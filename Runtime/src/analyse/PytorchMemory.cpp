/*
@author: Steven (Jiaxun) Tang <jtang@umass.edu>
@author: Tongping Liu <tongping.liu@bytedance.com>
*/
#include <iostream>
#include <fstream>
#include <array> 
#include <dlfcn.h>
#include <cstring>
#include <stdio.h>
#include <cuda_runtime.h>
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds
#include <c10/cuda/CUDACachingAllocator.h>
#include <pthread.h>

#include "common/Logging.h"
#include "common/Tool.h"
#include "trace/proxy/CUDAProxy.h"
#include "trace/proxy/PytorchMemProxy.h"
#include "analyse/PytorchMemory.h"
#include "analyse/DriverMemory.h"
#include "trace/hook/PyHook.h"
#include "common/DependencyLibVersionSpecifier.h"
#include "common/CUDAHelper.h"


using namespace std;
namespace mlinsight{
PytorchMemory torchMem;

/* 
The beginning of PyTorch allocator's detailed implementation, where the 
following functions should be aligned with the implementation of PyTorch's allocator
*/
//Copied from pytorch
constexpr size_t kMinBlockSize =
    512; // all sizes are rounded to at least 512 bytes
constexpr size_t kSmallSize = 1048576; // largest "small" allocation is 1 MiB
constexpr size_t kSmallBuffer =
    2097152; // "small" allocations are packed in 2 MiB blocks
constexpr size_t kLargeBuffer =
    20971520; // "large" allocations may be packed in 20 MiB blocks
constexpr size_t kMinLargeAlloc =
    10485760; // allocations between 1 and 10 MiB may use kLargeBuffer
constexpr size_t kRoundLarge = 2097152; // round up large allocations to 2 MiB


static size_t round_size(size_t size) {
    if (size < kMinBlockSize) {
        return kMinBlockSize;
    } else {
        return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
    }
}

static bool is_large_object(size_t size) {
    return size > kSmallBuffer ? true : false;  
}

bool should_split(size_t blockSize, size_t reqSize, bool isLargeBlock) {
    size_t remaining = blockSize - reqSize;
    if (!isLargeBlock) {
      return remaining >= kMinBlockSize; //512
    } else {
      return (remaining > kSmallSize); // 1M
    }
}


/* The end of PyTorch's specific implemntation */

void trackTorchCudaMalloc(void * devicePtr, ssize_t size) {
    torchMem.basic.curUsage += size;

    if(torchMem.basic.curUsage > torchMem.basic.peakUsage) {
        torchMem.basic.peakUsage = torchMem.basic.curUsage;
    }

    // Check and update maxExternalFrag if possible
    if((torchMem.basic.curUsage - torchMem.alloc.memAliveObjs) > torchMem.maxExternalFrag) {
        torchMem.maxExternalFrag = torchMem.basic.curUsage - torchMem.alloc.memAliveObjs; 
        torchMem.memoryAtMaxExternalFrag = torchMem.basic.curUsage;
        torchMem.requestSizeAtMaxExternalFrag = size; 
    }

    //INFO_LOGS(stderr, "trackTorchCudaMalloc: cudaMalloc ptr %p size %lx\n", devicePtr, size);
    // Recording the block information
    torchMem.countCudaMallocs += 1; 
    torchMem.numFreedObjects += 1; 
    torchMem.memFreedObjects += size; 
    if(is_large_object(size)) {
        torchMem.memFreedLargeObjects += size; 
    } 
    else {
        torchMem.memFreedSmallObjects += size; 
    }

    //INFO_LOGS(stderr, "trackTorchCudaMalloc: adding devicePtr %p size %lx to torchMem.mapFreeObjs\n", devicePtr, size);
 
    // This is a new allocated block, adding this block into the mapFreeObjs
    // We will use devicePtr as the key, as it can be used to get the object quickly later
    torchMem.mapFreeObjs[devicePtr] = new TorchObject(size, devicePtr);
}

void checkReadyForErase(TorchObject * object, void * ptr) {
    //INFO_LOGS("object %p size %lx ptr %p initSize %lx\n", object->ptr, object->size, ptr, object->initSize);
    assert(object->size == object->initSize);
    assert(object->ptr == ptr);
    assert(object->prev == nullptr && object->next == nullptr);
}

void trackTorchCudaFree(void * ptr, ssize_t size) {
    assert(torchMem.basic.curUsage >= size);
    torchMem.basic.curUsage -= size; 
    torchMem.countCudaFrees+=1; 
    torchMem.memCudaFrees += size; 

    torchMem.numFreedObjects -= 1; 
    torchMem.memFreedObjects -= size; 
    if(is_large_object(size)) {
        torchMem.memFreedLargeObjects -= size; 
    } 
    else {
        torchMem.memFreedSmallObjects -= size; 
    }

    // Remove this object from mapFreeObjs
    // TODO: check all blocks inside the original block 
    if(torchMem.mapFreeObjs.count(ptr) == 0) {
        INFO_LOG("ptr is not existing in mapFreeObjs!\n");
        assert(ptr == NULL); // Ugly, just force it to stop now
    }

    TorchObject * object = torchMem.mapFreeObjs[ptr];

    TorchObject * current = object; 
    // merge the adjacent object
    while(current->next != nullptr && current->next->allocated == false) {
        // Keep merging freed objects
        TorchObject * next = current->next;

        current->size += next->size;
        current->next = next->next;

        torchMem.mapFreeObjs.erase(next->ptr); 
        delete next; 

        torchMem.numFreedObjects -=1; 
    }

    INFO_LOGS("checkFree ptr %p size %lx\n", ptr, size);
    checkReadyForErase(object, ptr); 

    torchMem.mapFreeObjs.erase(ptr);
}

// Currently, this function is not used, which may be able to use it in the future.
#if TORCH_VERSION_MAJOR >= 2
void processCUDAOOMError(const c10::OutOfMemoryError&, ssize_t allocationSize)
#else
void processCUDAOOMError(const c10::CUDAOutOfMemoryError&, ssize_t allocationSize)
#endif
{

    using namespace c10::cuda::CUDACachingAllocator;
    typedef DeviceStats (*GetDevicePtr_t)(int device);
    int deviceID;

    //Reference https://github.com/pytorch/pytorch/blob/67ece03c8cd632cce9523cd96efde6f2d1cc8121/c10/cuda/CUDACachingAllocator.cpp#L1660C3-L1660C3
    CUDA_ASSERT(cudaGetDevice(&deviceID));
}

void trackPytorchAllocation(ssize_t size, void * ptr) {
    if(ptr == nullptr)
        return;

    INFO_LOGS("trackPytorchAllocation ptr %p size %lx\n", ptr, size);

    // Update the torch's allocation information
    torchMem.alloc.numAllocs += 1;
    torchMem.alloc.numAliveObjs += 1;

    // Update the information relate to the current object
    if(torchMem.mapFreeObjs.count(ptr) == 0) {
        INFO_LOGS("Error: torchallocation ptr %p with size %lx, NOT in the freeobjects\n", ptr, size);
        assert(ptr == nullptr);
    }

    TorchObject * current = torchMem.mapFreeObjs[ptr];

    INFO_LOGS("torchallocation ptr %p with size %lx before merging, current ptr %p size %lx\n", ptr, size, current->ptr, current->size);

    // Check whether we need to merge objects together
    /* If current->size == size, then no need to check. 
       (1) If the current object size is less than the requested size, 
           then definitely we would need to merge its next object. 
       (2) Even if the current object is larger than the requested size, 
           we need to check whether we could merge with the neighbor. 
           Otherwie, it will create an issue for the following allocation.
           0x8f600~0x8f623 
           0x8f623~0x8f640 

           If the allocation is 20 here, then the first allocation is 
           0x8f600. However, if 0x8f623 is not merged with 0x8f640 block, 
           then we may have the wase of 2 after the allocation of (0x8f600~0x8f620).
           But if the merge occurs, then we will have one big object (0x8f600~0x8f640), 
           after the first allocation (0x8f600~0x8f620), we will still have another block (0x8f620~0x8f640). 

           Note that this policy works for different versions of Pytorch Allocator, since free() will mark
           block->allocated to be false, and block->stream_uses to be empty as shown below. 
           That is, the checking of process_events() in malloc() function will be succeed. 
           As we only perform the merge inside malloc() (trackTorchAllocation()), which should be 
           always successful. 
             
           void free(Block* block) {

                block->allocated = false; 

                // When free, we will actually remove the stream from the block, but added 
                // an event to cuda_events, which will check and free memory blocks later.  
                if (!block->stream_uses.empty()) {
                    insert_events(block); // the insertion will make block->stream_uses empty
                }
                else {
                    free_block(block);
                }
            }
    */
    while((current->size != size) && ((current->size < size) ||
         (!should_split(current->size, size, is_large_object(current->initSize)) && current->next != nullptr && (current->next->allocated == false)))) {
        // Keep merging freed objects
        TorchObject * next = current->next;
        if(next == nullptr) {
            INFO_LOG("In checking merge, next is invalid!!");
            assert(next != nullptr);
        }

        current->size += next->size;
        current->next = next->next;

        if(current == next) {
            DBG_LOGS("current %p current->ptr %p, but erase next %p next->ptr %p\n", current, current->ptr, next, next->ptr);
            assert(current != next); 
        }
    //    DBG_LOGS("current %p current->ptr %p, but erase next %p next->ptr %p", current, current->ptr, next, next->ptr);

        torchMem.mapFreeObjs.erase(next->ptr); 
        delete next; 

        torchMem.numFreedObjects -=1; 

    }

    if(should_split(current->size, size, is_large_object(current->initSize))) {
        //Update mapFreeObjs table as the split occurs before this. 
        int blockSize = round_size(size);

        TorchObject * remaining = new TorchObject(current->initSize, current->size - blockSize);
        if(remaining == nullptr) {
            DBG_LOG("Out of CPU memory now. Exit!!!");
            exit(-1); 
        }

        if(current->next) {
            remaining->next = current->next;
            current->next->prev = remaining;
        }
        remaining->prev = current;
        remaining->ptr = static_cast<char*>(current->ptr) + blockSize;
        remaining->allocated = false; 

        if(current->ptr != ptr) {
            assert(current->ptr == ptr); 
        }

        // Adding this object to mapFreeObjs
        torchMem.mapFreeObjs[remaining->ptr] = remaining; 

        // Update the corresponding information of the current object
        current->next = remaining;
        current->size = blockSize; 
    } 
    else {
        torchMem.numFreedObjects -=1; 
    }

    current->fragment = current->size - size;
    current->allocated = true; 

    // Update statistics of torchMem
    torchMem.alloc.allocMem += current->size;
    torchMem.alloc.memAliveObjs += current->size; 
    torchMem.memFreedObjects -= current->size; 
    
    torchMem.internalFrag += current->fragment;
    if(torchMem.internalFrag > torchMem.maxInternalFrag) {
        torchMem.maxInternalFrag = torchMem.internalFrag;
    }

    if(is_large_object(current->initSize)) {
        torchMem.memFreedLargeObjects -= current->size; 
    } 
    else {
        torchMem.memFreedSmallObjects -= current->size; 
    }

    //printf("trackPytorchAllocation: allocating ptr %p size %lx now\n", ptr, size);

    //Get pytorch callstack
    current->updatePythonCallStack();

    // Remove the current object from the mapFreeObjs but inserting it into mapAliveObjs.
    torchMem.mapFreeObjs.erase(ptr);
    torchMem.alloc.mapAliveObjs[ptr] = current;
    if (ptr != current->ptr) {
        ERR_LOGS("trackPytorchAllocation ptr %p current->ptr %p size %lx\n", ptr, current->ptr, size);
    }
}


void trackPytorchFree(void * ptr) {
    if(ptr == nullptr)
        return;

    //DBG_LOG("Free: ptr %p", ptr);

    // Finding the entry in the hash map
    if(torchMem.alloc.mapAliveObjs.count(ptr) == 0) {
        ERR_LOGS("ERROR: tracking trackPytorchFree failed, where the pointer (%p) is not in the freeobjects\n", ptr);
        assert(ptr == NULL); // UGLY, but reporting an issue
    }

    TorchObject * current = torchMem.alloc.mapAliveObjs[ptr];
    // Sanity check
    if(ptr != current->ptr) {
        ERR_LOGS("trackPytorchFree failure, where the free pointer (%p) is not same as the pointer (%p) size %lx in block\n", ptr, current->ptr, current->size);
        assert(ptr == current->ptr); 
    }
    
    size_t curSize = current->size; 

    //DBG_LOGS("[%ld] trackPytorchFree: ptr %p ~ %p size %lx initSize %lx", pthread_self(), ptr, static_cast<char*>(ptr) + current->size, current->size, current->initSize);
    // Now updating the allocations information, which will also record 
    // the devicePtr to the allocated one. 
    torchMem.alloc.numFrees += 1;
    torchMem.alloc.freeMem += curSize;

    torchMem.alloc.memAliveObjs -= curSize; 
    torchMem.alloc.numAliveObjs -= 1;

    // Update the information of available objects
    torchMem.memFreedObjects += curSize;
    torchMem.internalFrag -= current->fragment; 

    if(is_large_object(current->initSize)) {
        torchMem.memFreedLargeObjects += curSize;
    }
    else {
        torchMem.memFreedSmallObjects += curSize;
    }
    
    current->allocated = false;

    // Check whether the current object (after possible merges) is the larger than maxFreedObjectSize
    // this is to detect whether a failing allocation is caused by external fragmentation
    if(current->size > torchMem.maxFreedObjectSize) {
        torchMem.maxFreedObjectSize = current->size; 
    }
            
    // Insert this object (with possible new ptr) to mapFreeObjs   
    torchMem.alloc.mapAliveObjs.erase(ptr);
    torchMem.mapFreeObjs[ptr] = current;

}

void printPythonCallstack(std::ofstream &output, CallStack<PyCallStack, PYTHON_CALL_STACK_LEVEL> * cs) {
    for(int i = 0; i < cs->levels; ++i) {
        const PyCallStack& pyCallStack= cs->array[i];
        output << pyCallStack.cachedCodeExtra->pythonSourceFileName << ":" 
               << pyCallStack.pythonSourceFileLineNumber << endl;
    }

    output << endl;
}


void printLeakyTorchObjects(std::ofstream &output) {

    struct CallStackHash {
        size_t operator()(const CallStack<PyCallStack, PYTHON_CALL_STACK_LEVEL>* key) const {
            size_t hashValue = 0xFFFFFFFF;
            for(int i = 0; i < key->levels; ++i){
                hashValue ^= std::hash<PyCodeExtra *>()(key->array[i].cachedCodeExtra+key->array[i].pythonSourceFileLineNumber);
            }
            return hashValue;
        }
    };

    struct CallStackComparator {
        size_t operator()(const CallStack<PyCallStack, PYTHON_CALL_STACK_LEVEL>* key1, const CallStack<PyCallStack, PYTHON_CALL_STACK_LEVEL>* key2) const {
            bool isEqual = key1->levels == key2->levels;
            if(isEqual){
                for(int i=0;i<key1->levels;++i){
                    if(!((key1->array[i].cachedCodeExtra == key2->array[i].cachedCodeExtra) 
                        && (key1->array[i].pythonSourceFileLineNumber == key2->array[i].pythonSourceFileLineNumber))){
                        isEqual=false;
                        break;
                    }
                }
            }
            return isEqual;   
        
        }
    };


    std::unordered_map<CallStack<PyCallStack, PYTHON_CALL_STACK_LEVEL>*, PotentialLeakOject *, CallStackHash, CallStackComparator> leakyObjsMap;

    int objectsNum = 0;
    int callstackNum = 0;
    int replicNum = 0; 
    int maxWaste = 0; 

    // Count the potentially-leaked objects based on the callsite
    auto & aliveObjs = torchMem.alloc.mapAliveObjs; 
    for(auto it = aliveObjs.begin(); it != aliveObjs.end(); it++){
        TorchObject * object = it->second;
        assert(object != nullptr); 

        objectsNum++;

        // Printing the callstack for non-torch objects. Do not replace if item is found.
        PotentialLeakOject* leakObj;
        if (leakyObjsMap.count(&object->callstack) == 0) {
            // When the current key is not existing, create a new leaky object
            leakObj = new PotentialLeakOject(object->size);
            leakyObjsMap[&object->callstack] = leakObj; 
            callstackNum++;
        }
        else {
            replicNum +=1; 
            leakObj = leakyObjsMap[&object->callstack]; 
        }

        leakObj->aliveCount += 1;
        leakObj->aliveMem += object->size;
        if(leakObj->aliveMem > maxWaste) {
            maxWaste = leakObj->aliveMem;  
        }
    }

    //Please do not use cout or printf here and use macro OUTPUT/OUTPUTS to output files into the disk.
    OUTPUTS("In the total of %d objects, there are %d callstacks, %d replicating callstacks, and the maximum waste %s !!!\n",
            objectsNum, callstackNum, replicNum, format_size(maxWaste).c_str());

    OUTPUTS("%d Pytorch alive objects include %d callstacks, %d replicating callstacks, and the maximum waste %s !!!\n",
            objectsNum,callstackNum,replicNum,format_size(maxWaste).c_str());


    std::vector<std::pair<size_t, CallStack<PyCallStack, PYTHON_CALL_STACK_LEVEL> *> > leakyObjects; 

    for(auto it = leakyObjsMap.begin(); it != leakyObjsMap.end(); it++) {
        PotentialLeakOject* object = it->second; 
        leakyObjects.push_back({object->aliveMem, it->first});
    }

    //DBG_LOGS("leakobject size %d\n", leakyObjects.size());
    // Sort the vector based on object size (you can customize the sorting order)
    std::sort(leakyObjects.begin(), leakyObjects.end(),
        [](const std::pair<size_t, CallStack<PyCallStack, PYTHON_CALL_STACK_LEVEL>*>& a, const std::pair<size_t, CallStack<PyCallStack, PYTHON_CALL_STACK_LEVEL>*>& b) {
            return a.first > b.first;
    });


    // Now printing all objects in decreasing order. 
    int i = 0; 
    for(auto it : leakyObjects) {

        PotentialLeakOject* leakObject = leakyObjsMap[it.second];
        CallStack<PyCallStack, PYTHON_CALL_STACK_LEVEL> * cs = it.second;
        output << i << "-th pytorch object: waste - " << format_size(leakObject->aliveMem) << ", alloc times: " << leakObject->aliveCount
               << ", unit size - "
               << format_size(leakObject->size) << ", callsite level: " << cs->levels << endl;

        printPythonCallstack(output, cs); 

        i++; 
    }

    //DBG_LOG("We finished the printing of python leaky objects");
}

void printPytorchMemoryProfile(std::ofstream & output) {

    output << endl;
    // Printing the pytorch information
    output << "Pytorch GPU information: current reserve  - " << format_size(torchMem.basic.curUsage) << ". Peak reserve - " << format_size(torchMem.basic.peakUsage) << endl;
    output << "\t Number of cudaMalloc: " << torchMem.countCudaMallocs << endl;
    output << "\t Number of cudaFree: " << torchMem.countCudaFrees << endl;
    output << "\t Number of allocations: " << torchMem.alloc.numAllocs << endl;
    output << "\t Total allocated memory: " << format_size(torchMem.alloc.allocMem) << endl; 
    output << "\t Number of frees:" << torchMem.alloc.numFrees << endl;
    output << "\t Total freed memory:" << format_size(torchMem.alloc.freeMem) << endl;
    //output << endl;
    output << "\t Number of alive objects: " << torchMem.alloc.numAliveObjs << endl; 
    output << "\t Memory of alive objects: " << format_size(torchMem.alloc.memAliveObjs) << endl;
    output << "\t Total internal fragmentation of alive objects: " << format_size(torchMem.internalFrag) << ". Maximum fragmentation:" << format_size(torchMem.maxInternalFrag) << endl; 
    output << "\t Maximum external fragmentation: " << format_size(torchMem.maxExternalFrag) << ", where memory usage at " << format_size(torchMem.memoryAtMaxExternalFrag) << " and request size " <<  format_size(torchMem.requestSizeAtMaxExternalFrag) << "." << endl;
    //output << "\t Number of freed objects: " << torchMem.numFreedObjects << endl; 
    //output << "\t Memory of freed objects: " << format_size(torchMem.memFreedObjects) << endl; 
    output << "\t Total available memory in allocator: " << format_size(torchMem.basic.curUsage - torchMem.alloc.memAliveObjs) << endl; 
    output << endl;

    printLeakyTorchObjects(output); 

    output << "****************************************************" << endl;
    output << "End of memory profile" << endl;
    output << "****************************************************" << endl;

}
}


