/*
@author: Steven (Jiaxun) Tang <jtang@umass.edu>
@author: Tongping Liu <tongping.liu@bytedance.com>
*/
#include <cstdint>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <string.h>
#include <sys/types.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <unistd.h>
#include "common/Logging.h"
#include "common/ProcInfoParser.h"
#include "common/Tool.h"
#include "analyse/DriverMemory.h"
#ifdef USE_TORCH
#include "analyse/PytorchMemory.h"
#endif
using namespace std;

namespace mlinsight {
    class MemInfo memory;
//class PytorchMemory memTorch;

    void updateTotalMemory(void) {
        size_t freeMem, totalMem;
        CUresult result = cuMemGetInfo(&freeMem, &totalMem);

        if (result != CUDA_SUCCESS) {
            ERR_LOG("MLInsight cannot invoke cuMemGetInfo, exiting now!!");
            return;
        }
        if (memory.totalMemory != totalMem) {
            memory.totalMemory = totalMem;
        }

        cout << "updateTotalMemory, current usage - " << format_size(totalMem - freeMem) << endl;
        ssize_t curUsage = totalMem - freeMem;
        memory.total.curUsage = curUsage;
        if (curUsage > memory.total.peakUsage) {
            memory.total.peakUsage = curUsage;
        }
    }

void updateDriverMemoryOnAlloc(ssize_t size, void *devicePtr, CallStack<void*, CPP_CALL_STACK_LEVEL>& callstackObject, bool isTorchAllocation) {
        // Updating the basic information at first
        memory.driver.basic.curUsage += size;

        if (memory.driver.basic.curUsage > memory.driver.basic.peakUsage) {
            //printf("Detecting allocation curUsage - %lx, peakUsage - %lx\n", memory.driver.basic.curUsage, memory.driver.basic.peakUsage);
            memory.driver.basic.peakUsage = memory.driver.basic.curUsage;
        }

        // Now updating the allocations information, which will also record
        // the devicePtr to the allocated one.
        memory.driver.alloc.numAllocs += 1;
        memory.driver.alloc.allocMem += size;
        memory.driver.alloc.memAliveObjs += size;
        memory.driver.alloc.numAliveObjs += 1;

        cout << "Detecting driver curUsage - " << format_size(memory.driver.basic.curUsage) << ". peakUsage - " << format_size(memory.driver.basic.peakUsage) << ". Alive objects - " << memory.driver.alloc.numAliveObjs << endl;;
        // Getting the callstacks of this allocation

        // Insert the current object into the hash map
        DriverObject * object = new DriverObject(size, callstackObject, isTorchAllocation);
        memory.driver.alloc.mapAliveObjs[devicePtr] = object;
    }


    void updateDriverMemoryOnFree(void *devicePtr) {
        if (devicePtr == NULL) {
            //fprintf(stderr, "wrong devicePtr %p\n", devicePtr);
            //mlinsight::print_stacktrace();
            return;
        }
    
        // Finding the entry in the hash map
        if(memory.driver.alloc.mapAliveObjs.count(devicePtr) == 0) {
            fprintf(stderr, "Free pointer %p not existing, but it points to %p!!!\n", devicePtr, *((void **)devicePtr));
            exit(-1);
        }

        DriverObject * object = memory.driver.alloc.mapAliveObjs[devicePtr];

        // Updating the basic information at first
        memory.driver.basic.curUsage -= object->size;

        // Now updating the allocations information, which will also record
        // the devicePtr to the allocated one.
        memory.driver.alloc.numFrees += 1;
        memory.driver.alloc.freeMem += object->size;
        memory.driver.alloc.memAliveObjs -= object->size;
        memory.driver.alloc.numAliveObjs -= 1;

        // check whether this object is Pytorch allocation
        if (object->isTorchAlloc) {
        #ifdef USE_TORCH
            trackTorchCudaFree(devicePtr, object->size);
            memory.driver.alivePytorchMemory -= object->size;
        #endif
        }
        else {
            memory.driver.aliveNormalMemory -= object->size;
        }

        // Delete the current object from the hash map
        memory.driver.alloc.mapAliveObjs.erase(devicePtr);
        delete object; 
        //printf("total objects in the map - %d\n", memory.driver.alloc.mapAliveObjs.getEntryNumber());
    }

    bool isPytorchAllocation(CallStack<void*, CPP_CALL_STACK_LEVEL>& callstackObject) {
        for (int i = 0; i < callstackObject.levels; i++) {
            void *addr = callstackObject.array[i];
            if (addr >= mlinsight::libc10_cuda_text_begin && addr <= mlinsight::libc10_cuda_text_end) {
                return true;
            }
        }

        return false;
    }

    // For each driver allocation, we will update the total, driver (and non-driver), and
    // pytroch allocator (if possible)'s information correspondingly.
    void trackDriverAllocation(ssize_t size, void *devicePtr) {

        //cout << "trackDriverAllocation allocatedSize - " << allocatedSize << ", devicePtr " << devicePtr << endl;
        if (devicePtr == nullptr) {
            return;
        }

        // Update the total information as shown in nvidia-smi
        updateTotalMemory();

        CallStack<void*, CPP_CALL_STACK_LEVEL> callstackObject;
        getCppStacktrace(callstackObject);

        bool isTorchAllocation = false; 
    #ifdef USE_TORCH
        isTorchAllocation = isPytorchAllocation(callstackObject);
    #endif
        // Update the information for driver allocation
        updateDriverMemoryOnAlloc(size, devicePtr, callstackObject, isTorchAllocation);
 
        // Update the Pytorch's allocation if necessary
        if (isTorchAllocation == true) {
            // Update the pytorch's related information. Note that this
            
            // the general information about Pytorch's allocator, but not about
            // the detailed allocations from Pytorch's scripts
    #ifdef USE_TORCH
            trackTorchCudaMalloc(devicePtr, size);
            memory.driver.alivePytorchMemory += size; 
    #endif
        }
        else {
            memory.driver.aliveNormalMemory += size; 
        }
        //INFO_LOGS("After cuMemAlloc malloc %zd, ptr %p\n", allocatedSize, devicePtr);
    }

    void trackDriverFree(void *devicePtr) {
        updateTotalMemory();
        updateDriverMemoryOnFree(devicePtr);
    }

    
    void printDriverObjects(std::ofstream &output) {
    struct CPPCallStackHash {
        size_t operator()(const CallStack<void*, CPP_CALL_STACK_LEVEL>* key) const {
            size_t hashValue = 0xFFFFFFFF;
            for(int i = 0; i < key->levels; ++i){
                hashValue ^= std::hash<void *>()(key->array[i]); // Correct
            }
            return hashValue;
        }
    };

    struct CPPCallStackCompare {
        size_t operator()(const CallStack<void*, CPP_CALL_STACK_LEVEL>* key1, const CallStack<void*, CPP_CALL_STACK_LEVEL>* key2) const {
            bool isEqual = key1->levels == key2->levels;
            if(isEqual){
                for(int i=0;i<key1->levels;++i){
                    if(!(key1->array[i]==key2->array[i])){
                        isEqual=false;
                        break;
                    }
                }
            }
            return isEqual;
        }
    };

    std::unordered_map<CallStack<void*, CPP_CALL_STACK_LEVEL>*, PotentialLeakOject *, CPPCallStackHash, CPPCallStackCompare> leakyObjsMap;

    int objectsNum = 0;
    int callstackNum = 0;
    int replicNum = 0; 
    int maxWaste = 0; 

    // Arrange potentially-leaked objects based on the callsite
    auto & aliveObjs = memory.driver.alloc.mapAliveObjs; 
    for(auto it = aliveObjs.begin(); it != aliveObjs.end(); it++){
        DriverObject * object = it->second;

        assert(object != nullptr); 

        if (object->isTorchAlloc) {
            continue;
        }

        objectsNum++;

        // Printing the callstack for non-torch objects. Do not replace if item is found.
        PotentialLeakOject* leakObj;
        CallStack<void*, CPP_CALL_STACK_LEVEL>* cs = &object->callstack; 
        if (leakyObjsMap.count(cs) == 0) {
            // When the current key is not existing, create a new leaky object
            leakObj = new PotentialLeakOject(object->size);

            assert(leakObj != nullptr);
            // create a temporary object (new, copy constructor)
            // = "insert"
            leakyObjsMap[cs] = leakObj; 
            callstackNum++;
        }
        else {
            replicNum +=1; 
            leakObj = leakyObjsMap[cs]; 
        }

        leakObj->aliveCount += 1;
        leakObj->aliveMem += object->size;
        if(leakObj->aliveMem > maxWaste) {
            maxWaste = leakObj->aliveMem;  
        }
    }

    cout << objectsNum << " driver objects include " << callstackNum << " callstacks, " << replicNum << " replicating callstacks, " 
           << "and the maximum waste " << format_size(maxWaste) << "!!!" << endl;
    
    cout << endl; 

    output << objectsNum << " driver objects include " << callstackNum << " callstacks, " << replicNum << " replicating callstacks, " 
           << "and the maximum waste " << format_size(maxWaste) << "!!!" << endl;

    output << endl;

    std::vector<std::pair<size_t, CallStack<void*, CPP_CALL_STACK_LEVEL>* > > leakyObjects; 

    for(auto it = leakyObjsMap.begin(); it != leakyObjsMap.end(); it++) {
        PotentialLeakOject* object = it->second; 
        leakyObjects.push_back({object->aliveMem, it->first});
        if(leakyObjsMap.count(it->first) == 0) {
            const CallStack<void*, CPP_CALL_STACK_LEVEL> * cs = it->first; 
            fprintf(stderr, "Error: aliveMem %lx size %lx (%p - %p) not existing in leakyObjsMap\n", object->aliveMem, object->size, cs->array[3], cs->array[4]);
        }
    }

    //fprintf(stderr, "leakobject size %d\n", leakyObjects.size());
    // Sort the vector based on object size (you can customize the sorting order)
    std::sort(leakyObjects.begin(), leakyObjects.end(),
        [](const std::pair<size_t, CallStack<void*, CPP_CALL_STACK_LEVEL>*>& a, const std::pair<size_t, CallStack<void*, CPP_CALL_STACK_LEVEL>*>& b) {
            return a.first > b.first;
    });


    output << endl;
    // Now printing all objects in decreasing order. 
    int i = 0; 
    for(auto it : leakyObjects) {
        PotentialLeakOject* leakObject = leakyObjsMap[it.second];

        assert(leakObject != nullptr);

        const CallStack<void*, CPP_CALL_STACK_LEVEL> * cs = it.second;
        output << i << "-th driver object: waste - " << format_size(leakObject->aliveMem) << ", alloc times: " << leakObject->aliveCount
               << ", unit size - "
               << format_size(leakObject->size) << ", callsite level: " << cs->levels << endl;
       
        char **strings;
        strings = backtrace_symbols(cs->array, cs->levels);
        if (cs->levels > 0 && strings != NULL) {
            for (int j = 0; j < cs->levels; j++) {
                output << strings[j] << endl;
            }

            free(strings);
            output << endl;
        }

        i++; 

        // Now we only print 5 objects here. 
    }
    }

    void reportMemoryProfile(ssize_t oomAllocSize) {
        size_t freeMem, totalMem;

        CUresult result = cuMemGetInfo(&freeMem, &totalMem);
        if (result != CUDA_SUCCESS) {
            ERR_LOG("MLInsight cannot invoke cuMemGetInfo, exiting now!!");
            return;
        }

        if(memory.totalMemory == 0) {
            memory.totalMemory = totalMem;
        }
        
        // Create a file using the process id.
        std::string fileName = "memoryprofile_" + std::to_string(getpid()) + ".txt";
        std::ofstream output(fileName, std::ios::app);

        printf("oomAllocSize %lx!!!!\n", oomAllocSize);

        // If size is given, there is an OOM failure when trying to allocate the given size
        if(oomAllocSize != 0) {
            output << endl;
            output << "OOM error when allocating " << format_size(oomAllocSize) << " at the following callsite: " << endl;
        #ifdef USE_TORCH
            TorchObject current; 
            current.updatePythonCallStack();
            printPythonCallstack(output, current.callstack);
        #endif

            if (oomAllocSize >= totalMem) {
                output << endl << "GPU capacity is " << format_size(totalMem) << ", which is less than the requested size - " << format_size(oomAllocSize) << endl; 
                output << "That is, this OOM is due to GPU capacity. Please use a larger GPU or adjust parameters like token number or batch size!!" << endl; 
            }
            else if(oomAllocSize >= freeMem) {
                output << "Allocation size - " << format_size(oomAllocSize) << " is larger than available GPU memory - " << format_size(freeMem) << endl;
            #ifdef USE_TORCH    
                if(oomAllocSize < (torchMem.alloc.memAliveObjs + freeMem)) {
                    output << "The problem is caused by external fragmentation of PyTorch allocator, which may "
                        << "be able to be fixed by invoking torch.cuda.empty_cache()!" << endl;
                }

                if(oomAllocSize < torchMem.internalFrag) {
                    output << "One major issue is caused by the internal fragmentation intorduced by PyTorch allocator." << endl; 
                }
            #endif
            }
        }

        output << endl;
        output << "****************************************************" << endl;
        output << "memory profile is shown as follows:" << endl;
        output << "****************************************************" << endl;

        if (memory.totalMemory != 0) {
            // Printing the total information
            output << "General GPU information: total " << format_size(memory.totalMemory) << ". Current usage - "
                   << format_size(memory.total.curUsage) << ". Peak usage - " << format_size(memory.total.peakUsage)
                   << endl;
            cout << "General GPU information: total " << format_size(memory.totalMemory) << ". Current usage - "
                 << format_size(memory.total.curUsage) << ". Peak usage - " << format_size(memory.total.peakUsage)
                 << endl;
        } else {
            output << "General GPU information: total " << format_size(totalMem) << ". Current usage - "
                   << format_size(memory.total.curUsage) << ". Peak usage - " << format_size(memory.total.peakUsage)
                   << endl;
            cout << "General GPU information: total " << format_size(memory.totalMemory) << ". Current usage - "
                 << format_size(memory.total.curUsage) << ". Peak usage - " << format_size(memory.total.peakUsage)
                 << endl;
        }
        output << endl;

        // Printing the driver information
        output << "Driver GPU information: current usage - " << format_size(memory.driver.basic.curUsage)
               << ". Peak usage - " << format_size(memory.driver.basic.peakUsage) << endl;


        cout << "Driver GPU information: current usage - " << format_size(memory.driver.basic.curUsage)
             << ". Peak usage - " << format_size(memory.driver.basic.peakUsage) << endl;

        output << "\t Within current memory, normal driver - " << format_size(memory.driver.aliveNormalMemory)
               << ". Pytorch - " << format_size(memory.driver.alivePytorchMemory) << endl;

        output << "\t Number of allocations: " << memory.driver.alloc.numAllocs << endl;
        output << "\t Total allocated memory: " << format_size(memory.driver.alloc.allocMem) << endl;
        output << "\t Number of frees:" << memory.driver.alloc.numFrees << endl;
        output << "\t Total freed memory:" << format_size(memory.driver.alloc.freeMem) << endl;
        //output << endl;
        output << "\t Number of alive objects: " << memory.driver.alloc.numAliveObjs << endl;
        output << "\t Memory of alive objects: " << format_size(memory.driver.alloc.memAliveObjs) << endl;

        printDriverObjects(output);
    #ifdef USE_TORCH
        printPytorchMemoryProfile(output);
    #endif
    }
}