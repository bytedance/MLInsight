#ifndef __MEMORY_LEAK_METRICS_PYTORCH_H__
#define __MEMORY_LEAK_METRICS_PYTORCH_H__

#if USE_TORCH
/**
 * This file defines classes to expand MemoryLeakAnalyzer for Pytorch
 */

#include <cstdio>
#include <iostream>
#include <sys/types.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <Python.h>
#include <frameobject.h>
#include <ceval.h>
#include <unordered_map>
#include "MemLeakMetrics.h"
//#include "common/HashMap.h"
#include "common/HashAndCompareFunctions.h"
#include "common/CallStack.h"
#include "trace/type/PyCodeExtra.h"
#include "trace/hook/PyHook.h"
#include "analyse/FlameGraph.h"
#include <vector>
#include "common/TensorObj.h"
#include "trace/proxy/PytorchMemProxy.h"
#include "trace/tool/Perfetto.h"

namespace mlinsight {
    extern void *pythonInterpreter_text_begin;
    extern void *pythonInterpreter_text_end;
}

namespace mlinsight::MemLeak::InternalFrag {

    /**
     * =================================================================================================================
     * Pytorch simulation related code. The content is basically copied from https://github.com/bytedance/MLInsight/blob/1cc26835ca52886f35b765818146314170673978/Runtime/src/analyse/PytorchMemory.cpp,
     * and is refactored. The stability of this code is pending tests.
     * Currently the code base does not use the following code.
     * =================================================================================================================
     */
    /**
     * Augment Tensor class with prev and next fields for fragmentation detection
     * todo: This code is currently disabled.
     */
    class Chunk {
    public:
        ssize_t initSize = 0;
        ssize_t size = 0;
        void *ptr = nullptr;
        Chunk *prev = nullptr; // previous block in the same cudaMalloc's allocation
        Chunk *next = nullptr; // next block in the same cudaMalloc's allocation
        bool allocated = false;
        ssize_t fragment = 0;

        Chunk() = default;

        /**
        * Constructor used when first allocating a block
        * @return
        */
        Chunk(ssize_t initSize, void *ptr) : initSize(initSize), size(initSize), ptr(ptr) {
        }

        /**
         * Constructor used when spliting block
         */
        Chunk(ssize_t initSize, ssize_t size) : size(size), initSize(initSize) {

        }
    };

    /**
     * This class is a subclass of AllocatorStatus.
     * We should add TensorObjSimulationMixIn to Tensor object in LEAK_OBJ_TYPE such as AllocatorStatusInternalFragTorchSimulation<Tensor< TorchSimulationTensorMixIn, ......Other mixins...... >>
     * Note that AllocatorStatus
     * todo: The simulation is mainly used to report internal fragmentation and is currently disabled.
     *
     * @tparam CTENSOR_TYPE Tensor< Mixins...... >
     */
    template<typename FRAMEWORK_TENSOR_TYPE>
    class TorchSimuMetric {
    public:
        ssize_t internalFrag = 0, maxInternalFrag = 0;
        ssize_t memFreedSmallObjects = 0;
        ssize_t memFreedLargeObjects = 0;
        ssize_t memFreedObjects = 0;     // Available freed objects in the memory
        ssize_t numFreedObjects = 0;     // Available freed objects in the memory
        ssize_t maxFreedObjectSize = 0; //A debug variable. Check detailed comments in the source code.

        std::unordered_map<void *, Chunk *> mapAliveChunks; // the map of free memory chunks
        std::unordered_map<void *, Chunk *> mapFreeChunks; // the map of free memory chunks

    public:


        /**
         * [Interface]
         * Check AllocatorStatus's constructor
         * @param genericMetric TorchSimuMetric need to use memAliveObjs in genericMetric.
         */
        explicit TorchSimuMetric(){

        }

        void onPostAllocDriver(ssize_t size, void *ptr){
            /**
             * The following commented metrics have been refactored to other classes to reuse code

            torchMem.basic.curUsage += size;

            if(torchMem.basic.curUsage > torchMem.basic.peakUsage) {
                torchMem.basic.peakUsage = torchMem.basic.curUsage;
            }

            // Check and update maxExternalFrag if possible
            if((torchMem.basic.curUsage - torchMem.alloc.memAliveObjs) > torchMem.maxExternalFrag) {
                torchMem.maxExternalFrag = torchMem.basic.curUsage - torchMem.alloc.memAliveObjs;
                torchMem.maxReserveAtMaxExternalFrag = torchMem.basic.curUsage;
                torchMem.requestSizeAtMaxExternalFrag = size;
            }

            //INFO_LOGS(stderr, "trackTorchCudaMalloc: cudaMalloc ptr %p size %lx\n", devicePtr, size);
            // Recording the block information
            torchMem.countCudaMallocs += 1;
            torchMem.numFreedObjects += 1;
            torchMem.memFreedObjects += size;
            */
            if (is_large_object(size)) {
                this->memFreedLargeObjects += size;
            } else {
                this->memFreedSmallObjects += size;
            }

            //INFO_LOGS(stderr, "trackTorchCudaMalloc: adding devicePtr %p size %lx to torchMem.mapFreeChunks\n", devicePtr, size);

            // This is a new allocated block, adding this block into the mapFreeChunks
            // We will use devicePtr as the key, as it can be used to get the object quickly later
            auto insertionIter = this->mapFreeChunks.try_emplace(ptr, new Chunk(size, ptr));
            assert(insertionIter.second == true); //ptr should not exist before
        }


        void  onPostAllocFramework(ssize_t size, void *ptr) {
            if (ptr == nullptr)
                return;

            //INFO_LOGS("trackPytorchAllocation ptr %p size %lx\n", ptr, size);



            /**
             * The following commented metrics have been refactored to other classes to reuse code
            // Update the torch's allocation information
            torchMem.alloc.numAllocs += 1;
            torchMem.alloc.numAliveObjs += 1;
            */
            auto findIter = this->mapFreeChunks.find(ptr);
#ifndef NDEBUG
            if (findIter == this->mapFreeChunks.end()) {
                fatalErrorS("Error: torchallocation ptr %p with size %lx, NOT in the freeobjects\n", ptr,
                            size);
            }
#endif
            Chunk *current = findIter->second;

            INFO_LOGS("torchallocation ptr %p with size %lx before merging, current ptr %p size %lx\n", ptr,
                      size, current->ptr, current->size);

            // Check whether we need to merge objects together
            /* If current->size == size, then no need to check.
               (1) If the current object size is less than the requested size,
                   then definitely we would need to merge its next object.
               (2) Even if the current object is larger than the requested size,
                   we need to check whether we could merge with the neighbor.
                   Otherwise, it will create an issue for the following allocation.
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
            while ((current->size != size) && ((current->size < size) ||
                                               (!should_split(current->size, size,
                                                              is_large_object(current->initSize)) &&
                                                current->next != nullptr &&
                                                !current->next->allocated))) {
                // Keep merging freed objects
                Chunk *next = current->next;
                if (next == nullptr) {
                    INFO_LOG("In checking merge, next is invalid!!");
                    assert(next != nullptr);
                }

                current->size += next->size;
                current->next = next->next;

                if (current == next) {
                    DBG_LOGS("current %p current->ptr %p, but erase next %p next->ptr %p\n", current,
                             current->ptr, next, next->ptr);
                    assert(current != next);
                }
                //    DBG_LOGS("current %p current->ptr %p, but erase next %p next->ptr %p", current, current->ptr, next, next->ptr);

                this->mapFreeChunks.erase(next->ptr);
                delete next;
                this->numFreedObjects -= 1;
            }

            if (should_split(current->size, size, is_large_object(current->initSize))) {
                //Update mapFreeChunks table as the split occurs before this.
                int blockSize = round_size(size);

                auto *remaining = new Chunk(current->initSize, current->size - blockSize);

                if (remaining == nullptr) {
                    fatalError("Out of CPU memory now. Exit!!!");
                }

                if (current->next) {
                    remaining->next = current->next;
                    current->next->prev = remaining;
                }
                remaining->prev = current;
                remaining->ptr = static_cast<char *>(current->ptr) + blockSize;
                remaining->allocated = false;

                //The upper simulated part should be equal to the actual allocation. Otherwise, the simulation is a failure.
                assert(current->ptr == ptr);

                // Adding this object to mapFreeChunks
                auto insertionRlt = this->mapFreeChunks.try_emplace(remaining->ptr, remaining);
                assert(insertionRlt.second = true);//Pointer should not already exist before

                // Update the corresponding information of the current object
                current->next = remaining;
                current->size = blockSize;
            } else {
                this->numFreedObjects -= 1;
            }

            current->fragment = current->size - size;
            current->allocated = true;

            /*
             * The following metrics have been refaactored to other classes to reuse code.
            // Update statistics of torchMem
            torchMem.alloc.memAllocs += current->size;
            torchMem.alloc.memAliveObjs += current->size;
            */

            //todo: Steven: Are there inconsistencies between memFreedObjects and numFreedObjects?
            this->memFreedObjects -= current->size;

            this->internalFrag += current->fragment;
            if (this->internalFrag > this->maxInternalFrag) {
                this->maxInternalFrag = this->internalFrag;
            }

            if (is_large_object(current->initSize)) {
                this->memFreedLargeObjects -= current->size;
            } else {
                this->memFreedSmallObjects -= current->size;
            }

            /*
             * The following metrics have been refaactored to other classes to reuse code.
             current->updatePythonCallStack();
             */

            // Remove the current object from the mapFreeChunks but inserting it into mapAliveObjs.
            this->mapFreeChunks.erase(ptr);
            auto insertionRlt = this->mapAliveChunks.try_emplace(ptr, current);
            assert(insertionRlt.second == true);

            //if (ptr != current->ptr) {
                //ERR_LOGS("trackPytorchAllocation ptr %p current->ptr %p size %lx\n", ptr, current->ptr,
                //         size);
            //}

        }


        void onPreFreeFramework(void *ptr) {
            if (ptr == nullptr) {
                return;
            }
            //DBG_LOG("Free: ptr %p", ptr);

            // Finding the entry in the hash map
            auto findIter = this->mapAliveChunks.find(ptr);
#ifndef NDEBUG
            if (findIter == this->mapAliveChunks.end()) {
                fatalErrorS(
                        "ERROR: tracking trackPytorchFree failed, where the pointer (%p) is not in the freeobjects\n",
                        ptr);
            }
#endif
            Chunk *current = findIter->second;
#ifndef NDEBUG
            // Sanity check
            if (ptr != current->ptr) {
                fatalErrorS(
                        "trackPytorchFree failure, where the free pointer (%p) is not same as the pointer (%p) size %lx in block\n",
                        ptr, current->ptr, current->size);
            }
#endif
            size_t curSize = current->size;

            /*
             * The following metrics have been refactored into other classes to reuse code.
             //DBG_LOGS("[%ld] trackPytorchFree: ptr %p ~ %p size %lx initSize %lx", pthread_self(), ptr, static_cast<char*>(ptr) + current->size, current->size, current->initSize);
             // Now updating the allocations information, which will also record
             // the devicePtr to the allocated one.
             torchMem.alloc.numFrees += 1;
             torchMem.alloc.memFrees += curSize;

             torchMem.alloc.memAliveObjs -= curSize;
             torchMem.alloc.numAliveObjs -= 1;
             */

            // Update the information of available objects
            this->memFreedObjects += curSize;
            this->internalFrag -= current->fragment;

            if (is_large_object(current->initSize)) {
                this->memFreedLargeObjects += curSize;
            } else {
                this->memFreedSmallObjects += curSize;
            }

            current->allocated = false;

            // Check whether the current object (after possible merges) is the larger than maxFreedObjectSize
            // this is to detect whether a failing allocation is caused by external fragmentation
            if (current->size > this->maxFreedObjectSize) {
                this->maxFreedObjectSize = current->size;
            }

            // Insert this object (with possible new ptr) to mapFreeChunks
            this->mapAliveChunks.erase(ptr);
            auto freeChunkInsertionIter = this->mapFreeChunks.try_emplace(ptr, current);
            assert(freeChunkInsertionIter.second == true); //ptr should not exist in this->mapFreeChunks
        }

        void  onPreFreeDriver(void *ptr) {
            /**
             * The following commented metrics have been refactored to other classes to reuse code
             * assert(torchMem.basic.curUsage >= size);
             * torchMem.basic.curUsage -= size;
             * torchMem.countCudaFrees+=1;
             * torchMem.memCudaFrees += size;
             */

            auto findIter = this->mapAliveChunks.find(ptr);
            assert(findIter != this->mapAliveChunks.end());
            ssize_t size = findIter->second->size;
            this->numFreedObjects -= 1;
            this->memFreedObjects -= size;

            if (is_large_object(size)) {
                this->memFreedLargeObjects -= size;
            } else {
                this->memFreedSmallObjects -= size;
            }

            // Remove this object from mapFreeChunks
            // TODO: check all blocks inside the original block
            auto freeObjIter = this->mapFreeChunks.find(ptr);
            assert(freeObjIter != this->mapFreeChunks.end());

            auto *object = freeObjIter->second;

            auto *current = object;
            // merge the adjacent object
            while (current->next != nullptr && !current->next->allocated) {
                // Keep merging freed objects
                auto *next = current->next;

                current->size += next->size;
                current->next = next->next;

                this->mapFreeChunks.erase(next->ptr);
                delete next; //todo: Maybe we should use object pool?

                this->numFreedObjects -= 1;
            }

            INFO_LOGS("checkFree ptr %p size %lx\n", ptr, size);
            checkReadyForErase(object, ptr);

            this->mapFreeChunks.erase(ptr);
        }

    protected:
        /*
          The beginning of PyTorch allocator's detailed implementation, where the
          following functions should be aligned with the implementation of PyTorch's allocator
          */
        //Copied from pytorch
        static constexpr size_t kMinBlockSize =
                512; // all sizes are rounded to at least 512 bytes
        static constexpr size_t kSmallSize = 1048576; // largest "small" allocation is 1 MiB
        static constexpr size_t kSmallBuffer =
                2097152; // "small" allocations are packed in 2 MiB blocks
        static constexpr size_t kLargeBuffer =
                20971520; // "large" allocations may be packed in 20 MiB blocks
        static constexpr size_t kMinLargeAlloc =
                10485760; // allocations between 1 and 10 MiB may use kLargeBuffer
        static constexpr size_t kRoundLarge = 2097152; // round up large allocations to 2 MiB


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

        void checkReadyForErase(Chunk *object, void *ptr) {
            //INFO_LOGS("object %p size %lx ptr %p initSize %lx\n", object->ptr, object->size, ptr, object->initSize);
            assert(object->size == object->initSize);
            assert(object->ptr == ptr);
            assert(object->prev == nullptr && object->next == nullptr);
        }
    };


    /**
   * A class to calculate internal fragmentation in tensor.
   * In this case, internal fragmentation of an object is just: driver reported size - allocator reported size
   * More precise internal fragmentation requires the simulation of memory allocator. Check
   *
   * The internal fragmentation is calculated as a lower-bound.
   * To calculate the all internal fragmentation, simulation is needed. Check the implementation in AllocatorStatusInternalFragTorchSimulation.
   * @tparam FRAMEWORK_TENSOR_TYPE: A Tensor that has mlinsight::MemLeak::FrameworkTensorMixin.
   */
    template<typename DRIVER_TENSOR_TYPE, typename FRAMEWORK_TENSOR_TYPE>
    class TorchSimpleSimuMetric:public CompleteCallback<DRIVER_TENSOR_TYPE, FRAMEWORK_TENSOR_TYPE> {
    public:
        ssize_t internalFragmentation = 0, maxInternalFragmentation = 0;
    public:
        
        /**
         * [Interface]
         * Check AllocatorStatus::onPostAllocFramework
         */
        void onPostAllocFramework(ssize_t size, void *ptr, FRAMEWORK_TENSOR_TYPE* newTensor);

        /**
        * [Interface]
        * Check AllocatorStatus::onPreFree
        */
        void onPreFreeFramework(void *ptr, FRAMEWORK_TENSOR_TYPE* justFreedTensor);
    protected:
        ssize_t lastDriverReturnedMemSize = 0;  // Records size at the driver allocation time so that we will know how much memory the CUDA allocator allocates.

        //Copied from pytorch
        static constexpr size_t kMinBlockSize =
                512; // all sizes are rounded to at least 512 bytes
        static constexpr size_t kSmallSize = 1048576; // largest "small" allocation is 1 MiB
        static constexpr size_t kSmallBuffer =
                2097152; // "small" allocations are packed in 2 MiB blocks
        static constexpr size_t kLargeBuffer =
                20971520; // "large" allocations may be packed in 20 MiB blocks
        static constexpr size_t kMinLargeAlloc =
                10485760; // allocations between 1 and 10 MiB may use kLargeBuffer
        static constexpr size_t kRoundLarge = 2097152; // round up large allocations to 2 MiB

        static size_t round_size(size_t size) {
            if (size < kMinBlockSize) {
                return kMinBlockSize;
            } else {
                return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
            }
        }
    };

}



namespace mlinsight::MemLeak::AllocStats {
    /**
     * A metric used to calculate the external
     */
    template<typename DRIVER_TENSOR_TYPE, typename FRAMEWORK_TENSOR_TYPE>
    class Metric:public CompleteCallback<DRIVER_TENSOR_TYPE, FRAMEWORK_TENSOR_TYPE> {
    public:
        ssize_t curActive = 0; //Current active memory from Pytorch Allocator
        ssize_t curReserve = 0;
        ssize_t nonReleasable = 0;
        ssize_t requestSizeAtMaxExternalFrag = 0, maxReserveAtMaxExternalFrag = 0;

        const int STAT_ALL=0;
        const int STAT_SMALL_POOL=1;
        const int STAT_LARGE_POOL=2;

        /**
         * Note that General::GeneralMetric must be calculated before external fragmentation
         * @param frameworkMetric
         */
        Metric() {
        }

    public:

        void onPostAllocFramework(ssize_t size, void *ptr, FRAMEWORK_TENSOR_TYPE* newTensor){

            /**
             * The following code is equivalent to:
             * reserved = torch.cuda.memory_reserved()
             * active = torch.cuda.memory_stats().get("active_bytes.all.current",0)
             */
            int deviceId=0;
            CUDA_ASSERT(cudaGetDevice(&deviceId));//todo: Cache deviceID per process
            //Ref: check statArrayToDict to check the meaning of
            this->curActive = allocatorProxy->realAllocator->getDeviceStats(deviceId).active_bytes[STAT_ALL].current;
            this->curReserve = allocatorProxy->realAllocator->getDeviceStats(deviceId).reserved_bytes[STAT_ALL].current;
            this->nonReleasable = allocatorProxy->realAllocator->getDeviceStats(deviceId).inactive_split_bytes[STAT_ALL].current;
        }

        void onPostFreeFramework(void *ptr,FRAMEWORK_TENSOR_TYPE* justFreedTensor) {
            int deviceId=0;
            CUDA_ASSERT(cudaGetDevice(&deviceId));//todo: Cache deviceID per process
            //Ref: check statArrayToDict to check the meaning of
            this->curActive = allocatorProxy->realAllocator->getDeviceStats(deviceId).active_bytes[STAT_ALL].current;
            this->curReserve = allocatorProxy->realAllocator->getDeviceStats(deviceId).reserved_bytes[STAT_ALL].current;
            this->nonReleasable = allocatorProxy->realAllocator->getDeviceStats(deviceId).inactive_split_bytes[STAT_ALL].current;
        }

    };


}


namespace mlinsight {

    template<typename CTENSOR_TYPE>
    class AliveObjectsRecord{
    public:
        // We will maintain a map between the object address and its related information :
        // such as allocatedSize, callsite. Such a map is important, as there is no allocatedSize information for deallocations.
        // In the end, we will use this map to find out driverMemRecord leaks.
        typedef typename std::unordered_map<void *, CTENSOR_TYPE>::iterator AliveObjIterator;

        AliveObjIterator onAlloc(ssize_t size, void *ptr) {
            auto emplaceRet = mapAliveObjs.try_emplace(ptr, CTENSOR_TYPE(size, ptr));
            assert(emplaceRet.second); //If the interception is complete, then unallocated address should not appear in mapAliveObjs
            return std::move(emplaceRet.first); //Save this iterator so that it can be used by subclass without looking up object again.
        }


        /**
         * Erase ptr from mapAliveObjs but do not free memory
         * @param ptr GPU memory pointer
         * @return Pointer to the erased TensorObject
         */
        AliveObjIterator onFree(void *ptr) {
            auto iterAliveObj = mapAliveObjs.find(ptr);
            return std::move(iterAliveObj);
        }

        void manualFreeObj(AliveObjIterator& iter) {
            mapAliveObjs.erase(iter);
        }
    protected:
        std::unordered_map<void *, CTENSOR_TYPE> mapAliveObjs;
    };


}



#endif //USE_TORCH

#endif //__MEMORY_LEAK_METRICS_PYTORCH_H__