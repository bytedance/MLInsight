//
// Created by user on 4/8/24.
//

#ifndef MLINSIGHT_TENSORMAP_H
#define MLINSIGHT_TENSORMAP_H
#include <unordered_map>
#include "common/MemoryHeap.h"

namespace mlinsight{
/**
 * A class that holds tensors in hashmap by pointer.
 * TensorMaps may be used by different analysis types. For example, MemLeak analysis might want to know the alive tensors.
 * Memory flamegraph may need to fetch torch objects by its pointers.
 * The purpose of creating a separate class to wrap an map is:
 * 1. Analyzers may need to access Tensor memory after it is removed from the alive objects map. So we need to keep object
 * not-freed even if it has just been moved out of the map.
 * 2. It is unclear when different types of analyzers should free memory. So memory free should be handled by a specific class.
 *
 * This TensorMaps operator work like this:
 *
 * 1. Allocation events happened (e.g.: onDriverAlloc)
 * 2. DriverTensor* newlyAllocatedTensor = TensorMaps<DRIVER_CTENSOR_TYPE>.onDriverAlloc (The tensor map size += 1)
 * 3. Analyzer1.onDriverAlloc(......, newlyAllocatedTensor); ...... AnalyzerN.onDriverAlloc(newlyAllocatedTensor);
 *
 * 1. DeAllocation events happened (e.g.: onDriverFree)
 * 2. DriverTensor* justFreedTensor = TensorMaps<DRIVER_CTENSOR_TYPE>.onDriverFree()  (The tensor map size -= 1, but the memory is not freed)
 * 3. Analyzer1.onDriverFree(......, justFreedTensor); ...... AnalyzerN.onDriverFree(justFreedTensor);
 * 4. TensorMaps<DRIVER_CTENSOR_TYPE>.freeObject(justFreedTensor);
 *
 *
 * @tparam FRAMEWORK_CTENSOR_TYPE
 * @tparam DRIVER_CTENSOR_TYPE
 */
template<typename CTENSOR_TYPE>
class TensorMaps{
public:
    std::unordered_map<void *, CTENSOR_TYPE*> mapObj;
public:

    /**
       * [Interface]
       * Invoked after the allocator allocates memory.  Insert a new Tensor into mapAliveObjs.
       * For each allocation, the sequence is [onPostAlloc(...... AllocationType::Driver] -> [onPostAlloc(...... AllocationType::Framework]
       * @param size The size of the allocation
       * @param ptr Memory pointer
       * @param type Indicate whether this is a driver allocation or framework allocation.
       */
    CTENSOR_TYPE* insert(ssize_t size, void *ptr) {
        if(!ptr){
            return nullptr;
        }
        auto* newTensor= objPool.alloc();
        new (newTensor) CTENSOR_TYPE(size, ptr);
        mapObj.emplace(ptr,newTensor);
        // auto emplaceRet = mapObj.try_emplace(ptr, newTensor);
        // assert(emplaceRet.second); //If the interception is complete, then unallocated address should not appear in mapAliveObjs
        // assert(newTensor!=nullptr);
        return newTensor;
    }


    /**
    * [Interface]
    * Invoked before the allocator frees memory. Remove a new Tensor from mapAliveObjs.
    * For each allocation, the sequence is [onPostAlloc(...... AllocationType::Framework] -> [onPostAlloc(...... AllocationType::Driver]
    * @param ptr Memory pointer. The user may pass nullptr to this parameter.
    * @param type Indicate whether this is a driver allocation or framework allocation.
    * @return The newly removed memory address. This function will return nullptr if ptr is null. This is possible when the user passes null to the parameter.
    */
    CTENSOR_TYPE* remove(void *ptr) {
        if(!ptr){
            return nullptr;
        }
        auto findIter = mapObj.find(ptr);
        if(findIter!=mapObj.end()){
            CTENSOR_TYPE*  retTensor = findIter->second;
            mapObj.erase(findIter);
            //Do not free object. Only free in freeTensorMemory;
            return retTensor;
        }else{
            return nullptr;
        }
    };

    /**
     * Actually free the memory of the last onPreFree return.
     * Currently it only supports freeing the last tensor to avoid data leaks. If need to preserve tensors, the analyzer should free
     */
    void erase(CTENSOR_TYPE* tensorPtr){
       tensorPtr->~CTENSOR_TYPE();
       objPool.dealloc(tensorPtr);
    }

    ssize_t getSize(){
        return mapObj.size();
    }
protected:
    ObjectPoolHeap<CTENSOR_TYPE> objPool;
};


/**
 * Singular metrics that can be calculated by events from either the driver or the tensor
 */
}
#endif //MLINSIGHT_TENSORMAP_H
