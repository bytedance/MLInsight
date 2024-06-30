#ifndef MLINSIGHT_MEMINCREMENTALANALYZER_H
#define MLINSIGHT_MEMINCREMENTALANALYZER_H

#include "analyse/MemLeak/MemLeakMetrics.h"
#include "analyse/MemLeak/MemoryLeakMetrics_Pytorch.h"
//#include "analyse/TensorMap.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include "analyse/TensorMap.h"
#include "analyse/GlobalVariables.h"
#include "analyse/CallBackInterface.h"
#include "analyse/MemLeak/MemoryLeakMetrics_Pytorch.h"

namespace mlinsight {


    /* For OOM, there are four reasons:
    1. external fragmentation (driverMemRecord blocks inside the torch allocator but can't be used for large allocation due to discontinous objects)
    2. internal fragmentation (how much driverMemRecord wasted due to unaligned driverMemRecord allocations)
    3. Memory leaks from specific callsites.
    4. Actual driverMemRecord usage is larger than the capacity of GPU driverMemRecord

    In first stage, we aims to understand the possibility of each reason, but not necessarily of
    the detailed information.
       1. For external fragmentation, we could track all freed objects inside the torch allocator and the pointer of un-used driverMemRecord
       2. For internal fragmentation, we will track the total waste for each allocation, and deduct it for each free
       3. For driverMemRecord leaks, we could track the number of allocations and deallocations for each cycle. However, how can we know the cycle?
          Or we could just use the trend of allocations (we will use 100 allocations as a pseudo cycle)
       4. For actual driverMemRecord usage, driverRecord + nondriver > capacity
    */
    template<typename DRIVER_CTENSOR_TYPE, typename FRAMEWORK_CTENSOR_TYPE>
    class MemIncrementalAnalyzer: public CompleteCallback<DRIVER_CTENSOR_TYPE,FRAMEWORK_CTENSOR_TYPE> {
    public:
        MemIncrementalAnalyzer(){

        }
        /**
       * [Interface]
       */
        void onPreAllocFramework(ssize_t size){

        }
        /**
        * [Interface]
        */
        void onPreAllocDriver(ssize_t size) {

        }

        /**
        * [Interface]
        */
        void onPostAllocDriver(ssize_t size, void *ptr, DRIVER_CTENSOR_TYPE *newTensor) {

        }

        /**
         * [Interface]
         */
        void onPostAllocFramework(ssize_t size, void *ptr, FRAMEWORK_CTENSOR_TYPE *newTensor) {

        }

        /**
        * [Interface]
        * Invoked before the allocator frees memory. Remove a new Tensor from mapAliveObjs.
        * For each allocation, the sequence is [onPostAlloc(...... AllocationType::Framework] -> [onPostAlloc(...... AllocationType::Driver]
        * @param ptr Memory pointer
        * @param type Indicate whether this is a driver allocation or framework allocation.
        */
        void onPreFreeFramework(void *ptr, FRAMEWORK_CTENSOR_TYPE *justFreedTensor) {

        }


        /**
        * [Interface]
        * Invoked before the allocator frees memory. Remove a new Tensor from mapAliveObjs.
        * For each allocation, the sequence is [onPostAlloc(...... AllocationType::Framework] -> [onPostAlloc(...... AllocationType::Driver]
        * @param ptr Memory pointer
        * @param type Indicate whether this is a driver allocation or framework allocation.
        */
        void onPreFreeDriver(void *ptr, DRIVER_CTENSOR_TYPE *justFreedTensor) {


        }

        /**
        * [Interface]
        */
        void onPostFreeDriver(void *ptr,DRIVER_CTENSOR_TYPE* justFreedTensor){

        }

        void onPostFreeFramework(void *ptr,FRAMEWORK_CTENSOR_TYPE* justFreedTensor) {

        }

        void onPreLayerForward(ssize_t layerId, MLExecutionStackFrame &curExecState) {

        }

        void onPostLayerForward(ssize_t layerId, MLExecutionStackFrame &curExecState) {

        }

        void onPreLayerBackward(ssize_t layerId, MLExecutionStackFrame &executionState) {

        }

        void onPostLayerBackward(ssize_t layerId, MLExecutionStackFrame &curExecState) {

        }
    };


    extern MemIncrementalAnalyzer<DriverTensorType, FramekworkTensorType> memIncrementalAnalyzer;


}

#endif //MLINSIGHT_MEMINCREMENTALANALYZER_H
