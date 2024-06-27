#ifndef DEBUG_ANALYZER_H
#define DEBUG_ANALYZER_H
//
// Created by user on 4/24/24.
//
#include "analyse/TensorMap.h"
#include "analyse/CallBackInterface.h"
#include "trace/proxy/PytorchMemProxy.h"
#if USE_TORCH
namespace mlinsight {

    /**
     * Debug analyzer to print out pytorch framework log events to the log file.
     * It is only used to analyze allocations and de-allocations in python.
     */
    template<typename DRIVER_CTENSOR_TYPE, typename FRAMEWORK_CTENSOR_TYPE>
    class DebugAnalyzer: public CompleteCallback<DRIVER_CTENSOR_TYPE,FRAMEWORK_CTENSOR_TYPE> {
    public:
        DebugAnalyzer();

        /**
       * [Interface]
       */
        void onPreAllocFramework(ssize_t size);

        /**
        * [Interface]
        */
        void onPreAllocDriver(ssize_t size);

        /**
        * [Interface]
        */
        void onPostAllocDriver(ssize_t size, void *ptr, DRIVER_CTENSOR_TYPE *newTensor);

        /**
         * [Interface]
         */
        void onPostAllocFramework(ssize_t size, void *ptr, FRAMEWORK_CTENSOR_TYPE *newTensor);

        /**
        * [Interface]
        * Invoked before the allocator frees memory. Remove a new Tensor from mapAliveObjs.
        * For each allocation, the sequence is [onPostAlloc(...... AllocationType::Framework] -> [onPostAlloc(...... AllocationType::Driver]
        * @param ptr Memory pointer
        * @param type Indicate whether this is a driver allocation or framework allocation.
        */
        void onPreFreeFramework(void *ptr, FRAMEWORK_CTENSOR_TYPE *justFreedTensor);


        /**
        * [Interface]
        * Invoked before the allocator frees memory. Remove a new Tensor from mapAliveObjs.
        * For each allocation, the sequence is [onPostAlloc(...... AllocationType::Framework] -> [onPostAlloc(...... AllocationType::Driver]
        * @param ptr Memory pointer
        * @param type Indicate whether this is a driver allocation or framework allocation.
        */
        void onPreFreeDriver(void *ptr, DRIVER_CTENSOR_TYPE *justFreedTensor);

        /**
        * [Interface]
        */
        void onPostFreeDriver(void *ptr,DRIVER_CTENSOR_TYPE* justFreedTensor);

        void onPostFreeFramework(void *ptr,FRAMEWORK_CTENSOR_TYPE* justFreedTensor);

    };




}
#endif
#endif
