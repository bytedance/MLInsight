/*
 * This file contains metric that is framework independent
 * @author: Steven (Jiaxun) Tang <jtang@umass.edu>
 * @author: Tongping Liu <tongping.liu@bytedance.com>
*/

#ifndef MLINSIGHT_MEMLEAKMETRICS_H
#define MLINSIGHT_MEMLEAKMETRICS_H

#include <sys/types.h>
#include <unordered_map>
#include "common/LinkedList.h"
#include "common/HashAndCompareFunctions.h"
#include <sys/types.h>
#include "MemLeakMetrics.h"
#include "common/CallStack.h"
#include "common/CUDAHelper.h"
#include "analyse/CallBackInterface.h"


namespace mlinsight::MemLeak {

    /**
     * This class includes the implementation to calculate base metrics that should be available in every allocator.
     */
    template<typename CTENSOR_TYPE>
    class GeneralMetric: public SimpleCallback<CTENSOR_TYPE> {
    public:
        /*
         * Add all subclasses here as friend. This not only let subclasses access private and protected members, but also let the users quickly identify different subclass in baseclass.
         * If a subclass is disabled by some macro, then it should be fine because freind is just a decleration.
         */

        /**
         * If maintainMapAliveObjs==false then this function will update stats only without maintaining mapAliveObjs.
         * Instead, mapAliveObjs will be maintained by subclass eg:AllocatorStatusInternalFragTorchSimulation
         */


        ssize_t curUsage = 0, peakUsage = 0;
        ssize_t numAllocs = 0, memAllocs = 0; //Total number and size of all allocations.
        ssize_t numFrees = 0, memFrees = 0; //Total number and size of all frees.

        // Tracking alive objects that are allocated but not freed objects
        // This is important to analyze OOM failures
        ssize_t numAliveObjs = 0; // total driverMemRecord of all alive objects
        ssize_t memAliveObjs = 0; // total driverMemRecord of all alive objects

    public:
        /**
         * [Interface]
         * Invoked after the allocator allocates memory.  Insert a new Tensor into mapAliveObjs.
         * For each allocation, the sequence is [onPostAlloc(...... AllocationType::Driver] -> [onPostAlloc(...... AllocationType::Framework]
         * @param size The size of the allocation
         * @param ptr Memory pointer. This might be null if the user passes a null pointer
         * @param type Indicate whether this is a driver allocation or framework allocation.
         */
        void onPostAlloc(ssize_t size, void *ptr, CTENSOR_TYPE *newTensor) {
            this->numAllocs += 1;
            if(newTensor){
                //This redirection gives sub-classes the opportunity to pass false to maintainMapAliveObjs
                this->memAllocs += size;
                numAliveObjs += 1;
                curUsage += size;
                if (curUsage > peakUsage) {
                    peakUsage = curUsage;
                }
            }else{
                //The allocation from the Pytorch allocator somehow failed, do not increase usage or add alive objects.
            }
        }
        /**
        * [Interface]
        * Invoked before the allocator frees memory. Remove a new Tensor from mapAliveObjs.
        * For each allocation, the sequence is [onPostAlloc(...... AllocationType::Framework] -> [onPostAlloc(...... AllocationType::Driver]
        * @param ptr Memory pointer
        * @param type Indicate whether this is a driver allocation or framework allocation.
        */
        void onPreFree(void *ptr, CTENSOR_TYPE *justFreedTensor) {
            this->numFrees += 1;
            if(justFreedTensor){
                assert(ptr);
                this->memFrees += justFreedTensor->size;
                //This pointer is in mapAliveObjs
                numAliveObjs -= 1;
                curUsage -= justFreedTensor->size;
            }else{
                //The user somehow passed a nullptr to the free function. Do not adjust alive objects.
            }
        }
    };

    /**
    * This class includes the metrics necessary to calculate between allocators
    */
    template<typename DRIVER_CTENSOR_TYPE,typename FRAMEWORK_CTENSOR_TYPE>
    class FrameworkGeneralMetric: public CompleteCallback<DRIVER_CTENSOR_TYPE,FRAMEWORK_CTENSOR_TYPE> {
    public:
        // Some information about cudaMalloc and cudaFree
        ssize_t countCudaMallocs=0;
        ssize_t countCudaFrees=0;
        ssize_t memCudaFrees=0;
        ssize_t curReserve=0, peakReserve=0;
        ssize_t nonFrameworkAllocatorNum = 0;
        ssize_t nonFrameworkAllocatorMem = 0;

        GeneralMetric<FRAMEWORK_CTENSOR_TYPE> frameworkGeneral;

        /**
        * [Interface]
        */
        void onPostAllocDriver(ssize_t size, void *ptr, DRIVER_CTENSOR_TYPE* newTensor);

        /**
        * [Interface]
        */
        void onPostAllocFramework(ssize_t size, void *ptr, FRAMEWORK_CTENSOR_TYPE* newTensor) {
            frameworkGeneral.onPostAlloc(size, ptr, newTensor);
            if(frameworkGeneral.curUsage > this->curReserve){
                fatalErrorS("frameworkGeneral.curUsage=%zd this->curReserve=%zd", frameworkGeneral.curUsage, this->curReserve);
            }
            assert(frameworkGeneral.curUsage <= this->curReserve);
        }

        void onPreFreeFramework(void *ptr, FRAMEWORK_CTENSOR_TYPE* justFreedTensor) {
            frameworkGeneral.onPreFree(ptr, justFreedTensor);
            assert(frameworkGeneral.curUsage <= this->curReserve);
        }


        void onPreFreeDriver(void *ptr, DRIVER_CTENSOR_TYPE* justFreedTensor);


    };


}
namespace mlinsight::MemLeak::Driver {
    enum class Type {
        UNSPECIFIED = 0,
        CUDA = 1
    };
    template<typename CTENSOR_TYPE, Type driverType>
    class Metric:public SimpleCallback<CTENSOR_TYPE>{
    public:
        /**
         * [Interface]
         * Invoked after the allocator allocates memory.  Insert a new Tensor into mapAliveObjs.
         * For each allocation, the sequence is [onPostAlloc(...... AllocationType::Driver] -> [onPostAlloc(...... AllocationType::Framework]
         * @param size The size of the allocation
         * @param ptr Memory pointer
         * @param type Indicate whether this is a driver allocation or framework allocation.
         */
        void onPostAlloc(ssize_t size, void *ptr, CTENSOR_TYPE* newTensor) {
            fatalError("Do not use base class");
        }

        /**
        * [Interface]
        * Invoked before the allocator frees memory. Remove a new Tensor from mapAliveObjs.
        * For each allocation, the sequence is [onPostAlloc(...... AllocationType::Framework] -> [onPostAlloc(...... AllocationType::Driver]
        * @param ptr Memory pointer
        * @param type Indicate whether this is a driver allocation or framework allocation.
        */
        void onPostFree(void *ptr, CTENSOR_TYPE* justFreedTensor) {
            fatalError("Do not use base class");
        }

    };

    template<typename CTENSOR_TYPE>
    class Metric<CTENSOR_TYPE,Type::CUDA>:public SimpleCallback<CTENSOR_TYPE>{
    public:
        size_t peakGPUMem, freeMem, totalMem;

    public:
        /**
         * [Interface]
         * Invoked after the allocator allocates memory.  Insert a new Tensor into mapAliveObjs.
         * For each allocation, the sequence is [onPostAlloc(...... AllocationType::Driver] -> [onPostAlloc(...... AllocationType::Framework]
         * @param size The size of the allocation
         * @param ptr Memory pointer
         * @param type Indicate whether this is a driver allocation or framework allocation.
         */
        void onPostAlloc(ssize_t size, void *ptr, CTENSOR_TYPE* newTensor) {
            CUDA_ASSERT(cudaMemGetInfo(&freeMem, &totalMem));
            peakGPUMem = std::max(peakGPUMem, totalMem - freeMem);
        }

        /**
        * [Interface]
        * Invoked before the allocator frees memory. Remove a new Tensor from mapAliveObjs.
        * For each allocation, the sequence is [onPostAlloc(...... AllocationType::Framework] -> [onPostAlloc(...... AllocationType::Driver]
        * @param ptr Memory pointer
        * @param type Indicate whether this is a driver allocation or framework allocation.
        */
        void onPostFree(void *ptr, CTENSOR_TYPE* justFreedTensor) {
            CUDA_ASSERT(cudaMemGetInfo(&freeMem, &totalMem));
            peakGPUMem = std::max(peakGPUMem,totalMem - freeMem);
        }


      
    };
}
namespace mlinsight::MemLeak::InternalFrag {
    /**
     * The FrameworkTensorMixin necessary for all classes in mlinsight::MemLeak::InternalFrag
     */
    class TensorMixin {
    public:
        ssize_t internalFragmentation;

        TensorMixin(ssize_t size, void *ptr){
            //Do not need to do anything here.
        }
    };


}

namespace mlinsight::MemLeak::ExternalFrag {


//    extern __thread Info threadLocalInfo; //This is not a problem as one thread can only access one variable at a time

    /**
     * A caching allocator consists of a driver part and a framework part.
     * Some metrics are directly calculated as
     *
     * @tparam DRIVER_LEAK_OBJ_TYPE
     * @tparam FRAMEWORK_LEAK_OBJ_TYPE
     */
    template<typename DRIVER_TENSOR_TYPE, typename FRAMEWORK_TENSOR_TYPE>
    class Metric:public CompleteCallback<DRIVER_TENSOR_TYPE, FRAMEWORK_TENSOR_TYPE> {
    public:
        ssize_t maxExternalFrag = 0;
        ssize_t requestSizeAtMaxExternalFrag = 0, maxReserveAtMaxExternalFrag = 0;

        FrameworkGeneralMetric<DRIVER_TENSOR_TYPE,FRAMEWORK_TENSOR_TYPE>& frameworkMetric;

        /**
         * Note that General::GeneralMetric must be calculated before external fragmentation
         * @param frameworkMetric
         */
        Metric(FrameworkGeneralMetric<DRIVER_TENSOR_TYPE,FRAMEWORK_TENSOR_TYPE>& frameworkMetric): frameworkMetric(frameworkMetric){

        }

    public:

        void onPostAllocFramework(ssize_t size, void *ptr, FRAMEWORK_TENSOR_TYPE* newTensor);
    };


}




#endif //MLINSIGHT_MEMLEAKMETRICS_H
