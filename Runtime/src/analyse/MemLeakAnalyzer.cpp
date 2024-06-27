#include "analyse/MemLeak/MemLeakMetrics.h"
#include "analyse/MemLeak/MemoryLeakMetrics_Tensorflow.h"
#include "analyse/MemLeak/MemoryLeakMetrics_Pytorch.h"
#include "analyse/GlobalVariables.h"
#include "common/CallStack.h"
namespace mlinsight::MemLeak {
    //Explicitly initialize
    template class FrameworkGeneralMetric<DriverTensorType,FramekworkTensorType>;
    /**
     * Definition moved here to avoid recursive inclusion of GlobalVariable.h
     * @tparam DRIVER_CTENSOR_TYPE
     * @tparam FRAMEWORK_CTENSOR_TYPE
     * @param size
     * @param ptr
     * @param newTensor
     */
    template<typename DRIVER_CTENSOR_TYPE,typename FRAMEWORK_CTENSOR_TYPE>
    void FrameworkGeneralMetric<DRIVER_CTENSOR_TYPE,FRAMEWORK_CTENSOR_TYPE>::onPostAllocDriver(ssize_t size, void *ptr, DRIVER_CTENSOR_TYPE* newTensor){
        if(globalExecutionState.isInvokingFrameworkMemOp){
            this->countCudaMallocs+=1;
            if(newTensor){
                assert(ptr);
                this->curReserve+=size;
                this->peakReserve=std::max(this->peakReserve,this->curReserve);
            }else{
                //This allocation is related to framework but failed, so we only record the counter but do not incrase curReserve.
            }
        }else{
            INFO_LOGS("[ALLOC] !isInvokingFrameworkMemOp for Tensor %p",newTensor);
            // This allocation is not from the Pytorch memory allocator
            //print_hybridstacktrace();
            if(newTensor){
                assert(ptr);
                nonFrameworkAllocatorNum+=1;
                assert(size>0);
                assert(size==newTensor->size);
                nonFrameworkAllocatorMem+=size;
            }
        }

    }

    template<typename DRIVER_CTENSOR_TYPE,typename FRAMEWORK_CTENSOR_TYPE>
    void FrameworkGeneralMetric<DRIVER_CTENSOR_TYPE,FRAMEWORK_CTENSOR_TYPE>::onPreFreeDriver(void *ptr, DRIVER_CTENSOR_TYPE* justFreedTensor){
        if(justFreedTensor){
            if(justFreedTensor->isAllocatedByFramework){
                this->countCudaFrees+=1;
                assert(ptr);
                this->memCudaFrees+=justFreedTensor->size;
                this->curReserve-=justFreedTensor->size;
            }else{
                // This allocation is not from the Pytorch memory allocator
                assert(ptr);
                nonFrameworkAllocatorNum-=1;
                assert(justFreedTensor->size>0);
                // if(nonFrameworkAllocatorMem<justFreedTensor->size){
                //     INFO_LOGS("nonFrameworkAllocatorMem=%zd justFreedTensor->size=%zd",nonFrameworkAllocatorMem,justFreedTensor->size);
                // }
                INFO_LOGS("[FREE] !isInvokingFrameworkMemOp for Tensor %p",justFreedTensor);

                INFO_LOGS("nonFrameworkAllocatorMem=%zd justFreedTensor->size=%zd",nonFrameworkAllocatorMem,justFreedTensor->size);

                fflush(logFileStd);
                assert(nonFrameworkAllocatorMem>=justFreedTensor->size);
                nonFrameworkAllocatorMem-=justFreedTensor->size;
                
                assert(nonFrameworkAllocatorMem>=0);
            }
        }
        

    }
}
namespace mlinsight{

    template class MemLeakAnalyzer<DriverTensorType,FramekworkTensorType>;

    template<typename DRIVER_CTENSOR_TYPE, typename FRAMEWORK_CTENSOR_TYPE>
    void MemLeakAnalyzer<DRIVER_CTENSOR_TYPE, FRAMEWORK_CTENSOR_TYPE>::printFrameworkInfo(std::ostream &output) {
        output << std::endl;
        // Printing the pytorch information
        output << "Pytorch GPU information: current reserve  - " << format_size(frameworkInfo.curReserve) << ". Peak reserve - " << format_size(frameworkInfo.peakReserve) << std::endl;
        output << "\t Number of cudaMalloc: " << frameworkInfo.countCudaMallocs << std::endl;
        output << "\t Number of cudaFree: " << frameworkInfo.countCudaFrees << std::endl;
        output << "\t Number of allocations: " << frameworkInfo.frameworkGeneral.numAllocs << std::endl;
        output << "\t Total allocated memory: " << format_size(frameworkInfo.frameworkGeneral.memAllocs) << std::endl;
        output << "\t Number of frees:" << frameworkInfo.frameworkGeneral.numFrees << std::endl;
        output << "\t Total freed memory:" << format_size(frameworkInfo.frameworkGeneral.memFrees) << std::endl;
        //output << std::endl;
        output << "\t Number of alive objects: " <<  mapFrameworkAliveObjs.getSize() << std::endl;
        output << "\t Memory of alive objects: " << format_size(frameworkInfo.frameworkGeneral.curUsage) << std::endl;
        output << "\t Total internal fragmentation of alive objects: " << format_size(internalFragMetric.internalFragmentation) << ". Maximum fragmentation:" << format_size(internalFragMetric.maxInternalFragmentation) << std::endl;
        output << "\t Maximum external fragmentation: " << format_size(externalFragMetric.maxExternalFrag) << ", where memory reserve at " << format_size(externalFragMetric.maxReserveAtMaxExternalFrag) << " and request size " <<  format_size(externalFragMetric.requestSizeAtMaxExternalFrag) << "." << std::endl;
        output << "\t Number of freed objects: " << frameworkInfo.frameworkGeneral.numFrees << std::endl;
        output << "\t Memory of freed objects: " << format_size(frameworkInfo.frameworkGeneral.memFrees) << std::endl;
        output << "\t Total available memory in allocator: " << format_size(frameworkInfo.curReserve - frameworkInfo.frameworkGeneral.curUsage) << std::endl;
        output << std::endl;
    }

    template<typename DRIVER_CTENSOR_TYPE, typename FRAMEWORK_CTENSOR_TYPE>
    void MemLeakAnalyzer<DRIVER_CTENSOR_TYPE, FRAMEWORK_CTENSOR_TYPE>::printOutput(std::ostream &output, ssize_t oomAllocSize) {

        this->printSummary(output,oomAllocSize);
        output << "****************************************************" << std::endl;
        output << "memory profile is shown as follows:" << std::endl;
        output << "****************************************************" << std::endl;
         this->printDriver(output,oomAllocSize);
         this->printFramework(output,oomAllocSize);
    }

    template<typename DRIVER_CTENSOR_TYPE, typename FRAMEWORK_CTENSOR_TYPE>
    void MemLeakAnalyzer<DRIVER_CTENSOR_TYPE, FRAMEWORK_CTENSOR_TYPE>::printSummary(std::ostream &output, ssize_t oomAllocSize) {
       this->printBasicInfo(output,oomAllocSize);
        output << std::endl;
    }

    template<typename DRIVER_CTENSOR_TYPE, typename FRAMEWORK_CTENSOR_TYPE>
    void MemLeakAnalyzer<DRIVER_CTENSOR_TYPE, FRAMEWORK_CTENSOR_TYPE>::printFramework(std::ostream &output, ssize_t oomAllocSize) {

        printFrameworkInfo(output);
        this->printLeakyObjects<FRAMEWORK_CTENSOR_TYPE, PyCallStack>(output,mapFrameworkAliveObjs);

    }

    template<typename DRIVER_CTENSOR_TYPE, typename FRAMEWORK_CTENSOR_TYPE>
    void MemLeakAnalyzer<DRIVER_CTENSOR_TYPE, FRAMEWORK_CTENSOR_TYPE>::printDriver(std::ostream &output, ssize_t oomAllocSize) {

        printDriverInfo(output);
        this->printLeakyObjects<DRIVER_CTENSOR_TYPE, CCallStack>(output,mapDriverAliveObjs);
    }


}

namespace mlinsight::MemLeak::InternalFrag {
    template class TorchSimpleSimuMetric<DriverTensorType,FramekworkTensorType>;


    template<typename DRIVER_TENSOR_TYPE, typename FRAMEWORK_TENSOR_TYPE>
    void TorchSimpleSimuMetric<DRIVER_TENSOR_TYPE, FRAMEWORK_TENSOR_TYPE>::onPostAllocFramework(ssize_t size, void *ptr,
                                                                                 FRAMEWORK_TENSOR_TYPE *newTensor) {
        //For an allocation through the Pytorch allocator, this branch must be the second invoked branch.

        if(newTensor){
            //Get the allocated tensor representation directly from parent to save one hashmap lookup time
            assert(ptr!=nullptr);
            assert(newTensor->ptr == ptr);

            auto *obj = static_cast<TensorMixin *>(newTensor);

            //Save internalFragmentation along with tensor object so that we can get this value on free.
            obj->internalFragmentation = round_size(size) - size;
            assert(obj->internalFragmentation >= 0);

            this->internalFragmentation += obj->internalFragmentation;
            this->maxInternalFragmentation = std::max(this->internalFragmentation,
                                                        this->maxInternalFragmentation);
        } else {
            //The pointer is empty, so there is no need to record fragmentation
        }
    }

    template<typename DRIVER_TENSOR_TYPE, typename FRAMEWORK_TENSOR_TYPE>
    void TorchSimpleSimuMetric<DRIVER_TENSOR_TYPE, FRAMEWORK_TENSOR_TYPE>::onPreFreeFramework(void *ptr,
                                                                               FRAMEWORK_TENSOR_TYPE *justFreedTensor) {
        if(justFreedTensor){
            assert(ptr);
            //For an allocation through the Pytorch allocator, this branch must be the first invoked branch.
            auto* obj = static_cast<TensorMixin*>(justFreedTensor);
            this->internalFragmentation -= justFreedTensor->internalFragmentation;
        }else{
            //The pointer is empty, this may be caused by the user passing a nullptr.
        }
    }
}

namespace mlinsight::MemLeak::ExternalFrag{
    template class Metric<DriverTensorType,FramekworkTensorType>;


    template<typename DRIVER_TENSOR_TYPE, typename FRAMEWORK_TENSOR_TYPE>
    void Metric<DRIVER_TENSOR_TYPE, FRAMEWORK_TENSOR_TYPE>::onPostAllocFramework(ssize_t size, void *ptr,
                                                                                 FRAMEWORK_TENSOR_TYPE *newTensor) {
        if(this->frameworkMetric.curReserve < this->frameworkMetric.frameworkGeneral.curUsage){
            fatalErrorS("this->frameworkMetric.curReserve=%zd this->frameworkMetric.frameworkGeneral.curUsage=%zd",
                        this->frameworkMetric.curReserve, this->frameworkMetric.frameworkGeneral.curUsage)
        }
        assert(this->frameworkMetric.curReserve >= this->frameworkMetric.frameworkGeneral.curUsage);

        ssize_t curExternalFragmentation=this->frameworkMetric.curReserve - this->frameworkMetric.frameworkGeneral.curUsage;
        if(curExternalFragmentation > this->maxExternalFrag){
            this->maxExternalFrag = curExternalFragmentation;
            this->maxReserveAtMaxExternalFrag = this->frameworkMetric.curReserve;
            this->requestSizeAtMaxExternalFrag = size;
        }

        //onPostAllocFramework must be called immediately after onPostAllocDriver. So it is safe to
    }
}
