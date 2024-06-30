#ifndef MLINSIGHT_PERFETTOTENSORTRACEANALYZER_H
#define MLINSIGHT_PERFETTOTENSORTRACEANALYZER_H
#ifdef USE_PERFETTO

#include <atomic>
#include "analyse/CallBackInterface.h"
#include "trace/tool/Perfetto.h"
#include "common/Logging.h"
#include "analyse/MemLeak/MemLeakAnalyzer.h"


namespace mlinsight::Perfetto {
    /**
    * The FrameworkTensorMixin necessary for all classes in mlinsight::MemLeak::InternalFrag
    */
    class TensorMixin {
    public:
        ssize_t perfettoSequenceId=-1;
        ssize_t allocTimestamp=-1;
        ssize_t freeTimestamp=-1;
        bool skippedRecording=false; //A marker indicating whether this allocation is filtered or not
        //HybridCallStack hybridCallStack; //A temporary implementation

        TensorMixin(ssize_t size, void *ptr){
            //Do not need to do anything here.
        }
    };
}

namespace mlinsight {

    extern ssize_t stepCounter;
    template<typename DRIVER_CTENSOR_TYPE, typename FRAMEWORK_CTENSOR_TYPE>
    class PerfettoTensorTraceAnalyser: public CompleteCallback<DRIVER_CTENSOR_TYPE,FRAMEWORK_CTENSOR_TYPE> {
    public:
        MemLeakAnalyzer<DRIVER_CTENSOR_TYPE,FRAMEWORK_CTENSOR_TYPE>& memLeakAnalyzer;
        /**
         * Perfetto rely on MemLeakAnalyzer, so memleakanalyzer should be called before PerfettoTensorTraceAnalyser.
        */
        PerfettoTensorTraceAnalyser(MemLeakAnalyzer<DRIVER_CTENSOR_TYPE,FRAMEWORK_CTENSOR_TYPE>& memLeakAnalyzer):memLeakAnalyzer(memLeakAnalyzer){
            
        }


        ssize_t perfettoFlowSequenceId=0; //Used to correlate allocation with free. TODO: Use lock or atomic variable to protect. Maybe kernels will use multithread and impacts this variable.
        /**
        * [Interface]
        * ptr may be null
        */
        void onPostAllocDriver(ssize_t size, void *ptr, DRIVER_CTENSOR_TYPE* newTensor);

        /**
        * [Interface]
        * ptr may be null
        */
        void onPostAllocFramework(ssize_t size, void *ptr, FRAMEWORK_CTENSOR_TYPE* newTensor);

        /**
        * [Interface]
        * ptr may be null
        */
        void onPreFreeFramework(void *ptr, FRAMEWORK_CTENSOR_TYPE* justFreedTensor);

        /**
        * [Interface]
        * ptr may be null
        */
        void onPreFreeDriver(void *ptr, DRIVER_CTENSOR_TYPE* justFreedTensor);

        /**
        * [Interface]
        * ptr may be null
        */
        void onPostFreeDriver(void *ptr,DRIVER_CTENSOR_TYPE* justFreedTensor);

        /**
        * [Interface]
        * ptr may be null
        */
        void onPostFreeFramework(void *ptr,FRAMEWORK_CTENSOR_TYPE* justFreedTensor);

        void onStepFinished(ssize_t stepCounter);

        void onOutOfMemoryFramework(ssize_t size);

        void onPreLayerForward(ssize_t layerId, MLExecutionStackFrame &curExecState);

        void onPostLayerForward(ssize_t layerId, MLExecutionStackFrame &curExecState);

        void onPreLayerBackward(ssize_t layerId, MLExecutionStackFrame &executionState);

        void onPostLayerBackward(ssize_t layerId, MLExecutionStackFrame &curExecState);

    };
    
    extern bool shouldProfileThisRank; //A boolean indicating whether MLInsight should output any data
}


#endif //USE_PERFETTO
#endif //MLINSIGHT_PERFETTOTENSORTRACEANALYZER_H
