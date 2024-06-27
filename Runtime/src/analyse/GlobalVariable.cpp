/*
* @author: Steven (Jiaxun) Tang <jtang@umass.edu>
* @author: Tongping Liu <tongping.liu@bytedance.com>
*/
#include <thread>         // std::this_thread::sleep_for

#include "common/Tool.h"
#include "trace/hook/PyHook.h"
#include "analyse/GlobalVariables.h"


namespace mlinsight {
    //PytorchMemRecord torchMem; todo: Move to flametree analysis types


    /**
     * =================================================================================================================
     * Pytorch allocator simulation related code
     * @tparam TENSOR_TYPE
     * @param ptr
     * =================================================================================================================
     */

    static ssize_t allocationCounter = 0; //For debugging flamegraph purpose


    PythonExecutionState<FramekworkTensorType> globalExecutionState;

    FlameGraphAnalyser<FramekworkTensorType> flameGraphAnalyser;

    MemLeakAnalyzer<DriverTensorType,FramekworkTensorType> memLeakAnalyzer;

    DebugAnalyzer<DriverTensorType,FramekworkTensorType> debugAnalyzer;
#if USE_PERFETTO
    PerfettoTensorTraceAnalyser<DriverTensorType, FramekworkTensorType> perfettoAnalyzer(memLeakAnalyzer);
#endif
    TensorMaps<FramekworkTensorType> mapFrameworkAliveObjs;

    TensorMaps<DriverTensorType> mapDriverAliveObjs;

    pthread_mutex_t analyzerLock;
#ifndef NDEBUG
    CuptiCrossChecker cuptiCrossChecker;
#endif

}


