#ifndef MLINSIGHT_GLOBALVARIABLES_H
#define MLINSIGHT_GLOBALVARIABLES_H

#include "common/TensorObj.h"
#include "analyse/MemLeak/MemLeakMetrics.h"
#include "analyse/MemLeak/MemoryLeakMetrics_Tensorflow.h"
#include "analyse/FlameGraph.h"
#include "trace/hook/PyHook.h"
#include "analyse/MemLeak/MemLeakAnalyzer.h"
#include "common/Logging.h"
#include "analyse/DebugAnalyzer.h"
#if USE_PERFETTO
#include <perfetto.h>
#include "analyse/PerfettoTensorTraceAnalyzer.h"
#endif //USE_PERFETTO
/**
 * All global variables goes here
 * DEPRECATED!!!!!!
 * todo: Remove this file. Using this file will cause dependency hell.
 */
namespace mlinsight{

    typedef TensorObj<
    FrameworkTensorMixin,
    MemLeak::InternalFrag::TensorMixin,
    FlameGraph::TensorMixin
#if USE_PERFETTO
    ,Perfetto::TensorMixin
#endif //USE_PERFETTO
    > FramekworkTensorType;

    typedef TensorObj<
    DriverTensorMixin,
    MemLeak::InternalFrag::TensorMixin
#if USE_PERFETTO
    ,Perfetto::TensorMixin
#endif //USE_PERFETTO
    > DriverTensorType;

    extern PythonExecutionState<FramekworkTensorType> globalExecutionState;
    extern FlameGraphAnalyser<FramekworkTensorType> flameGraphAnalyser;
    extern MemLeakAnalyzer<DriverTensorType, FramekworkTensorType> memLeakAnalyzer;
    extern DebugAnalyzer<DriverTensorType, FramekworkTensorType> debugAnalyzer;
#if USE_PERFETTO
    extern PerfettoTensorTraceAnalyser<DriverTensorType, FramekworkTensorType> perfettoAnalyzer;
#endif

    extern bool isPerfettoEnabled;
#if USE_PERFETTO
    extern std::unique_ptr<perfetto::TracingSession> tracingSession;
#endif

    extern TensorMaps<FramekworkTensorType> mapFrameworkAliveObjs;
    extern TensorMaps<DriverTensorType> mapDriverAliveObjs;
    extern pthread_mutex_t analyzerLock;

#ifndef NDEBUG
    extern CuptiCrossChecker cuptiCrossChecker;

#endif


}



#endif //MLINSIGHT_GLOBALVARIABLES_H
