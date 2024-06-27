#ifndef MLINSIGHT_PERFETTO_H
#define MLINSIGHT_PERFETTO_H

#if USE_PERFETTO
#include <perfetto.h>
#else
    namespace perfetto{
        class TracingSession;
    }
#endif //USE_PERFETTO

namespace mlinsight {

    void initializePerfetto() __attribute__((weak));

//     std::unique_ptr<perfetto::TracingSession> startTracing() __attribute__((weak));

//     void saveTracingData(std::unique_ptr<perfetto::TracingSession> tracing_session);
    
//     void stopTracing(std::unique_ptr<perfetto::TracingSession> tracing_session) __attribute__((weak));

}

const int FORWARD_PROP_TRACK = 1000;
const int BACKWARD_PROP_TRACK = 1001;
const int MEMORY_FLAMEGRAPH_TRACK = 1002;
const int MEMORY_FRAGMENTATION_TRACK = 1003;
const int PYTORCH_ALLOC_FREE_TRACK = 1004;
const int NOPYTORCH_ALLOC_FREE_TRACK = 1005;
const int NONPYTORCH_ACTIVE_TRACK = 1006;
const int STEP_TRACK = 1007;
const int CALLSTACK_ID_TRACK = 1008;

#if USE_PERFETTO

namespace mlinsight{
        /**
         * A custom data source used to write tensor-related trace
         */
        class MemoryDataSource: public perfetto::DataSource<MemoryDataSource> {
        public:
            void OnSetup(const SetupArgs&) override;

            // Optional callbacks for tracking the lifecycle of the data source.
            void OnStart(const StartArgs&) override;
            void OnStop(const StopArgs&) override;
        };
}

PERFETTO_DECLARE_DATA_SOURCE_STATIC_MEMBERS(mlinsight::MemoryDataSource);

// The set of track event categories that the example is using.
PERFETTO_DEFINE_CATEGORIES(
        perfetto::Category("forward")
                .SetDescription("Forward propagation"),
        perfetto::Category("backward")
                .SetDescription("Backward propagation"),
        perfetto::Category("flamegraph")
                .SetDescription("Flamegraph propagation"),
        perfetto::Category("MemStats")
                .SetDescription("Memory fragmentation"),
        perfetto::Category("tensor")
                .SetDescription("Tensor allocation and free"),
        perfetto::Category("rendering")
                .SetTags("stable")
                .SetDescription("Rendering and graphics events")
);
namespace mlinsight{
    extern ssize_t tensorTrackUUid;
    extern bool enablePerfettoTrace;
    extern bool shouldReportThisRankToPerfetto; //A boolean indicating whether MLInsight should output any data to this rank 
    extern bool shouldReportAtThisStep; //A boolean indicating whether MLInsight should output any data to this rank 
    extern bool waitForProfilingClient;
    extern bool showNonPytorchObjects;
    extern int64_t detailedMemInfoLowerBound,detailedMemInfoUpperBound;
    extern int32_t waitSteps, activeSteps, pauseSteps;
    /*
    * Call this function to turn perfetto trace on and off based on shouldReportAtThisStep and shouldReportThisRankToPerfetto.
    */
    inline void onTraceSwitchUpdate(){
        enablePerfettoTrace = shouldReportThisRankToPerfetto && shouldReportAtThisStep;
    }
    void waitForTracingStart();
    extern pthread_cond_t cv;

    #define MLINSIGHT_TRACE_EVENT_BEGIN(...)  {if(enablePerfettoTrace) {TRACE_EVENT_BEGIN(__VA_ARGS__);}}
    #define MLINSIGHT_TRACE_EVENT_END(...)  {if(enablePerfettoTrace) {TRACE_EVENT_END(__VA_ARGS__);}}
    #define MLINSIGHT_TRACE_COUNTER(...)  {if(enablePerfettoTrace) {TRACE_COUNTER(__VA_ARGS__);}}

}




#endif
#endif //MLINSIGHT_PERFETTO_H
