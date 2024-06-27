#include <fstream>
#include "trace/tool/Perfetto.h"
#include "common/Logging.h"
#include "analyse/GlobalVariables.h"
#include <condition_variable>


// Reserves internal static storage for our tracing categories.
PERFETTO_TRACK_EVENT_STATIC_STORAGE();
PERFETTO_DEFINE_DATA_SOURCE_STATIC_MEMBERS(mlinsight::MemoryDataSource);

namespace mlinsight {
    bool isPerfettoEnabled=false;
    std::unique_ptr<perfetto::TracingSession> tracingSession;
    ssize_t tensorTrackUUid=-1;
    bool shouldReportThisRankToPerfetto = true, shouldReportAtThisStep = true, waitForProfilingClient=false, enablePerfettoTrace=true, showNonPytorchObjects=false;
    int64_t detailedMemInfoLowerBound = 10*1024*1024;
    int64_t detailedMemInfoUpperBound = 9999L*1024*1024*1024;
    int32_t waitSteps = 0, activeSteps = 1, pauseSteps = 0;
    pthread_cond_t cv;

    void initializePerfetto() {
        perfetto::TracingInitArgs args;
        // The backends determine where trace events are recorded. For this example we
        // are going to use the in-process tracing service, which only includes in-app
        // events.
        //args.backends |= perfetto::kInProcessBackend;
        args.backends |= perfetto::kSystemBackend;
        args.shmem_size_hint_kb = 6400;
        perfetto::Tracing::Initialize(args);
        perfetto::TrackEvent::Register();

        // Register our custom data source. Only the name is required, but other
        // properties can be advertised too.
        perfetto::DataSourceDescriptor dsd;
        dsd.set_name("com.bytedance.mlinsight");
        MemoryDataSource::Register(dsd);

        auto desc = perfetto::ProcessTrack::Current().Serialize();
        char processName[1024];
        if(stoi(localRank) >= 0){
            snprintf(processName,sizeof(processName)/sizeof(char),"python3 Rank %s",localRank);
        }else{
            snprintf(processName,sizeof(processName)/sizeof(char),"python3 Main");
        }
        
        desc.mutable_process()->set_process_name(processName);
        perfetto::TrackEvent::SetTrackDescriptor(
        perfetto::ProcessTrack::Current(), desc);


        // Give a custom name for the traced process.
        perfetto::Track forwardTrack = perfetto::Track(FORWARD_PROP_TRACK);
        perfetto::protos::gen::TrackDescriptor descForward = forwardTrack.Serialize();
        descForward.set_name("Forward");
        perfetto::TrackEvent::SetTrackDescriptor(forwardTrack, descForward);

        perfetto::Track backwardTrack = perfetto::Track(BACKWARD_PROP_TRACK);
        perfetto::protos::gen::TrackDescriptor descBackward = backwardTrack.Serialize();
        descBackward.set_name("Backward");
        perfetto::TrackEvent::SetTrackDescriptor(backwardTrack, descBackward);

        perfetto::Track flamegraphTrack = perfetto::Track(MEMORY_FLAMEGRAPH_TRACK);
        perfetto::protos::gen::TrackDescriptor descFlamegraph = flamegraphTrack.Serialize();
        descFlamegraph.set_name("MemLeak Summary");
        perfetto::TrackEvent::SetTrackDescriptor(flamegraphTrack, descFlamegraph);


        perfetto::Track pytorchAllocFreeTrack = perfetto::Track(PYTORCH_ALLOC_FREE_TRACK);
        perfetto::protos::gen::TrackDescriptor descPytorchAllocFreeTrack = pytorchAllocFreeTrack.Serialize();
        descPytorchAllocFreeTrack.set_name("Pytorch Alloc & Free");
        perfetto::TrackEvent::SetTrackDescriptor(pytorchAllocFreeTrack, descPytorchAllocFreeTrack);


        perfetto::Track nonPytorchAllocFreeTrack = perfetto::Track(NOPYTORCH_ALLOC_FREE_TRACK);
        perfetto::protos::gen::TrackDescriptor descNonPytorchAllocFreeTrack = nonPytorchAllocFreeTrack.Serialize();
        descNonPytorchAllocFreeTrack.set_name("Non-Pytorch Alloc & Free");
        perfetto::TrackEvent::SetTrackDescriptor(nonPytorchAllocFreeTrack, descNonPytorchAllocFreeTrack);
        
        perfetto::Track nonPytorchActiveTrack = perfetto::Track(NONPYTORCH_ACTIVE_TRACK);
        perfetto::protos::gen::TrackDescriptor descNonPytorchActiveTrack = nonPytorchActiveTrack.Serialize();
        descNonPytorchActiveTrack.set_name("Non-Pytorch Active");
        perfetto::TrackEvent::SetTrackDescriptor(nonPytorchActiveTrack, descNonPytorchActiveTrack);

        perfetto::Track stepTrack = perfetto::Track(STEP_TRACK);
        perfetto::protos::gen::TrackDescriptor descStepTrack = stepTrack.Serialize();
        descStepTrack.set_name("Step Recognition");
        perfetto::TrackEvent::SetTrackDescriptor(stepTrack, descStepTrack);

        perfetto::Track callstackIDTrack = perfetto::Track(CALLSTACK_ID_TRACK);
        perfetto::protos::gen::TrackDescriptor descCallstackIDTrack = callstackIDTrack.Serialize();
        descCallstackIDTrack.set_name("CallStack ID");
        perfetto::TrackEvent::SetTrackDescriptor(callstackIDTrack, descCallstackIDTrack);

        tensorTrackUUid=pytorchAllocFreeTrack.uuid;
    }

    // std::unique_ptr<perfetto::TracingSession> startTracing() {
    //     isPerfettoEnabled=true;
    //     INFO_LOG("Start Perfetto tracing");

    //     // The trace config defines which types of data sources are enabled for
    //     // recording. In this example we just need the "track_event" data source,
    //     // which corresponds to the TRACE_EVENT trace points.
    //     perfetto::TraceConfig cfg;
    //     cfg.add_buffers()->set_size_kb(10240);
    //     auto *trackDsCfg = cfg.add_data_sources()->mutable_config();
    //     trackDsCfg->set_name("track_event");
    //     perfetto::protos::gen::TrackEventConfig track_event_config;
    //     track_event_config.clear_enabled_categories();
    //     track_event_config.clear_disabled_categories();
    //     track_event_config.clear_enabled_tags();
    //     track_event_config.clear_disabled_tags();
    //     track_event_config.add_enabled_categories("*");
    //     track_event_config.add_enabled_tags("*");
    //     trackDsCfg->set_track_event_config_raw(track_event_config.SerializeAsString());

    //     auto *memoryDsCfg = cfg.add_data_sources()->mutable_config();
    //     memoryDsCfg->set_name("com.bytedance.memorysource");


    //     // Give a custom name for the traced process.
    //     perfetto::Track forwardTrack = perfetto::Track(FORWARD_PROP_TRACK);
    //     perfetto::protos::gen::TrackDescriptor descForward = forwardTrack.Serialize();
    //     descForward.set_name("Forward");
    //     perfetto::TrackEvent::SetTrackDescriptor(forwardTrack, descForward);

    //     perfetto::Track backwardTrack = perfetto::Track(BACKWARD_PROP_TRACK);
    //     perfetto::protos::gen::TrackDescriptor descBackward = backwardTrack.Serialize();
    //     descBackward.set_name("Backward");
    //     perfetto::TrackEvent::SetTrackDescriptor(backwardTrack, descBackward);

    //     perfetto::Track flamegraphTrack = perfetto::Track(MEMORY_FLAMEGRAPH_TRACK);
    //     perfetto::protos::gen::TrackDescriptor descFlamegraph = flamegraphTrack.Serialize();
    //     descFlamegraph.set_name("MemFlamegraph");
    //     perfetto::TrackEvent::SetTrackDescriptor(flamegraphTrack, descFlamegraph);


    //     perfetto::Track pytorchAllocFreeTrack = perfetto::Track(PYTORCH_ALLOC_FREE_TRACK);
    //     perfetto::protos::gen::TrackDescriptor descPytorchAllocFreeTrack = pytorchAllocFreeTrack.Serialize();
    //     descPytorchAllocFreeTrack.set_name("Pytorch Alloc & Free");
    //     perfetto::TrackEvent::SetTrackDescriptor(pytorchAllocFreeTrack, descPytorchAllocFreeTrack);


    //     perfetto::Track nonPytorchAllocFreeTrack = perfetto::Track(NOPYTORCH_ALLOC_FREE_TRACK);
    //     perfetto::protos::gen::TrackDescriptor descNonPytorchAllocFreeTrack = nonPytorchAllocFreeTrack.Serialize();
    //     descNonPytorchAllocFreeTrack.set_name("Non-Pytorch Alloc & Free");
    //     perfetto::TrackEvent::SetTrackDescriptor(nonPytorchAllocFreeTrack, descNonPytorchAllocFreeTrack);
        
    //     perfetto::Track nonPytorchActiveTrack = perfetto::Track(NONPYTORCH_ACTIVE_TRACK);
    //     perfetto::protos::gen::TrackDescriptor descNonPytorchActiveTrack = nonPytorchActiveTrack.Serialize();
    //     descNonPytorchActiveTrack.set_name("Non-Pytorch Active");
    //     perfetto::TrackEvent::SetTrackDescriptor(nonPytorchActiveTrack, descNonPytorchActiveTrack);

    //     tensorTrackUUid=pytorchAllocFreeTrack.uuid;

    //     auto tracing_session = perfetto::Tracing::NewTrace();
    //     tracing_session->Setup(cfg);
    //     tracing_session->StartBlocking();
    //     return tracing_session;
    // }

    // void stopTracing(std::unique_ptr<perfetto::TracingSession> tracing_session) {
    //     assert(isPerfettoEnabled);
    //     isPerfettoEnabled=false;
    //     INFO_LOG("Stop Perfetto tracing");

    //     MemoryDataSource::Trace(
    //             [](MemoryDataSource::TraceContext ctx) { ctx.Flush(); });

    //     // Make sure the last event is closed for this example.
    //     perfetto::TrackEvent::Flush();

    //     // Stop tracing and read the trace data.
    //     tracing_session->StopBlocking();
    //     std::vector<char> trace_data(tracing_session->ReadTraceBlocking());

    //     // Write the result into a file.
    //     // Note: To save memory with longer traces, you can tell Perfetto to write
    //     // directly into a file by passing a file descriptor into Setup() above.
    //     std::ostream output;
    //     std::string saveFilePath = logProcessRootPath + "/perfetto.pftrace";
    //     output.open(saveFilePath, std::ios::out | std::ios::binary);
    //     output.write(&trace_data[0], std::streamsize(trace_data.size()));
    //     output.close();
    //     INFO_LOGS("Perfetto trace written in %s. To read this trace in "
    //               "text form, run `./tools/traceconv text example.pftrace`", saveFilePath.c_str());
    // }

void MemoryDataSource::OnSetup(const perfetto::DataSourceBase::SetupArgs &setupArgs) {
    pthread_mutex_lock(&analyzerLock);
    auto& mlinsightConfig = setupArgs.config->mlinsight_config();
    AROUTPUT("perfetto client has attached to this rank\n");
    const std::vector<int32_t>& rankToProfileVector=mlinsightConfig.ranktoprofile();
    
    shouldReportThisRankToPerfetto = true;
    
    /**
     * Check if rank matches
    */
    int localRankInt=std::stoi(localRank);
    if(mlinsightConfig.ranktoprofile().size()==1 && rankToProfileVector[0]==-2){
        //All ranks should be enabled bypass rank id checking
    } else {
        shouldReportThisRankToPerfetto = false;
        for(ssize_t i=0;i < rankToProfileVector.size();++i){
            if(localRankInt==rankToProfileVector[i]){
                shouldReportThisRankToPerfetto = true;
                AROUTPUT("Based on config.ranktoprofile, MLInsight will profile this rank\n");
                break;
            }
        }
        if(!shouldReportThisRankToPerfetto){
            AROUTPUT("Based on config.ranktoprofile, MLInsight will not profile this rank\n");

            onTraceSwitchUpdate();
            pthread_mutex_unlock(&analyzerLock);
            return;
        }
    }

    if(mlinsightConfig.has_detailedmeminfolowerbound()){
        detailedMemInfoLowerBound = mlinsightConfig.detailedmeminfolowerbound();
        AROUTPUTS("config.detailedMemInfoLowerBound = %s\n",format_size(detailedMemInfoLowerBound).c_str());
    }
    
    if(mlinsightConfig.has_detailedmeminfoupperbound()){
        detailedMemInfoUpperBound = mlinsightConfig.detailedmeminfoupperbound();
        AROUTPUTS("config.detailedMemInfoUpperBound = %s\n",format_size(detailedMemInfoUpperBound).c_str());
    }
    
    if(mlinsightConfig.has_schedule()){
        auto& schedule=mlinsightConfig.schedule();
        if(mlinsightConfig.schedule().has_active()){
            activeSteps = schedule.active();
            if(activeSteps<0){
                fatalError("config.schedule.active parameter range incorrect");
            }
            AROUTPUTS("config.schedule.active = %d steps\n",activeSteps);
        }
        if(mlinsightConfig.schedule().has_wait()){
            waitSteps = schedule.wait();
            if(waitSteps<0){
                fatalError("config.schedule.wait parameter range incorrect");
            }
            AROUTPUTS("config.schedule.wait = %d steps\n",waitSteps);
        }
        if(mlinsightConfig.schedule().has_pause()){
            pauseSteps = schedule.pause();
            if(pauseSteps<0){
                fatalError("config.schedule.pause parameter range incorrect");
            }
            AROUTPUTS("config.schedule.pause = %d steps\n",pauseSteps);
        }
    }

    if(mlinsightConfig.has_waitforprofilingclient()){
        waitForProfilingClient=mlinsightConfig.waitforprofilingclient();
        AROUTPUTS("config.waitForProfilingClient = %s\n",waitForProfilingClient?"true":"false");
    }
    
    if(mlinsightConfig.has_shownonpytorchobjects()){
        showNonPytorchObjects=mlinsightConfig.shownonpytorchobjects();
        AROUTPUTS("config.showNonPytorchObjects = %s\n",showNonPytorchObjects?"true":"false");
    }

    onTraceSwitchUpdate();
    pthread_mutex_unlock(&analyzerLock);
}

void MemoryDataSource::OnStart(const perfetto::DataSourceBase::StartArgs &) {
    pthread_mutex_lock(&analyzerLock);
    pthread_cond_broadcast(&cv);
    // Emit existing callstack ID events events
    // for(auto it=cCallStackRegistery->begin();it!=cCallStackRegistery->end();++it){
    //     MLINSIGHT_("tensor", perfetto::DynamicString{std::to_string(it->first->callstackID)}, perfetto::Track(CALLSTACK_ID_TRACK),perfetto::Flow::FromPointer(it->first),"MLInsight.callStackID",it->first->callstackID,"MLInsight.hybridCallstack",it->first->toString(),"MLInsight.source","Pytorch Allocator");
    //     MLINSIGHT_TRACE_EVENT_END("tensor", perfetto::Track(CALLSTACK_ID_TRACK));
    // }
    std::vector<PyCallStack*> sortedCallStackPtrArray(pyCallStackRegistery->size());
    for(ssize_t i=0;i<pyCallStackRegistery->size();++i){
        sortedCallStackPtrArray.emplace_back(nullptr);
    }
    for(auto it=pyCallStackRegistery->begin();it!=pyCallStackRegistery->end();++it){
        sortedCallStackPtrArray[it->first->callstackID]=it->first;
    }
    std::stringstream ss;
                
    for(ssize_t i=0;i<sortedCallStackPtrArray.size();++i){
        PyCallStack* pyStackPtr=sortedCallStackPtrArray[i];
        if(pyStackPtr){
            ss.str("");
            ss<<logProcessRootPath + "/callstack_" << pyStackPtr->callstackID<< ".txt";
            MLINSIGHT_TRACE_EVENT_BEGIN("tensor", perfetto::DynamicString{std::to_string(pyStackPtr->callstackID)}, perfetto::Track(CALLSTACK_ID_TRACK),perfetto::Flow::FromPointer(pyStackPtr),"MLInsight.callStackID",pyStackPtr->callstackID,"MLInsight.pyCallstack",pyStackPtr->toString(),"MLInsight.logfile",ss.str(),"MLInsight.source","Pytorch Allocator");
            MLINSIGHT_TRACE_EVENT_END("tensor", perfetto::Track(CALLSTACK_ID_TRACK));
        }
    }
   

    pthread_mutex_unlock(&analyzerLock);
}

void MemoryDataSource::OnStop(const perfetto::DataSourceBase::StopArgs &) {
    pthread_mutex_lock(&analyzerLock);
    shouldReportThisRankToPerfetto = false;
    onTraceSwitchUpdate();
    pthread_mutex_unlock(&analyzerLock);
}

void  waitForTracingStart() {
    pthread_mutex_lock(&analyzerLock);
    int waitResult=pthread_cond_wait(&cv, &analyzerLock);
    assert(waitResult==0);
    pthread_mutex_unlock(&analyzerLock);
}



}
