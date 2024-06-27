#include "analyse/PerfettoTensorTraceAnalyzer.h"
#include "analyse/GlobalVariables.h"
#include <condition_variable>

namespace mlinsight
{

    ssize_t stepCounter=0;//Training step counter
    ssize_t waitCounter=0;//Used to support "Wait" and "Active"

    // Explicitly initialize
    template class PerfettoTensorTraceAnalyser<DriverTensorType, FramekworkTensorType>;

    class Observer : public perfetto::TrackEventSessionObserver {
    public:
        Observer() { 
            perfetto::TrackEvent::AddSessionObserver(this); 
        }
        ~Observer() override { 
            perfetto::TrackEvent::RemoveSessionObserver(this); 
        }
        void OnSetup(const perfetto::DataSourceBase::SetupArgs& setupArgs) override {
           
        }

        void OnStart(const perfetto::DataSourceBase::StartArgs& startArgs) override {
            
        }

        void OnStop(const perfetto::DataSourceBase::StopArgs& stopArgs) override {
            pthread_mutex_lock(&analyzerLock);

            pthread_mutex_unlock(&analyzerLock);
        }
        
       
    };

    Observer observer;
    
    template <typename DRIVER_CTENSOR_TYPE, typename FRAMEWORK_CTENSOR_TYPE>
    void PerfettoTensorTraceAnalyser<DRIVER_CTENSOR_TYPE, FRAMEWORK_CTENSOR_TYPE>::onPostAllocDriver(ssize_t size, void *ptr, DRIVER_CTENSOR_TYPE *newTensor)
    {
        if (newTensor)
        {
            assert(ptr);
            auto *perfettoTensorPtr = static_cast<Perfetto::TensorMixin *>(newTensor);
            perfettoTensorPtr->allocTimestamp = perfetto::TrackEvent::GetTraceTimeNs();
           TRACE_COUNTER("MemStats", "GPU Free (GB)", this->memLeakAnalyzer.driverMetric.freeMem / (1024.0 * 1024 * 1024));

            if (!globalExecutionState.isInvokingFrameworkMemOp){
                this->perfettoFlowSequenceId += 1;
                newTensor->perfettoSequenceId=this->perfettoFlowSequenceId;
                
                if(showNonPytorchObjects){
                    if(newTensor->callstack->isNewCallStackId){
                        this->perfettoFlowSequenceId += 1;
                        MLINSIGHT_TRACE_EVENT_BEGIN("tensor", perfetto::DynamicString{std::to_string(newTensor->callstack->callstackID)}, perfetto::Track(CALLSTACK_ID_TRACK),perfetto::Flow::FromPointer(newTensor->callstack),"MLInsight.callStackID", newTensor->callstack->callstackID,"MLInsight.hybridCallstack",newTensor->callstack->toString(),"MLInsight.sourch","CUDA");
                        MLINSIGHT_TRACE_EVENT_END("tensor", perfetto::Track(CALLSTACK_ID_TRACK));

                        std::stringstream ss;
                        ss<<logProcessRootPath + "/callstack_" << newTensor->callstack->callstackID<< ".txt";
                        std::ofstream output(ss.str(), std::ios::app);
                        newTensor->callstack->print(output);
                        output.close();
                    }
                    
                    MLINSIGHT_TRACE_EVENT_BEGIN("tensor", "Alloc", perfetto::Track(NOPYTORCH_ALLOC_FREE_TRACK),perfetto::Flow::ProcessScoped(newTensor->perfettoSequenceId),perfetto::Flow::FromPointer(newTensor->callstack),"MLInsight.callStackID",newTensor->callstack->callstackID,"MLInsight.allocationSize",size / (1024.0 * 1024 * 1024));
                    MLINSIGHT_TRACE_EVENT_END("tensor", perfetto::Track(NOPYTORCH_ALLOC_FREE_TRACK));
                }


                assert(this->memLeakAnalyzer.frameworkInfo.nonFrameworkAllocatorMem>=0);
                
                MLINSIGHT_TRACE_COUNTER("MemStats", "NonPytorch Active (GB)", this->memLeakAnalyzer.frameworkInfo.nonFrameworkAllocatorMem / (1024.0 * 1024 * 1024));
            }
            
        }
    }

    template <typename DRIVER_CTENSOR_TYPE, typename FRAMEWORK_CTENSOR_TYPE>
    void PerfettoTensorTraceAnalyser<DRIVER_CTENSOR_TYPE, FRAMEWORK_CTENSOR_TYPE>::onPostAllocFramework(ssize_t size, void *ptr, FRAMEWORK_CTENSOR_TYPE *newTensor)
    {
        if (newTensor)
        {
            this->perfettoFlowSequenceId += 1;
            assert(ptr);
            auto *perfettoTensorPtr = static_cast<Perfetto::TensorMixin *>(newTensor);
            perfettoTensorPtr->allocTimestamp = perfetto::TrackEvent::GetTraceTimeNs();
            perfettoTensorPtr->perfettoSequenceId = this->perfettoFlowSequenceId;

            if(newTensor->callstack->isNewCallStackId){
                this->perfettoFlowSequenceId += 1;

                std::stringstream ss;
                ss<<logProcessRootPath + "/callstack_" << newTensor->callstack->callstackID<< ".txt";
                std::ofstream output(ss.str(), std::ios::app);
                newTensor->callstack->print(output);
                output.close();

                MLINSIGHT_TRACE_EVENT_BEGIN("tensor", perfetto::DynamicString{std::to_string(newTensor->callstack->callstackID)}, perfetto::Track(CALLSTACK_ID_TRACK),perfetto::Flow::FromPointer(newTensor->callstack),"MLInsight.callStackID", newTensor->callstack->callstackID,"MLInsight.pyCallstack",newTensor->callstack->toString(),"MLInsight.source","Pytorch Allocator","MLInsight.logfile",ss.str());
                MLINSIGHT_TRACE_EVENT_END("tensor", perfetto::Track(CALLSTACK_ID_TRACK));
            }

            TRACE_COUNTER("MemStats", "GPU Free (GB)", this->memLeakAnalyzer.driverMetric.freeMem / (1024.0 * 1024 * 1024)); //Add here to update GPU value often. The purpose is to prevent GPU free mem metric lost during a partial recording
            
            if(detailedMemInfoLowerBound <= size && size <= detailedMemInfoUpperBound){
                MLINSIGHT_TRACE_EVENT_BEGIN("tensor", "Alloc", perfetto::Track(PYTORCH_ALLOC_FREE_TRACK), perfetto::Flow::ProcessScoped(perfettoTensorPtr->perfettoSequenceId), perfetto::Flow::FromPointer(newTensor->callstack), "MLInsight.callStackID",newTensor->callstack->callstackID, "MLInsight.allocationSize (GB)",size / (1024.0*1024*1024));
                MLINSIGHT_TRACE_EVENT_END("tensor", perfetto::Track(PYTORCH_ALLOC_FREE_TRACK));
            }else{
                perfettoTensorPtr->skippedRecording=true;
            }

            MLINSIGHT_TRACE_COUNTER("MemStats", "Pytorch Active (GB)", this->memLeakAnalyzer.allocatorStatus.curActive / (1024.0 * 1024 * 1024));
            ssize_t unusedReserved = this->memLeakAnalyzer.allocatorStatus.curReserve - this->memLeakAnalyzer.allocatorStatus.curActive;

            MLINSIGHT_TRACE_COUNTER("MemStats", "Pytorch Unused Reserved (GB)", (unusedReserved) / (1024.0 * 1024 * 1024));
            MLINSIGHT_TRACE_COUNTER("MemStats", "Pytorch External Fragmentation Rate (%)", (unusedReserved/(double)this->memLeakAnalyzer.allocatorStatus.curReserve)*100.0);

            //MLINSIGHT_TRACE_COUNTER("MemStats", "Pytorch Non-Releasable (GB)", this->memLeakAnalyzer.allocatorStatus.nonReleasable / (1024.0 * 1024 * 1024));
            //MLINSIGHT_TRACE_COUNTER("MemStats", "Pytorch Internal Fragmentation (GB)", this->memLeakAnalyzer.internalFragMetric.internalFragmentation / (1024.0 * 1024  * 1024));


        }
    }

    template <typename DRIVER_CTENSOR_TYPE, typename FRAMEWORK_CTENSOR_TYPE>
    void PerfettoTensorTraceAnalyser<DRIVER_CTENSOR_TYPE, FRAMEWORK_CTENSOR_TYPE>::onPreFreeFramework(void *ptr, FRAMEWORK_CTENSOR_TYPE *justFreedTensor)
    {
        // Inert data into MemoryDatasource
        if (justFreedTensor != nullptr)
        {
            auto *perfettoTensorPtr = static_cast<Perfetto::TensorMixin *>(justFreedTensor);
            perfettoTensorPtr->freeTimestamp = perfetto::TrackEvent::GetTraceTimeNs();

            MLINSIGHT_TRACE_COUNTER("MemStats", "Pytorch Active (GB)", this->memLeakAnalyzer.allocatorStatus.curActive / (1024.0 * 1024 * 1024));
            MLINSIGHT_TRACE_COUNTER("MemStats", "Pytorch Unused Reserved (GB)", (this->memLeakAnalyzer.allocatorStatus.curReserve - this->memLeakAnalyzer.allocatorStatus.curActive) / (1024.0 * 1024 * 1024));
            //MLINSIGHT_TRACE_COUNTER("MemStats", "Pytorch Non-Releasable (GB)", this->memLeakAnalyzer.allocatorStatus.nonReleasable / (1024.0 * 1024 * 1024));
            //MLINSIGHT_TRACE_COUNTER("MemStats", "Pytorch Internal Fragmentation (GB)", this->memLeakAnalyzer.internalFragMetric.internalFragmentation / (1024.0 * 1024 * 1024));

            if(!perfettoTensorPtr->skippedRecording){
                MLINSIGHT_TRACE_EVENT_BEGIN("tensor", "Free", perfetto::Track(PYTORCH_ALLOC_FREE_TRACK), perfetto::Flow::ProcessScoped(justFreedTensor->perfettoSequenceId));
                MLINSIGHT_TRACE_EVENT_END("tensor", perfetto::Track(PYTORCH_ALLOC_FREE_TRACK));
            }

            // TRACE_EVENT("Tensor","Free",perfetto::Track(TENSOR_TRACK));

            //    MemoryDataSource::Trace([](MemoryDataSource::TraceContext ctx) {
            //                    auto packet = ctx.NewTracePacket();
            //                    packet->set_timestamp(42);
            //                    packet->set_for_testing()->set_str("Hello world!");
            //                });

            //                static perfetto::protos::pbzero::InternedData* interned_data = nullptr;

            //                MemoryDataSource::Trace([](MemoryDataSource::TraceContext ctx) {
            //                    auto packet = ctx.NewTracePacket();
            //                    packet->set_timestamp(43);
            //                    auto* event = packet->set_track_event();
            //                    event->add_categories("cat");
            //                    event->set_name("ev1");
            //                    event->set_type(perfetto::protos::pbzero::TrackEvent::TYPE_INSTANT);
            //                });
            //    MemoryDataSource::Trace([perfettoTensorPtr](MemoryDataSource::TraceContext ctx) {
            //        auto packet = ctx.NewTracePacket();
            //        packet->set_timestamp(perfettoTensorPtr->freeTimestamp);
            //        auto* trackEvent = packet->set_track_event();
            //        trackEvent->set_name("Alloc");
            //        trackEvent->add_categories("Tensor");
            //        trackEvent->set_type(perfetto::protos::pbzero::TrackEvent::TYPE_SLICE_BEGIN);
            //    });
            //    MemoryDataSource::Trace([perfettoTensorPtr](MemoryDataSource::TraceContext ctx) {
            //        auto packet = ctx.NewTracePacket();
            //        packet->set_timestamp(perfettoTensorPtr->freeTimestamp+100);
            //        auto* trackEvent = packet->set_track_event();
            //        trackEvent->set_name("Free");
            //        trackEvent->set_type(perfetto::protos::pbzero::TrackEvent::TYPE_SLICE_END);
            //        trackEvent->add_categories("Tensor");
            //    });
        }
    }

    template <typename DRIVER_CTENSOR_TYPE, typename FRAMEWORK_CTENSOR_TYPE>
    void PerfettoTensorTraceAnalyser<DRIVER_CTENSOR_TYPE, FRAMEWORK_CTENSOR_TYPE>::onPreFreeDriver(void *ptr, DRIVER_CTENSOR_TYPE *justFreedTensor)
    {
        if (justFreedTensor)
        {
            assert(ptr);
            auto *perfettoTensorPtr = static_cast<Perfetto::TensorMixin *>(justFreedTensor);
            perfettoTensorPtr->freeTimestamp = perfetto::TrackEvent::GetTraceTimeNs();

            TRACE_COUNTER("MemStats", "GPU Free (GB)", this->memLeakAnalyzer.driverMetric.freeMem / (1024.0 * 1024 * 1024));

            if (!justFreedTensor->isAllocatedByFramework){
                //MLINSIGHT_TRACE_EVENT_BEGIN("tensor", "Free", perfetto::Track(NOPYTORCH_ALLOC_FREE_TRACK),perfetto::Flow::ProcessScoped(justFreedTensor->perfettoSequenceId));
                //MLINSIGHT_TRACE_EVENT_END("tensor", perfetto::Track(NOPYTORCH_ALLOC_FREE_TRACK));
                // This allocation is not from the Pytorch memory allocator
                assert(this->memLeakAnalyzer.frameworkInfo.nonFrameworkAllocatorMem>=0);
                MLINSIGHT_TRACE_COUNTER("MemStats", "NonPytorch Active (GB)", this->memLeakAnalyzer.frameworkInfo.nonFrameworkAllocatorMem / (1024.0 * 1024 * 1024));
            }
        }
    }

    template <typename DRIVER_CTENSOR_TYPE, typename FRAMEWORK_CTENSOR_TYPE>
    void PerfettoTensorTraceAnalyser<DRIVER_CTENSOR_TYPE, FRAMEWORK_CTENSOR_TYPE>::onPostFreeDriver(void *ptr, DRIVER_CTENSOR_TYPE *justFreedTensor)
    {
        
    }

    template <typename DRIVER_CTENSOR_TYPE, typename FRAMEWORK_CTENSOR_TYPE>
    void PerfettoTensorTraceAnalyser<DRIVER_CTENSOR_TYPE, FRAMEWORK_CTENSOR_TYPE>::onPostFreeFramework(void *ptr, FRAMEWORK_CTENSOR_TYPE *justFreedTensor)
    {
        
    }

    template <typename DRIVER_CTENSOR_TYPE, typename FRAMEWORK_CTENSOR_TYPE>
    void PerfettoTensorTraceAnalyser<DRIVER_CTENSOR_TYPE, FRAMEWORK_CTENSOR_TYPE>::onStepFinished(){
        MLINSIGHT_TRACE_EVENT_BEGIN("flamegraph","Pytorch Snapshot",perfetto::Track(MEMORY_FLAMEGRAPH_TRACK));
        std::stringstream ss;
        // this->memLeakAnalyzer.printSummary(ss,0);
        // std::string memLeakSummary=ss.str();
        
        // ss.str("");
        this->memLeakAnalyzer.printFramework(ss,0);
        std::string memLeakTorchSummary=ss.str();

       
        
        
        MLINSIGHT_TRACE_EVENT_END("flamegraph",perfetto::Track(MEMORY_FLAMEGRAPH_TRACK),"MLInsight.memLeakTorchDetail",memLeakTorchSummary);

        if(showNonPytorchObjects){
            MLINSIGHT_TRACE_EVENT_BEGIN("flamegraph","Non-Pytorch Snapshot",perfetto::Track(MEMORY_FLAMEGRAPH_TRACK));
            ss.str("");
            this->memLeakAnalyzer.printDriver(ss,0);
            std::string memLeakNonTorchSummary=ss.str();
            MLINSIGHT_TRACE_EVENT_END("flamegraph",perfetto::Track(MEMORY_FLAMEGRAPH_TRACK),
                                                                            "MLInsight.memLeakNonTorchSummary",memLeakNonTorchSummary
                                                                            );
        }


        if(stepCounter < waitSteps){
            shouldReportAtThisStep=false;
        } else {
            // AROUTPUTS("waitCounter is %zd\n",waitCounter);

            if(waitCounter < activeSteps){
                shouldReportAtThisStep=true;
                // AROUTPUT("waitCounter < activeSteps\n");
                ++waitCounter;
            }else if(waitCounter< activeSteps+pauseSteps){
                // AROUTPUT("waitCounter < activeSteps+pauseSteps\n");
                shouldReportAtThisStep=false;
                ++waitCounter;
            }else{
                // AROUTPUT("waitCounter = 0\n");
                shouldReportAtThisStep=true;
                waitCounter=1;
            }
            onTraceSwitchUpdate();
            if(waitForProfilingClient && !perfetto::TrackEvent::IsEnabled()){
                if(isRankParentProcess){
                    fprintf(stderr,"MLInsight has paused the execution in rank %s and is waiting for perfetto client because mlinsight_config.waitForProfilingClient is set to true in the last run\n",localRank);
                }
                waitForTracingStart();
            }

        }

        if(stepCounter>0){
            MLINSIGHT_TRACE_EVENT_END("forward",perfetto::Track(STEP_TRACK));
        }

        char stepStr[255];
        sprintf(stepStr,"Step %zd",stepCounter);
        MLINSIGHT_TRACE_EVENT_BEGIN("forward",perfetto::DynamicString(stepStr),perfetto::Track(STEP_TRACK));
        ++stepCounter;
    }

    template <typename DRIVER_CTENSOR_TYPE, typename FRAMEWORK_CTENSOR_TYPE>
    void PerfettoTensorTraceAnalyser<DRIVER_CTENSOR_TYPE, FRAMEWORK_CTENSOR_TYPE>::onOutOfMemory(ssize_t size){
         MLINSIGHT_TRACE_EVENT_BEGIN("flamegraph","Snapshot at OOM",perfetto::Track(MEMORY_FLAMEGRAPH_TRACK));
        std::stringstream ss;
        this->memLeakAnalyzer.printFramework(ss,0);
        std::string memLeakTorchSummary=ss.str();

        ss.str("");
        this->memLeakAnalyzer.printDriver(ss,0);
        std::string memLeakNonTorchSummary=ss.str();
        
        MLINSIGHT_TRACE_EVENT_END("flamegraph",perfetto::Track(MEMORY_FLAMEGRAPH_TRACK),"MLInsight.memLeakTorchDetail",memLeakTorchSummary
                                                                            ,"MLInsight.memLeakNonTorchSummary",memLeakNonTorchSummary
                                                                            );
    }
}