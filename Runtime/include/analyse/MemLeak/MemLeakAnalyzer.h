#ifndef MLINSIGHT_MEMLEAKANALYZER_H
#define MLINSIGHT_MEMLEAKANALYZER_H

#include "analyse/MemLeak/MemLeakMetrics.h"
#include "analyse/MemLeak/MemoryLeakMetrics_Pytorch.h"
//#include "analyse/TensorMap.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include "analyse/TensorMap.h"
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
    class MemLeakAnalyzer: public CompleteCallback<DRIVER_CTENSOR_TYPE,FRAMEWORK_CTENSOR_TYPE> {
    public:

        MemLeak::GeneralMetric<DRIVER_CTENSOR_TYPE> driverInfo;
        MemLeak::FrameworkGeneralMetric<DRIVER_CTENSOR_TYPE,FRAMEWORK_CTENSOR_TYPE> frameworkInfo;
        MemLeak::Driver::Metric<DRIVER_CTENSOR_TYPE,MemLeak::Driver::Type::CUDA> driverMetric;
        /*
         * Combined metric that require events from both the driver and tensor
         */
        MemLeak::InternalFrag::TorchSimpleSimuMetric<DRIVER_CTENSOR_TYPE, FRAMEWORK_CTENSOR_TYPE> internalFragMetric;
        //MemLeak::InternalFrag::TorchSimuMetric<FRAMEWORK_CTENSOR_TYPE> torchSimuInternalFragMetric; //A beta implementation that can report more accurate internal fragmentation.

        MemLeak::ExternalFrag::Metric<DRIVER_CTENSOR_TYPE, FRAMEWORK_CTENSOR_TYPE> externalFragMetric; //Note that externalFragMetric depends on frameworkInfo. So frameworkInfo callbacks should always be called before externalFragMetric callbacks.

        MemLeak::AllocStats::Metric<DRIVER_CTENSOR_TYPE, FRAMEWORK_CTENSOR_TYPE> allocatorStatus;
//        MemLeak::AllocatorStatus::Metric<DRIVER_CTENSOR_TYPE, FRAMEWORK_CTENSOR_TYPE> allocatorStatus;
    public:
        MemLeakAnalyzer():externalFragMetric(frameworkInfo){

        }
        /**
       * [Interface]
       */
        void onPreAllocFramework(ssize_t size){
            frameworkInfo.onPreAllocFramework(size);
            internalFragMetric.onPreAllocFramework(size);
            externalFragMetric.onPreAllocFramework(size);
            allocatorStatus.onPreAllocFramework(size);
        }
        /**
        * [Interface]
        */
        void onPreAllocDriver(ssize_t size) {
            driverInfo.onPreAlloc(size);
            frameworkInfo.onPreAllocDriver(size);
            driverMetric.onPreAlloc(size);
            internalFragMetric.onPreAllocDriver(size);
            externalFragMetric.onPreAllocDriver(size);
            allocatorStatus.onPreAllocDriver(size);
        }

        /**
        * [Interface]
        */
        void onPostAllocDriver(ssize_t size, void *ptr, DRIVER_CTENSOR_TYPE *newTensor) {
            driverInfo.onPostAlloc(size, ptr, newTensor);
            frameworkInfo.onPostAllocDriver(size,ptr,newTensor);
            driverMetric.onPostAlloc(size,ptr,newTensor);
            internalFragMetric.onPostAllocDriver(size, ptr, newTensor);
            externalFragMetric.onPostAllocDriver(size, ptr, newTensor);
            allocatorStatus.onPostAllocDriver(size,ptr,newTensor);
        }

        /**
         * [Interface]
         */
        void onPostAllocFramework(ssize_t size, void *ptr, FRAMEWORK_CTENSOR_TYPE *newTensor) {
            frameworkInfo.onPostAllocFramework(size, ptr, newTensor);
            internalFragMetric.onPostAllocFramework(size, ptr, newTensor);
            //frameworkInfo must be calculated before externalFragMetric
            externalFragMetric.onPostAllocFramework(size, ptr, newTensor);
            allocatorStatus.onPostAllocFramework(size,ptr,newTensor);
        }

        /**
        * [Interface]
        * Invoked before the allocator frees memory. Remove a new Tensor from mapAliveObjs.
        * For each allocation, the sequence is [onPostAlloc(...... AllocationType::Framework] -> [onPostAlloc(...... AllocationType::Driver]
        * @param ptr Memory pointer
        * @param type Indicate whether this is a driver allocation or framework allocation.
        */
        void onPreFreeFramework(void *ptr, FRAMEWORK_CTENSOR_TYPE *justFreedTensor) {
            frameworkInfo.onPreFreeFramework(ptr, justFreedTensor);
            internalFragMetric.onPreFreeFramework(ptr, justFreedTensor);
            //frameworkInfo must be calculated before externalFragMetric
            externalFragMetric.onPreFreeFramework(ptr, justFreedTensor);
            allocatorStatus.onPreFreeFramework(ptr, justFreedTensor);
        }


        /**
        * [Interface]
        * Invoked before the allocator frees memory. Remove a new Tensor from mapAliveObjs.
        * For each allocation, the sequence is [onPostAlloc(...... AllocationType::Framework] -> [onPostAlloc(...... AllocationType::Driver]
        * @param ptr Memory pointer
        * @param type Indicate whether this is a driver allocation or framework allocation.
        */
        void onPreFreeDriver(void *ptr, DRIVER_CTENSOR_TYPE *justFreedTensor) {
            driverInfo.onPreFree(ptr, justFreedTensor);
            frameworkInfo.onPreFreeDriver(ptr, justFreedTensor);
            driverMetric.onPreFree(ptr,justFreedTensor);
            internalFragMetric.onPreFreeDriver(ptr, justFreedTensor);
            //frameworkInfo must be calculated before externalFragMetric
            externalFragMetric.onPreFreeDriver(ptr, justFreedTensor);
            allocatorStatus.onPreFreeDriver(ptr, justFreedTensor);

        }

        /**
        * [Interface]
        */
        void onPostFreeDriver(void *ptr,DRIVER_CTENSOR_TYPE* justFreedTensor){
            driverInfo.onPostFree(ptr,justFreedTensor);
            frameworkInfo.onPostFreeDriver(ptr, justFreedTensor);
            driverMetric.onPostFree(ptr,justFreedTensor);
            internalFragMetric.onPostFreeDriver(ptr,justFreedTensor);
            //frameworkInfo must be calculated before externalFragMetric
            externalFragMetric.onPostFreeDriver(ptr,justFreedTensor);
            allocatorStatus.onPostFreeDriver(ptr, justFreedTensor);
        }

        void onPostFreeFramework(void *ptr,FRAMEWORK_CTENSOR_TYPE* justFreedTensor) {
            frameworkInfo.onPostFreeFramework(ptr,justFreedTensor);
            internalFragMetric.onPostFreeFramework(ptr,justFreedTensor);
            //frameworkInfo must be calculated before externalFragMetric
            externalFragMetric.onPostFreeFramework(ptr,justFreedTensor);
            allocatorStatus.onPostFreeFramework(ptr, justFreedTensor);
        }


        void printOutput(std::ostream &output, ssize_t oomAllocSize);

        void printSummary(std::ostream &output, ssize_t oomAllocSize);

        void printFramework(std::ostream &output, ssize_t oomAllocSize);

        void printDriver(std::ostream &output, ssize_t oomAllocSize);


        void printDriverInfo(std::ostream &output) const {
            output <<"Callstack details saving folder: "<< logProcessRootPath << std::endl;
            //assert(driverMetric.totalMem != 0);
            // Printing the total information
            output << "General GPU information: total " << format_size(driverMetric.totalMem) << ". Current usage - "
                   << format_size(driverMetric.totalMem - driverMetric.freeMem) << ". Peak usage - " << format_size(
                    driverMetric.peakGPUMem)
                   << std::endl;

            // Printing the driver information
            output << "Driver GPU information: current usage - " << format_size(driverInfo.curUsage)
                   << ". Peak usage - " << format_size(driverInfo.peakUsage) << std::endl;

            output << "\t Within current memory, normal driver - " << format_size(driverInfo.curUsage - frameworkInfo.frameworkGeneral.curUsage)
                   << ". Pytorch - " << format_size(frameworkInfo.frameworkGeneral.curUsage) << std::endl;

            output << "\t Number of allocations: " << driverInfo.numAllocs << std::endl;
            output << "\t Total allocated memory: " << format_size(driverInfo.memAllocs) << std::endl;
            output << "\t Number of frees:" << driverInfo.numFrees << std::endl;
            output << "\t Total freed memory:" << format_size(driverInfo.memFrees) << std::endl;
            //output << endl;
            output << "\t Number of alive objects: " << driverInfo.numAliveObjs << std::endl;
            output << "\t Memory of alive objects: " << format_size(driverInfo.curUsage) << std::endl;
        }

        void printFrameworkInfo(std::ostream &output);

        void printBasicInfo(std::ostream &output, ssize_t oomAllocSize){
            DBG_LOGS("oomAllocSize %lx!!!!\n", oomAllocSize);

            // If size is given, there is an OOM failure when trying to allocate the given size
            if(oomAllocSize != 0) {
                output << std::endl;
                output << "OOM error when allocating " << format_size(oomAllocSize) << " at the following callsite: " << std::endl;
                print_stacktrace(output); //todo: Change to python stack trace

                if (oomAllocSize >=  this->driverMetric.totalMem) {
                    output << std::endl << "GPU capacity is " << format_size(this->driverMetric.totalMem) << ", which is less than the requested size - " << format_size(oomAllocSize) << std::endl;
                    output << "That is, this OOM is due to GPU capacity. Please use a larger GPU or adjust parameters like token number or batch size!!" << std::endl;
                }
                else if(oomAllocSize >= this->driverMetric.freeMem) {
                    output << "Allocation size - " << format_size(oomAllocSize) << " is larger than available GPU memory - " << format_size(this->driverMetric.freeMem) << std::endl;

                    if(oomAllocSize < (this->frameworkInfo.frameworkGeneral.curUsage + this->driverMetric.freeMem)) {
                        output << "The problem is caused by external fragmentation of PyTorch allocator, which may "
                               << "be able to be fixed by invoking torch.cuda.empty_cache()!" << std::endl;
                    }

                    if(oomAllocSize < this->internalFragMetric.internalFragmentation) {
                        output << "One major issue is caused by the internal fragmentation introduced by PyTorch allocator." << std::endl;
                    }
                }
            }
        }

        template<typename CTENSOR_TYPE, typename STACK_TYPE>
        void printLeakyObjects(std::ostream &output, TensorMaps<CTENSOR_TYPE>& aliveObjs) {
            //todo: If leaky object print incurs large overhead, we may generate callstack id at runtime.
            struct InfoByCallStack {
            public:
                size_t aliveCount=0;
                size_t aliveMem=0;
                STACK_TYPE* callstackPtr=nullptr;

            };

            int objectsNum = 0;
            int callstackNum = 0;
            int replicNum = 0;
            ssize_t maxWaste = 0;
            std::unordered_set<ssize_t> callstackNumberCalculator;
            std::vector<InfoByCallStack> infoByCallStackArray;
            std::vector<size_t> infoByCallStackSortedIndex; //infoByCallStackSortedIndex[<callstackID>] represents the desired output order of the infoByCallStackArray[<callstackID>]
            for(auto it = aliveObjs.mapObj.begin(); it != aliveObjs.mapObj.end(); it++){
                CTENSOR_TYPE* object = it->second;

                assert(object != nullptr);
                objectsNum++;
                assert(object->callstack!=nullptr);
                callstackNumberCalculator.emplace(object->callstack->callstackID);

                for(ssize_t i=infoByCallStackArray.size();i<object->callstack->callstackID+1;++i){
                    //Make sure there are room to store data.
                    infoByCallStackArray.emplace_back();
                    infoByCallStackSortedIndex.emplace_back(infoByCallStackSortedIndex.size());
                }   

                InfoByCallStack& curCallStackInfo=infoByCallStackArray[object->callstack->callstackID];
                curCallStackInfo.callstackPtr=object->callstack;
                replicNum +=1;
                curCallStackInfo.aliveCount+=1;
                curCallStackInfo.aliveMem+=object->size;

                if(curCallStackInfo.aliveMem > maxWaste) {
                    maxWaste = curCallStackInfo.aliveMem;
                }
            }
            callstackNum=callstackNumberCalculator.size();

            //Please do not use cout or printf as it may cause program crash. Use macro to output the files into a file.
            //OUTPUTS("%d objects include %d callstacks, %d replicating callstacks, and the maximum waste %s !!!\n",
            //        objectsNum,callstackNum, replicNum,format_size(maxWaste).c_str());

            output << objectsNum << " objects include " << callstackNum << " callstacks, " << replicNum << " replicating callstacks, "
                   << "and the maximum waste " << format_size(maxWaste) << "!!!" << std::endl;
            output << std::endl;

            // Sort the vector based on object size (you can customize the sorting order). Note that after this the order of callstack
            std::sort(infoByCallStackSortedIndex.begin(), infoByCallStackSortedIndex.end(),
                      [infoByCallStackArray](const size_t& callstackIDA, const size_t& callstackIDB) {
                            size_t ida=callstackIDA;
                            size_t idb=callstackIDB;
                            return infoByCallStackArray[ida].aliveMem > infoByCallStackArray[idb].aliveMem;
                        });


            output << std::endl;
            // Now printing all objects in decreasing order.
            int i = 0;
            for(auto curCallStackID : infoByCallStackSortedIndex) {
                auto& leakObject = infoByCallStackArray[curCallStackID];
                if(leakObject.aliveCount>0){
                    output << i << "-th object: waste - " << format_size(leakObject.aliveMem) << ", alloc times: " << leakObject.aliveCount
                        << ", callsite level: " << leakObject.callstackPtr->array.size() << ", callsiteID: "<<  leakObject.callstackPtr->callstackID<< std::endl;
                    // << ", unit size - " << format_size(leakObject->size)  I temporarily deleted unit size because the same callstack may have different allocation size.

                    leakObject.callstackPtr->print(output);
                    output << std::endl;

                    i++;
                }else{
                    //This is safe because the array is already sorted
                    break;
                }
                
                // Now we only print 5 objects here.
            }
        }

        void onOutOfMemoryFramework(ssize_t size) {
            std::string fileName = logProcessRootPath + "/memoryprofile_oom.txt";
            std::ofstream output(fileName, std::ios::app);
            //Save memory analyzer results
            this->printOutput(output,0);
            output.close();

            if(isRankParentProcess){
                std::stringstream ss;
                ss<< "memoryprofile_oom_";
                char* logRootPid=getenv("MLINSIGHT_LOGROOT_PID");
                if(logRootPid){
                    ss << logRootPid;
                } 
                ss <<"_" <<getpid()<<"_Rank"<<localRank<< ".txt";
                AROUTPUTS("The program tries to allocate %s, but pytorch allocator reported OOM MLInsight saved the summary log to %s.\n",format_size(size).c_str(),ss.str().c_str());

                std::string logFileName =ss.str();
                std::ofstream outputForUser(logFileName, std::ios::app);
                // Save memory analyzer results
                this->printOutput(outputForUser,0);
                // Save flame graph snapshots
                outputForUser.close();
            }
        }
    };




}

#endif //MLINSIGHT_MEMLEAKANALYZER_H
