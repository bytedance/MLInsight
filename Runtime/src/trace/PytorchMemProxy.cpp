/*
@author: Steven (Jiaxun) Tang <jtang@umass.edu>
@author: Tongping Liu <tongping.liu@bytedance.com>
*/
#include <cassert>
#include <stdio.h>
#include <sys/mman.h>
#include <iostream>
#include <string.h>
#include "common/Logging.h"
#include "trace/proxy/PytorchMemProxy.h"
#include "common/DependencyLibVersionSpecifier.h"
#include "analyse/GlobalVariables.h"

#if TORCH_VERSION_MAJOR >= 2
    namespace c10::detail {
        void deleteNothing(void*) __attribute__((weak));
    } // namespace c10::detail
#endif


namespace mlinsight {

    typedef c10::DataPtr (*allocate_t)(void *ptr, size_t bytes);

    /**
     * This class contains pytorch version non-specific code to dispatch the hook events to subclasses
     * @tparam CTENSOR_TYPE
     */
    template<typename CTENSOR_TYPE>
    class PytorchEventDispatcher: public SimpleCallback<CTENSOR_TYPE>{
    public:
        /**
        * [Interface]
        */
        void onPreAlloc(ssize_t size){
            globalExecutionState.onPreAlloc(size); //This should just be invoked before all other analyzer classes that use globalExecutionState
            debugAnalyzer.onPreAllocFramework(size);
            flameGraphAnalyser.onPreAlloc(size);
            memLeakAnalyzer.onPreAllocFramework(size);
            perfettoAnalyzer.onPreAllocFramework(size); //Perfetto analyzer must be called after its dependencies analyzers
        }

        /**
        * [Interface]
        */
        void onPostAlloc(ssize_t size, void *ptr,CTENSOR_TYPE* newTensor){
            memLeakAnalyzer.onPostAllocFramework(size, ptr, newTensor);
            flameGraphAnalyser.onPostAlloc(size, ptr, newTensor);
            debugAnalyzer.onPostAllocFramework(size, ptr, newTensor);
            perfettoAnalyzer.onPostAllocFramework(size,ptr,newTensor); //Perfetto analyzer must be called after its dependencies analyzers
            globalExecutionState.onPostAlloc(size, ptr, newTensor); //This should be the last analyzer to call
        }

        /**
        * [Interface]
        */
        void onPreFree(void *ptr,CTENSOR_TYPE* justFreedTensor){
            globalExecutionState.onPreFree(ptr, justFreedTensor); //This must be the first to call
            memLeakAnalyzer.onPreFreeFramework(ptr, justFreedTensor);
            flameGraphAnalyser.onPreFree(ptr, justFreedTensor);
            perfettoAnalyzer.onPreFreeFramework(ptr,justFreedTensor); //Perfetto analyzer must be called after its dependencies analyzers
        }
        /**
        * [Interface]
        */
        void onPostFree(void* ptr,CTENSOR_TYPE* justFreedTensor){
            flameGraphAnalyser.onPostFree(ptr, justFreedTensor);
            memLeakAnalyzer.onPostFreeFramework(ptr, justFreedTensor);
            perfettoAnalyzer.onPostFreeFramework(ptr,justFreedTensor); //Perfetto analyzer must be called after its dependencies analyzers
            globalExecutionState.onPostFree(ptr, justFreedTensor); //This must be last one to call
        }

    };

    Pytorch2AllocatorProxy *allocatorProxy=nullptr;
    PytorchEventDispatcher<FramekworkTensorType> eventDispatcher;
#if TORCH_VERSION_MAJOR >= 2
    std::atomic<c10::cuda::CUDACachingAllocator::CUDAAllocator *> *realPytorch2AllocatorPtr = nullptr;
    c10::DeleterFnPtr realDeleter = nullptr;

    static void deleter_proxy(void *ptr) {
        pthread_mutex_lock(&analyzerLock);
        assert(realDeleter != nullptr);
        FramekworkTensorType *justRemovedTensor = mapFrameworkAliveObjs.remove(ptr);
        eventDispatcher.onPreFree(ptr,justRemovedTensor);
        realDeleter(ptr);
        eventDispatcher.onPostFree(ptr,justRemovedTensor);
        pthread_mutex_unlock(&analyzerLock);

    }


    DataPtr Pytorch2AllocatorProxy::allocate(size_t n) const {
        pthread_mutex_lock(&analyzerLock);
        eventDispatcher.onPreAlloc(static_cast<ssize_t>(n));
        DataPtr data_ptr;
        try{
        data_ptr = realAllocator->allocate(n);
        }catch(const c10::Error& e){
            memLeakAnalyzer.onOutOfMemoryFramework(n);
            perfettoAnalyzer.onOutOfMemoryFramework(n);
            pthread_mutex_unlock(&analyzerLock);
            exit(-1);//todo: rethrow e will cause an error. It is possibly a Pytorch problem? So we currently directly call exit
        }
        void *ptr = data_ptr.get();

        FramekworkTensorType *newTensor = mapFrameworkAliveObjs.insert(static_cast<ssize_t>(n), ptr);
        if(newTensor) {
            //if ptr is nullptr newTensor would be null and this will be a failed allocation
            newTensor->updateCallstack();
        }
        eventDispatcher.onPostAlloc(static_cast<ssize_t>(n), ptr, newTensor);
        
        if (n == 0 || data_ptr.get() == NULL){
            pthread_mutex_unlock(&analyzerLock);
            return data_ptr;
        }

        realDeleter = data_ptr.get_deleter();
        // Switch the deleter so that we could interce the deleter function.
        bool success = data_ptr.compare_exchange_deleter(realDeleter, (c10::DeleterFnPtr) &deleter_proxy);
        const_cast<Pytorch2AllocatorProxy*>(this)->realRawDeleterMap[ptr]=(void*)realDeleter;
        if (!success) {
            fatalError("Failed to hook Pytorch deleter because pointer exchange failed. This should not happen");
        }
        pthread_mutex_unlock(&analyzerLock);


        // Track the allocation
        return data_ptr;
    }

    void Pytorch2AllocatorProxy::recordStream(const c10::DataPtr &ptr, c10::cuda::CUDAStream stream) {
            pthread_mutex_lock(&analyzerLock);
            if (!ptr.get()) {
                pthread_mutex_unlock(&analyzerLock);
                realAllocator->recordStream(ptr, stream);
                pthread_mutex_unlock(&analyzerLock);
                return;
            }

            auto findIter = this->realRawDeleterMap.find(ptr.get());
            const c10::DeleterFnPtr& realDeleter=(const c10::DeleterFnPtr&)findIter->second;
            c10::DataPtr &nonConstPtr=const_cast<c10::DataPtr&>(ptr);
            bool success = nonConstPtr.compare_exchange_deleter(deleter_proxy, realDeleter);
            if (!success) {
                fatalError("Failed to hook Pytorch deleter because pointer exchange failed. This should not happen");
            }
            realAllocator->recordStream(ptr, stream);
            success = nonConstPtr.compare_exchange_deleter(realDeleter, deleter_proxy);
            if (!success) {
                fatalError("Failed to hook Pytorch deleter because pointer exchange failed. This should not happen");
            }
            pthread_mutex_unlock(&analyzerLock);
    }
#else
//     allocate_t realAllocatePtr = nullptr;
//     raw_delete_t realRawDeletePtr = nullptr;
//     AllocatorGet_t realAllocatorGetPtr = nullptr;
//     c10::Allocator *cudaAllocatorProxyPtr = nullptr;
//     void *realGetDeviceStatsPtr = nullptr;


//     void raw_delete_proxy(void *ptr) {
//         DBG_MEMORY_RACE_CONDITION_DETECTOR_LOCK
//         assert(realRawDeletePtr != nullptr);
//         //INFO_LOGS("******raw_delete_proxy ptr %p now!!!!!!*****\n", ptr);
//         //todo: Temporary debug, let the IDE report type error

//         assert(realDeleter != nullptr);
//         FramekworkTensorType *justRemovedTensor = mapFrameworkAliveObjs.remove(ptr);
//         eventDispatcher.onPreFree(ptr,justRemovedTensor);
//         realRawDeletePtr(ptr);
//         eventDispatcher.onPostFree(ptr,justRemovedTensor);

//         mapFrameworkAliveObjs.erase(justRemovedTensor);


//         DBG_MEMORY_RACE_CONDITION_DETECTOR_UNLOCK
//     }


//     c10::Allocator *allocator_get_proxy(void) {
//         DBG_MEMORY_RACE_CONDITION_DETECTOR_LOCK

//         //Get the pointer of CUDACachingAllocator by invoke the real allocator_get function, and record this to allocatorPtr variable.
//         if (cudaAllocatorProxyPtr == nullptr) {
//             c10::Allocator *realAllocatorPtr = realAllocatorGetPtr();
//             cudaAllocatorProxyPtr = new CudaCachingAllocatorProxy(realAllocatorPtr);
//             fatalError("allocator_get_proxy");
//         }
//         DBG_MEMORY_RACE_CONDITION_DETECTOR_UNLOCK
//         return cudaAllocatorProxyPtr;
//     }

//     std::map<int, double> cudaCachingAllocatorFractionMap;

//     void setMemoryFraction_proxy(double fraction, int device) {
//         DBG_MEMORY_RACE_CONDITION_DETECTOR_LOCK
//         cudaCachingAllocatorFractionMap[device] = fraction;
//         DBG_MEMORY_RACE_CONDITION_DETECTOR_UNLOCK
//     }


//     c10::DataPtr CudaCachingAllocatorProxy::allocate(size_t size) const {
//     DBG_MEMORY_RACE_CONDITION_DETECTOR_LOCK
//     try {
//         eventDispatcher.onPreAlloc(size);
//         c10::DataPtr allocatePtr = realCUDACachineAllocatorPtr->allocate(size);
//         void* ptr = allocatePtr.get();
//         this->realRawDeleterMap.emplace_back(std::make_pair(ptr,allocatePtr.raw_deleter()));

//         FramekworkTensorType* newTensor = mapFrameworkAliveObjs.insert(size,ptr);
//         if(newTensor){
//             newTensor->updateCallstack();
//         }
//         eventDispatcher.onPostAlloc(size,ptr,newTensor);

//         //INFO_LOGS("trackPytorchAllocation ptr %p size %lx\n", ptr, size);

//         //todo: test
//         //std::ofstream outputForUser("logFileName.txt", std::ios::app);
//         //memLeakAnalyzer.printOutput(outputForUser,15);
//         return allocatePtr;
//     }
//     catch (const c10::CUDAOutOfMemoryError& e)
//     {  //c10/util/Exception.h
//         //reportMemoryProfile(size);
//         //processCUDAOOMError(e, allocatedSize);
//         DBG_MEMORY_RACE_CONDITION_DETECTOR_UNLOCK
//         throw e;
//     } catch (const std::exception &e) {
//         DBG_MEMORY_RACE_CONDITION_DETECTOR_UNLOCK
//         throw e;
//     }
// }
#endif //TORCH_VERSION_MAJOR >= 2

    void pytorch::onSettingHookHint(std::map<std::string, SymbolHookHint> &hookHintMap) {
#if TORCH_VERSION_MAJOR >= 2
        //Hook pytroch 2.x driverMemRecord allocator
        hookHintMap.insert(std::make_pair("_ZN3c104cuda20CUDACachingAllocator9allocatorE",
                                          SymbolHookHint(false, nullptr, (void **) &realPytorch2AllocatorPtr, 0,SymbolSpecialHandlingMarker::NO_SPECIAL_HANDLING)));


#else
        // //Hook Pytorch 1.x memory allocator
        // ProxySymbol proxySymbol[] = {
        //         {"setMemoryFraction", (void *) setMemoryFraction_proxy},
        //         {"_ZN3c104cuda20CUDACachingAllocator3getEv", (void *) allocator_get_proxy,
        //          (void **) &realAllocatorGetPtr},
        //         {"_ZN3c104cuda20CUDACachingAllocator10raw_deleteEPv", (void *) raw_delete_proxy,
        //          (void **) &realRawDeletePtr},
        //         {"_ZN3c104cuda20CUDACachingAllocator14getDeviceStatsEi", nullptr, (void **) &realGetDeviceStatsPtr}
        // };
        // const ssize_t proxySymbolArrSize = sizeof(proxySymbol) / sizeof(proxySymbol[0]);
        // for (int i = 0; i < proxySymbolArrSize; ++i) {
        //     hookHintMap.insert(std::make_pair(proxySymbol[i].name, SymbolHookHint(proxySymbol[i].address,
        //                                                                           proxySymbol[i].realAddressPtr)));
        // }
#endif
    }

    void pytorch::onHookInstallationFinished() {
#if TORCH_VERSION_MAJOR >= 2
        if (realPytorch2AllocatorPtr && !allocatorProxy) { //Address found but not installed
            //INFO_LOGS("Pid:%zd The address of pytorch driverMemRecord allocator is %p",getpid(),realPytorch2AllocatorPtr->load());
            allocatorProxy = new Pytorch2AllocatorProxy(realPytorch2AllocatorPtr->load());
            realPytorch2AllocatorPtr->store(allocatorProxy);
        }
#ifndef NDEBUG
        else if (realPytorch2AllocatorPtr && allocatorProxy) {
            //Address found and installed, check whether the replaced address is the same as we expected. (This verifies that Pytorch 2 only has one allocator instance)
            //When this branch hits it means that the symbol is found multiple times. To confirm the Pytorch allocator is indeed replaced to the correct value, we can compare the results again
            assert(realPytorch2AllocatorPtr->load() == allocatorProxy);
        }
#endif //NDEBUG

#endif //TORCH_VERSION_MAJOR >= 2
    }



}



