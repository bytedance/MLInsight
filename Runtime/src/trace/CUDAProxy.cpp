/*
@author: Steven (Jiaxun) Tang <jtang@umass.edu>
@author: Tongping Liu <tongping.liu@bytedance.com>
*/
#include <cstddef>

#include <dlfcn.h>
#include <cstring>
#include <iostream>
#include "common/Logging.h"
#include "common/Tool.h"
#include "trace/proxy/CUDAProxy.h"
#include "trace/hook/HookInstaller.h"
#include "trace/hook/HookInstaller.h"
#include "analyse/GlobalVariables.h"
#include "trace/hook/HookContext.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
using namespace std;

namespace mlinsight {
    unsigned int cudaCount = 0;

    typedef CUresult CUDAAPI(*CuGetProcAddrType)
            (
                    const char *symbol,
                    void **pfn,
                    int cudaVersion, cuuint64_t
                    flags);


    CUresult CUDAAPI cuGetProcAddress_proxy(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags) {
        //INFO_LOGS("CUDA Driver API dlsym: %s", symbol);
        if(!curContext){
            initTLS();
        }
#if CUDART_VERSION >= 12010
        CUdriverProcAddressQueryResult driverStatus;
        CUresult cudaResult = cuGetProcAddress(symbol, pfn, cudaVersion, flags, &driverStatus);
#elif CUDART_VERSION >= 11030 //todo: This value is not accurate
        CUresult cudaResult = cuGetProcAddress(symbol, pfn, cudaVersion, flags);

#else
        fatalError("This should be impossible. CUDA version below 11.x does not have cuGetProcAddress");
        return CUDA_ERROR_UNKNOWN;
        CUresult cudaResult;
#endif
        if (cudaResult != CUDA_SUCCESS || *pfn == nullptr) {
            //ERR_LOGS("MLInsight cannot hook CUDA API: %s because cuGetProcAddress failed", symbol);
            return cudaResult;
        }
        //todo: Currently, MLInsight has turned off the C++ API hook. That is why using this method works. Otherwise the return address would have to be retrived from MLInsight's stack.
        void *callerAddr = __builtin_return_address(0);
        assert(callerAddr != nullptr);

        HookInstaller *instance = HookInstaller::getInstance();
        void *mlinsightReturnAddress = nullptr;
        instance->installOnDlSym(symbol, *pfn, callerAddr, mlinsightReturnAddress);
        *pfn = mlinsightReturnAddress;

        return cudaResult;
    }


    CUresult CUDAAPI cuMemAlloc_proxy(CUdeviceptr *dptr, size_t bytesize) {
        if(!curContext){
            initTLS();
        }
        pthread_mutex_lock(&analyzerLock);
#ifndef NDEBUG
        if(cuptiCrossChecker.cuptiCrossCheckingEnabled){
            //This value will be used in cupti callback to detect whether MLInsight
            cuptiCrossChecker.insideMLInsightCuMemAllocProxy=true;
        }
#endif

        memLeakAnalyzer.onPreAllocDriver(bytesize);
        perfettoAnalyzer.onPreAllocDriver(bytesize); //Perfetto analyzer must be called after its dependencies analyzers

        CUresult ret = cuMemAlloc(dptr, bytesize);

        INFO_LOGS("cuMemAlloc_proxy(%p,%zd)",(void*)*dptr,bytesize);

        DriverTensorType* newTensor = mapDriverAliveObjs.insert(bytesize,(void*)*dptr);
        if(newTensor){
            newTensor->updateCallStack();
            newTensor->isAllocatedByFramework = globalExecutionState.isInvokingFrameworkMemOp;
        }

        memLeakAnalyzer.onPostAllocDriver(bytesize,(void*)*dptr,newTensor);
        perfettoAnalyzer.onPostAllocDriver(bytesize,(void*)*dptr,newTensor); //Perfetto analyzer must be called after its dependencies analyzers

#ifndef NDEBUG
        if (cuptiCrossChecker.cuptiCrossCheckingEnabled) {
            cuptiCrossChecker.cuMemAllocMLInsightSize += bytesize;
            cuptiCrossChecker.cuMemAllocMlInsightPtr = (void*)*dptr;

            INFO_LOG("CUPTI CROSSCHECK IS WORKING");
            if (cuptiCrossChecker.cuMemAllocCUPTISize != cuptiCrossChecker.cuMemAllocMLInsightSize ||
                                    cuptiCrossChecker.cuMemAllocMlInsightPtr != cuptiCrossChecker.cuMemAllocCuptiPtr) {
                fatalErrorS(
                        "Detected discrepancies between MLInsight and CUPTI interception of cuMemAlloc. Please check interception parts. MLInsight:%zd Cupti:%zd MLInsight:%p Cupti:%p",
                        cuptiCrossChecker.cuMemAllocMLInsightSize, cuptiCrossChecker.cuMemAllocCUPTISize,
                        cuptiCrossChecker.cuMemAllocMlInsightPtr, cuptiCrossChecker.cuMemAllocCuptiPtr);
            }
            cuptiCrossChecker.insideMLInsightCuMemAllocProxy=false;
        }
#endif

        //OUTPUTS("cuMemAlloc malloc %d.\n",bytesize);
        if (ret == CUDA_SUCCESS) {
            //memLeakAnalyzer.onPostAllocDriver(bytesize, (void*) dptr);
        }

        pthread_mutex_unlock(&analyzerLock);

        return ret;
    }


    CUresult cuMemFree_proxy(CUdeviceptr dptr) {
        if(!curContext){
            initTLS();
        }

        pthread_mutex_lock(&analyzerLock);

        INFO_LOGS("cuMemFree_proxy is called %p",(void*)dptr);
#ifndef NDEBUG
        if (cuptiCrossChecker.cuptiCrossCheckingEnabled) {
            cuptiCrossChecker.cuMemFreeMlInsightPtr = (void*) dptr;
            cuptiCrossChecker.insideMLInsightCuMemFreeProxy=true;
        }
#endif
        void* ptr= (void*)dptr;
        CUresult ret;
        if(ptr){
            DriverTensorType * justRemovedTensor = mapDriverAliveObjs.remove(ptr);
            memLeakAnalyzer.onPreFreeDriver(ptr,justRemovedTensor);
#if USE_PERFETTO
            perfettoAnalyzer.onPreFreeDriver(ptr,justRemovedTensor); //Perfetto analyzer must be called after its dependencies analyzers
#endif
            

            ret = cuMemFree(dptr);
            

            memLeakAnalyzer.onPostFreeDriver(ptr,justRemovedTensor);
#if USE_PERFETTO
            perfettoAnalyzer.onPostFreeDriver(ptr,justRemovedTensor); //Perfetto analyzer must be called after its dependencies analyzers
#endif
        }else{
            //Still call CUDA even if the pointer is null
            ret = cuMemFree(dptr);
        }

#ifndef NDEBUG
        if (cuptiCrossChecker.cuptiCrossCheckingEnabled) {

            if (cuptiCrossChecker.cuMemFreeMlInsightPtr != cuptiCrossChecker.cuMemFreeCuptiPtr) {
                fatalErrorS(
                        "Detected discrepancies between MLInsight and CUPTI interception of cuMemFree. Please check interception parts. MLInsight:%p Cupti:%p",
                        cuptiCrossChecker.cuMemFreeMlInsightPtr, cuptiCrossChecker.cuMemFreeCuptiPtr);
            }
            cuptiCrossChecker.insideMLInsightCuMemFreeProxy=false;
        }
#endif

        pthread_mutex_unlock(&analyzerLock);
        return ret;
    }

}
