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
#include "analyse/DriverMemory.h"

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
        //pthread_mutex_lock(&pytorchMemoryManagementLock);
        //INFO_LOGS("CUDA Driver API dlsym: %s", symbol);
        #include <cuda_runtime.h>
        #include <cuda_runtime_api.h>
        #ifdef CUDA_VERSION_121_LATER
            CUdriverProcAddressQueryResult driverStatus;
            CUresult cudaResult = cuGetProcAddress(symbol, pfn, cudaVersion, flags, &driverStatus); 
        #else
            CUresult cudaResult = cuGetProcAddress(symbol, pfn, cudaVersion, flags);
        #endif
        if (cudaResult != CUDA_SUCCESS || *pfn == nullptr) {
            //ERR_LOGS("MLInsight cannot hook CUDA API: %s because cuGetProcAddress failed", symbol);
            //pthread_mutex_unlock(&pytorchMemoryManagementLock);
            return cudaResult;
        }

        HookInstaller *instance = HookInstaller::getInstance();
        SymbolHookHint retSymbolHookHint;
        Elf64_Word bind = STB_GLOBAL;
        Elf64_Word type = STT_FUNC;
        
        instance->shouldHookThisSymbol(symbol, bind, type, retSymbolHookHint);
        //INFO_LOGS("cuGetProcAddress hooking %s (%d)", symbol, retSymbolHookHint.shouldHook);
        if(retSymbolHookHint.realAddressPtr){
            *(retSymbolHookHint.realAddressPtr)=*pfn;
        }

        if (retSymbolHookHint.shouldHook) {
            if (retSymbolHookHint.addressOverride) {
                //Use overrided address
                *pfn = retSymbolHookHint.addressOverride;
            }
            void *retAddr = nullptr;
            if (!instance->installDlSym(*pfn, retAddr)) {
                INFO_LOGS("CUDA API NOT hooked: name:%s bind:%zd type:%zd addr:%p",symbol,bind,type,*pfn);
                ERR_LOGS("Failed to hook %s because of installation failure", symbol);
            } else {
                //INFO_LOGS("Install the hook on %s", symbol);
                *pfn = retAddr;
                //INFO_LOGS("CUDA API hooked: name:%s bind:%zd type:%zd addr:%p",symbol,bind,type,*pfn);
            }
        }

        //pthread_mutex_unlock(&pytorchMemoryManagementLock);
        return cudaResult;
    }

    CUresult CUDAAPI cuMemcpyHtoD_proxy(CUdeviceptr_v1 dstDevice, const void *srcHost, unsigned int ByteCount) {
        //pthread_mutex_lock(&pytorchMemoryManagementLock);
        //INFO_LOGS("CUDA H2D for %u bytes", ByteCount);
        CUresult ret = cuMemcpyHtoD(dstDevice, srcHost, ByteCount);
        //pthread_mutex_unlock(&pytorchMemoryManagementLock);
        return ret;
    }

    CUresult CUDAAPI cuMemAlloc_proxy(CUdeviceptr *dptr, size_t bytesize) {
        //pthread_mutex_lock(&pytorchMemoryManagementLock);
        CUresult ret = cuMemAlloc(dptr, bytesize);
        
        //cout << "cuMemAlloc malloc " << bytesize << "." << endl; 
       // INFO_LOGS("cuMemAlloc malloc %zd bytes ptr %p", bytesize, dptr);
        if (ret == CUDA_SUCCESS) {
            trackDriverAllocation(bytesize, (void *)*dptr);
        }

        //pthread_mutex_unlock(&pytorchMemoryManagementLock);
        return ret;
    }

    CUresult cuMemAllocHost_proxy (void ** pp, size_t bytesize) {
        //pthread_mutex_lock(&pytorchMemoryManagementLock);
        CUresult ret = cuMemAllocHost(pp, bytesize);
        //INFO_LOGS("cuMemAllocHost malloc %zd bytes ptr %p", bytesize, *pp);
        //mlinsight::print_stacktrace(); 
        //INFO_LOGS("cuMemAllocHost malloc %zd bytes ptr %p", bytesize, *pp);
        //pthread_mutex_unlock(&pytorchMemoryManagementLock);
        return ret;
    }	

    CUresult cuMemHostAlloc_proxy (void ** pp, size_t bytesize, unsigned int flags) {
        //pthread_mutex_lock(&pytorchMemoryManagementLock);
        CUresult ret = cuMemHostAlloc(pp, bytesize, flags);
        //INFO_LOGS("cuMemAllocHost malloc %zd bytes ptr %p", bytesize, *pp);
        //mlinsight::print_stacktrace(); 
        //pthread_mutex_unlock(&pytorchMemoryManagementLock);
        return ret;
    }	

    CUresult cuMemAllocManaged_proxy ( CUdeviceptr* dptr, size_t bytesize, unsigned int  flags ) {
        //pthread_mutex_lock(&pytorchMemoryManagementLock);
        CUresult ret = cuMemAllocManaged(dptr, bytesize, flags);
        if (ret == CUDA_SUCCESS) {
            trackDriverAllocation(bytesize, (void *)*dptr);
        }
        //INFO_LOGS("cuMemAllocManaged malloc %zd bytes ptr %p", bytesize, dptr);
        //pthread_mutex_unlock(&pytorchMemoryManagementLock);
        return ret;
    }

    	
    CUresult cuMemFree_proxy (CUdeviceptr dptr) {
        //pthread_mutex_lock(&pytorchMemoryManagementLock);
        CUresult ret = cuMemFree(dptr);
        trackDriverFree((void *)dptr);
        //pthread_mutex_unlock(&pytorchMemoryManagementLock);
        return ret; 
    }

    CUresult cuMemFreeHost_proxy(void * ptr) {
        //pthread_mutex_lock(&pytorchMemoryManagementLock);
        CUresult ret = cuMemFreeHost(ptr); 
        trackDriverFree(ptr);
        //pthread_mutex_unlock(&pytorchMemoryManagementLock);
        return ret;        
    }

    CUresult cuMemFreeAsync_proxy ( CUdeviceptr dptr, CUstream hStream ) {
        //pthread_mutex_lock(&pytorchMemoryManagementLock);
        trackDriverFree((void *)dptr);
        CUresult rlt=cuMemFreeAsync(dptr, hStream); 
        //pthread_mutex_unlock(&pytorchMemoryManagementLock);
        return rlt;
    }


    CUresult cuMemAddressFree_proxy ( CUdeviceptr ptr, size_t size ){
        //pthread_mutex_lock(&pytorchMemoryManagementLock);
        trackDriverFree((void *)ptr);
        CUresult rlt=cuMemAddressFree(ptr, size);
        //printf("iiiiiiiin cuMemHostUnregister_proxy with pointer: %p iiiiiiiin\n", ptr);
        //pthread_mutex_unlock(&pytorchMemoryManagementLock);
        return rlt; 

    }

    CUresult cuMemHostUnregister_proxy ( void* ptr ) {
        //pthread_mutex_lock(&pytorchMemoryManagementLock);
        //printf("iiiiiiiin cuMemHostUnregister_proxy with pointer: %p iiiiiiiin\n", ptr);
        CUresult rlt=cuMemHostUnregister(ptr);
        //pthread_mutex_unlock(&pytorchMemoryManagementLock);
        return rlt;
    }

    CUresult cuMemUnmap_proxy ( CUdeviceptr ptr, size_t size ) {
        //pthread_mutex_lock(&pytorchMemoryManagementLock);

        //printf("iiiiiiiin cuMemUnmap_proxy with pointer: %p iiiiiiiin\n", ptr);
        trackDriverFree((void *)ptr);
        CUresult rlt=cuMemUnmap(ptr, size);
        //pthread_mutex_unlock(&pytorchMemoryManagementLock);
        return rlt; 
    }

    CUresult cuMemCreate_proxy(CUmemGenericAllocationHandle *handle, size_t size, const CUmemAllocationProp *prop, unsigned long long flags){
        CUresult rlt = cuMemCreate(handle, size, prop, flags);
        INFO_LOGS("cuMemCreate malloc %zd bytes ptr %p", size, handle);
        return rlt;
    }

    CUresult cuMemMap_proxy(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags){
        CUresult rlt = cuMemMap(ptr,size,offset,handle,flags);
        INFO_LOGS("cuMemMap malloc %zd bytes ptr %p", size, ptr);
        return rlt;
    }

}
