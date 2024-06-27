/*

@author: Steven Tang <steven.tang@bytedance.com>
*/

#ifndef MLINSIGHT_CUDAPROXY_H
#define MLINSIGHT_CUDAPROXY_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace mlinsight {
    typedef unsigned int CUdeviceptr_v1; //Copied from CUDA

    CUresult CUDAAPI cuGetProcAddress_proxy(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags);

    CUresult CUDAAPI cuMemAlloc_proxy(CUdeviceptr *dptr, size_t bytesize);

    CUresult cuMemFree_proxy(CUdeviceptr dptr);


#if CUDART_VERSION > 12010
    CUresult cuMemCreate_proxy(CUmemGenericAllocationHandle *handle, size_t size, const CUmemAllocationProp *prop, unsigned long long flags);
    CUresult cuMemMap_proxy(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags);


#endif
}
#endif
