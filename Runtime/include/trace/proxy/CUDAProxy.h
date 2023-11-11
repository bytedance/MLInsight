/*

@author: Steven Tang <steven.tang@bytedance.com>
*/

#ifndef MLINSIGHT_CUDAPROXY_H
#define MLINSIGHT_CUDAPROXY_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace mlinsight{
typedef unsigned int CUdeviceptr_v1; //Copied from CUDA
extern pthread_mutex_t pytorchMemoryManagementLock;

CUresult CUDAAPI cuGetProcAddress_proxy(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags);


CUresult CUDAAPI cuMemcpyHtoD_proxy(CUdeviceptr_v1 dstDevice, const void *srcHost, unsigned int ByteCount);
CUresult CUDAAPI cuMemAlloc_proxy(CUdeviceptr *dptr, size_t 	bytesize);	
CUresult CUDAAPI cuMemAllocHost_proxy(void ** pptr, size_t 	bytesize);	
CUresult CUDAAPI cuMemHostAlloc_proxy(void ** ptr, size_t bytesize, unsigned int  flags);	 
CUresult cuMemAllocManaged_proxy ( CUdeviceptr* dptr, size_t bytesize, unsigned int  flags );
CUresult cuMemFree_proxy (CUdeviceptr dptr);
CUresult cuMemFreeHost_proxy (void * ptr);
CUresult cuMemFreeAsync_proxy ( CUdeviceptr dptr, CUstream hStream );
CUresult cuMemAddressFree_proxy ( CUdeviceptr ptr, size_t size );
CUresult cuMemHostUnregister_proxy ( void* ptr );
CUresult cuMemUnmap_proxy ( CUdeviceptr ptr, size_t size );

#ifdef CUDA_VERSION_121_LATER
cudaError_t cudaMalloc_proxy ( void** devPtr, size_t size );
cudaError_t cudaMallocManaged_proxy ( void** devPtr, size_t size, unsigned int  flags);
cudaError_t cudaFree_proxy ( void* devPtr );
#endif
}
#endif