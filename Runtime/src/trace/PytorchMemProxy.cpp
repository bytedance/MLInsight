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
#include "analyse/PytorchMemory.h"
#include "analyse/DriverMemory.h"

namespace mlinsight{
typedef c10::DataPtr (*allocate_t)(void * ptr, size_t bytes);

#ifndef TORCH_VERSION_20_LATER
struct CudaCachingAllocatorProxy : public c10::Allocator {
   c10::Allocator* realCUDACachineAllocatorPtr;
   CudaCachingAllocatorProxy(Allocator* realCUDACachineAllocatorPtr):realCUDACachineAllocatorPtr(realCUDACachineAllocatorPtr){
  }



  c10::DataPtr allocate(size_t size) const override {
    //pthread_mutex_lock(&pytorchMemoryManagementLock);
    try {
      c10::DataPtr allocatePtr=realCUDACachineAllocatorPtr->allocate(size);
      trackPytorchAllocation(size, allocatePtr.get());
      return allocatePtr;
    }
    catch (const c10::CUDAOutOfMemoryError& e)
    {  //c10/util/Exception.h

      reportMemoryProfile(size);
      //processCUDAOOMError(e, allocatedSize);
      //pthread_mutex_unlock(&pytorchMemoryManagementLock);
      throw e;
    } catch (const std::exception& e){
      //pthread_mutex_unlock(&pytorchMemoryManagementLock);
      throw e;
    }
  }
};




raw_delete_t realRawDeletePtr=nullptr;
allocate_t realAllocatePtr=nullptr;
AllocatorGet_t realAllocatorGetPtr=nullptr;
c10::Allocator* cudaAllocatorProxyPtr=nullptr;
void* realGetDeviceStatsPtr=nullptr;

static void raw_delete_proxy(void* ptr){
  //pthread_mutex_lock(&pytorchMemoryManagementLock);
  assert(realRawDeletePtr!=nullptr);
  printf("******raw_delete_proxy ptr %p now!!!!!!*****\n", ptr);
  trackPytorchFree(ptr);
  realRawDeletePtr(ptr);
  //pthread_mutex_unlock(&pytorchMemoryManagementLock);
}


c10::Allocator* allocator_get_proxy(void) {
    //pthread_mutex_lock(&pytorchMemoryManagementLock);
    //Get the pointer of CUDACachingAllocator by invoke the real allocator_get function, and record this to allocatorPtr variable. 
    if(cudaAllocatorProxyPtr == nullptr){
        c10::Allocator* realAllocatorPtr = realAllocatorGetPtr();
        cudaAllocatorProxyPtr = new CudaCachingAllocatorProxy(realAllocatorPtr);
    }
    //pthread_mutex_unlock(&pytorchMemoryManagementLock);
    return cudaAllocatorProxyPtr;  
}

  std::map<int,double> cudaCachingAllocatorFractionMap;
  void setMemoryFraction_proxy(double fraction, int device){
    //pthread_mutex_lock(&pytorchMemoryManagementLock);
    cudaCachingAllocatorFractionMap[device]=fraction;
    //pthread_mutex_unlock(&pytorchMemoryManagementLock);
  }
#else
  c10::DeleterFnPtr realDeleter = nullptr;
  std::atomic<c10::cuda::CUDACachingAllocator::CUDAAllocator*>* realPytorch2AllocatorPtr=nullptr;
#endif

}
