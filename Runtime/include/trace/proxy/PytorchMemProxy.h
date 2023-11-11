#ifndef __PYTORCH_MEM_PROXY_H__
#define __PYTORCH_MEM_PROXY_H__
#include <c10/core/Allocator.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/core/Allocator.h>

#include <map>
namespace mlinsight{
extern pthread_mutex_t pytorchMemoryManagementLock;

typedef void (*raw_delete_t)(void*);
typedef c10::Allocator* (*AllocatorGet_t)(void);

extern raw_delete_t realRawDeletePtr;
extern AllocatorGet_t realAllocatorGetPtr;
extern std::atomic<c10::cuda::CUDACachingAllocator::CUDAAllocator*>* realPytorch2AllocatorPtr;

extern void* realGetDeviceStatsPtr;

void raw_delete_proxy(void* ptr);
c10::Allocator* allocator_get_proxy(void);

extern std::map<int,double> cudaCachingAllocatorFractionMap;
void setMemoryFraction_proxy(double fraction, int device);


using namespace c10::cuda::CUDACachingAllocator;
using namespace c10;
using namespace c10::cuda;

class Pytorch2AllocatorProxy : public c10::cuda::CUDACachingAllocator::CUDAAllocator {
  public:
  
  c10::cuda::CUDACachingAllocator::CUDAAllocator* realAllocator=nullptr;


  Pytorch2AllocatorProxy(c10::cuda::CUDACachingAllocator::CUDAAllocator* realAllocator):realAllocator(realAllocator){
    INFO_LOG("Pytorch2AllocatorProxy constructed");
    assert(realAllocator!=nullptr);
  }
    virtual DataPtr allocate(size_t n) const{
        INFO_LOGS("c10::cuda::CUDACachingAllocator allocate invoked+++ size=%zd",n);

        return realAllocator->allocate(n);
    }

  // If this returns a non nullptr, it means that allocate()
  // is guaranteed to return a unique_ptr with this deleter attached;
  // it means the rawAllocate and rawDeallocate APIs are safe to use.
  // This function MUST always return the same BoundDeleter.
  virtual DeleterFnPtr raw_deleter() const {
    return realAllocator->raw_deleter();
  }

  virtual void* raw_allocate(size_t n) {
    INFO_LOG("c10::cuda::CUDACachingAllocator raw_allocate invoked+++");
    return realAllocator->raw_allocate(n);
  }
  virtual void raw_deallocate(void* ptr) {
    INFO_LOG("c10::cuda::CUDACachingAllocator raw_deallocate invoked+++");
    realAllocator->raw_deallocate(ptr);
  }

  virtual void* raw_alloc(size_t nbytes){
    INFO_LOG("c10::cuda::CUDACachingAllocator raw_alloc invoked+++");
    return realAllocator->raw_alloc(nbytes);
  }
  virtual void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream){
    INFO_LOG("c10::cuda::CUDACachingAllocator raw_alloc_with_stream invoked+++");
    return realAllocator->raw_alloc_with_stream(nbytes,stream);

  }
  virtual void raw_delete(void* ptr){
    INFO_LOG("c10::cuda::CUDACachingAllocator raw_delete invoked+++");
    realAllocator->raw_delete(ptr);
  }
  virtual void init(int device_count){
    INFO_LOG("c10::cuda::CUDACachingAllocator init invoked+++");
    realAllocator->init(device_count);

  }
  virtual bool initialized() {
    INFO_LOG("c10::cuda::CUDACachingAllocator initialized invoked+++");
    return realAllocator->initialized();
  }

  virtual void setMemoryFraction(double fraction, int device) {
    INFO_LOG("c10::cuda::CUDACachingAllocator setMemoryFraction invoked+++");
    return realAllocator->setMemoryFraction(fraction,device);
  }

  virtual void emptyCache() {
    INFO_LOG("c10::cuda::CUDACachingAllocator emptyCache invoked+++");
    return realAllocator->emptyCache();

  }
  virtual void cacheInfo(int dev_id, size_t* largestBlock) {
    INFO_LOG("c10::cuda::CUDACachingAllocator cacheInfo invoked+++");
    return realAllocator->cacheInfo(dev_id,largestBlock);
  }
  virtual void* getBaseAllocation(void* ptr, size_t* size) {
    INFO_LOG("c10::cuda::CUDACachingAllocator getBaseAllocation invoked+++");
    return realAllocator->getBaseAllocation(ptr,size);

  }
  virtual void recordStream(const c10::DataPtr& ptr, c10::cuda::CUDAStream stream) {
    INFO_LOG("c10::cuda::CUDACachingAllocator recordStream invoked+++");
    realAllocator->recordStream(ptr,stream);
    
  }
  virtual c10::cuda::CUDACachingAllocator::DeviceStats getDeviceStats(int device) {
    INFO_LOG("c10::cuda::CUDACachingAllocator getDeviceStats invoked+++");
    return realAllocator->getDeviceStats(device);
  }
  virtual void resetAccumulatedStats(int device) {
    INFO_LOG("c10::cuda::CUDACachingAllocator resetAccumulatedStats invoked+++");
    return realAllocator->resetAccumulatedStats(device);
  }
  virtual void resetPeakStats(int device) {
    INFO_LOG("c10::cuda::CUDACachingAllocator resetPeakStats invoked+++");
    return realAllocator->resetPeakStats(device);
  }

  virtual c10::cuda::CUDACachingAllocator::SnapshotInfo snapshot() {
    INFO_LOG("c10::cuda::CUDACachingAllocator snapshot invoked+++");
    return realAllocator->snapshot();

  }
  
  virtual void beginAllocateStreamToPool(
      int device,
      cudaStream_t stream,
      MempoolId_t mempool_id) {
    INFO_LOG("c10::cuda::CUDACachingAllocator beginAllocateStreamToPool invoked+++");
    realAllocator->beginAllocateStreamToPool(device,stream,mempool_id);

  }

  virtual void endAllocateStreamToPool(int device, cudaStream_t stream) {
    INFO_LOG("c10::cuda::CUDACachingAllocator endAllocateStreamToPool invoked+++");

    realAllocator->endAllocateStreamToPool(device,stream);

  }

  virtual void releasePool(int device, MempoolId_t mempool_id) {
    INFO_LOG("c10::cuda::CUDACachingAllocator releasePool invoked+++");

    realAllocator->releasePool(device,mempool_id);
  }

  virtual std::shared_ptr<void> getIpcDevPtr(std::string handle) {
    INFO_LOG("c10::cuda::CUDACachingAllocator getIpcDevPtr invoked+++");
    return realAllocator->getIpcDevPtr(handle);
  }

  virtual void recordHistory(
      bool enabled,
      CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      RecordContext when) {
    INFO_LOG("c10::cuda::CUDACachingAllocator recordHistory invoked+++");

    realAllocator->recordHistory(enabled,context_recorder,alloc_trace_max_entries,when);

  }

  virtual void attachOutOfMemoryObserver(OutOfMemoryObserver observer){
    INFO_LOG("c10::cuda::CUDACachingAllocator attachOutOfMemoryObserver invoked+++");
    realAllocator->attachOutOfMemoryObserver(observer);
  }

  virtual void enablePeerAccess(int dev, int dev_to_access) {
    INFO_LOG("c10::cuda::CUDACachingAllocator enablePeerAccess invoked+++");

    realAllocator->enablePeerAccess(dev,dev_to_access);
  }

  virtual cudaError_t memcpyAsync(
      void* dst,
      int dstDevice,
      const void* src,
      int srcDevice,
      size_t count,
      cudaStream_t stream,
      bool p2p_enabled) {
    INFO_LOG("c10::cuda::CUDACachingAllocator memcpyAsync invoked+++");

    return realAllocator->memcpyAsync(dst,dstDevice,src,srcDevice,count,stream,p2p_enabled);
  }

  virtual std::shared_ptr<AllocatorState> getCheckpointState(int device,MempoolId_t id) {
    INFO_LOG("c10::cuda::CUDACachingAllocator getCheckpointState invoked+++");

        return realAllocator->getCheckpointState(device, id);

  }

  virtual CheckpointDelta setCheckpointPoolState(
      int device,
      std::shared_ptr<AllocatorState> pps){
    INFO_LOG("c10::cuda::CUDACachingAllocator setCheckpointPoolState invoked+++");

        return realAllocator->setCheckpointPoolState(device, pps);
  };

  virtual std::string name() {
    INFO_LOG("c10::cuda::CUDACachingAllocator name invoked+++");

        return realAllocator->name();
  };

};

}

#endif