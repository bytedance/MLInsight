#ifndef __PYTORCH_MEM_PROXY_H__
#define __PYTORCH_MEM_PROXY_H__

#include <map>
#include "trace/hook/HookInstaller.h"

/**
 * The following functions are interfaces used by non-pytorch part of the MLInsight
 */
namespace mlinsight{
    namespace pytorch {
        /**
         * Called at the initialization time of the HookInstaller
         */
        void onSettingHookHint(std::map<std::string, SymbolHookHint>& hookHintMap) __attribute__((weak));

        /**
         * Called after every installation of the hookInstaller
         */
        void onHookInstallationFinished() __attribute__((weak));

    }
}


/**
 * The following functions are closely related to Pytorch API and should not be referenced by non-pytorch code
 */
#ifdef USE_TORCH
#include <c10/core/Allocator.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/core/Allocator.h>
#include <unordered_map>
#include "common/DependencyLibVersionSpecifier.h"

namespace mlinsight {

#if TORCH_VERSION_MAJOR >= 2
    extern std::atomic<c10::cuda::CUDACachingAllocator::CUDAAllocator *> *realPytorch2AllocatorPtr;
    using namespace c10::cuda::CUDACachingAllocator;
    using namespace c10;
    using namespace c10::cuda;
    extern c10::DeleterFnPtr realDeleter;

    class Pytorch2AllocatorProxy : public c10::cuda::CUDACachingAllocator::CUDAAllocator {
    public:

        c10::cuda::CUDACachingAllocator::CUDAAllocator *realAllocator = nullptr;
        mutable std::unordered_map<void*, void*> realRawDeleterMap;

        Pytorch2AllocatorProxy(c10::cuda::CUDACachingAllocator::CUDAAllocator *realAllocator) : realAllocator(
                realAllocator) {
            assert(realAllocator != nullptr);
        }

        void init(int device_count) override {
            realAllocator->init(device_count);
        }

        bool initialized() override {
            return realAllocator->initialized();
        }

        void setMemoryFraction(double fraction, int device) override {
            realAllocator->setMemoryFraction(fraction, device);
        }

        void recordHistory(
                bool enabled,
                CreateContextFn context_recorder,
                size_t alloc_trace_max_entries,
                RecordContext when) override {
            realAllocator->recordHistory(enabled, context_recorder, alloc_trace_max_entries, when);
        }

        bool isHistoryEnabled() override {
            return realAllocator->isHistoryEnabled();
        }

        bool checkPoolLiveAllocations(
                int device,
                MempoolId_t mempool_id,
                const std::unordered_set<void *> &expected_live_allocations) override {
            return realAllocator->checkPoolLiveAllocations(device, mempool_id, expected_live_allocations);
        }


        void attachOutOfMemoryObserver(OutOfMemoryObserver observer) override {
            realAllocator->attachOutOfMemoryObserver(observer);
        }

        void emptyCache() override {
            return realAllocator->emptyCache();
        }

        void *getBaseAllocation(void *ptr, size_t *size) override {
            return realAllocator->getBaseAllocation(ptr, size);
        }

        void recordStream(const c10::DataPtr &ptr, c10::cuda::CUDAStream stream) override;

        c10::cuda::CUDACachingAllocator::SnapshotInfo snapshot() override {
            return realAllocator->snapshot();
        }

        std::shared_ptr<AllocatorState> getCheckpointState(int device, MempoolId_t id) override {
            return realAllocator->getCheckpointState(device, id);
        }

        CheckpointDelta setCheckpointPoolState(
                int device,
                std::shared_ptr<AllocatorState> pps) override {
            return realAllocator->setCheckpointPoolState(device, pps);
        };


        void beginAllocateStreamToPool(
                int device,
                cudaStream_t stream,
                MempoolId_t mempool_id) override {
            realAllocator->beginAllocateStreamToPool(device, stream, mempool_id);
        }

        void* convert2DevicePointer(DataPtr& ptr) const{
            return ptr.get();
        }

        DataPtr allocate(size_t n) const override;

        DataPtr callRealPytorchAllocator(ssize_t n) const{
            DataPtr data_ptr = this->realAllocator->allocate(n);
            return data_ptr;
        }

        void* converPytorchPtrToCPtr(DataPtr data_ptr){
            void *ptr = data_ptr.get();
            return ptr;
        }


        // If this returns a non nullptr, it means that allocate()
        // is guaranteed to return a unique_ptr with this deleter attached;
        // it means the rawAllocate and rawDeallocate APIs are safe to use.
        // This function MUST always return the same BoundDeleter.
        DeleterFnPtr raw_deleter() const override {
            return realAllocator->raw_deleter();
        }

        void cacheInfo(int dev_id, size_t *largestBlock) override {
            return realAllocator->cacheInfo(dev_id, largestBlock);
        }

        c10::cuda::CUDACachingAllocator::DeviceStats getDeviceStats(int device) override {
            return realAllocator->getDeviceStats(device);
        }

        void resetAccumulatedStats(int device) override {
            return realAllocator->resetAccumulatedStats(device);
        }

        void resetPeakStats(int device) override {
            return realAllocator->resetPeakStats(device);
        }

        void endAllocateStreamToPool(int device, cudaStream_t stream) override {
            realAllocator->endAllocateStreamToPool(device, stream);
        }

        void releasePool(int device, MempoolId_t mempool_id) override {
            realAllocator->releasePool(device, mempool_id);
        }

        void *raw_alloc(size_t nbytes) override {
            return realAllocator->raw_alloc(nbytes);
        }

        void *raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) override {
            return realAllocator->raw_alloc_with_stream(nbytes, stream);
        }

        void enablePeerAccess(int dev, int dev_to_access) override {
            realAllocator->enablePeerAccess(dev, dev_to_access);
        }

        cudaError_t memcpyAsync(
                void *dst,
                int dstDevice,
                const void *src,
                int srcDevice,
                size_t count,
                cudaStream_t stream,
                bool p2p_enabled) override {
            return realAllocator->memcpyAsync(dst, dstDevice, src, srcDevice, count, stream, p2p_enabled);
        }

        void raw_delete(void *ptr) override {
            realAllocator->raw_delete(ptr);
        }

        std::shared_ptr<void> getIpcDevPtr(std::string handle) override {
            return realAllocator->getIpcDevPtr(handle);
        }

        std::string name() override {
            return realAllocator->name();
        }

    };

    extern Pytorch2AllocatorProxy *allocatorProxy;

#else
    //Pytorch 1.x
    extern void *realGetDeviceStatsPtr;

    void raw_delete_proxy(void *ptr);

    c10::Allocator *allocator_get_proxy(void);

    extern std::map<int, double> cudaCachingAllocatorFractionMap;

    void setMemoryFraction_proxy(double fraction, int device);

    typedef void (*raw_delete_t)(void *);

    typedef c10::Allocator *(*AllocatorGet_t)(void);

    extern AllocatorGet_t realAllocatorGetPtr;
    extern raw_delete_t realRawDeletePtr;



    struct CudaCachingAllocatorProxy : public c10::Allocator {

        c10::Allocator *realCUDACachineAllocatorPtr;

        CudaCachingAllocatorProxy(Allocator *realCUDACachineAllocatorPtr) : realCUDACachineAllocatorPtr(
                realCUDACachineAllocatorPtr) {
        }


        c10::DataPtr allocate(size_t size) const override;
    };

#endif //TORCH_VERSION_MAJOR >= 2

}
#endif //USE_TORCH


#endif //__PYTORCH_MEM_PROXY_H__
