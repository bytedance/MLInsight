/*
@author: Steven (Jiaxun) Tang <jtang@umass.edu>
@author: Tongping Liu <tongping.liu@bytedance.com>
*/
#include "trace/proxy/TensorflowMemProxy.h"
#include <tensorflow/core/common_runtime/gpu/gpu_process_state.h>
#include "analyse/GlobalVariables.h"
namespace mlinsight {

    typedef ::tensorflow::GPUProcessState *(*singletonPtr_t)(::tensorflow::GPUProcessState *ps);
    typedef ssize_t TFAllocatorId ;

    class TFGPUProcessStateProxy;
    class TFAllocatorProxy;

    static singletonPtr_t realSingletonAddr=nullptr;

    static ::tensorflow::GPUProcessState * realProcessState=nullptr;
    static TFGPUProcessStateProxy* myGPUProcStatePtr=nullptr;
    static std::map<::tensorflow::Allocator*,TFAllocatorId> realAllocatorIdMap;
    static std::vector<TFAllocatorProxy*> tfAllocatorArray;




    inline ::tensorflow::Allocator* getRealAllocatorPointer(TFAllocatorProxy *thiz) {
        auto* thisPtr=(uint8_t*) thiz;
        auto** addrStrPtr= (::tensorflow::Allocator**)(thisPtr-sizeof(::tensorflow::Allocator*));
        return *addrStrPtr;
    }

    class TFAllocatorProxy {
    public:
        static constexpr size_t kAllocatorAlignment = 64;

        virtual ~TFAllocatorProxy(){

        }

        virtual std::string Name() {
            return getRealAllocatorPointer(this)->Name();
        }

        void onPreAllocMLInsight(size_t alignment,size_t num_bytes){
            INFO_LOGS("onPreAllocMLInsight alignment=%zd num_bytes=%zd pthread_self=%p",alignment, num_bytes,pthread_self());
            globalExecutionState.onPreAlloc(num_bytes); //This should just be invoked before all other analyzer classes that use globalExecutionState
            memLeakAnalyzer.onPreAllocFramework(num_bytes);
//            flameGraphAnalyser.onPreAlloc(num_bytes);

        }

        void onPostAllocMLInsight(size_t alignment, size_t num_bytes,void* ptr){
            FramekworkTensorType* newTensor = mapFrameworkAliveObjs.insert(num_bytes,ptr);
            newTensor->updateCallstack();

            INFO_LOGS("onPostAllocMLInsight alignment=%zd num_bytes=%zd returnPtr=%p pthread_self=%p  MLInsightTensor:%p", alignment, num_bytes, ptr, pthread_self(), newTensor);
            memLeakAnalyzer.onPostAllocFramework(num_bytes, ptr, newTensor);
//            flameGraphAnalyser.onPostAlloc(num_bytes,ptr,newTensor);
            globalExecutionState.onPostAlloc(num_bytes,ptr,newTensor);
        }

        FramekworkTensorType* onPreFreeMLInsight(void* ptr){
            DBG_MEMORY_RACE_CONDITION_DETECTOR_LOCK
            INFO_LOGS("onPreFreeMLInsight ptr=%p pthread_self=%p",ptr,pthread_self());

            FramekworkTensorType* justRemovedTensor = mapFrameworkAliveObjs.remove(ptr);
            globalExecutionState.onPreFree(ptr,justRemovedTensor); //This must be the first to call
            memLeakAnalyzer.onPreFreeFramework(ptr, justRemovedTensor);
//            flameGraphAnalyser.onPreFree(ptr,justRemovedTensor);
            return justRemovedTensor;
        }

        void onPostFreeMLInsight(void* ptr, FramekworkTensorType* justRemovedTensor){
            INFO_LOGS("onPostFreeMLInsight ptr=%p pthread_self=%p MLInsightTensor:%p",ptr,pthread_self(),justRemovedTensor);

            memLeakAnalyzer.onPostFreeFramework(ptr,justRemovedTensor);
//            flameGraphAnalyser.onPostFree(ptr,justRemovedTensor);
            globalExecutionState.onPostFree(ptr,justRemovedTensor); //This must be last one to call
            mapFrameworkAliveObjs.erase(justRemovedTensor);
        }


        virtual void* AllocateRaw(size_t alignment, size_t num_bytes){
            DBG_MEMORY_RACE_CONDITION_DETECTOR_LOCK
            this->onPreAllocMLInsight(alignment,num_bytes);

            void* retVal=getRealAllocatorPointer(this)->AllocateRaw(alignment,num_bytes);

            this->onPostAllocMLInsight(alignment,num_bytes,retVal);

            cudaPointerAttributes attributes;
            cudaPointerGetAttributes(&attributes,retVal);

            const char* memTypeArray[]={"cudaMemoryTypeUnregistered",
                                      "cudaMemoryTypeHost",
                                      "cudaMemoryTypeDevice",
                                      "cudaMemoryTypeManaged"
            };
            INFO_LOGS("onPostAllocMLInsight alignment=%zd num_bytes=%zd returnPtr=%p pthread_self=%p Type:%s", alignment, num_bytes, retVal, pthread_self(), memTypeArray[attributes.type]);
            DBG_MEMORY_RACE_CONDITION_DETECTOR_UNLOCK
            return retVal;
        }

        virtual void* AllocateRaw(size_t alignment, size_t num_bytes,
                                  const ::tensorflow::AllocationAttributes& allocation_attr) {
            DBG_MEMORY_RACE_CONDITION_DETECTOR_LOCK
            this->onPreAllocMLInsight(alignment,num_bytes);
            // The default behavior is to use the implementation without any allocation
            // attributes.
            void* retVal = getRealAllocatorPointer(this)->AllocateRaw(alignment,num_bytes,allocation_attr);
            this->onPostAllocMLInsight(alignment,num_bytes,retVal);
            const char* memTypeArray[]={"cudaMemoryTypeUnregistered",
                                        "cudaMemoryTypeHost",
                                        "cudaMemoryTypeDevice",
                                        "cudaMemoryTypeManaged"
            };
            cudaPointerAttributes attributes;
            cudaPointerGetAttributes(&attributes,retVal);
            INFO_LOGS("onPostAllocMLInsight alignment=%zd num_bytes=%zd returnPtr=%p pthread_self=%p Type:%s", alignment, num_bytes, retVal, pthread_self(),memTypeArray[attributes.type]);
            DBG_MEMORY_RACE_CONDITION_DETECTOR_UNLOCK
            return retVal;
        }

        virtual void DeallocateRaw(void* ptr) {
            DBG_MEMORY_RACE_CONDITION_DETECTOR_LOCK
            FramekworkTensorType* justRemovedTensor=this->onPreFreeMLInsight(ptr);

            getRealAllocatorPointer(this)->DeallocateRaw(ptr);

            this->onPostFreeMLInsight(ptr,justRemovedTensor);
            DBG_MEMORY_RACE_CONDITION_DETECTOR_UNLOCK
        }

        virtual bool TracksAllocationSizes()  {
            return getRealAllocatorPointer(this)->TracksAllocationSizes();
        }

        virtual bool AllocatesOpaqueHandle()  {
            return getRealAllocatorPointer(this)->AllocatesOpaqueHandle();

        }

        virtual size_t RequestedSize(const void* ptr)  {
            return getRealAllocatorPointer(this)->RequestedSize(ptr);
        }

        virtual size_t AllocatedSize(const void* ptr)  {
            return getRealAllocatorPointer(this)->AllocatedSize(ptr);
        }

        virtual ::tensorflow::int64 AllocationId(const void* ptr)  {
            return getRealAllocatorPointer(this)->AllocationId(ptr);
        }

        virtual size_t AllocatedSizeSlow(const void* ptr)  {
            return getRealAllocatorPointer(this)->AllocatedSizeSlow(ptr);
        }

        virtual absl::optional<::tensorflow::AllocatorStats> GetStats() {
            return getRealAllocatorPointer(this)->GetStats();
        }

        virtual void ClearStats() {
            getRealAllocatorPointer(this)->ClearStats();
        }

        virtual void SetSafeFrontier(::tensorflow::uint64 count) {
            getRealAllocatorPointer(this)->SetSafeFrontier(count);
        }

    };
    static TFAllocatorProxy* myObj=nullptr;


    class TFGPUProcessStateProxy{
    public:

        static TFGPUProcessStateProxy *singleton_proxy(::tensorflow::GPUProcessState *ps = nullptr) {
            INFO_LOG("singleton_proxy is invoked");
            if(!myGPUProcStatePtr){
                realProcessState=realSingletonAddr(ps);
                myGPUProcStatePtr=new TFGPUProcessStateProxy();
            }
            return myGPUProcStatePtr;
        }
        // Query whether any GPU device has been created so far.
        // Disable thread safety analysis since a race is benign here.
        bool HasGPUDevice() const TF_NO_THREAD_SAFETY_ANALYSIS {
            return realProcessState->HasGPUDevice();
        }

        // Set the flag to indicate a GPU device has been created.
        // Disable thread safety analysis since a race is benign here.
        void EnableGPUDevice() TF_NO_THREAD_SAFETY_ANALYSIS {
            return realProcessState->EnableGPUDevice();
        }

        // Returns the one GPU allocator used for the indexed GPU.
        // Note that this is a system GPU index, not (necessarily) a brain
        // device index.
        //
        // 'total_bytes' is the total number of bytes that should be made
        // available to the allocator.  The first call to this function for
        // a given tf_gpu_id creates the allocator, so only the total_bytes
        // used on that first call is used.
        //
        // "Allocator type" describes the type of algorithm to use for the
        // underlying allocator.  REQUIRES: Must be a valid type (see
        // config.proto for the list of supported strings.).
        //
        // REQUIRES: tf_gpu_id must be a valid id for a BaseGPUDevice available in the
        // current system environment.  Otherwise returns nullptr.

        virtual ::tensorflow::Allocator* GetGPUAllocator(const ::tensorflow::GPUOptions& options,
                                                         ::tensorflow::TfDeviceId tf_gpu_id, size_t total_bytes,const std::vector<::tensorflow::TfDeviceId>& peer_gpu_ids){

            ::tensorflow::Allocator* realGpuAllocatorAddr = realProcessState->GetGPUAllocator(options, tf_gpu_id, total_bytes,peer_gpu_ids);

            ::tensorflow::Allocator* retValue=nullptr;
            auto iter=realAllocatorIdMap.find(realGpuAllocatorAddr);
            if(iter == realAllocatorIdMap.end()){
                //TODO: merge find and insertion in to one step
                realAllocatorIdMap[realGpuAllocatorAddr]=tfAllocatorArray.size();
                uint8_t* ptr = (uint8_t*) malloc(sizeof(::tensorflow::Allocator*)+ sizeof(TFAllocatorProxy));

                auto** tmp=(::tensorflow::Allocator** )ptr;
                *tmp=realGpuAllocatorAddr;
                auto* newMyAllocatorPtr = reinterpret_cast<TFAllocatorProxy *>(ptr + sizeof(::tensorflow::Allocator *));
                new (newMyAllocatorPtr) TFAllocatorProxy();

                tfAllocatorArray.emplace_back(newMyAllocatorPtr);
                INFO_LOGS("realGpuAllocatorAddr=%p linked with %p, address stored in %p Allocator type:%s", realGpuAllocatorAddr, newMyAllocatorPtr, newMyAllocatorPtr, options.allocator_type().c_str());

                return (::tensorflow::Allocator*) newMyAllocatorPtr;
            }else{
                return (::tensorflow::Allocator*) tfAllocatorArray[iter->second];
            }
        }

        virtual ::tensorflow::Allocator* GetGpuHostAllocator(int numa_node){
            return realProcessState->GetGpuHostAllocator(numa_node);

        }

        // Registers a Visitor to be invoked on new chunks of memory allocated by the
        // SubAllocator of every GPU proximate to the specified bus.  The AllocVisitor
        // is provided with a memory pointer, a GPU id, and the size of the area it
        // identifies.  The pointer is not guaranteed to be valid after the call
        // terminates.  The intention is for this interface to be used for network
        // device memory registration.  "bus_id" is platform-specific.  On many
        // platforms it should be 0.  On machines with multiple PCIe buses, it should
        // be the index of one of the PCIe buses (maybe the NUMA node at which the
        // PCIe is rooted).  If the bus_id is invalid, results are undefined.
        virtual void AddGPUAllocVisitor(int bus_id,
                                        const ::tensorflow::SubAllocator::Visitor& visitor){
            return realProcessState->AddGPUAllocVisitor(bus_id,visitor);
        }

        // Registers a Visitor to be invoked on new chunks of memory allocated by
        // the SubAllocator of the GpuHostAllocator for the given numa_node.
        virtual void AddGpuHostAllocVisitor(int numa_node,
                                            const ::tensorflow::SubAllocator::Visitor& visitor){
            return realProcessState->AddGpuHostAllocVisitor(numa_node,visitor);
        }

        // Registers a Visitor to be invoked on each chunk handed back for freeing to
        // the SubAllocator of the GpuHostAllocator for the given numa_node.
        virtual void AddGpuHostFreeVisitor(int numa_node,
                                           const ::tensorflow::SubAllocator::Visitor& visitor){
            return realProcessState->AddGpuHostFreeVisitor(numa_node,visitor);
        }

        // Returns bus_id for the given GPU id.
        virtual int BusIdForGPU(::tensorflow::TfDeviceId tf_gpu_id){
            return realProcessState->BusIdForGPU(tf_gpu_id);
        }

        ::tensorflow::SharedCounter* GPUAllocatorCounter(::tensorflow::TfDeviceId tf_gpu_id){
            return realProcessState->GPUAllocatorCounter(tf_gpu_id);
        }


    };

    namespace tensorflow{


        void onSettingHookHint(std::map<std::string, SymbolHookHint>& hookHintMap) {
            ProxySymbol proxySymbol[] = {
                    {"_ZN10tensorflow15GPUProcessState9singletonEPS0_", (void*) TFGPUProcessStateProxy::singleton_proxy, (void **) &realSingletonAddr}
            };

            const ssize_t proxySymbolArrSize = sizeof(proxySymbol) / sizeof(proxySymbol[0]);
            for (int i = 0; i < proxySymbolArrSize; ++i) {
                hookHintMap.insert(std::make_pair(proxySymbol[i].name, SymbolHookHint(proxySymbol[i].address,
                                                                                      proxySymbol[i].realAddressPtr)));
            }

        }


        void onHookInstallationFinished(){
            //assert(realSingletonAddr!=nullptr);
        }
    }

}