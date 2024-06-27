#include "analyse/DebugAnalyzer.h"
#include "common/TensorObj.h"
#include "analyse/GlobalVariables.h"
namespace mlinsight {

    template class DebugAnalyzer<DriverTensorType ,FramekworkTensorType>;

    template<typename DRIVER_CTENSOR_TYPE, typename FRAMEWORK_CTENSOR_TYPE>
    DebugAnalyzer<DRIVER_CTENSOR_TYPE, FRAMEWORK_CTENSOR_TYPE>::DebugAnalyzer() {

    }

    template<typename DRIVER_CTENSOR_TYPE, typename FRAMEWORK_CTENSOR_TYPE>
    void DebugAnalyzer<DRIVER_CTENSOR_TYPE, FRAMEWORK_CTENSOR_TYPE>::onPreAllocFramework(ssize_t size) {

    }

    template<typename DRIVER_CTENSOR_TYPE, typename FRAMEWORK_CTENSOR_TYPE>
    void DebugAnalyzer<DRIVER_CTENSOR_TYPE, FRAMEWORK_CTENSOR_TYPE>::onPreAllocDriver(ssize_t size) {
        // Detect whether the allocator is still our replaced one.
        assert(realPytorch2AllocatorPtr->load() == allocatorProxy);
    }

    template<typename DRIVER_CTENSOR_TYPE, typename FRAMEWORK_CTENSOR_TYPE>
    void DebugAnalyzer<DRIVER_CTENSOR_TYPE, FRAMEWORK_CTENSOR_TYPE>::onPostAllocDriver(ssize_t size, void *ptr,
                                                                                       DRIVER_CTENSOR_TYPE *newTensor) {
    }

    template<typename DRIVER_CTENSOR_TYPE, typename FRAMEWORK_CTENSOR_TYPE>
    void DebugAnalyzer<DRIVER_CTENSOR_TYPE, FRAMEWORK_CTENSOR_TYPE>::onPostAllocFramework(ssize_t size, void *ptr,
                                                                                          FRAMEWORK_CTENSOR_TYPE *newTensor) {
        //OUTPUTS("[trackPytorchAllocation] ptr %p size %zu\n", ptr, size);
    }

    template<typename DRIVER_CTENSOR_TYPE, typename FRAMEWORK_CTENSOR_TYPE>
    void DebugAnalyzer<DRIVER_CTENSOR_TYPE, FRAMEWORK_CTENSOR_TYPE>::onPreFreeFramework(void *ptr,
                                                                                        FRAMEWORK_CTENSOR_TYPE *justFreedTensor) {

    }

    template<typename DRIVER_CTENSOR_TYPE, typename FRAMEWORK_CTENSOR_TYPE>
    void DebugAnalyzer<DRIVER_CTENSOR_TYPE, FRAMEWORK_CTENSOR_TYPE>::onPreFreeDriver(void *ptr,
                                                                                     DRIVER_CTENSOR_TYPE *justFreedTensor) {
    }

    template<typename DRIVER_CTENSOR_TYPE, typename FRAMEWORK_CTENSOR_TYPE>
    void DebugAnalyzer<DRIVER_CTENSOR_TYPE, FRAMEWORK_CTENSOR_TYPE>::onPostFreeDriver(void *ptr,
                                                                                      DRIVER_CTENSOR_TYPE *justFreedTensor) {

    }

    template<typename DRIVER_CTENSOR_TYPE, typename FRAMEWORK_CTENSOR_TYPE>
    void DebugAnalyzer<DRIVER_CTENSOR_TYPE, FRAMEWORK_CTENSOR_TYPE>::onPostFreeFramework(void *ptr,
                                                                                         FRAMEWORK_CTENSOR_TYPE *justFreedTensor) {
        //OUTPUTS("[trackPytorchFree] ptr %p\n", ptr);
    }

}