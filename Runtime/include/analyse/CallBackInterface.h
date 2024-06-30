
#ifndef MLINSIGHT_CALLBACKINTERFACE_H
#define MLINSIGHT_CALLBACKINTERFACE_H
#include <cstdio>

namespace mlinsight{

enum class MLExecutionState : char {
    UNSPECIFIED_ML_EXECUTION_STATE = 0,
    FORWARD_STATE = 1,
    BACKWARD_STATE = 2
};

struct MLExecutionStackFrame {
    ssize_t layerId = -1;
    MLExecutionState mlExecutionState = MLExecutionState::UNSPECIFIED_ML_EXECUTION_STATE;
    void* modulePointer=nullptr; //Used to reduce unordered map operation

    inline MLExecutionStackFrame() = default;

    inline MLExecutionStackFrame(ssize_t pyTorchModuleId, MLExecutionState pyTorchModuleExecutionState, void* modulePointer) : layerId(
            pyTorchModuleId), mlExecutionState(pyTorchModuleExecutionState), modulePointer(modulePointer) {

    }
};

static const char *toString(const MLExecutionState &execState) {
    switch (execState) {
        case MLExecutionState::UNSPECIFIED_ML_EXECUTION_STATE:
            return "Unspecified";
        case MLExecutionState::FORWARD_STATE:
            return "Forward";
        case MLExecutionState::BACKWARD_STATE:
            return "Backward";
    }
    return "Error";
}

/**
 * Callbacks for classes that only handles allocation from one allocator.
 * To prevent performance overhead, we do not allow the use of virtual function. That is why user should never upcast to this base class.
 * The purpose of this class is to provide a "mental" interface for subclasses developers. In practice, subclass have the right to modify these interfaces (eg: Return different values) based on requirements.
 * Besides, this class make it possible for subclasses to only implement methods they care, but the proxy functions can still call all types of these callbacks regardless of subclass type.
 * Th methods are ordered by invocation order when performing memory allocation and frees.
 */
template<typename CTENSOR_TYPE>
class SimpleCallback{
public:
    /**
    * [Interface]
    */
    void onPreAlloc(ssize_t size){}
    /**
    * [Interface]
    */
    void onPostAlloc(ssize_t size, void *ptr,CTENSOR_TYPE* newTensor){}
    /**
    * [Interface]
    */
    void onPreFree(void *ptr,CTENSOR_TYPE* justFreedTensor){}
    /**
    * [Interface]
    */
    void onPostFree(void* ptr,CTENSOR_TYPE* justFreedTensor){}

protected:
    SimpleCallback(){}
};

/**
 * Callbacks for classes that only handles allocation from one allocator.
 * To prevent performance overhead, we do not allow the use of virtual function. That is why subclass should never upcast to the base class.
 * The purpose of this class is to provide a "mental" interface for subclasses developers.
 * Besides, this class make it possible for subclasses to only implement methods they care, but the proxy functions can still call all types of these callbacks regardless of subclass type.
 * Th methods in this class are ordered by invocation order when performing memory allocation and frees.
 */
template<typename DRIVER_TENSOR_TYPE, typename FRAMEWORK_TENSOR_TYPE>
class CompleteCallback{
public:
    /**
    * [Interface]
    */
    void onPreAllocFramework(ssize_t size){}

    /**
    * [Interface]
    */
    void onPreAllocDriver(ssize_t size) {}

    /**
    * [Interface]
    * ptr may be null
    */
    void onPostAllocDriver(ssize_t size, void *ptr, DRIVER_TENSOR_TYPE* newTensor){}

    /**
    * [Interface]
    * ptr may be null
    */
    void onPostAllocFramework(ssize_t size, void *ptr, FRAMEWORK_TENSOR_TYPE* newTensor) {}

    /**
    * [Interface]
    * ptr may be null
    */
    void onPreFreeFramework(void *ptr, FRAMEWORK_TENSOR_TYPE* justFreedTensor) {}

    /**
    * [Interface]
    * ptr may be null
    */
    void onPreFreeDriver(void *ptr, DRIVER_TENSOR_TYPE* justFreedTensor) {}

    /**
    * [Interface]
    * ptr may be null
    */
    void onPostFreeDriver(void *ptr,DRIVER_TENSOR_TYPE* justFreedTensor) {}

    /**
    * [Interface]
    * ptr may be null
    */
    void onPostFreeFramework(void *ptr,FRAMEWORK_TENSOR_TYPE* justFreedTensor) {}

    /**
     * [Interface]
     * Called before throwing Pytorch out of memory error
     * @param size 
     */
    void onOutOfMemoryFramework(ssize_t size) {}

    void onPreLayerForward(ssize_t layerId, MLExecutionStackFrame &curExecState) {}

    void onPostLayerForward(ssize_t layerId, MLExecutionStackFrame &curExecState) {}

    void onPreLayerBackward(ssize_t layerId, MLExecutionStackFrame &curExecState) {}

    void onPostLayerBackward(ssize_t layerId, MLExecutionStackFrame &curExecState) {}


protected:
    CompleteCallback(){}
};

}
#endif //MLINSIGHT_CALLBACKINTERFACE_H
