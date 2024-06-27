#ifndef MLINSIGHT_PYHOOK_H
#define MLINSIGHT_PYHOOK_H

#include <Python.h>
#include <frameobject.h>
#include <ceval.h>
#include "trace/type/PyCodeExtra.h"
#include "analyse/CallBackInterface.h"
#include <regex>

namespace mlinsight {
    extern bool isPyInterpreterInstalled;
    extern PyObject *mlInsightPythonModule;
    extern PyInterpreterState *pythonInterpreterState;
    extern bool pyTorchHookInstalled;
    extern PyObject * pythonNoneObj; //Do not use Py_None to prevent linker error
    extern PyObject *mlInsightPythonModule;
    extern PyObject *forward_pre_hook_ptr;
    extern PyObject *forward_hook_ptr;
    extern PyObject *full_backward_pre_hook_ptr;
    extern PyObject *full_backward_hook_ptr;
    extern PyObject *module_registration_hook_ptr;
    extern PyObject *parameter_registration_hook_ptr;
    extern bool hasMainFunctionStarted; 

    inline PyObject* returnPyNone(){
        Py_IncRef(pythonNoneObj);
        return pythonNoneObj;
    }
    struct RegexAndFriendlyName {
        std::regex pkgNameRegex;
        std::string friendlyName;

        RegexAndFriendlyName() = default;

        RegexAndFriendlyName(std::string pkgNameRegex, std::string friendlyName) : pkgNameRegex(pkgNameRegex),
                                                                                   friendlyName(friendlyName) {
        }
    };


    bool installAfterPythonInit();

    /**
     * This function must be called before the Python Interpreter is loaded
     */
    void installBeforePythonInit();


    PythonFrameExtra_t *getPyCodeExtra(PyFrameObject *f);


    RecTuple &allocateTimeRecordingTuple(PythonFrameExtra_t *prevCodeExtra, PythonFrameExtra_t *curCodeExtra);


    enum class PyTorchModuleState : char {
        UNSPECIFIED_PYTORCH_MODULE_STATE = 0,
        FORWARD_STATE = 1,
        BACKWARD_STATE = 2
    };

    struct ExecutionState {
        FileID pyTorchModuleId = -1;
        PyTorchModuleState pyTorchModuleExecutionState = PyTorchModuleState::UNSPECIFIED_PYTORCH_MODULE_STATE;

        inline ExecutionState() = default;

        inline ExecutionState(FileID pyTorchModuleId, PyTorchModuleState pyTorchModuleExecutionState) : pyTorchModuleId(
                pyTorchModuleId), pyTorchModuleExecutionState(pyTorchModuleExecutionState) {

        }
    };

    static const char *toString(const PyTorchModuleState &execState) {
        switch (execState) {
            case PyTorchModuleState::UNSPECIFIED_PYTORCH_MODULE_STATE:
                return "Unspecified";
            case PyTorchModuleState::FORWARD_STATE:
                return "Forward";
            case PyTorchModuleState::BACKWARD_STATE:
                return "Backward";
        }
        return "Error";
    }

    typedef ssize_t CORRELATION_ID;
    const ssize_t UNSPECIFIED_CORRELATION_ID=-1;
    /**
     * A class that stores the current state of the python interpreter.
     * The member variables can be read by Python invoked libraries to check the current module .etc.
     * This class is thread safe
     */
    template<typename FRAMEWORK_TENSOR_TYPE>
    class PythonExecutionState:public SimpleCallback<FRAMEWORK_TENSOR_TYPE> {
    public:
        ssize_t pyModuleId = -1; //The currently executed fileId of the python module. If this value is -1, it means that there is no python function executing. Beware that this value may be -1.
        std::vector<ExecutionState> pyTorchModuleStack;

        /**
         * This variable is set to false before framework memory allocation. And is true after driver memory allocation
         * is invoked.
         *
         * In this way AllocationStatus class can check this flag to check whether the newly allocated object is newly
         * allocated by driver or is it a cached object.
         */
        ssize_t isInvokingFrameworkMemOp=0; //Indicates whether this memory operation (allocation/deallocation) comes from framework or not.
        CORRELATION_ID allocationCorrelationId=UNSPECIFIED_CORRELATION_ID; // Used to correlate driver and framework allocation
        CORRELATION_ID freeCorrelationId=UNSPECIFIED_CORRELATION_ID; // Used to correlate driver and framework free
        /**
        * [Interface]
        * This must be called BEFORE all metrics that uses objects of class
        */
        void onPreAlloc(ssize_t size){
            assert(isInvokingFrameworkMemOp>=0);
            isInvokingFrameworkMemOp+=1;
            allocationCorrelationId+=1;
        }

        /**
        * [Interface]
        * This must be called AFTER all metrics that uses objects of class
        */
        void onPostAlloc(ssize_t size, void *ptr,FRAMEWORK_TENSOR_TYPE* newTensor){
            isInvokingFrameworkMemOp-=1; 
            assert(isInvokingFrameworkMemOp>=0);
        }

        /**
        * [Interface]
        */
        void onPreFree(void *ptr,FRAMEWORK_TENSOR_TYPE* justFreedTensor){
            assert(isInvokingFrameworkMemOp>=0);
            isInvokingFrameworkMemOp+=1;
            freeCorrelationId+=1;
        }

        /**
        * [Interface]
        */
        void onPostFree(void* ptr,FRAMEWORK_TENSOR_TYPE* justFreedTensor){
            isInvokingFrameworkMemOp-=1;
            assert(isInvokingFrameworkMemOp>=0);
        }

    };


}

#endif