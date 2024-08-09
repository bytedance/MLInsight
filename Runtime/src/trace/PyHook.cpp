/*
@author: Steven Tang <steven.tang@bytedance.com>
@author: Tongping Liu <tongping.liu@bytedance.com>
*/
#include <cstdio>
#include <Python.h>
#include <patchlevel.h>
#include <frameobject.h>
#include <ceval.h>
#include <map>
#include <regex>
#include "analyse/GlobalVariables.h"
//#include <pybind11/pybind11.h>

#include "common/MemoryHeap.h"
#include "trace/hook/HookInstaller.h"
#include "common/Tool.h"
#include "trace/type/RecordingDataStructure.h"
#include "trace/type/PyCodeExtra.h"
#include "trace/hook/HookContext.h"
#include "trace/hook/PyHook.h"
#include "analyse/LogicalClock.h"
#include "analyse/PieChartAnalyzer.h"
#include "trace/proxy/PythonModuleDef.h"
#include <Python.h>


#define PY_VERSION_36 ((3 << 24) | (6 << 16))
#define PY_VERSION_37 ((3 << 24) | (7 << 16))
#define PY_VERSION_38 ((3 << 24) | (8 << 16))
#define PY_VERSION_39 ((3 << 24) | (9 << 16))
#define PY_VERSION_3916 ((3 << 24) | (9 << 16)) | (16 << 8)

namespace mlinsight {


    extern __thread HookContext *curContext;
    bool bInit = false;
    bool isPyInterpreterInstalled = false;
    PyObject *mlInsightPythonModule = nullptr;
    PyObject *forward_pre_hook_ptr = nullptr;
    PyObject *forward_hook_ptr = nullptr;
    PyObject *full_backward_pre_hook_ptr = nullptr;
    PyObject *full_backward_hook_ptr = nullptr;
    PyObject *module_registration_hook_ptr = nullptr;
    PyObject *parameter_registration_hook_ptr = nullptr;
    bool hasMainFunctionStarted=false;
    bool pyTorchHookInstalled = false;
    PyInterpreterState *pythonInterpreterState = nullptr;
    PyObject * pythonNoneObj=nullptr;
//    PythonExecutionState globalExecutionState;
    PyObject * (*realEvalFrameDefaultPtr)(PyThreadState *tstate, PyFrameObject *f, int exc);
    ssize_t pyFrameReplacementCounter=0;

    std::vector<RegexAndFriendlyName> pyModuleSummaryAttributionArray;

    /*
     * Performance analyzer required fields
     */
    int PyCodeExtra_Index = -1; //Should not conflict with other frameworks that uses this value
    const int PyPackage_Level = 5; //How many pyCallStackLevel of packages should MLInsight treat as packages. 1 means if hello.a and hello.b will be treated as the same package hello
    typedef uint64_t (*PyPreHookHandler)(HookContext *curContextPtr);

    typedef void (*PyPostHookHandler)(uint64_t preHookTimestamp, HookContext *curContextPtr, RecTuple &curRecTuple);

    ObjectPoolHeap<PythonFrameExtra_t>* pyCodeExtraHeap=new ObjectPoolHeap<PythonFrameExtra_t>(); //todo: This object itself is a driverMemRecord leak.

    //HookContext *curContextPtr = NULL;
/*
 * Memory analyzer required fields
 */

/*The following functions are only defined if python interpreter is 3.9*/
    PyPreHookHandler preHookHandlerPtr = nullptr;
    PyPostHookHandler postHookHandlerPtr = nullptr;


    /**
    * Get the full python package name for the current Python frame.
    */
    std::string getPyModuleName(PyFrameObject *f) {// Get the global namespace dictionary
        PyObject *
        globals = f->f_globals; //Reference https://github.com/python/cpython/blob/c688c0f130906ff7725a126fff143d1389884f89/Python/ceval.c#L2497

        /**
         * Parse the module name of this symbol and correlate with the timing recording structure.
         */

        // Get the value of the "__name__" variable from the global namespace
        PyObject * nameObj = PyDict_GetItemString(globals, "__name__");

        std::string pyNameVar("<empty>");
        //printf("getPyCodeExtra 11\n");
        if (nameObj != NULL) {
            const char *pyNameCStr = PyUnicode_AsUTF8(nameObj);
            if (pyNameCStr) {
                pyNameVar = std::string(pyNameCStr);
                //INFO_LOGS("Python reports that %s is the package name", pyNameCStr);
            }
        }

        //Get the Python module name at the specified package pyCallStackLevel
        ssize_t subStrEndLoc = 0;
        for (int i = 0; i < PyPackage_Level; ++i) {
            subStrEndLoc = pyNameVar.find('.', subStrEndLoc + 1);
            if (subStrEndLoc == -1) {
                //Use the entire string
                subStrEndLoc = pyNameVar.length();
                break;
            }
            subStrEndLoc = -1;
        }
        std::string pyModuleName = pyNameVar.substr(0, subStrEndLoc) + " [Py]";

        return std::move(pyModuleName);
    }


    /**
     * This function allocates a callerFileId for newly allocated PyCodeExtra objects.
     * Beware that this function only works for the current frame. So argument curFrameCodeExtra must belong to the current frame.
     * This function will also update the package level field in pycodeExtra
     * @param curFrameCodeExtra The PyCodeExtra object for the current Python frame.
     */
    template<typename FILE_INFO_TYPE, typename... FILE_INFO_ARGS>
    FileID getFileIdPyModule(const std::string &strToBeStored,
                             Array<FILE_INFO_TYPE> &fileInfos,
                             std::unordered_map<std::string, FileID> &reverseIdMap,
                             FILE *fileStrTbl,
                             FILE_INFO_ARGS... fileInfoConstructionArgs) {
//        string pyModuleName = getPyModuleName(f);
        //No lock here because python process is protected by the GIL.
        auto fileIdIter = reverseIdMap.find(strToBeStored);

        ssize_t newGlobalFileId = -1;
        //Check if
        if (fileIdIter == reverseIdMap.end()) {
            pthread_mutex_lock(&hookInstallerInstance->dynamicLoadingLock);

            //Write filename to the string table
            fprintf(fileStrTbl, "%s\n", strToBeStored.c_str());
            //This file has not been hooked before. Allocate a new fileID.
            //Need lock protection because may these variables may be accessed by non-python processes.
            newGlobalFileId = fileInfos.getSize();

            fileInfos.pushBack(fileInfoConstructionArgs...);

            reverseIdMap[strToBeStored] = newGlobalFileId;

            pthread_mutex_unlock(&hookInstallerInstance->dynamicLoadingLock);
            //INFO_LOGS("Allocated a new callerFileId %d for module %s",newGlobalFileId,pyModuleName.c_str());
        } else {
            //Found fileId before.
            //This is possible because different API may belong to the same python Module, source file.
            newGlobalFileId = fileIdIter->second;
        }

        //Record new recId in code extra
//        curFrameCodeExtra->calleeFileId = newGlobalFileId;
//        curFrameCodeExtra->globalPyModuleId=curFrameCodeExtra->calleeFileId;
        return newGlobalFileId;
    }


    /**
     * Retrieve an existing or allocating a new recording entry and APICallInfo entry in the per-thread recording data structure.
     * @param prevCodeExtra PyCodeExtra object for the previous frame. (Caller)
     * @param curCodeExtra PyCodeExtra object for the current frame. (Callee)
     * @param curFrame The current frame object
     * @return the newly allocated symbolId
     */
    RecTuple &allocateTimeRecordingTuple(PythonFrameExtra_t *prevCodeExtra, PythonFrameExtra_t *curCodeExtra) {
        //MLInsight records the recording Id for each API in the following way: calleePyCodeExtra[callerPyCodeExtra.calleeFileId] = The current API's recording location in the per-thread recording array.
        //This is necessary because the fact that different modules invoking  the same API should be recorded in different entries in the recording array.
        ssize_t originalSize = curCodeExtra->pyModuleRecArrMap.getSize();
        //The caller may not necessarily be the one with the smallest calleeFileId, so we need to insert -1 to ensure pyModuleRecArrMap[calleeFileId] is valid.
        for (ssize_t i = originalSize - 1; i <= prevCodeExtra->globalPyModuleId; ++i) {
            curCodeExtra->pyModuleRecArrMap.pushBack(-1);
            //Indicate this invocation relation has no allocated entry in recording array yet.
        }
        assert(prevCodeExtra != nullptr);
        ssize_t &curRecId = curCodeExtra->pyModuleRecArrMap[prevCodeExtra->globalPyModuleId];
        //INFO_LOGS("Existing record Id is: %zd",curRecId);

        HookContext *curContextPtr = curContext;
        assert(curContextPtr != nullptr);
        //THere is no need to acquire lock here because we are manipulating per-thread context.
        if (curRecId == -1) {
            //Allocate new recording entry and record the ID.
            //Allocate a new recEntry in recTuple
            //Be aware that python functions are not exactly the same as C/C++ APIs. In Python, functionInfo in Python actually denotes an invocation relationship.
            pthread_mutex_lock(&hookInstallerInstance->dynamicLoadingLock);
            //Record symbol name to disk
            fprintf(hookInstallerInstance->nativeAPIInfoFile, "%s,%ld\n", curCodeExtra->pythonFunctionName.c_str(),
                    curCodeExtra->globalPyModuleId);

            curRecId = hookInstallerInstance->allExtSymbol.getSize();
            APICallInfo &newSym = hookInstallerInstance->allExtSymbol.pushBack();
            newSym.apiType = APIType::PY_API;
            newSym.callerFileId = prevCodeExtra->globalPyModuleId; //For .rela.plt and .rela.dyn APIs, callerFileId is known at the parsing time.
            newSym.calleeFileId = curCodeExtra->globalPyModuleId;
            newSym.symIdInFile = 0; //Python does not support this.
            newSym.initialGap = 0; //Always hook all python functions
            newSym.addressOverride = nullptr; //Currently python do not support address overrde.  shouldHookThisSymbol is also not called to save some time. For the special handling of Python functions, please refer to

            pthread_mutex_unlock(&hookInstallerInstance->dynamicLoadingLock);
        }

        //INFO_LOGS("Existing record Id is: %zd",curRecId);
        if (curRecId >= curContextPtr->recordArray.getSize()) {
            //This is possible because recording entries may be created by other threads and this thread has not yet synced these changes.
            curContextPtr->recordArray.allocateArray(curRecId + 1 - curContextPtr->recordArray.getSize());
        }

        return curContextPtr->recordArray[curRecId];

    }

    void registerPyTorchHook();

    //A marker that indicates whether MLInsight has found function _find_and_load_unlocked
    //This flag can help saving strcmp operations
    bool foundFindAndLoadUnlocked = false;

    PythonFrameExtra_t *getPyCodeExtra(PyFrameObject *f) {
        PythonFrameExtra_t *retPyCodeExtra;

        int pyCodeExtraRlt = _PyCode_GetExtra((PyObject *) f->f_code, PyCodeExtra_Index,
                                              (void **) &retPyCodeExtra);

        assert(pyCodeExtraRlt != -1);

        if (!retPyCodeExtra) {
            //PyCodeObject is null. Allocate and initialize this new function.
            retPyCodeExtra = pyCodeExtraHeap->alloc();
            new(retPyCodeExtra) PythonFrameExtra_t();

            //INFO_LOGS("Allocated pycode object at %p %zd %zd",curCodeExtra,curCodeExtra->calleeFileId,curCodeExtra->pyModuleRecArrMap.getSize());

            int setRlt = _PyCode_SetExtra((PyObject *) f->f_code, PyCodeExtra_Index, (void *) retPyCodeExtra);
            assert(setRlt == 0);

            /*
            * Record callstack related information
            */

            retPyCodeExtra->pythonSourceFileName = PyUnicode_AsUTF8(f->f_code->co_filename);
            retPyCodeExtra->pythonFunctionName = PyUnicode_AsUTF8(f->f_code->co_name);

            //Allocate pyModuleId
            std::string pyModuleName = getPyModuleName(f);
            PyModuleType pyModuleType = PyModuleType::USER_DONT_CARE;
            ssize_t pyModuleSummaryAttributionId = -1;
            for (ssize_t regexId = 0; regexId < pyModuleSummaryAttributionArray.size(); ++regexId) {
                if (regex_match(pyModuleName, pyModuleSummaryAttributionArray[regexId].pkgNameRegex)) {
                    pyModuleType = PyModuleType::USER_CARE;
                    pyModuleSummaryAttributionId = regexId;
                }
            }

            retPyCodeExtra->globalPyModuleId = getFileIdPyModule(pyModuleName,
                                                                 hookInstallerInstance->pyModuleInfoMap,
                                                                 hookInstallerInstance->pyModuleIdMap,
                                                                 hookInstallerInstance->pyModuleStrTbl,
                                                                 pyModuleName,
                                                                 pyModuleType,
                                                                 pyModuleSummaryAttributionId
            );

            retPyCodeExtra->pyFrameExtraID=callstack::python::globalPyCodeExtraIdCounter.fetch_add(1);

            assert(hookInstallerInstance->pyModuleInfoMap.getSize() > 0);
//The following code aggregates data by thread
//            const std::string& pySrcFileName = retPyCodeExtra->pythonSourceFileName;
//            retPyCodeExtra->globalPySrcFileId = getFileIdPyModule(pySrcFileName,
//                                                                  instance->pySrcFileInfoMap,
//                                                                  instance->pySrcFileIdReverseMap,
//                                                                  instance->pySrcFileStrTbl,
//                                                                  pySrcFileName
//            );
//            assert(instance->pySrcFileInfoMap.getSize()>0);


            if (!foundFindAndLoadUnlocked && retPyCodeExtra->pythonFunctionName == "_find_and_load_unlocked") {
                foundFindAndLoadUnlocked = true;
                retPyCodeExtra->functionalMarker = FUNCTIONAL_MARKER::PYTORCH_IMPORT_FINISHED_MARKER;
            }

        }

        return retPyCodeExtra;
    }


    inline void getCodeExtraIndex() {
        //printf("In entry of getCodeExtraIndex, PyCodeExtra_Index is %d\n", PyCodeExtra_Index);
        assert(PyCodeExtra_Index == -1);
        PyCodeExtra_Index = _PyEval_RequestCodeExtraIndex(freePyCodeExtra);
        //printf("In exit of getCodeExtraIndex, PyCodeExtra_Index is %d\n", PyCodeExtra_Index);
    }


    PyFrameObject *findRootFrame() {
        PyFrameObject *previousFrame = PyEval_GetFrame();

        while (previousFrame->f_back != nullptr) {
            previousFrame = previousFrame->f_back;
        }
        return previousFrame;
    }


    ssize_t getFrameNumber(PyFrameObject *f) {
        ssize_t fNumber = 1; //One frame has a depth value of 1. Two frame has depth of 2.
        PyFrameObject *previousFrame = f->f_back;
        while (previousFrame != nullptr) {
            ++fNumber;
            previousFrame = previousFrame->f_back;
        }
        return fNumber;
    }

    void initializeExistingPyFrames(HookContext *curContextPtr) {
        assert(curContextPtr!=nullptr);

        PyFrameObject *curFrame = PyEval_GetFrame();
        //Find the depth of

        //Install MLInsight on existing python frames
        //findRootFrame();
        ssize_t frameNumber = getFrameNumber(curFrame);
        //INFO_LOGS("Current frame number is %ld", frameNumber);

        if (frameNumber > MAX_CALL_DEPTH) {
            //Skip exceeded frames
            for (ssize_t i = 0; i < MAX_CALL_DEPTH - frameNumber; ++i) {
                curFrame = curFrame->f_back;
            }
            frameNumber = MAX_CALL_DEPTH;
        }

        for (ssize_t i = frameNumber - 1; i >= 0; --i) {
            //DBG_LOGS("Python frame at installation time %p", curFrame);
            //Initialize pycode object for this frame
            //DBG_LOGS("Initial install frame %zd", i);

            //todo: Handle logical clock correctly here.
            PythonFrameExtra_t *pyCodeExtraPtr = getPyCodeExtra(curFrame);

            curFrame = curFrame->f_back;
        }

        //Some initial Python interpreter frames may exist before the installation. These frames cannot be intercepted. We create a dummy module name for such modules.
        //We mark these modules as <main> to denote user programs.
        globalExecutionState.pyModuleId = getFileIdPyModule("<main> [Py]",
                                                            hookInstallerInstance->pyModuleInfoMap,
                                                            hookInstallerInstance->pyModuleIdMap,
                                                            hookInstallerInstance->pyModuleStrTbl,
                                                            "<main> [Py]",
                                                            PyModuleType::USER_CARE,
                                                            -1
        );

    }

    template<typename PYFRAME_T, typename... FRAME_EVAL_Args>
    extern PyObject *evalFrameFuncGeneral(PYFRAME_T curFrame, FRAME_EVAL_Args... args);

#if PY_VERSION_38 <= PY_VERSION_HEX && PY_VERSION_HEX < PY_VERSION_39
    //TODO the support is incomplete compared to Py39
    //#elif PY_VERSION_HEX > PY_VERSION_39
        template<typename PYFRAME_T,typename... FRAME_EVAL_Args>
        PyObject *executePythonFrame(PyFrameObject *f, int throwflag) {
            PyObject *ret = _PyEval_EvalFrameDefault(f, throwflag);
            return ret;
        }

        PyObject *evalFrameFunc(PyFrameObject *f, int throwflag) {
            return evalFrameFuncGeneral<PyFrameObject *, int>(f, throwflag);
        }

        inline bool installPyFrameInterceptor(PyPreHookHandler preHookHandler, PyPostHookHandler postHookHandler) {

            assert(preHookHandler!=nullptr && postHookHandler!=nullptr);
            PyInterpreterState *pyInterpreterStateMain = PyInterpreterState_Main();

#if 0
            typedef PyObject* (*_PyFrameEvalFunction)(PyThreadState *tstate, PyFrameObject *, int);
            _PyFrameEvalFunction prevEvalFrameFunc = pyInterpreterStateMain->eval_frame;
            if (prevEvalFrameFunc == evalFrameFunc) {
                return false;
            }

            pyInterpreterStateMain->eval_frame = evalFrameFunc;
#endif
            preHookHandlerPtr = preHookHandler;
            postHookHandlerPtr = postHookHandler;
            return true;
        }
#elif PY_VERSION_39 <= PY_VERSION_HEX

    template<typename PYFRAME_T, typename... FRAME_EVAL_Args>
    PyObject *executePythonFrame(PyFrameObject *f, PyThreadState *ts, int throwflag) {
        PyObject * ret = _PyEval_EvalFrameDefault(ts, f, throwflag);
        return ret;
    }

    PyObject *evalFrameFunc(PyThreadState * tstate, PyFrameObject * f, int
    throwflag) {

    return
    evalFrameFuncGeneral<PyFrameObject *, PyThreadState *, int>(f, tstate, throwflag
    );
}

bool installPyFrameInterceptor(PyPreHookHandler preHookHandler, PyPostHookHandler postHookHandler) {

    assert(preHookHandler != nullptr && postHookHandler != nullptr);
    // Retrieve the current frame evaluation function for the main interpreter state
    assert(pythonInterpreterState == nullptr);
    pythonInterpreterState = PyInterpreterState_Main();
    _PyFrameEvalFunction prevEvalFrameFunc = _PyInterpreterState_GetEvalFrameFunc(pythonInterpreterState);
    if (prevEvalFrameFunc == evalFrameFunc) {
        //Python frame interceptor has been installed before
        return false;
    }

    _PyInterpreterState_SetEvalFrameFunc(pythonInterpreterState, evalFrameFunc);
    preHookHandlerPtr = pyPreHookAttribution;
    postHookHandlerPtr = pyPostHookAttribution;
    return true;
}
#elif PY_VERSION_37 <= PY_VERSION_HEX && PY_VERSION_HEX < PY_VERSION_38
    //#elif PY_VERSION_HEX > PY_VERSION_39
    template<typename PYFRAME_T, typename... FRAME_EVAL_Args>
    PyObject* executePythonFrame(PyFrameObject *f, int throwflag) {
        PyObject *ret = _PyEval_EvalFrameDefault(f,throwflag);
        return ret;
    }

    PyObject* evalFrameFunc(PyFrameObject *f, int throwflag) {
        return evalFrameFuncGeneral<PyFrameObject*, int>(f,throwflag);
    }

    inline bool installPyFrameInterceptor(PyPreHookHandler preHookHandler, PyPostHookHandler postHookHandler) {
        assert(preHookHandler!=nullptr && postHookHandler!=nullptr);
        PyInterpreterState* pyInterpreterStateMain = PyInterpreterState_Main();

        typedef PyObject* (*_PyFrameEvalFunction)(struct _frame*, int);
        _PyFrameEvalFunction prevEvalFrameFunc = pyInterpreterStateMain->eval_frame;
        if(prevEvalFrameFunc == evalFrameFunc) {
            return false;
        }

        pyInterpreterStateMain->eval_frame = evalFrameFunc;

        preHookHandlerPtr = preHookHandler;
        postHookHandlerPtr = postHookHandler;
        return true;
    }
#else
#error MLInsight require a minimum version of python 3.6.
#endif

int pyCallStackLevel = 0; //Records the callstack level. This variable is mainly used for debug logging purposes.

template<typename PYFRAME_T, typename... FRAME_EVAL_Args>
PyObject *evalFrameFuncGeneral(PYFRAME_T curFrame, FRAME_EVAL_Args... args) {
    if (!curContext) {
        initTLS();
    }
    HookContext *curContextPtr = curContext;
    

    if (bypassCHooks == MLINSIGHT_TRUE) {
        //This flag instructs MLInsight to skip this frame. This is probably because MLInsight is invoking some Python APIs by itself.
        return executePythonFrame<PYFRAME_T, FRAME_EVAL_Args...>(curFrame, std::forward<FRAME_EVAL_Args>(args)...);
    }

    assert(curContextPtr != nullptr);

    if (bInit == false) {
        initializeExistingPyFrames(curContextPtr);
        bInit = true;
    }

    uint64_t preHookTimestamp = preHookHandlerPtr(curContextPtr);

    PyFrameObject *previousFrame = curFrame->f_back;
    pyCallStackLevel += 1;
    ssize_t codeLine = PyFrame_GetLineNumber(curFrame);
//        if(pyTorchHookInstalled){
//            INFO_LOGS("********Tracing[Callee](%d) started %s:%zd for function %s strlen:%zu\r\n",
//                      pyCallStackLevel,
//                      PyUnicode_AsUTF8(curFrame->f_code->co_filename),
//                      codeLine,
//                      PyUnicode_AsUTF8(curFrame->f_code->co_name),
//                      strlen(PyUnicode_AsUTF8(curFrame->f_code->co_name))
//            );
//        }
    PythonFrameExtra_t *curCodeExtra = nullptr;
    PythonFrameExtra_t *prevCodeExtra = nullptr;

    FileID previousPyFileId = globalExecutionState.pyModuleId;
    if (previousFrame) {
        //Only record if previous frame is available, otherwise there are strange errors in getPyModuleName function.
        //These two code extras must be obtained before function execution for attribution to work correctly.
        curCodeExtra = getPyCodeExtra(curFrame);
        //todo: If _PyCode_GetExtra is proved to be too slow, then we may reduce one _PyCode_GetExtra call by implementing a custom stack.
        prevCodeExtra = getPyCodeExtra(previousFrame);
        //Check if the current module is a new module or not. We can check this by requesting code extra.
        globalExecutionState.pyModuleId = curCodeExtra->globalPyModuleId;
    }
    PyObject * ret = executePythonFrame<PYFRAME_T, FRAME_EVAL_Args...>(curFrame,
                                                                       std::forward<FRAME_EVAL_Args>(args)...);

    _PyFrameEvalFunction prevEvalFrameFunc = _PyInterpreterState_GetEvalFrameFunc(pythonInterpreterState);
    if(prevEvalFrameFunc != evalFrameFunc) {
        ERR_LOGS("Pytorch dynamo was installed in process pid=%d. Currently, MLinsight does not perform Pytorch Dynamo interception. This will affect data attribution results may will not cause correctness problem",getpid());
        //todo: Install Pytorch dynamo interception here. Pytorch dynamo will not crash MLInsight.
    }

    globalExecutionState.pyModuleId = previousPyFileId;

    pyCallStackLevel -= 1;

    assert(curFrame != nullptr);

    //Record the time of this API invocation
    RecTuple *curRecTuple = nullptr;
    if (curCodeExtra && prevCodeExtra) {
        curRecTuple = &allocateTimeRecordingTuple(prevCodeExtra, curCodeExtra);
        postHookHandlerPtr(preHookTimestamp, curContextPtr, *curRecTuple);

        if (!pyTorchHookInstalled &&
            curCodeExtra->functionalMarker == FUNCTIONAL_MARKER::PYTORCH_IMPORT_FINISHED_MARKER) {
            //todo: When previousFrame is not NULL, reading module name from f->globals will cause error.
            //Try to install PyTorch
            //todo: There are issues in registerPyTorchHook when testing with ColossalAI
        registerPyTorchHook();
        }

    } else {
        //Shhould be impossible, but we still need to handle this
        //todo:Attribute time to application
        //printf("Detected exits");
    }

//        INFO_LOGS("********Tracing[Callee](%d) finished %s:%d for function %s\r\n",
//                 pyCallStackLevel,
//                 PyUnicode_AsUTF8(curFrame->f_code->co_filename),
//                 codeLine,
//                 PyUnicode_AsUTF8(curFrame->f_code->co_name)
//                 );
//
//        //According to cpython codebase, this function denotes a library import
//        if(strncmp(PyUnicode_AsUTF8(curFrame->f_code->co_name),"_find_and_load",14)==0) {
////            //todo: Should parse these arguments before function execution rather than other.
//            OUTPUTS("_find_and_loadName: %s PID:%d", PyUnicode_AsUTF8(curFrame->f_localsplus[0]),getpid());
//            OUTPUT("\n");
//            getchar();
//        }


    // if(curFrame->f_back){
    //     codeLine=PyFrame_GetLineNumber(curFrame);
    //     INFO_LOGS("********Tracing[Caller] %s:%d for function %s\r\n",
    //         PyUnicode_AsUTF8(curFrame->f_back->f_code->co_filename),
    //         codeLine,
    //         PyUnicode_AsUTF8(curFrame->f_back->f_code->co_name)
    //         );
    // }
    return ret;
}


/**
 * This function must be called after MLInsight is loaded
 */
void registerMLInsightAsAPythonModule();


void findPyNone(){
    if(!pythonNoneObj){
        pythonNoneObj=Py_BuildValue("");
        Py_IncRef(pythonNoneObj);
    }
}


bool installAfterPythonInit() {
    if (isPythonAvailable()) {
        INFO_LOG("Python is available");
        if (Py_IsInitialized() == 0) {
            INFO_LOG("*******Python interpreter not initialized*******");
            //Py_Initialize();
        }

        findPyNone();
        

        getCodeExtraIndex();

        //Register mlinsight as a python Module
        registerMLInsightAsAPythonModule();

        installPyFrameInterceptor(pyPreHookAttribution, pyPostHookAttribution);


        INFO_LOG("*******Installing python interpreter done*******");

        //Insert user-care package name regex for flame graph visualization
        pyModuleSummaryAttributionArray.emplace_back("cruise.*", "Cruise");
        pyModuleSummaryAttributionArray.emplace_back("deepspeed.*", "Deepspeed");
        pyModuleSummaryAttributionArray.emplace_back("torch.*", "PyTorch");
    } else {
        INFO_LOG("Python is not available");
    }
    return true;
}


PyObject *torchStrObj = nullptr;
PyObject *modulesStringObject = nullptr;

void registerPyTorchHook() {
    assert(!pyTorchHookInstalled);
    //if(!pyTorchHookInstalled) {
    //It is important to disable
    bypassCHooks = MLINSIGHT_TRUE;
    //Check if PyTorch is imported
    auto *modulesDict = PyImport_GetModuleDict();
    if (!PyDict_Contains(modulesDict, PyUnicode_FromString("torch"))) {
        INFO_LOG("PyTorch still no imported yet. Skip installation. ======================");
        bypassCHooks = MLINSIGHT_FALSE;
        return;
    }
    if (!torchStrObj) {
        torchStrObj = PyUnicode_FromString("torch.nn.modules.module");
    }

    
    auto *torchModuleObject = PyImport_GetModule(torchStrObj);
    if (torchModuleObject == NULL) {
        bypassCHooks = MLINSIGHT_FALSE;
        //INFO_LOG("Cannot import PyTorch. Skip installation. ======================");
        return;
    } else {
        pyTorchHookInstalled = true;

        auto* torchRootModuleobject =  PyImport_GetModule(PyUnicode_FromString("torch"));
        assert(torchRootModuleobject);

        auto *pyTorchRootModuleDict = PyModule_GetDict(torchRootModuleobject);
        assert(pyTorchRootModuleDict);
        auto *torchVersion = PyDict_GetItem(pyTorchRootModuleDict, PyUnicode_FromString(
                "__version__"));
        if(!torchVersion){
            fatalError("Failed to detect torch.__version__. Maybe this pytorch interface has changed?");
        }
        const char* pytorchRuntimeVersion = PyUnicode_AsUTF8(torchVersion);
        assert(pytorchRuntimeVersion);
        if(strcmp(pytorchRuntimeVersion,PYTORCH_VERSION_STR)!=0){
            fatalErrorS("The runtime version of pytorch %s does not match compile time pytorch %s\n",pytorchRuntimeVersion,PYTORCH_VERSION_STR);
        }
       
        auto *pyTorchNNModuleDict = PyModule_GetDict(torchModuleObject);
        assert(pyTorchNNModuleDict!=nullptr && pyTorchNNModuleDict!=NULL);

        auto *register_module_forward_pre_hook = PyDict_GetItem(pyTorchNNModuleDict, PyUnicode_FromString(
                "register_module_forward_pre_hook"));

        auto *register_module_forward_hook = PyDict_GetItem(pyTorchNNModuleDict,
                                                            PyUnicode_FromString("register_module_forward_hook"));
        auto *register_module_full_backward_pre_hook = PyDict_GetItem(pyTorchNNModuleDict, PyUnicode_FromString(
                "register_module_full_backward_pre_hook"));
        auto *register_module_full_backward_hook = PyDict_GetItem(pyTorchNNModuleDict, PyUnicode_FromString(
                "register_module_full_backward_hook"));
        auto *register_module_module_registration_hook = PyDict_GetItem(pyTorchNNModuleDict, PyUnicode_FromString(
                "register_module_module_registration_hook"));
        auto *register_module_parameter_registration_hook = PyDict_GetItem(pyTorchNNModuleDict, PyUnicode_FromString(
                "register_module_parameter_registration_hook"));

        if (register_module_forward_pre_hook == NULL ||
            register_module_forward_hook == NULL ||
            register_module_full_backward_pre_hook == NULL ||
            register_module_full_backward_hook == NULL ||
            register_module_module_registration_hook == NULL ||
            register_module_parameter_registration_hook == NULL) {
            pyTorchHookInstalled = false;
            bypassCHooks = MLINSIGHT_FALSE;
            return;
        }

        assert(forward_pre_hook_ptr != nullptr);
        auto *pyArgList1 = Py_BuildValue("(O)", forward_pre_hook_ptr);
        Py_IncRef((forward_pre_hook_ptr));
        Py_IncRef(pyArgList1);
        assert(register_module_forward_pre_hook != NULL);
        //PyObject_Print(register_module_forward_pre_hook, logFileStd, 0);
        //OUTPUT("\n");
        PyObject_CallObject(register_module_forward_pre_hook, pyArgList1);
//                Py_DecRef(pyArgList1);

        assert(forward_hook_ptr != nullptr);
        auto *pyArgList2 = Py_BuildValue("(O)", forward_hook_ptr);
        Py_IncRef((forward_hook_ptr));
        Py_IncRef(pyArgList2);

        
        assert(register_module_forward_hook != NULL);
        PyObject_CallObject(register_module_forward_hook, pyArgList2);

/*
        According to https://github.com/pytorch/pytorch/issues/61519, the following two interfaces will conflict with the Pytorch in-place operator.
        auto *pyArgList3 = Py_BuildValue("(O)", full_backward_pre_hook_ptr);
        Py_IncRef((full_backward_pre_hook_ptr));
        Py_IncRef(pyArgList3);
        PyObject_CallObject(register_module_full_backward_pre_hook, pyArgList3);


        auto *pyArgList4 = Py_BuildValue("(O)", full_backward_hook_ptr);
        Py_IncRef((full_backward_hook_ptr));
        Py_IWncRef(pyArgList4);
        PyObject_CallObject(register_module_full_backward_hook, pyArgList4);
*/

        auto *pyArgList5 = Py_BuildValue("(O)", module_registration_hook_ptr);
        Py_IncRef((module_registration_hook_ptr));
        Py_IncRef(pyArgList5);
        PyObject_CallObject(register_module_module_registration_hook, pyArgList5);


        auto *pyArgList6 = Py_BuildValue("(O)", parameter_registration_hook_ptr);
        Py_IncRef((parameter_registration_hook_ptr));
        Py_IncRef(pyArgList6);
        PyObject_CallObject(register_module_parameter_registration_hook, pyArgList6);
        INFO_LOG("PyTorch hook installation scucceded");
    }

    bypassCHooks = MLINSIGHT_FALSE;
    //}
}

void registerMLInsightAsAPythonModule() {
    mlInsightPythonModule = PyModule_Create(&mlInsightModuleDef);
    Py_IncRef(mlInsightPythonModule);
    auto *methodDict = PyModule_GetDict(mlInsightPythonModule);
    //Get function pointer from the module object
    forward_pre_hook_ptr = PyDict_GetItem(methodDict, PyUnicode_FromString("forward_pre_hook"));
    Py_IncRef(forward_pre_hook_ptr);
    forward_hook_ptr = PyDict_GetItem(methodDict, PyUnicode_FromString("forward_hook"));
    Py_IncRef(forward_hook_ptr);
    full_backward_pre_hook_ptr = PyDict_GetItem(methodDict, PyUnicode_FromString("full_backward_pre_hook"));
    Py_IncRef(full_backward_pre_hook_ptr);
    full_backward_hook_ptr = PyDict_GetItem(methodDict, PyUnicode_FromString("full_backward_hook"));
    Py_IncRef(full_backward_hook_ptr);
    module_registration_hook_ptr = PyDict_GetItem(methodDict, PyUnicode_FromString("module_registration_hook"));
    Py_IncRef(module_registration_hook_ptr);
    parameter_registration_hook_ptr = PyDict_GetItem(methodDict, PyUnicode_FromString("parameter_registration_hook"));
    Py_IncRef(parameter_registration_hook_ptr);
}




void installBeforePythonInit() {
    struct _inittab mods[] =
            {
                    {"mlinsightpyapi", []() -> PyObject * { return PyModule_Create(&mlInsightPyApiDef); }},
                    {nullptr,          nullptr}
            };
    if (PyImport_ExtendInittab(mods) < 0) {
        fatalError("Cannot register mlinsightapi")
    }
};

}