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
//#include <pybind11/pybind11.h>

#include "common/MemoryHeap.h"
#include "trace/hook/HookInstaller.h"
#include "common/Tool.h"
#include "trace/type/RecordingDataStructure.h"
#include "trace/type/PyCodeExtra.h"
#include "trace/hook/HookContext.h"
#include "trace/hook/PyHook.h"
#include "analyse/LogicalClock.h"
#include "analyse/PieChartAttributor.h"

#define PY_VERSION_36 ((3 << 24) | (6 << 16))
#define PY_VERSION_37 ((3 << 24) | (7 << 16))
#define PY_VERSION_38 ((3 << 24) | (8 << 16))
#define PY_VERSION_39 ((3 << 24) | (9 << 16))
#define PY_VERSION_3916 ((3 << 24) | (9 << 16)) | (16 << 8)

#define SKIP_PYFRAMES 4

namespace mlinsight {
    extern __thread HookContext *curContext;
    int invokeNum = 0; 
    bool bInit = false; 
/*
 * Performance analyzer required fields
 */
    int PyCodeExtra_Index = -1; //Should not conflict with other frameworks that uses this value
    const int PyPackage_Level = 1; //How many level of packages should MLInsight treat as packages. 1 means if hello.a and hello.b will be treated as the same package hello
    typedef uint64_t (*PyPreHookHandler)(HookContext *curContextPtr);

    typedef void (*PyPostHookHandler)(uint64_t preHookTimestamp, HookContext *curContextPtr, RecTuple &curRecTuple);

    ObjectPoolHeap<PyCodeExtra> *pyCodeExtraHeap = nullptr; //todo: This object itself is a memory leak.
    std::map<std::string, FileID> *fileNameFileIdMap = nullptr; //todo: This object itself is a memory leak. Map fileName string to mlinsight FileID
    //HookContext *curContextPtr = NULL;
/*
 * Memory analyzer required fields
 */

/*The following functions are only defined if python interpreter is 3.9*/
    PyPreHookHandler preHookHandlerPtr = nullptr;
    PyPostHookHandler postHookHandlerPtr = nullptr;

    PyCodeExtra *getPyCodeExtra(PyFrameObject *f) {
        PyCodeExtra *curCodeExtra = nullptr;
        int pyCodeExtraRlt = _PyCode_GetExtra((PyObject *) f->f_code, PyCodeExtra_Index,
                                              (void **) &curCodeExtra);
        assert(pyCodeExtraRlt != -1);

        if (!curCodeExtra) {
            //PyCodeObject is null. Initialize pycode object.
            assert(pyCodeExtraHeap!=nullptr);
            curCodeExtra = pyCodeExtraHeap->alloc();

            new (curCodeExtra) PyCodeExtra();
            //curCodeExtra=new PyCodeExtra();
            //INFO_LOGS("Allocated pycode object at %p %zd %zd",curCodeExtra,curCodeExtra->pyModuleFileId,curCodeExtra->pyModuleRecArrMap.getSize());

            assert(curCodeExtra->pyModuleRecArrMap.getSize() == 0);

            int setRlt = _PyCode_SetExtra((PyObject *) f->f_code, PyCodeExtra_Index, (void *) curCodeExtra);
            assert(setRlt == 0);

            /*
            * Record callstack related information
            */
            curCodeExtra->pythonSourceFileName=PyUnicode_AsUTF8(f->f_code->co_filename);
            curCodeExtra->pythonFunctionName=PyUnicode_AsUTF8(f->f_code->co_name);
            
#if 0 //Block timing code
            /*
            * Allocate code timing block 
            */
            // Get the global namespace dictionary
            PyObject* globals = PyEval_GetGlobals();

            // Get the value of the "__name__" variable from the global namespace
            PyObject* nameObj = PyDict_GetItemString(globals, "__name__");

            //printf("getPyCodeExtra 11\n");
            if(nameObj == 0)
                return curCodeExtra;

            // Convert the Python object to a C string
            const char* nameStr = PyUnicode_AsUTF8(nameObj);
            std::string pyNameVar(nameStr);


            ssize_t subStrEndLoc = 0;
            for (int i = 0; i < PyPackage_Level; ++i) { 
                subStrEndLoc = pyNameVar.find('.', subStrEndLoc);
                if (subStrEndLoc == -1) {
                    //Use the entire string
                    subStrEndLoc = pyNameVar.length();
                    break;
                }
            }
            std::string pyModuleName = pyNameVar.substr(0, subStrEndLoc);

            auto fileIdIter = fileNameFileIdMap[0].find(pyModuleName);
            ssize_t newFileId = -1;
            if (fileIdIter == fileNameFileIdMap[0].end()) {
                HookInstaller *inst = mlinsight::HookInstaller::getInstance();
                
                //Allocate a new file Id
                //INFO_LOGS("thread:%p pthread_mutex_lock(&inst->dynamicLoadingLock)",pthread_self());
    
                pthread_mutex_lock(&inst->dynamicLoadingLock);
                newFileId = inst->elfImgInfoMap.getSize();
                ELFImgInfo *curElfImgInfo = inst->elfImgInfoMap.pushBack();
                fileNameFileIdMap[0][pyModuleName] = newFileId;

                //INFO_LOGS("thread:%p pthread_mutex_unlock(&inst->dynamicLoadingLock)",pthread_self());

                pthread_mutex_unlock(&inst->dynamicLoadingLock);
                //INFO_LOGS("Allocated a new fileId %d for module %s",newFileId,pyModuleName.c_str());
            } else {
                //Found fileId
                newFileId = fileIdIter->second;
            }

            //Record new recId in code extra
            curCodeExtra->pyModuleFileId = newFileId;
#endif
        }
        //else{
        //INFO_LOGS("ensureMlTracerIsInstalled %p %zd", curCodeExtra,curCodeExtra->pyModuleFileId);
        //}
        return curCodeExtra;
    }

    inline RecTuple &getRecTuple(PyCodeExtra *prevCodeExtra, PyCodeExtra *curCodeExtra, PyFrameObject *curFrame) {
        ssize_t originalSize = curCodeExtra->pyModuleRecArrMap.getSize();
        //Make sure there are enough space to store this entry
        //INFO_LOGS("i=%ld;i<=%zd;++i; curCodeExtra=%p prevCodeExtra=%p",originalSize-1,prevCodeExtra->pyModuleFileId,curCodeExtra,prevCodeExtra);
        for (ssize_t i = originalSize - 1; i <= prevCodeExtra->pyModuleFileId; ++i) {
            curCodeExtra->pyModuleRecArrMap.pushBack(-1);
            //Indicate this invocation relation has no allocated entry in recording array yet.
        }

        ssize_t &curRecId = curCodeExtra->pyModuleRecArrMap[prevCodeExtra->pyModuleFileId];
        //INFO_LOGS("Existing record Id is: %zd",curRecId);

        HookContext *curContextPtr = curContext;
        assert(curContextPtr != nullptr);
        if (curRecId == -1) {
            //todo: Acquire lock and insert symbol entry
            //Allocate new recording entry and record the ID.
            //Allocate a new recEntry in recTuple
            curRecId = curContextPtr->recordArray.getSize();
            //INFO_LOGS("Current function name is %s",PyUnicode_AsUTF8(curFrame->f_code->co_name));
            //INFO_LOGS("Allocate a new record Id. Id updated to: %zd",curRecId);
            return curContextPtr->recordArray.pushBack();
        } else {
            //INFO_LOGS("Existing record Id is: %zd",curRecId);
            if (curRecId >= curContextPtr->recordArray.getSize()) {
                curContextPtr->recordArray.allocateArray(curRecId + 1 - curContextPtr->recordArray.getSize());
            }

            return curContextPtr->recordArray[curRecId];
        }
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
            //Push
            PyCodeExtra *curCodeExtra = getPyCodeExtra(curFrame);

            curContextPtr->hookTuple[i].callerAddr = 0; //Python do not need to record return pointer
            curContextPtr->hookTuple[i].id.fileId = curCodeExtra->pyModuleFileId;
            curFrame = curFrame->f_back;
        }

    }

    template<typename PYFRAME_T,typename... FRAME_EVAL_Args>
    extern PyObject *evalFrameFuncGeneral(PYFRAME_T f,FRAME_EVAL_Args... args);

#if PY_VERSION_38 <= PY_VERSION_HEX && PY_VERSION_HEX < PY_VERSION_39

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
//TODO
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
    template<typename PYFRAME_T,typename... FRAME_EVAL_Args>
    PyObject *executePythonFrame(PyFrameObject* f, PyThreadState * ts, int throwflag) {
        PyObject *ret = _PyEval_EvalFrameDefault(ts, f, throwflag);
        return ret;
    }

    PyObject *evalFrameFunc(PyThreadState * tstate, PyFrameObject * f, int throwflag){
    
        return evalFrameFuncGeneral<PyFrameObject *, PyThreadState *, int>(f, tstate, throwflag);
    }

    bool installPyFrameInterceptor(PyPreHookHandler preHookHandler, PyPostHookHandler postHookHandler) {
        
        assert(preHookHandler!=nullptr && postHookHandler!=nullptr);
        // Retrieve the current frame evaluation function for the main interpreter state
        _PyFrameEvalFunction prevEvalFrameFunc = _PyInterpreterState_GetEvalFrameFunc(PyInterpreterState_Main());
        if (prevEvalFrameFunc == evalFrameFunc) {
            return false;
        }

        _PyInterpreterState_SetEvalFrameFunc(PyInterpreterState_Main(), evalFrameFunc);
        preHookHandlerPtr = pyPreHookAttribution;
        postHookHandlerPtr = pyPostHookAttribution;

        return true;
    }
#else
    #error MLInsight require a minimum version of python 3.6.
#endif

    template<typename PYFRAME_T,typename... FRAME_EVAL_Args>
    PyObject *evalFrameFuncGeneral(PYFRAME_T f,FRAME_EVAL_Args... args) {
        //printf("Inside evalFrameFuncGeneral NOOOOOOO\n");
        if(invokeNum <= SKIP_PYFRAMES) {
            invokeNum += 1; 
            return executePythonFrame<PYFRAME_T,FRAME_EVAL_Args...>(f,args...);
        }

        HookContext *curContextPtr = curContext;
        if (curContextPtr == nullptr) {
            INFO_LOGS("curContextPtr==nullptr pthread_self=%p", (void*)pthread_self());
        }
        assert(curContextPtr != nullptr);
        if (bInit == false) {
            initializeExistingPyFrames(curContextPtr);
            bInit = true;
        }
        
        uint64_t preHookTimestamp = preHookHandlerPtr(curContextPtr);

        PyFrameObject *previousFrame = f->f_back;
        if(previousFrame){
            //Since we use ObjectPoolHeap, the code extra address will remain unchanged.
            PyCodeExtra *prevCodeExtra = getPyCodeExtra(previousFrame);
            //Record call stack information into per-thread ringbuffer
            //curContextPtr->callStackRingBuffer.forceEnqueue(prevCodeExtra);
        }

        PyObject *ret = executePythonFrame<PYFRAME_T,FRAME_EVAL_Args...>(f,std::forward<FRAME_EVAL_Args>(args)...);

        if(previousFrame){
            //curContextPtr->callStackRingBuffer.dequeue();
        }

#if 0 //Block Timing code
        RecTuple *curRecTuple = nullptr;
        if (e) {
            PyCodeExtra *prevCodeExtra = getPyCodeExtra(previousFrame);
            //Check if the current module is a new module or not. We can check this by requesting code extra
            PyCodeExtra *curCodeExtra = getPyCodeExtra(f);
            if (prevCodeExtra == NULL or curCodeExtra == NULL) {
                // TODO
                printf("code extra will be NULL NOW!!!!");
                return ret; 
            }

            curRecTuple = &getRecTuple(prevCodeExtra, curCodeExtra, f);
            postHookHandlerPtr(preHookTimestamp, curContextPtr, *curRecTuple);
        } else {
            //Shhould be impossible, but we still need to handle this
            //todo:Attribute time to application
            //printf("Detected exits");
        }
#endif

        // ssize_t codeLine=PyFrame_GetLineNumber(f);
        // INFO_LOGS("********Tracing[Callee] %s:%d for function %s\r\n",
        //         PyUnicode_AsUTF8(f->f_code->co_filename),
        //         codeLine,
        //         PyUnicode_AsUTF8(f->f_code->co_name)
        //         ); 
        // if(f->f_back){
        //     codeLine=PyFrame_GetLineNumber(f);
        //     INFO_LOGS("********Tracing[Caller] %s:%d for function %s\r\n",
        //         PyUnicode_AsUTF8(f->f_back->f_code->co_filename),
        //         codeLine,
        //         PyUnicode_AsUTF8(f->f_back->f_code->co_name)
        //         ); 
        // }
        return ret;
    }


    bool installPythonInterceptor(){
        pyCodeExtraHeap=new ObjectPoolHeap<PyCodeExtra>();
        fileNameFileIdMap=new std::map<std::string,FileID>();
        if(isPythonAvailable()) {
            INFO_LOG("Python is available");
            if (Py_IsInitialized() == 0) {
                INFO_LOG("*******Python interpreter not initialized*******");
                Py_Initialize();
            }
            getCodeExtraIndex();


            installPyFrameInterceptor(pyPreHookAttribution, pyPostHookAttribution);

            INFO_LOG("*******Installing python interpreter done*******");
        }else {
            INFO_LOG("Python is not available");
        }
        return true;
    }

}