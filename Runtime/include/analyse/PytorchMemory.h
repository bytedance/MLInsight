#ifndef __PYTORCH_MEMORY_H__
#define __PYTORCH_MEMORY_H__
#include <cstdio>
#include <iostream>
#include <sys/types.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <Python.h>
#include <frameobject.h>
#include <ceval.h>
#include <unordered_map>
#include "analyse/CommonMemory.h"
//#include "common/HashMap.h"
#include "common/HashAndCompareFunctions.h"
#include "common/CallStack.h"
#include "trace/type/PyCodeExtra.h"
#include "trace/hook/PyHook.h"


namespace mlinsight{
class TorchObject {
public:
    size_t  initSize=0;  // initial block allocatedSize;
    ssize_t size=0;      // current block allocatedSize;
    TorchObject * prev=nullptr; // previous block in the same cudaMalloc's allocation
    TorchObject * next=nullptr; // next block in the same cudaMalloc's allocation
    void        * ptr=nullptr; // The real memory address
    bool          allocated=false;
    // The following is only available for an alive object
    CallStack<PyCallStack, PYTHON_CALL_STACK_LEVEL> callstack;
    ssize_t     fragment=0;

public:
    //TODO: Simplify this
    TorchObject(): callstack(){
        //This empty constructor is necessary because we need to support map
        this->prev = nullptr; 
        this->next = nullptr; 
    }

    /**
     * Constructor used when first allocating a block
     * @return
     */
    TorchObject(int initSize, void * ptr):initSize(initSize),size(initSize),callstack(),ptr(ptr),fragment(0),allocated(false){
        this->prev = nullptr; 
        this->next = nullptr; 
    }

    /**
     * Constructor used when spliting block
     */
    TorchObject(int initSize, int size):initSize(initSize),size(size),callstack() {
        this->prev = nullptr; 
        this->next = nullptr; 
    }

    ~TorchObject() {
    }

    bool is_split() const {
        return (prev != nullptr) || (next != nullptr);
    }

    void updatePythonCallStack() {
        //Acquire GIL
        PyGILState_STATE gstate;
        gstate = PyGILState_Ensure();
        PyFrameObject* currentFrame= PyEval_GetFrame();
        if(callstack.array == nullptr) {
            ERR_LOGS("updatePythonCallStack: After PyEval_GetFrame currentFrame=%p callstack.array %p",currentFrame, callstack.array);
            assert(callstack.array != nullptr);
        }

        int i = 0;
        ssize_t callStackMaxDepth=callstack.getMaxDepth();
        for(i = 0; i < callStackMaxDepth; ++i){
            if(currentFrame == NULL)
                break; 

            //Cache the name and line number of current python frame
            PyCodeExtra * codeExtra = getPyCodeExtra(currentFrame);

            callstack.array[i].cachedCodeExtra = codeExtra;
            callstack.array[i].pythonSourceFileLineNumber=PyFrame_GetLineNumber(currentFrame); //Get code line number
            
            //Go to next frame
            currentFrame=currentFrame->f_back;
        }
        callstack.levels=i;

        //INFO_LOGS("Collecting callstacks with level %d\n", callstack.levels);
        //Release GIL
        PyGILState_Release(gstate);
        //Perform a test print
        //printPythonCallStack();
    }

   /**
     * Debug only
    */
    void printPythonCallStack(){
        for(int i=0;i<this->callstack.levels;++i){
            const PyCallStack& pyCallStack=this->callstack.array[i];
            INFO_LOGS("%d Function: %s Line: %s:%zd",i,pyCallStack.cachedCodeExtra->pythonFunctionName.c_str(),
                      pyCallStack.cachedCodeExtra->pythonSourceFileName.c_str(),pyCallStack.pythonSourceFileLineNumber);
        }
    }
};

class PytorchMemory {
public:
    MemBasic basic;
    MemAlloc<TorchObject*> alloc;

    // Some information about cudaMalloc and cudaFree
    ssize_t countCudaMallocs=0;
    ssize_t countCudaFrees=0;
    ssize_t memCudaFrees=0;

    ssize_t numAllocs=0;
    ssize_t memFreedObjects=0;     // Available freed objects in the memory
    ssize_t numFreedObjects=0;     // Available freed objects in the memory
    ssize_t maxFreedObjectSize=0;
    ssize_t memFreedSmallObjects=0;
    ssize_t memFreedLargeObjects=0;
    ssize_t maxInternalFrag=0;
    ssize_t internalFrag=0; // Memory wasted due to internal fragmentation
    ssize_t maxExternalFrag = 0; 
    ssize_t requestSizeAtMaxExternalFrag = 0;
    ssize_t memoryAtMaxExternalFrag = 0;
    
    std::unordered_map<void *, TorchObject*> mapFreeObjs;

    PytorchMemory() {

    }
};

extern PytorchMemory torchMem;  

void printPytorchMemoryProfile(std::ofstream & output);

void printLeakyTorchObjects(std::ofstream &output);


ssize_t getInternalFragment(void * ptr, ssize_t size);
#if TORCH_VERSION_MAJOR >= 2
void processCUDAOOMError(const c10::OutOfMemoryError&, ssize_t allocationSize);
#else
void processCUDAOOMError(const c10::CUDAOutOfMemoryError&, ssize_t allocationSize);
#endif
void trackPytorchFree(void * ptr);
void trackPytorchAllocation(ssize_t size, void * ptr);
void trackTorchCudaMalloc(void* devicePtr, ssize_t size);
void trackTorchCudaFree(void* devicePtr, ssize_t size);
void printPythonCallstack(std::ofstream &output, CallStack<PyCallStack, PYTHON_CALL_STACK_LEVEL> & cs);

}

#endif
