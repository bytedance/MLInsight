#include "trace/type/PyCodeExtra.h"
#include "Python.h"
#include "trace/hook/PyHook.h"
#include "trace/hook/HookInstaller.h"

namespace mlinsight
{
    namespace callstack::python{
        std::atomic<ssize_t> globalPyCodeExtraIdCounter;
    }

    std::unordered_map<PyCallStack*,ssize_t,callstack::HasherPtr<PyCallStack>,callstack::ComparaterPtr<PyCallStack>>* pyCallStackRegistery = new std::unordered_map<PyCallStack*,ssize_t,callstack::HasherPtr<PyCallStack>,callstack::ComparaterPtr<PyCallStack>>();
    ObjectPoolHeap<PyCallStack>* pyCallStackHeap=new ObjectPoolHeap<PyCallStack>();


    std::string PythonFrameExtra_t::toString(){

        if(this->toStringCache.length()==0){
            char retString[4096];
            snprintf(retString,sizeof(retString)/sizeof(char), "%s:%zd",
                            this->pythonSourceFileName.c_str(),
                            this->pythonSourceFileLineNumber);
            toStringCache =std::string(retString);
        }

        return toStringCache;
    }

    void PythonFrameExtra_t::print(std::ostream &os){
        os<<this->toString()<<std::endl;
    }

    std::string PyCallStack::toString(){

        if(this->strCache.length()==0){
            if(this->array.size()==0){
                std::stringstream ss;
                for (int i = 0; i < this->array.size(); i++)
                {  
                    this->array[i].extra->print(ss);
                }
                this->strCache=ss.str();
            }else{
                this->strCache="This is a pure-C stack and does not have Python callstack frames";
            }
        }

        return this->strCache;
    }
    void PyCallStack::print(std::ostream &output)
    {
        output<<this->toString();;
    }

    void PyCallStack::snapshot()
    {
        this->array.clear();
        this->strCache.clear();
        // Acquire GILls
        PyGILState_STATE gstate;
        gstate = PyGILState_Ensure();
        PyFrameObject *currentFrame = PyEval_GetFrame();

        while(currentFrame != NULL){
            // Cache the name and line number of current python frame
            PythonFrameExtra_t *codeExtraPtr = getPyCodeExtra(currentFrame);
            this->array.emplace_back(codeExtraPtr->pyFrameExtraID,codeExtraPtr); //Uses globalPyCodeExtraIdCounter to reduce the conflict iin
            // Go to next frame
            currentFrame = currentFrame->f_back;
        }

        // INFO_LOGS("Collecting callstacks with pyCallStackLevel %d\n", callstack.levels);
        // Release GIL
        PyGILState_Release(gstate);
        // Perform a test print
        // printPythonCallStack();

        // //Get the calstack id
        // auto insertionIter = pyCallStackRegistery->find(*this);
        // if(insertionIter==pyCallStackRegistery->end()){
        //     ssize_t newCallStackId=globalCallStackIdCounter.fetch_add(1);
        //     this->callstackID=newCallStackId;
        //     pyCallStackRegistery->emplace_hint(insertionIter,*this, newCallStackId);
        //     this->isNewCallStackId=true;
        // }else{
        //     this->callstackID=insertionIter->second;
        //     this->isNewCallStackId=false;
        // }
    }

    size_t PyCallStack::hash() const{
        size_t hashValue = 0xFFFFFFFF;
            for (int i = 0; i < this->array.size(); ++i) {
                hashValue ^= this->array[i].hash();
            }
        return hashValue;
    }


    //todo: Refactor these extern
    extern void *pythonInterpreter_text_begin;
    extern void *pythonInterpreter_text_end;
    extern int PyCodeExtra_Index;

//     void HybridCallStack::snapshot()
//     {
//         stringstream ss;

//         ssize_t pythonStackLevels=0;
//         if (isPythonAvailable() && Py_IsInitialized())
//         {
//             PyGILState_STATE gstate;
//             gstate = PyGILState_Ensure();
//             PyFrameObject *currentFrame = PyEval_GetFrame();
//             // Add the python part
//             while (currentFrame != NULL)
//             {
//                 pythonStackLevels+=1;
//                 //todo: Slow, use pycodeextra instead
//                 const char *pythonSourceFileName = PyUnicode_AsUTF8(currentFrame->f_code->co_filename);
//                 const char *pythonFunctionName = PyUnicode_AsUTF8(currentFrame->f_code->co_name);

//                 ss<<pythonSourceFileName<<":"<<PyFrame_GetLineNumber(currentFrame)<<"("<<pythonFunctionName<<")"<<std::endl;

//                 // Go to next frame
//                 currentFrame = currentFrame->f_back;
//             }
//             // Release GIL
//             PyGILState_Release(gstate);
//         }
        
//         if(pythonStackLevels==0){
//             // Find how many levels of stacks are below the first Python interpreter stack.
//             using namespace std;
//             const int hybridCStackLevel = 10;
//             void *array[hybridCStackLevel];
//             int size = backtrace(array, hybridCStackLevel);
//             int bottomCCallStackSize = 0;

//             for (int i = 0; i < size; ++i)
//             {
//                 if (pythonInterpreter_text_begin <= array[i] && array[i] <= pythonInterpreter_text_end)
//                 {
//                     // Is python interpreter stack frame
//                     bottomCCallStackSize = i;
//                     break;
//                 }
//             }
//             if (bottomCCallStackSize == 0)
//             {
//                 bottomCCallStackSize = size;
//             }

//             // Insert C stack into

//             char **strings = backtrace_symbols(array, bottomCCallStackSize);
//             for (int i = 0; i < bottomCCallStackSize; ++i)
//             {
//                 ss<<strings[i]<<std::endl;
//             }
//             free(strings);
//         }

//         this->callStackLines=ss.str();
//     }
}