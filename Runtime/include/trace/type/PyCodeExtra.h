#ifndef MLINSIGHT_PYCODEEXTRA_H
#define MLINSIGHT_PYCODEEXTRA_H
#include "common/Array.h"
#include "trace/type/RecordingDataStructure.h"

namespace mlinsight{
    class PyCodeExtra{
    public: 
        Array<FuncID> pyModuleRecArrMap; //Map python module (caller) to funcId in the recording array
        FileID pyModuleFileId=0; //The fileID of the newly loaded python module (callee)
        std::string pythonSourceFileName;
        std::string pythonFunctionName;
    };


    class PyCallStack{
    public:
        //For every single code object, only one code extra is allocated and cached.
        PyCodeExtra* cachedCodeExtra=nullptr;
        ssize_t pythonSourceFileLineNumber=0;

        bool operator==(PyCallStack& rho){
            return this->cachedCodeExtra==rho.cachedCodeExtra && this->pythonSourceFileLineNumber==rho.pythonSourceFileLineNumber;
        }

        void printCallstack(std::ofstream & output) {
            
        }
    };


    inline void freePyCodeExtra(void* obj) {
        //Pycode extra is allocated in chunk, do not delete here.
        //assert(obj!=nullptr);
        //PyCodeExtra* pycodeExtra = (PyCodeExtra*)obj;
        //delete pycodeExtra;
    }
    
}
#endif