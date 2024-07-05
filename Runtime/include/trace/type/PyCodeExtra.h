#ifndef MLINSIGHT_PYCODEEXTRA_H
#define MLINSIGHT_PYCODEEXTRA_H

#include <utility>
#include <vector>
#include <string>

#include "common/Array.h"
#include "trace/type/RecordingDataStructure.h"
#include "common/CallStack.h"
namespace mlinsight {
    enum FUNCTIONAL_MARKER {
        UNSPECIFIED = 0,
        PYTORCH_IMPORT_FINISHED_MARKER = 1
    };

    /**
     * Python callstack
     */
    namespace callstack::python
    {
        class FrameExtra: public callstack::FrameExtra {
        public:
            Array<FuncID> pyModuleRecArrMap; // Map python module (caller) to funcId in the recording array.
            FileID globalPyModuleId = 0; // Actually equals to calleeFileId in the current implementation. Which means, the timing module will time based on the Python package.
            FileID globalPySrcFileId = 0; // The fileId for python source code file.
            ssize_t pyFrameExtraID=-1; //Map python callstack FrameExtra to an id
            std::string pythonSourceFileName;
            std::string pythonFunctionName;
            FUNCTIONAL_MARKER functionalMarker;
            std::string toStringCache;

            /**
             * Convert the callstack inforamtion into a string.
             */
            std::string toString(); 

            /**
             * Output the callstack information into std::ostream
             */
            void print(std::ostream &os);

        };

        extern std::atomic<ssize_t> globalPyCodeExtraIdCounter; //This variable is used to calculate a unique ID for PyCodeExtra. With this id, the hash conflict rate will be lower than address.

        class IntegerFrameKey : public callstack::FrameKey<ssize_t,FrameExtra>
        {
        public:
            ssize_t pythonSourceFileLineNumber;
            
            // The version of elem. If library is unloaded, then the address maybe reused. So one address may corresponds to multiple FrameKey-FrameExtra pairs.
            IntegerFrameKey() = default;
            inline IntegerFrameKey(ssize_t frameExtraID, FrameExtra *extra, ssize_t lineNumber) : callstack::FrameKey<ssize_t,FrameExtra>(frameExtraID,extra), pythonSourceFileLineNumber(lineNumber)
            {
            }

            /**
             * Compare callstack keys
             */
            inline bool operator==(const IntegerFrameKey &rho) const
            {
                return this->key == rho.key;
            }

            /**
             * Returns the hash of the this callstack key.
             */
            inline size_t hash() const
            {
                return std::hash<ssize_t>()(this->key);
            }

        };


    }

    typedef callstack::python::FrameExtra PythonFrameExtra_t;
    typedef callstack::python::IntegerFrameKey PythonFrameKey_t;

    class PyCallStack : public callstack::CallStack<PythonFrameKey_t, PYTHON_CALL_STACK_LEVEL>
    {
    public:
        std::string toString();

        void print(std::ostream &output);

        void snapshot();

        size_t hash() const;

    protected:
        std::string strCache;

    };

    /**
     * A callstack implementation for python that combines both C and Python
    */
    // class HybridPyCallStack : public callstack::CallStack<PythonFrameKey_t, PYTHON_CALL_STACK_LEVEL>
    // {
    // public:

    //     void print(std::ostream &output);

    //     void snapshot();

    //     size_t hash() const;
    // };

    // typedef callstackold::Hash<PyCallStackElem, PYTHON_CALL_STACK_LEVEL> PYStackHash_t;
    // typedef callstackold::Comparator<PyCallStackElem, PYTHON_CALL_STACK_LEVEL> PYStackCmp_t;

    extern std::unordered_map<PyCallStack*,ssize_t,callstack::HasherPtr<PyCallStack>, callstack::ComparaterPtr<PyCallStack>>* pyCallStackRegistery;
    extern ObjectPoolHeap<PyCallStack>* pyCallStackHeap;

    /**
     * This information is used in the flame graph visualization.
     */
    enum PyModuleType {
        UNKNOWN = 0,
        USER_CARE = 1, //User care about this package.
        USER_DONT_CARE = 2 //User do not care about this package.
    };

    /**
     * Similar to ELFImgInfo, marks the information for specific python modules.
     */
    struct PyModuleInfo {
        std::string moduleName;
        PyModuleType moduleType = PyModuleType::UNKNOWN;
        ssize_t pyPackageSummaryAttributionId = -1;

        inline PyModuleInfo() = default;

        inline explicit PyModuleInfo(std::string moduleName, const PyModuleType &pyModuleType,
                                     ssize_t pyPackageSummaryAttributionId) :
                moduleName(std::move(moduleName)),
                moduleType(pyModuleType),
                pyPackageSummaryAttributionId(pyPackageSummaryAttributionId) {
        }
    };


    /**
     * Similar to ELFImgInfo, marks the information for specific python source files.
     */
    struct PySrcFileInfo {
        std::string fileName;

        inline PySrcFileInfo() = default;

        inline explicit PySrcFileInfo(std::string fileName) : fileName(std::move(fileName)) {
        }
    };


    /**
     * Similar to ELFImgInfo, marks the information for specific python modules.
     */
    struct PyTorchModuleInfo {
        std::string moduleName;

        inline PyTorchModuleInfo() = default;

        inline explicit PyTorchModuleInfo(std::string moduleName) : moduleName(std::move(moduleName)) {
        }
    };

    inline void freePyCodeExtra(void *obj) {
        //Pycode extra is allocated in chunk, do not delete here.
        //assert(obj!=nullptr);
        //PyCodeExtra* pycodeExtra = (PyCodeExtra*)obj;
        //delete pycodeExtra;
    }

}
#endif