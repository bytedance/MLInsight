//
// Created by user on 3/27/24.
//

#ifndef MLINSIGHT_MLINSIGHTPYAPI_H
#define MLINSIGHT_MLINSIGHTPYAPI_H

#include <Python.h>
#include "common/Logging.h"

/**
 * TODO: The entire file is just a piece of POC code and will be refactored soon.
 */
namespace mlinsight
{
    static PyObject *infoLog(PyObject *self, PyObject *args)
    {
        char *input = nullptr;
        if (!PyArg_ParseTuple(args, "s", &input))
        {
            fatalError("MLInsight failed to parse Python arguments. Has the pytorch API interface changed?");
        }
        assert(input != nullptr);
        INFO_LOG(input);
        // Py_IncRef(self);
        // auto* val=Py_BuildValue("");
        return returnPyNone();
    }

    struct MLInsightPyMonkeyPatchInfo
    {
        PyCFunctionWithKeywords realFuncAddr = nullptr;
        const char *funcName = nullptr;
        PyObject *callBackFuncObj = nullptr;

        MLInsightPyMonkeyPatchInfo(PyCFunctionWithKeywords realFuncAddr, const char *funcName,
                                   PyObject *callBackFuncObj) : realFuncAddr(realFuncAddr), funcName(funcName),
                                                                callBackFuncObj(callBackFuncObj)
        {
        }
    };

    // todo: Change to hashmap
    std::map<MLInsightPyMonkeyPatchInfo, PyCFunctionWithKeywords *> pyShadowTable;

    static PyObject *generalPyHook(PyObject *self, PyObject *args, PyObject **kwargs)
    {
        return self;
    }

    const int PYID_SAVER_BIN_SIZE = 23;
    struct PythonIdSaverBinWrapper
    {
        uint8_t idSaverBin[PYID_SAVER_BIN_SIZE] = {
            /**
             * Read context pointer (TLS)
             */
            // mov $0x1122334455667788,%rcx | move the address of real function to rcx
            0x48, 0xb9, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            // movq $0x1122334455667788,%r11 | Move the address of frameworkGeneral python proxy function
            0x49, 0xBB, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            // jmpq *%r11 | Jump to frameworkGeneral python proxy function
            0x41, 0xFF, 0xE3};
    };

    static ObjectPoolHeap<PythonIdSaverBinWrapper> pyIdSaversObjPool;
    typedef int64_t PyMonkeyPatchId; // Must be a 64 bits variable
    static std::vector<MLInsightPyMonkeyPatchInfo> monkeyPatchInfo;
    static bool enablePyAPICB = true;
    static PyObject *monkeyPatchGeneralInstaller(PyObject *self, PyObject *args, PyObject *kwargs, PyMonkeyPatchId patchId)
    {
        auto &tmpObj = monkeyPatchInfo[patchId];
        //        INFO_LOGS("Inside monkeyPatchGeneralInstaller calling %zd:%s at %p",patchId, tmpObj.funcName,tmpObj.realFuncAddr);

        PyObject *retObj = tmpObj.realFuncAddr(self, args, kwargs);

        if (enablePyAPICB)
        {
            Py_IncRef(retObj);
            Py_IncRef(args);
            Py_IncRef(kwargs);

            PyObject *d = PyDict_New();
            bool hasKwargs = true;
            if (kwargs == nullptr)
            {
                hasKwargs = false;
                kwargs = PyDict_New();
                assert(kwargs != nullptr);
            }
            // todo: the
            auto *mlinsightRetObjKey = PyUnicode_FromString("mlinsight_retobj");
            auto *mlinsightNameKey = PyUnicode_FromString("mlinsight_name");
            Py_IncRef(mlinsightRetObjKey);
            Py_IncRef(mlinsightNameKey);

            // todo: Cache string object
            PyDict_SetItem(kwargs, mlinsightNameKey, PyUnicode_FromString(tmpObj.funcName));
            PyDict_SetItem(kwargs, mlinsightRetObjKey, retObj);
            assert(tmpObj.callBackFuncObj != nullptr);
            //        fatalErrorS("apiMonkeyPatchCallBackFunc=%lu",apiMonkeyPatchCallBackFunc);
            //        PyObject_Print(apiMonkeyPatchCallBackFunc, stderr,NULL);
            //        PyObject* retBuildVal= Py_BuildValue("(O)",PyUnicode_FromString("mlinsight_name"),
            //                      PyUnicode_FromString(tmpObj.funcName),
            //                      PyUnicode_FromString("mlinsight_objret"),retObj);
            enablePyAPICB = false;
            PyObject_Call(tmpObj
                              .callBackFuncObj,
                          args, kwargs);
            enablePyAPICB = true;
            if (!hasKwargs)
            {
                Py_DecRef(kwargs);
                kwargs = nullptr;
            }
            Py_DecRef(mlinsightNameKey);
            Py_DecRef(mlinsightRetObjKey);

            Py_DecRef(retObj);
            Py_DecRef(args);
            Py_DecRef(kwargs);
        }
        return retObj;
    }

    static void fillAddr2PyIdSaver(uint8_t *idSaverEntry, PyMonkeyPatchId monkeyPatchId)
    {
        const int REAL_FUNCTION_OFFSET = 2;
        const int PYGENERAL_HOOK_OFFSET = 12;
        void *monkeyPatchGeneralInstallerPtr = reinterpret_cast<void *>(&monkeyPatchGeneralInstaller);
        memcpy(idSaverEntry + REAL_FUNCTION_OFFSET, (void *)&monkeyPatchId, sizeof(PyMonkeyPatchId));
        memcpy(idSaverEntry + PYGENERAL_HOOK_OFFSET, (void *)&monkeyPatchGeneralInstallerPtr, sizeof(void *));
    }

    static PyObject *monkeyPatchInstaller(PyObject *self, PyObject *args)
    {
        INFO_LOG("Monkey patching pytorch code......");
        PyTypeObject *pyTypeObject;
        PyObject *apiMonkeyPatchCallBackFuncTmp = nullptr;

        assert(PyTuple_Size(args) == 2);
        if (!PyArg_ParseTuple(args, "OO", &pyTypeObject, &apiMonkeyPatchCallBackFuncTmp))
        {
            fatalError("MLInsight failed to parse Python arguments. Has the pytorch API interface changed?");
        }

        PyMethodDef *defPtr = pyTypeObject->tp_methods;
        while (defPtr != nullptr && !(defPtr->ml_meth == 0 && defPtr->ml_doc == 0 && defPtr->ml_flags == 0 &&
                                      defPtr->ml_name == 0))
        {
            INFO_LOGS("Check %s", defPtr->ml_name);

            // Valid entry
            // todo: Check flags to ensure the function prototype is correctly checked.
            if (defPtr->ml_flags & METH_VARARGS && defPtr->ml_flags & METH_KEYWORDS)
            {
                // Replace variable
                // todo: Here, PythonIdSaverBinWrapper shoudl be aligned to page size in order for permission to work correctly. This is the same for other mmap code.
                // todo: The program did not crash because we always allow RWE permission
                PythonIdSaverBinWrapper *pyIdSaverBinWrapper = new PythonIdSaverBinWrapper();
                fillAddr2PyIdSaver(pyIdSaverBinWrapper->idSaverBin, monkeyPatchInfo.size());

                bool adjPermSuccess = adjustMemPerm(pyIdSaverBinWrapper->idSaverBin,
                                                    pyIdSaverBinWrapper->idSaverBin + PYID_SAVER_BIN_SIZE,
                                                    PROT_READ | PROT_WRITE | PROT_EXEC);
                if (!adjPermSuccess)
                {
                    fatalError("Cannot adjust permission for PyIdSaverBinWrapper");
                }
                INFO_LOGS("Monkeypatch Name:%s hooked PID:%d Ori=%p IdSaverBinAddr=%p", defPtr->ml_name, getpid(),
                          defPtr->ml_meth, pyIdSaverBinWrapper->idSaverBin);
                Py_IncRef(apiMonkeyPatchCallBackFuncTmp);
                monkeyPatchInfo.emplace_back((PyCFunctionWithKeywords)defPtr->ml_meth, defPtr->ml_name,
                                             apiMonkeyPatchCallBackFuncTmp);
                defPtr->ml_meth = reinterpret_cast<PyCFunction>(pyIdSaverBinWrapper->idSaverBin);
                //                getchar();
            }
            defPtr += 1;
        }
        Py_IncRef(self);
        return self;
    }

    static PyObject *monkeyPatchFunctionInstaller(PyObject *self, PyObject *args)
    {
        INFO_LOG("Monkey patching pytorch code......");
        PyTypeObject *pyTypeObject;
        PyObject *callBackFuncTmp = nullptr;
        const char *funcitonName = nullptr;
        assert(PyTuple_Size(args) == 3);
        if (!PyArg_ParseTuple(args, "OsO", &pyTypeObject, &funcitonName, &callBackFuncTmp))
        {
            fatalError("MLInsight failed to parse Python arguments. Has the pytorch API interface changed?");
        }
        std::string funcNameStr(funcitonName);
        PyMethodDef *defPtr = pyTypeObject->tp_methods;
        while (defPtr != nullptr && !(defPtr->ml_meth == 0 && defPtr->ml_doc == 0 && defPtr->ml_flags == 0 &&
                                      defPtr->ml_name == 0))
        {
            if (defPtr->ml_name != NULL)
            {
                INFO_LOGS("Checking function %s", defPtr->ml_name);
            }
            // Valid entry
            // todo: Check flags to ensure the function prototype is correctly checked.
            if (defPtr != NULL && defPtr->ml_name != NULL &&
                strncmp(defPtr->ml_name, funcNameStr.c_str(), funcNameStr.size()) == 0)
            {
                // Replace variable
                // todo: Here, PythonIdSaverBinWrapper shoudl be aligned to page size in order for permission to work correctly. This is the same for other mmap code.
                // todo: The program did not crash because we always allow RWE permission
                PythonIdSaverBinWrapper *pyIdSaverBinWrapper = new PythonIdSaverBinWrapper();
                fillAddr2PyIdSaver(pyIdSaverBinWrapper->idSaverBin, monkeyPatchInfo.size());

                bool adjPermSuccess = adjustMemPerm(pyIdSaverBinWrapper->idSaverBin,
                                                    pyIdSaverBinWrapper->idSaverBin + PYID_SAVER_BIN_SIZE,
                                                    PROT_READ | PROT_WRITE | PROT_EXEC);
                if (!adjPermSuccess)
                {
                    fatalError("Cannot adjust permission for PyIdSaverBinWrapper");
                }
                INFO_LOGS("Monkeypatch Name:%s hooked PID:%d Ori=%p IdSaverBinAddr=%p", defPtr->ml_name, getpid(),
                          defPtr->ml_meth, pyIdSaverBinWrapper->idSaverBin);
                Py_IncRef(callBackFuncTmp);
                monkeyPatchInfo.emplace_back((PyCFunctionWithKeywords)defPtr->ml_meth, defPtr->ml_name, callBackFuncTmp);
                defPtr->ml_meth = reinterpret_cast<PyCFunction>(pyIdSaverBinWrapper->idSaverBin);
            }
            defPtr += 1;
        }
        Py_IncRef(self);
        return self;
    }

}
#endif // MLINSIGHT_MLINSIGHTPYAPI_H
