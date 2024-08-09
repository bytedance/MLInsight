#include <Python.h>
#include "common/Logging.h"
#include "trace/hook/PyHook.h"
#include "trace/hook/HookInstaller.h"
#include "trace/proxy/PytorchMemProxy.h"
#include "analyse/GlobalVariables.h"
#include "trace/tool/Perfetto.h"
#include "analyse/MemLeak/MemLeakAnalyzer.h"
#include "trace/proxy/PyTorchCallBacks.h"
namespace mlinsight {
    //Check Pytorch/test/nn/test_modulehooks.py

    ssize_t stepCounter=-1;//Training step counter

    std::set<PyObject* > moduleSet; // A set used to find the root module
    //PyObject* rootModule=nullptr;
    PyObject *forward_pre_hook(PyObject * self, PyObject * args) {
        PyObject * module = nullptr;
        PyObject * input = nullptr;
        if (!PyArg_ParseTuple(args, "OO", &module, &input)) {
            fatalError("MLInsight failed to parse Python arguments. Has the pytorch API interface changed?");
        }

        // todo: Move the following code to perfetto analyzer
        Py_BEGIN_ALLOW_THREADS

        pthread_mutex_lock(&analyzerLock);

        if(moduleSet.find(module)!=moduleSet.end()){
            perfettoAnalyzer.onStepFinished(stepCounter);
            ++stepCounter;
        }

        auto findRet = hookInstallerInstance->pytorchModuleIdMap.find(module);
        if (findRet != hookInstallerInstance->pytorchModuleIdMap.end()) {
            FileID layerId = (*findRet).second;
            globalExecutionState.pyTorchModuleStack.emplace_back(layerId, MLExecutionState::FORWARD_STATE, module);
            perfettoAnalyzer.onPreLayerForward(layerId, globalExecutionState.pyTorchModuleStack.back());
        }else{
            //ERR_LOG("Encountered not registered module: ");
            //PyObject_Print(module,logFileStd,NULL);
        }
        pthread_mutex_unlock(&analyzerLock);

        Py_END_ALLOW_THREADS 

//        INFO_LOGS("[MLInsight ForwardPreHook: %s\n",);
//        PyObject_Print(module,logFileStd,NULL);
//        OUTPUT("]\n");
        return returnPyNone();
    }


    PyObject *forward_hook(PyObject * self, PyObject * args) {
        PyObject * module = nullptr;
        PyObject * input = nullptr;
        PyObject * output = nullptr;


        if (!PyArg_ParseTuple(args, "OOO", &module, &input, &output)) {
            fatalError("MLInsight failed to parse Python arguments. Has the pytorch API interface changed?");
        }

        Py_BEGIN_ALLOW_THREADS

        pthread_mutex_lock(&analyzerLock);
//        OUTPUT("argPrint:")
//        PyObject_Print(args,logFileStd,NULL);
//        OUTPUT("\n")
//        assert(hookInstallerInstance->pytorchModuleIdMap.find(module)!=hookInstallerInstance->pytorchModuleIdMap.end());
//        FileID moduleId = hookInstallerInstance->pytorchModuleIdMap[module];
//        INFO_LOGS("Pytorch Module %s forward End", hookInstallerInstance->pytorchModuleInfoMap[moduleId].moduleName.c_str());

        auto findRet = hookInstallerInstance->pytorchModuleIdMap.find(module);
        if (findRet != hookInstallerInstance->pytorchModuleIdMap.end()) {
            FileID moduleId = (*findRet).second;
            assert(globalExecutionState.pyTorchModuleStack.back().mlExecutionState ==
                   MLExecutionState::FORWARD_STATE);
            perfettoAnalyzer.onPostLayerForward(moduleId, globalExecutionState.pyTorchModuleStack.back());
            globalExecutionState.pyTorchModuleStack.pop_back();
            //INFO_LOGS("===[Pytorch Module] %s forward End",
            //          hookInstallerInstance->pytorchModuleInfoMap[moduleId].moduleName.c_str());
        }
        pthread_mutex_unlock(&analyzerLock);

        Py_END_ALLOW_THREADS 
        return returnPyNone();
    }

    PyObject *full_backward_pre_hook(PyObject * self, PyObject * args) {
        PyObject * module = nullptr;
        PyObject * output = nullptr;
        if (!PyArg_ParseTuple(args, "OO", &module, &output)) {
            fatalError("MLInsight failed to parse Python arguments. Has the pytorch API interface changed?");
        }
//        assert(hookInstallerInstance->pytorchModuleIdMap.find(module)!=hookInstallerInstance->pytorchModuleIdMap.end());
//        FileID moduleId = hookInstallerInstance->pytorchModuleIdMap[module];
//        INFO_LOGS("Pytorch Module %s backward Start", hookInstallerInstance->pytorchModuleInfoMap[moduleId].moduleName.c_str());
        Py_BEGIN_ALLOW_THREADS
        
        pthread_mutex_lock(&analyzerLock);
        auto findRet = hookInstallerInstance->pytorchModuleIdMap.find(module);
        if (findRet != hookInstallerInstance->pytorchModuleIdMap.end()) {
            FileID moduleId = (*findRet).second;
            globalExecutionState.pyTorchModuleStack.emplace_back(moduleId, MLExecutionState::BACKWARD_STATE, module);
            perfettoAnalyzer.onPreLayerBackward(moduleId, globalExecutionState.pyTorchModuleStack.back());
        } else{
            //ERR_LOG("Encountered not registered module: ");
            //PyObject_Print(module,logFileStd,NULL);
        }
        pthread_mutex_unlock(&analyzerLock);

        Py_END_ALLOW_THREADS        
        return returnPyNone();
    }

    PyObject *full_backward_hook(PyObject * self, PyObject * args) {
        PyObject * module = nullptr;
        PyObject * input = nullptr;
        PyObject * output = nullptr;


        if (!PyArg_ParseTuple(args, "OOO", &module, &input, &output)) {
            fatalError("MLInsight failed to parse Python arguments. Has the pytorch API interface changed?");
        }

        Py_BEGIN_ALLOW_THREADS

        pthread_mutex_lock(&analyzerLock);
//        assert(hookInstallerInstance->pytorchModuleIdMap.find(module)!=hookInstallerInstance->pytorchModuleIdMap.end());
//        FileID moduleId = hookInstallerInstance->pytorchModuleIdMap[module];
//        INFO_LOGS("Pytorch Module %s backward End", hookInstallerInstance->pytorchModuleInfoMap[moduleId].moduleName.c_str());
        //INFO_LOG("ForwardPreHook");

        auto findRet = hookInstallerInstance->pytorchModuleIdMap.find(module);
        if (findRet != hookInstallerInstance->pytorchModuleIdMap.end()) {
            FileID moduleId = (*findRet).second;
            
            //assert(globalExecutionState.pyTorchModuleStack.back().layerId == moduleId);
            //assert(globalExecutionState.pyTorchModuleStack.back().mlExecutionState ==
            //       MLExecutionState::BACKWARD_STATE);
            perfettoAnalyzer.onPostLayerBackward(moduleId, globalExecutionState.pyTorchModuleStack.back());
            globalExecutionState.pyTorchModuleStack.pop_back();
            //INFO_LOGS("===[Pytorch Module] %s backward End",
            //          hookInstallerInstance->pytorchModuleInfoMap[moduleId].moduleName.c_str());
        }

        //ERR_LOGS("pthread_mutex_unlock %p",pthread_self());
        pthread_mutex_unlock(&analyzerLock);
        
        Py_END_ALLOW_THREADS 

        return returnPyNone();
    }

    PyObject *module_registration_hook(PyObject * self, PyObject * args) {
        PyObject * module = nullptr;
        PyObject * name = nullptr;
        PyObject * submodule = nullptr;

        if (!PyArg_ParseTuple(args, "OOO", &module, &name, &submodule)) {
            fatalError("MLInsight failed to parse Python arguments. Has the pytorch API interface changed?");
        }
        pthread_mutex_lock(&analyzerLock);
        //INFO_LOG("ForwardPreHook");
        const char *pyTorchModuleName = PyUnicode_AsUTF8(name);
        INFO_LOGS("PyTorch Module Registration: %p %s moduleId=%zd", module, pyTorchModuleName,
                  hookInstallerInstance->pytorchModuleInfoMap.getSize());
        moduleSet.emplace(module);
        moduleSet.erase(submodule);
        
        //module_registeration_hook instance should only have one instance
        if(hookInstallerInstance->pytorchModuleIdMap.find(submodule) ==
               hookInstallerInstance->pytorchModuleIdMap.end()){
            PyObject* retObj1 = PyObject_CallMethod(submodule, "register_full_backward_pre_hook","(O)", full_backward_pre_hook_ptr);
            if(retObj1==nullptr){
                fatalError("MLInsight did not managed to call register_full_backward_pre_hook on a pytorch module. Maybe the interface changed?");
            }

            PyObject* retObj2 = PyObject_CallMethod(submodule, "register_full_backward_hook","(O)", full_backward_hook_ptr);
            if(retObj2==nullptr){
                fatalError("MLInsight did not managed to call register_full_backward_hook on a pytorch module. Maybe the interface changed?");
            }

            hookInstallerInstance->pytorchModuleIdMap[submodule] = hookInstallerInstance->pytorchModuleInfoMap.getSize();
            hookInstallerInstance->pytorchModuleInfoMap.pushBack(pyTorchModuleName);
        }
        pthread_mutex_unlock(&analyzerLock);
        return returnPyNone();
    }

    PyObject *parameter_registration_hook(PyObject * self, PyObject * args) {
        PyObject * module = nullptr;
        PyObject * name = nullptr;
        PyObject * parm = nullptr;


        if (!PyArg_ParseTuple(args, "OOO", &module, &name, &parm)) {
            fatalError("MLInsight failed to parse Python arguments. Has the pytorch API interface changed?");
        }

        pthread_mutex_lock(&analyzerLock);
        const char *pyTorchParameterName = PyUnicode_AsUTF8(name);
        INFO_LOGS("PyTorch Parameter Registration: %s", pyTorchParameterName);
        //INFO_LOG("ForwardPreHook");
        pthread_mutex_unlock(&analyzerLock);
        return returnPyNone();
    }
}