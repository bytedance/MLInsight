
#include <FuncWithDiffParms.h>

/*

@author: Steven Tang <steven.tang@bytedance.com>
*/
#include <pybind11/pybind11.h>
#include <cstdio>
#include <iostream>
#include <pthread.h>
#include <thread>
#include <chrono>

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <frameobject.h>
#include <dlfcn.h>


namespace mlinsight {
    typedef void (*FuncAType)();


    void callNativeFuncA(){
        funcA();
    }

    void *libHandle =nullptr;
    void callNativeFuncAThroughDlSym(){
        printf("TestDL\n");
        if(!libHandle){
            libHandle = dlopen(CMAKE_BINARY_DIR "/libTestlib-FuncCall.so",RTLD_LAZY);
        }
        if (!libHandle) {
            fprintf(stderr,"Cannot open " CMAKE_BINARY_DIR "/libTestlib-FuncCall.so""\n");
        }else{
            auto funcAPtr = (FuncAType) dlsym(libHandle, "funcA");
            if(!funcAPtr){
                fprintf(stderr,"Cannot find funcA\n");
            }else{
                funcAPtr();
            }
        }
    }


}


PYBIND11_MODULE(_testextensionmodule, m) {
    m.doc() = "NativeFuncCallWrapper";
    m.def("callNativeFuncA", &mlinsight::callNativeFuncA, "callNativeFuncA");
    //m.def("callNativeCallFuncA", &mlinsight::callNativeCallFuncA, "callNativeCallFuncA");
    m.def("callNativeFuncAThroughDlSym", &mlinsight::callNativeFuncAThroughDlSym, "callNativeFuncAThroughDlSym");
}