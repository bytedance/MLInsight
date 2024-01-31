/*
@author: Steven (Jiaxun) Tang <jtang@umass.edu>
@author: Tongping Liu <tongping.liu@bytedance.com>
*/
#include <dlfcn.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <csignal>
#include <unistd.h>
#include <cuda.h>
#include "common/Tool.h"
#include "common/Logging.h"
#include "trace/proxy/LibcProxy.h"
#include "trace/hook/HookInstaller.h"
#include "trace/hook/HookContext.h"
#include "trace/hook/PyHook.h"
#include "analyse/DriverMemory.h"
#if CUDA_ENABLED
#include "common/CUDAHelper.h"
#endif


namespace mlinsight {
    Array<mlinsight::HookContext *> threadContextMap;
    extern bool installed;
    pthread_mutex_t pytorchMemoryManagementLock;
}

extern "C" {
main_fn_t real_main;

#ifndef MANUAL_INSTALL
int __libc_start_main(main_fn_t, int, char **, void (*)(), void (*)(), void (*)(),
                      void *) __attribute__((weak, alias("_libc_start_main")));


typedef int (*exitFunc)(int __status);


void handleSIGUSR2(int signum) {
    using namespace mlinsight;
    DBG_LOG("In handling SIGUSR2!!");
    OUTPUTS("Received SIGUSR2 (signal %d). Pausing execution.\n",signum);

    // Handle the pausing logic here
    mlinsight::reportMemoryProfile(0);
}



int customMainEntry(int argc, char **argv, char **envp) {
    using namespace mlinsight;

    /**
     * Write special rules here to skip unwanted process.
     * Hooking an unwanted process will not affect program correctness.
     * Developers should think carefully whether hard-coded rules should be used here.
     */
    if (strncmp(argv[0], "time", 4) == 0 || mlinsight::strEndsWith(argv[0], "/time")) {
        OUTPUTS("MLInsight %s skipped program %s, because it is the time program.\n\n", MLINSIGHT_VERSION, argv[0]);
        return real_main(argc, argv, envp);
    }

    OUTPUTS("MLInsight %s hooked program: %s ", MLINSIGHT_VERSION, argv[0]);
    for(int i=1;i<argc;++i){
        OUTPUTS("%s ", argv[i]);
    }
    OUTPUT("\n");
    DBG_LOGS("Main thread id is %lu", pthread_self());

    std::stringstream ss;
    ss <<logProcessRootPath<< "/timeprofile_" << getunixtimestampms();
    //INFO_LOGS("Folder name is %s", ss.str().c_str());

    mlinsight::HookInstaller::getInstance(ss.str())->install();
    //Calculate the main application time
    installed = true;



    mlinsight::HookContext *curContextPtr = mlinsight::curContext;
    curContextPtr->threadCreatorFileId = 0;
    curContextPtr->isMainThread = true;

    DBG_LOGS("At the beginning of process %d\n", getpid());
    /**
     * Register this thread with the main thread
     */
    threadContextMap.pushBack(curContextPtr);

    // Install the python interceptor here. 
    mlinsight::installPythonInterceptor();

    // Install the handlers for SIGCONT (18) and SIGSTP (19) signal
    signal(SIGUSR2, handleSIGUSR2);

    int ret = real_main(argc, argv, envp);

    //printf("inside the end of main() ret %d!!\n", ret);

    // After exiting the main function, we will invoke saveData to 
    // save the results of the main thread.
    saveData(curContextPtr, true);
    return ret;
}

char *findExecutionFile(char *string) {
    char *lastSlash = strrchr(string, '/');
    if (lastSlash != NULL) {
        return lastSlash + 1;
    } else {
        return string;
    }
}

int _libc_start_main(main_fn_t main_fn, int argc, char **argv, void (*init)(), void (*fini)(),
                     void (*rtld_fini)(), void *stack_end) {
    using namespace mlinsight;
    //This function must be invoked before using any log macro
    initLog();

    // Find the real __libc_start_main
    auto real_libc_start_main = (decltype(__libc_start_main) *) dlsym(RTLD_NEXT, "__libc_start_main");
    if (!real_libc_start_main) {
        fatalError("Cannot find __libc_start_main.");
        return -1;
    }
    // Save the program's real main function
    real_main = main_fn;

    bool toSkipInterception = true;

#if 0
    char * skipProcesses[] = {
        "as",
        "nvcc",
        "ptxas",
        "fatbinary",
        "c++",
        "ninja",
        "lscpu",
        "cc1plus",
        "cudafe++",
        "gcc",
        "cicc",
        "collect2",
    };

    // Run the real __libc_start_main, but pass the custom's main function
    for(int i = 0; i < sizeof(skipProcesses)/sizeof(skipProcesses[0]); i++) {
        if(strcmp(findExecutionFile(argv[0]), skipProcesses[i]) == 0) {
            toSkipInterception = true;
            break;
        }
    }

    if(toSkipInterception== false && argc >=2 && (strstr(argv[1], "ninja") != nullptr || strstr(argv[1], "cpuinfo") != nullptr))
       toSkipInterception = true;
#endif

        //Initialize pthread_mutex_t pytorchMemoryManagementLock;
        pthread_mutexattr_t attr;
        pthread_mutexattr_init(&attr);
        pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
        pthread_mutex_init(&pytorchMemoryManagementLock, &attr);
        return real_libc_start_main(customMainEntry, argc, argv, init, fini, rtld_fini, stack_end);
}

void exit(int __status) {

    auto realExitFunc = (exitFunc) dlsym(RTLD_NEXT, "exit");

    if (mlinsight::installed) {
        //printf("inside the exit()!!\n");
        saveData(mlinsight::curContext, true);
    }
    realExitFunc(__status);
}

#endif
}


