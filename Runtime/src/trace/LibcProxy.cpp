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
#include "analyse/MemLeak/MemLeakAnalyzer.h"

#if CUDA_ENABLED

#include "common/CUDAHelper.h"
#include "trace/proxy/GPUEventTrace.h"
#include "analyse/GlobalVariables.h"
#include "trace/hook/PyHook.h"
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
    OUTPUTS("Received SIGUSR2 (signal %d). Pausing execution.\n", signum);

    //stopTracing(std::move(tracingSession));
    //tracingSession = startTracing();
    // Handle the pausing logic here
    //saveData(curContext, true);
}

void initLogAPI(){
    mlinsight::initLog();
}

void onPreMainFunction(int argc, char **argv, char **envp){
    using namespace mlinsight;
    hasMainFunctionStarted = true;
    AROUTPUTS("MLInsight %s hooked program: %s ", MLINSIGHT_VERSION, argv[0]);
    for (int i = 1; i < argc; ++i) {
        OUTPUTS("%s ", argv[i]);
    }
    AROUTPUT("\n");
    DBG_LOGS("Main thread id is %lu", pthread_self());
    AROUTPUT("Python installation pending...\n");
    std::stringstream ss;
    ss << logProcessRootPath << "/timeprofile_" << getunixtimestampms();
    //INFO_LOGS("Folder name is %s", ss.str().c_str());



    mlinsight::HookInstaller::getInstance(ss.str())->install();
    //Calculate the main application time
    installed = true;
    INFO_LOGS("cuMemAlloc_proxy is called pid:%d",getpid());
    // INFO_LOGS("Waiting for debugger %d.",getpid());
    //INFO_LOGS("Local Rank: %s",getenv("LOCAL_RANK"));
    
    //Currently, initCuptiTrace is not mandatory as it is only used for corss-checking correctness.
    //INFO_LOG("Open CUPTI GPU activity trace (Async)");
    // INFO_LOGS("Waiting for debugger %zd",getpid());
    // while(!DEBUGGER_CONTINUE){
    //     usleep(1000);
    // }
    bypassCHooks = MLINSIGHT_TRUE; //initCuptiTrace will invoke a dlopen and may gets intercepted by MLInsight if cupti is linked as a dynamic library. If we do not add bypassCHook, MLInsight will think that it is the Python interpreter that invoked the dlopen and may call some python APIs preMaturely. Basically, dlopen_proxy originated by MLinsight should be bypassed. 
    char* MLINSIGHT_LOGROOT_PIDcpy = getenv("MLINSIGHT_LOGROOT_PID");
    initCuptiTrace();
    if(MLINSIGHT_LOGROOT_PIDcpy){
        //In Colossal AI launcher and FSDP, initCuptiTrace will clear out environment variable.
        //To mitigate this, we pre-save the environment variable and set it again.
        setenv("MLINSIGHT_LOGROOT_PID",MLINSIGHT_LOGROOT_PIDcpy,0);
    }
    bypassCHooks = MLINSIGHT_FALSE;

    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&analyzerLock, &attr);


    mlinsight::HookContext *curContextPtr = mlinsight::curContext;
    assert(curContextPtr!=nullptr);
    curContextPtr->threadCreatorFileId = 0;
    curContextPtr->isMainThread = true;

    DBG_LOGS("At the beginning of process %d\n", getpid());
    /**
     * Register this thread with the main thread
     */
    threadContextMap.pushBack(curContextPtr);

    // Install the handlers for SIGCONT (18) and SIGSTP (19) signal
    signal(SIGUSR2, handleSIGUSR2);
}

void onPostMainFunction(int argc, char **argv, char **envp, int ret){
    using namespace mlinsight;
    HookContext *curContextPtr = curContext;
    
    // if(isPerfettoEnabled && pyTorchHookInstalled){
    //     stopTracing(std::move(tracingSession));
    // }

    //printf("inside the end of main() ret %d!!\n", ret);
    INFO_LOG("finiTrace()");
    finiTrace();
    // After exiting the main function, we will invoke saveData to
    // save the results of the main thread.
    saveData(curContextPtr, true);

}

int customMainEntry(int argc, char **argv, char **envp) {
    using namespace mlinsight;
    onPreMainFunction(argc, argv, envp);
    /**
     * Write special rules here to skip unwanted process.
     * Hooking an unwanted process will not affect program correctness.
     * Developers should think carefully whether hard-coded rules should be used here.
     */
    if (strncmp(argv[0], "time", 4) == 0 || mlinsight::strEndsWith(argv[0], "/time")) {
        OUTPUTS("MLInsight %s skipped program %s, because it is the time program.\n\n", MLINSIGHT_VERSION, argv[0]);
        return real_main(argc, argv, envp);
    }
    int ret = real_main(argc, argv, envp);
    onPostMainFunction(argc, argv, envp, ret);
    
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


