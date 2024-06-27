#include "Logging.h"
#include <stdio.h>
#include <string.h>

extern "C" {
typedef int (*main_fn_t)(int, char **, char **);

main_fn_t real_main;

#ifndef MANUAL_INSTALL
int __libc_start_main(main_fn_t, int, char **, void (*)(), void (*)(), void (*)(),
                      void *) __attribute__((weak, alias("_libc_start_main"), visibility("default")));


typedef int (*exitFunc)(int __status);


typedef bool (*onPreMainFunction_t)(int argc, char **argv, char **envp);

typedef void (*onPostMainFunction_t)(int argc, char **argv, char **envp, int ret);

typedef void (*initLog_t)();

typedef void (*onPreDlOpen_t)(const char *__file, int __mode, void* callerAddress);
typedef void* (*onPostDlOpen_t)(const char *__file, int __mode, void* callerAddress, void* ret);
typedef void* (*dlopen_t)(const char *__file, int __mode) __THROWNL;

onPreMainFunction_t onPreMainFunctionPtr = nullptr;
onPostMainFunction_t onPostMainFunctionPtr = nullptr;
initLog_t initLogPtr = nullptr;
onPreDlOpen_t onPreDlOpenPtr=nullptr;
onPostDlOpen_t onPostDlOpenPtr=nullptr;
dlopen_t dlopenPtr = nullptr;

int customMainEntry(int argc, char **argv, char **envp) {
    using namespace mlinsight;
    if(!dlopenPtr){
        //dlopenPtr
        dlopenPtr = (dlopen_t) dlsym(RTLD_NEXT, "dlopen");
    }
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
    for (int i = 1; i < argc; ++i) {
        OUTPUTS("%s ", argv[i]);
    }
    OUTPUT("\n");
    DBG_LOGS("Main thread id is %lu", pthread_self());
    char savingFolder[4096];
    snprintf(savingFolder,4096,"%s/timeprofile_%ld",logProcessRootPath.c_str(), getunixtimestampms());
    //INFO_LOGS("Folder name is %s", ss.str().c_str());


    void* dlopenRet = dlopenPtr("libmlinsight.so",RTLD_LAZY|RTLD_LOCAL|RTLD_DEEPBIND);
    if(!dlopenRet){
        OUTPUTS("Cannot open libmlinsight.so because %s. MLInsight will be skipped for this process. \n", dlerror());
    }else{

        onPreMainFunctionPtr=(onPreMainFunction_t)dlsym(dlopenRet,"onPreMainFunction");
        if(onPreMainFunctionPtr==nullptr){
            fatalErrorS("Cannot find onPreMainFunction inside libmlinsight.so because: %s. This should not happen.",dlerror());
        }

        onPostMainFunctionPtr=(onPostMainFunction_t)dlsym(dlopenRet,"onPostMainFunction");
        if(onPostMainFunctionPtr==nullptr){
            fatalErrorS("Cannot find onPostMainFunction inside libmlinsight.so because: %s. This should not happen.",dlerror());
        }

        initLogPtr=(initLog_t)dlsym(dlopenRet,"_ZN9mlinsight7initLogEv");
        if(initLogPtr==nullptr){
            fatalErrorS("Cannot find mlinsight::initLog inside libmlinsight.so because: %s. This should not happen.",dlerror());
        }

        onPreDlOpenPtr = (onPreDlOpen_t)dlsym(dlopenRet,"_ZN9mlinsight11onPreDlOpenEPKciPv");
        if(onPreDlOpenPtr==nullptr){
            fatalErrorS("Cannot find mlinsight::onPreDlOpen inside libmlinsight.so because: %s. This should not happen.",dlerror());
        }

        onPostDlOpenPtr = (onPostDlOpen_t)dlsym(dlopenRet,"_ZN9mlinsight12onPostDlOpenEPKciPvS2_");
        if(onPostDlOpenPtr==nullptr){
            fatalErrorS("Cannot find mlinsight::onPostDlOpen inside libmlinsight.so because: %s. This should not happen.",dlerror());
        }
        DBG_LOGS("MLInsight Loader succesfully installed on pid %d",getpid());
        initLogPtr();
        onPreMainFunctionPtr(argc,argv,envp);
        fflush(logFileStd);
        
        int ret = real_main(argc, argv, envp);
        onPostMainFunctionPtr(argc, argv, envp, ret);
        return ret;
    }
    return real_main(argc, argv, envp);

}

char *findExecutionFile(char *string) {
    char *lastSlash = strrchr(string, '/');
    if (lastSlash != NULL) {
        return lastSlash + 1;
    } else {
        return string;
    }
}

void* dlopen(const char *__file, int __mode) __THROWNL  __attribute__((visibility("default")));
void* dlopen(const char *__file, int __mode) __THROWNL{

    if(!dlopenPtr){
        //dlopenPtr
        dlopenPtr = (dlopen_t) dlsym(RTLD_NEXT, "dlopen");
    }

    void *callerAddr = __builtin_return_address(0);


    if(!onPreDlOpenPtr || !onPostDlOpenPtr || !callerAddr){
        //It is possible that dlopen is invoked before main function and will not have address
        return dlopenPtr(__file, __mode);
    }

    //Forward dlopen request

    onPreDlOpenPtr(__file,__mode, callerAddr);

    void *ret = dlopenPtr(__file, __mode);

    return onPostDlOpenPtr(__file, __mode, callerAddr, ret);
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

#endif
}