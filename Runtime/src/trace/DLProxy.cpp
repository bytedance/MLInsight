/*
@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#include <cstdlib>
#include "trace/proxy/DLProxy.h"
#include "common/Logging.h"
#include "common/MemoryHeap.h"
#include "trace/hook/HookInstaller.h"
#include "common/DependencyLibVersionSpecifier.h"
#include "trace/hook/PyHook.h"
#include "trace/hook/HookContext.h"
#include "trace/tool/Perfetto.h"
#include "analyse/GlobalVariables.h"

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

namespace mlinsight {
    /**
     * An API function that supports both 
    */
    void onPreDlOpen(const char *__file, int __mode, void* callerAddress){
        if(!curContext){
            initTLS();
        }

    }

    void * onPostDlOpen(const char *__file, int __mode, void* callerAddress, void* ret){
        if (!ret && __file != NULL) {
            // Try agagin with relative import
            ssize_t filePathLen = strnlen(__file, PATH_MAX);
            if (filePathLen > 0 && __file[0] != '/') {
                // For link system, path always starts with '/'. If starts with '.' or an alphabet then dlopen may need to search for `pwd`
                DBG_LOG("dlopen is trying to use relative path. Push the caller id to the end of LD_PRELOAD.")
                //todo: Currently, MLInsight has turned off the C++ API hook. That is why using this method works. Otherwise the return address would have to be retrived from MLInsight's stack.
                void *callerAddr = callerAddress;
                assert(callerAddr != nullptr);
                assert(hookInstallerInstance != nullptr);
                
                std::string ldCallerFilePath; 

                Dl_info info;
                int dladdrRet = dladdr(callerAddr, &info);
                if(dladdrRet){
                    assert(info.dli_fname);
                    ldCallerFilePath=info.dli_fname;
                }else{
                    hookInstallerInstance->pmParser.parsePMMap();
                    ssize_t callerFileID = hookInstallerInstance->pmParser.findFileIdByAddr(callerAddr);
                    hookInstallerInstance->pmParser.getFileEntry(callerFileID).filePath;
                    ldCallerFilePath = hookInstallerInstance->pmParser.getFileEntry(callerFileID).filePath;
                }
                
                ssize_t rFindLoc = ldCallerFilePath.rfind('/');
                ldCallerFilePath = ldCallerFilePath.substr(0, rFindLoc) + "/" + __file;
                INFO_LOGS("Replaced relative to absolute path %s", ldCallerFilePath.c_str());
                ret = dlopen(ldCallerFilePath.c_str(), __mode);
            }
        }

        if (callerAddress==nullptr  || !hasMainFunctionStarted || bypassCHooks == MLINSIGHT_TRUE) {
            //Yes, it is possible to have no return address or dlopen is called before main function or dlopen is invoked by MLinsight itself.
            return ret;
        }
        INFO_LOGS("Installing onto dlopened file %s", __file);
        if (!isPyInterpreterInstalled) {
            isPyInterpreterInstalled = true;
            //todo: Assume the first dlopen to be single process
            installAfterPythonInit();
        }

        #ifdef USE_PERFETTO
        if(initializePerfetto && !isPerfettoEnabled){
            if(isRankParentProcess){
                setenv("MLINSIGHT_INSTALLED_RANK",localRank,1);
                isPerfettoEnabled = true;
                initializePerfetto();
                //assert(startTracing);
                //tracingSession = startTracing();
            } else if (pyTorchHookInstalled){
                //Is not a pytorch rank byt pytorch is installed, it means this process is main process
                isRankParentProcess=true;
                localRank="-1";

            }
        }
        #endif

        //INFO_LOGS("realPytorch2AllocatorPtr=%p",realPytorch2AllocatorPtr);

        if (ret) {
            //Successfully opened the library
            //DBG_LOGS("Installing on to open %s",__file);
            HookInstaller *inst = HookInstaller::getInstance();
            if (!inst) {
                ERR_LOG("MLInsight hook failed because MLInsight is not initialized yet.");
                return ret;
            }

            inst->installOnDlOpen();
            return ret;
        } else {
            DBG_LOGS("dlopen for %s failed. MLInsight will not install on this file. %d", __file, getpid());
            return ret;
        }
    }

    void *dlopen_proxy(const char *__file, int __mode) __THROWNL {
        void *callerAddr = __builtin_return_address(0);
        onPreDlOpen(__file,__mode, callerAddr);

        void *ret = dlopen(__file, __mode);

        return onPostDlOpen(__file, __mode, callerAddr, ret);
    }

    typedef void *(dlsym_t)(void *__restrict __handle, const char *__restrict __name) __THROW __nonnull ((2));

    typedef void *(dlvsym_t)(void *__restrict __handle, const char *__restrict __name,
                             const char *__restrict __version) __THROW __nonnull ((2, 3));


/* Find the run-time address in the shared object HANDLE refers to
   of the symbol called NAME.  */
    void *dlsym_proxy(void *__restrict __handle, const char *__restrict __name) __THROW {
        if (unlikely(bypassCHooks == MLINSIGHT_TRUE)) {
            return dlsym(__handle, __name);
        }
        //print_stacktrace();

        HookInstaller *instance = HookInstaller::getInstance();
        if(!curContext){
            initTLS();
        }
        HookContext *curHookContextPtr = curContext;
        
        void *realFuncAddr = nullptr;

        //todo: Currently, MLInsight has turned off the C++ API hook. That is why using this method works. Otherwise the return address would have to be retrived from MLInsight's stack.
        void *callerAddr = __builtin_return_address(0);
        assert(callerAddr != nullptr);

        if (__handle == RTLD_NEXT) {
            //This will be invalid if preHookInstaller is invoked because asmHookHandle replaces the return address
            //assert(callerAddr != nullptr);
            //realFuncAddr = searchNextSymbol<dlsym_t>(callerAddr, __name, dlsym);
            fatalError("This branch should not hit. Check whether dlsymJumper works correctly.");
        } else {
            realFuncAddr = dlsym(__handle, __name);
        }


        void *scalerReturnAddr = nullptr;
        instance->installOnDlSym(__name, realFuncAddr, callerAddr, scalerReturnAddr);
        return scalerReturnAddr;
    }

#ifdef __USE_GNU

    void *dlmopen_proxy(Lmid_t __nsid, const char *__file, int __mode) __THROW {

        //todo: Handle correct private namespace
        INFO_LOG("dlmopen Interception Start");
        INFO_LOG("dlmopen Interception End");
        void *rlt = dlmopen(__nsid, __file, __mode);
        if (bypassCHooks == MLINSIGHT_TRUE) {
            return rlt;
        }
        if (rlt) {
            //Successfully opened the library
            //DBG_LOGS("Installing on to open %s", __file);
            HookInstaller *inst = HookInstaller::getInstance();

            if (!inst) {
                ERR_LOG("Scaler hook failed because Scaler is not initialized yet.");
                return rlt;
            }

            //Actual installation
            inst->installOnDlOpen();

            return rlt;
        } else {
            DBG_LOGS("dlmopen for %s failed. Scaler will not install on this file.", __file);
            return rlt;
        }

    }

    int dlclose_proxy(void *__handle) __THROWNL {
        HookInstaller *inst = HookInstaller::getInstance();
        if(!__handle){
            INFO_LOGS("Waiting for debugger in PID %d",getpid());
            while(!DEBUGGER_CONTINUE){
                usleep(1000);
            }
        }
        inst->parseRealFileId();
        int rlt = dlclose(__handle);
        return rlt;
    }


    void *dlvsym_proxy(void *__restrict __handle,
                       const char *__restrict __name,
                       const char *__restrict __version)
    __THROW {
        HookInstaller *instance = HookInstaller::getInstance();
        void *realFuncAddr = nullptr;
        void *callerAddr = __builtin_return_address(0);
        assert(callerAddr != nullptr);
        if (__handle == RTLD_NEXT) {
            //This will be invalid if preHookInstaller is invoked because asmHookHandle replaces the return address
            //todo: Currently, MLInsight has turned off the C++ API hook. That is why using this method works. Otherwise the return address would have to be retrived from MLInsight's stack.
            //realFuncAddr = searchNextSymbol<dlvsym_t, const char *__restrict>(callerAddr, __name, dlvsym, __version);
            fatalError("This branch should not hit. Check whether dlsymJumper works correctly.");
        } else {
            realFuncAddr = dlvsym(__handle, __name, __version);
        }
        void *scalerReturnAddr = nullptr;
        instance->installOnDlSym(__name, realFuncAddr, callerAddr, scalerReturnAddr);
        return scalerReturnAddr;
    }


#endif

}
