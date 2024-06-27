#ifndef MLINSIGHT_TENSORFLOWMEMPROXY_H
#define MLINSIGHT_TENSORFLOWMEMPROXY_H

#include <map>
#include <dlfcn.h>
#include "trace/hook/HookInstaller.h"
namespace mlinsight{


    namespace tensorflow{
        /**
       * Called at the initialization time of the HookInstaller
       */
        void onSettingHookHint(std::map<std::string, SymbolHookHint>& hookHintMap) __attribute__((weak));

        /**
         * Called after every installation of the hookInstaller
         */
        void onHookInstallationFinished() __attribute__((weak));


    }

}
#if USE_TENSORFLOW
namespace mlinsight{
    typedef const char* (*TFVersionPtr)();
    inline void checkTFVersion(const char* fullTFLibPath){
        bypassCHooks = MLINSIGHT_TRUE;
        INFO_LOGS("Check TF version for %s",fullTFLibPath);
        void* libHandle = dlopen(fullTFLibPath,RTLD_LAZY);
        if(!libHandle){
            ERR_LOGS("dlopen failed because: %s",dlerror());
        }
        assert(libHandle != nullptr);
        auto* tfVersionPtr = (TFVersionPtr) dlsym(libHandle,"TF_Version");
        assert(tfVersionPtr != nullptr);
        fatalErrorS("TF version is: %s",tfVersionPtr());
        bypassCHooks = MLINSIGHT_FALSE;
    }
}
#endif //USE_TENSORFLOW



#endif //MLINSIGHT_TENSORFLOWMEMPROXY_H