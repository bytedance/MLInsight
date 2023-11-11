/*
@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#include <cstdlib>
#include "trace/proxy/DLProxy.h"
#include "common/Logging.h"
#include "common/MemoryHeap.h"
#include "trace/hook/HookInstaller.h"
#include <c10/core/Allocator.h>
#include <c10/cuda/CUDACachingAllocator.h>


namespace mlinsight{

std::atomic<c10::cuda::CUDACachingAllocator::CUDAAllocator*> pytorch2AllocationAtomicPtr;
extern std::atomic<c10::cuda::CUDACachingAllocator::CUDAAllocator*>* realPytorch2AllocatorPtr;


void *dlopen_proxy(const char *__file, int __mode) __THROWNL {
    void *ret = dlopen(__file, __mode);
    INFO_LOGS("Installing on to open %s",__file);
    INFO_LOGS("realPytorch2AllocatorPtr=%p",realPytorch2AllocatorPtr);

    if(ret){
        //Successfully opened the library
        //DBG_LOGS("Installing on to open %s",__file);
        mlinsight::HookInstaller* inst=mlinsight::HookInstaller::getInstance();
        if(!inst){
            ERR_LOG("MLInsight hook failed because MLInsight is not initialized yet.");
            return ret;
        }

        inst->installAPI();

        return ret;
    } else {
        //DBG_LOGS("dlopen for %s failed. MLInsight will not install on this file.",__file);
        return ret;
    }
}

/**
 * Return address
*/
void* searchNextSymbol(void* returnAddr, const char *__restrict __name){
    mlinsight::HookInstaller* instance=mlinsight::HookInstaller::getInstance();

    void* retFuncAddress=nullptr;

    //Search return address in pmParser
    bool found = false;
    ssize_t pmEntryId;
    //INFO_LOGS("Find address:%p",returnAddr);

    instance->pmParser.findPmEntryIdByAddr(returnAddr,pmEntryId,found);
    if(!found){
        //findPmEntryIdByAddr returns a lower bound, pmEntryId must -=1 to get the correct entry
        pmEntryId-=1;
        assert(pmEntryId>=0); 
    }

    const PMEntry & pmEntry=instance->pmParser.getPmEntry(pmEntryId);
    //pmEntryId returns an address lower bound, the return address must be contained by the pmEntry. We perfrom a simple assertion check here.
    assert(pmEntry.addrStart <returnAddr);
    assert(returnAddr<pmEntry.addrEnd);

    //Search symbol in subsequent libraries
    ssize_t fileEntryArraySize=instance->pmParser.getFileEntryArraySize();
    for(int curFileId=pmEntry.globalFileId+1;curFileId<fileEntryArraySize;++curFileId){
            //Get fileEntry from pmParser
        const FileEntry& fileEntry= instance->pmParser.getFileEntry(curFileId);
        
        //Get fileName and get fileHandle
        //DBG_LOGS("Try to open %s",inst->pmParser.getStr(fileEntry.pathNameStartIndex));
        void* libraryHandle = dlopen(instance->pmParser.getStr(fileEntry.pathNameStartIndex),RTLD_LAZY);
        if(libraryHandle==nullptr){
                
#ifndef NDEBUG
            if(fileEntry.valid){
                fatalError("File is considered valid by MLInsight, but dlopen failed on existing library %s. This should be impossible.");
            }
#endif
            continue;
        }

        // inst->pmParser.fileEntryArray.getSize()
        retFuncAddress=dlsym(libraryHandle,__name);
        if(retFuncAddress!=nullptr){
            //DBG_LOGS("Symbol %s found in %s",__name,inst->pmParser.getStr(fileEntry.pathNameStartIndex));
            break;
        }
    }

    //if(retFuncAddress==nullptr){
        //DBG_LOGS("Symbol %s not found",__name);
    //}
    return retFuncAddress;
}


/* Find the run-time address in the shared object HANDLE refers to
   of the symbol called NAME.  */
void *dlsym_proxy(void *__restrict __handle, const char *__restrict __name) __THROW {
    mlinsight::HookInstaller* instance=mlinsight::HookInstaller::getInstance();
    pthread_mutex_lock(&instance->dynamicLoadingLock);

     void *realFuncAddr = nullptr;
    if(__handle==RTLD_NEXT){
        //Special case handling since RTLD_NEXT will not yield correct results if we directly pass this parameter to dlsym.
        void *retAddr = __builtin_return_address(0);
        assert(retAddr!=nullptr);
        realFuncAddr=searchNextSymbol(retAddr,__name);
    }else{
        realFuncAddr = dlsym(__handle, __name);
    }

    Elf64_Word bind=STB_GLOBAL;
    Elf64_Word type=STT_FUNC;
    
    SymbolHookHint retSymbolHookHint;
    instance->shouldHookThisSymbol(__name,bind,type,retSymbolHookHint);
    if(retSymbolHookHint.realAddressPtr){
        *(retSymbolHookHint.realAddressPtr)=realFuncAddr;
    }

    if(retSymbolHookHint.shouldHook){
        //INFO_LOGS("dlsym API hooked: name:%s bind:%zd type:%zd addr:%p",__name,bind,type,realFuncAddr);
           
        if(retSymbolHookHint.addressOverride){
            realFuncAddr=retSymbolHookHint.addressOverride;
        }
        //TODO: Lock protect
        void* retAddr=nullptr;
        if(!mlinsight::HookInstaller::getInstance()->installDlSym(realFuncAddr, retAddr)){
            ERR_LOGS("Failed to hook on %s",__name);
            //INFO_LOGS("thread:%p pthread_mutex_unlock(&inst->dynamicLoadingLock)",pthread_self());
            pthread_mutex_unlock(&instance->dynamicLoadingLock);
            return realFuncAddr;
        }else{
            //INFO_LOG("Dlsym Interception End");
            //INFO_LOGS("thread:%p pthread_mutex_unlock(&inst->dynamicLoadingLock)",pthread_self());
            pthread_mutex_unlock(&instance->dynamicLoadingLock);
            return retAddr;
        }
    }else{
         INFO_LOGS("dlsym API NOT hooked: name:%s bind:%zd type:%zd addr:%p",__name,bind,type,realFuncAddr);
    }
    //INFO_LOGS("thread:%p pthread_mutex_unlock(&inst->dynamicLoadingLock)",pthread_self());
    pthread_mutex_unlock(&instance->dynamicLoadingLock);
    return realFuncAddr;
}

#ifdef __USE_GNU

//TODO: Handle this like dlopen

void *dlmopen_proxy(Lmid_t __nsid, const char *__file, int __mode) __THROW {
    INFO_LOGS("The \"dlmopen\" support is underway and the current MLInsight verison will not install on %s",__file);
    //INFO_LOG("dlmopen Interception End");
    return dlmopen(__nsid,__file,__mode);
}

//TODO: Handle this like dlsym

void *dlvsym_proxy(void *__restrict __handle,
                   const char *__restrict __name,
                   const char *__restrict __version)
__THROW {
    INFO_LOGS("The \"dlvsym\" support is underway and the current MLInsight verison will not hook %s",__name);
    //INFO_LOG("dlvsym Interception End");
    return dlvsym(__handle,__name,__version);
}

#endif

}