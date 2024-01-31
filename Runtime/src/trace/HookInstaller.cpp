/*
@author: Steven (Jiaxun) Tang <jtang@umass.edu>
@author: Tongping Liu <tongping.liu@bytedance.com>
*/
#ifdef __linux


#include <sys/mman.h>
#include <cassert>
#include <elf.h>
#include <set>
#include <utility>
#include "common/ProcInfoParser.h"
#include "common/ELFParser.h"
#include "trace/tool/Math.h"
#include "common/Tool.h"
#include "trace/type/RecordingDataStructure.h"
#include "trace/type/ELFInfo.h"
#include "trace/hook/HookHandlers.h"
#include "trace/hook/HookContext.h"

#include "common/Logging.h"
#include "trace/proxy/DLProxy.h"
#include "trace/proxy/SystemProxy.h"
#include "trace/proxy/PthreadProxy.h"
#include "analyse/SerializationDataStructure.h"
#include "analyse/LogicalClock.h"
#include "trace/proxy/PytorchMemProxy.h"
#include "common/CUDAHelper.h"

#ifdef USE_TORCH
#include "common/DependencyLibVersionSpecifier.h"
#include "trace/proxy/PytorchMemProxy.h"
#endif

#ifdef TENSOR_FLOW

#endif


#ifdef CUDA_ENABLED
#include "trace/proxy/CUDAProxy.h"
#include <c++/11/cxxabi.h>
#endif


namespace mlinsight {
    bool installed = false;

    bool HookInstaller::install() {
        DBG_LOG("*******HookInstaller::install*****");

        createRecordingFolder();

        if (!initTLS()) {
            ERR_LOG("Failed to initialize TLS");
            //This is the main thread
            return false;
        }


        initLogicalClock(curContext->cachedWallClockSnapshot, curContext->cachedLogicalClock,
                         curContext->cachedThreadNum);
        __cxxabiv1::__cxa_atexit([](void*)->void {saveData(curContext);},NULL,NULL);
        //Register datasaver hook
        return installAPI();
    }


    bool HookInstaller::installAPI() {
        //DBG_LOGS("Install with loadingId=%zd", loadingId);
        //DBG_LOG("Install DlOpen");
        parseRequiredInfo();

        populateRecordingArray(*this); //Invoke this again because new loadingID is loaded
        //DBG_LOG("Replace PLT entry");

        HookContext *curContextPtr = curContext;
        
        //DBG_LOGS("ContextPtr=%p",curContextPtr);
        //DBG_LOGS("&ContextPtr.recordArray=%p",&curContextPtr->recordArray[0].internalArray);
        //DBG_LOGS("ContextPtr.recordArray=%p",&curContextPtr->recordArray[0].internalArray[2].count);
        //INFO_LOGS("&realPytorch2AllocatorPtr=%p realPytorch2AllocatorPtr=%p",&realPytorch2AllocatorPtr,realPytorch2AllocatorPtr);


        return replacePltEntry();
    }


    bool HookInstaller::uninstall() {
//        //todo: release oriPltCode oriPltSecCode
//        //Decallocate recordbuffer for main thread
//
//        for (SymID curSymId: hookedExtSymbol) {
//            auto &curSymbol = allExtSymbol[curSymId];
//
//            //DBG_LOGS("[%s] %s hooked (ID:%zd)\n", curELFImgInfo.filePath.c_str(), curSymbol.symbolName.c_str(),
//            //curSymbol.extSymbolId);
//            void *oriCodePtr = nullptr;
//
//            if (curSymbol.oriPltSecCode != nullptr) {
//                //.plt.sec table exists
//                //todo: adjust the permission back after this
//                if (!adjustMemPerm(
//                        (uint8_t *) curSymbol.pltSecEntry,
//                        (uint8_t *) curSymbol.pltSecEntry + 16,
//                        PROT_READ | PROT_WRITE | PROT_EXEC)) {
//                    ERR_LOG("Cannot adjust memory permission");
//                    continue;
//                }
//                memcpy((uint8_t *) curSymbol.pltSecEntry,
//                       curSymbol.oriPltSecCode, 16);
//                free(curSymbol.oriPltSecCode);
//                curSymbol.oriPltSecCode = nullptr;
//
//            }
//
//            if (curSymbol.oriPltCode != nullptr) {
//                //todo: what is this doesn't exist (for example, installer failed at this symbol)
//                if (!adjustMemPerm(
//                        (uint8_t *) curSymbol.pltEntry,
//                        (uint8_t *) curSymbol.pltEntry + 16,
//                        PROT_READ | PROT_WRITE | PROT_EXEC)) {
//                    ERR_LOG("Cannot adjust memory permission");
//                    continue;
//                }
//                memcpy((uint8_t *) curSymbol.pltEntry,
//                       curSymbol.oriPltCode, 16);
//                free(curSymbol.oriPltCode);
//                curSymbol.oriPltCode = nullptr;
//            }
//        }
//        installed = false;
        return true;
    }

    HookInstaller *HookInstaller::instance = nullptr;

    struct ProxySymbol{
        std::string name="";
        void* address=nullptr;
        void** realAddressPtr=nullptr;
    };

    HookInstaller *HookInstaller::getInstance(std::string folderName) {
        if (!instance) {
            instance = new HookInstaller(std::move(folderName));
            if (!instance) {
                fatalError("Cannot allocate memory for ExtFuncCallHookAsm");
                return nullptr;
            }
            //The following arrays should be mutually exclusive!
            std::string skipSymbol[]={
                "oom","err","jump","exit","fail","verr","errx","_exit","abort","_Exit","verrx","_ZdlPv","_dl_sym","longjmp","_setjmp","_longjmp","__assert","thrd_exit",
                "__longjmp","siglongjmp","quick_exit","__chk_fail","__REDIRECT","__sigsetjmp","__do_cancel","__cxa_throw","pthread_exit","__libc_fatal","__longjmp_chk",
                "__assert_fail","__cxa_rethrow","__tls_get_addr","__pthread_exit","_startup_fatal","__ia64_longjmp","__libc_longjmp","__novmxlongjmp","nscd_run_prune",
                "main_loop_poll","__libc_message","__cxa_bad_cast","____longjmp_chk","__novmx_longjmp","nscd_run_worker","_dl_catch_error","__REDIRECT_NTHNL","__pthread_unwind",
                "_dl_fatal_printf","_dl_signal_error","__longjmp_cancel","__novmx__longjmp","_dl_allocate_tls","__call_tls_dtors","__tunable_get_val","futex_fatal_error",
                "__novmxsiglongjmp","__libc_siglongjmp","libc_hidden_proto","rtld_hidden_proto","__cxa_begin_catch","_dl_reloc_bad_type","__assert_fail_base","termination_handler",
                "receive_print_stats","_dl_catch_exception","_dl_signal_exception","__assert_perror_fail","_ZSt13get_terminatev","__cxa_free_exception","_dl_exception_create",
                "__pthread_unwind_next","__novmx__libc_longjmp","_dl_allocate_tls_init","_Unwind_RaiseException","_dl_find_dso_for_object","svctcp_rendezvous_abort",
                "_Unwind_DeleteException","svcunix_rendezvous_abort","__novmx__libc_siglongjmp","__cxa_allocate_exception","__cxa_init_primary_exception","__cxa_current_exception_type",
                "__cxa_free_dependent_exception","__cxa_allocate_dependent_exception"
            };
            ProxySymbol proxySymbol[]={
                {"fork",(void*)fork_proxy},{"dlsym",(void*)dlsym_proxy},{"dlopen",(void*)dlopen_proxy},
                {"dlvsym",(void*) dlvsym_proxy},{"dlmopen",(void*) dlmopen_proxy},
                {"cuMemFree",(void*)cuMemFree_proxy},
                {"cuMemAlloc",(void*)cuMemAlloc_proxy},
                {"cuMemAllocHost",(void*)cuMemAllocHost_proxy},
                {"cuMemHostAlloc",(void*)cuMemHostAlloc_proxy},
                {"cuMemUnmap",(void*)cuMemUnmap_proxy},
                {"cuMemcpyHtoD",(void*)cuMemcpyHtoD_proxy},
                //{"cuMemFreeHost",(void*)cuMemFreeHost_proxy},
                {"cuMemFreeAsync",(void*)cuMemFreeAsync_proxy},
                {"cuGetProcAddress",(void*)cuGetProcAddress_proxy},
                {"cuGetProcAddress_v2",(void*)cuGetProcAddress_proxy},
                {"cuMemAddressFree",(void*)cuMemAddressFree_proxy},
                {"cuMemAllocManaged",(void*)cuMemAllocManaged_proxy},
                {"pthread_create",(void*)pthread_create_proxy},
                {"cuMemHostUnregister",(void*)cuMemHostUnregister_proxy},
            #if CUDART_VERSION > 12010
                {"cuMemCreate",(void *)cuMemCreate_proxy},
                {"cuMemMap",(void *)cuMemMap_proxy},
            #endif
			#ifdef USE_TORCH
              #if TORCH_VERSION_MAJOR >= 2
                {"setMemoryFraction",(void*)setMemoryFraction_proxy},
                {"_ZN3c104cuda20CUDACachingAllocator3getEv",(void*)allocator_get_proxy,(void**)&realAllocatorGetPtr},
                {"_ZN3c104cuda20CUDACachingAllocator10raw_deleteEPv",(void*)raw_delete_proxy,(void**)&realRawDeletePtr},
                {"_ZN3c104cuda20CUDACachingAllocator14getDeviceStatsEi",nullptr,(void**)&realGetDeviceStatsPtr}
              #endif
            #endif
            };
            const ssize_t skipSymbolArrSize=sizeof(skipSymbol)/sizeof(skipSymbol[0]);
            for(int i=0;i<skipSymbolArrSize;++i){
                //INFO_LOGS("Here %s %d",skipSymbol[i].c_str(),i);
                instance->hookHintMap.insert(skipSymbol[i]);
            }
            const ssize_t proxySymbolArrSize=sizeof(proxySymbol)/sizeof(proxySymbol[0]);
            for(int i=0;i<proxySymbolArrSize;++i){
                instance->hookHintMap.insert(proxySymbol[i].name,proxySymbol[i].address, proxySymbol[i].realAddressPtr);
            }

            //Hook pytroch 2.x memory allocator
            instance->hookHintMap.insert("_ZN3c104cuda20CUDACachingAllocator9allocatorE",false,nullptr,(void**)&realPytorch2AllocatorPtr,0);
            
        }
        return instance;
    }

    HookInstaller *HookInstaller::getInstance() {
        if (!instance) {
            fatalError(
                    "Cannot create object using  this function. Users can use getInstance(std::string folderName) first and then call this function");
            return nullptr;
        }
        return instance;
    }

    HookInstaller::~HookInstaller() {
        //todo: release oriPltCode oriPltSecCode
        uninstall();
        pthread_mutex_destroy(&dynamicLoadingLock);
    }

    HookInstaller::HookInstaller(std::string folderName) : folderName(folderName), pmParser(folderName),
                                                           elfImgInfoMap(1024), allExtSymbol(1024),
                                                           hookHintMap() {
        pthread_mutexattr_t attr;
        pthread_mutexattr_init(&attr);
        pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
        pthread_mutex_init(&dynamicLoadingLock, &attr);
    }

    bool HookInstaller::makeGOTWritable(ELFSecInfo &gotSec, bool writable) {
        if (writable) {
            return adjustMemPerm(gotSec.startAddr, gotSec.startAddr + gotSec.size, PROT_READ | PROT_WRITE);
        } else {
            return adjustMemPerm(gotSec.startAddr, gotSec.startAddr + gotSec.size, PROT_READ);
        }
    }

    inline bool HookInstaller::parseRelaSection(ssize_t& validRelaEntrySize, const FunctionType& symbolType, ELFParser& parser, ELFImgInfo &curImgInfo, FileID fileId, ELFSecInfo &pltSection,
                                         ELFSecInfo &pltSecureSection, Elf64_Rela*& relaSection, ssize_t relaEntrySize, FILE *symInfoFile, uint8_t *baseAddr,
                                         uint8_t *startAddr, uint8_t *endAddr){

        //INFO_LOGS("%d: parseRelaSection now", getpid());
        for (ssize_t i = 0; i < relaEntrySize; ++i) {
            const char *funcName;
            Elf64_Word type;
            Elf64_Word bind;
            parser.getExtSymbolInfo(i, funcName, bind, type,relaSection);

            uint8_t **gotAddr = (uint8_t **) autoAddBaseAddr((uint8_t *) (parser.getRelaOffset(i,relaSection)), baseAddr, startAddr, endAddr);
            //INFO_LOGS("shouldHookThisSymbol ? id:%zd name:%s bind:%zd type:%zd addr:%p", allExtSymbol.getSize(),funcName,bind,type,gotAddr);
            SymbolHookHint retSymbolHookHint;
            shouldHookThisSymbol(funcName, bind, type,retSymbolHookHint);
            
            if(strcmp(funcName,"_ZN3c104cuda20CUDACachingAllocator9allocatorE")==0 && realPytorch2AllocatorPtr == nullptr){
                realPytorch2AllocatorPtr=(std::atomic<c10::cuda::CUDACachingAllocator::CUDAAllocator*>*)*gotAddr;
                //INFO_LOGS("Pid:%zd The address of pytorch memory allocator is %p",getpid(),realPytorch2AllocatorPtr->load());
                Pytorch2AllocatorProxy* allocatorProxy=new Pytorch2AllocatorProxy(realPytorch2AllocatorPtr->load());
                realPytorch2AllocatorPtr->store(allocatorProxy);
            }

            if(retSymbolHookHint.realAddressPtr){
                *(retSymbolHookHint.realAddressPtr)=*gotAddr;
            }

            if (!retSymbolHookHint.shouldHook) {
                //INFO_LOGS("API NOT hooked: fileId:%zd symbolId:%zd name:%s bind:%zd type:%zd addr:%p",fileId,allExtSymbol.getSize(),funcName,bind,type,gotAddr);
                continue;
            }else{
            //    INFO_LOGS("API hooked: fileId:%zd symbolId:%zd name:%s bind:%zd type:%zd addr:%p",fileId,allExtSymbol.getSize(),funcName,bind,type,gotAddr);
            }


            //Get function id from plt entry

            
            //INFO_LOGS("Symbol %d types is %d name is %s", allExtSymbol.getSize(), symbolType,funcName);

            //Make sure space is enough, if space is enough, array won't allocateArray
            FunctionInfo& newSym = allExtSymbol.pushBack();
            newSym.symbolType = symbolType;
            newSym.fileId = fileId;
            newSym.symIdInFile = i;
            newSym.initialGap = retSymbolHookHint.initialGap;
            newSym.addressOverride=retSymbolHookHint.addressOverride;
            
            if(symbolType == FunctionType::C_PLT_API) {
                uint8_t *pltSecEntry = nullptr;
                if (curImgInfo.pltSecStartAddr) {
                    pltSecEntry = curImgInfo.pltSecStartAddr + pltSecureSection.entrySize * i;
                }
                uint8_t *pltEntry = curImgInfo.pltStartAddr + pltSection.entrySize * (i + 1);

                //DBG_LOGS("curImgInfo.pltStartAddr = %p\n", curImgInfo.pltStartAddr);
                uint32_t pltStubId = parsePltStubId(pltEntry); //Note that the first entry is not valid

                newSym.realAddrPtr = gotAddr; //In PLT interpostion, the address is stored in got entry
                newSym.sharedAddr1.pltEntryAddr = pltEntry;
                newSym.pltSecEntryAddr = pltSecEntry;
                newSym.pltStubId = pltStubId;
                
                //if (addressOverride) {
                //  INFO_LOGS("%p", gotAddr);  moved to replacePLT function
                //  *newSym->realAddrPtr = reinterpret_cast<uint8_t *>(addressOverride);
                //}
            }else if(symbolType == FunctionType::C_DYN_API) {
                newSym.sharedAddr1.gotEntryAddr = gotAddr;
                
                //INFO_LOGS("Setting symbol Id gotAddr to %p",gotAddr);
                // newSym->realAddrPtr will be filled at installation time
            }

            //DBG_LOGS("%s,%ld,%ld\n", funcName, newSym->fileId, newSym->symIdInFile);
            //Write this symbol to symbol file            
            //fprintf(symInfoFile, "%s,%ld,%ld\n", funcName, newSym.fileId, newSym.symIdInFile);

            validRelaEntrySize+=1;
            //INFO_LOGS(
            //       "id:%ld funcName:%s gotAddr:%p *gotAddr:%p globalFileId:%zd symIdInFile:%zd sharedAddr1.pltEntryAddr:%p pltSecEntryAddr:%p pltStubId:%lu\n",
            //       allExtSymbol.getSize() - 1, funcName, gotAddr, *gotAddr
            //       fileId,
            //       newSym->symIdInFile, newSym->sharedAddr1.pltEntryAddr, newSym->pltSecEntryAddr, newSym->pltStubId);
        }
        return true;
    }

    bool HookInstaller::parseSymbolInfo(ELFParser &parser, ssize_t fileId, uint8_t *baseAddr,
                                        ELFSecInfo &pltSection,
                                        ELFSecInfo &pltSecureSection, ELFSecInfo &gotSec, uint8_t *startAddr,
                                        uint8_t *endAddr) {

        //assert(sizeof(ExtSymInfo) % 32 == 0); //Force memory allignment
        //INFO_LOGS("sizeof(ExtSymInfo)=%d", a);

        ELFImgInfo &curImgInfo = elfImgInfoMap[fileId];
        curImgInfo.firstSymIndex = allExtSymbol.getSize();
        //Allocate space for all rela entries in this file
        //DBG_LOGS("First sym index=%ld", curImgInfo.firstSymIndex);

        adjustMemPerm(pltSection.startAddr, pltSection.startAddr + pltSection.size, PROT_READ | PROT_WRITE | PROT_EXEC);
        adjustMemPerm(gotSec.startAddr, gotSec.startAddr + gotSec.size, PROT_READ | PROT_WRITE);
        //INFO_LOGS("gotSec %p-%p",gotSec.startAddr, gotSec.startAddr + gotSec.allocatedSize);
        
        if (pltSecureSection.startAddr) {
            // DBG_LOGS("Adjusting mem permission from:%p to:%p", pltSecureSection.internalArray,
            //         pltSecureSection.internalArray + pltSecureSection.allocatedSize);
            adjustMemPerm(pltSecureSection.startAddr, pltSecureSection.startAddr + pltSecureSection.size,
                          PROT_READ | PROT_WRITE | PROT_EXEC);
        }
        // std::stringstream ss;
        // ss << mlinsight::HookInstaller::instance->folderName << "/symbolInfo.txt";
        FILE *symInfoFile = NULL; //fopen(ss.str().c_str(), "a");
        // if (!symInfoFile) {
        //     fatalErrorS("Cannot open %s because:%s", ss.str().c_str(), strerror(errno))
        // }

        //INFO_LOGS("parser.pltEntrySize=%zd",pltSection.allocatedSize / pltSection.entrySize);
        //INFO_LOGS("parser.relaPLTEntrySize=%zd",parser.relaPLTEntrySize);
        assert(pltSection.size / pltSection.entrySize == parser.relaPLTEntrySize+1);
        
        //Install RELA PLT
        ssize_t validRelaPLTEntrySize=0;
        bool parseRelaPLTSuccess=parseRelaSection(validRelaPLTEntrySize, FunctionType::C_PLT_API, parser, curImgInfo, fileId, pltSection, pltSecureSection, parser.relaPLTSection, parser.relaPLTEntrySize, symInfoFile, baseAddr, startAddr, endAddr);
        if(!parseRelaPLTSuccess){
            ERR_LOG("Failed to parse .rela.plt");
            fclose(symInfoFile);
            return false;
        }
        validRelaPltSize+=validRelaPLTEntrySize;

        ssize_t validRelaDYNEntrySize=0;
        //Install RELA DYN
        bool parseRelaDYNSuccess=parseRelaSection(validRelaDYNEntrySize, FunctionType::C_DYN_API, parser, curImgInfo, fileId, pltSection, pltSecureSection, parser.relaDYNSection, parser.relaDYNEntrySize, symInfoFile, baseAddr, startAddr, endAddr);
        if(!parseRelaDYNSuccess){
            ERR_LOG("Failed to parse .rela.dyn");
            fclose(symInfoFile);
            return false;
        }
        validRelaDynSize+=validRelaDYNEntrySize;

        //fclose(symInfoFile);
        return true;
    }


    void HookInstaller::shouldHookThisSymbol(const char *funcName, Elf64_Word &bind, Elf64_Word &type, SymbolHookHint& retSymbolInfo) {

        std::string funcNameStr=funcName;

        //Find built-in case by name
        SymbolHookHint* symbolHookHint= hookHintMap.find(funcNameStr);
        if(symbolHookHint!=nullptr){
            retSymbolInfo=*symbolHookHint;
            //If the hint asks us to store the real funciton address, then store it.
            //INFO_LOGS("predefined function %s shouldHook? %s",funcName,symbolHookHint->shouldHook?"true":"false");
            return;
        }

        //Handle more general case
        if (funcNameStr.length() == 0) {
            //Do not hook function that does not have explicit function name
            retSymbolInfo.shouldHook=false;
            return;
        }

        if (mlinsight::strStartsWith(funcName, "__")) {
            //Do not hook function that starts with "__" as these functions are usually internal APIs
            retSymbolInfo.shouldHook=false;
            return;
        }


        bool bindCorrect=(bind==STB_GLOBAL);
        bool typeCorrect=(type==STT_FUNC);
        if (!bindCorrect || !typeCorrect) {
            //INFO_LOG("No installation");
            retSymbolInfo.shouldHook=false;
            return;
        }


        //No special handling, return directly
        //todo: we temporarily turn hook off bu default to ensure program stability
        retSymbolInfo.shouldHook=false;
        return;
    }


    bool HookInstaller::parseSecInfos(ELFParser &elfParser, ELFSecInfo &pltInfo, ELFSecInfo &pltSecInfo,
                                      ELFSecInfo &gotInfo,
                                      uint8_t *baseAddr, uint8_t *startAddr, uint8_t *endAddr) {
        Elf64_Shdr pltHdr;
        if (!elfParser.getSecHeader(SHT_PROGBITS, ".plt", pltHdr)) {
            ERR_LOG("Cannot read .plt header");
            return false;
        }
        pltInfo.size = pltHdr.sh_size;
        pltInfo.entrySize = pltHdr.sh_entsize;

        Elf64_Shdr pltSecHdr;
        pltSecInfo.entrySize = 0;
        if (elfParser.getSecHeader(SHT_PROGBITS, ".plt.sec", pltSecHdr)) {
            pltSecInfo.size = pltSecHdr.sh_size;
            pltSecInfo.entrySize = pltSecHdr.sh_entsize;
        }


        Elf64_Shdr gotHdr;
        if (!elfParser.getSecHeader(SHT_PROGBITS, ".got", gotHdr)) {
            ERR_LOG("Cannot read .got header");
            return false;
        }
        gotInfo.size = gotHdr.sh_size;
        gotInfo.entrySize = gotHdr.sh_entsize;


        pltInfo.startAddr = autoAddBaseAddr((uint8_t *) pltHdr.sh_addr, baseAddr, startAddr, endAddr);
        gotInfo.startAddr = autoAddBaseAddr((uint8_t *) gotHdr.sh_addr, baseAddr, startAddr, endAddr);

        if (pltSecInfo.entrySize > 0) {
            //Have .plt.sec table
            pltSecInfo.startAddr = autoAddBaseAddr((uint8_t *) pltSecHdr.sh_addr, baseAddr, startAddr, endAddr);
        } else {
            pltSecInfo.startAddr = nullptr;
        }

        return pltInfo.startAddr != nullptr && gotInfo.startAddr != nullptr;
    }

    //16bytes aligned. 0x90 are for alignment purpose
    uint8_t pltEntryBin[] = {0x49, 0xBB, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                             0x00, 0x00, 0x41, 0xff, 0xE3, 0x90, 0x90, 0x90};
    //32bytes aligned. 0x90 are for alignment purpose

    const int READ_TLS_PART_START = 0;
    const int COUNT_TLS_ARR_ADDR = READ_TLS_PART_START + 2;

    const int COUNTING_PART_START = READ_TLS_PART_START + 20;
    const int REC_ARRAY_OFFSET1 = COUNTING_PART_START + 5, DYMAIC_LOADING_OFFSET1_SIZE = 32;
    const int COUNT_OFFSET1 = COUNTING_PART_START + 12, COUNT_OFFSET1_SIZE = 32;
    const int COUNT_OFFSET2 = COUNTING_PART_START + 23, COUNT_OFFSET2_SIZE = 32;
    const int GAP_OFFSET = COUNTING_PART_START + 30, GAP_OFFSET_SIZE = 32;

    const int SKIP_PART_START = COUNTING_PART_START + 45;
    const int GOT_ADDR = SKIP_PART_START + 2, GOT_ADDR_SIZE = 64;
    const int CALL_LD_INST = SKIP_PART_START + 13;
    const int PLT_STUB_ID = SKIP_PART_START + 14, PLT_STUB_ID_SIZE = 32;
    const int PLT_START_ADDR = SKIP_PART_START + 20, PLT_START_ADDR_SIZE = 64;

    const int LDARR_OFFSET_IN_CONTEXT = 0x628;
    const int INTERNALARR_OFFSET_IN_LDARR = 0x18;
    const int COUNT_OFFSET_IN_RECARR = 0x10;
    const int GAP_OFFSET_IN_RECARR = 0x18;

    const int TIMING_PART_START = SKIP_PART_START + 31;
    const int LOW_BITS_GOTENTRYADDR = TIMING_PART_START + 4;
    const int HIGH_BITS_GOTENTRYADDR = TIMING_PART_START + 12;
    const int SYM_ID = TIMING_PART_START + 21, FUNC_ID_SIZE = 32;
    const int ASM_HOOK_HANDLER_ADDR = TIMING_PART_START + 27, ASM_HOOK_HANDLER_ADDR_SIZE = 64;



    uint32_t HookInstaller::parsePltStubId(uint8_t *dest) {
        int pushOffset = -1;
        if (*dest == 0xFF || *dest==0xCC) {
            pushOffset = 7;
        } else if (*dest == 0xF3) {
            pushOffset = 5;
        } else {
            fatalError("Plt entry format illegal. Cannot find instruction \"push id\"");
        }

        //Debug tips: Add breakpoint after this statement, and *pltStubId should be 0 at first, 2 at second .etc
        uint32_t *pltStubId = reinterpret_cast<uint32_t *>(dest + pushOffset);
        return *pltStubId;
    }

    bool HookInstaller::fillAddr2pltEntry(uint8_t *funcAddr, uint8_t *retPltEntry) {
        //Copy code
        memcpy(retPltEntry, pltEntryBin, sizeof(pltEntryBin));
        //Copy address
        assert(sizeof(uint8_t **) == 8);
        memcpy(retPltEntry + 2, &funcAddr, sizeof(uint8_t **));
        return true;
    }

    bool HookInstaller::fillAddrAndSymId2IdSaver(uint8_t **realAddrPtr, uint8_t *firstPltEntry, uint32_t symId,
                                                 uint32_t pltStubId, uint32_t recArrayOffset,
                                                 uint32_t countOffset, uint32_t gapOffset, uint8_t *idSaverEntry) {



        assert(sizeof(uint8_t **) == 8);

        memcpy(idSaverEntry + COUNT_TLS_ARR_ADDR, tlsOffset, sizeof(void *));

        //Fill TLS offset (Address filled directly i
        memcpy(idSaverEntry + REC_ARRAY_OFFSET1, &recArrayOffset, sizeof(uint32_t));
        memcpy(idSaverEntry + COUNT_OFFSET1, &countOffset, sizeof(uint32_t));
        memcpy(idSaverEntry + COUNT_OFFSET2, &countOffset, sizeof(uint32_t));

        memcpy(idSaverEntry + GAP_OFFSET, &gapOffset, sizeof(uint32_t));

        //Fill got address
        memcpy(idSaverEntry + GOT_ADDR, &realAddrPtr, sizeof(uint8_t **));
        //Fill function id
        memcpy(idSaverEntry + PLT_STUB_ID, &pltStubId, sizeof(uint32_t));
        //Fill first plt address
        memcpy(idSaverEntry + PLT_START_ADDR, &firstPltEntry, sizeof(uint8_t *));

        //INFO_LOG("Here");

        uint32_t realAddrPtrHi = ((uint64_t) realAddrPtr) >> 32;
        uint32_t realAddrPtrLo = ((uint64_t) realAddrPtr) & 0xffffffff;
        //INFO_LOGS("GOT_ADDR=%p", realAddrPtr);
        //INFO_LOGS("GOT_HI=0x%x GOT_LOW", realAddrPtrHi);
        //INFO_LOGS("GOT_LO=0x%x GOT_LOW", realAddrPtrLo);

        memcpy(idSaverEntry + LOW_BITS_GOTENTRYADDR, &realAddrPtrLo, sizeof(uint32_t));
        memcpy(idSaverEntry + HIGH_BITS_GOTENTRYADDR, &realAddrPtrHi, sizeof(uint32_t));

        //INFO_LOG("Fill Symbol Id");

        //Fill symId
        memcpy(idSaverEntry + SYM_ID, &symId, sizeof(uint32_t));

        //INFO_LOG("Fill asmTimingHandler");

        uint8_t *asmHookPtr = (uint8_t *) &asmTimingHandler;
        //Fill asmTimingHandler
        memcpy(idSaverEntry + ASM_HOOK_HANDLER_ADDR, (void *) &asmHookPtr, sizeof(void *));
        //INFO_LOG("Here");

        return true;
    }


    inline void HookInstaller::parseRequiredInfo() {
        //INFO_LOGS("thread:%p pthread_mutex_lock(&inst->dynamicLoadingLock)",pthread_self());

        pthread_mutex_lock(&dynamicLoadingLock);
        //Initialize existing loading id

        ELFParser elfParser;
        if (!pmParser.parsePMMap()) {
            fatalError("Cannot parsePmMap");
        }
        //Find new file from exising PMMaps
        Array<FileID> newFileEntryId;
        pmParser.getNewFileEntryIds(newFileEntryId, true);

        if(tlsOffset == nullptr){
            parseTLSOffset();
        }

        for(ssize_t i=elfImgInfoMap.getSize();i<pmParser.getFileEntryArraySize();++i){
            elfImgInfoMap.pushBack();
        }

        //print_stacktrace();
        //print_pystacktrace();

        //Get segment info from /proc/self/maps
        for (ssize_t fileId = 0; fileId < newFileEntryId.getSize(); ++fileId) {
            FileID globalFileId = newFileEntryId[fileId];
            //INFO_LOGS("globalFileId=%zd", globalFileId);
            FileEntry &curFileEntry = pmParser.getFileEntry(globalFileId);
            const char *curFilePathName = pmParser.getStr(curFileEntry.pathNameStartIndex);
            //DBG_LOGS("Install newly discovered file:%s fileId:%zd", curFilePathName, globalFileId);
            ELFImgInfo& curElfImgInfo = elfImgInfoMap[globalFileId];
            if (elfParser.parse(curFilePathName)) {
                //Find the entry allocatedSize of plt and got
                ELFSecInfo pltInfo{};
                ELFSecInfo pltSecInfo{};
                ELFSecInfo gotInfo{};

                //todo: We assume plt and got entry allocatedSize is the same.
                if (!parseSecInfos(elfParser, pltInfo, pltSecInfo, gotInfo, curFileEntry.baseStartAddr,
                                   curFileEntry.baseStartAddr, curFileEntry.baseEndAddr)) {
                    fatalError("Failed to parse plt related sections.");
                    exit(-1);
                }
                curElfImgInfo.pltStartAddr = pltInfo.startAddr;
                curElfImgInfo.pltSecStartAddr = pltSecInfo.startAddr;
                curElfImgInfo.gotStartAddr = gotInfo.startAddr;

                //INFO_LOGS("CurFileName=%s",curFilePathName);
                //Install hook on this file
                if (!parseSymbolInfo(elfParser, globalFileId, curFileEntry.baseStartAddr, pltInfo, pltSecInfo,
                                     gotInfo, curFileEntry.baseStartAddr,
                                     curFileEntry.baseEndAddr)) {
                    fatalErrorS("installation for file %s failed.", curFilePathName);
                    exit(-1);
                }
                curElfImgInfo.valid = true;
            }
        }
        //INFO_LOGS("thread:%p pthread_mutex_unlock(&inst->dynamicLoadingLock)",pthread_self());

        pthread_mutex_unlock(&dynamicLoadingLock);
    }

    void HookInstaller::parseTLSOffset(){
        ssize_t scalerFileId = pmParser.getMLInsightFileId();
        assert(scalerFileId != -1);
        ELFParser parser;
        FileEntry& curFileEntry=pmParser.getFileEntry(scalerFileId);
        parser.parse(pmParser.getStr(curFileEntry.pathNameStartIndex));
        
        

        uint8_t *baseAddr = curFileEntry.baseStartAddr;
        uint8_t *endAddr = curFileEntry.baseEndAddr;
        ELFSecInfo pltInfo{};
        ELFSecInfo pltSecInfo{};
        ELFSecInfo gotInfo{};

        if(!parseSecInfos(parser, pltInfo, pltSecInfo, gotInfo,baseAddr,baseAddr,endAddr)){
            fatalError("Failed to parse TLS offset related sections.");
            exit(-1);
        }

        for (ssize_t i = 0; i < parser.relaDYNEntrySize; ++i){
            const char* funcName;
            Elf64_Word type;
            Elf64_Word bind;
            parser.getExtSymbolInfo(i, funcName, bind, type, parser.relaDYNSection);
            ssize_t stringLength = strlen(funcName);

            if(stringLength==26 && strncmp(funcName,"_ZN9mlinsight10curContextE",stringLength)==0){
                tlsOffset = (uint8_t **) autoAddBaseAddr((uint8_t *)(parser.getRelaOffset(i,parser.relaDYNSection)),baseAddr, baseAddr, endAddr);
                //INFO_LOGS("Parse TLSOffset is successful %p %p",tlsOffset,*tlsOffset);
                break;
            }
        }
    }

    bool HookInstaller::replacePltEntry() {
        //Allocate callIdSaver
        //todo: memory leak
        if(allExtSymbol.getSize()<=installedSymbolSize) {
            //ERR_LOGS("No new symbols discovered but replacePltEntry was invoked anyways. allExtSymbol.getSize()=%zd, installedSymbolSize=%zd",
            //        allExtSymbol.getSize(),
            //        installedSymbolSize);
            return false;
        }
        uint8_t*& relaIdSaver=relaIdSavers.pushBack();
        //INFO_LOGS("Install %zd symbols", allExtSymbol.getSize()-installedSymbolSize);
        relaIdSaver = static_cast<uint8_t *>(mmap(NULL, (allExtSymbol.getSize()-installedSymbolSize) * ID_SAVER_BIN_SIZE,
                                                   PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        //Allocate relaDynRealAddrSaver
        //todo: memory leak
        uint8_t** relaDynRealAddrSaver=nullptr;
        if(validRelaDynSize<=installedRelaDynSize) {
            //DBG_LOG("No new DYN symbols discovered. Will not install this part");
        }else{
            uint8_t**& _relaDynRealAddrSaver=relaDynRealAddrSavers.pushBack();
            _relaDynRealAddrSaver=static_cast<uint8_t **>(mmap(NULL, (validRelaDynSize-installedRelaDynSize) * sizeof(uint8_t *),
                                                   PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
            relaDynRealAddrSaver=_relaDynRealAddrSaver;
        }
        
        /**
         * Prepare callIdSaver
         */
        uint8_t *curRelaIdSaver = relaIdSaver;
        uint8_t **curRelaDynRealAddrSaver = relaDynRealAddrSaver;
        IdSaverBinWrapper idSaverBinWrapper;//Install this in order to get 
        //Fill address and ids in callIdSaver
        ssize_t cachedInstalledSymbolSize=installedSymbolSize; //Only install newly loaded API
        for (int curSymId = cachedInstalledSymbolSize; curSymId < allExtSymbol.getSize(); ++curSymId) {
            //Fetch symbol info
            FunctionInfo &curSymInfo = allExtSymbol[curSymId];
            ELFImgInfo &curImgInfo = elfImgInfoMap[curSymInfo.fileId];
            
            if (curSymInfo.symbolType == FunctionType::C_DYN_API){
                curSymInfo.realAddrPtr=curRelaDynRealAddrSaver; //Set realAddrPtr to newly allocated address storage location
                curRelaDynRealAddrSaver+=1;
                *curSymInfo.realAddrPtr=*curSymInfo.sharedAddr1.gotEntryAddr;
            }

            //Set realAddrPtr to correct place
            if(curSymInfo.addressOverride){
                //INFO_LOGS("Symbol %d addressOverrided at %p from %p to %p ",curSymId, curSymInfo.realAddrPtr,*curSymInfo.realAddrPtr,curSymInfo.addressOverride);
                *curSymInfo.realAddrPtr = reinterpret_cast<uint8_t *>(curSymInfo.addressOverride);
                //INFO_LOGS("I mean really *%p=%p",curSymInfo.realAddrPtr,*curSymInfo.realAddrPtr);
            }

            //Place pltStubId to the idSaverBin
            memcpy(curRelaIdSaver, idSaverBinWrapper.idSaverBin, ID_SAVER_BIN_SIZE);
            //INFO_LOGS("Install symbol %d curSymInfo.realAddrPtr=%p *curSymInfo.realAddrPtr=%p"
            //,curSymId,curSymInfo.realAddrPtr,*curSymInfo.realAddrPtr);
            if (!fillAddrAndSymId2IdSaver(curSymInfo.realAddrPtr,
                                          curImgInfo.pltStartAddr,
                                          curSymId,
                                          curSymInfo.pltStubId,
                                          LDARR_OFFSET_IN_CONTEXT + INTERNALARR_OFFSET_IN_LDARR,
                                          curSymId * sizeof(RecTuple) + COUNT_OFFSET_IN_RECARR,
                                          curSymId * sizeof(RecTuple) + GAP_OFFSET_IN_RECARR,
                                          curRelaIdSaver)) {
                fatalError("fillAddrAndSymId2IdSaver failed, this should not happen");
            }
            curRelaIdSaver += ID_SAVER_BIN_SIZE;
        }
        installedSymbolSize = allExtSymbol.getSize();


        /**
         * Replace plt entry or replace .plt (Or directly replace .plt.sec)
         */
        curRelaIdSaver = relaIdSaver;
        for (int curSymId = cachedInstalledSymbolSize; curSymId < allExtSymbol.getSize(); ++curSymId) {
            FunctionInfo &curSym = allExtSymbol[curSymId];
            ELFImgInfo &curImgInfo = elfImgInfoMap[curSym.fileId];

            if (curSym.symbolType == FunctionType::C_PLT_API){
                //todo: Check symbol type
                if (curSym.pltSecEntryAddr) {
                    //Replace .plt.sec
                    if (!fillAddr2pltEntry(curRelaIdSaver, curSym.pltSecEntryAddr)) {
                        fatalError("pltSecAddr installation failed, this should not happen");
                    }
                } else {
                    //Replace .plt
                    if (!fillAddr2pltEntry(curRelaIdSaver, curSym.sharedAddr1.pltEntryAddr)) {
                        fatalError("pltEntry installation failed, this should not happen");
                    }
                }
                //Replace got entry, 16 is the allocatedSize of a plt entry
                if (abs(curSym.sharedAddr1.pltEntryAddr - *curSym.realAddrPtr) < 16) {
                    //Address not resolved, perform lazy loading and call ld.so
                    *curSym.realAddrPtr = curRelaIdSaver + CALL_LD_INST;
                }
            } else if (curSym.symbolType == FunctionType::C_DYN_API){
                //Replace got directly to relaIdSaver
                //INFO_LOGS("curRelaIdSaver=%p",curRelaIdSaver);
                adjustMemPerm(curSym.sharedAddr1.gotEntryAddr, curSym.sharedAddr1.gotEntryAddr+1,
                              PROT_READ | PROT_WRITE);
                /*INFO_LOGS("C_DYN_API replace SumId=%zd *%p= from %p to %p realIdPtr=%p",curSymId,curSym.sharedAddr1.gotEntryAddr,
                    *curSym.sharedAddr1.gotEntryAddr,
                    curRelaIdSaver,
                    curSym.realAddrPtr
                );*/
                *curSym.sharedAddr1.gotEntryAddr=curRelaIdSaver;
            } else {
                fatalErrorS("Impossible case. Symbol type %d should not appear here",curSym.symbolType);
            }

           

            curRelaIdSaver += ID_SAVER_BIN_SIZE;
        }

        //DBG_LOG("replace PLT finished");
        return true;
    }

    void HookInstaller::createRecordingFolder() const {
        //sprintf(folderName, "mlinsightdata_%lu", getunixtimestampms());
        //The timing code is blocked so there is no need to create recording folder
        //if (mkdir(folderName.c_str(), 0755) == -1) {
        //    fatalErrorS("Cannot mkdir ./%s because: %s", folderName.c_str(),
        //                strerror(errno));
        //}
    }

    /**
     * Intercept and install dlsym
     * @return successful or not
     */
    bool HookInstaller::installDlSym(void *realFuncAddr, void*& retAddr) {
        if(realFuncAddr){
            HookContext* hookContextPtr=curContext;
            //realFuncAddr is valid
            //todo: what if two APIs has the same address?
            auto findIter=dlsymRealAddrGOTEntryMap.find(realFuncAddr);
            if(findIter==dlsymRealAddrGOTEntryMap.end()){
                //Should install
                IdSaverBinWrapper* idSaverBinWrapper= dlSymIdSavers.alloc();
                new (idSaverBinWrapper) IdSaverBinWrapper(); //Allocate idsaver
                adjustMemPerm(idSaverBinWrapper->idSaverBin, idSaverBinWrapper->idSaverBin + ID_SAVER_BIN_SIZE,
                              PROT_READ | PROT_WRITE | PROT_EXEC);
                idSaverBinWrapper->realFuncAddr=realFuncAddr;

                //todo: Also record symbol entry
    

                ssize_t newSymId=hookContextPtr->recordArray.getSize();
                hookContextPtr->recordArray.pushBack(); //Insert new entry to record dlsym loaded symbol

                if(!fillAddrAndSymId2IdSaver((uint8_t**)&idSaverBinWrapper->realFuncAddr, NULL, newSymId, 0,
                                         LDARR_OFFSET_IN_CONTEXT + INTERNALARR_OFFSET_IN_LDARR,
                                         newSymId * sizeof(RecTuple) + COUNT_OFFSET_IN_RECARR,
                                         newSymId * sizeof(RecTuple) + GAP_OFFSET_IN_RECARR,
                                         idSaverBinWrapper->idSaverBin)){
                    fatalError("fillAddrAndSymId2IdSaver failed, this should not happen");
                }
                //Save idSaver address to dlsymRealAddrGOTEntryMap
                dlsymRealAddrGOTEntryMap[realFuncAddr]=idSaverBinWrapper->idSaverBin;
                retAddr=idSaverBinWrapper->idSaverBin;

                //INFO_LOGS("thread:%p pthread_mutex_lock(&inst->dynamicLoadingLock)",pthread_self());

                pthread_mutex_lock(&dynamicLoadingLock);
                FunctionInfo& symInfo=instance->allExtSymbol.pushBack();
                installedSymbolSize += 1;
                symInfo.symbolType=FunctionType::C_DL_API;
                
                //INFO_LOGS("thread:%p pthread_mutex_unlock(&inst->dynamicLoadingLock)",pthread_self());

                pthread_mutex_unlock(&dynamicLoadingLock);

                return true;
            }else{
                retAddr=findIter->second;
                return true;
            }
        }else{
            ERR_LOG("dlsym returns null");
        }
        return false;
    }


}

#endif
