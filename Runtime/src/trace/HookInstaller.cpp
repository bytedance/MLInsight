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
#include "trace/hook/PyHook.h"

#ifdef USE_TORCH

#include "common/DependencyLibVersionSpecifier.h"
#include "trace/proxy/PytorchMemProxy.h"

#endif

#ifdef TENSOR_FLOW

#endif


#ifdef CUDA_ENABLED

#include "trace/proxy/CUDAProxy.h"
#include "trace/hook/HookInstaller.h"
#include "trace/proxy/TensorflowMemProxy.h"

#include <cxxabi.h>

#endif


namespace mlinsight {
    bool installed = false;
    void *libc10_cuda_text_begin = nullptr;
    void *libc10_cuda_text_end = nullptr;
    void *pythonInterpreter_text_begin = nullptr;
    void *pythonInterpreter_text_end = nullptr;

    //Used in other performance sensitive C code to avoid an extra function call.
    HookInstaller *hookInstallerInstance = nullptr;

    bool HookInstaller::install() {
        //tensorflow::checkTFVersion("/workspace/user/pilot_service/libtensorflow_cc.so.1");
        DBG_LOG("*******HookInstaller::install*****");

        createRecordingFolder();

        if (!initTLS()) {
            ERR_LOG("Failed to initialize TLS");
            //This is the main thread
            return false;
        }


        initLogicalClock(curContext->cachedWallClockSnapshot, curContext->cachedLogicalClock,
                         curContext->cachedThreadNum);
        __cxxabiv1::__cxa_atexit([](void *) -> void { saveData(curContext); }, NULL, NULL);

        if (isPythonAvailable()) {
            installBeforePythonInit();
        }

        //Register datasaver hook
        return installOnDlOpen();
    }


    bool HookInstaller::installOnDlOpen() {
        //DBG_LOGS("Install with loadingId=%zd", loadingId);
        //DBG_LOG("Install DlOpen");
        parseRequiredInfo();

        //DBG_LOG("Replace PLT entry");
        bool installationRet=replaceEntries();

        if(pytorch::onHookInstallationFinished){
            pytorch::onHookInstallationFinished();
        }
        if(tensorflow::onHookInstallationFinished){
            tensorflow::onHookInstallationFinished();
        }
        return installationRet;
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
//                    ERR_LOG("Cannot adjust driverMemRecord permission");
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
//                    ERR_LOG("Cannot adjust driverMemRecord permission");
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


    HookInstaller *HookInstaller::getInstance(std::string folderName) {
        if (!instance) {
            instance = new HookInstaller(std::move(folderName));
            if (!instance) {
                fatalError("Cannot allocate driverMemRecord for HookInstaller");
                return nullptr;
            }
            hookInstallerInstance = instance;


            //The following arrays should be mutually exclusive!
            std::string skipSymbol[] = {
                    "oom", "err", "jump", "exit", "fail", "verr", "errx", "_exit", "abort", "_Exit", "verrx", "_ZdlPv",
                    "_dl_sym", "longjmp", "_setjmp", "_longjmp", "__assert", "thrd_exit",
                    "__longjmp", "siglongjmp", "quick_exit", "__chk_fail", "__REDIRECT", "__sigsetjmp", "__do_cancel",
                    "__cxa_throw", "pthread_exit", "__libc_fatal", "__longjmp_chk",
                    "__assert_fail", "__cxa_rethrow", "__tls_get_addr", "__pthread_exit", "_startup_fatal",
                    "__ia64_longjmp", "__libc_longjmp", "__novmxlongjmp", "nscd_run_prune",
                    "main_loop_poll", "__libc_message", "__cxa_bad_cast", "____longjmp_chk", "__novmx_longjmp",
                    "nscd_run_worker", "_dl_catch_error", "__REDIRECT_NTHNL", "__pthread_unwind",
                    "_dl_fatal_printf", "_dl_signal_error", "__longjmp_cancel", "__novmx__longjmp", "_dl_allocate_tls",
                    "__call_tls_dtors", "__tunable_get_val", "futex_fatal_error",
                    "__novmxsiglongjmp", "__libc_siglongjmp", "libc_hidden_proto", "rtld_hidden_proto",
                    "__cxa_begin_catch", "_dl_reloc_bad_type", "__assert_fail_base", "termination_handler",
                    "receive_print_stats", "_dl_catch_exception", "_dl_signal_exception", "__assert_perror_fail",
                    "_ZSt13get_terminatev", "__cxa_free_exception", "_dl_exception_create",
                    "__pthread_unwind_next", "__novmx__libc_longjmp", "_dl_allocate_tls_init", "_Unwind_RaiseException",
                    "_dl_find_dso_for_object", "svctcp_rendezvous_abort",
                    "_Unwind_DeleteException", "svcunix_rendezvous_abort", "__novmx__libc_siglongjmp",
                    "__cxa_allocate_exception", "__cxa_init_primary_exception", "__cxa_current_exception_type",
                    "__cxa_free_dependent_exception", "__cxa_allocate_dependent_exception"
            };
            ProxySymbol proxySymbol[] = {
                    {"fork", (void *) fork_proxy},
                    //{"dlsym", (void *) dlsym_proxy},
                    // {"dlopen", (void *) dlopen_proxy},
                    //{"dlvsym", (void *) dlvsym_proxy},
                    // {"dlmopen", (void *) dlmopen_proxy},
                    {"dlclose", (void *) dlclose_proxy},
                    {"cuMemFree", (void *) cuMemFree_proxy},
                    {"cuMemFree_v2", (void *) cuMemFree_proxy},
                    {"cuMemAlloc", (void *) cuMemAlloc_proxy},
                    {"cuMemAlloc_v2", (void *) cuMemAlloc_proxy},
                    {"cuGetProcAddress", (void *) cuGetProcAddress_proxy},
                    {"cuGetProcAddress_v2", (void *) cuGetProcAddress_proxy},
                    {"pthread_create", (void *) pthread_create_proxy},
            };
            const ssize_t skipSymbolArrSize = sizeof(skipSymbol) / sizeof(skipSymbol[0]);
            for (int i = 0; i < skipSymbolArrSize; ++i) {
                //INFO_LOGS("Here %s %d",skipSymbol[i].c_str(),i);
                instance->hookHintMap.insert(std::make_pair(skipSymbol[i], SymbolHookHint()));
            }
            const ssize_t proxySymbolArrSize = sizeof(proxySymbol) / sizeof(proxySymbol[0]);
            for (int i = 0; i < proxySymbolArrSize; ++i) {
                instance->hookHintMap.insert(std::make_pair(proxySymbol[i].name, SymbolHookHint(proxySymbol[i].address,
                                                                                                proxySymbol[i].realAddressPtr)));
            }

            // Perform special handling of dlsym 
            instance->hookHintMap.insert(std::make_pair("dlsym", SymbolHookHint(true, nullptr, nullptr, 0,SymbolSpecialHandlingMarker::DLSYM_RTLD_NEXT_BYPASS)));
            instance->hookHintMap.insert(std::make_pair("dlvsym", SymbolHookHint(true, nullptr, nullptr, 0,SymbolSpecialHandlingMarker::DLVSYM_RTLD_NEXT_BYPASS)));



            if(pytorch::onSettingHookHint){
                pytorch::onSettingHookHint(instance->hookHintMap);
            }
            if(tensorflow::onSettingHookHint){
                tensorflow::onSettingHookHint(instance->hookHintMap);
            }

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

    HookInstaller::HookInstaller(std::string folderName) : folderName(folderName),
                                                           elfImgInfoMap(1024), allExtSymbol(1024),
                                                           hookHintMap(),dlsymJumperWrapper((void*)&dlsym_proxy,(void*)&dlsym),
                                                           dlvsymJumperWrapper((void*)&dlvsym_proxy,(void*)&dlvsym)
                                                            { 
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

    inline bool
    HookInstaller::parseRelaSection(ssize_t &validRelaEntrySize, const APIType &symbolType, ELFParser &parser,
                                    ELFImgInfo &curImgInfo, FileID fileId, ELFSecInfo &pltSection,
                                    ELFSecInfo &pltSecureSection, Elf64_Rela *&relaSection, ssize_t relaEntrySize,
                                    FILE *symInfoFile, uint8_t *baseAddr,
                                    uint8_t *startAddr, uint8_t *endAddr) {

        //INFO_LOGS("%d: parseRelaSection now", getpid());
        for (ssize_t i = 0; i < relaEntrySize; ++i) {
            const char *funcName;
            Elf64_Word type;
            Elf64_Word bind;
            parser.getExtSymbolInfo(i, funcName, bind, type, relaSection);


            uint8_t **gotAddr = (uint8_t **) autoAddBaseAddr((uint8_t *) (parser.getRelaOffset(i, relaSection)),
                                                             baseAddr, startAddr, endAddr);
//            INFO_LOGS("shouldHookThisSymbol ? id:%zd name:%s bind:%zd type:%zd addr:%p", allExtSymbol.getSize(),funcName,bind,type,gotAddr);
            SymbolHookHint retSymbolHookHint;
            shouldHookThisSymbol(funcName, bind, type, retSymbolHookHint);

            /**
             * Perform special hanling of certain APIs
            */
            if (retSymbolHookHint.realAddressPtr) {
                //The user requires mlinsight to store the real address of a symbol to another pointer.
                *(retSymbolHookHint.realAddressPtr) = *gotAddr;
            }

            if (!retSymbolHookHint.shouldHook) {
                //INFO_LOGS("API NOT hooked: symbolId:%zd name:%s bind:%zd type:%zd addr:%p",allExtSymbol.getSize(),funcName,bind,type,gotAddr);
                continue;
            }else{
                //INFO_LOGS("API hooked: symbolId:%zd name:%s bind:%zd type:%zd addr:%p",allExtSymbol.getSize(),funcName,bind,type,gotAddr);
            }

            //Get function id from plt entry
            INFO_LOGS("File %s,Symbol %ld name is %s",pmParser.getFileEntry(fileId).filePath.c_str(), allExtSymbol.getSize(),funcName);

            uint8_t *curGotDesk = *gotAddr;
            ssize_t symbolNumber = allExtSymbol.getSize();

            //Make sure space is enough, if space is enough, array won't allocateArray
            APICallInfo &newSym = allExtSymbol.pushBack();
            newSym.callerFileId = fileId; //For .rela.plt and .rela.dyn APIs, callerFileId is known at the parsing time.
            newSym.symIdInFile = i;
            newSym.initialGap = retSymbolHookHint.initialGap;
            newSym.specialHandlingMarker=retSymbolHookHint.specialHandlingMarker;
            newSym.apiType = symbolType;
            newSym.addressOverride = retSymbolHookHint.addressOverride;
            

            if (symbolType == APIType::C_PLT_API) {
                assert(curImgInfo.pltStartAddr != nullptr);

                uint8_t *pltSecEntry = nullptr;
                if (curImgInfo.pltSecStartAddr) {
                    pltSecEntry = curImgInfo.pltSecStartAddr + pltSecureSection.entrySize * i;
                }
                uint8_t *pltEntry = curImgInfo.pltStartAddr + pltSection.entrySize * (i + 1);

                //DBG_LOGS("curImgInfo.pltStartAddr = %p\n", curImgInfo.pltStartAddr);
                uint32_t pltStubId = parsePltStubId(pltEntry); //Note that the first entry is not valid

                newSym.realAddrPtr = gotAddr; //In PLT interpostion, the address is stored in got entry
                newSym.gotEntryAddr = gotAddr;
                newSym.pltEntryAddr = pltEntry;
                newSym.pltSecEntryAddr = pltSecEntry;
                newSym.pltStubId = pltStubId;
            } else if (symbolType == APIType::C_DYN_API) {
                newSym.gotEntryAddr = gotAddr;
            }

            fprintf(symInfoFile, "%s,%ld\n", funcName, newSym.callerFileId);

            validRelaEntrySize += 1;
            //INFO_LOGS(
            //       "id:%ld funcName:%s gotAddr:%p *gotAddr:%p calleeFileId:%zd symIdInFile:%zd sharedAddr1.pltEntryAddr:%p pltSecEntryAddr:%p pltStubId:%lu\n",
            //       allExtSymbol.getSize() - 1, funcName, gotAddr, *gotAddr
            //       callerFileId,
            //       newSym->symIdInFile, newSym->sharedAddr1.pltEntryAddr, newSym->pltSecEntryAddr, newSym->pltStubId);
        }
        return true;
    }

    bool HookInstaller::parseSymbolInfo(ELFParser &parser, ssize_t fileId, FileEntry &fileEntry,
                                        ELFSecInfo &pltSection,
                                        ELFSecInfo &pltSecureSection, ELFSecInfo &gotSec,ELFSecInfo &gotPltSec) {

        //assert(sizeof(ExtSymInfo) % 32 == 0); //Force driverMemRecord allignment
        //INFO_LOGS("sizeof(ExtSymInfo)=%d", a);

        ELFImgInfo &curImgInfo = elfImgInfoMap[fileId];
        curImgInfo.firstSymIndex = allExtSymbol.getSize();
        //Allocate space for all rela entries in this file
        //DBG_LOGS("First sym index=%ld", curImgInfo.firstSymIndex);
        if(pltSection.startAddr){
            adjustMemPerm(pltSection.startAddr, pltSection.startAddr + pltSection.size, PROT_READ | PROT_WRITE | PROT_EXEC);
        }
        if(gotSec.startAddr){
            adjustMemPerm(gotSec.startAddr, gotSec.startAddr + gotSec.size, PROT_READ | PROT_WRITE);
            //INFO_LOGS("Assign writable permission to GOT. gotSec %p-%p gotSec.size=%u",gotSec.startAddr, gotSec.startAddr + gotSec.size,gotSec.size);
        }
        if(gotPltSec.startAddr){
            //INFO_LOGS("Assign writable permission to .got.plt. gotPltSec %p-%p gotPltSec.size=%u",gotPltSec.startAddr, gotPltSec.startAddr + gotPltSec.size,gotPltSec.size);
            adjustMemPerm(gotPltSec.startAddr, gotPltSec.startAddr + gotPltSec.size, PROT_READ | PROT_WRITE);
        }
        if (pltSecureSection.startAddr) {
            // DBG_LOGS("Adjusting mem permission from:%p to:%p", pltSecureSection.internalArray,
            //         pltSecureSection.internalArray + pltSecureSection.allocatedSize);
            adjustMemPerm(pltSecureSection.startAddr, pltSecureSection.startAddr + pltSecureSection.size,
                          PROT_READ | PROT_WRITE | PROT_EXEC);
        }

        uint8_t *baseAddr = (uint8_t *) pmParser.getPmEntry(fileEntry.pmEntryRange[0].first).addrStart;
        uint8_t *endAddr = (uint8_t *) pmParser.getPmEntry(fileEntry.pmEntryRange.back().second).addrEnd;

//        INFO_LOGS("parser.pltEntrySize=%zd",pltSection.size / pltSection.entrySize);
//        INFO_LOGS("parser.relaPLTEntrySize=%zd",parser.relaPLTEntrySize);
//        assert(pltSection.size / pltSection.entrySize == parser.relaPLTEntrySize + 1);


        //Install RELA PLT
        ssize_t validRelaPLTEntrySize = 0;
        bool parseRelaPLTSuccess = parseRelaSection(validRelaPLTEntrySize, APIType::C_PLT_API, parser, curImgInfo,
                                                    fileId, pltSection, pltSecureSection, parser.relaPLTSection,
                                                    parser.relaPLTEntrySize, nativeAPIInfoFile, baseAddr, baseAddr,
                                                    endAddr);


        ssize_t validRelaDYNEntrySize = 0;
        //Install RELA DYN
        bool parseRelaDYNSuccess = parseRelaSection(validRelaDYNEntrySize, APIType::C_DYN_API, parser, curImgInfo,
                                                    fileId, pltSection, pltSecureSection, parser.relaDYNSection,
                                                    parser.relaDYNEntrySize, nativeAPIInfoFile, baseAddr, baseAddr,
                                                    endAddr);
        if (!parseRelaPLTSuccess && !parseRelaDYNSuccess) {
            return false;
        }
        if(parseRelaPLTSuccess){
            if(pltSection.entrySize == 0 || pltSection.size / pltSection.entrySize != parser.relaPLTEntrySize + 1){
                DBG_LOGS(".rela.plt is not empty, but the pltSection for %s is empty. This is most likely be normal.",fileEntry.filePath.c_str());
                return false;
            }
        }

        //Image is valid
        return true;
    }


    void HookInstaller::shouldHookThisSymbol(const char *funcName, Elf64_Word &bind, Elf64_Word &type,
                                             SymbolHookHint &retSymbolInfo) {

        std::string funcNameStr = funcName;

        auto hookHintIterator = hookHintMap.find(funcNameStr);
        //Find built-in case by name
        if (hookHintIterator != hookHintMap.end()) {
            retSymbolInfo = hookHintIterator->second;
            //If the hint asks us to store the real funciton address, then store it.
            //INFO_LOGS("predefined function %s shouldHook? %s",funcName,symbolHookHint->shouldHook?"true":"false");
            return;
        }

        //Handle more frameworkGeneral case
        if (funcNameStr.length() == 0) {
            //Do not hook function that does not have explicit function name
            retSymbolInfo.shouldHook = false;
            return;
        }

        if (mlinsight::strStartsWith(funcName, "__")) {
            //Do not hook function that starts with "__" as these functions are usually internal APIs
            retSymbolInfo.shouldHook = false;
            return;
        }


        bool bindCorrect = (bind == STB_GLOBAL);
        bool typeCorrect = (type == STT_FUNC);
        if (!bindCorrect || !typeCorrect) {
            //INFO_LOG("No installation");
            retSymbolInfo.shouldHook = false;
            return;
        }


        //No special handling, return directly
        //todo: we temporarily turn hook off bu default to ensure program stability
        retSymbolInfo.shouldHook = false;
        return;
    }


    bool HookInstaller::parseSecInfos(ELFParser &elfParser, ELFSecInfo &pltInfo, ELFSecInfo &pltSecInfo,
                                      ELFSecInfo &gotInfo,ELFSecInfo &gotPltInfo, FileEntry &fileEntry) {

        //Even though a file may consists of multiple PMEntries, we can still only use the first and the last pmEntry to check for address validity.
        uint8_t *baseAddr = (uint8_t *) pmParser.getPmEntry(fileEntry.pmEntryRange[0].first).addrStart;
        uint8_t *endAddr = (uint8_t *) pmParser.getPmEntry(fileEntry.pmEntryRange.back().second).addrEnd;

        Elf64_Shdr pltHdr;
        if (elfParser.getSecHeader(SHT_PROGBITS, ".plt", pltHdr)) {
            pltInfo.startAddr = autoAddBaseAddr((uint8_t *) pltHdr.sh_addr, baseAddr, baseAddr, endAddr);
            pltInfo.size = pltHdr.sh_size;
            pltInfo.entrySize = pltHdr.sh_entsize;
        }else{
            ERR_LOG("Cannot read .plt header");
            pltInfo.startAddr=nullptr;
        }


        Elf64_Shdr pltSecHdr;
        pltSecInfo.entrySize = 0;
        if (elfParser.getSecHeader(SHT_PROGBITS, ".plt.sec", pltSecHdr)) {
            pltSecInfo.size = pltSecHdr.sh_size;
            pltSecInfo.entrySize = pltSecHdr.sh_entsize;
            pltSecInfo.startAddr = autoAddBaseAddr((uint8_t *) pltSecHdr.sh_addr, baseAddr, baseAddr, endAddr);
        }else{
            pltSecInfo.startAddr = nullptr;
        }


        Elf64_Shdr gotHdr;
        if (elfParser.getSecHeader(SHT_PROGBITS, ".got", gotHdr)) {
            gotInfo.size = gotHdr.sh_size;
            gotInfo.entrySize = gotHdr.sh_entsize;
            gotInfo.startAddr = autoAddBaseAddr((uint8_t *) gotHdr.sh_addr, baseAddr, baseAddr, endAddr);
        }else{
            DBG_LOG("Cannot read .got header");
            gotInfo.startAddr=nullptr;
        }

        Elf64_Shdr gotPltHdr;
        if (elfParser.getSecHeader(SHT_PROGBITS, ".got.plt", gotPltHdr)) {
            gotPltInfo.startAddr = autoAddBaseAddr((uint8_t *) gotPltHdr.sh_addr, baseAddr, baseAddr, endAddr);
            gotPltInfo.size=gotPltHdr.sh_size;
            gotPltInfo.entrySize=gotPltHdr.sh_entsize;

        }else{
            DBG_LOG("Cannot read .got.plt header");
            gotPltInfo.startAddr=nullptr;
        }

        return true;
    }

    //16bytes aligned. 0x90 are for alignment purpose
    
    //32bytes aligned. 0x90 are for alignment purpose



    uint32_t HookInstaller::parsePltStubId(uint8_t *dest) {
        int pushOffset = -1;
        if (*dest == 0xFF || *dest == 0xCC) {
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


    inline void HookInstaller::parseRequiredInfo() {
        //INFO_LOGS("thread:%p pthread_mutex_lock(&inst->dynamicLoadingLock)",pthread_self());

        pthread_mutex_lock(&dynamicLoadingLock);
        //Initialize existing loading id

        ELFParser elfParser;
        if (!pmParser.parsePMMap()) {
            fatalError("Cannot parsePmMap");
        }
        parseC10FileId();
        parsePythonInterpreterFileId();

        //Find new file from exising PMMaps
        std::vector<FileID> newFileEntryId;
        pmParser.getNewFileEntryIds(newFileEntryId, this->lastLoadingCounter,true);
        this->lastLoadingCounter=pmParser.getLoadingCounter();

        if (tlsOffset == nullptr) {
            parseTLSOffset();
        }

        for (ssize_t i = elfImgInfoMap.getSize(); i < pmParser.getFileEntryArraySize(); ++i) {
            elfImgInfoMap.pushBack();
        }

        //print_stacktrace();
        //print_pystacktrace();

        //Get segment info from /proc/self/maps
        for (ssize_t fileId = 0; fileId < newFileEntryId.size(); ++fileId) {
            FileID globalFileId = newFileEntryId[fileId];
            //INFO_LOGS("calleeFileId=%zd", calleeFileId);
            FileEntry &curFileEntry = pmParser.getFileEntry(globalFileId);
            //DBG_LOGS("Install newly discovered file:%s callerFileId:%zd", curFilePathName, calleeFileId);
            ELFImgInfo &curElfImgInfo = elfImgInfoMap[globalFileId];
            if (elfParser.parse(curFileEntry.filePath.c_str())) {
                //Find the entry allocatedSize of plt and got
                ELFSecInfo pltInfo{};
                ELFSecInfo pltSecInfo{};
                ELFSecInfo gotInfo{};
                ELFSecInfo gotPltInfo{};

                parseSecInfos(elfParser, pltInfo, pltSecInfo, gotInfo,gotPltInfo, curFileEntry);
                //todo: We assume plt and got entry allocatedSize is the same.
                curElfImgInfo.pltStartAddr = pltInfo.startAddr?pltInfo.startAddr:nullptr;
                curElfImgInfo.pltSecStartAddr = pltSecInfo.startAddr?pltSecInfo.startAddr:nullptr;
                curElfImgInfo.gotStartAddr = gotInfo.startAddr?gotInfo.startAddr:nullptr;
                curElfImgInfo.gotPltStartAddr = gotPltInfo.startAddr?gotPltInfo.startAddr:nullptr;


                INFO_LOGS("CurFileName=%s",curFileEntry.filePath.c_str());
                //Install hook on this file
                if (!parseSymbolInfo(elfParser, globalFileId, curFileEntry, pltInfo, pltSecInfo,
                                     gotInfo,gotPltInfo)) {
                    ERR_LOGS("installation for file %s failed.", curFileEntry.filePath.c_str());
                    curElfImgInfo.valid = false;
                }else{
                    curElfImgInfo.valid = true;
                }

            }
        }
        //INFO_LOGS("thread:%p pthread_mutex_unlock(&inst->dynamicLoadingLock)",pthread_self());

        pthread_mutex_unlock(&dynamicLoadingLock);
    }

    bool HookInstaller::replaceEntries() {
        //Allocate callIdSaver
        //todo: driverMemRecord leak
        if (allExtSymbol.getSize() <= installedSymbolSize) {
            //ERR_LOGS("No new symbols discovered but replacePltEntry was invoked anyways. allExtSymbol.getSize()=%zd, installedSymbolSize=%zd",
            //        allExtSymbol.getSize(),
            //        installedSymbolSize);
            return false;
        }
        
        /**
         * Prepare callIdSaver
         */
        //Fill address and ids in callIdSaver
        ssize_t cachedInstalledSymbolSize = installedSymbolSize; //Only install newly loaded API
        for (ssize_t curSymId = cachedInstalledSymbolSize; curSymId < allExtSymbol.getSize(); ++curSymId) {
            //Fetch symbol info
            APICallInfo &curSymInfo = allExtSymbol[curSymId];

            if (curSymInfo.apiType == APIType::PY_API) {
                //Py API should not be installed here
                continue;
            }

            ELFImgInfo &curImgInfo = elfImgInfoMap[curSymInfo.callerFileId];
            
            
            if (curSymInfo.apiType == APIType::C_PLT_API) {
                curSymInfo.realAddrPtr = curSymInfo.gotEntryAddr;
                if(curSymInfo.addressOverride){
                    *curSymInfo.realAddrPtr = reinterpret_cast<uint8_t *>(curSymInfo.addressOverride);
                }

            } else if (curSymInfo.apiType == APIType::C_DYN_API) {
                uint8_t ** curAddrStorageLocationPtr = this->relaDynRealAddrFields.alloc();
                curSymInfo.realAddrPtr = curAddrStorageLocationPtr; //Set realAddrPtr to newly allocated address storage location
                 if(curSymInfo.addressOverride){
                    *curSymInfo.realAddrPtr = reinterpret_cast<uint8_t *>(curSymInfo.addressOverride);
                }

            } else if (curSymInfo.apiType == APIType::C_DL_API) {
                // This type of API has already been installed at the beginning of dlsym
            } else {
                fatalError("You added new API type, please handle them specifically here");
            }

            /**
             * Perform special handling for certain symbols
            */
            if(curSymInfo.specialHandlingMarker!=SymbolSpecialHandlingMarker::NO_SPECIAL_HANDLING){
                if(curSymInfo.specialHandlingMarker==SymbolSpecialHandlingMarker::DLSYM_RTLD_NEXT_BYPASS){
                    curSymInfo.realAddrPtr=dlsymJumperWrapper.dlSymJumperAddrPtr;
                    //*curSymInfo.realAddrPtr=dlsymJumperWrapper.dlSymJumperBin; This has already been done in the constructor of dlsymJumperWrapper 
                }else if(curSymInfo.specialHandlingMarker==SymbolSpecialHandlingMarker::DLSYM_RTLD_NEXT_BYPASS){
                    curSymInfo.realAddrPtr=dlvsymJumperWrapper.dlSymJumperAddrPtr;
                    //*curSymInfo.realAddrPtr=dlvsymJumperWrapper.dlSymJumperBin; This has already been done in the constructor of dlsymJumperWrapper
                }
                
            }

            APICallInfo &curSym = allExtSymbol[curSymId];
            if (curSym.apiType == APIType::C_PLT_API) {
                IdSaverBinWrapper* curIdSaverBin = idSaverBinWrapperHeap.alloc();
                assert(tlsOffset!=nullptr);
                new (curIdSaverBin) IdSaverBinWrapper(tlsOffset,curSymInfo.realAddrPtr,curImgInfo.pltStartAddr, curSymId, curSymInfo.pltStubId, (void*)&asmTimingHandler);

                ELFImgInfo &curImgInfo = elfImgInfoMap[curSym.callerFileId];

                PltEntryWrapper* curPltEntryWrapper = pltEntryWrapperHeap.alloc();
                
                //Replace got entry, 16 is the allocatedSize of a plt entry
                if (abs(curSym.pltEntryAddr - *curSym.realAddrPtr) < 16) {
                    //Address not resolved, perform lazy loading and call ld.so
                    *curSym.realAddrPtr = curIdSaverBin->idSaverBin + curIdSaverBin->CALL_LD_INST;
                }

                //Replace PLT
                if (curSym.pltSecEntryAddr) {
                    new (curPltEntryWrapper) PltEntryWrapper(curIdSaverBin->idSaverBin);
                     memcpy(curSym.pltSecEntryAddr, curPltEntryWrapper->pltEntryBin, curPltEntryWrapper->PLT_ENTRY_BIN_SIZE);
                } else {
                    new (curPltEntryWrapper) PltEntryWrapper(curIdSaverBin->idSaverBin);
                    memcpy(curSym.pltEntryAddr, curPltEntryWrapper->pltEntryBin, curPltEntryWrapper->PLT_ENTRY_BIN_SIZE);
                }
                
            } else if (curSym.apiType == APIType::C_DYN_API) {

                IdSaverBinWrapper* curIdSaverBin = idSaverBinWrapperHeap.alloc();
                assert(tlsOffset!=nullptr);
                new (curIdSaverBin) IdSaverBinWrapper(tlsOffset,curSymInfo.realAddrPtr,curImgInfo.pltStartAddr, curSymId, curSymInfo.pltStubId,(void*)&asmTimingHandler);

                ELFImgInfo &curImgInfo = elfImgInfoMap[curSym.callerFileId];

                //Replace got directly to relaIdSaver
                //INFO_LOGS("curRelaIdSaver=%p",curRelaIdSaver);
                //This is safe because gotEntry is naturally placed in a read-only page. Adding write permission will not touch other parts
                adjustMemPerm(curSym.gotEntryAddr, curSym.gotEntryAddr + 1,
                              PROT_READ | PROT_WRITE); 
                /*INFO_LOGS("C_DYN_API replace SumId=%zd *%p= from %p to %p realIdPtr=%p",curSymId,curSym.sharedAddr1.gotEntryAddr,
                    *curSym.sharedAddr1.gotEntryAddr,
                    curRelaIdSaver,
                    curSym.realAddrPtr
                );*/
                *curSym.gotEntryAddr = curIdSaverBin->idSaverBin;
                //adjustMemPerm(curSym.gotEntryAddr, curSym.gotEntryAddr + 1,
                //              PROT_READ);
            } else {
                //This is not a C/C++ symbol, so we do not need to install it.
            }




        }
        installedSymbolSize = allExtSymbol.getSize();


       
        //DBG_LOG("replace PLT finished");
        return true;
    }


    void HookInstaller::createRecordingFolder() {
        //sprintf(folderName, "scalerdata_%lu", getunixtimestampms());
        if (mkdir(folderName.c_str(), 0755) == -1) {
            fatalErrorS("Cannot mkdir %s because: %s", folderName.c_str(),
                        strerror(errno));
        }
        initializeRecordingFileHandles("nativeAPIInfoFile.txt", instance->nativeAPIInfoFile);
        fprintf(instance->nativeAPIInfoFile, "%s,%s,%s\n", "funcName", "calleeFileId", "symIdInFile");
        initializeRecordingFileHandles("pythonAPIInfoFile.txt", instance->pythonAPIInfoFile);
        fprintf(instance->pythonAPIInfoFile, "%s,%s\n", "funcName", "pyModuleId");
        initializeRecordingFileHandles("elfImgStrTbl.txt", instance->elfImgStrTbl);
        initializeRecordingFileHandles("pyModuleStrTbl.txt", instance->pyModuleStrTbl);
        initializeRecordingFileHandles("pySrcFileStrTbl.txt", instance->pySrcFileStrTbl);

        pmParser = PmParser(instance->elfImgStrTbl);
    }

    /**
     * Intercept and install dlsym
     * @return successful or not
     */
    bool HookInstaller::installOnDlSym(const char *__name, void *realFuncAddr, void *callerAddr, void *&retAddr) {
        if (!realFuncAddr) {
            ERR_LOGS("dlsym(\"%s\") returns null. Scaler cannot install for this API.", __name);
            retAddr = nullptr;
            return false;
        }

         if(!curContext){
            initTLS();
        }
        HookContext *hookContextPtr = curContext;
        assert(hookContextPtr!=nullptr);
       
        pthread_mutex_lock(&dynamicLoadingLock);

        //realFuncAddr is valid
        //todo: what if two APIs has the same address?
        auto findIter = dlsymRealAddrGOTEntryMap.find(realFuncAddr);
        FileID calleeFileId = instance->pmParser.findFileIdByAddr(realFuncAddr);
        FileEntry &fileEntry = instance->pmParser.getFileEntry(calleeFileId);
        if (findIter != instance->dlsymRealAddrGOTEntryMap.end()) {
            DlsymInstallInfo &info = findIter->second;

            if (info.calleeFileId == calleeFileId && info.loadingCounter == fileEntry.loadingCounter) {
                //Skip hooking this API because it has been hooked by the same file version
                DBG_LOG("Address was hooked before, MLInsight will not install");
                retAddr = info.idSaverEntry;
                pthread_mutex_unlock(&instance->dynamicLoadingLock);
                return false;
            }
        }

        /**
         * Check if the API matches the requirement.
         */
        Elf64_Word bind = STB_GLOBAL;
        Elf64_Word type = STT_FUNC;
        SymbolHookHint retSymbolHookHint;
        instance->shouldHookThisSymbol(__name, bind, type, retSymbolHookHint);
        if (!retSymbolHookHint.shouldHook) {
            //Should not hook
            retAddr = realFuncAddr;
            pthread_mutex_unlock(&instance->dynamicLoadingLock);
            return false;
        }

        auto pmParserIterator = pmParser.findPmEntryIdByAddr(realFuncAddr);
        if (!pmParserIterator->isE()) {
            //This address does not point to the code section. It might be a global variable.
            retAddr = realFuncAddr;
            pthread_mutex_unlock(&instance->dynamicLoadingLock);
            return false;
        }

        if (retSymbolHookHint.addressOverride) {
            realFuncAddr = retSymbolHookHint.addressOverride;
        }

        ssize_t newSymId = hookContextPtr->recordArray.getSize();

        //Should install, allocate a new saverbinwrapper
        IdSaverBinWrapper *idSaverBinWrapper = idSaverBinWrapperHeap.alloc();
        assert(tlsOffset!=nullptr);
        new(idSaverBinWrapper) IdSaverBinWrapper(tlsOffset,(uint8_t **) &idSaverBinWrapper->realFuncAddr,NULL, newSymId, 0,(void*)&asmTimingHandler); //Allocate idsaver
        idSaverBinWrapper->realFuncAddr = realFuncAddr;

        hookContextPtr->recordArray.pushBack(); //Insert new entry to record dlsym loaded symbol

        //Save idSaver address to dlsymRealAddrGOTEntryMap

        const auto emplaceRlt = dlsymRealAddrGOTEntryMap.emplace(std::make_pair(realFuncAddr, DlsymInstallInfo()));
        emplaceRlt.first->second.idSaverEntry = idSaverBinWrapper->idSaverBin;
        emplaceRlt.first->second.calleeFileId = calleeFileId;
        emplaceRlt.first->second.loadingCounter = fileEntry.loadingCounter;

        retAddr = idSaverBinWrapper->idSaverBin;

        APICallInfo &symInfo = instance->allExtSymbol.pushBack();
        installedSymbolSize += 1;

        //The callerFileId is calculated based on the caller of the dlsym function.
        symInfo.callerFileId = pmParser.findFileIdByAddr(callerAddr);
        //The calleeFileId is calculated based on the return value of the system's dlsym function (The actual address of the function).
        symInfo.calleeFileId = pmParser.findFileIdByAddr(realFuncAddr);
        symInfo.dlsymRealAddr = static_cast<uint8_t *>(realFuncAddr);
        symInfo.apiType = APIType::C_DL_API;


        fprintf(nativeAPIInfoFile, "%s,%ld,%d\n", __name, symInfo.callerFileId, -1);
        pthread_mutex_unlock(&dynamicLoadingLock);
        return true;
    }

    bool HookInstaller::parseRealFileId() {
        //Before DlClose time, we need to parse the real address of parsed symbols
        ssize_t allExtSymbolSize = instance->allExtSymbol.getSize();
        for (ssize_t i = 0; i < allExtSymbolSize; ++i) {
            APICallInfo &curFunctionInfo = instance->allExtSymbol[i];
            //curFunctionInfo.callerFileId

            if (curFunctionInfo.calleeFileId == -1 &&
                instance->pmParser.getFileEntry(curFunctionInfo.callerFileId).fileExists) {
                //RealFileId has not been resolved yet
                if (curFunctionInfo.apiType == APIType::C_DYN_API || curFunctionInfo.apiType == APIType::C_PLT_API) {
                    if (curFunctionInfo.realAddrPtr == nullptr) {
                        INFO_LOGS("Fatal error occured in symbolId=%zd apiType=%d", i, curFunctionInfo.apiType);
                    }
                    assert(curFunctionInfo.realAddrPtr != nullptr);
                    curFunctionInfo.calleeFileId = instance->pmParser.findFileIdByAddr(*(curFunctionInfo.realAddrPtr));
                }
#ifndef NDEBUG
                else if (curFunctionInfo.apiType == APIType::C_DL_API) {
                    //DL calleeFileId has been resolved at installation time. No need to resolve again.
                    fatalError("DL API should be linked with module id before the first invocation.");
                } else if (curFunctionInfo.apiType == APIType::PY_API) {
                    fatalError("Python API should be linked with module id before the first invocation.");
                }
#endif
            }
        }


        return true;
    }

    void HookInstaller::parseTLSOffset() {
        ssize_t scalerFileId = pmParser.getMLInsightFileId();
        assert(scalerFileId != -1);
        ELFParser parser;
        FileEntry &curFileEntry = pmParser.getFileEntry(scalerFileId);
        parser.parse(curFileEntry.filePath.c_str());

        uint8_t *baseAddr = pmParser.getPmEntry(curFileEntry.pmEntryRange[0].first).addrStart;
        uint8_t *endAddr = pmParser.getPmEntry(curFileEntry.pmEntryRange.back().second).addrEnd;

        ELFSecInfo pltInfo{};
        ELFSecInfo pltSecInfo{};
        ELFSecInfo gotInfo{};
        ELFSecInfo gotPltInfo{};

        parseSecInfos(parser, pltInfo, pltSecInfo, gotInfo, gotPltInfo,curFileEntry);
        if (!gotInfo.startAddr) {
            fatalError("Failed to parse TLS offset related sections.");
            exit(-1);
        }

        for (ssize_t i = 0; i < parser.relaDYNEntrySize; ++i) {
            const char *funcName;
            Elf64_Word type;
            Elf64_Word bind;
            parser.getExtSymbolInfo(i, funcName, bind, type, parser.relaDYNSection);
            ssize_t stringLength = strlen(funcName);

            if (stringLength == 26 && strncmp(funcName, "_ZN9mlinsight10curContextE", stringLength) == 0) {
                tlsOffset = (uint8_t **) autoAddBaseAddr((uint8_t *) (parser.getRelaOffset(i, parser.relaDYNSection)),
                                                         baseAddr, baseAddr, endAddr);
                INFO_LOGS("Parse TLSOffset is successful %p %p",tlsOffset,*tlsOffset);
                break;
            }
        }
    }

    bool HookInstaller::initializeRecordingFileHandles(const std::string &fileName, FILE *&retFileHandle) {
        std::stringstream ss;
        ss << mlinsight::HookInstaller::instance->folderName << "/" << fileName;

        retFileHandle = fopen(ss.str().c_str(), "a");
        if (!retFileHandle) {
            fatalErrorS("Cannot open %s because:%s", ss.str().c_str(), strerror(errno));
            return false;
        }

        return true;
    }

    inline void HookInstaller::parseC10FileId() {
        //The file ID of Scaler itself.
        FileID libc10CUDAFileID = pmParser.getC10CUDAFileID();
        if (libc10CUDAFileID >= 0 && libc10_cuda_text_begin == nullptr) {
            pmParser.getAddressRangeByFileId(libc10CUDAFileID, libc10_cuda_text_begin, libc10_cuda_text_end);
        }
    }

    inline void HookInstaller::parsePythonInterpreterFileId() {
        FileID pyInterpreterFileID = pmParser.getPythonInterpreterFileId();
        if (pyInterpreterFileID >= 0 && pythonInterpreter_text_begin == nullptr) {
            pmParser.getAddressRangeByFileId(pyInterpreterFileID, pythonInterpreter_text_begin,
                                             pythonInterpreter_text_end);
        }

    }


}

#endif
