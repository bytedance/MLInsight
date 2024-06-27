/*

@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#ifndef MLINSIGHT_HOOKINSTALLER_H
#define MLINSIGHT_HOOKINSTALLER_H

#ifdef __linux

#include <string>
#include <vector>
#include <atomic>
#include "common/ELFParser.h"
#include "common/ProcInfoParser.h"
#include "common/HashMap.h"
#include "trace/type/RecordingDataStructure.h"
#include "trace/type/ELFInfo.h"
#include "common/MemoryHeap.h"
#include <Python.h>
#include <unordered_map>
#include "trace/type/PyCodeExtra.h"

namespace mlinsight {

    /**
    * Determine whether a symbol should be hooked
    */
    typedef bool SYMBOL_FILTER(std::string fileName, std::string funcName);
    
     /**
     * Also called "global shadow table" in table. This struct combines idSaver, pseudo Plt entry, and invocation counter into one executable assembly.
     * Please note that DlSymJumperWarpper can only be stored in page-alinged datastructure for correct memory permission adjustments (eg: ObjectPoolHeap). 
     * Otherwise adjustMemPerm might override permission of the adjancent memory regions which will cause unwanted segmentation fault.
    */
    class IdSaverBinWrapper {
        public:
        static const int ID_SAVER_BIN_SIZE = 134;
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


        uint8_t idSaverBin[ID_SAVER_BIN_SIZE] = {
                /**
                 * Read context pointer (TLS)
                 */
                //mov $0x1122334455667788,%r11 | move context's TLS offset to r11
                0x49, 0xBB, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                //mov %fs:(%r11),%r11 | move per-thread context's address to r11 
                0x64, 0x4D, 0x8B, 0x1B,
                //cmpq $0,%r11 | Check whether per-thread context is initialized or not
                0x49, 0x83, 0xFB, 0x00,
                //jmp APIInvocation | If it is not initialized, jump to APIInvocation to avoid segmentation fault.  (Currently, the performance analysis part is still under testing. So we skip these parts)
                0xEB, 0x2D,

                /**
                 * Counting part: counter. per-thread data for all APIs 
                 *      gap: 
                 */
                //push %r10 | Save register r10, as we will use r10 register soon.
                0x41, 0x52,
                //mov    0x00000000(%r11),%r11 | mov the address of Context.recordArray.internalArray to r11
                0x4D, 0x8B, 0x9B, 0x00, 0x00, 0x00, 0x00,
                //mov    0x00000000(%r11),%r10 | mov the value of current API's invocation counter (``count'' in RecTuple) to r10
                0x4D, 0x8B, 0x93, 0x00, 0x00, 0x00, 0x00,
                //add    $0x1,%r10 | Increase counter by 1
                0x49, 0x83, 0xC2, 0x01,
                //mov    %r10,0x11223344(%r11) | Store the updated counter back
                0x4D, 0x89, 0x93, 0x00, 0x00, 0x00, 0x00,
                //mov    0x11223344(%r11),%r11 | move the value of current API's gap to r10
                0x4D, 0x8B, 0x9B, 0x00, 0x00, 0x00, 0x00,
                //and    %r11,%r10 | counter % gap: we only care about the bits with value 1
                0x4D, 0x21, 0xDA,
                //cmpq   $0x0,%r10 | Check the ``and'' result and output the result to a flag register. 
                0x49, 0x83, 0xFA, 0x00,
                //pop    %r10 | Restore the value of r10
                0x41, 0x5A,
                //jz TIMING_JUMPER | If counter % gap == 0, jump to TIMING part.  
                0x74, 0x1F,

                /**
                 * APIInvocation: Invoking the real API.
                 *     For dlsym,  the API is returned by dlsym. 
                 *     For lazy mode before the address resolution, GOT's address is PLT's "push index" instruction address. The address here is
                 *          pushq_address below. 
                 *     For .plt.got or eager mode, lazy mode after resolution, GOT's address is API address
                 *     
                 *     Typically, only context is not initialized or timing is not required, 
                 *     we will invoke the API here. 
                 */
                //movq $0x1122334455667788,%r11 | Move current API's address (as shown above) to r11
                0x49, 0xBB, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                //jmpq (%r11) | Jump to the address as shown above
                0x41, 0xFF, 0x23,

                // pushq_address
                /* The following few instructions is the actually 2nd and 3rd instructons of the original PLT entry. 
                 * Instead, we are using far jump (can't use near jump). Then we will rely on
                 * the original system to perform the address resolution (ld-linux.so)
                 */
                //pushq $0x11223344 
                0x68, 0x00, 0x00, 0x00, 0x00,
                //movq $0x1122334455667788,%r11 | move the address of PLT[0] to r11
                0x49, 0xBB, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                //jmpq *%r11 | Jump to PLT[0]
                0x41, 0xFF, 0xE3,

                /**
                 * TIMING part: 
                 *    Save some key values/pointers to the stack, then invoke asmTimingHandler
                 */

                //movl $0x44332211,0x28(%rsp) | Save low bits of realAddrPtr
                0xC7, 0x44, 0x24, 0xF0, 0x00, 0x00, 0x00, 0x00,
                //movl $0x44332211,0x10(%rsp) | Save high bits of realAddrPtr
                0xC7, 0x44, 0x24, 0xF4, 0x00, 0x00, 0x00, 0x00,
                //movl $0x44332211,0x0(%rsp) | Save symId to stack
                0x48, 0xC7, 0x44, 0x24, 0xF8, 0x00, 0x00, 0x00, 0x00,
                //movq $0x1122334455667788,%r11 | Move the address of asmTimingHandler to r11
                0x49, 0xBB, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                //jmpq *%r11 | Jump to asmTimingHandler
                0x41, 0xFF, 0xE3
        };
        void *realFuncAddr;
        //todo: Put code that changes the idSaverBin into this class like DLSymJumperWarpper

        IdSaverBinWrapper(uint8_t ** tlsOffset, uint8_t **realAddrPtr, uint8_t *firstPltEntry, uint32_t symId,
                                                 uint32_t pltStubId, void* asmTimingHandlerPtr){
            adjustMemPerm(this->idSaverBin, this->idSaverBin + ID_SAVER_BIN_SIZE, PROT_READ | PROT_WRITE | PROT_EXEC);
                    assert(sizeof(uint8_t **) == 8);

            memcpy(this->idSaverBin + COUNT_TLS_ARR_ADDR, tlsOffset, sizeof(void *));

            uint32_t recArrayOffset= LDARR_OFFSET_IN_CONTEXT + INTERNALARR_OFFSET_IN_LDARR;
            memcpy(this->idSaverBin + REC_ARRAY_OFFSET1, &recArrayOffset, sizeof(uint32_t));

            uint32_t countOffset = symId * sizeof(RecTuple) + COUNT_OFFSET_IN_RECARR;
            uint32_t gapOffset = symId * sizeof(RecTuple) + GAP_OFFSET_IN_RECARR;

            memcpy(this->idSaverBin + COUNT_OFFSET1, &countOffset, sizeof(uint32_t));
            memcpy(this->idSaverBin + COUNT_OFFSET2, &countOffset, sizeof(uint32_t));

            memcpy(this->idSaverBin + GAP_OFFSET, &gapOffset, sizeof(uint32_t));

            //Fill got address
            memcpy(this->idSaverBin + GOT_ADDR, &realAddrPtr, sizeof(uint8_t **));
            //Fill function id
            memcpy(this->idSaverBin + PLT_STUB_ID, &pltStubId, sizeof(uint32_t));
            //Fill first plt address
            memcpy(this->idSaverBin + PLT_START_ADDR, &firstPltEntry, sizeof(uint8_t *));

            //INFO_LOG("Here");

            uint32_t realAddrPtrHi = ((uint64_t) realAddrPtr) >> 32;
            uint32_t realAddrPtrLo = ((uint64_t) realAddrPtr) & 0xffffffff;
            //INFO_LOGS("GOT_ADDR=%p", realAddrPtr);
            //INFO_LOGS("GOT_HI=0x%x GOT_LOW", realAddrPtrHi);
            //INFO_LOGS("GOT_LO=0x%x GOT_LOW", realAddrPtrLo);

            memcpy(this->idSaverBin + LOW_BITS_GOTENTRYADDR, &realAddrPtrLo, sizeof(uint32_t));
            memcpy(this->idSaverBin + HIGH_BITS_GOTENTRYADDR, &realAddrPtrHi, sizeof(uint32_t));

            //INFO_LOG("Fill Symbol Id");

            //Fill symId
            memcpy(this->idSaverBin + SYM_ID, &symId, sizeof(uint32_t));

            //INFO_LOG("Fill asmTimingHandler");

            //Fill asmTimingHandler
            memcpy(this->idSaverBin + ASM_HOOK_HANDLER_ADDR, (void *) &asmTimingHandlerPtr, sizeof(void *));
            //INFO_LOG("Here");

            // adjustMemPerm(this->idSaverBin, this->idSaverBin + ID_SAVER_BIN_SIZE, PROT_READ | PROT_EXEC);
        }

    };

   
    /**
     * A jumper that selects dlsym implementation based on handle type
     * Please note that DlSymJumperWarpper can only be stored in page-alinged datastructure for correct memory permission adjustments (eg: ObjectPoolHeap). 
     * Otherwise adjustMemPerm might override permission of the adjancent memory regions which will cause unwanted segmentation fault.
    */
    class DlSymJumperWarpper {
    public:
    static const int DL_SYM_JUMPER_SIZE = 32;

        uint8_t internalArr[DL_SYM_JUMPER_SIZE] = {
            //mov $r11, 0x1122334455667788
            0x49, 0xbb, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11,
            //cmp $rdi,-0x1
            0x48, 0x83, 0xff, 0xff,
            //jz SKIP_DLSYM
            0x74, 0x03,
            //jmp $r11
            0x41, 0xff, 0xe3,
            //mov $r11, 0x1122334455667788
            0x49, 0xbb, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11,
            //jmp $r11
            0x41, 0xff, 0xe3,
        };
        uint8_t* dlSymJumperBin=internalArr;
        uint8_t** dlSymJumperAddrPtr=&dlSymJumperBin;

        inline DlSymJumperWarpper(void* dlsymProxyAddress, void* realDlsymAddress){
            //Please note that DlSymJumperWarpper can only be stored in page-alinged datastructure (eg: ObjectPoolHeap). Otherwise adjustMemPerm might override adjancent memory regions which will cause unwanted behavior.
            //Even if adjustMemPerm overrides dlSymJumperBin by a few bytes it will not matter because this modification only adds PROT_WRITE, and keeps PROT_EXEC. 
            adjustMemPerm(this->dlSymJumperBin, this->dlSymJumperBin + DL_SYM_JUMPER_SIZE, PROT_READ | PROT_WRITE | PROT_EXEC);
            memcpy(this->dlSymJumperBin + 2,&dlsymProxyAddress,sizeof(void*));
            memcpy(this->dlSymJumperBin + 21,&realDlsymAddress,sizeof(void*));
            //adjustMemPerm(this->dlSymJumperBin, this->dlSymJumperBin + DL_SYM_JUMPER_SIZE, PROT_READ | PROT_EXEC);
            INFO_LOGS("this->dlSymJumperBin=%p pid=%d tid=%zd",this->dlSymJumperBin,getpid(),pthread_self());
            INFO_LOGS("dlsymProxyAddress=%p realDlsymAddress=%p",dlsymProxyAddress,realDlsymAddress);
            
            
        }

    };

    /**
     * A plt entry replacement.
     * Please note that PltEntryWrapper can only be stored in page-alinged datastructure for correct memory permission adjustments (eg: ObjectPoolHeap). 
     * Otherwise adjustMemPerm might override permission of the adjancent memory regions which will cause unwanted segmentation fault.
    */
    class PltEntryWrapper{
    public:
        static const int PLT_ENTRY_BIN_SIZE=16;
        uint8_t pltEntryBin[PLT_ENTRY_BIN_SIZE] = {0x49, 0xBB, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                             0x00, 0x00, 0x41, 0xff, 0xE3, 0x90, 0x90, 0x90};
        inline PltEntryWrapper(uint8_t *funcAddr){
            //Copy address
            adjustMemPerm(this->pltEntryBin, this->pltEntryBin + PLT_ENTRY_BIN_SIZE, PROT_READ | PROT_WRITE | PROT_EXEC);
            assert(sizeof(uint8_t **) == 8);
            memcpy(this->pltEntryBin + 2, &funcAddr, sizeof(uint8_t **));
            //adjustMemPerm(this->pltEntryBin, this->pltEntryBin + PLT_ENTRY_BIN_SIZE, PROT_READ | PROT_EXEC);
        }
            
    };


   
    class SymbolHookHint {
    public:
        ssize_t initialGap;
        void *addressOverride;
        bool shouldHook;
        void **realAddressPtr; //Real address should be saved to this destination
        SymbolSpecialHandlingMarker specialHandlingMarker;
        
        /*
        * For symbol defined in special hook hint, do hook by default
        */
        SymbolHookHint() : shouldHook(false), addressOverride(nullptr), initialGap(0), realAddressPtr(nullptr),specialHandlingMarker(SymbolSpecialHandlingMarker::NO_SPECIAL_HANDLING) {

        }

        /*
        * Should replace address
        */
        SymbolHookHint(void *addressOverride, void **realAddressPtr = nullptr, ssize_t initialGap = 0) :
                shouldHook(true), addressOverride(addressOverride), initialGap(initialGap),
                realAddressPtr(realAddressPtr),specialHandlingMarker(SymbolSpecialHandlingMarker::NO_SPECIAL_HANDLING) {

        }


        /*
        * Fine-grained parameter
        */
        SymbolHookHint(bool shouldHook, void *addressOverride, void **realAddressPtr, ssize_t initialGap,SymbolSpecialHandlingMarker specialHandlingMarker) :
                shouldHook(shouldHook), addressOverride(addressOverride), initialGap(initialGap),
                realAddressPtr(realAddressPtr),specialHandlingMarker(specialHandlingMarker) {

        }
    };

    /**
     * A struct recording dlsym installed APIs.
     * The purpose of this struct is to detect repeated dlsym of different APIs
     */
    struct DlsymInstallInfo {
        std::string apiName;
        void *idSaverEntry = nullptr; //The previously allocated idSaverEntry
        FileID calleeFileId = -1; //Which file is this API correlated to.
        ssize_t loadingCounter = -1; //Which version of the file is this API correlated to. Used to detect API re-load. eg: Same address of different APIs or same address, same API but different version. Scaler should only install when the <callerFileId, loadingCounter, api address> does not match with existing records.
    };

    class HookInstaller;

    //Avoid the invocation of getInstance to save runtime overhead.
    extern HookInstaller *hookInstallerInstance;

    class HookInstaller {
    public:
        HookInstaller(std::string folderName);

        HookInstaller(HookInstaller &) = delete;

        HookInstaller(HookInstaller &&) = delete;

        virtual bool install();

        /**
         * Scan for APIs at dlopen and dlmopen times.
         * @return Installation succeeded or not.
         */
        virtual bool installOnDlOpen();

        /**
         * Link API with libraries.
         * This function may need to be invoked many times as libraries are constantly changing.
         * @return The parsing is correct or not
         */
        virtual bool parseRealFileId();

        virtual bool installOnDlSym(const char *__name, void *realFuncAddr, void *callerAddr, void *&retAddr);

        virtual bool uninstall();

        virtual ~HookInstaller();

        PmParser pmParser;

        /*
         * There are currently three types of globalIds, please be sure not to mix them:
         * 1. globalCFileId: A unique id for each elf images. Used to index elements in the elfImgInfoMap. Used to work with APIType::PY_API, APIType::C_PLT_API, APIType::C_DYN_API, and APIType::C_DL_API.
         * 2. globalPyModuleId: A unique id for each python modules. Used to index elements in the pyModuleReverseIdMap. Used to work with APIType::PY_API.
         * 3. globalPySrcFileId: A unique id for each python source file. Used to index elements in the pySrcFileInfoMap. Used to work with APIType::PY_API.
         */

        /**
         * Fields used to record C++ related information.
         */
        //Note that the following variable only stores information for C funciton. It's index only includes C function rather than python modules
        Array<ELFImgInfo> elfImgInfoMap; //Mapping fileID to ELFImgInfo. It's write must be protected by dynamicLoadingLock. It's read can be unprotected because there will be no array expansion.
        /**
         * Fields used to record Python module and file information
         */
        Array<PyModuleInfo> pyModuleInfoMap;
        std::unordered_map<std::string, FileID> pyModuleIdMap; //Map pyModule information to fileId
        Array<PySrcFileInfo> pySrcFileInfoMap;
        std::unordered_map<std::string, FileID> pySrcFileIdReverseMap; //Map pyModule information to fileId
        Array<PyTorchModuleInfo> pytorchModuleInfoMap;
        std::unordered_map<PyObject *, FileID> pytorchModuleIdMap; //Map pytorch module to


        Array<APICallInfo> allExtSymbol; //Includes all types of APIs. It's write must be protected by dynamicLoadingLock. It's read can be unprotected because there will be no array expansion.
        ssize_t lastLoadingCounter = -9999; //Used to compare with ProcInfoParser::PmParser::loadingCounter
        ssize_t installedSymbolSize = 0; //Used to find newly found symbol
        //Protect possible contention in thread context initialization. All context initialization/dynamic loading should acquire this lock. It's read can be unprotected because there will be no array expansion.
        //But hook handler does not need to acquire this lock to save overhead.
        //Because there is no expansion possibility for elfImgInfoMap and allExtSymbol. Already initialized fields stays initialized and dynamic loading will not impact them.
        //Dynamically loaded fields will only be invoked by the loading thread or other thread after the loading completes.
        //Python has GIL, so there is no real simultaenous access. Besides, python modules are seperated with others by using different loading id.
        pthread_mutex_t dynamicLoadingLock;

        std::string folderName;
        uint8_t **tlsOffset = nullptr;
        std::map<void *, DlsymInstallInfo> dlsymRealAddrGOTEntryMap; //Convert dynamically loaded symbol address to slots in dlSymIdSavers

        //The reaon for seperate store is for efficiency. These driverMemRecord cannot be freed once allocated. To support dynamic loading, we make it
        ObjectPoolHeap<IdSaverBinWrapper> idSaverBinWrapperHeap; //Holds the idsaver installed by dlsym.
        ObjectPoolHeap<PltEntryWrapper> pltEntryWrapperHeap;
        DlSymJumperWarpper dlsymJumperWrapper; //Used for special handling lf dlsym
        DlSymJumperWarpper dlvsymJumperWrapper; //Used for special handling lf dlsym
        
        ObjectPoolHeap<uint8_t *> relaDynRealAddrFields; //Holds the real address of .got.plt entries

        std::map<std::string, SymbolHookHint> hookHintMap;

        FILE *nativeAPIInfoFile = nullptr;
        FILE *pythonAPIInfoFile = nullptr;
        FILE *elfImgStrTbl = nullptr;
        FILE *pyModuleStrTbl = nullptr; //The string table to store python module name.
        FILE *pySrcFileStrTbl = nullptr; //The string table to store python source file name.

        /**
         * Private constructor
         */

        inline bool isSymbolAddrResolved(APICallInfo &symInfo) {
            assert(symInfo.apiType==APIType::C_PLT_API);
            //Check whether its value has 6 bytes offset as its plt entry start address
            ELFImgInfo &curImg = elfImgInfoMap[symInfo.callerFileId];
            int64_t myPltStartAddr = (int64_t) curImg.pltStartAddr;
            int64_t curGotAddr = (int64_t) symInfo.realAddrPtr;
            int64_t offset = curGotAddr - myPltStartAddr;
            return offset > 6 || offset < -6;
        }

        static HookInstaller *getInstance(std::string folderName);

        static HookInstaller *getInstance();

        static HookInstaller *instance;

        void
        shouldHookThisSymbol(const char *funcName, Elf64_Word &bind, Elf64_Word &type, SymbolHookHint &symbolHookHint);

    protected:


        inline bool
        parseSecInfos(ELFParser &elfParser, ELFSecInfo &pltInfo, ELFSecInfo &pltSecInfo, ELFSecInfo &gotInfo,ELFSecInfo &gotPltInfo,
                      FileEntry &fileEntry);

        bool
        parseSymbolInfo(ELFParser &parser, ssize_t fileId, FileEntry &fileEntry, ELFSecInfo &pltSection,
                        ELFSecInfo &pltSecureSection, ELFSecInfo &gotSec,ELFSecInfo &gotPltSec);

        bool makeGOTWritable(ELFSecInfo &gotSec, bool writable);


        uint32_t parsePltStubId(uint8_t *dest);


        inline bool parseRelaSection(ssize_t &validRelaEntrySize, const APIType &symbolType, ELFParser &parser,
                                     ELFImgInfo &curImgInfo, FileID fileId, ELFSecInfo &pltSection,
                                     ELFSecInfo &pltSecureSection, Elf64_Rela *&relaSection, ssize_t relaEntrySize,
                                     FILE *symInfoFile, uint8_t *baseAddr, uint8_t *startAddr, uint8_t *endAddr);


        inline void parseRequiredInfo();

        /**
         * Actual entry
         * @return
         */
        bool replaceEntries();

        void createRecordingFolder();

        void parseTLSOffset();

        bool initializeRecordingFileHandles(const std::string &fileName, FILE *&retFileHandle);

        void parseC10FileId();

        void parsePythonInterpreterFileId();

    };

    struct ProxySymbol {
        std::string name = "";
        void *address = nullptr;
        void **realAddressPtr = nullptr;
    };


}


extern "C" {


__pid_t fork(void);

}


#endif

#endif
