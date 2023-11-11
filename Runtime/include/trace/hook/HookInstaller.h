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


namespace mlinsight {

    /**
    * Determine whether a symbol should be hooked
    */
    typedef bool SYMBOL_FILTER(std::string fileName, std::string funcName);

    const int ID_SAVER_BIN_SIZE=134;
    struct IdSaverBinWrapper {
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
        void* realFuncAddr;
    };

    class SymbolHookHint{
    public:
        ssize_t initialGap;
        void *addressOverride; 
        bool shouldHook;
        void** realAddressPtr; //Real address should be saved to this destiation

        /*
        * For symbol defined in special hook hint, do hook by defult
        */
        SymbolHookHint():shouldHook(false),addressOverride(nullptr),initialGap(0),realAddressPtr(nullptr) {

        }

        /*
        * Should replace address
        */
        SymbolHookHint(void *addressOverride,void** realAddressPtr=nullptr,ssize_t initialGap=0):
                        shouldHook(true),addressOverride(addressOverride),initialGap(initialGap),realAddressPtr(realAddressPtr) {

        }



        /*
        * Fine-grained parameter
        */
        SymbolHookHint(bool shouldHook, void *addressOverride,void** realAddressPtr,ssize_t initialGap):
                        shouldHook(shouldHook),addressOverride(addressOverride),initialGap(initialGap),realAddressPtr(realAddressPtr) {

        }
    };
    

    class HookInstaller {
    public:
        HookInstaller(std::string folderName);

        HookInstaller(HookInstaller &) = delete;

        HookInstaller(HookInstaller &&) = delete;

        virtual bool install();

        virtual bool installAPI();

        virtual bool installDlSym(void *realFuncAddr,void*& retAddr);

        virtual bool uninstall();

        virtual ~HookInstaller();

        PmParser pmParser;

        Array<ELFImgInfo> elfImgInfoMap;//Mapping fileID to ELFImgInfo. It's write must be protected by dynamicLoadingLock. It's read can be unprotected because there will be no array expansion.
        Array<FunctionInfo> allExtSymbol; //All external symbols in ELF image. It's write must be protected by dynamicLoadingLock. It's read can be unprotected because there will be no array expansion.
        ssize_t installedSymbolSize=0; //Used to find newly found symbol
        //Protect possible contention in thread context initialization. All context initialization/dynamic loading should acquire this lock. It's read can be unprotected because there will be no array expansion.
        //But hook handler does not need to acquire this lock to save overhead.
        //Because there is no expansion possibility for elfImgInfoMap and allExtSymbol. Already initialized fields stays initialized and dynamic loading will not impact them.
        //Dynamically loaded fields will only be invoked by the loading thread or other thread after the loading completes.
        //Python has GIL, so there is no real simultaenous access. Besides, python modules are seperated with others by using different loading id.
        pthread_mutex_t dynamicLoadingLock;
        ssize_t validRelaPltSize=0; //All API that has .rela.plt in allExtSymbol
        ssize_t validRelaDynSize=0; //All API that has .rela.dyn in allExtSymbol
        ssize_t installedRelaDynSize=0; //Used to find newly found rela location


        std::string folderName;
        uint8_t **tlsOffset = nullptr;
        std::map<void*,uint8_t*> dlsymRealAddrGOTEntryMap; //Convert dynamically loaded symbol address to slots in dlSymIdSavers

        //The reaon for seperate store is for efficiency. These memory cannot be freed once allocated. To support dynamic loading, we make it 
        ObjectPoolHeap<IdSaverBinWrapper> dlSymIdSavers; //Holds the idsaver installed by dlsym.
        Array<uint8_t*> relaIdSavers; //Holds idsaver of individual symbols installed install() and dlopen_proxy(), used in replacePltEntry
        Array<uint8_t**> relaDynRealAddrSavers; //Holds the real address of .got.plt entries
        
        HashMap<std::string, SymbolHookHint,ObjectPoolHeap> hookHintMap;
        /**
         * Private constructor
         */

        inline bool isSymbolAddrResolved(FunctionInfo &symInfo) {
            //Check whether its value has 6 bytes offset as its plt entry start address
            ELFImgInfo &curImg = elfImgInfoMap[symInfo.fileId];
            int64_t myPltStartAddr = (int64_t) curImg.pltStartAddr;
            int64_t curGotAddr = (int64_t) symInfo.realAddrPtr;
            int64_t offset = curGotAddr - myPltStartAddr;
            return offset > 6 || offset < -6;
        }

        static HookInstaller *getInstance(std::string folderName);

        static HookInstaller *getInstance();

        static HookInstaller *instance;

        void shouldHookThisSymbol(const char *funcName, Elf64_Word &bind, Elf64_Word &type,SymbolHookHint& symbolHookHint);

    protected:

        inline bool
        parseSecInfos(ELFParser &elfParser, ELFSecInfo &pltInfo, ELFSecInfo &pltSecInfo, ELFSecInfo &gotInfo,
                      uint8_t *baseAddr, uint8_t *startAddr, uint8_t *endAddr);

        bool
        parseSymbolInfo(ELFParser &parser, ssize_t fileId, uint8_t *baseAddr, ELFSecInfo &pltSection,
                        ELFSecInfo &pltSecureSection,
                        ELFSecInfo &gotSec, uint8_t *startAddr, uint8_t *endAddr);

        bool makeGOTWritable(ELFSecInfo &gotSec, bool writable);


        uint32_t parsePltStubId(uint8_t *dest);

        bool fillAddr2pltEntry(uint8_t *funcAddr, uint8_t *retPltEntry);

        bool fillAddrAndSymId2IdSaver(uint8_t **gotAddr, uint8_t *firstPltEntry, uint32_t symId,
                                      uint32_t pltStubId, uint32_t recArrayOffset, uint32_t countOffset,
                                       uint32_t gapOffset, uint8_t *idSaverEntry);

        inline bool parseRelaSection(ssize_t& validRelaEntrySize, const FunctionType& symbolType, ELFParser& parser, ELFImgInfo &curImgInfo, FileID fileId, ELFSecInfo &pltSection,
                              ELFSecInfo &pltSecureSection, Elf64_Rela*& relaSection, ssize_t relaEntrySize, FILE *symInfoFile, uint8_t *baseAddr, uint8_t *startAddr, uint8_t *endAddr);


        inline void parseRequiredInfo();

        /**
         * Actual entry
         * @return
         */
        bool replacePltEntry();

        void createRecordingFolder() const;

        void parseTLSOffset();
    };

}


extern "C" {


__pid_t fork(void);

}


#endif

#endif
