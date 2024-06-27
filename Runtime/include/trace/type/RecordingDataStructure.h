/*

@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#ifndef MLINSIGHT_RECORDINGDATASTRUCTURE_H
#define MLINSIGHT_RECORDINGDATASTRUCTURE_H

#include <string>

namespace mlinsight {

    typedef ssize_t FileID;
    typedef ssize_t SymID;
    typedef ssize_t FuncID;

    //Please keep this struct in ascending order because installer will use PLACE_HOLDER_* variable to determine whether a variable needs to allocate an entry in the global shadow table or not.
    enum APIType : int8_t {
        /*APIs that does not need an entry in the global shadow table start*/
        NULL_API = 0,      //Uninitialized type
        PY_API = 1,     //Python API
        C_PLT_API = 3, //API that uses .plt.sec .plt and .got.plt
        C_DYN_API = 4, //API that uses .plt.got and .got
        C_DL_API = 5,  //API that is dynamically loaded
        /*APIs that needs an entry in the global shadow table end*/
    };

    union SharedAddr1 {
        uint8_t *pltEntryAddr; //Valid only when the type is C_PLT_API. Stores the address of .plt entry.
        uint8_t **gotEntryAddr; //Valid only when the type is C_DYN_API. Stores the address of .got entry.
    };

    enum class SymbolSpecialHandlingMarker{
        NO_SPECIAL_HANDLING=0,
        DLSYM_RTLD_NEXT_BYPASS=1,
        DLVSYM_RTLD_NEXT_BYPASS=2
    };

    /**
    * Symbol information. Each entry
    */
    class APICallInfo {
    public:
        int64_t callerFileId = -1;//(8 bytes) Store fileID for this symbol. Depending on the apiType, this field maybe globalCFileId, globalPyModuleId, globalPySrcFileId, check description in HookInstaller.h.
        int64_t symIdInFile = -1; //(8 bytes) todo: change the definition of symbol id so we only need to save one id. (Put all symbols in one list, one simple id corresponds to one symbol in a file. Same simple in different file is considered as different)
        int64_t calleeFileId = -1; //(8 bytes) Stores the origin file that holds this symbol. Depending on the apiType, this field maybe globalCFileId, globalPyModuleId, globalPySrcFileId, check description in HookInstaller.h.
        uint64_t pltStubId = 0; //(8 bytes)
        int32_t initialGap = 0;//8 Bytes. Initial gap value
        uint8_t **realAddrPtr = nullptr; //(8 bytes) The address of a symbol. If not resolved, == nullptr //This can be modified without lock because all threads will resolve to the same value. Replacing this driverMemRecord with same value won't cause conflict.
        uint8_t *dlsymRealAddr = nullptr; //Dlsym loaded symbol do not have
        uint8_t *pltEntryAddr = nullptr; //Used for C_PLT_API
        uint8_t *pltSecEntryAddr = nullptr; //(8 bytes)
        uint8_t **gotEntryAddr; //Used for C_PLT_API and C_DYN_API
        void *addressOverride = nullptr; //(8 bytes) A metric recording that MLInsight wants to change the function to its own proxy funciton. Writes from the parsing stage and reads from the installation stage.
        APIType apiType = APIType::NULL_API; //The type of this symbol
        SymbolSpecialHandlingMarker specialHandlingMarker=SymbolSpecialHandlingMarker::NO_SPECIAL_HANDLING;
    };

    /**
     * This struct is the format that we record time and save to disk.
     */
    class RecTuple {
    public:
        uint64_t totalClockCycles = 0; //8
        uint64_t totalClockCyclesUnScaled = 0; //8
        int64_t count = 0; //8
        int32_t gap = 0; //4 Whether we will skip using (counter & gap). If gap is 0, we will do timing.
        // If gap is 0b11 (keeping the last two bits), then we will do timing at 0, 4, 8, xxx.
        float meanClockTick = 0; //4
        uint32_t flags = 0; //4
    };
}
#endif