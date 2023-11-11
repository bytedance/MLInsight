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


    enum FunctionType : int8_t{
        NULL_API = 0,      //Uninitialized type
        C_PLT_API = 1, //API that uses .plt.sec .plt and .got.plt 
        C_DYN_API = 2, //API that uses .plt.got and .got
        C_DL_API = 3,  //API that is dynamically loaded
        PY_API = 4     //Python API
    };

    union SharedAddr1{
        uint8_t*  pltEntryAddr; //Valid only when the type is C_PLT_API. Stores the address of .plt entry.
        uint8_t**  gotEntryAddr; //Valid only when the type is C_DYN_API. Stores the address of .got entry.
    };


    /**
    * Symbol information
    */
    class FunctionInfo {
    public:
        int64_t fileId = -1;//(8 bytes) Store fileID for this symbol
        int64_t symIdInFile = -1; //(8 bytes) todo: change the definition of symbol id so we only need to save one id. (Put all symbols in one list, one simple id corresponds to one symbol in a file. Same simple in different file is considered as different)
        uint64_t pltStubId = 0; //(8 bytes)
        int32_t initialGap = 0;//8 Bytes. Initial gap value
        uint8_t **realAddrPtr = nullptr; //(8 bytes) The address of a symbol. If not resolved, == nullptr //This can be modified without lock because all threads will resolve to the same value. Replacing this memory with same value won't cause conflict.
        SharedAddr1 sharedAddr1; //(8 bytes)
        uint8_t *pltSecEntryAddr = nullptr; //(8 bytes)
        void *addressOverride = nullptr; //(8 bytes) A metric recording that MLInsight wants to change the function to its own proxy funciton. Writes from the parsing stage and reads from the installation stage.
        FunctionType symbolType =FunctionType::NULL_API; //The type of this symbol
    };

    /**
     * This struct is the format that we record time and save to disk.
     */
    class RecTuple {
    public:
        uint64_t totalClockCycles=0; //8
        uint64_t totalClockCyclesUnScaled=0; //8
        int64_t count=0; //8
        int32_t gap=0; //4 Whether we will skip using (counter & gap). If gap is 0, we will do timing. 
                       // If gap is 0b11 (keeping the last two bits), then we will do timing at 0, 4, 8, xxx. 
        float meanClockTick=0; //4
        uint32_t flags=0; //4
    };
}
#endif