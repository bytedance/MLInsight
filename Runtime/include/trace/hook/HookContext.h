/*
@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#ifndef MLINSIGHT_HOOKCONTEXT_H
#define MLINSIGHT_HOOKCONTEXT_H

#include <cstdio>
#include <atomic>

#include "HookInstaller.h"
#include "trace/type/RecordingDataStructure.h"
#include "common/Array.h"
#include "common/RingBuffer.h"
#include "trace/type/PyCodeExtra.h"

namespace mlinsight{

extern uint32_t threadNum; //The current number of threads
extern std::atomic<uint64_t> wallclockSnapshot; //Used as update timestamp
extern uint64_t logicalClock;

#define MAX_CALL_DEPTH 64 //N+1 because of dummy variable
#define MAX_DLOPEN_NUMBER 4096 //N+1 because of dummy variable


union IdUnion
{
  int64_t symId;
  int64_t fileId;
};

struct HookTuple {
    uint64_t callerAddr; //8
    uint64_t logicalClockCycles; //8
    IdUnion id; //8
};

struct HookContext {
    /**
     * Frequently read/write variables
     */
    uint64_t cachedWallClockSnapshot;  //8bytes
    uint64_t cachedLogicalClock; //8bytes
    uint32_t indexPosi;//4bytes
    uint32_t cachedThreadNum; //4bytes
    //Variables used to determine whether it's called by hook handler or not
    int64_t threadCreatorFileId = 1; //Which library created the current thread? The default one is main thread
    HookInstaller *_this = nullptr; //8bytes
    HookTuple hookTuple[MAX_CALL_DEPTH]; //8bytes aligned
    Array<RecTuple> recordArray; 
    //Records which symbol is called for how many times, the index is mlinsight id (Only contains hooked function)
    RingBuffer<PyCodeExtra*> callStackRingBuffer=RingBuffer<PyCodeExtra*>(32);

    /**
     * Infrequently read/write variables
     */
    uint64_t threadExecTime; //Used for application time attribution
    pthread_t threadId;
    uint8_t dataSaved = false;
    uint8_t isMainThread = false;
    uint8_t initialized = 0;
};

const uint8_t MLINSIGHT_TRUE = 145;
const uint8_t MLINSIGHT_FALSE = 167;

void exitHandler(HookContext *context, bool finalize = false);
void saveData(HookContext *context, bool finalize);


extern __thread HookContext *curContext;

extern __thread uint8_t bypassCHooks; //Anything that is not MLINSIGHT_FALSE should be treated as MLINSIGHT_FALSE

extern mlinsight::Array<HookContext *> threadContextMap;

extern pthread_mutex_t threadDataSavingLock;

bool initTLS();

void populateRecordingArray(mlinsight::HookInstaller& inst);


}
#endif