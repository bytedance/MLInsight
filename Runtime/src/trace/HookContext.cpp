/*

@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#include <cxxabi.h>
#include <cassert>
#include "trace/hook/HookContext.h"

#include "common/Tool.h"
#include "analyse/SerializationDataStructure.h"
#include "trace/tool/AtomicSpinLock.h"
#include "analyse/LogicalClock.h"
#include "trace/type/RecordingDataStructure.h"
#include "common/Array.h"
#include "analyse/DriverMemory.h"


namespace mlinsight{
uint64_t logicalClock;
std::atomic<uint64_t> wallclockSnapshot;
uint64_t updateSpinlock;
AtomicSpinLock threadUpdateLock; //Lock used in LogicalClock.h to update thread counters
uint32_t threadNum = 0; //Actual thread number recorded

HookContext *constructContext(mlinsight::HookInstaller &inst) {
    HookContext *ret = new HookContext();
    if (!ret) {
        fatalError("Cannot allocate memory for HookContext")
    }
    //INFO_LOGS("inst.curLoadingId.load(std::memory_order_acquire)+1=%zd",inst.curLoadingId.load(std::memory_order_acquire)+1);

    //Push a dummy value in the stack (Especially for callAddr, because we need to avoid null problem)
    ret->hookTuple[ret->indexPosi].callerAddr = 0;
    ret->hookTuple[ret->indexPosi].logicalClockCycles = 0;
    ret->hookTuple[ret->indexPosi].id.symId = 0;
    ret->indexPosi = 1;

    __atomic_store_n(&ret->dataSaved, false, __ATOMIC_RELEASE);

    assert(mlinsight::HookInstaller::instance != nullptr);
    ret->_this = mlinsight::HookInstaller::instance;

    ret->threadId = pthread_self();

    return ret;
}

bool destructContext() {
    HookContext *curContextPtr = curContext;
    munmap(curContextPtr, sizeof(HookContext) +
                          sizeof(mlinsight::Array<uint64_t>) +
                          sizeof(pthread_mutex_t));
    curContext = nullptr;
    return true;
}


void __attribute__((used, noinline, optimize(3))) printRecOffset() {

    auto i = (uint8_t *) curContext;
    auto j  = (uint8_t *) &curContext->recordArray;
    auto k  = (uint8_t *) &curContext->recordArray.internalArray;
    auto l  = (uint8_t *) &curContext->recordArray.internalArray[0];
    auto m  = (uint8_t *) &curContext->recordArray.internalArray[0].count;
    auto n  = (uint8_t *) &curContext->recordArray.internalArray[0].gap;

    printf("\nTLS offset: Check assembly\n"
           "LDARR_OFFSET_IN_CONTEXT: 0x%lx\n"
           "INTERNALARR_OFFSEjT_IN_LDARR Offset: 0x%lx\n"
           "COUNT_OFFSET_IN_RECARR: 0x%lx\n"
           "GAP_OFFSET_IN_RECARR: 0x%lx\n", j - i, k - j, m - l, n - l);


}


bool initTLS() {
    //INFO_LOG("initTLS is called here");
    assert(mlinsight::HookInstaller::instance != nullptr);
    mlinsight::HookInstaller &inst = *mlinsight::HookInstaller::getInstance();
    //INFO_LOGS("thread:%p pthread_mutex_lock(&inst->dynamicLoadingLock)",pthread_self());

    pthread_mutex_lock(&(inst.dynamicLoadingLock));
    //Put a dummy variable to avoid null checking
    //Initialize saving data structure
    curContext = constructContext(inst);

    //INFO_LOGS("initTLS: curContext %p\n", curContext);
    
#ifdef PRINT_DBG_LOG
    //printRecOffset();
#endif


    if (!curContext) {
        pthread_mutex_unlock(&inst.dynamicLoadingLock);
        fatalError("Failed to allocate memory for Context");
        return false;
    }
    //Populate the recording array of existing loading ids

    populateRecordingArray(inst);

    curContext->initialized = MLINSIGHT_TRUE;
    //INFO_LOGS("thread:%p pthread_mutex_unlock(&inst->dynamicLoadingLock)",pthread_self());

    pthread_mutex_unlock(&inst.dynamicLoadingLock);

    return true;
}

void populateRecordingArray(mlinsight::HookInstaller& inst) {
    //Do not need to acquire lock because lock ƒhas been acquired in
    //No contention because parent function will acquire a lock
    //Allocate recArray
    HookContext* curContextPtr=curContext;
    curContextPtr->recordArray.allocateArray(inst.allExtSymbol.getSize());

    //Initialize gap to one
    for (ssize_t symId = 0; symId < inst.allExtSymbol.getSize(); ++symId) {
        //number mod 2^n is equivalent to stripping off all but the n lowest-order
        curContextPtr->recordArray[symId].gap = inst.allExtSymbol[symId].initialGap;//0b11 if %4, because 4=2^2 Initially time everything
        curContextPtr->recordArray[symId].count = 0;
    }

}


__thread HookContext *curContext __attribute((tls_model("initial-exec")));
__thread uint8_t bypassCHooks __attribute((tls_model("initial-exec"))) = MLINSIGHT_FALSE; //Anything that is not MLINSIGHT_FALSE should be treated as MLINSIGHT_FALSE

inline void savePerThreadTimingData(std::stringstream &ss, HookContext *curContextPtr) {
    ss.str("");
    ss << mlinsight::HookInstaller::instance->folderName << "/threadTiming_" << curContextPtr->threadId << ".bin";
    //INFO_LOGS("Saving timing data to %s", ss.str().c_str());

    int fd;
    size_t realFileIdSizeInBytes =
            sizeof(ThreadCreatorInfo) + sizeof(ArrayDescriptor) + curContextPtr->recordArray.getSize() * sizeof(mlinsight::RecTuple);
    uint8_t *fileContentInMem = nullptr;
    if (!mlinsight::fOpen4Write<uint8_t>(ss.str().c_str(), fd, realFileIdSizeInBytes, fileContentInMem)) {
        fatalErrorS("Cannot fopen %s because:%s", ss.str().c_str(), strerror(errno));
    }
    uint8_t *_fileContentInMem = fileContentInMem;
    /**
     * Record who created the thread
     */
    ThreadCreatorInfo *threadCreatorInfo = reinterpret_cast<ThreadCreatorInfo *>(fileContentInMem);
    threadCreatorInfo->threadExecutionCycles = curContextPtr->threadExecTime;
    threadCreatorInfo->threadCreatorFileId = curContextPtr->threadCreatorFileId;
    threadCreatorInfo->magicNum = 167;
    fileContentInMem += sizeof(ThreadCreatorInfo);

    /**
     * Record allocatedSize information about the recorded array
     */
    ArrayDescriptor *arrayDescriptor = reinterpret_cast<ArrayDescriptor *>(fileContentInMem);
    arrayDescriptor->arrayElemSize = sizeof(mlinsight::RecTuple);
    arrayDescriptor->arraySize = curContextPtr->recordArray.getSize();
    arrayDescriptor->magicNum = 167;
    fileContentInMem += sizeof(ArrayDescriptor);

//    for(int i=0;i<curContextPtr->recordArray->getSize();++i){
//        if(curContextPtr->recordArray->internalArray[i].count>0){
//            printf("%ld\n",curContextPtr->recordArray->internalArray[i].count);
//        }
//    }
    /**
     * Write recording tuple onto the disk
     */
    memcpy(fileContentInMem, curContextPtr->recordArray.data(),
           curContextPtr->recordArray.getTypeSizeInBytes() * curContextPtr->recordArray.getSize());

    if (!mlinsight::fClose<uint8_t>(fd, realFileIdSizeInBytes, _fileContentInMem)) {
        fatalErrorS("Cannot close file %s, because %s", ss.str().c_str(), strerror(errno));
    }

    DBG_LOGS("Saving data to %s, %lu", mlinsight::HookInstaller::instance->folderName.c_str(), pthread_self());
}

//inline void saveApiInvocTimeByLib(std::stringstream &ss, HookContext *curContextPtr){
//    ss.str("");
//    ss << mlinsight::HookInstaller::instance->folderName << "/apiInvocTimeByLib_"<< curContextPtr->threadId << ".bin";
//    //The real id of each function is resolved in after hook, so I can only save it in datasaver
//
//    int fd;
//    ssize_t selfTimeSizeInBytes = sizeof(ArrayDescriptor) + (curContextPtr->selfTimeArr->getSize()) * sizeof(uint64_t);
//    uint8_t *fileContentInMem = nullptr;
//    if (!mlinsight::fOpen4Write<uint8_t>(ss.str().c_str(), fd, selfTimeSizeInBytes, fileContentInMem)) {
//        fatalErrorS(
//                "Cannot open %s because:%s", ss.str().c_str(), strerror(errno))
//    }
//    uint8_t *_fileContentInMem = fileContentInMem;
//
//    /**
//     * Write array descriptor first
//     */
//    ArrayDescriptor *arrayDescriptor = reinterpret_cast<ArrayDescriptor *>(fileContentInMem);
//    arrayDescriptor->arrayElemSize = sizeof(uint64_t);
//    arrayDescriptor->arraySize = curContextPtr->selfTimeArr->getSize();
//    arrayDescriptor->magicNum = 167;
//    fileContentInMem += sizeof(ArrayDescriptor);
//
//    uint64_t *realFileIdMem = reinterpret_cast<uint64_t *>(fileContentInMem);
//    for (int i = 0; i < curContextPtr->selfTimeArr->getSize(); ++i) {
//        realFileIdMem[i] = curContextPtr->selfTimeArr->internalArray[i];
//    }
//
//    if (!mlinsight::fClose<uint8_t>(fd, selfTimeSizeInBytes, _fileContentInMem)) {
//        fatalErrorS("Cannot close file %s, because %s", ss.str().c_str(), strerror(errno));
//    }
//}

inline void saveRealFileId(std::stringstream &ss, HookContext *curContextPtr) {
    ss.str("");
    ss << mlinsight::HookInstaller::instance->folderName << "/realFileId.bin";
    //The real id of each function is resolved in after hook, so I can only save it in datasaver

    int fd;
    ssize_t realFileIdSizeInBytes = sizeof(ArrayDescriptor) +
                                    (curContextPtr->_this->allExtSymbol.getSize()) * sizeof(uint64_t);
    uint8_t *fileContentInMem = nullptr;
    if (!mlinsight::fOpen4Write<uint8_t>(ss.str().c_str(), fd, realFileIdSizeInBytes, fileContentInMem)) {
        fatalErrorS("Cannot open %s because:%s", ss.str().c_str(), strerror(errno))
    }
    uint8_t *_fileContentInMem = fileContentInMem;

    /**
     * Write array descriptor first
     */
    ArrayDescriptor *arrayDescriptor = reinterpret_cast<ArrayDescriptor *>(fileContentInMem);
    arrayDescriptor->arrayElemSize = sizeof(uint64_t);
    arrayDescriptor->arraySize = curContextPtr->_this->allExtSymbol.getSize();
    arrayDescriptor->magicNum = 167;
    fileContentInMem += sizeof(ArrayDescriptor);

    uint64_t *realFileIdMem = reinterpret_cast<uint64_t *>(fileContentInMem);
    for (int i = 0; i < curContextPtr->_this->allExtSymbol.getSize(); ++i) {
        realFileIdMem[i] = curContextPtr->_this->pmParser.findFileIdByAddr(
                *(curContextPtr->_this->allExtSymbol[i].realAddrPtr));
    }

    if (!mlinsight::fClose<uint8_t>(fd, realFileIdSizeInBytes, _fileContentInMem)) {
        fatalErrorS("Cannot close file %s, because %s", ss.str().c_str(), strerror(errno));
    }
}

inline void saveDataForAllOtherThread(std::stringstream &ss, HookContext *curContextPtr) {
    DBG_LOG("Save data of all existing threads");
    for (int i = 0; i < threadContextMap.getSize(); ++i) {
        HookContext *threadContext = threadContextMap[i];
        saveData(threadContext, false);
    }
}

void saveData(HookContext *curContextPtr, bool finalize) {
    bypassCHooks = MLINSIGHT_TRUE;
    // if(logFileStd){
    //     //Flush log all the time
    //     fflush(logFileStd);
    // }
    /* CS: Check whether data for the current thread has been saved. Make sure data is only saved once */
    if (__atomic_test_and_set(&curContextPtr->dataSaved, __ATOMIC_ACQUIRE)) {
        //INFO_LOGS("Thread data already saved, skip %d/%zd", i, threadContextMap.getSize());
        return;
    }
    /* CS: End */


    if (finalize == true) {
        //The main thread ends
        reportMemoryProfile(0);
        OUTPUT("MLInsight memory profile logs has been saved memoryprofile_*.txt.\n");
    }
    return;

    uint64_t curLogicalClock = threadTerminatedRecord();
//    INFO_LOGS("AttributingThreadEndTime+= %lu - %lu", curLogicalClock, curContextPtr->threadExecTime);
    curContextPtr->threadExecTime = curLogicalClock -
                                    curContextPtr->threadExecTime; //curContextPtr->threadExecTime is set to logical clock in the beginning


    std::stringstream ss;

    savePerThreadTimingData(ss, curContextPtr);
//    saveApiInvocTimeByLib(ss, curContextPtr);

    if (curContextPtr->isMainThread || finalize) {
//        INFO_LOGS("Data saved to %s", mlinsight::HookInstaller::instance->folderName.c_str());
        saveRealFileId(ss, curContextPtr);
        saveDataForAllOtherThread(ss, curContextPtr);
    }

}


}

