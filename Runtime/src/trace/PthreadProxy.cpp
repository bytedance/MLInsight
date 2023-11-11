/*

@author: Steven (Jiaxun) Tang <jtang@umass.edu>
@author: Tongping Liu <tongping.liu@bytedance.com>
*/
#include "trace/proxy/PthreadProxy.h"
#include "trace/hook/HookContext.h"

#include "analyse/LogicalClock.h"

namespace mlinsight{

// Define structure to hold arguments for dummy thread function
struct dummy_thread_function_args {
    void *(*actual_thread_function)(void *data);

    void *pthreadCreateRetAddr;
    void *data;
};


// Define the dummy thread function
// Entering this function means the thread has been successfully created
// Instrument thread beginning, call the original thread function, instrument thread end
void *dummy_thread_function(void *data) {
    //INFO_LOGS("pthread_create thread %p initialized",pthread_self());
    /**
     * Perform required actions at beginning of thread
     */
    initTLS();
    /**
    * Update logical clock
    */
    uint64_t curLogicalClock = threadCreatedRecord(curContext->cachedWallClockSnapshot, curContext->cachedLogicalClock,
                                                   curContext->cachedThreadNum);
    curContext->threadExecTime = curLogicalClock;
//    INFO_LOGS("Thread %ld creates at logical clock %ld", pthread_self(), curContext->threadExecTime);

    /**
     * Call actual thread function
     */
    // Extract arguments and call actual thread function
    auto *args = static_cast<dummy_thread_function_args *>(data);
    void *argData = args->data;
    auto actualFuncPtr = args->actual_thread_function;
    void *pthreadCreateRetAddr = args->pthreadCreateRetAddr;
    free(args);
    args = nullptr;

    HookContext *curContextPtr = curContext;
    assert(curContextPtr != NULL);
    curContextPtr->threadCreatorFileId = curContextPtr->_this->pmParser.findFileIdByAddr(
            pthreadCreateRetAddr);
//    DBG_LOGS("Thread is created by %ld", curContextPtr->_this->pmParser.findFileIdByAddr(
//            pthreadCreateRetAddr));

    /**
     * Register this thread with the main thread
     */
    threadContextMap.pushBack(curContextPtr);

    void *threadFuncRet = actualFuncPtr(argData);
    /**
     * Perform required actions after each thread function completes
     */
//    curContextPtr->endTImestamp = getunixtimestampms();
    //printf("inside the end of thread!!\n");
    exitHandler(curContextPtr);
    return threadFuncRet;
}

extern bool installed;

// Main Pthread wrapper functions.
int pthread_create_proxy(pthread_t *thread, const pthread_attr_t *attr, void *(*start)(void *), void *arg) {
    //register uint64_t rbp asm ("rbp");
    //register uint64_t rsp asm ("rsp");c
    //void **callerAddr1 = reinterpret_cast<void **>(rbp+8);
    //void *callerAddrPtr = *reinterpret_cast<void **>(rsp + 0x8 + 0x40);
    //uint8_t callOpCode = *reinterpret_cast<uint8_t *>((uint64_t) callerAddrPtr - 0x5);
    //if (callOpCode != 0xE8 && callOpCode != 0xFF && callOpCode != 0x9A) {
    //    fatalError("Failed to parse the caller address for pthread_create")
    //    exit(-1);
    //}


    if (!installed) {
        //ERR_LOG("pthread_create failed because of not installed");

        return pthread_create(thread, attr, start, (void *) arg);
    }

    auto threadID = pthread_self();
    //INFO_LOGS("pthread_create %p", pthread_self());

    auto args = (struct dummy_thread_function_args *) malloc(sizeof(struct dummy_thread_function_args));
    args->actual_thread_function = start;
    args->data = arg;
    args->pthreadCreateRetAddr = __builtin_return_address(0);
    // Call the actual pthread_create

    int retVal = pthread_create(thread, attr, dummy_thread_function, (void *) args);

    HookContext *curContextPtr = curContext;
    //Attribute time to pthread_create

    return retVal;
}

}