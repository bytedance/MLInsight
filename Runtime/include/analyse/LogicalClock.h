/*
@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#ifndef MLINSIGHT_LOGICALCLOCK_H
#define MLINSIGHT_LOGICALCLOCK_H

#include <atomic>

#include "trace/tool/AtomicSpinLock.h"
#include "common/Tool.h"

namespace mlinsight {
    extern uint64_t logicalClock; //The logical clock
    extern uint32_t threadNum; //The current number of threads
    extern std::atomic<uint64_t> wallclockSnapshot; //Used as update timestamp
    extern AtomicSpinLock threadUpdateLock;

/**
 * Initialize logical clock, should be invoked before main thread starts execution
 * @param cachedWallClockSnapshot Reference to thread local timestamp used to determine modification or not.
 * @return Current wall clock snapshot
 */
    inline uint64_t
    initLogicalClock(uint64_t &cachedWallClockSnapshot, uint64_t &cachedLogicalClock, uint32_t &cachedThreadNum) {
        threadUpdateLock.lock();
        threadNum = 1;
        cachedThreadNum = 1;
        logicalClock = 0; //Logical clock should be 0 at first
        cachedLogicalClock = 0;
        uint64_t curWallclockTimestamp = getunixtimestampms();
        cachedWallClockSnapshot = curWallclockTimestamp;

        wallclockSnapshot.store(curWallclockTimestamp,
                                std::memory_order_release); //Publish thread number and logical clock

//    INFO_LOGS("Logical clock value initialized to = %lu", logicalClock);
        threadUpdateLock.unlock();
        return wallclockSnapshot;
    }

/**
 * Update logical clock when thread creates/finished
 * Should only be invoked by API functions
 * @param threadNumChange +1 means a thread is being created, -1 means a thread is being destroyed
 * @param cachedWallClockSnapshot Used to prevent first lock. For thread creation, pass thread local variable. For thread termination, pass dummy value,
 * @return
 */
    inline uint64_t
    updateLogicalClockAndThreadNum(int8_t threadNumChange, uint64_t &cachedWallClockSnapshot,
                                   uint64_t &cachedLogicalClock,
                                   uint32_t &cachedThreadNum) {
        threadUpdateLock.lock(); //Make sure that the following non-atomic operation will not cause data race.
        uint64_t curWallClockCurSnapshot = getunixtimestampms(); //Record current time
        uint64_t prevWallClockSnapshot = wallclockSnapshot.load(std::memory_order_acquire);

        logicalClock += (curWallClockCurSnapshot - prevWallClockSnapshot) / threadNum; //Update logical clock
        threadNum += threadNumChange; //Change thread number

        /**
         * Update cache to prevent first time locking
         */
        cachedWallClockSnapshot = curWallClockCurSnapshot;
        cachedThreadNum = threadNum;
        cachedLogicalClock = logicalClock;

        //Update cached wallclocksnapshot to avoid first time lock
        wallclockSnapshot.store(curWallClockCurSnapshot, std::memory_order_release);
        threadUpdateLock.unlock();

        return cachedLogicalClock;
    }

/**
 * API interface, use the following functions in hook
 */
    inline uint64_t
    threadCreatedRecord(uint64_t &cachedWallClockSnapshot, uint64_t &cachedLogicalClock, uint32_t &cachedThreadNum) {

        return updateLogicalClockAndThreadNum(1, cachedWallClockSnapshot, cachedLogicalClock, cachedThreadNum);
    }

    inline uint64_t threadTerminatedRecord() {
        uint64_t tmp; //We do not need to update cachedWallClockSnapshot at exit
        uint32_t tmp1;
        return updateLogicalClockAndThreadNum(-1, tmp, tmp, tmp1);
    }

    inline uint64_t
    calcCurrentLogicalClock(uint64_t &cachedWallClockSnapshot, uint64_t &cachedLogicalClock,
                            uint32_t &cachedThreadNum) {
        uint64_t prevWallClockSnapshot = wallclockSnapshot.load(std::memory_order_acquire);
        //Updates performed by wallclockSnapshot must have been done. (Ensured by C++11 standard)
        uint64_t result = 0;
        if (prevWallClockSnapshot != cachedWallClockSnapshot) {
            //There is change to timestamp, get the lock and acquire again to ensure there are no data race on "threadNum" variable and "logicalClock" variable
            threadUpdateLock.lock();
            prevWallClockSnapshot = wallclockSnapshot.load(std::memory_order_acquire);
            //Updates performed by wallclockSnapshot must have been done. (Ensured by C++11 standard)
            cachedWallClockSnapshot = prevWallClockSnapshot;
            cachedThreadNum = threadNum;
            cachedLogicalClock = logicalClock;
            uint64_t curWallClockSnapshot = getunixtimestampms();

            result = (curWallClockSnapshot - prevWallClockSnapshot) / cachedThreadNum + cachedLogicalClock;

            threadUpdateLock.unlock();
        } else {
            uint64_t curWallClockSnapshot = getunixtimestampms();

            result = (curWallClockSnapshot - cachedWallClockSnapshot) / cachedThreadNum + cachedLogicalClock;

        }
        return result;
    }
}
#endif
