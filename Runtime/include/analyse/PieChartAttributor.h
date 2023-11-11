#ifndef __MLINSIGHT_PIECHARTATTRIBUTOR_H__
#define __MLINSIGHT_PIECHARTATTRIBUTOR_H__

#include "trace/hook/HookContext.h"
#include "analyse/LogicalClock.h"

namespace mlinsight {
    inline void preHookAttribution(HookContext *curContextPtr) {
        curContextPtr->hookTuple[curContextPtr->indexPosi].logicalClockCycles =
                calcCurrentLogicalClock(curContextPtr->cachedWallClockSnapshot,
                                        curContextPtr->cachedLogicalClock, curContextPtr->cachedThreadNum);
    }

    inline void postHookAttribution(HookContext *curContextPtr) {
        uint64_t preLogicalClockCycle = curContextPtr->hookTuple[curContextPtr->indexPosi].logicalClockCycles;

        uint64_t postLogicalClockCycle = calcCurrentLogicalClock(curContextPtr->cachedWallClockSnapshot,
                                                                 curContextPtr->cachedLogicalClock,
                                                                 curContextPtr->cachedThreadNum);
        uint64_t clockCyclesDuration = (int64_t) (postLogicalClockCycle - preLogicalClockCycle);
        //Attribute scaled clock cycle to API
        mlinsight::SymID symbolId = curContextPtr->hookTuple[curContextPtr->indexPosi].id.symId;

        curContextPtr->recordArray.internalArray[symbolId].totalClockCycles += clockCyclesDuration;
    }

    inline uint64_t pyPreHookAttribution(HookContext* curContextPtr){
        uint64_t preLogicalClockCycle = calcCurrentLogicalClock(curContextPtr->cachedWallClockSnapshot,
                                                                curContextPtr->cachedLogicalClock, curContextPtr->cachedThreadNum);
        return preLogicalClockCycle;
    }

    inline void pyPostHookAttribution(uint64_t preHookTimestamp,HookContext* curContextPtr,RecTuple& curRecTuple){
        uint64_t postLogicalClockCycle = calcCurrentLogicalClock(curContextPtr->cachedWallClockSnapshot,
                                                                 curContextPtr->cachedLogicalClock, curContextPtr->cachedThreadNum);
        uint64_t clockCyclesDuration = (int64_t) (postLogicalClockCycle - preHookTimestamp);
        curRecTuple.totalClockCycles+=clockCyclesDuration;
    }
}

#endif
