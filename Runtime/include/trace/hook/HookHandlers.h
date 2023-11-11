/*

@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#ifndef MLINSIGHT_HOOKHANDLERS_H
#define MLINSIGHT_HOOKHANDLERS_H

#include "trace/type/RecordingDataStructure.h"

extern "C"{


extern uint8_t *callIdSavers;
void __attribute__((naked)) asmTimingHandler();

/**
* A handler written in C. It calls custom handler and calculates actual function address
* In the new code, .plt and .plt.sec uses the same handler. Since we currently don't calculate
* based on the first address.
* @param callerFuncAddr The next caller
* @param oriRBPLoc The rsp location before saving all registers
* @return Original function pointer
*/
__attribute__((used)) void *preHookHandler(uint64_t nextCallAddr,int64_t symId,void** realAddrPtr);


__attribute__((used)) void *postHookHandler();

}
#endif