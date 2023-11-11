/*
@author: Steven (Jiaxun) Tang <jtang@umass.edu>
@author: Tongping Liu <tongping.liu@bytedance.com>
*/
#ifndef __DRIVER_MEMORY_H__
#define __DRIVER_MEMORY_H__

#include <sys/types.h>
#include "analyse/CommonMemory.h"
#include "common/CallStack.h"

namespace mlinsight{


/* For OOM, there are four reasons:
    1. external fragmentation (memory blocks inside the torch allocator but can't be used for large allocation due to discontinous objects)
    2. internal fragmentation (how much memory wasted due to unaligned memory allocations)
    3. Memory leaks from specific callsites. 
    4. Actual memory usage is larger than the capacity of GPU memory

In first stage, we aims to understand the possibility of each reason, but not necessarily of 
the detailed information. 
    1. For external fragmentation, we could track all freed objects inside the torch allocator and the pointer of un-used memory
    2. For internal fragmentation, we will track the total waste for each allocation, and deduct it for each free
    3. For memory leaks, we could track the number of allocations and deallocations for each cycle. However, how can we know the cycle?
       Or we could just use the trend of allocations (we will use 100 allocations as a pseudo cycle)
    4. For actual memory usage, driver + nondriver > capacity
*/

class DriverObject {
public:
    ssize_t size=0;
    CallStack<void*, CPP_CALL_STACK_LEVEL> callstack;
    bool    isTorchAlloc=false;

public:

    /**
     *
     * @param sz
     * @param cstack Must pass rvalue to save overhead. Do not support lvalue.
     * @param level
     * @param isTorchAllocation
     */
    DriverObject(ssize_t sz, CallStack<void*, CPP_CALL_STACK_LEVEL>& cstack, bool isTorchAllocation):size(sz), callstack(cstack),
                                                                    isTorchAlloc(isTorchAllocation) {
    }

};

class DriverMemory {
public:
    MemBasic basic;
    MemAlloc<DriverObject*> alloc;
    ssize_t alivePytorchMemory = 0;
    ssize_t aliveNormalMemory = 0;
public:
    DriverMemory(){
    }
}; 


class MemInfo {
public:
    MemBasic total;  // curUsage can be read from nvidia-smi
    ssize_t  totalMemory; 
    DriverMemory  driver; 

public:
    MemInfo():driver() {
        // Initiliaze the toal information
        total.curUsage = 0;
        total.peakUsage = 0;
        totalMemory = 0;
    }
};



/*
 For driver  or pytorch memory allocations, we will maintain a hash map to track all allocated objects. 
*/

void trackDriverAllocation(ssize_t size, void * devicePtr);
void trackDriverFree(void * devicePtr);
void reportMemoryProfile(ssize_t reportMemoryProfile);

}
#endif