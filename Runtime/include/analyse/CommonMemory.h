#ifndef __COMMON_MEMORY_H__
#define __COMMON_MEMORY_H__

#include <sys/types.h>
#include <unordered_map>
#include "common/LinkedList.h"
#include "common/HashAndCompareFunctions.h"
//#include "common/HashMap.h"

namespace mlinsight{

class MemBasic {
public:
    ssize_t curUsage=0;
    ssize_t peakUsage=0;
};

class PotentialLeakOject {
public:
    size_t size=0;
    size_t aliveCount=0;
    size_t aliveMem=0;
  
    PotentialLeakOject(int size):size(size){
    }
};

template <class OBJ_TYPE>
class MemAlloc {
public:
    ssize_t numAllocs=0; // The number of allocations in total
    ssize_t numFrees=0;  // The number of deallocations in total
    ssize_t allocMem=0;  // The total of allocated memory in bytes
    ssize_t freeMem=0;   // The total of freed memory in bytes

    // Tracking alive objects that are allocated but not freed objects
    // This is important to analyze OOM failures
    ssize_t memAliveObjs=0; // total memory of all alive objects
    ssize_t numAliveObjs=0; // number of all alive objects

    // We will maintain a map between the object address and its related information :
    // such as allocatedSize, callsite. Such a map is important, as there is no allocatedSize information for deallocations.
    // In the end, we will use this map to find out memory leaks. 
    std::unordered_map<void *, OBJ_TYPE> mapAliveObjs; // the map of alive objects

    //Force right hand operator initialization to ensure speed
    MemAlloc(){

    }
};
}

#endif