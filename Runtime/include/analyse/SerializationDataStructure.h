/*
@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#ifndef MLINSIGHT_SERIALIZATIONDATASTRUCTURE_H
#define MLINSIGHT_SERIALIZATIONDATASTRUCTURE_H


/**
 * This struct stores the total allocatedSize and element allocatedSize of an array.
 * On disk, this struct is followed by array elements
 */
struct ArrayDescriptor {
    uint64_t arrayElemSize;
    uint64_t arraySize;
    uint8_t magicNum = 167;  //1 Used to ensure the collected data format is recognized in python scripts.
};

/**
 * This struct is the format that we record detailed timing and save to disk.
 */
typedef int64_t TIMING_TYPE;

struct DetailedTimingDescriptor {
    TIMING_TYPE timingSize;
};

struct ThreadCreatorInfo {
    uint64_t threadCreatorFileId;
    uint64_t threadExecutionCycles;
    uint8_t magicNum = 167;  //1 Used to ensure the collected data format is recognized in python scripts.
};


#endif
