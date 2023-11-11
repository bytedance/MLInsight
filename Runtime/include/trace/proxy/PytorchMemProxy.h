#ifndef __PYTORCH_MEM_PROXY_H__
#define __PYTORCH_MEM_PROXY_H__
#include <c10/core/Allocator.h>
#include <map>
namespace mlinsight{
extern pthread_mutex_t pytorchMemoryManagementLock;

typedef void (*raw_delete_t)(void*);
typedef c10::Allocator* (*AllocatorGet_t)(void);

extern raw_delete_t realRawDeletePtr;
extern AllocatorGet_t realAllocatorGetPtr;
extern void* realGetDeviceStatsPtr;

void raw_delete_proxy(void* ptr);
c10::Allocator* allocator_get_proxy(void);

extern std::map<int,double> cudaCachingAllocatorFractionMap;
void setMemoryFraction_proxy(double fraction, int device);

}

#endif