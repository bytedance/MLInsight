#ifndef MLINSIGHT_GPUEVENTTRACE_H
#define MLINSIGHT_GPUEVENTTRACE_H

#include <cupti_activity.h>

namespace mlinsight {
    static const char *getKindString(CUpti_ActivityKind kind);

    static const char *getMemoryOperationTypeString(CUpti_ActivityMemoryOperationType type);

    static const char *getMemorycpyKindString(uint8_t kind);

    static const char *getMemoryKindString(CUpti_ActivityMemoryKind kind);

    static const char *getMemoryPoolTypeString(CUpti_ActivityMemoryPoolType type);

    static const char *getMemoryPoolOperationTypeString(CUpti_ActivityMemoryPoolOperationType type);

    static void printActivity(CUpti_Activity *record);

    void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords);

    void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize);

    void initCuptiTrace();

    void finiTrace();

}

#endif
