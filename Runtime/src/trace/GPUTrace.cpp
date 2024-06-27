#include <trace/proxy/GPUEventTrace.h>
#include <stdio.h>
#include <string.h>
#include <cupti.h>
#include <stdlib.h>
#include <common/Logging.h>
#include "analyse/GlobalVariables.h"
#include <cuda.h>
#include <cassert>

namespace mlinsight {
/*
 * Copyright 2021 NVIDIA Corporation. All rights reserved
 *
 * Sample CUPTI app to print trace of CUDA driverMemRecord operations.
 * The sample also traces CUDA driverMemRecord operations done via
 * default driverMemRecord pool.
 *cu
 */



#define CUPTI_CALL(call)                                                          \
    do {                                                                          \
        CUptiResult _status = call;                                               \
        if (_status != CUPTI_SUCCESS) {                                           \
            const char *errstr;                                                   \
            cuptiGetResultString(_status, &errstr);                               \
            fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",  \
                    __FILE__, __LINE__, #call, errstr);                           \
            exit(EXIT_FAILURE);                                                    \
        }                                                                         \
    } while (0)

#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                               \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

    static const char *getKindString(CUpti_ActivityKind kind) {
        switch (kind) {
            case CUPTI_ACTIVITY_KIND_MEMCPY:
                return "memcpy";
            case CUPTI_ACTIVITY_KIND_MEMCPY2:
                return "memcpyp2p";
            default:
                return "<unknown>";
        }
    }


    static const char *getMemorycpyKindString(uint8_t kind) {
        switch (kind) {
            case CUPTI_ACTIVITY_MEMCPY_KIND_UNKNOWN:
                return "UNKNOWN";
            case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
                return "HTOD";
            case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
                return "DTOH";
            case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
                return "HTOA";
            case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
                return "ATOH";
            case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
                return "ATOA";
            case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
                return "ATOD";
            case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
                return "DTOA";
            case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
                return "DTOD";
            case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
                return "HTOH";
            case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
                return "PTOP";
            default:
                return "<unknown>";
        }
    }

    static const char *getMemoryKindString(uint8_t kind) {
        switch (kind) {
            case CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN:
                return "UNKNOWN";
            case CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE:
                return "PAGEABLE";
            case CUPTI_ACTIVITY_MEMORY_KIND_PINNED:
                return "PINNED";
            case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE:
                return "DEVICE";
            case CUPTI_ACTIVITY_MEMORY_KIND_ARRAY:
                return "ARRAY";
            default:
                return "<unknown>";
        }
    }

    static const char *getMemoryPoolTypeString(CUpti_ActivityMemoryPoolType type) {
        switch (type) {
            case CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL:
                return "LOCAL";
            case CUPTI_ACTIVITY_MEMORY_POOL_TYPE_IMPORTED:
                return "IMPORTED";
            default:
                return "<unknown>";
        }
    }

    static const char *getMemoryPoolOperationTypeString(CUpti_ActivityMemoryPoolOperationType type) {
        switch (type) {
            case CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_CREATED:
                return "MEM_POOL_CREATED";
            case CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_DESTROYED:
                return "MEM_POOL_DESTROYED";
            case CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_TRIMMED:
                return "MEM_POOL_TRIMMED";
            default:
                return "<unknown>";
        }
    }

#if CUPTI_API_VERSION > 14
    static const char *getMemorycpyChannelType(CUpti_ChannelType type) {
        switch (type) {
            case CUPTI_CHANNEL_TYPE_ASYNC_MEMCPY:
                return "ASYNC_MEMCPY";
            case CUPTI_CHANNEL_TYPE_COMPUTE:
                return "COMPUTE";
            case CUPTI_CHANNEL_TYPE_INVALID:
                return "INVALID";
            default:
                return "<unknown>";
        }
    }
#else

#endif
    static void printActivity(CUpti_Activity *record) {
        switch (record->kind) {
#if CUPTI_API_VERSION > 14 //todo: Inaccurate
            case CUPTI_ACTIVITY_KIND_MEMCPY: {
                CUpti_ActivityMemcpy5 *memory = (CUpti_ActivityMemcpy5 *) (void *) record;
                INFO_LOGS("%s copyKind:%s, srcKind:%s, dstKind:%s, bytes:%lu, start:%lu, end:%lu, devId:%u, ctxId:%u, "
                          "corrId:%u, streamId:%u, runtimeCorrelationId:%u, graphNodeId:%lu, graphId:%u, channelID:%u, channelType:%s isAsync:%s",

                          "CUPTI_ACTIVITY_KIND_MEMCPY",
                          getMemorycpyKindString(memory->copyKind),
                          getMemoryKindString(memory->srcKind),
                          getMemoryKindString(memory->dstKind),
                          memory->bytes,
                          memory->start,
                          memory->end,
                          memory->deviceId,
                          memory->contextId,
                          memory->correlationId,
                          memory->streamId,
                          memory->runtimeCorrelationId,
                          memory->graphNodeId,
                          memory->graphId,
                          memory->channelID,
                          getMemorycpyChannelType(memory->channelType),
                          (CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC & memory->flags) == 1 ? "Async" : "Sync");

                break;
            }
#endif
            case CUPTI_ACTIVITY_KIND_DRIVER: {
                CUpti_ActivityAPI *runtimeApi = (CUpti_ActivityAPI *) (void *) record;
                INFO_LOGS("[CUPTI reported CUDA driverRecord API call] CallBackID:%d CorrelationID:%d Start:%lu End:%lu",
                          runtimeApi->cbid, runtimeApi->correlationId, runtimeApi->start, runtimeApi->end);
                break;
            }
            default:
                break;
        }
    }

    void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
        INFO_LOG("Buffer requested");
        uint8_t *bfr = (uint8_t *) malloc(BUF_SIZE + ALIGN_SIZE);

        if (bfr == NULL) {
            OUTPUT("Error: out of driverMemRecord.\n");
            exit(EXIT_FAILURE);
        }

        *size = BUF_SIZE;
        *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
        *maxNumRecords = 0;
    }

    void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize) {
        INFO_LOG("Buffer completed");
        CUptiResult status;
        CUpti_Activity *record = NULL;

        if (validSize > 0) {
            do {
                status = cuptiActivityGetNextRecord(buffer, validSize, &record);
                if (status == CUPTI_SUCCESS) {
                    printActivity(record);
                } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
                    break;
                } else {
                    CUPTI_CALL(status);
                }
            } while (1);

            // report any records dropped from the queue
            size_t dropped;
            CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
            if (dropped != 0) {
                OUTPUTS("Warning: Dropped %u activity records.\n", (unsigned int) dropped);
            }
        }

        free(buffer);
    }

    const char* cuptiStrMap[]={"CUPTI_DRIVER_TRACE_CBID_INVALID","CUPTI_DRIVER_TRACE_CBID_cuInit","CUPTI_DRIVER_TRACE_CBID_cuDriverGetVersion","CUPTI_DRIVER_TRACE_CBID_cuDeviceGet","CUPTI_DRIVER_TRACE_CBID_cuDeviceGetCount","CUPTI_DRIVER_TRACE_CBID_cuDeviceGetName","CUPTI_DRIVER_TRACE_CBID_cuDeviceComputeCapability","CUPTI_DRIVER_TRACE_CBID_cuDeviceTotalMem","CUPTI_DRIVER_TRACE_CBID_cuDeviceGetProperties","CUPTI_DRIVER_TRACE_CBID_cuDeviceGetAttribute","CUPTI_DRIVER_TRACE_CBID_cuCtxCreate","CUPTI_DRIVER_TRACE_CBID_cuCtxDestroy","CUPTI_DRIVER_TRACE_CBID_cuCtxAttach","CUPTI_DRIVER_TRACE_CBID_cuCtxDetach","CUPTI_DRIVER_TRACE_CBID_cuCtxPushCurrent","CUPTI_DRIVER_TRACE_CBID_cuCtxPopCurrent","CUPTI_DRIVER_TRACE_CBID_cuCtxGetDevice","CUPTI_DRIVER_TRACE_CBID_cuCtxSynchronize","CUPTI_DRIVER_TRACE_CBID_cuModuleLoad","CUPTI_DRIVER_TRACE_CBID_cuModuleLoadData","CUPTI_DRIVER_TRACE_CBID_cuModuleLoadDataEx","CUPTI_DRIVER_TRACE_CBID_cuModuleLoadFatBinary","CUPTI_DRIVER_TRACE_CBID_cuModuleUnload","CUPTI_DRIVER_TRACE_CBID_cuModuleGetFunction","CUPTI_DRIVER_TRACE_CBID_cuModuleGetGlobal","CUPTI_DRIVER_TRACE_CBID_cu64ModuleGetGlobal","CUPTI_DRIVER_TRACE_CBID_cuModuleGetTexRef","CUPTI_DRIVER_TRACE_CBID_cuMemGetInfo","CUPTI_DRIVER_TRACE_CBID_cu64MemGetInfo","CUPTI_DRIVER_TRACE_CBID_cuMemAlloc","CUPTI_DRIVER_TRACE_CBID_cu64MemAlloc","CUPTI_DRIVER_TRACE_CBID_cuMemAllocPitch","CUPTI_DRIVER_TRACE_CBID_cu64MemAllocPitch","CUPTI_DRIVER_TRACE_CBID_cuMemFree","CUPTI_DRIVER_TRACE_CBID_cu64MemFree","CUPTI_DRIVER_TRACE_CBID_cuMemGetAddressRange","CUPTI_DRIVER_TRACE_CBID_cu64MemGetAddressRange","CUPTI_DRIVER_TRACE_CBID_cuMemAllocHost","CUPTI_DRIVER_TRACE_CBID_cuMemFreeHost","CUPTI_DRIVER_TRACE_CBID_cuMemHostAlloc","CUPTI_DRIVER_TRACE_CBID_cuMemHostGetDevicePointer","CUPTI_DRIVER_TRACE_CBID_cu64MemHostGetDevicePointer","CUPTI_DRIVER_TRACE_CBID_cuMemHostGetFlags","CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD","CUPTI_DRIVER_TRACE_CBID_cu64MemcpyHtoD","CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH","CUPTI_DRIVER_TRACE_CBID_cu64MemcpyDtoH","CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD","CUPTI_DRIVER_TRACE_CBID_cu64MemcpyDtoD","CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoA","CUPTI_DRIVER_TRACE_CBID_cu64MemcpyDtoA","CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoD","CUPTI_DRIVER_TRACE_CBID_cu64MemcpyAtoD","CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoA","CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoH","CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoA","CUPTI_DRIVER_TRACE_CBID_cuMemcpy2D","CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DUnaligned","CUPTI_DRIVER_TRACE_CBID_cuMemcpy3D","CUPTI_DRIVER_TRACE_CBID_cu64Memcpy3D","CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync","CUPTI_DRIVER_TRACE_CBID_cu64MemcpyHtoDAsync","CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync","CUPTI_DRIVER_TRACE_CBID_cu64MemcpyDtoHAsync","CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync","CUPTI_DRIVER_TRACE_CBID_cu64MemcpyDtoDAsync","CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoAAsync","CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoHAsync","CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DAsync","CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DAsync","CUPTI_DRIVER_TRACE_CBID_cu64Memcpy3DAsync","CUPTI_DRIVER_TRACE_CBID_cuMemsetD8","CUPTI_DRIVER_TRACE_CBID_cu64MemsetD8","CUPTI_DRIVER_TRACE_CBID_cuMemsetD16","CUPTI_DRIVER_TRACE_CBID_cu64MemsetD16","CUPTI_DRIVER_TRACE_CBID_cuMemsetD32","CUPTI_DRIVER_TRACE_CBID_cu64MemsetD32","CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D8","CUPTI_DRIVER_TRACE_CBID_cu64MemsetD2D8","CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D16","CUPTI_DRIVER_TRACE_CBID_cu64MemsetD2D16","CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D32","CUPTI_DRIVER_TRACE_CBID_cu64MemsetD2D32","CUPTI_DRIVER_TRACE_CBID_cuFuncSetBlockShape","CUPTI_DRIVER_TRACE_CBID_cuFuncSetSharedSize","CUPTI_DRIVER_TRACE_CBID_cuFuncGetAttribute","CUPTI_DRIVER_TRACE_CBID_cuFuncSetCacheConfig","CUPTI_DRIVER_TRACE_CBID_cuArrayCreate","CUPTI_DRIVER_TRACE_CBID_cuArrayGetDescriptor","CUPTI_DRIVER_TRACE_CBID_cuArrayDestroy","CUPTI_DRIVER_TRACE_CBID_cuArray3DCreate","CUPTI_DRIVER_TRACE_CBID_cuArray3DGetDescriptor","CUPTI_DRIVER_TRACE_CBID_cuTexRefCreate","CUPTI_DRIVER_TRACE_CBID_cuTexRefDestroy","CUPTI_DRIVER_TRACE_CBID_cuTexRefSetArray","CUPTI_DRIVER_TRACE_CBID_cuTexRefSetAddress","CUPTI_DRIVER_TRACE_CBID_cu64TexRefSetAddress","CUPTI_DRIVER_TRACE_CBID_cuTexRefSetAddress2D","CUPTI_DRIVER_TRACE_CBID_cu64TexRefSetAddress2D","CUPTI_DRIVER_TRACE_CBID_cuTexRefSetFormat","CUPTI_DRIVER_TRACE_CBID_cuTexRefSetAddressMode","CUPTI_DRIVER_TRACE_CBID_cuTexRefSetFilterMode","CUPTI_DRIVER_TRACE_CBID_cuTexRefSetFlags","CUPTI_DRIVER_TRACE_CBID_cuTexRefGetAddress","CUPTI_DRIVER_TRACE_CBID_cu64TexRefGetAddress","CUPTI_DRIVER_TRACE_CBID_cuTexRefGetArray","CUPTI_DRIVER_TRACE_CBID_cuTexRefGetAddressMode","CUPTI_DRIVER_TRACE_CBID_cuTexRefGetFilterMode","CUPTI_DRIVER_TRACE_CBID_cuTexRefGetFormat","CUPTI_DRIVER_TRACE_CBID_cuTexRefGetFlags","CUPTI_DRIVER_TRACE_CBID_cuParamSetSize","CUPTI_DRIVER_TRACE_CBID_cuParamSeti","CUPTI_DRIVER_TRACE_CBID_cuParamSetf","CUPTI_DRIVER_TRACE_CBID_cuParamSetv","CUPTI_DRIVER_TRACE_CBID_cuParamSetTexRef","CUPTI_DRIVER_TRACE_CBID_cuLaunch","CUPTI_DRIVER_TRACE_CBID_cuLaunchGrid","CUPTI_DRIVER_TRACE_CBID_cuLaunchGridAsync","CUPTI_DRIVER_TRACE_CBID_cuEventCreate","CUPTI_DRIVER_TRACE_CBID_cuEventRecord","CUPTI_DRIVER_TRACE_CBID_cuEventQuery","CUPTI_DRIVER_TRACE_CBID_cuEventSynchronize","CUPTI_DRIVER_TRACE_CBID_cuEventDestroy","CUPTI_DRIVER_TRACE_CBID_cuEventElapsedTime","CUPTI_DRIVER_TRACE_CBID_cuStreamCreate","CUPTI_DRIVER_TRACE_CBID_cuStreamQuery","CUPTI_DRIVER_TRACE_CBID_cuStreamSynchronize","CUPTI_DRIVER_TRACE_CBID_cuStreamDestroy","CUPTI_DRIVER_TRACE_CBID_cuGraphicsUnregisterResource","CUPTI_DRIVER_TRACE_CBID_cuGraphicsSubResourceGetMappedArray","CUPTI_DRIVER_TRACE_CBID_cuGraphicsResourceGetMappedPointer","CUPTI_DRIVER_TRACE_CBID_cu64GraphicsResourceGetMappedPointer","CUPTI_DRIVER_TRACE_CBID_cuGraphicsResourceSetMapFlags","CUPTI_DRIVER_TRACE_CBID_cuGraphicsMapResources","CUPTI_DRIVER_TRACE_CBID_cuGraphicsUnmapResources","CUPTI_DRIVER_TRACE_CBID_cuGetExportTable","CUPTI_DRIVER_TRACE_CBID_cuCtxSetLimit","CUPTI_DRIVER_TRACE_CBID_cuCtxGetLimit","CUPTI_DRIVER_TRACE_CBID_cuD3D10GetDevice","CUPTI_DRIVER_TRACE_CBID_cuD3D10CtxCreate","CUPTI_DRIVER_TRACE_CBID_cuGraphicsD3D10RegisterResource","CUPTI_DRIVER_TRACE_CBID_cuD3D10RegisterResource","CUPTI_DRIVER_TRACE_CBID_cuD3D10UnregisterResource","CUPTI_DRIVER_TRACE_CBID_cuD3D10MapResources","CUPTI_DRIVER_TRACE_CBID_cuD3D10UnmapResources","CUPTI_DRIVER_TRACE_CBID_cuD3D10ResourceSetMapFlags","CUPTI_DRIVER_TRACE_CBID_cuD3D10ResourceGetMappedArray","CUPTI_DRIVER_TRACE_CBID_cuD3D10ResourceGetMappedPointer","CUPTI_DRIVER_TRACE_CBID_cuD3D10ResourceGetMappedSize","CUPTI_DRIVER_TRACE_CBID_cuD3D10ResourceGetMappedPitch","CUPTI_DRIVER_TRACE_CBID_cuD3D10ResourceGetSurfaceDimensions","CUPTI_DRIVER_TRACE_CBID_cuD3D11GetDevice","CUPTI_DRIVER_TRACE_CBID_cuD3D11CtxCreate","CUPTI_DRIVER_TRACE_CBID_cuGraphicsD3D11RegisterResource","CUPTI_DRIVER_TRACE_CBID_cuD3D9GetDevice","CUPTI_DRIVER_TRACE_CBID_cuD3D9CtxCreate","CUPTI_DRIVER_TRACE_CBID_cuGraphicsD3D9RegisterResource","CUPTI_DRIVER_TRACE_CBID_cuD3D9GetDirect3DDevice","CUPTI_DRIVER_TRACE_CBID_cuD3D9RegisterResource","CUPTI_DRIVER_TRACE_CBID_cuD3D9UnregisterResource","CUPTI_DRIVER_TRACE_CBID_cuD3D9MapResources","CUPTI_DRIVER_TRACE_CBID_cuD3D9UnmapResources","CUPTI_DRIVER_TRACE_CBID_cuD3D9ResourceSetMapFlags","CUPTI_DRIVER_TRACE_CBID_cuD3D9ResourceGetSurfaceDimensions","CUPTI_DRIVER_TRACE_CBID_cuD3D9ResourceGetMappedArray","CUPTI_DRIVER_TRACE_CBID_cuD3D9ResourceGetMappedPointer","CUPTI_DRIVER_TRACE_CBID_cuD3D9ResourceGetMappedSize","CUPTI_DRIVER_TRACE_CBID_cuD3D9ResourceGetMappedPitch","CUPTI_DRIVER_TRACE_CBID_cuD3D9Begin","CUPTI_DRIVER_TRACE_CBID_cuD3D9End","CUPTI_DRIVER_TRACE_CBID_cuD3D9RegisterVertexBuffer","CUPTI_DRIVER_TRACE_CBID_cuD3D9MapVertexBuffer","CUPTI_DRIVER_TRACE_CBID_cuD3D9UnmapVertexBuffer","CUPTI_DRIVER_TRACE_CBID_cuD3D9UnregisterVertexBuffer","CUPTI_DRIVER_TRACE_CBID_cuGLCtxCreate","CUPTI_DRIVER_TRACE_CBID_cuGraphicsGLRegisterBuffer","CUPTI_DRIVER_TRACE_CBID_cuGraphicsGLRegisterImage","CUPTI_DRIVER_TRACE_CBID_cuWGLGetDevice","CUPTI_DRIVER_TRACE_CBID_cuGLInit","CUPTI_DRIVER_TRACE_CBID_cuGLRegisterBufferObject","CUPTI_DRIVER_TRACE_CBID_cuGLMapBufferObject","CUPTI_DRIVER_TRACE_CBID_cuGLUnmapBufferObject","CUPTI_DRIVER_TRACE_CBID_cuGLUnregisterBufferObject","CUPTI_DRIVER_TRACE_CBID_cuGLSetBufferObjectMapFlags","CUPTI_DRIVER_TRACE_CBID_cuGLMapBufferObjectAsync","CUPTI_DRIVER_TRACE_CBID_cuGLUnmapBufferObjectAsync","CUPTI_DRIVER_TRACE_CBID_cuVDPAUGetDevice","CUPTI_DRIVER_TRACE_CBID_cuVDPAUCtxCreate","CUPTI_DRIVER_TRACE_CBID_cuGraphicsVDPAURegisterVideoSurface","CUPTI_DRIVER_TRACE_CBID_cuGraphicsVDPAURegisterOutputSurface","CUPTI_DRIVER_TRACE_CBID_cuModuleGetSurfRef","CUPTI_DRIVER_TRACE_CBID_cuSurfRefCreate","CUPTI_DRIVER_TRACE_CBID_cuSurfRefDestroy","CUPTI_DRIVER_TRACE_CBID_cuSurfRefSetFormat","CUPTI_DRIVER_TRACE_CBID_cuSurfRefSetArray","CUPTI_DRIVER_TRACE_CBID_cuSurfRefGetFormat","CUPTI_DRIVER_TRACE_CBID_cuSurfRefGetArray","CUPTI_DRIVER_TRACE_CBID_cu64DeviceTotalMem","CUPTI_DRIVER_TRACE_CBID_cu64D3D10ResourceGetMappedPointer","CUPTI_DRIVER_TRACE_CBID_cu64D3D10ResourceGetMappedSize","CUPTI_DRIVER_TRACE_CBID_cu64D3D10ResourceGetMappedPitch","CUPTI_DRIVER_TRACE_CBID_cu64D3D10ResourceGetSurfaceDimensions","CUPTI_DRIVER_TRACE_CBID_cu64D3D9ResourceGetSurfaceDimensions","CUPTI_DRIVER_TRACE_CBID_cu64D3D9ResourceGetMappedPointer","CUPTI_DRIVER_TRACE_CBID_cu64D3D9ResourceGetMappedSize","CUPTI_DRIVER_TRACE_CBID_cu64D3D9ResourceGetMappedPitch","CUPTI_DRIVER_TRACE_CBID_cu64D3D9MapVertexBuffer","CUPTI_DRIVER_TRACE_CBID_cu64GLMapBufferObject","CUPTI_DRIVER_TRACE_CBID_cu64GLMapBufferObjectAsync","CUPTI_DRIVER_TRACE_CBID_cuD3D11GetDevices","CUPTI_DRIVER_TRACE_CBID_cuD3D11CtxCreateOnDevice","CUPTI_DRIVER_TRACE_CBID_cuD3D10GetDevices","CUPTI_DRIVER_TRACE_CBID_cuD3D10CtxCreateOnDevice","CUPTI_DRIVER_TRACE_CBID_cuD3D9GetDevices","CUPTI_DRIVER_TRACE_CBID_cuD3D9CtxCreateOnDevice","CUPTI_DRIVER_TRACE_CBID_cu64MemHostAlloc","CUPTI_DRIVER_TRACE_CBID_cuMemsetD8Async","CUPTI_DRIVER_TRACE_CBID_cu64MemsetD8Async","CUPTI_DRIVER_TRACE_CBID_cuMemsetD16Async","CUPTI_DRIVER_TRACE_CBID_cu64MemsetD16Async","CUPTI_DRIVER_TRACE_CBID_cuMemsetD32Async","CUPTI_DRIVER_TRACE_CBID_cu64MemsetD32Async","CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D8Async","CUPTI_DRIVER_TRACE_CBID_cu64MemsetD2D8Async","CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D16Async","CUPTI_DRIVER_TRACE_CBID_cu64MemsetD2D16Async","CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D32Async","CUPTI_DRIVER_TRACE_CBID_cu64MemsetD2D32Async","CUPTI_DRIVER_TRACE_CBID_cu64ArrayCreate","CUPTI_DRIVER_TRACE_CBID_cu64ArrayGetDescriptor","CUPTI_DRIVER_TRACE_CBID_cu64Array3DCreate","CUPTI_DRIVER_TRACE_CBID_cu64Array3DGetDescriptor","CUPTI_DRIVER_TRACE_CBID_cu64Memcpy2D","CUPTI_DRIVER_TRACE_CBID_cu64Memcpy2DUnaligned","CUPTI_DRIVER_TRACE_CBID_cu64Memcpy2DAsync","CUPTI_DRIVER_TRACE_CBID_cuCtxCreate_v2","CUPTI_DRIVER_TRACE_CBID_cuD3D10CtxCreate_v2","CUPTI_DRIVER_TRACE_CBID_cuD3D11CtxCreate_v2","CUPTI_DRIVER_TRACE_CBID_cuD3D9CtxCreate_v2","CUPTI_DRIVER_TRACE_CBID_cuGLCtxCreate_v2","CUPTI_DRIVER_TRACE_CBID_cuVDPAUCtxCreate_v2","CUPTI_DRIVER_TRACE_CBID_cuModuleGetGlobal_v2","CUPTI_DRIVER_TRACE_CBID_cuMemGetInfo_v2","CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2","CUPTI_DRIVER_TRACE_CBID_cuMemAllocPitch_v2","CUPTI_DRIVER_TRACE_CBID_cuMemFree_v2","CUPTI_DRIVER_TRACE_CBID_cuMemGetAddressRange_v2","CUPTI_DRIVER_TRACE_CBID_cuMemHostGetDevicePointer_v2","CUPTI_DRIVER_TRACE_CBID_cuMemcpy_v2","CUPTI_DRIVER_TRACE_CBID_cuMemsetD8_v2","CUPTI_DRIVER_TRACE_CBID_cuMemsetD16_v2","CUPTI_DRIVER_TRACE_CBID_cuMemsetD32_v2","CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D8_v2","CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D16_v2","CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D32_v2","CUPTI_DRIVER_TRACE_CBID_cuTexRefSetAddress_v2","CUPTI_DRIVER_TRACE_CBID_cuTexRefSetAddress2D_v2","CUPTI_DRIVER_TRACE_CBID_cuTexRefGetAddress_v2","CUPTI_DRIVER_TRACE_CBID_cuGraphicsResourceGetMappedPointer_v2","CUPTI_DRIVER_TRACE_CBID_cuDeviceTotalMem_v2","CUPTI_DRIVER_TRACE_CBID_cuD3D10ResourceGetMappedPointer_v2","CUPTI_DRIVER_TRACE_CBID_cuD3D10ResourceGetMappedSize_v2","CUPTI_DRIVER_TRACE_CBID_cuD3D10ResourceGetMappedPitch_v2","CUPTI_DRIVER_TRACE_CBID_cuD3D10ResourceGetSurfaceDimensions_v2","CUPTI_DRIVER_TRACE_CBID_cuD3D9ResourceGetSurfaceDimensions_v2","CUPTI_DRIVER_TRACE_CBID_cuD3D9ResourceGetMappedPointer_v2","CUPTI_DRIVER_TRACE_CBID_cuD3D9ResourceGetMappedSize_v2","CUPTI_DRIVER_TRACE_CBID_cuD3D9ResourceGetMappedPitch_v2","CUPTI_DRIVER_TRACE_CBID_cuD3D9MapVertexBuffer_v2","CUPTI_DRIVER_TRACE_CBID_cuGLMapBufferObject_v2","CUPTI_DRIVER_TRACE_CBID_cuGLMapBufferObjectAsync_v2","CUPTI_DRIVER_TRACE_CBID_cuMemHostAlloc_v2","CUPTI_DRIVER_TRACE_CBID_cuArrayCreate_v2","CUPTI_DRIVER_TRACE_CBID_cuArrayGetDescriptor_v2","CUPTI_DRIVER_TRACE_CBID_cuArray3DCreate_v2","CUPTI_DRIVER_TRACE_CBID_cuArray3DGetDescriptor_v2","CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2","CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2","CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2","CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2","CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2","CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2","CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoH_v2","CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoHAsync_v2","CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoD_v2","CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoA_v2","CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoA_v2","CUPTI_DRIVER_TRACE_CBID_cuMemcpy2D_v2","CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DUnaligned_v2","CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DAsync_v2","CUPTI_DRIVER_TRACE_CBID_cuMemcpy3D_v2","CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DAsync_v2","CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoA_v2","CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoAAsync_v2","CUPTI_DRIVER_TRACE_CBID_cuMemAllocHost_v2","CUPTI_DRIVER_TRACE_CBID_cuStreamWaitEvent","CUPTI_DRIVER_TRACE_CBID_cuCtxGetApiVersion","CUPTI_DRIVER_TRACE_CBID_cuD3D10GetDirect3DDevice","CUPTI_DRIVER_TRACE_CBID_cuD3D11GetDirect3DDevice","CUPTI_DRIVER_TRACE_CBID_cuCtxGetCacheConfig","CUPTI_DRIVER_TRACE_CBID_cuCtxSetCacheConfig","CUPTI_DRIVER_TRACE_CBID_cuMemHostRegister","CUPTI_DRIVER_TRACE_CBID_cuMemHostUnregister","CUPTI_DRIVER_TRACE_CBID_cuCtxSetCurrent","CUPTI_DRIVER_TRACE_CBID_cuCtxGetCurrent","CUPTI_DRIVER_TRACE_CBID_cuMemcpy","CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync","CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel","CUPTI_DRIVER_TRACE_CBID_cuProfilerStart","CUPTI_DRIVER_TRACE_CBID_cuProfilerStop","CUPTI_DRIVER_TRACE_CBID_cuPointerGetAttribute","CUPTI_DRIVER_TRACE_CBID_cuProfilerInitialize","CUPTI_DRIVER_TRACE_CBID_cuDeviceCanAccessPeer","CUPTI_DRIVER_TRACE_CBID_cuCtxEnablePeerAccess","CUPTI_DRIVER_TRACE_CBID_cuCtxDisablePeerAccess","CUPTI_DRIVER_TRACE_CBID_cuMemPeerRegister","CUPTI_DRIVER_TRACE_CBID_cuMemPeerUnregister","CUPTI_DRIVER_TRACE_CBID_cuMemPeerGetDevicePointer","CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeer","CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeerAsync","CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DPeer","CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DPeerAsync","CUPTI_DRIVER_TRACE_CBID_cuCtxDestroy_v2","CUPTI_DRIVER_TRACE_CBID_cuCtxPushCurrent_v2","CUPTI_DRIVER_TRACE_CBID_cuCtxPopCurrent_v2","CUPTI_DRIVER_TRACE_CBID_cuEventDestroy_v2","CUPTI_DRIVER_TRACE_CBID_cuStreamDestroy_v2","CUPTI_DRIVER_TRACE_CBID_cuTexRefSetAddress2D_v3","CUPTI_DRIVER_TRACE_CBID_cuIpcGetMemHandle","CUPTI_DRIVER_TRACE_CBID_cuIpcOpenMemHandle","CUPTI_DRIVER_TRACE_CBID_cuIpcCloseMemHandle","CUPTI_DRIVER_TRACE_CBID_cuDeviceGetByPCIBusId","CUPTI_DRIVER_TRACE_CBID_cuDeviceGetPCIBusId","CUPTI_DRIVER_TRACE_CBID_cuGLGetDevices","CUPTI_DRIVER_TRACE_CBID_cuIpcGetEventHandle","CUPTI_DRIVER_TRACE_CBID_cuIpcOpenEventHandle","CUPTI_DRIVER_TRACE_CBID_cuCtxSetSharedMemConfig","CUPTI_DRIVER_TRACE_CBID_cuCtxGetSharedMemConfig","CUPTI_DRIVER_TRACE_CBID_cuFuncSetSharedMemConfig","CUPTI_DRIVER_TRACE_CBID_cuTexObjectCreate","CUPTI_DRIVER_TRACE_CBID_cuTexObjectDestroy","CUPTI_DRIVER_TRACE_CBID_cuTexObjectGetResourceDesc","CUPTI_DRIVER_TRACE_CBID_cuTexObjectGetTextureDesc","CUPTI_DRIVER_TRACE_CBID_cuSurfObjectCreate","CUPTI_DRIVER_TRACE_CBID_cuSurfObjectDestroy","CUPTI_DRIVER_TRACE_CBID_cuSurfObjectGetResourceDesc","CUPTI_DRIVER_TRACE_CBID_cuStreamAddCallback","CUPTI_DRIVER_TRACE_CBID_cuMipmappedArrayCreate","CUPTI_DRIVER_TRACE_CBID_cuMipmappedArrayGetLevel","CUPTI_DRIVER_TRACE_CBID_cuMipmappedArrayDestroy","CUPTI_DRIVER_TRACE_CBID_cuTexRefSetMipmappedArray","CUPTI_DRIVER_TRACE_CBID_cuTexRefSetMipmapFilterMode","CUPTI_DRIVER_TRACE_CBID_cuTexRefSetMipmapLevelBias","CUPTI_DRIVER_TRACE_CBID_cuTexRefSetMipmapLevelClamp","CUPTI_DRIVER_TRACE_CBID_cuTexRefSetMaxAnisotropy","CUPTI_DRIVER_TRACE_CBID_cuTexRefGetMipmappedArray","CUPTI_DRIVER_TRACE_CBID_cuTexRefGetMipmapFilterMode","CUPTI_DRIVER_TRACE_CBID_cuTexRefGetMipmapLevelBias","CUPTI_DRIVER_TRACE_CBID_cuTexRefGetMipmapLevelClamp","CUPTI_DRIVER_TRACE_CBID_cuTexRefGetMaxAnisotropy","CUPTI_DRIVER_TRACE_CBID_cuGraphicsResourceGetMappedMipmappedArray","CUPTI_DRIVER_TRACE_CBID_cuTexObjectGetResourceViewDesc","CUPTI_DRIVER_TRACE_CBID_cuLinkCreate","CUPTI_DRIVER_TRACE_CBID_cuLinkAddData","CUPTI_DRIVER_TRACE_CBID_cuLinkAddFile","CUPTI_DRIVER_TRACE_CBID_cuLinkComplete","CUPTI_DRIVER_TRACE_CBID_cuLinkDestroy","CUPTI_DRIVER_TRACE_CBID_cuStreamCreateWithPriority","CUPTI_DRIVER_TRACE_CBID_cuStreamGetPriority","CUPTI_DRIVER_TRACE_CBID_cuStreamGetFlags","CUPTI_DRIVER_TRACE_CBID_cuCtxGetStreamPriorityRange","CUPTI_DRIVER_TRACE_CBID_cuMemAllocManaged","CUPTI_DRIVER_TRACE_CBID_cuGetErrorString","CUPTI_DRIVER_TRACE_CBID_cuGetErrorName","CUPTI_DRIVER_TRACE_CBID_cuOccupancyMaxActiveBlocksPerMultiprocessor","CUPTI_DRIVER_TRACE_CBID_cuCompilePtx","CUPTI_DRIVER_TRACE_CBID_cuBinaryFree","CUPTI_DRIVER_TRACE_CBID_cuStreamAttachMemAsync","CUPTI_DRIVER_TRACE_CBID_cuPointerSetAttribute","CUPTI_DRIVER_TRACE_CBID_cuMemHostRegister_v2","CUPTI_DRIVER_TRACE_CBID_cuGraphicsResourceSetMapFlags_v2","CUPTI_DRIVER_TRACE_CBID_cuLinkCreate_v2","CUPTI_DRIVER_TRACE_CBID_cuLinkAddData_v2","CUPTI_DRIVER_TRACE_CBID_cuLinkAddFile_v2","CUPTI_DRIVER_TRACE_CBID_cuOccupancyMaxPotentialBlockSize","CUPTI_DRIVER_TRACE_CBID_cuGLGetDevices_v2","CUPTI_DRIVER_TRACE_CBID_cuDevicePrimaryCtxRetain","CUPTI_DRIVER_TRACE_CBID_cuDevicePrimaryCtxRelease","CUPTI_DRIVER_TRACE_CBID_cuDevicePrimaryCtxSetFlags","CUPTI_DRIVER_TRACE_CBID_cuDevicePrimaryCtxReset","CUPTI_DRIVER_TRACE_CBID_cuGraphicsEGLRegisterImage","CUPTI_DRIVER_TRACE_CBID_cuCtxGetFlags","CUPTI_DRIVER_TRACE_CBID_cuDevicePrimaryCtxGetState","CUPTI_DRIVER_TRACE_CBID_cuEGLStreamConsumerConnect","CUPTI_DRIVER_TRACE_CBID_cuEGLStreamConsumerDisconnect","CUPTI_DRIVER_TRACE_CBID_cuEGLStreamConsumerAcquireFrame","CUPTI_DRIVER_TRACE_CBID_cuEGLStreamConsumerReleaseFrame","CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2_ptds","CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2_ptds","CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2_ptds","CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoA_v2_ptds","CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoD_v2_ptds","CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoA_v2_ptds","CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoH_v2_ptds","CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoA_v2_ptds","CUPTI_DRIVER_TRACE_CBID_cuMemcpy2D_v2_ptds","CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DUnaligned_v2_ptds","CUPTI_DRIVER_TRACE_CBID_cuMemcpy3D_v2_ptds","CUPTI_DRIVER_TRACE_CBID_cuMemcpy_ptds","CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeer_ptds","CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DPeer_ptds","CUPTI_DRIVER_TRACE_CBID_cuMemsetD8_v2_ptds","CUPTI_DRIVER_TRACE_CBID_cuMemsetD16_v2_ptds","CUPTI_DRIVER_TRACE_CBID_cuMemsetD32_v2_ptds","CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D8_v2_ptds","CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D16_v2_ptds","CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D32_v2_ptds","CUPTI_DRIVER_TRACE_CBID_cuGLMapBufferObject_v2_ptds","CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync_ptsz","CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoAAsync_v2_ptsz","CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoHAsync_v2_ptsz","CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2_ptsz","CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2_ptsz","CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2_ptsz","CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DAsync_v2_ptsz","CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DAsync_v2_ptsz","CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeerAsync_ptsz","CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DPeerAsync_ptsz","CUPTI_DRIVER_TRACE_CBID_cuMemsetD8Async_ptsz","CUPTI_DRIVER_TRACE_CBID_cuMemsetD16Async_ptsz","CUPTI_DRIVER_TRACE_CBID_cuMemsetD32Async_ptsz","CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D8Async_ptsz","CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D16Async_ptsz","CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D32Async_ptsz","CUPTI_DRIVER_TRACE_CBID_cuStreamGetPriority_ptsz","CUPTI_DRIVER_TRACE_CBID_cuStreamGetFlags_ptsz","CUPTI_DRIVER_TRACE_CBID_cuStreamWaitEvent_ptsz","CUPTI_DRIVER_TRACE_CBID_cuStreamAddCallback_ptsz","CUPTI_DRIVER_TRACE_CBID_cuStreamAttachMemAsync_ptsz","CUPTI_DRIVER_TRACE_CBID_cuStreamQuery_ptsz","CUPTI_DRIVER_TRACE_CBID_cuStreamSynchronize_ptsz","CUPTI_DRIVER_TRACE_CBID_cuEventRecord_ptsz","CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz","CUPTI_DRIVER_TRACE_CBID_cuGraphicsMapResources_ptsz","CUPTI_DRIVER_TRACE_CBID_cuGraphicsUnmapResources_ptsz","CUPTI_DRIVER_TRACE_CBID_cuGLMapBufferObjectAsync_v2_ptsz","CUPTI_DRIVER_TRACE_CBID_cuEGLStreamProducerConnect","CUPTI_DRIVER_TRACE_CBID_cuEGLStreamProducerDisconnect","CUPTI_DRIVER_TRACE_CBID_cuEGLStreamProducerPresentFrame","CUPTI_DRIVER_TRACE_CBID_cuGraphicsResourceGetMappedEglFrame","CUPTI_DRIVER_TRACE_CBID_cuPointerGetAttributes","CUPTI_DRIVER_TRACE_CBID_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags","CUPTI_DRIVER_TRACE_CBID_cuOccupancyMaxPotentialBlockSizeWithFlags","CUPTI_DRIVER_TRACE_CBID_cuEGLStreamProducerReturnFrame","CUPTI_DRIVER_TRACE_CBID_cuDeviceGetP2PAttribute","CUPTI_DRIVER_TRACE_CBID_cuTexRefSetBorderColor","CUPTI_DRIVER_TRACE_CBID_cuTexRefGetBorderColor","CUPTI_DRIVER_TRACE_CBID_cuMemAdvise","CUPTI_DRIVER_TRACE_CBID_cuStreamWaitValue32","CUPTI_DRIVER_TRACE_CBID_cuStreamWaitValue32_ptsz","CUPTI_DRIVER_TRACE_CBID_cuStreamWriteValue32","CUPTI_DRIVER_TRACE_CBID_cuStreamWriteValue32_ptsz","CUPTI_DRIVER_TRACE_CBID_cuStreamBatchMemOp","CUPTI_DRIVER_TRACE_CBID_cuStreamBatchMemOp_ptsz","CUPTI_DRIVER_TRACE_CBID_cuNVNbufferGetPointer","CUPTI_DRIVER_TRACE_CBID_cuNVNtextureGetArray","CUPTI_DRIVER_TRACE_CBID_cuNNSetAllocator","CUPTI_DRIVER_TRACE_CBID_cuMemPrefetchAsync","CUPTI_DRIVER_TRACE_CBID_cuMemPrefetchAsync_ptsz","CUPTI_DRIVER_TRACE_CBID_cuEventCreateFromNVNSync","CUPTI_DRIVER_TRACE_CBID_cuEGLStreamConsumerConnectWithFlags","CUPTI_DRIVER_TRACE_CBID_cuMemRangeGetAttribute","CUPTI_DRIVER_TRACE_CBID_cuMemRangeGetAttributes","CUPTI_DRIVER_TRACE_CBID_cuStreamWaitValue64","CUPTI_DRIVER_TRACE_CBID_cuStreamWaitValue64_ptsz","CUPTI_DRIVER_TRACE_CBID_cuStreamWriteValue64","CUPTI_DRIVER_TRACE_CBID_cuStreamWriteValue64_ptsz","CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel","CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz","CUPTI_DRIVER_TRACE_CBID_cuEventCreateFromEGLSync","CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice","CUPTI_DRIVER_TRACE_CBID_cuFuncSetAttribute","CUPTI_DRIVER_TRACE_CBID_cuDeviceGetUuid","CUPTI_DRIVER_TRACE_CBID_cuStreamGetCtx","CUPTI_DRIVER_TRACE_CBID_cuStreamGetCtx_ptsz","CUPTI_DRIVER_TRACE_CBID_cuImportExternalMemory","CUPTI_DRIVER_TRACE_CBID_cuExternalMemoryGetMappedBuffer","CUPTI_DRIVER_TRACE_CBID_cuExternalMemoryGetMappedMipmappedArray","CUPTI_DRIVER_TRACE_CBID_cuDestroyExternalMemory","CUPTI_DRIVER_TRACE_CBID_cuImportExternalSemaphore","CUPTI_DRIVER_TRACE_CBID_cuSignalExternalSemaphoresAsync","CUPTI_DRIVER_TRACE_CBID_cuSignalExternalSemaphoresAsync_ptsz","CUPTI_DRIVER_TRACE_CBID_cuWaitExternalSemaphoresAsync","CUPTI_DRIVER_TRACE_CBID_cuWaitExternalSemaphoresAsync_ptsz","CUPTI_DRIVER_TRACE_CBID_cuDestroyExternalSemaphore","CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture","CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture_ptsz","CUPTI_DRIVER_TRACE_CBID_cuStreamEndCapture","CUPTI_DRIVER_TRACE_CBID_cuStreamEndCapture_ptsz","CUPTI_DRIVER_TRACE_CBID_cuStreamIsCapturing","CUPTI_DRIVER_TRACE_CBID_cuStreamIsCapturing_ptsz","CUPTI_DRIVER_TRACE_CBID_cuGraphCreate","CUPTI_DRIVER_TRACE_CBID_cuGraphAddKernelNode","CUPTI_DRIVER_TRACE_CBID_cuGraphKernelNodeGetParams","CUPTI_DRIVER_TRACE_CBID_cuGraphAddMemcpyNode","CUPTI_DRIVER_TRACE_CBID_cuGraphMemcpyNodeGetParams","CUPTI_DRIVER_TRACE_CBID_cuGraphAddMemsetNode","CUPTI_DRIVER_TRACE_CBID_cuGraphMemsetNodeGetParams","CUPTI_DRIVER_TRACE_CBID_cuGraphMemsetNodeSetParams","CUPTI_DRIVER_TRACE_CBID_cuGraphNodeGetType","CUPTI_DRIVER_TRACE_CBID_cuGraphGetRootNodes","CUPTI_DRIVER_TRACE_CBID_cuGraphNodeGetDependencies","CUPTI_DRIVER_TRACE_CBID_cuGraphNodeGetDependentNodes","CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiate","CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch","CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz","CUPTI_DRIVER_TRACE_CBID_cuGraphExecDestroy","CUPTI_DRIVER_TRACE_CBID_cuGraphDestroy","CUPTI_DRIVER_TRACE_CBID_cuGraphAddDependencies","CUPTI_DRIVER_TRACE_CBID_cuGraphRemoveDependencies","CUPTI_DRIVER_TRACE_CBID_cuGraphMemcpyNodeSetParams","CUPTI_DRIVER_TRACE_CBID_cuGraphKernelNodeSetParams","CUPTI_DRIVER_TRACE_CBID_cuGraphDestroyNode","CUPTI_DRIVER_TRACE_CBID_cuGraphClone","CUPTI_DRIVER_TRACE_CBID_cuGraphNodeFindInClone","CUPTI_DRIVER_TRACE_CBID_cuGraphAddChildGraphNode","CUPTI_DRIVER_TRACE_CBID_cuGraphAddEmptyNode","CUPTI_DRIVER_TRACE_CBID_cuLaunchHostFunc","CUPTI_DRIVER_TRACE_CBID_cuLaunchHostFunc_ptsz","CUPTI_DRIVER_TRACE_CBID_cuGraphChildGraphNodeGetGraph","CUPTI_DRIVER_TRACE_CBID_cuGraphAddHostNode","CUPTI_DRIVER_TRACE_CBID_cuGraphHostNodeGetParams","CUPTI_DRIVER_TRACE_CBID_cuDeviceGetLuid","CUPTI_DRIVER_TRACE_CBID_cuGraphHostNodeSetParams","CUPTI_DRIVER_TRACE_CBID_cuGraphGetNodes","CUPTI_DRIVER_TRACE_CBID_cuGraphGetEdges","CUPTI_DRIVER_TRACE_CBID_cuStreamGetCaptureInfo","CUPTI_DRIVER_TRACE_CBID_cuStreamGetCaptureInfo_ptsz","CUPTI_DRIVER_TRACE_CBID_cuGraphExecKernelNodeSetParams","CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture_v2","CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture_v2_ptsz","CUPTI_DRIVER_TRACE_CBID_cuThreadExchangeStreamCaptureMode","CUPTI_DRIVER_TRACE_CBID_cuDeviceGetNvSciSyncAttributes","CUPTI_DRIVER_TRACE_CBID_cuOccupancyAvailableDynamicSMemPerBlock","CUPTI_DRIVER_TRACE_CBID_cuDevicePrimaryCtxRelease_v2","CUPTI_DRIVER_TRACE_CBID_cuDevicePrimaryCtxReset_v2","CUPTI_DRIVER_TRACE_CBID_cuDevicePrimaryCtxSetFlags_v2","CUPTI_DRIVER_TRACE_CBID_cuMemAddressReserve","CUPTI_DRIVER_TRACE_CBID_cuMemAddressFree","CUPTI_DRIVER_TRACE_CBID_cuMemCreate","CUPTI_DRIVER_TRACE_CBID_cuMemRelease","CUPTI_DRIVER_TRACE_CBID_cuMemMap","CUPTI_DRIVER_TRACE_CBID_cuMemUnmap","CUPTI_DRIVER_TRACE_CBID_cuMemSetAccess","CUPTI_DRIVER_TRACE_CBID_cuMemExportToShareableHandle","CUPTI_DRIVER_TRACE_CBID_cuMemImportFromShareableHandle","CUPTI_DRIVER_TRACE_CBID_cuMemGetAllocationGranularity","CUPTI_DRIVER_TRACE_CBID_cuMemGetAllocationPropertiesFromHandle","CUPTI_DRIVER_TRACE_CBID_cuMemGetAccess","CUPTI_DRIVER_TRACE_CBID_cuStreamSetFlags","CUPTI_DRIVER_TRACE_CBID_cuStreamSetFlags_ptsz","CUPTI_DRIVER_TRACE_CBID_cuGraphExecUpdate","CUPTI_DRIVER_TRACE_CBID_cuGraphExecMemcpyNodeSetParams","CUPTI_DRIVER_TRACE_CBID_cuGraphExecMemsetNodeSetParams","CUPTI_DRIVER_TRACE_CBID_cuGraphExecHostNodeSetParams","CUPTI_DRIVER_TRACE_CBID_cuMemRetainAllocationHandle","CUPTI_DRIVER_TRACE_CBID_cuFuncGetModule","CUPTI_DRIVER_TRACE_CBID_cuIpcOpenMemHandle_v2","CUPTI_DRIVER_TRACE_CBID_cuCtxResetPersistingL2Cache","CUPTI_DRIVER_TRACE_CBID_cuGraphKernelNodeCopyAttributes","CUPTI_DRIVER_TRACE_CBID_cuGraphKernelNodeGetAttribute","CUPTI_DRIVER_TRACE_CBID_cuGraphKernelNodeSetAttribute","CUPTI_DRIVER_TRACE_CBID_cuStreamCopyAttributes","CUPTI_DRIVER_TRACE_CBID_cuStreamCopyAttributes_ptsz","CUPTI_DRIVER_TRACE_CBID_cuStreamGetAttribute","CUPTI_DRIVER_TRACE_CBID_cuStreamGetAttribute_ptsz","CUPTI_DRIVER_TRACE_CBID_cuStreamSetAttribute","CUPTI_DRIVER_TRACE_CBID_cuStreamSetAttribute_ptsz","CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiate_v2","CUPTI_DRIVER_TRACE_CBID_cuDeviceGetTexture1DLinearMaxWidth","CUPTI_DRIVER_TRACE_CBID_cuGraphUpload","CUPTI_DRIVER_TRACE_CBID_cuGraphUpload_ptsz","CUPTI_DRIVER_TRACE_CBID_cuArrayGetSparseProperties","CUPTI_DRIVER_TRACE_CBID_cuMipmappedArrayGetSparseProperties","CUPTI_DRIVER_TRACE_CBID_cuMemMapArrayAsync","CUPTI_DRIVER_TRACE_CBID_cuMemMapArrayAsync_ptsz","CUPTI_DRIVER_TRACE_CBID_cuGraphExecChildGraphNodeSetParams","CUPTI_DRIVER_TRACE_CBID_cuEventRecordWithFlags","CUPTI_DRIVER_TRACE_CBID_cuEventRecordWithFlags_ptsz","CUPTI_DRIVER_TRACE_CBID_cuGraphAddEventRecordNode","CUPTI_DRIVER_TRACE_CBID_cuGraphAddEventWaitNode","CUPTI_DRIVER_TRACE_CBID_cuGraphEventRecordNodeGetEvent","CUPTI_DRIVER_TRACE_CBID_cuGraphEventWaitNodeGetEvent","CUPTI_DRIVER_TRACE_CBID_cuGraphEventRecordNodeSetEvent","CUPTI_DRIVER_TRACE_CBID_cuGraphEventWaitNodeSetEvent","CUPTI_DRIVER_TRACE_CBID_cuGraphExecEventRecordNodeSetEvent","CUPTI_DRIVER_TRACE_CBID_cuGraphExecEventWaitNodeSetEvent","CUPTI_DRIVER_TRACE_CBID_cuArrayGetPlane","CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync","CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync_ptsz","CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync","CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync_ptsz","CUPTI_DRIVER_TRACE_CBID_cuMemPoolTrimTo","CUPTI_DRIVER_TRACE_CBID_cuMemPoolSetAttribute","CUPTI_DRIVER_TRACE_CBID_cuMemPoolGetAttribute","CUPTI_DRIVER_TRACE_CBID_cuMemPoolSetAccess","CUPTI_DRIVER_TRACE_CBID_cuDeviceGetDefaultMemPool","CUPTI_DRIVER_TRACE_CBID_cuMemPoolCreate","CUPTI_DRIVER_TRACE_CBID_cuMemPoolDestroy","CUPTI_DRIVER_TRACE_CBID_cuDeviceSetMemPool","CUPTI_DRIVER_TRACE_CBID_cuDeviceGetMemPool","CUPTI_DRIVER_TRACE_CBID_cuMemAllocFromPoolAsync","CUPTI_DRIVER_TRACE_CBID_cuMemAllocFromPoolAsync_ptsz","CUPTI_DRIVER_TRACE_CBID_cuMemPoolExportToShareableHandle","CUPTI_DRIVER_TRACE_CBID_cuMemPoolImportFromShareableHandle","CUPTI_DRIVER_TRACE_CBID_cuMemPoolExportPointer","CUPTI_DRIVER_TRACE_CBID_cuMemPoolImportPointer","CUPTI_DRIVER_TRACE_CBID_cuMemPoolGetAccess","CUPTI_DRIVER_TRACE_CBID_cuGraphAddExternalSemaphoresSignalNode","CUPTI_DRIVER_TRACE_CBID_cuGraphExternalSemaphoresSignalNodeGetParams","CUPTI_DRIVER_TRACE_CBID_cuGraphExternalSemaphoresSignalNodeSetParams","CUPTI_DRIVER_TRACE_CBID_cuGraphAddExternalSemaphoresWaitNode","CUPTI_DRIVER_TRACE_CBID_cuGraphExternalSemaphoresWaitNodeGetParams","CUPTI_DRIVER_TRACE_CBID_cuGraphExternalSemaphoresWaitNodeSetParams","CUPTI_DRIVER_TRACE_CBID_cuGraphExecExternalSemaphoresSignalNodeSetParams","CUPTI_DRIVER_TRACE_CBID_cuGraphExecExternalSemaphoresWaitNodeSetParams","CUPTI_DRIVER_TRACE_CBID_cuGetProcAddress","CUPTI_DRIVER_TRACE_CBID_cuFlushGPUDirectRDMAWrites","CUPTI_DRIVER_TRACE_CBID_cuGraphDebugDotPrint","CUPTI_DRIVER_TRACE_CBID_cuStreamGetCaptureInfo_v2","CUPTI_DRIVER_TRACE_CBID_cuStreamGetCaptureInfo_v2_ptsz","CUPTI_DRIVER_TRACE_CBID_cuStreamUpdateCaptureDependencies","CUPTI_DRIVER_TRACE_CBID_cuStreamUpdateCaptureDependencies_ptsz","CUPTI_DRIVER_TRACE_CBID_cuUserObjectCreate","CUPTI_DRIVER_TRACE_CBID_cuUserObjectRetain","CUPTI_DRIVER_TRACE_CBID_cuUserObjectRelease","CUPTI_DRIVER_TRACE_CBID_cuGraphRetainUserObject","CUPTI_DRIVER_TRACE_CBID_cuGraphReleaseUserObject","CUPTI_DRIVER_TRACE_CBID_cuGraphAddMemAllocNode","CUPTI_DRIVER_TRACE_CBID_cuGraphAddMemFreeNode","CUPTI_DRIVER_TRACE_CBID_cuDeviceGraphMemTrim","CUPTI_DRIVER_TRACE_CBID_cuDeviceGetGraphMemAttribute","CUPTI_DRIVER_TRACE_CBID_cuDeviceSetGraphMemAttribute","CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiateWithFlags","CUPTI_DRIVER_TRACE_CBID_cuDeviceGetExecAffinitySupport","CUPTI_DRIVER_TRACE_CBID_cuCtxCreate_v3","CUPTI_DRIVER_TRACE_CBID_cuCtxGetExecAffinity","CUPTI_DRIVER_TRACE_CBID_cuDeviceGetUuid_v2","CUPTI_DRIVER_TRACE_CBID_cuGraphMemAllocNodeGetParams","CUPTI_DRIVER_TRACE_CBID_cuGraphMemFreeNodeGetParams","CUPTI_DRIVER_TRACE_CBID_cuGraphNodeSetEnabled","CUPTI_DRIVER_TRACE_CBID_cuGraphNodeGetEnabled","CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx","CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz","CUPTI_DRIVER_TRACE_CBID_cuArrayGetMemoryRequirements","CUPTI_DRIVER_TRACE_CBID_cuMipmappedArrayGetMemoryRequirements","CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiateWithParams","CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiateWithParams_ptsz","CUPTI_DRIVER_TRACE_CBID_cuGraphExecGetFlags","CUPTI_DRIVER_TRACE_CBID_cuStreamWaitValue32_v2","CUPTI_DRIVER_TRACE_CBID_cuStreamWaitValue32_v2_ptsz","CUPTI_DRIVER_TRACE_CBID_cuStreamWaitValue64_v2","CUPTI_DRIVER_TRACE_CBID_cuStreamWaitValue64_v2_ptsz","CUPTI_DRIVER_TRACE_CBID_cuStreamWriteValue32_v2","CUPTI_DRIVER_TRACE_CBID_cuStreamWriteValue32_v2_ptsz","CUPTI_DRIVER_TRACE_CBID_cuStreamWriteValue64_v2","CUPTI_DRIVER_TRACE_CBID_cuStreamWriteValue64_v2_ptsz","CUPTI_DRIVER_TRACE_CBID_cuStreamBatchMemOp_v2","CUPTI_DRIVER_TRACE_CBID_cuStreamBatchMemOp_v2_ptsz","CUPTI_DRIVER_TRACE_CBID_cuGraphAddBatchMemOpNode","CUPTI_DRIVER_TRACE_CBID_cuGraphBatchMemOpNodeGetParams","CUPTI_DRIVER_TRACE_CBID_cuGraphBatchMemOpNodeSetParams","CUPTI_DRIVER_TRACE_CBID_cuGraphExecBatchMemOpNodeSetParams","CUPTI_DRIVER_TRACE_CBID_cuModuleGetLoadingMode","CUPTI_DRIVER_TRACE_CBID_cuMemGetHandleForAddressRange","CUPTI_DRIVER_TRACE_CBID_cuOccupancyMaxPotentialClusterSize","CUPTI_DRIVER_TRACE_CBID_cuOccupancyMaxActiveClusters","CUPTI_DRIVER_TRACE_CBID_cuGetProcAddress_v2","CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadData","CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadFromFile","CUPTI_DRIVER_TRACE_CBID_cuLibraryUnload","CUPTI_DRIVER_TRACE_CBID_cuLibraryGetKernel","CUPTI_DRIVER_TRACE_CBID_cuLibraryGetModule","CUPTI_DRIVER_TRACE_CBID_cuKernelGetFunction","CUPTI_DRIVER_TRACE_CBID_cuLibraryGetGlobal","CUPTI_DRIVER_TRACE_CBID_cuLibraryGetManaged","CUPTI_DRIVER_TRACE_CBID_cuKernelGetAttribute","CUPTI_DRIVER_TRACE_CBID_cuKernelSetAttribute","CUPTI_DRIVER_TRACE_CBID_cuKernelSetCacheConfig","CUPTI_DRIVER_TRACE_CBID_cuGraphAddKernelNode_v2","CUPTI_DRIVER_TRACE_CBID_cuGraphKernelNodeGetParams_v2","CUPTI_DRIVER_TRACE_CBID_cuGraphKernelNodeSetParams_v2","CUPTI_DRIVER_TRACE_CBID_cuGraphExecKernelNodeSetParams_v2","CUPTI_DRIVER_TRACE_CBID_cuStreamGetId","CUPTI_DRIVER_TRACE_CBID_cuStreamGetId_ptsz","CUPTI_DRIVER_TRACE_CBID_cuCtxGetId","CUPTI_DRIVER_TRACE_CBID_cuGraphExecUpdate_v2","CUPTI_DRIVER_TRACE_CBID_cuTensorMapEncodeTiled","CUPTI_DRIVER_TRACE_CBID_cuTensorMapEncodeIm2col","CUPTI_DRIVER_TRACE_CBID_cuTensorMapReplaceAddress","CUPTI_DRIVER_TRACE_CBID_cuLibraryGetUnifiedFunction","CUPTI_DRIVER_TRACE_CBID_cuCoredumpGetAttribute","CUPTI_DRIVER_TRACE_CBID_cuCoredumpGetAttributeGlobal","CUPTI_DRIVER_TRACE_CBID_cuCoredumpSetAttribute","CUPTI_DRIVER_TRACE_CBID_cuCoredumpSetAttributeGlobal","CUPTI_DRIVER_TRACE_CBID_cuCtxSetFlags","CUPTI_DRIVER_TRACE_CBID_cuMulticastCreate","CUPTI_DRIVER_TRACE_CBID_cuMulticastAddDevice","CUPTI_DRIVER_TRACE_CBID_cuMulticastBindMem","CUPTI_DRIVER_TRACE_CBID_cuMulticastBindAddr","CUPTI_DRIVER_TRACE_CBID_cuMulticastUnbind","CUPTI_DRIVER_TRACE_CBID_cuMulticastGetGranularity","CUPTI_DRIVER_TRACE_CBID_SIZE","CUPTI_DRIVER_TRACE_CBID_FORCE_INT" };

    void cuptiAPICallBack(void *userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void *cbdata) {
        assert(domain == CUPTI_CB_DOMAIN_DRIVER_API);
        const CUpti_CallbackData *cbInfo = (CUpti_CallbackData *) cbdata;

        if(
                cbid == CUPTI_DRIVER_TRACE_CBID_cu64MemAlloc ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuMemAllocPitch ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cu64MemAllocPitch ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cu64MemFree ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuMemGetAddressRange ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cu64MemGetAddressRange ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuMemAllocHost ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuMemFreeHost ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuMemHostAlloc ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuArrayCreate ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuArrayDestroy ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuArray3DCreate ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuArray3DGetDescriptor ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuTexRefCreate ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuTexRefDestroy ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuD3D10MapResources ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuD3D9MapResources ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuD3D9UnmapResources ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuD3D9MapVertexBuffer ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuD3D9UnmapVertexBuffer ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuGLMapBufferObject ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuGLUnmapBufferObject ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuGLMapBufferObjectAsync ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuGLUnmapBufferObjectAsync ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cu64GLMapBufferObject ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cu64GLMapBufferObjectAsync ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cu64MemHostAlloc ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cu64ArrayCreate ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cu64Array3DCreate ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuGLMapBufferObject_v2 ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuGLMapBufferObjectAsync_v2 ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuMemHostAlloc_v2 ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuArrayCreate_v2 ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuArray3DCreate_v2 ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuMemAllocHost_v2 ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuMemHostRegister ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuMemHostUnregister ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuMemPeerRegister ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuMemPeerUnregister ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuMemAllocManaged ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuMemAdvise ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuMemPrefetchAsync ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuMemPrefetchAsync_ptsz ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuMemAddressReserve ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuMemAddressFree ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuMemCreate ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuMemRelease ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuMemMap ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuMemUnmap ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuMemMapArrayAsync ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuMemMapArrayAsync_ptsz ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync_ptsz ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync_ptsz ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuMemPoolCreate ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuMemPoolDestroy ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuMemAllocFromPoolAsync ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuMemAllocFromPoolAsync_ptsz
#if CUPTI_API_VERSION > 14 //todo: Inaccurate
                || cbid == CUPTI_DRIVER_TRACE_CBID_cuGraphAddMemAllocNode ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuGraphAddMemFreeNode ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuGraphAddBatchMemOpNode
#endif
                ){
            const char* cuptiFuncStr="CUPTI_DRIVER_TRACE_CBID_FORCE_INT";
            if(cbid!=CUPTI_DRIVER_TRACE_CBID_FORCE_INT){
                cuptiFuncStr=cuptiStrMap[cbid];
            }
            //print_hybridstacktrace();
            //ERR_LOGS("Warning: The program used unsupported memory operation %s. MLInsight currently has not provided such support.",cuptiFuncStr);
        }

        if (cbid == CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2 || cbid == CUPTI_DRIVER_TRACE_CBID_cuMemAlloc) {
#ifndef NDEBUG
            if (cuptiCrossChecker.cuptiCrossCheckingEnabled) {
                assert(cuptiCrossChecker.insideMLInsightCuMemAllocProxy);
            }

            if (cbInfo->callbackSite == CUPTI_API_EXIT) {
                auto *params = (cuMemAlloc_v2_params *) (cbInfo->functionParams);
                //INFO_LOGS("CUPTI_cuMemAlloc_v2_%zu %p", params->bytesize,*params->dptr);

                if (cuptiCrossChecker.cuptiCrossCheckingEnabled) {
                    cuptiCrossChecker.cuMemAllocCuptiPtr = (void*) *params->dptr;
                    cuptiCrossChecker.cuMemAllocCUPTISize += params->bytesize;
                }
            }
#endif
        } else if (cbid == CUPTI_DRIVER_TRACE_CBID_cuMemFree_v2 || cbid == CUPTI_DRIVER_TRACE_CBID_cuMemFree) {
#ifndef NDEBUG
            if (cuptiCrossChecker.cuptiCrossCheckingEnabled) {
                assert(cuptiCrossChecker.insideMLInsightCuMemFreeProxy);
            }

            if (cbInfo->callbackSite == CUPTI_API_EXIT) {
                auto *params = (cuMemFree_v2_params *) (cbInfo->functionParams);
                cuptiCrossChecker.cuMemFreeCuptiPtr = (void*) (params->dptr);
                //if(params->dptr){
                //    INFO_LOGS("CUPTI_cuMemFree_v2 %p", params->dptr);
                //}

            }
#endif

        }
    }

    bool cuptiInitialize=false;
    void initCuptiTrace() {
//        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));  // Non-P2P memcpy
//        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY2)); // P2P memcpy only
//        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMORY_POOL));
//        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
//        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));

        if(!cuptiInitialize){
            cuptiInitialize=true;
            // Register callbacks for buffer requests and for buffers completed by CUPTI.
    //        CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));

            // Register callbacks for CUDA runtime and driverRecord API
            // The proWblem with CUDA callback api is that it does not provide correlations like activity API

    #ifndef NDEBUG
            CUpti_SubscriberHandle subscriber;
            CUptiResult cuptiRet= cuptiSubscribe(&subscriber, (CUpti_CallbackFunc) cuptiAPICallBack, nullptr);
            if(cuptiRet == CUPTI_SUCCESS){
                CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API));
                cuptiCrossChecker.cuptiCrossCheckingEnabled = true;
            }else{
                ERR_LOGS("Warning, cupti instialized failed. Cupti cross-checking will not be effective for process %d", getpid());
                cuptiInitialize=false;
            }
            
    #endif
        }
    }

    void finiTrace() {
        if(cuptiInitialize){
            // Force flush any remaining activity buffers before termination of the application
            CUPTI_CALL(cuptiActivityFlushAll(1));
            cuptiInitialize=false;
        }
    }


}