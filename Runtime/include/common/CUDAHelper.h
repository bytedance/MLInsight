#ifndef MLINSIGHT_CUDAHELPER_H
#define MLINSIGHT_CUDAHELPER_H

#ifdef CUDA_ENABLED

#include "common/Logging.h"
#include <cuda_runtime.h>
#include <cuda.h>

namespace mlinsight {

/**
 * Exception Handling
 */
    inline void cudaAssert(cudaError_t err, const char *__file, int __line) {
        if (cudaSuccess != err) {
            OUTPUTS("result != cudaSuccess %s %d %d\n", err != cudaSuccess ? "true" : "false", err, cudaSuccess);
            OUTPUTS("ERR: %s:%d  CUDA runtime error code=%d(%s) \"%s\" \n", __file, __line, err, cudaGetErrorName(err),
                    cudaGetErrorString(err));
            exit(-1);
        }
    }

    inline void cuAssert(CUresult err, const char *__file, ssize_t __line) {
        if (CUDA_SUCCESS != err) {
            const char *errorName = nullptr;
            const char *errorDescription = nullptr;
            cuGetErrorName(err, &errorName);
            cuGetErrorString(err, &errorDescription);

            OUTPUTS("ERR: %s:%ld  CUDA driverRecord error code=%d(%s) \"%s\" \n", __file, __line, err, errorName,
                    errorDescription);
            exit(-1);
        }
    }

// Kill program and print error message if a CUDA runtime API fails.
#define CUDA_ASSERT(result) {std::string srcFile=__FILE__; cudaAssert(result,srcFile.c_str(),__LINE__);}
//Kill program and print error message if a CUDA driverRecord API fails.
#define CU_ASSERT(result) {std::string srcFile=__FILE__; cuAssert(result,srcFile.c_str(),__LINE__);}

/**
 * Initialization
 */

/**
 * Before calling CUDA driverRecord API, the initialization must be done here.
 * @return
 */
    inline void initCUDADriver() {
        CU_ASSERT(cuInit(0));

        CUdevice device;
        CUcontext pctx;
        CU_ASSERT(cuDeviceGet(&device, 0));
        CU_ASSERT(cuCtxCreate(&pctx, CU_CTX_SCHED_AUTO, device));
    }

}

#endif

#endif
