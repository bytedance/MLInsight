#@author: Steven (Jiaxun) Tang <jtang@umass.edu>
#@author: Steven  Tang <steven.tang@bytedance.com>

project(Runtime VERSION 0.2.0)

option(USE_TORCH "Use PyTorch" ON)
option(USE_TENSORFLOW "Use Tensorflow" OFF)
option(USE_PERFETTO "Use Perfetto" ON)

# Required
include(GNUInstallDirs)

set(COMPILATION_FLAGS "-O0" "-g")
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Python3_EXECUTABLE=${Python3_EXECUTABLE}")

# A list of optional source files based on compilation flags
set(OPTIONAL_SRC)
# A list of optional include directories based on compilation flags
set(OPTIONAL_INCLUDE_DIR)
set(OPTIONAL_DEFINITIONS)

# The compilation must either be Pytorch or Tensorflow
if (USE_TORCH)
    message(STATUS "Compiling with MLInsight Pytorch Version.")
    list(APPEND OPTIONAL_SRC
            src/trace/PytorchMemProxy.cpp
            src/trace/PyTorchCallbacks.cpp
            src/analyse/DebugAnalyzer.cpp
    )
    execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c "import os;import torch; print(''.join([os.path.dirname(torch.__file__),os.sep,'include']))"
            OUTPUT_VARIABLE PYTORCH_INCLUDE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import os;import torch; print(torch.__version__)"
        OUTPUT_VARIABLE PYTORCH_VERSION_STR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "Pytorch Version: ${PYTORCH_VERSION_STR}")
    message(STATUS "Pytorch Include Path: ${PYTORCH_INCLUDE_DIR}")
    list(APPEND OPTIONAL_INCLUDE_DIR
            ${PYTORCH_INCLUDE_DIR}
    )
    list(APPEND OPTIONAL_DEFINITIONS USE_TORCH=1)
    list(APPEND OPTIONAL_DEFINITIONS PYTORCH_VERSION_STR="${PYTORCH_VERSION_STR}")


elseif (USE_TENSORFLOW)
    message(STATUS "Compiling with MLInsight Tensorflow Version.")
    list(APPEND OPTIONAL_SRC
            src/trace/TensorflowMemProxy.cpp
    )
    # todo: Remove the dependency of absolute path for tensorflow
    list(APPEND OPTIONAL_INCLUDE_DIR /workspace/user/tensorflow /workspace/usr/py37/lib/python3.7/site-packages/tensorflow/include /workspace/usr/py37/lib/python3.7/site-packages/tensorflow_core/include)
    list(APPEND OPTIONAL_DEFINITIONS USE_TENSORFLOW=1)
else ()
    message(FATAL_ERROR "Either USE_TORCH or USE_TENSORFLOW should be enabled")
endif ()

if (USE_PERFETTO)
    message(STATUS "Compiling with perfetto")
    list(APPEND OPTIONAL_SRC
            src/trace/Perfetto.cpp
            lib/perfetto/sdk/perfetto.cc
            src/analyse/PerfettoTensorTraceAnalyzer.cpp
    )
    list(APPEND OPTIONAL_INCLUDE_DIR
            lib/perfetto/sdk
    )
    list(APPEND OPTIONAL_DEFINITIONS USE_PERFETTO=1)
endif ()
message(STATUS "OPTIONAL_SRC=${OPTIONAL_SRC}")
set(HOOK_SRC
        src/trace/HookInstaller.cpp
        src/trace/PyHook.cpp
        src/trace/HookContext.cpp
        src/common/ProcInfoParser.cpp
        src/common/ElfParser.cpp
        src/common/Tool.cpp
        src/common/Logging.cpp
        src/trace/SystemProxy.cpp
        src/trace/PthreadProxy.cpp
        src/trace/APIHookHandlers.cpp
        src/trace/DLProxy.cpp
        src/trace/LibcProxy.cpp
        src/trace/CUDAProxy.cpp
        src/common/Logging.cpp
        src/trace/GPUTrace.cpp
        src/analyse/GlobalVariable.cpp
        src/analyse/MemLeakAnalyzer.cpp
        src/analyse/TensorObj.cpp
        src/trace/PyCodeExtra.cpp
        src/common/CallStack.cpp
        ${OPTIONAL_SRC}
)


#If cuda not found. Try to change this variable by adding -D CUDAToolkit_ROOT=<cuda toolkiet root>
set(CUDAToolkit_ROOT "/usr/local/cuda" CACHE STRING "CUDA toolkit root path")
find_package(CUDAToolkit)

if (${CUDAToolkit_FOUND})
    message(STATUS "Found CUDA. Will build MLInsight-CUDA")
    add_library(mlinsight SHARED ${HOOK_SRC})
    message(STATUS "MLInsight will compile ${HOOK_SRC}")
    target_include_directories(mlinsight PUBLIC include lib/inireader ${OPTIONAL_INCLUDE_DIR} ${Python3_INCLUDE_DIRS})# ${PROTO_SRC_DIR})
    target_link_libraries(mlinsight PUBLIC pthread dl CUDA::cuda_driver CUDA::cudart CUDA::cupti)
    target_compile_options(mlinsight PRIVATE ${COMPILATION_FLAGS})
#     target_link_options(mlinsight PRIVATE "LINKER:--unresolved-symbols=ignore-all")
    target_compile_definitions(mlinsight PUBLIC CUDA_ENABLED MLINSIGHT_VERSION="${PROJECT_VERSION}" ${OPTIONAL_DEFINITIONS}) #NDEBUG
    install(TARGETS mlinsight LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
else ()
    message(FATAL_ERROR "Currently, MLInsight must be compiled with CUDAToolkit.")
endif ()

# add_subdirectory(tests)
