#@author: Steven (Jiaxun) Tang <jtang@umass.edu>
#@author: Steven  Tang <steven.tang@bytedance.com>

project(Runtime VERSION 0.2.0)

option(USE_TORCH "Use PyTorch" ON)

# Required
include(GNUInstallDirs)

set(COMPILATION_FLAGS "-O3" "-g")
# https://json.nlohmann.me/integration/cmake/


if (USE_TORCH)
   add_definitions(-DUSE_TORCH)
   set (OptionSrc 
    src/trace/PytorchMemProxy.cpp
    src/analyse/PytorchMemory.cpp
   )
   execute_process(
    COMMAND "python3" -c "import os;import torch; print(''.join([os.path.dirname(torch.__file__),os.sep,'include']))"
    OUTPUT_VARIABLE PYTORCH_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
   )
   message(STATUS "Pytorch Include Path: ${PYTORCH_INCLUDE_DIR}")
else()
   add_definitions(-DUSE_TENSORFLOW)
   set (OptionSrc
   )	
endif()

set(HookSrc
        src/trace/HookInstaller.cpp
        src/trace/PyHook.cpp
        src/trace/HookContext.cpp
        src/common/ProcInfoParser.cpp
        src/common/ElfParser.cpp
        src/common/Tool.cpp
        src/trace/SystemProxy.cpp
        src/trace/PthreadProxy.cpp
        src/trace/APIHookHandlers.cpp
        src/trace/DLProxy.cpp
        src/trace/LibcProxy.cpp 
        src/trace/CUDAProxy.cpp  
        src/analyse/DriverMemory.cpp 
    	)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Python Version: ${Python3_EXECUTABLE}")
message(STATUS "Python Include Path: ${Python3_INCLUDE_DIRS}")

 #If cuda not found. Try to change this variable by adding -D CUDAToolkit_ROOT=<cuda toolkiet root>
if(EXISTS "/usr/local/cuda")
    option(CUDAToolkit_ROOT "CUDA toolkit root path" "/usr/local/cuda")
    set(CUDAToolkit_ROOT "/usr/local/cuda")
    find_package(CUDAToolkit)
else()
    option(CUDAToolkit_ROOT "CUDA toolkit root path" "")
    find_package(CUDAToolkit)
endif()

if(${CUDAToolkit_FOUND})
    message(STATUS "Found CUDA. Will build MLInsight-CUDA")

    add_library(mlinsight SHARED ${HookSrc} ${OptionSrc}) 

    target_include_directories(mlinsight PUBLIC include lib/inireader ${PYTORCH_INCLUDE_DIR} ${Python3_INCLUDE_DIRS})# ${PROTO_SRC_DIR})
    target_link_libraries(mlinsight PUBLIC pthread dl CUDA::cuda_driver CUDA::cudart)
    target_compile_options(mlinsight PRIVATE ${COMPILATION_FLAGS})
    target_compile_definitions(mlinsight PUBLIC CUDA_ENABLED MLINSIGHT_VERSION="${PROJECT_VERSION}") #NDEBUG

    install(TARGETS mlinsight LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

#add_subdirectory(tests)
