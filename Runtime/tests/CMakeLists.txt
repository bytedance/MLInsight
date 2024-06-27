# @author: Steven (Jiaxun) Tang <jtang@umass.edu>

project(Trace-Tests)

enable_testing()
find_package(GTest)

# ============================================================
# Test libs
# ============================================================

set(TEST_FLAGS "-fstack-protector-all")

if(${GTest_FOUND})

## A library that does a range of function call
#add_library(Testlib-FuncCall SHARED lib/FunctionCall/FuncWithDiffParms.cpp
#        lib/FunctionCall/TenThousandFunc.cpp)
#target_include_directories(Testlib-FuncCall PUBLIC ${CAKE_ROOT} lib/FunctionCall/include)
#target_compile_options(Testlib-FuncCall PRIVATE ${TEST_FLAGS} -Werror)
#target_link_options(Testlib-FuncCall PUBLIC "-zlazy")
## A library that calls Testlib-FuncCall

#add_library(Testlib-CallFuncCall SHARED lib/CallFunctionCall/CallFunctionCall.cpp)
#target_include_directories(Testlib-CallFuncCall PUBLIC lib/CallFunctionCall/include)
#target_link_libraries(Testlib-CallFuncCall Testlib-FuncCall)
#target_compile_options(Testlib-CallFuncCall PRIVATE ${TEST_FLAGS} -Werror)

# ============================================================
# Demo Application
# ============================================================

#option(Trace-demoapps_AUTO_INSTALL_HOOK "If ON, then Trace-Auto will be linked. Otherwise Trace-HookManualAsm-C will be linked" ON)
#
#add_executable(Trace-demoapps-Exceptions src/demoapps/TestExceptions.cpp)
#target_link_libraries(Trace-demoapps-Exceptions PUBLIC Trace-Auto Testlib-FuncCall
#        Testlib-CallFuncCall  dl)
#target_compile_options(Trace-demoapps-Exceptions PRIVATE ${TEST_FLAGS} -Werror)
#
#add_executable(Trace-demoapps-FuncCall src/demoapps/TestFuncCall.cpp)
#target_link_libraries(Trace-demoapps-FuncCall PUBLIC Trace-Auto Testlib-FuncCall
#        Testlib-CallFuncCall  dl)
#target_compile_options(Trace-demoapps-FuncCall PRIVATE ${TEST_FLAGS} -Werror)
#
#
#add_executable(Trace-demoapps-TimingAccuracy src/demoapps/TestTimingAccuracy.cpp)
#target_link_libraries(Trace-demoapps-TimingAccuracy PUBLIC Testlib-FuncCall Testlib-CallFuncCall  dl)
#target_compile_options(Trace-demoapps-TimingAccuracy PRIVATE ${TEST_FLAGS} -Werror)
#target_link_options(Trace-demoapps-TimingAccuracy PRIVATE "-z" "lazy")
#
#add_executable(Trace-demoapps-HookEverything src/demoapps/TestHookEverything.cpp)
#target_include_directories(Trace-demoapps-HookEverything PUBLIC libtest/header)
#target_link_libraries(Trace-demoapps-HookEverything PUBLIC Trace-Auto Testlib-FuncCall
#        Testlib-CallFuncCall  dl)
#target_compile_options(Trace-demoapps-HookEverything PRIVATE ${TEST_FLAGS} -Werror)
#
#add_executable(Trace-demoapps-FuncCallDelay src/demoapps/TestFuncCallDelay.cpp)
#target_link_libraries(Trace-demoapps-FuncCallDelay PUBLIC Trace-Auto Testlib-FuncCall
#        Testlib-CallFuncCall dl pthread)
#target_compile_options(Trace-demoapps-FuncCallDelay PRIVATE ${TEST_FLAGS} -Werror)
#
#add_executable(Trace-demoapps-FuncNestedCall src/demoapps/TestNestedFuncCallPthread.cpp)
#target_link_libraries(Trace-demoapps-FuncNestedCall PUBLIC Trace-Auto Testlib-FuncCall
#        Testlib-CallFuncCall dl pthread)
#target_compile_options(Trace-demoapps-FuncNestedCall PRIVATE ${TEST_FLAGS} -Werror)
#
#
#add_executable(Trace-demoapps-Pthread src/demoapps/TestPthread.cpp)
#target_include_directories(Trace-demoapps-Pthread PUBLIC libtest/header)
#target_link_libraries(Trace-demoapps-Pthread Trace-Auto Testlib-FuncCall pthread dl)
#target_compile_options(Trace-demoapps-Pthread PRIVATE ${TEST_FLAGS} -Werror)
#
#add_executable(Trace-demoapps-BinaryAccess src/demoapps/TestBinaryAccess.cpp)
#target_include_directories(Trace-demoapps-BinaryAccess PUBLIC libtest/header)
#
#
#add_executable(Trace-proof-Threadlocal src/demoapps/TestThreadLocalAssembly.cpp)
#target_include_directories(Trace-proof-Threadlocal PUBLIC libtest/header)
#target_link_libraries(Trace-proof-Threadlocal pthread dl)
#target_compile_options(Trace-proof-Threadlocal PRIVATE ${TEST_FLAGS} -Werror)
#
#
#add_executable(Trace-demoapps-StdPthread src/demoapps/TestStdThread.cpp)
#target_include_directories(Trace-demoapps-StdPthread PUBLIC libtest/header)
#target_link_libraries(Trace-demoapps-StdPthread Trace-Auto Testlib-FuncCall Testlib-CallFuncCall pthread dl)
#target_compile_options(Trace-demoapps-StdPthread PRIVATE ${TEST_FLAGS} -Werror)
#
#
#add_executable(Trace-demoapps-ComplexThreadop src/demoapps/TestComplexThreadOps.cpp)
#target_include_directories(Trace-demoapps-ComplexThreadop PUBLIC libtest/header)
#target_link_libraries(Trace-demoapps-ComplexThreadop Trace-Auto Testlib-FuncCall Testlib-CallFuncCall pthread dl)
#target_compile_options(Trace-demoapps-ComplexThreadop PRIVATE ${TEST_FLAGS} -Werror)
#
#
#add_executable(Trace-demoapps-Semaphore src/demoapps/TestSemaphore.cpp)
#target_include_directories(Trace-demoapps-Semaphore PUBLIC)
#target_link_libraries(Trace-demoapps-Semaphore Trace-Auto pthread dl)
#target_compile_options(Trace-demoapps-Semaphore PRIVATE ${TEST_FLAGS} -Werror)
#
#
#add_executable(Trace-demoapps-Signal src/demoapps/TestSingalHandler.cpp)
#target_include_directories(Trace-demoapps-Signal PUBLIC)
#target_link_libraries(Trace-demoapps-Signal Trace-Auto pthread dl)
#target_compile_options(Trace-demoapps-Signal PRIVATE ${TEST_FLAGS} -Werror)
#
#add_executable(Trace-demoapps-InlineFunc src/demoapps/TestInlineFunction.cpp)
#target_include_directories(Trace-demoapps-InlineFunc PUBLIC)
#target_link_libraries(Trace-demoapps-InlineFunc Trace-Auto)
#target_compile_options(Trace-demoapps-InlineFunc PRIVATE ${TEST_FLAGS} -Werror)
#
#
#add_executable(Trace-proof-TestTLS src/demoapps/TestTLS.cpp)
#target_include_directories(Trace-proof-TestTLS PUBLIC)
#target_link_libraries(Trace-proof-TestTLS pthread)
#target_compile_options(Trace-proof-TestTLS PRIVATE ${TEST_FLAGS} -Werror)
#
#add_library(libWorkload SHARED src/demoapps/libWorkload.cpp)
#target_include_directories(libWorkload PUBLIC src/demoapps/include)
#
#add_executable(Trace-demoapps-SingleThreadedWorkload src/demoapps/TestSimpleFuncCall.cpp)
#target_link_libraries(Trace-demoapps-SingleThreadedWorkload PUBLIC libWorkload)
#
#add_executable(Trace-demoapps-ThreadInterleave src/demoapps/TestThreadInterleave.cpp)
#target_link_libraries(Trace-demoapps-ThreadInterleave PUBLIC libWorkload pthread)
#
#add_executable(Trace-demoapps-ThreadSimultaneouslyEnd src/demoapps/TestThreadSimultaenouslyEnd.cpp)
#target_link_libraries(Trace-demoapps-ThreadSimultaneouslyEnd PUBLIC libWorkload pthread)


find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
add_executable(unittests
        src/unittests/TestLinkedList.cpp
        src/unittests/TestHashMap.cpp
        src/unittests/TestObjectPool.cpp
        src/unittests/TestArray.cpp
        src/unittests/TestRingBuffer.cpp
        src/unittests/TestLazyValue.cpp
        src/unittests/TestFlameGraph.cpp
        ../src/common/Logging.cpp
)
target_link_libraries(unittests PUBLIC gtest gtest_main dl pthread)
target_include_directories(unittests PUBLIC ../../Runtime/include  ${Python3_INCLUDE_DIRS})
target_compile_options(unittests PRIVATE "-fno-access-control" "-fno-elide-constructors" ${TEST_FLAGS} -Werror)
target_compile_definitions(unittests PRIVATE -DUSE_TORCH)
# Temporary measure.
#target_link_libraries(unittests PUBLIC mlinsight)
#target_link_options(unittests PRIVATE "LINKER:--unresolved-symbols=ignore-all")


else()
    message(STATUS "Unit tests are disabled because gtest not found. Please install it to enable unittest.")
endif()

