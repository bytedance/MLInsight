#@author: Steven Tang <steven.tang@bytedance.com>

project("TestExtensionModule")
# This module is expected to build alone.
# Required
include(GNUInstallDirs)

#install(
#    TARGETS Testlib-FuncCall
#    EXPORT TestLibFuncCallTargets
#    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
add_library(Testlib-FuncCall SHARED lib/FunctionCall/FuncWithDiffParms.cpp
        lib/FunctionCall/TenThousandFunc.cpp)
target_include_directories(Testlib-FuncCall PUBLIC lib/FunctionCall/include)
target_compile_options(Testlib-FuncCall PRIVATE ${TEST_FLAGS} -Werror)
target_link_options(Testlib-FuncCall PUBLIC "-zlazy")

install(
    TARGETS Testlib-FuncCall
    EXPORT Testlib-FuncCallTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

# Python build definition
if(DEFINED SKBUILD)
    # prevent an unused variable warning
    set(ignoreMe "${SKBUILD}")

    # call pybind11-config to obtain the root of the cmake package (pybind11 should be install via pip).
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pybind11 --cmakedir
            OUTPUT_VARIABLE pybind11_ROOT_RAW)
    string(STRIP ${pybind11_ROOT_RAW} pybind11_ROOT)
    find_package(pybind11 REQUIRED)

    pybind11_add_module(_testextensionmodule MODULE "testextensionmodule/NativeFuncCallWrapper.cpp")

    target_link_libraries(_testextensionmodule PRIVATE dl Testlib-FuncCall )
    target_compile_definitions(_testextensionmodule PRIVATE CMAKE_BINARY_DIR="${CMAKE_BINARY_DIR}")
    
    # Installing the extension module to the root of the package
    install(TARGETS _testextensionmodule DESTINATION .)

    if(APPLE)
        set_target_properties(_testextensionmodule PROPERTIES INSTALL_RPATH "@loader_path/${CMAKE_INSTALL_LIBDIR}")
    else()
        set_target_properties(_testextensionmodule PROPERTIES INSTALL_RPATH "$ORIGIN/${CMAKE_INSTALL_LIBDIR}")
    endif()

endif()