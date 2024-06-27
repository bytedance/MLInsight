add_library(mlinsightloader SHARED src/Logging.cpp src/main.cpp)
target_include_directories(mlinsightloader PUBLIC include)
target_compile_definitions(mlinsightloader PUBLIC MLINSIGHT_VERSION="${PROJECT_VERSION}") #NDEBUG
target_link_libraries(mlinsightloader dl) # Please do not statically link libstdc++ libgcc
set_target_properties(mlinsightloader PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(mlinsightloader PROPERTIES BUILD_RPATH "${CMAKE_BINARY_DIR}/Runtime")
#https://gcc.gnu.org/onlinedocs/gcc/C_002b_002b-Dialect-Options.html
