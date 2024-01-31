
cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0079 NEW)
project(MLInsight)

#add_compile_definitions(CUDA_VERSION_121_LATER)
#add_compile_definitions(TORCH_VERSION_20_LATER)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_subdirectory(Runtime)



