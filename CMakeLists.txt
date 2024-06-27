
cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0079 NEW)
project(MLInsight)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_subdirectory(Runtime)
add_subdirectory(Loader)



