cmake_minimum_required(VERSION 3.12)
project(lambdamart)

include(CMakeLibs.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fno-tree-vectorize")

include_directories(include)
file (GLOB SOURCES
        src/model/*.cpp
        src/dataset/*.cpp
        src/tree/*.cpp)
add_executable(lambdamart main.cpp ${SOURCES})
target_link_libraries(lambdamart pcm)
