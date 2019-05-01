cmake_minimum_required(VERSION 3.12)
project(lambdamart)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)
file (GLOB SOURCES
        src/model/*.cpp
        src/dataset/*.cpp
        src/tree/*.cpp)
add_executable(lambdamart main.cpp ${SOURCES})
