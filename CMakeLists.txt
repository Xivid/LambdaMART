cmake_minimum_required(VERSION 3.12)
project(lambdamart)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)
include_directories(src)
add_executable(lambdamart main.cpp src/lambdamart.h src/dataset/dataset.cpp src/dataset/dataset.h src/dataset/bin.cpp src/dataset/bin.h src/tree/TreeLearner.cpp src/tree/TreeLearner.h src/boosting/Booster.cpp src/boosting/Booster.h src/boosting/LambdaRank.cpp src/boosting/LambdaRank.h include/config.h include/common.h include/log.h)