cmake_minimum_required(VERSION 3.15)
project(DeepLearning)

set(CMAKE_CXX_STANDARD 14)

include_directories(include)

add_executable(DeepLearning src/main.cpp include/two_layer_net.h include/vector_function.h)