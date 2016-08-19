cmake_minimum_required( VERSION 2.6)
project( testCUDACode)
file(GLOB SOURCES *.cpp)

add_definitions( --std=c++11)
add_executable( testCudaCode ${SOURCES})
