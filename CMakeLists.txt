cmake_minimum_required( VERSION 2.8)

find_package( CUDA QUIET REQUIRED)

set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_30,code=sm_30)

if (APPLE)
  link_directories( /usr/local/lib)
else()
  link_directories( /usr/local/share/arrayfire-3/lib
                    /usr/local/lib)
endif()

cuda_add_executable( Driver
                     parameters.h
                     EventDrivenMap.hpp
                     EventDrivenMap.cu
                     Driver.cu)

target_link_libraries( Driver -lafcuda)
