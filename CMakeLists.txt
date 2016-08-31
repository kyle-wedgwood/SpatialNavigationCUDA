cmake_minimum_required( VERSION 2.8)

find_package( CUDA QUIET REQUIRED)

set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; --std=c++11 -gencode arch=compute_30,code=sm_30)

if (APPLE)
  link_directories( /usr/local/lib)
else()
  link_directories( /usr/local/share/arrayfire-3/lib
                    /usr/local/lib
                    /usr/local/cuda-7.5/nvvm/lib64)
  include_directories( /usr/local/share/arrayfire-3/include/)
  #set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}; -Xlinker -rpath -Xlinker /usr/local/cuda-7.5/nvvm/lib64/")
endif()

cuda_add_executable( Driver
                     parameters.h
                     EventDrivenMap.hpp
                     EventDrivenMap.cu
                     Driver.cu)

target_link_libraries( Driver -lafcuda)


# nvcc -x=cu --std=c++11 testVisualisationMoving.cu -I/usr/local/share/arrayfire-3/include -lafcuda -L/usr/local/share/arrayfire-3/lib -Xlinker -rpath -Xlinker /usr/local/cuda-7.5/nvvm/lib64/ -o testVisualisation
