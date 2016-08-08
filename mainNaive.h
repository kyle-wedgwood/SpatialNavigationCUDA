#ifndef MAINNAIVEHEADERGUARD
#define MAINNAIVEHEADERGUARD

#include "parameters.h"
#include "vector_types.h"

#define CUDA_ERROR_CHECK
#define CUDA_CALL( err) __cudaCall( err, __FILE__, __LINE__ )
#define CURAND_CALL( err) __curandCall( err, __FILE__, __LINE__)

inline void __cudaCall( cudaError err, const char *file, const int line );

inline void __cudaCheckError( const char *file, const int line );

__device__ float fun2( float t, float v0, float n0, float u0, float y0, float thresh);

__device__ float fun3( float t, float v0, float n0, float u0, float y0, float thresh);

__device__ float dfun1( float t, float v0, float n0, float u0, float y0);

__device__ float dfun2( float t, float v0, float n0, float u0, float y0);

__device__ float dfun3( float t, float v0, float n0, float u0, float y0);

__device__ float eventTimeZone1( float v0, float n0, float u0, float y0);

__device__ void eventTimeZone2( float v0, float n0, float u0, float y0,
                                 float *t, unsigned short *cross);

__device__ void eventTimeZone3( float v0, float n0, float u0, float y0,
                                 float *t, unsigned short *cross);

__global__ void eventTimeZone1Kernel( const float4* pGlobal_state,
                                      const unsigned short* pGlobalZone,
                                      struct firing* pVal);

__global__ void eventTimeZone2Kernel( const float4* pGlobal_state,
                                      const unsigned short* pGlobalZone,
                                      struct firing* pVal);

__global__ void eventTimeZone3Kernel( const float4* pGlobal_state,
                                      const unsigned short* pGlobalZone,
                                      struct firing* pVal);

__global__ void eventTimeZone4Kernel( const float *pRefractTime,
                                      const unsigned short* pGlobalZone,
                                      struct firing* pVal);

__inline__ __device__ struct firing warpReduceMin( struct firing val);

__inline__ __device__ struct firing blockReduceMin( struct firing val);

__global__ void deviceReduceMinKernel( const struct firing* in,
                                       const unsigned int npts,
                                       struct firing* out);

__global__ void updateZone1Kernel( float* p_global_v,
                                   float* p_global_n,
                                   float* p_global_u,
                                   float* p_global_y,
                                   float eventTime);

int main( int argc , char *argv[]);

#endif
