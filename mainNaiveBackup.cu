/* Code to do one dimensional spiking model from Mayte's note */
#include <iostream>
#include <cstdlib>
#include <cmath>
#include "parameters.hpp"
#include "vector_types.h"

#define CUDA_ERROR_CHECK
#define CUDA_CALL( err) __cudaCall( err, __FILE__, __LINE__ )
#define CURAND_CALL( err) __curandCall( err, __FILE__, __LINE__)

inline void __cudaCall( cudaError err, const char *file, const int line )
{
  #ifdef CUDA_ERROR_CHECK
  if ( cudaSuccess != err )
  {
    fprintf( stderr, "cudaCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
    exit( -1 );
  }
  #endif
  return;
}

inline void __cudaCheckError( const char *file, const int line )
{
  #ifdef CUDA_ERROR_CHECK
  cudaError err = cudaGetLastError();
if ( cudaSuccess != err )
  {
    fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
    exit( -1 );
  }
  #endif
}

struct firing
{
  float time;
  unsigned int index;
  unsigned short cross;
};

__device__ float fun1( float t, float v0, float n0, float u0, float y0)
{
  return v0*expf(-t/tau)
  +(beta_left*gh*(tau-tau_h+tau_h*expf(-t/tau_h)))/(tau-tau_h)
  -(beta_left*gh*tau*expf(-t/tau))/(tau-tau_h)
  -(gh*n0*tau_h*expf(-t/tau_h))/(tau-tau_h)
  +(gh*n0*tau_h*expf(-t/tau))/(tau-tau_h)
  +(gs*expf(-alpha*t)*(alpha*tau*u0-u0-alpha*t*y0+alpha*tau*y0+powf(alpha,2)*t*tau*y0))/powf(alpha*tau-1,2)
  -(gs*expf(-t/tau)*(alpha*tau*u0-u0+alpha*tau*y0))/powf(alpha*tau-1,2)
  -I+I*expf(-t/tau)-V_left;
}


__device__ float fun2( float t, float v0, float n0, float u0, float y0, float thresh)
{
  return
  1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))+tau*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))+tau_h*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))-tau_h*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h)))*(-v0+(beta_centre*(tau*tau)*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(gamma_centre*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*-2.0+gamma_centre*tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0+gamma_centre*tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)+(beta_centre*(tau_h*tau_h)*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(gamma_centre*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*-2.0+gamma_centre*tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0+gamma_centre*tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)+(I*tau_h*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))+(I*tau_h*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(I*(tau_h*tau_h)*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(beta_centre*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(gamma_centre*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)))-(beta_centre*tau*tau_h*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(-gamma_centre*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+gamma_centre*tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+gamma_centre*tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))+(gs*tau_h*u0*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))-1.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(beta_centre*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(gamma_centre*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)))-alpha*gs*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0+tau*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(t*tau+t*tau_h-tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0)+(alpha*gs*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0+tau*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(t*tau+t*tau_h-tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*t*tau*tau_h*2.0)*2.0)*(1.0/2.0))/tau+alpha*gs*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0-tau*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(-t*tau-t*tau_h+tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0)+(alpha*gs*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0-tau*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(-t*tau-t*tau_h+tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+alpha*t*tau*tau_h*2.0)*2.0)*(1.0/2.0))/tau+(I*tau*tau_h*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(I*(tau_h*tau_h)*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(I*tau*tau_h*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(gs*(tau_h*tau_h)*u0*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(gs*tau_h*u0*exp(-alpha*t)*(exp(alpha*t)-exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(beta_centre*(tau*tau)*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(gamma_centre*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)))-(beta_centre*(tau_h*tau_h)*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(gamma_centre*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)))+(alpha*gs*tau_h*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0+tau*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(t*tau+t*tau_h-tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/tau-(alpha*gs*tau_h*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0-tau*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(-t*tau-t*tau_h+tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/tau+(gs*tau*tau_h*u0*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(gs*(tau_h*tau_h)*u0*exp(-alpha*t)*(exp(alpha*t)-exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h)))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)+(beta_centre*tau*tau_h*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(gamma_centre*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)))+(gs*tau*tau_h*u0*exp(-alpha*t)*(exp(alpha*t)-exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h)))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0))*(-1.0/2.0)+(1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*((tau*tau)*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))-(tau*tau)*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))+(tau_h*tau_h)*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))-(tau_h*tau_h)*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))-exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau_h*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*2.0+tau*tau_h*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*2.0)*(n0+(beta_centre*tau*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))+(beta_centre*tau*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(beta_centre*(tau*tau)*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(beta_centre*tau*tau_h*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(beta_centre*(tau*tau)*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(beta_centre*tau*tau_h*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))+alpha*gamma_centre*gs*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0+tau*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(t*tau+t*tau_h-tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*gamma_centre*gs*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0-tau*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(-t*tau-t*tau_h+tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-(I*gamma_centre*tau*tau_h*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*2.0)/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(I*gamma_centre*tau*tau_h*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(gamma_centre*gs*tau*tau_h*u0*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(gamma_centre*gs*tau*tau_h*u0*exp(-alpha*t)*(exp(alpha*t)-exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h)))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0))*(1.0/4.0))/(gamma_centre*tau)
  - thresh;
}

__device__ float fun3( float t, float v0, float n0, float u0, float y0, float thresh)
{
  return v0*expf(-t/tau)
  +(beta_right*gh*(tau-tau_h+tau_h*expf(-t/tau_h)))/(tau-tau_h)
  -(beta_right*gh*tau*expf(-t/tau))/(tau-tau_h)
  -(gh*n0*tau_h*expf(-t/tau_h))/(tau-tau_h)
  +(gh*n0*tau_h*expf(-t/tau))/(tau-tau_h)
  +(gs*expf(-alpha*t)*(alpha*tau*u0-u0-alpha*t*y0+alpha*tau*y0+powf(alpha,2)*t*tau*y0))/powf(alpha*tau-1,2)
  -(gs*expf(-t/tau)*(alpha*tau*u0-u0+alpha*tau*y0))/powf(alpha*tau-1,2)
  -I+I*expf(-t/tau) - thresh;
}

__device__ float dfun1( float t, float v0, float n0, float u0, float y0)
{
  return
    (beta_right*gh*exp(-t/tau))/(tau-tau_h)-(v0*exp(-t/tau))/tau-(I*exp(-t/tau))/tau-(beta_right*gh*exp(-t/tau_h))/(tau-tau_h)+(gh*n0*exp(-t/tau_h))/(tau-tau_h)-(gs*exp(-alpha*t)*(-tau*y0*alpha*alpha+y0*alpha))/powf(alpha*tau-1.0f,2.0f)+(gs*exp(-t/tau)*(alpha*tau*u0-u0+alpha*tau*y0))/(tau*powf(alpha*tau-1.0f,2.0f))-(alpha*gs*exp(-alpha*t)*(alpha*tau*u0-u0-alpha*t*y0+alpha*tau*y0+alpha*alpha*t*tau*y0))/powf(alpha*tau-1.0f,2.0f)-(gh*n0*tau_h*exp(-t/tau))/(tau*(tau-tau_h));
}

__device__ float dfun2( float t, float v0, float n0, float u0, float y0)
{
  return
    1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))+tau*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))+tau_h*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))-tau_h*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h)))*(I*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(-1.0/2.0)+(I*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*(1.0/2.0))/tau+(I*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*(1.0/2.0))/tau+(gs*u0*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*(1.0/2.0))/tau+(I*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(beta_centre*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/gamma_centre-(beta_centre*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(-gamma_centre*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+gamma_centre*tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+gamma_centre*tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))+gs*u0*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0)-alpha*gs*y0*((exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*(-t*tau-t*tau_h+tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+alpha*t*tau*tau_h*2.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(tau*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*2.0)/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0)-(alpha*gs*y0*((exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*(-t*tau-t*tau_h+tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+alpha*t*tau*tau_h*2.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(tau*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*2.0)/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0))*(1.0/2.0))/tau+(I*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/tau-alpha*gs*y0*((exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*(t*tau+t*tau_h-tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*t*tau*tau_h*2.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)+(tau*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*2.0)/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0)+(alpha*gs*y0*((exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*(t*tau+t*tau_h-tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*t*tau*tau_h*2.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)+(tau*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*2.0)/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0))*(1.0/2.0))/tau-(gs*tau_h*u0*exp(-alpha*t)*(alpha*exp(alpha*t)-(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h)))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(gs*tau_h*u0*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/tau-(beta_centre*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/4.0))/(gamma_centre*tau*tau_h)-(I*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0))-(beta_centre*tau*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/4.0))/(gamma_centre*tau_h)-(beta_centre*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/4.0))/(gamma_centre*tau)+(beta_centre*tau*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau_h*(gamma_centre*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*-2.0+gamma_centre*tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0+gamma_centre*tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0))+(beta_centre*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*(gamma_centre*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*-2.0+gamma_centre*tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0+gamma_centre*tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0))+(beta_centre*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/4.0))/(gamma_centre*tau*tau_h)+(alpha*gs*tau_h*y0*((exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*(t*tau+t*tau_h-tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*t*tau*tau_h*2.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)+(tau*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*2.0)/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/tau+(alpha*gs*tau_h*u0*exp(-alpha*t)*(exp(alpha*t)-exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(gs*(tau_h*tau_h)*u0*exp(-alpha*t)*(alpha*exp(alpha*t)-(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)+(alpha*gs*tau_h*y0*((exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*(-t*tau-t*tau_h+tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+alpha*t*tau*tau_h*2.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(tau*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*2.0)/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/tau+(alpha*gs*(tau_h*tau_h)*u0*exp(-alpha*t)*(exp(alpha*t)-exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h)))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)+(gs*tau*tau_h*u0*exp(-alpha*t)*(alpha*exp(alpha*t)-(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(alpha*gs*tau*tau_h*u0*exp(-alpha*t)*(exp(alpha*t)-exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h)))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0))*(-1.0/2.0)+1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*((exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/tau-(exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/tau_h-(exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/tau+(exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/tau_h+(exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h)+(exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(-v0+(beta_centre*(tau*tau)*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(gamma_centre*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*-2.0+gamma_centre*tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0+gamma_centre*tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)+(beta_centre*(tau_h*tau_h)*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(gamma_centre*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*-2.0+gamma_centre*tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0+gamma_centre*tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)+(I*tau_h*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))+(I*tau_h*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(I*(tau_h*tau_h)*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(beta_centre*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(gamma_centre*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)))-(beta_centre*tau*tau_h*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(-gamma_centre*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+gamma_centre*tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+gamma_centre*tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))+(gs*tau_h*u0*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))-1.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(beta_centre*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(gamma_centre*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)))-alpha*gs*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0+tau*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(t*tau+t*tau_h-tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0)+(alpha*gs*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0+tau*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(t*tau+t*tau_h-tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*t*tau*tau_h*2.0)*2.0)*(1.0/2.0))/tau+alpha*gs*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0-tau*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(-t*tau-t*tau_h+tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0)+(alpha*gs*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0-tau*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(-t*tau-t*tau_h+tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+alpha*t*tau*tau_h*2.0)*2.0)*(1.0/2.0))/tau+(I*tau*tau_h*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(I*(tau_h*tau_h)*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(I*tau*tau_h*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(gs*(tau_h*tau_h)*u0*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(gs*tau_h*u0*exp(-alpha*t)*(exp(alpha*t)-exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(beta_centre*(tau*tau)*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(gamma_centre*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)))-(beta_centre*(tau_h*tau_h)*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(gamma_centre*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)))+(alpha*gs*tau_h*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0+tau*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(t*tau+t*tau_h-tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/tau-(alpha*gs*tau_h*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0-tau*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(-t*tau-t*tau_h+tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/tau+(gs*tau*tau_h*u0*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(gs*(tau_h*tau_h)*u0*exp(-alpha*t)*(exp(alpha*t)-exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h)))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)+(beta_centre*tau*tau_h*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(gamma_centre*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)))+(gs*tau*tau_h*u0*exp(-alpha*t)*(exp(alpha*t)-exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h)))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0))*(1.0/2.0)+(1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*((tau*tau)*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))-(tau*tau)*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))+(tau_h*tau_h)*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))-(tau_h*tau_h)*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))-exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau_h*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*2.0+tau*tau_h*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*2.0)*(beta_centre*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(-1.0/2.0)+(beta_centre*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*(1.0/2.0))/tau_h+(beta_centre*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*(1.0/2.0))/tau_h+(beta_centre*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+I*gamma_centre*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-gamma_centre*gs*u0*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-(I*gamma_centre*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)))/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(beta_centre*tau*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/tau_h+alpha*gamma_centre*gs*y0*((exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*(t*tau+t*tau_h-tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*t*tau*tau_h*2.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)+(tau*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*2.0)/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-(beta_centre*tau*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau_h*(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0))+alpha*gamma_centre*gs*y0*((exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*(-t*tau-t*tau_h+tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+alpha*t*tau*tau_h*2.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(tau*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*2.0)/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-(gamma_centre*gs*tau*tau_h*u0*exp(-alpha*t)*(alpha*exp(alpha*t)-(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)+(alpha*gamma_centre*gs*tau*tau_h*u0*exp(-alpha*t)*(exp(alpha*t)-exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h)))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0))*(1.0/4.0))/(gamma_centre*tau)+(1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))+(tau*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/tau_h+(tau_h*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/tau-(tau*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/tau_h-(tau_h*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/tau+(exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h)-(exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(n0+(beta_centre*tau*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))+(beta_centre*tau*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(beta_centre*(tau*tau)*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(beta_centre*tau*tau_h*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(beta_centre*(tau*tau)*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(beta_centre*tau*tau_h*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))+alpha*gamma_centre*gs*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0+tau*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(t*tau+t*tau_h-tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*gamma_centre*gs*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0-tau*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(-t*tau-t*tau_h+tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-(I*gamma_centre*tau*tau_h*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*2.0)/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(I*gamma_centre*tau*tau_h*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(gamma_centre*gs*tau*tau_h*u0*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(gamma_centre*gs*tau*tau_h*u0*exp(-alpha*t)*(exp(alpha*t)-exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h)))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0))*(1.0/4.0))/(gamma_centre*tau);
}

__device__ float dfun3( float t, float v0, float n0, float u0, float y0)
{
  return
    (beta_right*gh*exp(-t/tau))/(tau-tau_h)-(v0*exp(-t/tau))/tau-(I*exp(-t/tau))/tau-(beta_right*gh*exp(-t/tau_h))/(tau-tau_h)+(gh*n0*exp(-t/tau_h))/(tau-tau_h)-(gs*exp(-alpha*t)*(-tau*y0*alpha*alpha+y0*alpha))/powf(alpha*tau-1.0f,2.0f)+(gs*exp(-t/tau)*(alpha*tau*u0-u0+alpha*tau*y0))/(tau*powf(alpha*tau-1.0f,2.0f))-(alpha*gs*exp(-alpha*t)*(alpha*tau*u0-u0-alpha*t*y0+alpha*tau*y0+alpha*alpha*t*tau*y0))/powf(alpha*tau-1.0f,2.0f)-(gh*n0*tau_h*exp(-t/tau))/(tau*(tau-tau_h));
}

__device__ float eventTimeZone1( float v0, float n0, float u0, float y0)
{
  float f, df, estimatedTime = 0.0f;

  f  = fun1( estimatedTime, v0, n0, u0, y0);
  df = dfun1( estimatedTime, v0, n0, u0, y0);

  while (fabs(f)>tol) {
    estimatedTime -= f/df;
    f  = fun1( estimatedTime, v0, n0, u0, y0);
    df = dfun1( estimatedTime, v0, n0, u0, y0);
  }

  return estimatedTime;
}

__device__ void eventTimeZone2( float v0, float n0, float u0, float y0,
                                 float *t, unsigned short *cross)
{
  float f, df;
  float estimatedTimeLeft  = 0.0f;
  float estimatedTimeRight = 0.0f;

  f  = fun2( estimatedTimeLeft, v0, n0, u0, y0, V_left);
  df = dfun2( estimatedTimeLeft, v0, n0, u0, y0);

  while (fabs(f)>tol) {
    estimatedTimeLeft -= f/df;
    f  = fun2( estimatedTimeLeft, v0, n0, u0, y0, V_left);
    df = dfun2( estimatedTimeLeft, v0, n0, u0, y0);
  }

  f  = fun2( estimatedTimeRight, v0, n0, u0, y0, V_right);
  df = dfun2( estimatedTimeRight, v0, n0, u0, y0);

  while (fabs(f)>tol) {
    estimatedTimeRight -= f/df;
    f  = fun2( estimatedTimeRight, v0, n0, u0, y0, V_right);
    df = dfun2( estimatedTimeRight, v0, n0, u0, y0);
  }

  *cross = 2;

  if (estimatedTimeRight<estimatedTimeLeft)
  {
    estimatedTimeLeft = estimatedTimeRight;
    *cross = 3;
  }
  *t = estimatedTimeLeft;
}

__device__ void eventTimeZone3( float v0, float n0, float u0, float y0,
    float *t, unsigned short *cross)
{
  float f, df;
  float estimatedTimeLeft  = 0.0f;
  float estimatedTimeRight = 0.0f;

  f  = fun3( estimatedTimeLeft, v0, n0, u0, y0, V_right);
  df = dfun3( estimatedTimeLeft, v0, n0, u0, y0);

  while (fabs(f)>tol) {
    estimatedTimeLeft -= f/df;
    f  = fun3( estimatedTimeLeft, v0, n0, u0, y0, V_right);
    df = dfun3( estimatedTimeLeft, v0, n0, u0, y0);
  }

  f  = fun3( estimatedTimeRight, v0, n0, u0, y0, V_th);
  df = dfun3( estimatedTimeRight, v0, n0, u0, y0);

  while (fabs(f)>tol) {
    estimatedTimeRight -= f/df;
    f  = fun3( estimatedTimeRight, v0, n0, u0, y0, V_th);
    df = dfun3( estimatedTimeRight, v0, n0, u0, y0);
  }

  *cross = 4;
  if (estimatedTimeRight<estimatedTimeLeft)
  {
    estimatedTimeLeft = estimatedTimeRight;
    *cross = 5;
  }
  *t = estimatedTimeLeft;
}

__global__ void eventTimeZone1Kernel( const float4* pGlobal_state,
                                      const unsigned short* pGlobalZone,
                                      struct firing* pVal)
{
  unsigned int index = threadIdx.x+blockDim.x*blockIdx.x;
  bool correct_zone  = (pGlobalZone[index] == 1);
  (*pVal).time = 100000.0f;
  if (correct_zone)
  {
    float4 local_state = pGlobal_state[index];
    pVal[index].time  = eventTimeZone1(local_state.x,local_state.y,local_state.z,local_state.w);
    pVal[index].index = index;
    pVal[index].cross = 1;
  }
}

__global__ void eventTimeZone2Kernel( const float4* pGlobal_state,
                                      const unsigned short* pGlobalZone,
                                      struct firing* pVal)
{
  unsigned int index = threadIdx.x+blockDim.x*blockIdx.x;
  bool correct_zone  = (pGlobalZone[index] == 2);
  float local_v = pGlobal_v[index];
  float local_n = pGlobal_n[index];
  float local_u = pGlobal_u[index];
  float local_y = pGlobal_y[index];
  float local_time = 1000000.0f;
  unsigned short cross;
  if (correct_zone)
  {
    eventTimeZone2(local_v,local_n,local_u,local_y,&local_time,&cross);
    pVal[index].time  = local_time;
    pVal[index].index = index;
    pVal[index].cross = cross;
  }
}

__global__ void eventTimeZone3Kernel( const float* pGlobal_state,
                                      const unsigned short* pGlobalZone,
                                      struct firing* pVal)
{
  unsigned int index = threadIdx.x+blockDim.x*blockIdx.x;
  bool correct_zone  = (pGlobalZone[index] == 3);
  float local_v = pGlobal_v[index];
  float local_n = pGlobal_n[index];
  float local_u = pGlobal_u[index];
  float local_y = pGlobal_y[index];
  float local_time = 1000000.0f;
  unsigned short cross;
  if (correct_zone)
  {
    eventTimeZone3(local_v,local_n,local_u,local_y,&local_time,&cross);
    pVal[index].time  = local_time;
    pVal[index].index = index;
    pVal[index].cross = cross;
  }
}

__global__ void eventTimeZone4Kernel( const float *pRefractTime,
                                      const unsigned short* pGlobalZone,
                                      struct firing* pVal)
{
  unsigned int index = threadIdx.x+blockDim.x*blockIdx.x;
  bool correct_zone  = (pGlobalZone[index] == 4);
  if (correct_zone)
  {
    pVal[index].time  = tau_r-pRefractTime[index];
    pVal[index].index = index;
    pVal[index].cross = 6;
  }
}

__inline__ __device__ struct firing warpReduceMin( struct firing val)
{
  float dummyTime;
  unsigned int dummyIndex;
  unsigned short dummyCross;
  for (int offset = warpSize/2; offset>0; offset/=2) {
    dummyTime  = __shfl_down( val.time, offset);
    dummyIndex = __shfl_down( val.index, offset);
    dummyCross = __shfl_down( val.cross, offset);
    if (dummyTime<val.time)
    {
      val.time  = dummyTime;
      val.index = dummyIndex;
      val.cross = dummyCross;
    }
  }
  return val;
}

__inline__ __device__ struct firing blockReduceMin( struct firing val)
{
  __shared__ struct EventDrivenMap::firing shared[32];
  int lane = threadIdx.x % warpSize;
  int wid  = threadIdx.x / warpSize;

  val = warpReduceMin( val);

  if (lane==0) {
    shared[wid] = val;
  }
  __syncthreads();

  val.time  = (threadIdx.x<blockDim.x/warpSize) ? shared[lane].time  : 0.0f;
  val.index = (threadIdx.x<blockDim.x/warpSize) ? shared[lane].index : 0;
  val.cross = (threadIdx.x<blockDim.x/warpSize) ? shared[lane].cross : 0;

  if (wid==0) {
    val = warpReduceMin( val);
  }

  return val;
}

__global__ void deviceReduceMinKernel( const struct firing* in,
                                       const unsigned int npts,
                                       struct firing* out);
{
  float time = 1000000.0f
  struct firing dummy;
  //reduce multiple elements per thread
  for (int i=blockIdx.x*blockDim.x+threadIdx.x;i<npts;i+=blockDim.x*gridDim.x)
  {
    dummy = in[i];
    if (dummy.time < time)
    {
      val  = dummy;
      time = dummy.time;
    }
  }
  val = blockReduceMin( val);
  if (threadIdx.x==0)
  {
    out[blockIdx.x] = val;
  }
}

__global__ void updateZone1Kernel( float* p_global_v,
                                   float* p_global_n,
                                   float* p_global_u,
                                   float* p_global_y,
                                   float eventTime)
{
  unsigned int index = threadIdx.x+blockDim.x*blockIdx.x;
  bool correct_zone  = (pGlobalZone[index] == 1);
  if (correct_zone)
  {
    
  }
}

int main( int argc , char *argv[])
{
  // Allocate memory
  float4* p_global_state;
  float* p_refract_time;
  struct firing* p_firing_val;
  struct firing* p_firing_val_temp;
  unsigned short* p_global_zone;

  unsigned int noThreads = 512;
  unsigned int noBlocks  = (N+noThreads-1)/noThreads;

  // Allocate memory
  CUDA_CALL( cudaMalloc( &p_global_state, N*sizeof(float4)));
  CUDA_CALL( cudaMalloc( &p_refract_time, N*sizeof(float)));
  CUDA_CALL( cudaMalloc( &p_global_zone, N*sizeof(short)));
  CUDA_CALL( cudaMalloc( &p_firing_val, N*sizeof(firing)));
  CUDA_CALL( cudaMalloc( &p_firing_val_temp, noBlocks*sizeof(firing)));

  InitialiseKernel<<<noBlocks,noThreads>>>(p_global_v,
                                           p_global_n,
                                           p_global_u,
                                           p_global_y);

  float finalTime   = 100.0f;
  float currentTime = 0.0f; // use pinned memory for this

  while (currentTime<finalTime)
  {
    eventTimeZone1Kernel<<<noBlocks,noThreads>>>
       (p_global_state,
        p_global_zone,
        p_firing_val);
    eventTimeZone2Kernel<<<noBlocks,noThreads>>>
       (p_global_state,
        p_global_zone,
        p_firing_val);
    eventTimeZone3Kernel<<<noBlocks,noThreads>>>
       (p_global_state,
        p_global_zone,
        p_firing_val);
    eventTimeZone4Kernel<<<noBlocks,noThreads>>>
      ( p_refract_time,
        p_global_zone,
        p_firing_val);

    // Find minimum spike time
    deviceReduceMinKernel<<<noBlocks,noThreads>>>
      ( p_firing_val, N, p_firing_val_temp);
    deviceReduceMinKernel<<<1,noThreads>>>
      ( p_firing_val_temp, noBlocks, p_firing_val_temp);

    // Update - assume transfer to page-locked memory
    updateZone1Kernel<<<noBlocks,noThreads>>>
      ( 

    if (crossType!=

  }

  cudaFree( global_v);
  cudaFree( global_n);
  cudaFree( global_u);
  cudaFree( global_y);
}
