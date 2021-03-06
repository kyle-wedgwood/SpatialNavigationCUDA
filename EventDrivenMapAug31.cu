/* Code to do one dimensional spiking model from Mayte's note */
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <curand.h>
#include <arrayfire.h>
#include "parameters.h"
#include "EventDrivenMap.hpp"
#include "vector_types.h"

#define CUDA_ERROR_CHECK
#define CUDA_CALL( err) __cudaCall( err, __FILE__, __LINE__ )
#define CUDA_CHECK_ERROR()    __cudaCheckError( __FILE__, __LINE__ )

__constant__ float I_app;

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

inline void __curandCall( curandStatus_t err, const char *file, const int line)
{
#ifdef CURAND_ERROR_CHECK
  if ( CURAND_STATUS_SUCCESS != err)
  {
    fprintf( stderr, "curandCall() failed at %s:%i",
                 file, line);
    exit( -1 );
  }
  #endif
}

EventDrivenMap::EventDrivenMap( const ParameterList* pParameterList)
{

  mNetworkSize = (*pParameterList).networkSize;
  mNoThreads   = (*pParameterList).noThreads;
  mNoBlocks    = (mNetworkSize+mNoThreads-1)/mNoThreads;
  mDomainSize  = (*pParameterList).domainSize;
  mDx          = 2.0f*mDomainSize/(mNetworkSize);
  mDt          = (*pParameterList).timestep;
  mPrintOutput = (*pParameterList).printOutput;

  // For plotting
  mpPlotVarX = new af::array(mNetworkSize,f32);
  mpPlotVarY = new af::array(mNetworkSize,f32);
  mpPlotVarYHelper = (*mpPlotVarY).device<float>();
  SetXAxis( mpPlotVarX);

  // Allocate memory for GPU variables
  CUDA_CALL( cudaMalloc( &mpGlobalState, mNetworkSize*sizeof(float4)));
  CUDA_CALL( cudaMalloc( &mpRefractTime, mNetworkSize*sizeof(float)));
  CUDA_CALL( cudaMalloc( &mpGlobalZone, mNetworkSize*sizeof(int)));
  CUDA_CALL( cudaMalloc( &mpFiringVal, mNetworkSize*sizeof(firing)));
  CUDA_CALL( cudaMalloc( &mpFiringValTemp, mNoBlocks*sizeof(firing)));
  CUDA_CALL( cudaMalloc( &mpEventNo, sizeof(int)));
  CUDA_CALL( cudaMallocHost( &mpHost_eventNo, sizeof(int)));

  // Allocate memory for CPU variables
  mpHost_firingVal = (firing*) malloc(sizeof(firing));

  // Set applied curent to zero
  SetAppliedCurrent( 0.0f);

  // Set some other options
  CalculateSpatialExtent();
  SetPlottingWindow();
  printf("Created network object with %d neurons.\n",mNetworkSize);
}

EventDrivenMap::~EventDrivenMap()
{
  (*mpPlotVarY).unlock();

  cudaFree( mpGlobalState);
  cudaFree( mpGlobalZone);
  cudaFree( mpRefractTime);
  cudaFree( mpFiringVal);
  cudaFree( mpFiringValTemp);
  cudaFree( mpEventNo);
  cudaFree( mpHost_firingVal);
  cudaFree( mpHost_eventNo);
  cudaFree( mpPlotVarYHelper);
  cudaFree( mpW);

  free( mpPlotVarX);
  free( mpPlotVarY);
  free( mpWindow);
}

void EventDrivenMap::SetAppliedCurrent( const float val)
{
  float temp_I = (-val/gl_steve);
  CUDA_CALL( cudaMemcpyToSymbol( I_app, &temp_I, sizeof(float)));
  printf("Applied current set to %f.\n",val);
}

void EventDrivenMap::SimulateNetwork( const float finalTime, bool extend)
{
  unsigned int counter = 0;
  if (!extend)
  {
    printf("Hello\n");
    InitialiseNetwork();
  }

  printf("Starting simulation.\n");
  do {
    SimulateStep();

    // Code to plot output
    if (counter%100==0)
    {
      PlotData();
    }

    // Prepare for next step
    mTime += mDt;
    counter++;
    if (mPrintOutput)
      printf("Current time = %f.\n",mTime);
  } while ((mTime<finalTime) & (!(*mpWindow).close()));

  printf("Simulation finished successfully.\n");
}

void EventDrivenMap::InitialiseNetwork()
{
  mTime = 0.0f;
  InitialiseNetworkKernel<<<mNoBlocks,mNoThreads>>>( mpGlobalState,
    mpGlobalZone, mNetworkSize);
  CUDA_CHECK_ERROR();
  ResetMemoryKernel<<<mNoBlocks,mNoThreads>>>( mpFiringVal, mNetworkSize, mpFiringValTemp, mNoBlocks, mDt);
  CUDA_CHECK_ERROR();
  printf("Network initialised.\n");
}

__global__ void InitialiseNetworkKernel( float4* pGlobalState,
                                         unsigned int* pGlobalZone,
                                         unsigned int networkSize)
{
  unsigned int index = threadIdx.x+blockDim.x*blockIdx.x;
  if (index<networkSize)
  {
    pGlobalState[index].x = V_left+10.0f;
    pGlobalState[index].y = gamma_centre*(V_left+10.0f)+beta_centre;
    pGlobalState[index].x = V_r;
    pGlobalState[index].y = 0.0f;
    pGlobalState[index].z = 0.0f;
    pGlobalState[index].w = 0.0f;
    pGlobalZone[index] = 2;
  }
}

void EventDrivenMap::CalculateSpatialExtent()
{
  float x = 0.0f;
  int i;
  for (i=0;i<mNetworkSize/2;++i)
  {
    if (W(x)<spatial_cutoff)
    {
      break;
    }
    x += mDx;
  }
  mSpatialExtent = i;
  printf("Trimmed synaptic kernel at tolerance = %f.\n",tol);
}

void EventDrivenMap::SimulateStep()
{
  float local_time = 0.0f;
  while (local_time<mDt)
  {
    // First, find spiking cell
    CUDA_CALL( cudaMemset( mpEventNo, 0, sizeof(int)));
    FindMinimumSpikeTime( mDt-local_time);
    if (mPrintOutput)
      printf("Found any spikes times.\n");
    CUDA_CALL( cudaMemcpy( mpHost_firingVal, mpFiringValTemp, sizeof(firing), cudaMemcpyDeviceToHost));
    if (mPrintOutput)
      printf("Taking step of size %f\n",(*mpHost_firingVal).time);

    // Update all cells
    if ((*mpHost_firingVal).time>tol)
    {
      UpdateStateKernel<<<mNoBlocks,mNoThreads>>>( (*mpHost_firingVal).time,
          mpGlobalState, mpGlobalZone, mpRefractTime, mNetworkSize);
      CUDA_CHECK_ERROR();
      if (mPrintOutput)
        printf("Updated neuron states.\n");
    }
    /*
    UpdateZone1Kernel<<<mNoBlocks,mNoThreads>>>( (*mpHost_firingVal).time,
       mpGlobalState, mpGlobalZone);
    if (mPrintOutput)
      printf("Updated neurons in zone 1.\n");
    CUDA_CHECK_ERROR();
    UpdateZone2Kernel<<<mNoBlocks,mNoThreads>>>( (*mpHost_firingVal).time,
       mpGlobalState, mpGlobalZone);
    if (mPrintOutput)
      printf("Updated neurons in zone 2.\n");
    CUDA_CHECK_ERROR();
    UpdateZone3Kernel<<<mNoBlocks,mNoThreads>>>( (*mpHost_firingVal).time,
       mpGlobalState, mpGlobalZone);
    if (mPrintOutput)
      printf("Updated neurons in zone 3.\n");
    CUDA_CHECK_ERROR();
    UpdateZone4Kernel<<<mNoBlocks,mNoThreads>>>( (*mpHost_firingVal).time,
       mpGlobalState, mpGlobalZone, mpRefractTime);
    CUDA_CHECK_ERROR();
    if (mPrintOutput)
      printf("Updated neurons in zone 4.\n");
    */

    // Update time
    local_time += (*mpHost_firingVal).time;

    // Reset neuron that fired
    if (*mpHost_eventNo>0)
    {
      ApplyResetKernel<<<(2*mSpatialExtent+mNoThreads-1)/mNoThreads,mNoThreads>>>( mpGlobalState,
          mpGlobalZone, (*mpHost_firingVal).index, mpRefractTime, mDx, mSpatialExtent);
      CUDA_CHECK_ERROR();
    }

    // Reset memory
    if ((*mpHost_firingVal).time<mDt)
    {
      if (*mpHost_eventNo==0)
      {
        ResetMemoryKernel<<<mNoBlocks,mNoThreads>>>( mpFiringVal, mNetworkSize, mpFiringValTemp, mNoBlocks, mDt);
        CUDA_CHECK_ERROR();
      }
      else
      {
        ResetMemoryKernel<<<(*mpHost_eventNo+mNoThreads-1)/mNoThreads,mNoThreads>>>( mpFiringVal, *mpHost_eventNo, mpFiringValTemp, mNoBlocks, mDt-local_time);
        CUDA_CHECK_ERROR();
      }
    }
    if (mPrintOutput)
    {
      printf("Memory reset.\n");
      printf("Step finished.\n");
    }
  }
}

void EventDrivenMap::SetXAxis( af::array* mpPlotVarX)
{
  *mpPlotVarX = af::range( mNetworkSize);
  *mpPlotVarX *= 2*mDomainSize/(mNetworkSize);
  *mpPlotVarX -= mDomainSize;
  //for (int i=0;i<mNetworkSize;++i)
  //{
  //  (*mpPlotVarX)(i) = -mDomainSize+2*mDomainSize/(mNetworkSize)*i;
  //}
}

void EventDrivenMap::SetPlottingWindow()
{
  mpWindow = new af::Window(mNetworkSize,512,"1D IF Network");
  printf("Plotting window created.\n");
}

void EventDrivenMap::PlotData()
{
  CopyDataToPlotBufferKernel<<<mNoBlocks,mNoThreads>>>( mpPlotVarYHelper, mpGlobalState, mNetworkSize);
  char str[50];
  sprintf(str,"1D IF Network. Time = %f",mTime);
  mpWindow->setTitle(str);
  mpWindow->plot( *mpPlotVarX, *mpPlotVarY);
  mpWindow->show();
}

__global__ void CopyDataToPlotBufferKernel( float* pPlotVarY, const float4* pGlobalState, const unsigned int networkSize)
{
  unsigned int index = threadIdx.x+blockDim.x*blockIdx.x;
  if (index<networkSize)
  {
    pPlotVarY[index] = pGlobalState[index].x;
  }
}

__global__ void UpdateStateKernel( const float eventTime,
                                   float4* pGlobalState,
                                   unsigned int* pGlobalZone,
                                   float* pRefractTime,
                                   const unsigned int networkSize)
{
  unsigned int index = threadIdx.x+blockDim.x*blockIdx.x;
  unsigned int local_zone;
  float4 local_state;
  short change_flag = 0;
  if (index<networkSize)
  {
    local_zone  = pGlobalZone[index];
    local_state = pGlobalState[index];
    if (local_zone==1)
    {
      local_state = UpdateZone1( eventTime, local_state, index, &change_flag);
    }
    if (local_zone==2)
    {
      local_state = UpdateZone2( eventTime, local_state, index, &change_flag);
    }
    if (local_zone==3)
    {
      local_state = UpdateZone3( eventTime, local_state, index, &change_flag);
    }
    if (local_zone==4)
    {
      float local_refract_time = pRefractTime[index];
      local_state = UpdateZone4( eventTime, local_state, &local_refract_time,
          index, &change_flag);
      pRefractTime[index] = local_refract_time;
    }
    pGlobalState[index] = local_state;
    pGlobalZone[index]  = local_zone+change_flag;
  }
}

__global__ void UpdateZone1Kernel( const float eventTime,
                                   float4* pGlobalState,
                                   unsigned int *pGlobalZone)
{
  unsigned int index = threadIdx.x+blockDim.x*blockIdx.x;
  if (pGlobalZone[index]==1)
  {
    float4 local_state = pGlobalState[index];
    //local_state = UpdateZone1( eventTime, local_state);
    pGlobalState[index] = local_state;
    pGlobalZone[index] += (local_state.x>V_left);
  }
}

__device__ float4 UpdateZone1( float eventTime,
                               float4 state,
                               unsigned int index,
                               short* pChangeFlag)
{
  float crossTime = eventTime;
  short changeZoneFlag = 0;
  float local_I =
    (-I_steve/gl_steve)+I_app*(index>=I_app_first)*(index<I_app_last);
  float v = fun1( crossTime, state.x, state.y, state.z, state.w, local_I, 0.0f);
  if (threadIdx.x==0)
    printf("Voltage = %f\n",v);
  if (v > V_left)
  {
    changeZoneFlag = 1;
    float dv = dfun1( crossTime, state.x, state.y, state.z, state.w, local_I);
    int iterate = 0;
    while (fabs(v)>tol)
    {
      crossTime -= 0.1f*v/dv;
      v  = fun1( crossTime, state.x, state.y, state.z, state.w, local_I, V_left);
      dv = dfun1( crossTime, state.x, state.y, state.z, state.w, local_I);
      if (threadIdx.x==0)
        printf("Iterate = %d,\t v = %f,\t zone = 1,\t time = %f\n",iterate,v+V_left,crossTime);
      iterate++;
    }
  }
  state = UpdateStateZone1( crossTime,local_I, state);
  if (changeZoneFlag)
  {
    state.x = V_left;
    state = UpdateStateZone2( eventTime-crossTime, local_I, state);
  }

  *pChangeFlag = changeZoneFlag; // NOTE: since we are defining change_flags if the function that calls this one, we don't actually need another variable here. This is mostly to protect against failure to define variables
  return state;
}

__device__ float4 UpdateStateZone1( float t, float I, float4 state)
{
  state.x = (state.x*expf(-t/tau)
    +(beta_left*gh*(tau-tau_h+tau_h*expf(-t/tau_h)))/(tau-tau_h)
    -(beta_left*gh*tau*expf(-t/tau))/(tau-tau_h)
    -(gh*state.y*tau_h*expf(-t/tau_h))/(tau-tau_h)
    +(gh*state.y*tau_h*expf(-t/tau))/(tau-tau_h)
    +(gs*expf(-alpha*t)*(alpha*tau*state.z-state.z-alpha*t*state.w+alpha*tau*state.w+alpha*alpha*t*tau*state.w))/powf(alpha*tau-1.0f,2)
    -(gs*expf(-t/tau)*(alpha*tau*state.z-state.z+alpha*tau*state.w))/powf(alpha*tau-1.0f,2)
    -I+I*exp(-t/tau));
  state.y = (beta_left*(1.0f-expf(-t/tau_h))+state.y*expf(-t/tau_h));
  state.z = (state.z+alpha*state.w*t)*expf(-alpha*t);
  state.w = state.w*expf(-alpha*t);
  return state;
}


__global__ void UpdateZone2Kernel( const float eventTime,
                                   float4* pGlobalState,
                                   unsigned int* pGlobalZone)
{
  unsigned int index = threadIdx.x+blockDim.x*blockIdx.x;
  unsigned int local_zone = pGlobalZone[index];
  if (local_zone==2)
  {
    float4 local_state = pGlobalState[index];

    // Update state
    //local_state = UpdateZone2( eventTime, local_state);
    pGlobalState[index] = local_state;

    // Update zone
    local_zone += (local_state.x>V_right);
    local_zone -= (local_state.x<V_left);
    if (index==0)
      printf("zone = %d\n",local_zone);
    pGlobalZone[index] = local_zone;
  }
}

__device__ float4 UpdateZone2( float eventTime,
                               float4 state,
                               unsigned int index,
                               short* pChangeFlag)
{
  float crossTime = eventTime;
  short changeZoneFlag = 0;
  float local_I =
    (-I_steve/gl_steve)+I_app*(index>=I_app_first)*(index<I_app_last);

  float v = fun2( crossTime, state.x, state.y, state.z, state.w, local_I, 0.0f);
  if (v > V_right)
  {
    changeZoneFlag = 1;
    v -= V_right;
    float dv = dfun2( crossTime, state.x, state.y, state.z, state.w, local_I);
    int iterate = 0;
    while (fabs(v)>tol)
    {
      crossTime -= v/dv;
      v  = fun2( crossTime, state.x, state.y, state.z, state.w, local_I, V_right);
      dv = dfun2( crossTime, state.x, state.y, state.z, state.w, local_I);
      if (threadIdx.x==0)
        printf("Iterate = %d,\t v = %f,\t zone = 2, \t time = %f\n",iterate,v+V_right,crossTime);
      iterate++;
    }
  }
  if (v < V_left)
  {
    changeZoneFlag = -1;
    v -= V_left;
    int iterate = 0;
    float dv = dfun2( crossTime, state.x, state.y, state.z, state.w, local_I);
    while (fabs(v)>tol)
    {
      crossTime -= v/dv;
      v  = fun2( crossTime, state.x, state.y, state.z, state.w, local_I, V_left);
      dv = dfun2( crossTime, state.x, state.y, state.z, state.w, local_I);
      if (threadIdx.x==0)
        printf("Iterate = %d,\t v = %f,\t zone = 2,\t time = %f\n",iterate,(v+V_left),crossTime);
      iterate++;
    }
  }
  state = UpdateStateZone2( crossTime, local_I, state);
  if (changeZoneFlag==1)
  {
    state.x = V_right;
    state = UpdateStateZone3( eventTime-crossTime, local_I, state);
  }
  if (changeZoneFlag==-1)
  {
    state.x = V_left;
    state = UpdateStateZone1( eventTime-crossTime, local_I, state);
  }
  if (threadIdx.x==0)
    printf("Voltage = %f.\n",state.x);

  *pChangeFlag = changeZoneFlag; // NOTE: See note for UpdateZone1
  return state;
}

__device__ float4 UpdateStateZone2( float t, float I, float4 state)
{
  float v0 = state.x;
  state.x =
    (1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))+tau*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))+tau_h*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))-tau_h*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h)))*(-v0+(beta_centre*(tau*tau)*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(gamma_centre*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*-2.0+gamma_centre*tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0+gamma_centre*tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)+(beta_centre*(tau_h*tau_h)*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(gamma_centre*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*-2.0+gamma_centre*tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0+gamma_centre*tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)+(I*tau_h*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))+(I*tau_h*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(I*(tau_h*tau_h)*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(beta_centre*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(gamma_centre*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)))-(beta_centre*tau*tau_h*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(-gamma_centre*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+gamma_centre*tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+gamma_centre*tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))+(gs*tau_h*state.z*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))-1.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(beta_centre*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(gamma_centre*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)))-alpha*gs*state.w*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0+tau*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(t*tau+t*tau_h-tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0)+(alpha*gs*state.w*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0+tau*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(t*tau+t*tau_h-tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*t*tau*tau_h*2.0)*2.0)*(1.0/2.0))/tau+alpha*gs*state.w*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0-tau*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(-t*tau-t*tau_h+tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0)+(alpha*gs*state.w*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0-tau*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(-t*tau-t*tau_h+tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+alpha*t*tau*tau_h*2.0)*2.0)*(1.0/2.0))/tau+(I*tau*tau_h*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(I*(tau_h*tau_h)*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(I*tau*tau_h*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(gs*(tau_h*tau_h)*state.z*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(gs*tau_h*state.z*exp(-alpha*t)*(exp(alpha*t)-exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(beta_centre*(tau*tau)*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(gamma_centre*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)))-(beta_centre*(tau_h*tau_h)*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(gamma_centre*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)))+(alpha*gs*tau_h*state.w*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0+tau*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(t*tau+t*tau_h-tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/tau-(alpha*gs*tau_h*state.w*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0-tau*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(-t*tau-t*tau_h+tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/tau+(gs*tau*tau_h*state.z*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(gs*(tau_h*tau_h)*state.z*exp(-alpha*t)*(exp(alpha*t)-exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h)))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)+(beta_centre*tau*tau_h*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(gamma_centre*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)))+(gs*tau*tau_h*state.z*exp(-alpha*t)*(exp(alpha*t)-exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h)))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0))*(-1.0/2.0)+(1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*((tau*tau)*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))-(tau*tau)*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))+(tau_h*tau_h)*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))-(tau_h*tau_h)*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))-exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau_h*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*2.0+tau*tau_h*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*2.0)*(state.y+(beta_centre*tau*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))+(beta_centre*tau*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(beta_centre*(tau*tau)*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(beta_centre*tau*tau_h*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(beta_centre*(tau*tau)*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(beta_centre*tau*tau_h*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))+alpha*gamma_centre*gs*state.w*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0+tau*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(t*tau+t*tau_h-tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*gamma_centre*gs*state.w*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0-tau*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(-t*tau-t*tau_h+tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-(I*gamma_centre*tau*tau_h*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*2.0)/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(I*gamma_centre*tau*tau_h*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(gamma_centre*gs*tau*tau_h*state.z*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(gamma_centre*gs*tau*tau_h*state.z*exp(-alpha*t)*(exp(alpha*t)-exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h)))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0))*(1.0/4.0))/(gamma_centre*tau));
  state.y =
    /*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(exp(-(t*tau*(1.0/2.0)+t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+exp(-(t*tau*(1.0/2.0)+t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*exp(-(t*tau*(1.0/2.0)+t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))+tau*exp(-(t*tau*(1.0/2.0)+t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))+tau_h*exp(-(t*tau*(1.0/2.0)+t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))-tau_h*exp(-(t*tau*(1.0/2.0)+t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h)))*(n0+(beta_centre*tau*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))+(beta_centre*tau*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(beta_centre*(tau*tau)*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(beta_centre*tau*tau_h*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(beta_centre*(tau*tau)*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(beta_centre*tau*tau_h*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))+alpha*gamma_centre*gs*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0+tau*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(t*tau+t*tau_h-tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*gamma_centre*gs*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0-tau*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(-t*tau-t*tau_h+tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-(I*gamma_centre*tau*tau_h*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*2.0)/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(I*gamma_centre*tau*tau_h*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(gamma_centre*gs*tau*tau_h*u0*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(gamma_centre*gs*tau*tau_h*u0*exp(-alpha*t)*(exp(alpha*t)-exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h)))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0))*(1.0/2.0)-(gamma_centre*tau*exp(-(t*tau*(1.0/2.0)+t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))-gamma_centre*tau*exp(-(t*tau*(1.0/2.0)+t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h)))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(-v0+(beta_centre*(tau*tau)*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(gamma_centre*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*-2.0+gamma_centre*tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0+gamma_centre*tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)+(beta_centre*(tau_h*tau_h)*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(gamma_centre*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*-2.0+gamma_centre*tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0+gamma_centre*tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)+(I*tau_h*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))+(I*tau_h*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(I*(tau_h*tau_h)*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(beta_centre*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(gamma_centre*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)))-(beta_centre*tau*tau_h*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(-gamma_centre*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+gamma_centre*tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+gamma_centre*tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))+(gs*tau_h*u0*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))-1.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(beta_centre*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(gamma_centre*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)))-alpha*gs*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0+tau*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(t*tau+t*tau_h-tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0)+(alpha*gs*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0+tau*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(t*tau+t*tau_h-tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*t*tau*tau_h*2.0)*2.0)*(1.0/2.0))/tau+alpha*gs*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0-tau*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(-t*tau-t*tau_h+tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0)+(alpha*gs*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0-tau*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(-t*tau-t*tau_h+tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+alpha*t*tau*tau_h*2.0)*2.0)*(1.0/2.0))/tau+(I*tau*tau_h*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(I*(tau_h*tau_h)*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(I*tau*tau_h*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(gs*(tau_h*tau_h)*u0*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(gs*tau_h*u0*exp(-alpha*t)*(exp(alpha*t)-exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(beta_centre*(tau*tau)*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(gamma_centre*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)))-(beta_centre*(tau_h*tau_h)*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(gamma_centre*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)))+(alpha*gs*tau_h*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0+tau*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(t*tau+t*tau_h-tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/tau-(alpha*gs*tau_h*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0-tau*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(-t*tau-t*tau_h+tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/tau+(gs*tau*tau_h*u0*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(gs*(tau_h*tau_h)*u0*exp(-alpha*t)*(exp(alpha*t)-exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h)))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)+(beta_centre*tau*tau_h*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(gamma_centre*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)))+(gs*tau*tau_h*u0*exp(-alpha*t)*(exp(alpha*t)-exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h)))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0));
      */
    (1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))-tau*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))-tau_h*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))+tau_h*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h)))*(state.y+(beta_centre*tau*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))+(beta_centre*tau*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(beta_centre*(tau*tau)*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(beta_centre*tau*tau_h*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(beta_centre*(tau*tau)*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(beta_centre*tau*tau_h*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))+alpha*gamma_centre*gs*state.w*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0+tau*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(t*tau+t*tau_h-tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*gamma_centre*gs*state.w*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0-tau*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(-t*tau-t*tau_h+tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-(I*gamma_centre*tau*tau_h*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*2.0)/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(I*gamma_centre*tau*tau_h*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(gamma_centre*gs*tau*tau_h*state.z*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(gamma_centre*gs*tau*tau_h*state.z*exp(-alpha*t)*(exp(alpha*t)-exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h)))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0))*(1.0/2.0)+(gamma_centre*tau*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))-gamma_centre*tau*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h)))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(-v0+(beta_centre*(tau*tau)*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(gamma_centre*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*-2.0+gamma_centre*tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0+gamma_centre*tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)+(beta_centre*(tau_h*tau_h)*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(gamma_centre*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*-2.0+gamma_centre*tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0+gamma_centre*tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)+(I*tau_h*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))+(I*tau_h*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(I*(tau_h*tau_h)*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(beta_centre*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(gamma_centre*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)))-(beta_centre*tau*tau_h*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(-gamma_centre*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+gamma_centre*tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+gamma_centre*tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))+(gs*tau_h*state.z*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))-1.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(beta_centre*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(gamma_centre*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)))-alpha*gs*state.w*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0+tau*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(t*tau+t*tau_h-tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0)+(alpha*gs*state.w*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0+tau*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(t*tau+t*tau_h-tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*t*tau*tau_h*2.0)*2.0)*(1.0/2.0))/tau+alpha*gs*state.w*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0-tau*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(-t*tau-t*tau_h+tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0)+(alpha*gs*state.w*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0-tau*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(-t*tau-t*tau_h+tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+alpha*t*tau*tau_h*2.0)*2.0)*(1.0/2.0))/tau+(I*tau*tau_h*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(I*(tau_h*tau_h)*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(I*tau*tau_h*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(gs*(tau_h*tau_h)*state.z*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(gs*tau_h*state.z*exp(-alpha*t)*(exp(alpha*t)-exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(beta_centre*(tau*tau)*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(gamma_centre*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)))-(beta_centre*(tau_h*tau_h)*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(gamma_centre*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)))+(alpha*gs*tau_h*state.w*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0+tau*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(t*tau+t*tau_h-tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/tau-(alpha*gs*tau_h*state.w*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0-tau*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(-t*tau-t*tau_h+tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/tau+(gs*tau*tau_h*state.z*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(gs*(tau_h*tau_h)*state.z*exp(-alpha*t)*(exp(alpha*t)-exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h)))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)+(beta_centre*tau*tau_h*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(gamma_centre*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)))+(gs*tau*tau_h*state.z*exp(-alpha*t)*(exp(alpha*t)-exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h)))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)));
  state.z = (state.z+alpha*state.w*t)*expf(-alpha*t);
  state.w = state.w*expf(-alpha*t);
  return state;
}

__global__ void UpdateZone3Kernel( const float eventTime,
                                   float4* pGlobalState,
                                   unsigned int* pGlobalZone)
{
  unsigned int index = threadIdx.x+blockDim.x*blockIdx.x;
  bool correct_zone  = (pGlobalZone[index] == 3);
  if (correct_zone)
  {
    float4 local_state = pGlobalState[index];

    // Update state
    //local_state = UpdateZone3( eventTime, local_state);
    pGlobalState[index] = local_state;

    // Update zone
    pGlobalZone[index] -= (local_state.x<V_right);
  }
}

__device__ float4 UpdateZone3( float eventTime,
                               float4 state,
                               unsigned index,
                               short* pChangeFlag)
{
  float crossTime = eventTime;
  short changeZoneFlag = 0;
  float local_I =
    (-I_steve/gl_steve)+I_app*(index>=I_app_first)*(index<I_app_last);
  float v = fun3( crossTime, state.x, state.y, state.z, state.w, local_I, 0.0f);
  if (v < V_right)
  {
    changeZoneFlag = -1;
    v  = fun3( crossTime, state.x, state.y, state.z, state.w, local_I, V_right);
    float dv = dfun3( crossTime, state.x, state.y, state.z, state.w, local_I);
    while (fabs(v)>tol)
    {
      int iterate = 0;
      crossTime -= v/dv;
      v  = fun3( crossTime, state.x, state.y, state.z, state.w, local_I, V_right);
      dv = dfun3( crossTime, state.x, state.y, state.z, state.w, local_I);
      if (threadIdx.x==0)
        printf("Iterate = %d,\t v = %f,\t zone = 3, \t time = %f\n",iterate,v,crossTime);
      iterate++;
    }
  }
  state = UpdateStateZone3( crossTime, local_I, state);
  if (changeZoneFlag==-1)
  {
    state.x = V_right;
    state = UpdateStateZone2( eventTime-crossTime, local_I, state);
  }

  *pChangeFlag = changeZoneFlag; // NOTE: See note for UpdateZone1
  return state;
}

__device__ float4 UpdateStateZone3( float t, float I, float4 state)
{
  state.x =
        (state.x*expf(-t/tau)
    +(beta_right*gh*(tau-tau_h+tau_h*expf(-t/tau_h)))/(tau-tau_h)
    -(beta_right*gh*tau*expf(-t/tau))/(tau-tau_h)
    -(gh*state.y*tau_h*expf(-t/tau_h))/(tau-tau_h)
    +(gh*state.y*tau_h*expf(-t/tau))/(tau-tau_h)
    +(gs*expf(-alpha*t)*(alpha*tau*state.z-state.z-alpha*t*state.w+alpha*tau*state.w+alpha*alpha*t*tau*state.w))/powf(alpha*tau-1.0f,2)
    -(gs*expf(-t/tau)*(alpha*tau*state.z-state.z+alpha*tau*state.w))/powf(alpha*tau-1.0f,2)
    -I +I*exp(-t/tau));
  state.y = (beta_right*(1.0f-expf(-t/tau_h))+state.y*expf(-t/tau_h));
  state.z = (state.z+alpha*state.w*t)*expf(-alpha*t);
  state.w = state.w*expf(-alpha*t);
  return state;
}

__device__ float4 UpdateZone4( const float eventTime,
                               float4 state,
                               float* pRefractTime,
                               unsigned int index,
                               short* pChangeFlag)
{
  float local_refract_time = *pRefractTime;
  short local_change_flag = 0;
  float sim_time = min( *pRefractTime, eventTime);
  float local_I =
    (-I_steve/gl_steve)+I_app*(index>=I_app_first)*(index<I_app_last);

  state = UpdateStateZone4( sim_time, state);
  local_refract_time -= eventTime;
  if (local_refract_time<0.0f)
  {
    state = UpdateStateZone2( -local_refract_time, local_I, state);
    local_refract_time = 0.0f;
    local_change_flag  = -2;
  }

  *pRefractTime = local_refract_time;
  *pChangeFlag  = local_change_flag;
  return state;
}

__global__ void UpdateZone4Kernel( const float eventTime,
                                   float4* pGlobalState,
                                   unsigned int* pGlobalZone,
                                   float* pRefractTime)
{
  unsigned int index = threadIdx.x+blockDim.x*blockIdx.x;
  bool correct_zone  = (pGlobalZone[index] == 4);
  if (correct_zone)
  {
    // Update state
    float4 local_state = pGlobalState[index];
    float local_refract_time = pRefractTime[index];
    float sim_time = min( local_refract_time, eventTime);
    float local_I =
      (-I_steve/gl_steve)+I_app*(index>=I_app_first)*(index<I_app_last);
    local_state = UpdateStateZone4( sim_time, local_state);
    local_refract_time -= eventTime;
    local_refract_time *= (-1);
    if (local_refract_time>0.0f)
    {
      local_state = UpdateStateZone2( local_refract_time, local_I, local_state);
    }

    // Update zone
    local_refract_time *= (-1);
    if (local_refract_time > 0.0f)
    {
      local_refract_time = 0.0f;
      pGlobalZone[index]  = 2;
    }
    pRefractTime[index] = local_refract_time;
  }
}

__device__ float4 UpdateStateZone4( float t, float4 state)
{
  state.x = V_r;
  state.y = ((gamma_centre*V_r+beta_centre)*(1.0f-expf(-t/tau_h))+state.y*expf(-t/tau_h));
  state.z = (state.z+alpha*state.w*t)*expf(-alpha*t);
  state.w = state.w*expf(-alpha*t);
  return state;
}

__global__ void ResetMemoryKernel( EventDrivenMap::firing* pFiringVal,
                                   const unsigned int resetSize,
                                   EventDrivenMap::firing *pFiringValTemp,
                                   const unsigned int resetSizeTemp,
                                   const float stepSize)
{
  unsigned int index =  threadIdx.x+blockDim.x*blockIdx.x;
  if (index<resetSize)
  {
    pFiringVal[index].time  = stepSize;
    pFiringVal[index].index = 0;
  }
  if (index<resetSizeTemp)
  {
    pFiringValTemp[index].time  = stepSize;
    pFiringValTemp[index].index = 0;
  }
}

__device__ float fun1( float t, float v0, float n0, float u0, float y0, float I, float thresh)
{
  return v0*expf(-t/tau)
  +(beta_left*gh*(tau-tau_h+tau_h*expf(-t/tau_h)))/(tau-tau_h)
  -(beta_left*gh*tau*expf(-t/tau))/(tau-tau_h)
  -(gh*n0*tau_h*expf(-t/tau_h))/(tau-tau_h)
  +(gh*n0*tau_h*expf(-t/tau))/(tau-tau_h)
  +(gs*expf(-alpha*t)*(alpha*tau*u0-u0-alpha*t*y0+alpha*tau*y0+powf(alpha,2)*t*tau*y0))/powf(alpha*tau-1,2)
  -(gs*expf(-t/tau)*(alpha*tau*u0-u0+alpha*tau*y0))/powf(alpha*tau-1,2)
  -I+I*expf(-t/tau)-thresh;
}

__device__ float fun2( float t, float v0, float n0, float u0, float y0, float I, float thresh)
{
  return
  1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))+tau*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))+tau_h*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))-tau_h*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h)))*(-v0+(beta_centre*(tau*tau)*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(gamma_centre*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*-2.0+gamma_centre*tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0+gamma_centre*tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)+(beta_centre*(tau_h*tau_h)*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(gamma_centre*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*-2.0+gamma_centre*tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0+gamma_centre*tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)+(I*tau_h*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))+(I*tau_h*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(I*(tau_h*tau_h)*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(beta_centre*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(gamma_centre*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)))-(beta_centre*tau*tau_h*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(-gamma_centre*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+gamma_centre*tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+gamma_centre*tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))+(gs*tau_h*u0*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))-1.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(beta_centre*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(gamma_centre*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)))-alpha*gs*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0+tau*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(t*tau+t*tau_h-tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0)+(alpha*gs*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0+tau*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(t*tau+t*tau_h-tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*t*tau*tau_h*2.0)*2.0)*(1.0/2.0))/tau+alpha*gs*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0-tau*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(-t*tau-t*tau_h+tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0)+(alpha*gs*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0-tau*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(-t*tau-t*tau_h+tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+alpha*t*tau*tau_h*2.0)*2.0)*(1.0/2.0))/tau+(I*tau*tau_h*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(I*(tau_h*tau_h)*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(I*tau*tau_h*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(gs*(tau_h*tau_h)*u0*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(gs*tau_h*u0*exp(-alpha*t)*(exp(alpha*t)-exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(beta_centre*(tau*tau)*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(gamma_centre*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)))-(beta_centre*(tau_h*tau_h)*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(gamma_centre*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)))+(alpha*gs*tau_h*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0+tau*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(t*tau+t*tau_h-tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/tau-(alpha*gs*tau_h*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0-tau*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(-t*tau-t*tau_h+tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/tau+(gs*tau*tau_h*u0*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(gs*(tau_h*tau_h)*u0*exp(-alpha*t)*(exp(alpha*t)-exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h)))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)+(beta_centre*tau*tau_h*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(gamma_centre*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)))+(gs*tau*tau_h*u0*exp(-alpha*t)*(exp(alpha*t)-exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h)))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0))*(-1.0/2.0)+(1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*((tau*tau)*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))-(tau*tau)*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))+(tau_h*tau_h)*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))-(tau_h*tau_h)*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))-exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau_h*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*2.0+tau*tau_h*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*2.0)*(n0+(beta_centre*tau*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))+(beta_centre*tau*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(beta_centre*(tau*tau)*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(beta_centre*tau*tau_h*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(beta_centre*(tau*tau)*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(beta_centre*tau*tau_h*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))+alpha*gamma_centre*gs*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0+tau*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(t*tau+t*tau_h-tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*gamma_centre*gs*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0-tau*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(-t*tau-t*tau_h+tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-(I*gamma_centre*tau*tau_h*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*2.0)/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(I*gamma_centre*tau*tau_h*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(gamma_centre*gs*tau*tau_h*u0*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(gamma_centre*gs*tau*tau_h*u0*exp(-alpha*t)*(exp(alpha*t)-exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h)))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0))*(1.0/4.0))/(gamma_centre*tau)
  - thresh;
}

__device__ float fun3( float t, float v0, float n0, float u0, float y0, float I, float thresh)
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

__device__ float dfun1( float t, float v0, float n0, float u0, float y0, float I)
{
  return
    (beta_right*gh*exp(-t/tau))/(tau-tau_h)-(v0*exp(-t/tau))/tau-(I*exp(-t/tau))/tau-(beta_right*gh*exp(-t/tau_h))/(tau-tau_h)+(gh*n0*exp(-t/tau_h))/(tau-tau_h)-(gs*exp(-alpha*t)*(-tau*y0*alpha*alpha+y0*alpha))/powf(alpha*tau-1.0f,2.0f)+(gs*exp(-t/tau)*(alpha*tau*u0-u0+alpha*tau*y0))/(tau*powf(alpha*tau-1.0f,2.0f))-(alpha*gs*exp(-alpha*t)*(alpha*tau*u0-u0-alpha*t*y0+alpha*tau*y0+alpha*alpha*t*tau*y0))/powf(alpha*tau-1.0f,2.0f)-(gh*n0*tau_h*exp(-t/tau))/(tau*(tau-tau_h));
}

__device__ float dfun2( float t, float v0, float n0, float u0, float y0, float I)
{
  return
    1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))+tau*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))+tau_h*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))-tau_h*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h)))*(I*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(-1.0/2.0)+(I*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*(1.0/2.0))/tau+(I*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*(1.0/2.0))/tau+(gs*u0*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*(1.0/2.0))/tau+(I*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(beta_centre*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/gamma_centre-(beta_centre*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(-gamma_centre*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+gamma_centre*tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+gamma_centre*tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))+gs*u0*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0)-alpha*gs*y0*((exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*(-t*tau-t*tau_h+tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+alpha*t*tau*tau_h*2.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(tau*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*2.0)/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0)-(alpha*gs*y0*((exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*(-t*tau-t*tau_h+tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+alpha*t*tau*tau_h*2.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(tau*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*2.0)/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0))*(1.0/2.0))/tau+(I*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/tau-alpha*gs*y0*((exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*(t*tau+t*tau_h-tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*t*tau*tau_h*2.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)+(tau*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*2.0)/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0)+(alpha*gs*y0*((exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*(t*tau+t*tau_h-tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*t*tau*tau_h*2.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)+(tau*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*2.0)/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0))*(1.0/2.0))/tau-(gs*tau_h*u0*exp(-alpha*t)*(alpha*exp(alpha*t)-(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h)))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(gs*tau_h*u0*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/tau-(beta_centre*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/4.0))/(gamma_centre*tau*tau_h)-(I*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0))-(beta_centre*tau*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/4.0))/(gamma_centre*tau_h)-(beta_centre*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/4.0))/(gamma_centre*tau)+(beta_centre*tau*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau_h*(gamma_centre*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*-2.0+gamma_centre*tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0+gamma_centre*tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0))+(beta_centre*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*(gamma_centre*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*-2.0+gamma_centre*tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0+gamma_centre*tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0))+(beta_centre*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/4.0))/(gamma_centre*tau*tau_h)+(alpha*gs*tau_h*y0*((exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*(t*tau+t*tau_h-tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*t*tau*tau_h*2.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)+(tau*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*2.0)/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/tau+(alpha*gs*tau_h*u0*exp(-alpha*t)*(exp(alpha*t)-exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(gs*(tau_h*tau_h)*u0*exp(-alpha*t)*(alpha*exp(alpha*t)-(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)+(alpha*gs*tau_h*y0*((exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*(-t*tau-t*tau_h+tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+alpha*t*tau*tau_h*2.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(tau*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*2.0)/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/tau+(alpha*gs*(tau_h*tau_h)*u0*exp(-alpha*t)*(exp(alpha*t)-exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h)))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)+(gs*tau*tau_h*u0*exp(-alpha*t)*(alpha*exp(alpha*t)-(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(alpha*gs*tau*tau_h*u0*exp(-alpha*t)*(exp(alpha*t)-exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h)))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0))*(-1.0/2.0)+1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*((exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/tau-(exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/tau_h-(exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/tau+(exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/tau_h+(exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h)+(exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(-v0+(beta_centre*(tau*tau)*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(gamma_centre*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*-2.0+gamma_centre*tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0+gamma_centre*tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)+(beta_centre*(tau_h*tau_h)*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(gamma_centre*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*-2.0+gamma_centre*tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0+gamma_centre*tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)+(I*tau_h*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))+(I*tau_h*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(I*(tau_h*tau_h)*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(beta_centre*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(gamma_centre*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)))-(beta_centre*tau*tau_h*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(-gamma_centre*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+gamma_centre*tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+gamma_centre*tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))+(gs*tau_h*u0*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))-1.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(beta_centre*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(gamma_centre*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)))-alpha*gs*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0+tau*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(t*tau+t*tau_h-tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0)+(alpha*gs*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0+tau*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(t*tau+t*tau_h-tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*t*tau*tau_h*2.0)*2.0)*(1.0/2.0))/tau+alpha*gs*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0-tau*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(-t*tau-t*tau_h+tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0)+(alpha*gs*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0-tau*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(-t*tau-t*tau_h+tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+alpha*t*tau*tau_h*2.0)*2.0)*(1.0/2.0))/tau+(I*tau*tau_h*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(I*(tau_h*tau_h)*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(I*tau*tau_h*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(gs*(tau_h*tau_h)*u0*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(gs*tau_h*u0*exp(-alpha*t)*(exp(alpha*t)-exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(beta_centre*(tau*tau)*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(gamma_centre*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)))-(beta_centre*(tau_h*tau_h)*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(gamma_centre*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)))+(alpha*gs*tau_h*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0+tau*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(t*tau+t*tau_h-tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/tau-(alpha*gs*tau_h*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0-tau*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(-t*tau-t*tau_h+tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/tau+(gs*tau*tau_h*u0*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(gs*(tau_h*tau_h)*u0*exp(-alpha*t)*(exp(alpha*t)-exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h)))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)+(beta_centre*tau*tau_h*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(gamma_centre*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)))+(gs*tau*tau_h*u0*exp(-alpha*t)*(exp(alpha*t)-exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h)))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0))*(1.0/2.0)+(1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*((tau*tau)*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))-(tau*tau)*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))+(tau_h*tau_h)*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))-(tau_h*tau_h)*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))-exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau_h*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*2.0+tau*tau_h*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*2.0)*(beta_centre*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(-1.0/2.0)+(beta_centre*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*(1.0/2.0))/tau_h+(beta_centre*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*(1.0/2.0))/tau_h+(beta_centre*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+I*gamma_centre*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-gamma_centre*gs*u0*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-(I*gamma_centre*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)))/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(beta_centre*tau*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/tau_h+alpha*gamma_centre*gs*y0*((exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*(t*tau+t*tau_h-tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*t*tau*tau_h*2.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)+(tau*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*2.0)/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-(beta_centre*tau*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau_h*(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0))+alpha*gamma_centre*gs*y0*((exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*(-t*tau-t*tau_h+tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+alpha*t*tau*tau_h*2.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(tau*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*2.0)/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-(gamma_centre*gs*tau*tau_h*u0*exp(-alpha*t)*(alpha*exp(alpha*t)-(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)+(alpha*gamma_centre*gs*tau*tau_h*u0*exp(-alpha*t)*(exp(alpha*t)-exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h)))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0))*(1.0/4.0))/(gamma_centre*tau)+(1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))+(tau*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/tau_h+(tau_h*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/tau-(tau*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/tau_h-(tau_h*exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/tau+(exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)-t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h)-(exp((t*tau*(-1.0/2.0)-t*tau_h*(1.0/2.0)+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*(1.0/2.0))/(tau*tau_h))*(n0+(beta_centre*tau*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))+(beta_centre*tau*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(beta_centre*(tau*tau)*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(beta_centre*tau*tau_h*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0))/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(beta_centre*(tau*tau)*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(beta_centre*tau*tau_h*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))+alpha*gamma_centre*gs*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0+tau*tau_h*exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(t*tau+t*tau_h-tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*gamma_centre*gs*y0*((tau*tau)*(tau_h*tau_h)*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*4.0-tau*tau_h*exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))*1.0/pow(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0,2.0)*(-t*tau-t*tau_h+tau*tau_h*2.0+t*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+alpha*t*tau*tau_h*2.0)*2.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-(I*gamma_centre*tau*tau_h*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*2.0)/(tau*tau_h*2.0+tau*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)+tau_h*sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-tau*tau-tau_h*tau_h-gamma_centre*gh*tau*tau_h*4.0)+(I*gamma_centre*tau*tau_h*(exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))-(gamma_centre*gs*tau*tau_h*u0*(exp((t*(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)*(1.0/2.0))/(tau*tau_h))-1.0)*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)/(tau+tau_h-sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0)-(gamma_centre*gs*tau*tau_h*u0*exp(-alpha*t)*(exp(alpha*t)-exp((t*(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0))*(1.0/2.0))/(tau*tau_h)))*1.0/sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)*2.0)/(tau+tau_h+sqrt(tau*tau_h*-2.0+tau*tau+tau_h*tau_h+gamma_centre*gh*tau*tau_h*4.0)-alpha*tau*tau_h*2.0))*(1.0/4.0))/(gamma_centre*tau);
}

__device__ float dfun3( float t, float v0, float n0, float u0, float y0, float I)
{
  return
    (beta_right*gh*exp(-t/tau))/(tau-tau_h)-(v0*exp(-t/tau))/tau-(I*exp(-t/tau))/tau-(beta_right*gh*exp(-t/tau_h))/(tau-tau_h)+(gh*n0*exp(-t/tau_h))/(tau-tau_h)-(gs*exp(-alpha*t)*(-tau*y0*alpha*alpha+y0*alpha))/powf(alpha*tau-1.0f,2.0f)+(gs*exp(-t/tau)*(alpha*tau*u0-u0+alpha*tau*y0))/(tau*powf(alpha*tau-1.0f,2.0f))-(alpha*gs*exp(-alpha*t)*(alpha*tau*u0-u0-alpha*t*y0+alpha*tau*y0+alpha*alpha*t*tau*y0))/powf(alpha*tau-1.0f,2.0f)-(gh*n0*tau_h*exp(-t/tau))/(tau*(tau-tau_h));
}


__global__ void ApplyResetKernel( float4* pGlobalState,
                                  unsigned int* pGlobalZone,
                                  unsigned int firingIndex,
                                  float* pRefractTime,
                                  float dx,
                                  int spatialExtent)
{
  int local_index = threadIdx.x+blockDim.x*blockIdx.x+firingIndex-spatialExtent;
  if ((local_index>0) && (local_index<=2*spatialExtent))
  {
    local_index -= spatialExtent;
    pGlobalState[local_index].w += alpha*W(local_index*dx)*dx;
  }
  if (threadIdx.x==0)
  {
    pGlobalState[firingIndex].x = V_r;
    pGlobalZone [firingIndex] = 2;
    pRefractTime[firingIndex] = tau_r;
  }
}

__host__ __device__ float W( const float x)
{
  return w0/2*(tanh(beta*(sigma-x))+tanh(beta*(sigma+x)));
}

__device__ float FindSpikeTime( const float4 state, const float local_I)
{
  float spikeTime = 0.0f;
  float f, df;
  f  = fun3( spikeTime, state.x, state.y, state.z, state.w, local_I, V_th);
  df = dfun3( spikeTime, state.x, state.y, state.z, state.w, local_I);

  while (fabs(f)>tol) {
    spikeTime -= f/df;
    f  = fun3( spikeTime, state.x, state.y, state.z, state.w, local_I, V_th);
    df = dfun3( spikeTime, state.x, state.y, state.z, state.w, local_I);
  }

  return spikeTime;
}

__global__ void FindSpikeTimeKernel( const float4* pGlobalState,
                                     const unsigned int* pGlobalZone,
                                     const float stepTime,
                                     EventDrivenMap::firing* pFiringVal,
                                     unsigned int* pEventNo)
{
  unsigned int index = threadIdx.x+blockDim.x*blockIdx.x;
  bool correct_zone  = (pGlobalZone[index] == 3);
  if (correct_zone)
  {
    float local_I =
      (-I_steve/gl_steve)+I_app*(index>=I_app_first)*(index<I_app_last);
    float4 local_state = pGlobalState[index];
    float local_v = fun3( stepTime, local_state.x, local_state.y, local_state.z,
        local_state.w, local_I, 0.0f);
    float spikeTime = stepTime;
    unsigned int storage_id;
    if (local_v>V_th)
    {
      spikeTime = FindSpikeTime( local_state, local_I);
      storage_id = atomicAdd( pEventNo, 1);

      // Store values
      pFiringVal[storage_id].time  = spikeTime;
      pFiringVal[storage_id].index = index;
    }
  }
}

void EventDrivenMap::FindMinimumSpikeTime( float timestep)
{
  FindSpikeTimeKernel<<<mNoBlocks,mNoThreads>>>( mpGlobalState, mpGlobalZone,
                       timestep, mpFiringVal, mpEventNo);
  CUDA_CHECK_ERROR();
  if (mPrintOutput)
    printf("Spike times found.\n");

  // Find minimum spike time
  CUDA_CALL( cudaMemcpy( mpHost_eventNo, mpEventNo, sizeof(int), cudaMemcpyDeviceToHost));
  if (mPrintOutput)
    printf("No of spiking cells = %d\n",*mpHost_eventNo);

  if (*mpHost_eventNo>0)
  {
    deviceReduceMinKernel<<<mNoBlocks,mNoThreads>>>
      ( mpFiringVal, *mpEventNo, mpFiringValTemp);
    CUDA_CHECK_ERROR();
    deviceReduceMinKernel<<<1,mNoThreads>>>
      ( mpFiringValTemp, mNoBlocks, mpFiringValTemp);
    CUDA_CHECK_ERROR();
  }
}

__inline__ __device__ EventDrivenMap::firing warpReduceMin( EventDrivenMap::firing val)
{
  float dummyTime;
  unsigned int dummyIndex;
  for (int offset = warpSize/2; offset>0; offset/=2) {
    dummyTime  = __shfl_down( val.time, offset);
    dummyIndex = __shfl_down( val.index, offset);
    if (dummyTime<val.time)
    {
      val.time  = dummyTime;
      val.index = dummyIndex;
    }
  }
  return val;
}

__inline__ __device__ struct EventDrivenMap::firing blockReduceMin( EventDrivenMap::firing val)
{
  __shared__ EventDrivenMap::firing shared[32];
  int lane = threadIdx.x % warpSize;
  int wid  = threadIdx.x / warpSize;

  val = warpReduceMin( val);

  if (lane==0) {
    shared[wid] = val;
  }
  __syncthreads();

  val = (threadIdx.x<blockDim.x/warpSize) ? shared[lane] : (EventDrivenMap::firing){100.0f,0};

  if (wid==0) {
    val = warpReduceMin( val);
  }

  return val;
}

__global__ void deviceReduceMinKernel( const EventDrivenMap::firing* in,
                                       const unsigned int npts,
                                       EventDrivenMap::firing* out)
{
  float time = 1000000.0f;
  struct EventDrivenMap::firing dummy;
  struct EventDrivenMap::firing val;
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
