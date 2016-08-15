#ifndef EVENTDRIVENMAPHEADERDEF
#define EVENTDRIVENMAPHEADERDEF
/* Code to do one dimensional spiking model from Mayte's note */
#include <iostream>
#include <cstdlib>
#include <cmath>
#include "parameters.h"
#include "vector_types.h"

class EventDrivenMap
{

  public:

    struct ParameterList
    {
      ParameterList()
      {
        networkSize = 1000;
        noThreads   = 1024;
        domainSize  = 120.0;
        timestep    = 0.1;
      }

      unsigned int networkSize;
      unsigned int noThreads;
      float domainSize;
      float timestep;
    };

    EventDrivenMap( const ParameterList* pParameterList);

    ~EventDrivenMap();

    void InitialiseNetwork();

    void SimulateNetwork( const float finalTime);

    void SimulateStep();

    struct __align__(8) firing
    {
      float time;
      unsigned int index;
    };

  private:

    // Hiding default constructor
    EventDrivenMap();

    unsigned int mNetworkSize;
    unsigned int mNoThreads;
    unsigned int mNoBlocks;
    float mDomainSize;
    float mDx;

    float4* mpGlobalState;
    float* mpRefractTime;
    short* mpGlobalZone;
    firing* mpFiringVal;
    firing* mpFiringValTemp;
    firing* mpHost_firingVal;

    unsigned int* mpEventNo;
    unsigned int* mpHost_eventNo;

    float mTime;
    float mDt;

    unsigned int* mpEventNo;

    void FindMinimumSpikeTime( float timestep);

};

// Initialise network
__global__ void InitialiseNetworkKernel( float4* pGlobalState, unsigned int* pGlobalZone);

// Functions to find minimum spike time
__device__ float FindSpikeTime( const float4 state);
__global__ void FindSpikeTimeKernel( const float4* pGlobalState,
                                     const short* pGlobalZone,
                                     const float stepTime,
                                     EventDrivenMap::firing* pFiringVal,
                                     unsigned int* pEventNo);

__global__ void deviceReduceMinKernel( const EventDrivenMap::firing* in,
                                       const unsigned int npts,
                                       EventDrivenMap::firing* out);

__inline__ __device__ EventDrivenMap::firing warpReduceMin( EventDrivenMap::firing val);
__inline__ __device__ EventDrivenMap::firing blockReduceMin( EventDrivenMap::firing val);

// Zone 1 functions
__global__ void UpdateZone1Kernel( const float eventTime, float4* pGlobalState, unsigned int* pGlobalZone);
__device__ float4 UpdateZone1( float eventTime, float4 state);
__device__ float4 UpdateStateZone1( float eventTime, float4 state);
__device__ float fun1( float t, float v0, float n0, float u0, float y0);
__device__ float dfun1( float t, float v0, float n0, float u0, float y0);

// Zone 2 functions
__global__ void UpdateZone2Kernel( const float eventTime, float4* pGlobalState, unsigned int* pGlobalZone);
__device__ float4 UpdateZone2( float eventTime, float4 state);
__device__ float4 UpdateStateZone2( float eventTime, float4 state);
__device__ float fun2( float t, float v0, float n0, float u0, float y0, float thresh);
__device__ float dfun2( float t, float v0, float n0, float u0, float y0);

// Zone 3 functions
__global__ void UpdateZone3Kernel( const float eventTime, float4* pGlobalState, unsigned int* pGlobalZone);
__device__ float4 UpdateZone3( float eventTime, float4 state);
__device__ float4 UpdateStateZone3( float eventTime, float4 state);
__device__ float fun3( float t, float v0, float n0, float u0, float y0, float thresh);
__device__ float dfun3( float t, float v0, float n0, float u0, float y0);

// Zone 4 functions
__global__ void UpdateZone4Kernel( const float eventTime, float4* pGlobalState, unsigned int* pGlobalZone, float* pRefractTime);
__device__ float4 UpdateStateZone4( float t, float4 state);

// Update synaptic input to cells
__global__ void ApplyResetKernel( float4* pGlobalState, unsigned int* pGlobalZone, firing* pFiringVal, float* pRefractTime);

// Clear memory
__global__ void ResetMemoryKernel( firing pFiringVal, const unsigned int networkSize, const float stepTime);

#endif
