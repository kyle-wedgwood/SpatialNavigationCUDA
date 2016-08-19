#ifndef EVENTDRIVENMAPHEADERDEF
#define EVENTDRIVENMAPHEADERDEF
/* Code to do one dimensional spiking model from Mayte's note */
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <arrayfire.h>
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
        printOutput = 0;
      }

      unsigned int networkSize;
      unsigned int noThreads;
      float domainSize;
      float timestep;
      bool printOutput;
    };

    EventDrivenMap( const ParameterList* pParameterList);

    ~EventDrivenMap();

    void InitialiseNetwork();

    void SimulateNetwork( const float finalTime);

    void SimulateStep();

    void SetXAxis( af::array* mpPlotVarX);

    void PlotData();

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
    unsigned int mSpatialExtent;
    bool mPrintOutput;

    float4* mpGlobalState;
    float* mpRefractTime;
    unsigned int* mpGlobalZone;
    firing* mpFiringVal;
    firing* mpFiringValTemp;
    firing* mpHost_firingVal;

    unsigned int* mpEventNo;
    unsigned int* mpHost_eventNo;

    // For plotting
    af::Window* mpWindow;
    af::array* mpPlotVarY;
    af::array* mpPlotVarX;
    float* mpPlotVarYHelper;

    float mTime;
    float mDt;

    void FindMinimumSpikeTime( float timestep);
    void CalculateSpatialExtent();
    void SetPlottingWindow();
};

// Initialise network
__global__ void InitialiseNetworkKernel( float4* pGlobalState, unsigned int* pGlobalZone, unsigned int networkSize);

// Functions to find minimum spike time
__device__ float FindSpikeTime( const float4 state);
__global__ void FindSpikeTimeKernel( const float4* pGlobalState,
                                     const unsigned int* pGlobalZone,
                                     const float stepTime,
                                     EventDrivenMap::firing* pFiringVal,
                                     unsigned int* pEventNo);

__global__ void deviceReduceMinKernel( const EventDrivenMap::firing* in,
                                       const unsigned int npts,
                                       EventDrivenMap::firing* out);

__inline__ __device__ EventDrivenMap::firing warpReduceMin( EventDrivenMap::firing val);
__inline__ __device__ EventDrivenMap::firing blockReduceMin( EventDrivenMap::firing val);

// Main kernel to update state
__global__ void UpdateStateKernel( const float eventTime, float4* pGlobalState, unsigned int* pGlobalZone, float* pRefractTime, const unsigned int networkSize);

// Zone 1 functions
__global__ void UpdateZone1Kernel( const float eventTime, float4* pGlobalState, unsigned int* pGlobalZone);
__device__ float4 UpdateZone1( float eventTime, float4 state, short* pChangeFlag);
__device__ float4 UpdateStateZone1( float eventTime, float4 state);
__device__ float fun1( float t, float v0, float n0, float u0, float y0, float thresh);
__device__ float dfun1( float t, float v0, float n0, float u0, float y0);

// Zone 2 functions
__global__ void UpdateZone2Kernel( const float eventTime, float4* pGlobalState, unsigned int* pGlobalZone);
__device__ float4 UpdateZone2( float eventTime, float4 state, short* pChangeFlag);
__device__ float4 UpdateStateZone2( float eventTime, float4 state);
__device__ float fun2( float t, float v0, float n0, float u0, float y0, float thresh);
__device__ float dfun2( float t, float v0, float n0, float u0, float y0);

// Zone 3 functions
__global__ void UpdateZone3Kernel( const float eventTime, float4* pGlobalState, unsigned int* pGlobalZone);
__device__ float4 UpdateZone3( float eventTime, float4 state, short* pChangeFlag);
__device__ float4 UpdateStateZone3( float eventTime, float4 state);
__device__ float fun3( float t, float v0, float n0, float u0, float y0, float thresh);
__device__ float dfun3( float t, float v0, float n0, float u0, float y0);

// Zone 4 functions
__global__ void UpdateZone4Kernel( const float eventTime, float4* pGlobalState, unsigned int* pGlobalZone, float* pRefractTime);
__device__ float4 UpdateZone4( const float eventTime, float4 state, float* pRefractTime, short* pChangeFlag);
__device__ float4 UpdateStateZone4( float t, float4 state);

// Update synaptic input to cells
__global__ void ApplyResetKernel( float4* pGlobalState, unsigned int* pGlobalZone, unsigned int index, float* pRefractTime, float dx, unsigned int spatial_extent);

// Clear memory
__global__ void ResetMemoryKernel( EventDrivenMap::firing* pFiringVal, const unsigned int resetSize, EventDrivenMap::firing* pFiringValTemp, const unsigned int resetSizeTemp, const float stepSize);

// Transfer data for plotting
__global__ void CopyDataToPlotBufferKernel( float* pPlotVarY, const float4* pGlobalState, const unsigned int networkSize);

#endif
