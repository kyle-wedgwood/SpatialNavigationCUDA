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

    float mTime;
    float mDt;

    unsigned int* mpEventNo;

    void FindMinimumSpikeTime();

};

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

#endif
