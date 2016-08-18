#include <iostream>
#include <cstdlib>
#include <cmath>
#include "EventDrivenMap2D.hpp"

int main( int argc, char* argv[])
{
  EventDrivenMap::ParameterList pars;
  pars.networkSizeX = 1024;
  pars.networkSizeY = 1024;
  pars.noThreads   = 1024;
  pars.domainSize  = 120.0;
  pars.timestep    = 0.1;

  EventDrivenMap* p_event = new EventDrivenMap( &pars);

  float simulation_time = 100.0f;
  p_event->InitialiseNetwork();
  p_event->SimulateStep();
  p_event->SimulateNetwork( simulation_time);

  delete(p_event);

  return 0;
}
