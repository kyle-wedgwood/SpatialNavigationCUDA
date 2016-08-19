#include <iostream>
#include <cstdlib>
#include <cmath>
#include "EventDrivenMap.hpp"

int main( int argc, char* argv[])
{
  EventDrivenMap::ParameterList pars;
  pars.networkSize = 1024;
  pars.noThreads   = 1024;
  pars.domainSize  = 120.0;
  pars.timestep    = 0.1;

  EventDrivenMap* p_event = new EventDrivenMap( &pars);

  float simulation_time = 1000.0f;
  //p_event->InitialiseNetwork();
  //p_event->SimulateStep();
  p_event->SimulateNetwork( simulation_time);

  delete(p_event);

  return 0;
}
