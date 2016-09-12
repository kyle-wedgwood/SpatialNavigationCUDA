#include <iostream>
#include <cstdlib>
#include <cmath>
#include "EventDrivenMap.hpp"
#include "parameters.h"

using namespace std;

int main( int argc, char* argv[])
{
  EventDrivenMap::ParameterList pars;
  pars.networkSize = 1000;
  pars.noThreads   = 1024;
  pars.domainSize  = 10.0*sigma;
  pars.timestep    = 0.1;
  pars.plotFreq    = 100;
  pars.printOutput = 1;

  EventDrivenMap* p_event = new EventDrivenMap( &pars);

  float simulation_time = 1000.0f;
  bool extend_simulation = true;

  // First simulate to settle at steady state
  p_event->SimulateNetwork( simulation_time);
  cout<< "Transient simulation finished" << endl;

  // Now hyperpolarise subset of cells
  cout<< "Hyperpolarising some cells" << endl;
  p_event->SetAppliedCurrent( -30.0f);

  simulation_time += 250.0f;
  p_event->SimulateNetwork( simulation_time, extend_simulation);

  // Remove applied current
  cout<< "Removing hyperpolarising current" << endl;
  pars.plotFreq = 10;
  p_event->SetParameters( &pars);
  p_event->SetAppliedCurrent( 0.0f);
  simulation_time += 100000.f;
  p_event->SimulateNetwork( simulation_time, extend_simulation);

  delete(p_event);

  return 0;
}
