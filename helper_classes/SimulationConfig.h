//
// Created by mahdi on 19.08.24.
//

#ifndef SIMULATIONCONFIG_H
#define SIMULATIONCONFIG_H


#include "ns3/core-module.h"

using namespace ns3;

// class SimulationConfig : public Object
class SimulationConfig
{
public:
//   static TypeId GetTypeId (void)
//   {
//     static TypeId tid = TypeId ("ns3::SimulationConfig")
//       .SetParent<Object> ()
//       .AddConstructor<SimulationConfig> ()
//       .AddAttribute ("MyVariable",
//                      "Description of my variable",
//                      DoubleValue (5.0), // Default value
//                      MakeDoubleAccessor (&SimulationConfig::m_myVariable),
//                      MakeDoubleChecker<double> ());
//     return tid;
//   }

  SimulationConfig ();

  double GetMyVariable () const;

private:
  double m_myVariable;
};

#endif // SIMULATIONCONFIG_H