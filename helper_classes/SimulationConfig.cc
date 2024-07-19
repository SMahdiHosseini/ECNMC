//
// Created by mahdi on 19.08.24.
//


#include "SimulationConfig.h"

using namespace ns3;

SimulationConfig::SimulationConfig() : m_myVariable (5.0) {} // Default value
double SimulationConfig::GetMyVariable() const { return m_myVariable; }
