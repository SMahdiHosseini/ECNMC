//
// Created by nal on 18.01.22.
//

#ifndef WEHE_P_TOMOGRAPHY_HELPERMETHODS_H
#define WEHE_P_TOMOGRAPHY_HELPERMETHODS_H

#include <string>
#include <chrono>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <regex>
#include <random>

#include "ns3/core-module.h"
#include "ns3/internet-module.h"

using namespace std;

namespace helper_methods {

    extern vector<string> SplitStr (const string &s, char delim);
    template<class T> extern string VectorToString(vector<T> vector, string separator);
    extern uint32_t GetRandomNumber(uint32_t min, uint32_t max);
    extern uint32_t GetSubDirCount(const string& dirPath);
    extern string ComputeQueueSize(const string& linkRate, const vector<string>& linksDelay);
    extern ns3::Ipv4Address GetNodeIP(ns3::Ptr<ns3::Node> node, uint32_t interface);
    extern bool DoesPolicerLocationMatch(const string& linkLocation, const string& expectedPolicerLocation);
    extern bool IsPolicerTypePerFlowPolicer(int policerType);
    extern uint8_t Dscp2Tos(uint8_t dscp);
    extern string GetSocketFactory(bool isTCP);

}

#endif //WEHE_P_TOMOGRAPHY_HELPERMETHODS_H
