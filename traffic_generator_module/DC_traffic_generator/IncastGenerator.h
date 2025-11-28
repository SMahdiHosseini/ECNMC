//
// Created by nal on 26.11.25.
//

#ifndef INCAST_GENERATOR_H
#define INCAST_GENERATOR_H

#include "ns3/core-module.h"
#include "ns3/internet-module.h"

#include "../../helper_classes/HelperMethods.h"
#include "../background_replay/IncastReceiverHelper.h"
#include "../background_replay/TraceReplayReceiverHelper.h"

using namespace ns3;
using namespace std;
using namespace helper_methods;
 
class IncastGenerator {

private:

    vector<Ptr<Node>> nodes;
    static uint32_t SOCKET_COUNT;
    int incastFactor;
    uint32_t messageSize;
    Time incastperiod;
    Time trafficStartTime;
    Time trafficEndTime;
    Ptr<UniformRandomVariable> m_uv;
    void GenerateTraffic();
    void ConnectionSucceeded(Ptr<Socket> socket);
    void ConnectionClosed(Ptr<Socket> socket);
    void Recv(Ptr<Socket> socket);
public:
    IncastGenerator(const vector<NodeContainer>& nodes, int incastFactor, uint32_t messageSize, Time incastperiod, Time trafficStartTime, Time trafficEndTime);
    void Start();
};


#endif //INCAST_GENERATOR_H
