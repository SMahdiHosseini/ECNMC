//
// Created by nal on 30.06.25.
//

#ifndef PROBE_GENERATOR_H
#define PROBE_GENERATOR_H

#include "ns3/core-module.h"
#include "ns3/internet-module.h"

#include "../../helper_classes/HelperMethods.h"
#include "../background_replay/TraceReplayReceiverHelper.h"
#include "ProbeApp.h"

using namespace ns3;
using namespace std;
using namespace helper_methods;
 
class ProbeGenerator {

private:
    Ptr<Node> _sender;
    Ptr<Node> _receiver;
    double _avgRate;
    Time trafficStartTime;
    Time trafficEndTime;

    Address receiverAddresses;
    Address establishPairConnections();
public:
    ProbeGenerator(const Ptr<Node>& sender, const Ptr<Node>& receiver, double avgRate, Time trafficStartTime, Time trafficEndTime);

    void GenrateTraffic();
};


#endif //PROBE_GENERATOR_H
