//
// Created by nal on 05.05.25.
//

#ifndef DC_WORKLOAD_GEN_H
#define DC_WORKLOAD_GEN_H

#include "ns3/core-module.h"
#include "ns3/internet-module.h"

#include "../../helper_classes/HelperMethods.h"
#include "../background_replay/TraceReplayReceiverHelper.h"
#include "WorkloadApp.h"

using namespace ns3;
using namespace std;
using namespace helper_methods;
 
class DCWorkloadGenerator {

private:
    static uint32_t SOCKET_COUNT;
    Ptr<Node> _sender;
    vector<Ptr<Node>> _receivers;
    double _avgRate;
    uint32_t _poolSize;
    string _workloadPath;
    string protocol;
    Time trafficStartTime;
    Time trafficEndTime;

    vector<vector<Address>> receiversAddresses;
    vector<Address> establishPairConnections(uint32_t receiverId);
public:
    DCWorkloadGenerator(const Ptr<Node>& sender, const vector<Ptr<Node>>& receivers, double avgRate, uint32_t poolSize, const string workloadPath, const string protocol, Time trafficStartTime, Time trafficEndTime);

    void GenrateTraffic();
};


#endif //DC_WORKLOAD_GEN_H
