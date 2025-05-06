//
// Created by Mahdi on 05.05.25.
//

#include "ns3/applications-module.h"
#include "DCWorkloadGenerator.h"

uint32_t DCWorkloadGenerator::SOCKET_COUNT = 0;

DCWorkloadGenerator::DCWorkloadGenerator(const Ptr<Node>& sender, const vector<Ptr<Node>>& receivers, double avgRate, uint32_t poolSize, const string workloadPath, const string protocol, Time trafficStartTime, Time trafficEndTime) :
        _sender(sender), _receivers(receivers), _avgRate(avgRate), _poolSize(poolSize), _workloadPath(workloadPath), protocol(protocol), trafficStartTime(trafficStartTime), trafficEndTime(trafficEndTime) {}

vector<Address> 
DCWorkloadGenerator::establishPairConnections(uint32_t receiverId) {
    Ptr<Node> receiver = _receivers[receiverId];
    vector<Address> receiverAddresses;
    for (uint32_t i = 0; i < _poolSize; i++) {
        uint32_t connectionId = ++SOCKET_COUNT;
        InetSocketAddress receiverAddress = InetSocketAddress(GetNodeIP(receiver, 1), 4000 + connectionId);
        receiverAddresses.push_back(receiverAddress.ConvertTo());

        // create sink at receiver
        TraceReplayReceiverHelper replayHelperServer(receiverAddress);
        replayHelperServer.SetAttribute("Protocol", StringValue(protocol));
        ApplicationContainer replayAppServer = replayHelperServer.Install(receiver);
        replayAppServer.Start(Simulator::Now());
    }
    return receiverAddresses;
}


void 
DCWorkloadGenerator::GenrateTraffic() {
    for (uint32_t i = 0; i < _receivers.size(); i++) {
        receiversAddresses.push_back(establishPairConnections(i));
    }

    ObjectFactory factory;
    factory.SetTypeId(WorkloadApp::GetTypeId());
    factory.Set("StartTime", TimeValue(trafficStartTime));
    factory.Set("StopTime", TimeValue(trafficEndTime));
    factory.Set("Protocol", StringValue(protocol));
    factory.Set("Rate", DoubleValue(_avgRate));
    factory.Set("WorkloadPath", StringValue(_workloadPath));
    Ptr<WorkloadApp> nodeAppsHandler = factory.Create<WorkloadApp>();
    nodeAppsHandler->SetReceiverAddress(receiversAddresses);
    _sender->AddApplication(nodeAppsHandler);
}
