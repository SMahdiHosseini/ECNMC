//
// Created by Mahdi on 30.06.25.
//

#include "ns3/applications-module.h"
#include "ProbeGenerator.h"

ProbeGenerator::ProbeGenerator(const Ptr<Node>& sender, const Ptr<Node>& receiver, double avgRate, Time trafficStartTime, Time trafficEndTime) :
        _sender(sender), _receiver(receiver), _avgRate(avgRate), trafficStartTime(trafficStartTime), trafficEndTime(trafficEndTime) {}

Address
ProbeGenerator::establishPairConnections() {

    InetSocketAddress receiverAddress = InetSocketAddress(GetNodeIP(_receiver, 1), 9000);

    // create sink at receiver
    TraceReplayReceiverHelper replayHelperServer(receiverAddress);
    replayHelperServer.SetAttribute("Protocol", StringValue("ns3::TcpSocketFactory"));
    ApplicationContainer replayAppServer = replayHelperServer.Install(_receiver);
    replayAppServer.Start(Simulator::Now());

    return receiverAddress.ConvertTo();
}


void 
ProbeGenerator::GenrateTraffic() {
    receiverAddresses = establishPairConnections();
    // cout << "connection at the serverside established with address: " << InetSocketAddress::ConvertFrom(receiverAddresses).GetIpv4() << " port: " << InetSocketAddress::ConvertFrom(receiverAddresses).GetPort() << endl;
    ObjectFactory factory;
    factory.SetTypeId(ProbeApp::GetTypeId());
    factory.Set("StartTime", TimeValue(trafficStartTime - Seconds(0.001))); // Start a bit earlier to ensure the connection is established
    factory.Set("StopTime", TimeValue(trafficEndTime));
    factory.Set("Protocol", StringValue("ns3::TcpSocketFactory"));
    factory.Set("Rate", DoubleValue(_avgRate));
    Ptr<ProbeApp> nodeAppsHandler = factory.Create<ProbeApp>();
    nodeAppsHandler->SetReceiverAddress(receiverAddresses);
    _sender->AddApplication(nodeAppsHandler);
}
