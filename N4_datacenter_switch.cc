// The nework Topology is as follows:
//  
//  s1                s3
//  |                  |
//  T1 -----(10Gbps)-- T2 -(1Gbps)-R2
//  |                  |    
//  s2                R1

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/traffic-control-module.h"
#include "ns3/flow-monitor-module.h"
#include "monitors_module/PacketMonitor.h"
#include "monitors_module/SwitchMonitor.h"
#include "monitors_module/PoissonSampler.h"
#include "traffic_generator_module/background_replay/BackgroundReplay.h"
#include <iomanip>
#include <iostream>
#include <string>

using namespace ns3;
using namespace std;

void queueDiscSize(uint32_t oldValue, uint32_t newValue) {
    std::cout << Simulator::Now().GetNanoSeconds() << ": Queue Disc Size: " << newValue << endl;
}

void queueSize(uint32_t oldValue, uint32_t newValue) {
    std::cout << Simulator::Now().GetNanoSeconds() << ": Queue Size Measure: " << newValue << endl;
}

void queueSize2(uint32_t oldValue, uint32_t newValue) {
    std::cout << Simulator::Now().GetNanoSeconds() << ": Queue Size Cross: " << newValue << endl;
}

void dequeue(Ptr< const Packet > packet){
    std::cout << "Packet dequeued: ";
    packet->Print(std::cout);
    std::cout << endl;
}

void enqueue(Ptr< const Packet > packet){
    std::cout << "Packet enqueued: ";
    packet->Print(std::cout);
    std::cout << endl;
}

void enqueueDisc(Ptr< const QueueDiscItem > item){
    std::cout << Simulator::Now().GetNanoSeconds() << ": Packet enqueued Disc: ";
    item->Print(std::cout);
    item->GetPacket()->Print(std::cout);
    std::cout << endl;
}

int main(int argc, char* argv[])
{
    auto start = std::chrono::high_resolution_clock::now();
    cout << endl<< "Start" << endl;
    /* ########## START: Config ########## */
    string hostToTorLinkRate = "53Mbps";               // Links bandwith between hosts and ToR switches
    string hostToTorLinkRateCrossTraffic = "53Mbps";   // Links bandwith between hosts and ToR switches for the cross traffic
    string hostToTorLinkDelay = "10us";                // Links delay between hosts and ToR switches
    string torToAggLinkRate = "10Mbps";                // Links bandwith between ToR and Agg switches
    string torToAggLinkDelay = "10us";                 // Links delay between ToR and Agg switches
    string aggToCoreLinkRate = "10Mbps";               // Links bandwith between Agg and Core switches
    string aggToCoreLinkDelay = "10us";                // Links delay between Agg and Core switches
    string appDataRate = "10Mbps";                     // Application data rate
    string duration = "10";                            // Duration of the simulation
    double pctPacedBack = 0.8;                         // the percentage of tcp flows of the CAIDA trace to be paced
    bool enableSwitchECN = true;                       // Enable ECN on the switches
    bool enableECMP = false;                           // Enable ECMP on the switches
    double sampleRate = 10;                            // Sample rate for the PoissonSampler

    /*command line input*/
    CommandLine cmd;
    cmd.AddValue("hostToTorLinkRate", "Links bandwith between hosts and ToR switches", hostToTorLinkRate);
    cmd.AddValue("hostToTorLinkDelay", "Links delay between hosts and ToR switches", hostToTorLinkDelay);
    cmd.AddValue("torToAggLinkRate", "Links bandwith between ToR and Agg switches", torToAggLinkRate);
    cmd.AddValue("torToAggLinkDelay", "Links delay between ToR and Agg switches", torToAggLinkDelay);
    cmd.AddValue("aggToCoreLinkRate", "Links bandwith between Agg and Core switches", aggToCoreLinkRate);
    cmd.AddValue("aggToCoreLinkDelay", "Links delay between Agg and Core switches", aggToCoreLinkDelay);
    cmd.AddValue("appDataRate", "Application data rate", appDataRate);
    cmd.AddValue("enableSwichECN", "Enable ECN on the switches", enableSwitchECN);
    cmd.AddValue("enableECMP", "Enable ECMP on the switches", enableECMP);
    cmd.AddValue("duration", "Duration of the simulation", duration);
    cmd.AddValue("pctPacedBack", "the percentage of tcp flows of the CAIDA trace to be paced", pctPacedBack);
    cmd.AddValue("sampleRate", "Sample rate for the PoissonSampler", sampleRate);
    cmd.AddValue("hostToTorLinkRateCrossTraffic", "Links bandwith between hosts and ToR switches for the cross traffic", hostToTorLinkRateCrossTraffic);
    cmd.Parse(argc, argv);

    /*set default values*/
    Time startTime = Seconds(0);
    Time stopTime = Seconds(stof(duration));
    Time convergenceTime = Seconds(0.005);

    Config::SetDefault("ns3::TcpL4Protocol::SocketType", StringValue("ns3::TcpDctcp"));
    Config::SetDefault("ns3::Ipv4GlobalRouting::RandomEcmpRouting", BooleanValue(enableECMP));
    Config::SetDefault("ns3::RedQueueDisc::UseEcn", BooleanValue(enableSwitchECN));
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(1448));
    Config::SetDefault("ns3::TcpSocket::DelAckCount", UintegerValue(2));
    GlobalValue::Bind("ChecksumEnabled", BooleanValue(false));
    Config::SetDefault("ns3::RedQueueDisc::UseHardDrop", BooleanValue(false));
    Config::SetDefault("ns3::RedQueueDisc::MeanPktSize", UintegerValue(1500));
    // Triumph and Scorpion switches used in DCTCP Paper have 4 MB of buffer
    // If every packet is 1500 bytes, 2666 packets can be stored in 4 MB
    Config::SetDefault("ns3::RedQueueDisc::MaxSize", QueueSizeValue(QueueSize("2666p")));
    Config::SetDefault("ns3::RedQueueDisc::QW", DoubleValue(1));
    Config::SetDefault("ns3::RedQueueDisc::MinTh", DoubleValue(20));
    Config::SetDefault("ns3::RedQueueDisc::MaxTh", DoubleValue(60));
    /* ########## END: Config ########## */



    /* ########## START: Ceating the topology ########## */
    int nHosts = 4;
    int nRacks = 2;
    vector<NodeContainer> racks;
    racks.reserve(nRacks);
    NodeContainer torSwitches;
    // NodeContainer aggSwitches;
    // NodeContainer coreSwitches;

    // Create the racks and switches containers
    for (int i = 0; i < nRacks; i++) {
        NodeContainer rack;
        rack.Create(nHosts);
        racks.push_back(rack);
    }
    torSwitches.Create(nRacks);

    // connecting the hosts to the ToR switches
    vector<vector<NetDeviceContainer>> hostsToTorsNetDevices;

    PointToPointHelper p2pHostToTorMeasurementTraffic;
    p2pHostToTorMeasurementTraffic.SetDeviceAttribute("DataRate", StringValue(hostToTorLinkRate));
    p2pHostToTorMeasurementTraffic.SetChannelAttribute("Delay", StringValue(hostToTorLinkDelay));

    PointToPointHelper p2pHostToTorCrossTraffic;
    p2pHostToTorCrossTraffic.SetDeviceAttribute("DataRate", StringValue(hostToTorLinkRateCrossTraffic));
    p2pHostToTorCrossTraffic.SetChannelAttribute("Delay", StringValue(hostToTorLinkDelay));


    for (int i = 0; i < nRacks; i++) {
        vector<NetDeviceContainer> hostsToTors;
        for (int j = 0; j < nHosts; j++) {
            // if (i == 0) {
                if (j < nHosts/2) {
                    hostsToTors.push_back(p2pHostToTorMeasurementTraffic.Install(racks[i].Get(j), torSwitches.Get(i)));
                } else {
                    hostsToTors.push_back(p2pHostToTorCrossTraffic.Install(racks[i].Get(j), torSwitches.Get(i)));
                }
            // }
            // else {
            //     hostsToTors.push_back(p2pHostToTorCrossTraffic.Install(racks[i].Get(j), torSwitches.Get(i)));
            // }
        }
        hostsToTorsNetDevices.push_back(hostsToTors);
    }

    // connecting the Tor Switches to each other
    vector<NetDeviceContainer> torToTorNetDevices;
    PointToPointHelper p2pTorToTor;
    p2pTorToTor.SetDeviceAttribute("DataRate", StringValue(torToAggLinkRate));
    p2pTorToTor.SetChannelAttribute("Delay", StringValue(torToAggLinkDelay));
    
    for (int i = 0; i < nRacks - 1; i++) {
        torToTorNetDevices.push_back(p2pTorToTor.Install(torSwitches.Get(i), torSwitches.Get(i + 1)));
    }

    InternetStackHelper stack;
    stack.InstallAll();

    TrafficControlHelper torToTorTCH;
    torToTorTCH.SetRootQueueDisc("ns3::RedQueueDisc", 
                                  "LinkBandwidth", StringValue(torToAggLinkRate),
                                  "LinkDelay", StringValue(torToAggLinkDelay), 
                                  "MinTh", DoubleValue(50),
                                  "MaxTh", DoubleValue(150));
    vector<QueueDiscContainer> torTotorQueueDiscs;
    for (int i = 0; i < nRacks - 1; i++) {
        QueueDiscContainer qdisc = torToTorTCH.Install(torToTorNetDevices[i]);
        torTotorQueueDiscs.push_back(qdisc);
    }

    TrafficControlHelper hosToTorTCH;
    hosToTorTCH.SetRootQueueDisc("ns3::RedQueueDisc", 
                                 "LinkBandwidth", StringValue(hostToTorLinkRate),
                                 "LinkDelay", StringValue(hostToTorLinkDelay), 
                                 "MinTh", DoubleValue(20),
                                 "MaxTh", DoubleValue(60));
    vector<vector<QueueDiscContainer>> hostToTorQueueDiscs;
    for (int i = 0; i < nRacks; i++) {
        vector<QueueDiscContainer> qdiscs;
        for (int j = 0; j < nHosts; j++) {
            QueueDiscContainer qdisc = hosToTorTCH.Install(hostsToTorsNetDevices[i][j].Get(1));
            qdiscs.push_back(qdisc);
        }
        hostToTorQueueDiscs.push_back(qdiscs);
    }
    
    // Assign IP addresses
    uint16_t nbSubnet = 0;
    Ipv4AddressHelper address;
    vector<vector<Ipv4InterfaceContainer>> ipsRacks;
    ipsRacks.reserve(nRacks);
    for (int i = 0; i < nRacks; i++) {
        vector<Ipv4InterfaceContainer> ips;
        address.SetBase(("10." + to_string(++nbSubnet) + ".1.0").c_str(), "255.255.255.0");
        for (int j = 0; j < nHosts; j++) {
            ips.push_back(address.Assign(hostsToTorsNetDevices[i][j]));
            address.NewNetwork();
        }
        ipsRacks.push_back(ips);
    }

    // set the ips between the ToR switches
    address.SetBase(("10." + to_string(++nbSubnet) + ".1.0").c_str(), "255.255.255.0");
    vector<Ipv4InterfaceContainer> ipsTorToTor;
    for (int i = 0; i < nRacks - 1; i++) {
        ipsTorToTor.push_back(address.Assign(torToTorNetDevices[i]));
        address.NewNetwork();
    }
    
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();
    /* ########## END: Ceating the topology ########## */



    /* ########## START: Application Setup ########## */
    // r0h0 -> r1h0
    auto* appTraffic = new BackgroundReplay(racks[0].Get(0), racks[1].Get(0));
    appTraffic->SetPctOfPacedTcps(pctPacedBack);
    // string tracesPath = "/home/mahdi/Documents/NAL/Data/chicago_2010_traffic_10min_2paths/path0";
    string tracesPath = "/home/mahdi/Documents/Data/chicago_2010_traffic_10min_2paths/path0";
    if (std::filesystem::exists(tracesPath)) {
        appTraffic->RunAllTCPTraces(tracesPath, 0);
    } else {
        cout << "requested Background Directory does not exist" << endl;
    }

    // R0h1 -> R1h1
    auto* appTraffic2 = new BackgroundReplay(racks[0].Get(1), racks[1].Get(1));
    appTraffic2->SetPctOfPacedTcps(pctPacedBack);
    // string tracesPath2 = "/home/mahdi/Documents/NAL/Data/chicago_2010_traffic_10min_2paths/path1";
    string tracesPath2 = "/home/mahdi/Documents/Data/chicago_2010_traffic_10min_2paths/path1";
    if (std::filesystem::exists(tracesPath2)) {
        appTraffic2->RunAllTCPTraces(tracesPath2, 0);
    } else {
        cout << "requested Background Directory does not exist" << endl;
    }

    // // R1h2 -> R1h0
    // auto* appTraffic3 = new BackgroundReplay(racks[1].Get(2), racks[1].Get(0));
    // appTraffic3->SetPctOfPacedTcps(pctPacedBack);
    // string tracesPath3 = "/home/mahdi/Documents/NAL/Data/chicago_2010_traffic_10min_2paths/path0";
    // if (std::filesystem::exists(tracesPath3)) {
    //     appTraffic3->RunAllTraces(tracesPath3, 0);
    // } else {
    //     cout << "requested Background Directory does not exist" << endl;
    // }

    // r0h2 -> r1h2
    auto* appTraffic4 = new BackgroundReplay(racks[0].Get(2), racks[1].Get(2));
    appTraffic4->SetPctOfPacedTcps(pctPacedBack);
    // string tracesPath4 = "/home/mahdi/Documents/NAL/Data/chicago_2010_traffic_10min_2paths/path1";
    string tracesPath4 = "/home/mahdi/Documents/Data/chicago_2010_traffic_10min_2paths/path1";
    if (std::filesystem::exists(tracesPath4)) {
        appTraffic4->RunAllTraces(tracesPath4, 0);
    } else {
        cout << "requested Background Directory does not exist" << endl;
    }

    // r0h3 -> r1h3
    auto* appTraffic5 = new BackgroundReplay(racks[0].Get(3), racks[1].Get(3));
    appTraffic5->SetPctOfPacedTcps(pctPacedBack);
    // string tracesPath5 = "/home/mahdi/Documents/NAL/Data/chicago_2010_traffic_10min_2paths/path0";
    string tracesPath5 = "/home/mahdi/Documents/Data/chicago_2010_traffic_10min_2paths/path0";
    if (std::filesystem::exists(tracesPath5)) {
        appTraffic5->RunAllTraces(tracesPath5, 0);
    } else {
        cout << "requested Background Directory does not exist" << endl;
    }

    // NS3 application
    // // r0h1 -> r1h1
    // uint16_t port = 50000;
    // Address sinkLocalAddress(InetSocketAddress(Ipv4Address::GetAny(), port));
    // PacketSinkHelper sinkHelper("ns3::TcpSocketFactory", sinkLocalAddress);
    // ApplicationContainer sinkApp = sinkHelper.Install(racks[1].Get(1));
    // Ptr<PacketSink> s2r2PacketSink = sinkApp.Get(0)->GetObject<PacketSink>();
    // sinkApp.Start(startTime);
    // sinkApp.Stop(stopTime);
    // OnOffHelper S2ClientHelper("ns3::TcpSocketFactory", Address());
    // S2ClientHelper.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
    // S2ClientHelper.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
    // S2ClientHelper.SetAttribute("DataRate", DataRateValue(DataRate(appDataRate)));
    // S2ClientHelper.SetAttribute("PacketSize", UintegerValue(1000));
    // ApplicationContainer S2ClientApp;
    // AddressValue remoteAddress(InetSocketAddress(ipsRacks[1][1].GetAddress(0), port));
    // S2ClientHelper.SetAttribute("Remote", remoteAddress);
    // S2ClientApp.Add(S2ClientHelper.Install(racks[0].Get(1)));
    // S2ClientApp.Start(startTime);
    // S2ClientApp.Stop(stopTime);

    // // r0h0 -> r1h0
    // uint16_t port2 = 50001;
    // Address sinkLocalAddress2 = Address(InetSocketAddress(Ipv4Address::GetAny(), port2));
    // PacketSinkHelper sinkHelper2 = PacketSinkHelper("ns3::TcpSocketFactory", sinkLocalAddress2);
    // ApplicationContainer sinkApp2 = sinkHelper2.Install(racks[1].Get(0));
    // Ptr<PacketSink> s1r1PacketSink = sinkApp2.Get(0)->GetObject<PacketSink>();
    // sinkApp2.Start(startTime);
    // sinkApp2.Stop(stopTime);
    // OnOffHelper S1ClientHelper("ns3::TcpSocketFactory", Address());
    // S1ClientHelper.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
    // S1ClientHelper.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
    // S1ClientHelper.SetAttribute("DataRate", DataRateValue(DataRate(appDataRate)));
    // S1ClientHelper.SetAttribute("PacketSize", UintegerValue(1000));
    // ApplicationContainer S1ClientApp;
    // AddressValue remoteAddress2 = AddressValue(InetSocketAddress(ipsRacks[1][0].GetAddress(0), port2));
    // S1ClientHelper.SetAttribute("Remote", remoteAddress2);
    // S1ClientApp.Add(S1ClientHelper.Install(racks[0].Get(0)));
    // S1ClientApp.Start(startTime);
    // S1ClientApp.Stop(stopTime);

    // // r1h2 -> r1h0
    // uint16_t port3 = 50002;
    // Address sinkLocalAddress3 = Address(InetSocketAddress(Ipv4Address::GetAny(), port3));
    // PacketSinkHelper sinkHelper3 = PacketSinkHelper("ns3::TcpSocketFactory", sinkLocalAddress3);
    // ApplicationContainer sinkApp3 = sinkHelper3.Install(racks[1].Get(0));
    // Ptr<PacketSink> s3r1PacketSink = sinkApp3.Get(0)->GetObject<PacketSink>();
    // sinkApp3.Start(startTime);
    // sinkApp3.Stop(stopTime);
    // OnOffHelper S3ClientHelper("ns3::TcpSocketFactory", Address());
    // S3ClientHelper.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
    // S3ClientHelper.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
    // S3ClientHelper.SetAttribute("DataRate", DataRateValue(DataRate(appDataRate)));
    // S3ClientHelper.SetAttribute("PacketSize", UintegerValue(1000));
    // ApplicationContainer S3ClientApp;
    // AddressValue remoteAddress3 = AddressValue(InetSocketAddress(ipsRacks[1][0].GetAddress(0), port3));
    // S3ClientHelper.SetAttribute("Remote", remoteAddress3);
    // S3ClientApp.Add(S3ClientHelper.Install(racks[1].Get(2)));
    // S3ClientApp.Start(startTime);
    // S3ClientApp.Stop(stopTime);

    // r0h2 -> r1h2: BulkSendApplication
    // uint16_t port4 = 50003;
    // Address sinkLocalAddress4 = Address(InetSocketAddress(Ipv4Address::GetAny(), port4));
    // PacketSinkHelper sinkHelper4 = PacketSinkHelper("ns3::TcpSocketFactory", sinkLocalAddress4);
    // ApplicationContainer sinkApp4 = sinkHelper4.Install(racks[1].Get(2));
    // Ptr<PacketSink> s4r2PacketSink = sinkApp4.Get(0)->GetObject<PacketSink>();
    // sinkApp4.Start(startTime);
    // sinkApp4.Stop(stopTime);
    // BulkSendHelper S4ClientHelper("ns3::TcpSocketFactory", Address());
    // S4ClientHelper.SetAttribute("MaxBytes", UintegerValue(0));
    // ApplicationContainer S4ClientApp;
    // AddressValue remoteAddress4 = AddressValue(InetSocketAddress(ipsRacks[1][2].GetAddress(0), port4));
    // S4ClientHelper.SetAttribute("Remote", remoteAddress4);
    // S4ClientApp.Add(S4ClientHelper.Install(racks[0].Get(2)));
    // S4ClientApp.Start(startTime);
    // S4ClientApp.Stop(stopTime);

    // // r0h3 -> r1h3: BulkSendApplication
    // uint16_t port5 = 50004;
    // Address sinkLocalAddress5 = Address(InetSocketAddress(Ipv4Address::GetAny(), port5));
    // PacketSinkHelper sinkHelper5 = PacketSinkHelper("ns3::TcpSocketFactory", sinkLocalAddress5);
    // ApplicationContainer sinkApp5 = sinkHelper5.Install(racks[1].Get(3));
    // Ptr<PacketSink> s5r2PacketSink = sinkApp5.Get(0)->GetObject<PacketSink>();
    // sinkApp5.Start(startTime);
    // sinkApp5.Stop(stopTime);
    // BulkSendHelper S5ClientHelper("ns3::TcpSocketFactory", Address());
    // S5ClientHelper.SetAttribute("MaxBytes", UintegerValue(0));
    // ApplicationContainer S5ClientApp;
    // AddressValue remoteAddress5 = AddressValue(InetSocketAddress(ipsRacks[1][3].GetAddress(0), port5));
    // S5ClientHelper.SetAttribute("Remote", remoteAddress5);
    // S5ClientApp.Add(S5ClientHelper.Install(racks[0].Get(3)));
    // S5ClientApp.Start(startTime);
    // S5ClientApp.Stop(stopTime);

    /* ########## END: Application Setup ########## */



    /* ########## START: Monitoring ########## */
    // p2pHostToTor.EnablePcapAll("N4_datacenter_switch_");
    ns3::PacketMetadata::Enable();

    // End to End Monitors
    vector<PacketMonitor *> endToendMonitors, endToEndCrossMonior;
    // r0h0 -> r1h0 Monitor
    auto *R0h0R1h0Monitor = new PacketMonitor(startTime, stopTime + convergenceTime, racks[0].Get(0), racks[1].Get(0), "R0h0R1h0");
    R0h0R1h0Monitor->AddAppKey(AppKey(ipsRacks[0][0].GetAddress(0), ipsRacks[1][0].GetAddress(0), 0, 0));
    endToendMonitors.push_back(R0h0R1h0Monitor);

    // r0h1 -> r1h1 Monitor
    auto *R0h1R1h1Monitor = new PacketMonitor(startTime, stopTime + convergenceTime, racks[0].Get(1), racks[1].Get(1), "R0h1R1h1");
    R0h1R1h1Monitor->AddAppKey(AppKey(ipsRacks[0][1].GetAddress(0), ipsRacks[1][1].GetAddress(0), 0, 0));
    endToendMonitors.push_back(R0h1R1h1Monitor);

    // // r1h2 -> r1h0 Monitor
    // auto *R1h2R1h1Monitor = new PacketMonitor(startTime, stopTime + convergenceTime, racks[1].Get(2), racks[1].Get(0), "R1h2R1h0");
    // R1h2R1h1Monitor->AddAppKey(AppKey(ipsRacks[1][2].GetAddress(0), ipsRacks[1][0].GetAddress(0), 0, 0));
    // endToendMonitors.push_back(R1h2R1h1Monitor);

    // r0h2 -> r1h2 Monitor
    auto *R0h2R1h2Monitor = new PacketMonitor(startTime, stopTime + convergenceTime, racks[0].Get(2), racks[1].Get(2), "R0h2R1h2");
    R0h2R1h2Monitor->AddAppKey(AppKey(ipsRacks[0][2].GetAddress(0), ipsRacks[1][2].GetAddress(0), 0, 0));
    endToEndCrossMonior.push_back(R0h2R1h2Monitor);

    // r0h3 -> r1h3 Monitor
    auto *R0h3R1h3Monitor = new PacketMonitor(startTime, stopTime + convergenceTime, racks[0].Get(3), racks[1].Get(3), "R0h3R1h3");
    R0h3R1h3Monitor->AddAppKey(AppKey(ipsRacks[0][3].GetAddress(0), ipsRacks[1][3].GetAddress(0), 0, 0));
    endToEndCrossMonior.push_back(R0h3R1h3Monitor);

    // Switch Monitors
    vector<SwitchMonitor *> switchMonitors;
    // T0 Switch Monitor
    auto *T0SwitchMonitor = new SwitchMonitor(startTime, stopTime + convergenceTime, torSwitches.Get(0), "T0");
    T0SwitchMonitor->AddAppKey(AppKey(ipsRacks[0][0].GetAddress(0), ipsRacks[1][0].GetAddress(0), 0, 0));
    T0SwitchMonitor->AddAppKey(AppKey(ipsRacks[0][1].GetAddress(0), ipsRacks[1][1].GetAddress(0), 0, 0));
    T0SwitchMonitor->AddAppKey(AppKey(ipsRacks[0][2].GetAddress(0), ipsRacks[1][2].GetAddress(0), 0, 0));
    T0SwitchMonitor->AddAppKey(AppKey(ipsRacks[0][3].GetAddress(0), ipsRacks[1][3].GetAddress(0), 0, 0));
    switchMonitors.push_back(T0SwitchMonitor);

    // T1 Switch Monitor
    auto *T1SwitchMonitor = new SwitchMonitor(startTime, stopTime + convergenceTime, torSwitches.Get(1), "T1");
    T1SwitchMonitor->AddAppKey(AppKey(ipsRacks[0][0].GetAddress(0), ipsRacks[1][0].GetAddress(0), 0, 0));
    T1SwitchMonitor->AddAppKey(AppKey(ipsRacks[0][1].GetAddress(0), ipsRacks[1][1].GetAddress(0), 0, 0));
    T1SwitchMonitor->AddAppKey(AppKey(ipsRacks[0][2].GetAddress(0), ipsRacks[1][2].GetAddress(0), 0, 0));
    T1SwitchMonitor->AddAppKey(AppKey(ipsRacks[0][3].GetAddress(0), ipsRacks[1][3].GetAddress(0), 0, 0));
    switchMonitors.push_back(T1SwitchMonitor);

    // Poisson Samplers on the ToR switches
    vector<PoissonSampler *> poissonSamplers;
    // T0 -> T1 Poisson Sampler
    auto *T0PoissonSampler = new PoissonSampler(startTime, stopTime + convergenceTime, DynamicCast<RedQueueDisc>(torTotorQueueDiscs[0].Get(0)), DynamicCast<PointToPointNetDevice>(torToTorNetDevices[0].Get(0))->GetQueue(), DynamicCast<PointToPointNetDevice>(torToTorNetDevices[0].Get(0)), "T0T1", sampleRate);
    poissonSamplers.push_back(T0PoissonSampler);

    // T1 -> R1h0 Poisson Sampler
    auto *T1R1h0PoissonSampler = new PoissonSampler(startTime, stopTime + convergenceTime, DynamicCast<RedQueueDisc>(hostToTorQueueDiscs[1][0].Get(0)), DynamicCast<PointToPointNetDevice>(hostsToTorsNetDevices[1][0].Get(1))->GetQueue(), DynamicCast<PointToPointNetDevice>(hostsToTorsNetDevices[1][0].Get(1)), "T1.R1h0", sampleRate);
    poissonSamplers.push_back(T1R1h0PoissonSampler);

    // T1 -> R1h1 Poisson Sampler
    auto *T1R1h1PoissonSampler = new PoissonSampler(startTime, stopTime + convergenceTime, DynamicCast<RedQueueDisc>(hostToTorQueueDiscs[1][1].Get(0)), DynamicCast<PointToPointNetDevice>(hostsToTorsNetDevices[1][1].Get(1))->GetQueue(), DynamicCast<PointToPointNetDevice>(hostsToTorsNetDevices[1][1].Get(1)), "T1.R1h1", sampleRate);
    poissonSamplers.push_back(T1R1h1PoissonSampler);
    /* ########## END: Monitoring ########## */



    /* ########## START: Check Config ########## */ 
    // print hosts and switches IP addresses
    cout << "Hosts and Switches IP addresses" << endl;
    for (int i = 0; i < nRacks; i++) {
        for (int j = 0; j < nHosts; j++) {
            cout << "Rack: " << i << " Host: "<< j << " Id:" << racks[i].Get(j)->GetId() << " IP: " << ipsRacks[i][j].GetAddress(0) << endl;
        }
    }
    //print config parameters
    cout << "Config Parameters" << endl;
    cout << "hostToTorLinkRate: " << hostToTorLinkRate << endl;
    cout << "hostToTorLinkRateCrossTraffic: " << hostToTorLinkRateCrossTraffic << endl;
    cout << "hostToTorLinkDelay: " << hostToTorLinkDelay << endl;
    cout << "torToAggLinkRate: " << torToAggLinkRate << endl;
    cout << "torToAggLinkDelay: " << torToAggLinkDelay << endl;
    cout << "aggToCoreLinkRate: " << aggToCoreLinkRate << endl;
    cout << "aggToCoreLinkDelay: " << aggToCoreLinkDelay << endl;
    cout << "appDataRate: " << appDataRate << endl;
    cout << "duration: " << duration << endl;
    cout << "pctPacedBack: " << pctPacedBack << endl;
    cout << "enableSwitchECN: " << enableSwitchECN << endl;
    cout << "enableECMP: " << enableECMP << endl;
    cout << "sampleRate: " << sampleRate << endl;

    /* ########## END: Check Config ########## */


    /* ########## START: Scheduling and  Running ########## */
    // DynamicCast<RedQueueDisc>(torTotorQueueDiscs[0].Get(0))->TraceConnectWithoutContext("PacketsInQueue", MakeCallback(&queueDiscSize));
    // DynamicCast<PointToPointNetDevice>(torToTorNetDevices[0].Get(0))->GetQueue()->TraceConnectWithoutContext("PacketsInQueue", MakeCallback(&queueSize));

    // DynamicCast<RedQueueDisc>(hostToTorQueueDiscs[1][0].Get(0))->TraceConnectWithoutContext("Enqueue", MakeCallback(&enqueueDisc));
    // DynamicCast<PointToPointNetDevice>(hostsToTorsNetDevices[1][0].Get(1))->GetQueue()->TraceConnectWithoutContext("PacketsInQueue", MakeCallback(&queueSize));
    // DynamicCast<PointToPointNetDevice>(hostsToTorsNetDevices[1][2].Get(1))->GetQueue()->TraceConnectWithoutContext("PacketsInQueue", MakeCallback(&queueSize2));
    // DynamicCast<PointToPointNetDevice>(torToTorNetDevices[0].Get(0))->GetQueue()->TraceConnectWithoutContext("Dequeue", MakeCallback(&dequeue));

    Simulator::Stop(stopTime + convergenceTime + convergenceTime);
    Simulator::Run();
    Simulator::Destroy();

    for (auto monitor: endToendMonitors) {
        monitor->SavePacketRecords((string) (getenv("PWD")) + "/results/" + monitor->GetMonitorTag() + "_EndToEnd.csv");
    }
    for (auto monitor: switchMonitors) {
        monitor->SavePacketRecords((string) (getenv("PWD")) + "/results/" + monitor->GetMonitorTag() + "_Switch.csv");
    }
    for (auto sampler: poissonSamplers) {
        sampler->SaveSamples((string) (getenv("PWD")) + "/results/" + sampler->GetSampleTag() + "_PoissonSampler.csv");
    }
    for (auto monitor: endToEndCrossMonior){
        monitor->SavePacketRecords((string) (getenv("PWD")) + "/results/" + monitor->GetMonitorTag() + "_EndToEnd_crossTraffic.csv");
    }
    /* ########## END: Scheduling and  Running ########## */



    cout << "Done " << endl;

    auto stop = std::chrono::high_resolution_clock::now();
    cout << "Total execution time = " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " microsecond" << endl;
    return 0;
}