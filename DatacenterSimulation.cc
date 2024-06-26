//
// Created by Mahdi Hosseini on 5.06.24.
//
// Signiture:
// ****** Mahdi Change ***** (START) ***** // 

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
#include "monitors_module/RegularSampler.h"
#include "monitors_module/NetDeviceMonitor.h"
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
    std::cout << Simulator::Now().GetNanoSeconds() << "Packet enqueued: ";
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
    bool enableECMP = true;                            // Enable ECMP on the switches
    double sampleRate = 10;                            // Sample rate for the PoissonSampler
    int minTh = 9000;                                  // RED Queue Disc MinTh
    int maxTh = 28000;                                 // RED Queue Disc MaxTh
    int experiment = 1;                                // Experiment number
    double errorRate = 0.005;                          // Silent Packet Drop Error rate

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
    cmd.AddValue("minTh", "RED Queue Disc MinTh", minTh);
    cmd.AddValue("maxTh", "RED Queue Disc MaxTh", maxTh);
    cmd.AddValue("experiment", "Experiment number", experiment);
    cmd.AddValue("errorRate", "Silent Packet Drop Error rate", errorRate);
    cmd.Parse(argc, argv);

    /*set default values*/
    ns3::RngSeedManager::SetSeed(experiment);
    Time startTime = Seconds(0);
    Time stopTime = Seconds(stof(duration));
    Time stopTime_1 = Seconds(stof(duration));  
    Time convergenceTime = Seconds(0.5);
    stopTime = stopTime + convergenceTime;

    Config::SetDefault("ns3::TcpL4Protocol::SocketType", StringValue("ns3::TcpDctcp"));
    Config::SetDefault("ns3::Ipv4GlobalRouting::RandomEcmpRouting", BooleanValue(enableECMP));
    Config::SetDefault("ns3::RedQueueDisc::UseEcn", BooleanValue(enableSwitchECN));
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(1448));
    Config::SetDefault("ns3::TcpSocket::DelAckCount", UintegerValue(2));
    GlobalValue::Bind("ChecksumEnabled", BooleanValue(false));
    Config::SetDefault("ns3::RedQueueDisc::UseHardDrop", BooleanValue(false));
    Config::SetDefault("ns3::RedQueueDisc::MeanPktSize", UintegerValue(1500));
    Config::SetDefault("ns3::RedQueueDisc::MaxSize", QueueSizeValue(QueueSize("37.5KB")));
    Config::SetDefault("ns3::RedQueueDisc::QW", DoubleValue(1));
    Config::SetDefault("ns3::RedQueueDisc::MinTh", DoubleValue(minTh));
    Config::SetDefault("ns3::RedQueueDisc::MaxTh", DoubleValue(maxTh));

    // GlobalValue::Bind("SimulatorImplementationType", StringValue("ns3::DistributedSimulatorImpl"));
    // MPI_Init(&argc, &argv);
    // MPI_Comm splitComm = MPI_COMM_WORLD;
    // MpiInterface::Enable(splitComm);
    /* ########## END: Config ########## */



    /* ########## START: Ceating the topology ########## */
    int nHosts = 6;
    int nRacks = 4;
    int nAggSwitches = 2;
    int nCoreSwitches = 1;

    vector<NodeContainer> racks;
    racks.reserve(nRacks);
    NodeContainer torSwitches;
    NodeContainer aggSwitches;
    NodeContainer coreSwitches;

    // Create the racks and switches containers
    for (int i = 0; i < nRacks; i++) {
        NodeContainer rack;
        rack.Create(nHosts);
        racks.push_back(rack);
    }
    torSwitches.Create(nRacks);
    aggSwitches.Create(nAggSwitches);
    coreSwitches.Create(nCoreSwitches);

    // connecting the hosts to the ToR switches
    vector<vector<NetDeviceContainer>> hostsToTorsNetDevices;

    PointToPointHelper p2pHostToTor;
    p2pHostToTor.SetDeviceAttribute("DataRate", StringValue(hostToTorLinkRate));
    p2pHostToTor.SetChannelAttribute("Delay", StringValue(hostToTorLinkDelay));

    for (int i = 0; i < nRacks; i++) {
        vector<NetDeviceContainer> hostsToTors;
        for (int j = 0; j < nHosts; j++) {
            hostsToTors.push_back(p2pHostToTor.Install(racks[i].Get(j), torSwitches.Get(i)));
        }
        hostsToTorsNetDevices.push_back(hostsToTors);
    }

    // connecting the Tor Switches to the Agg Switches
    vector<vector<NetDeviceContainer>> torToAggNetDevices;
    PointToPointHelper p2pTorToTor;
    p2pTorToTor.SetDeviceAttribute("DataRate", StringValue(torToAggLinkRate));
    p2pTorToTor.SetChannelAttribute("Delay", StringValue(torToAggLinkDelay));
    
    for (int i = 0; i < nRacks; i++) {
        vector<NetDeviceContainer> torToAgg;
        for (int j = 0; j < nAggSwitches; j++) {
            torToAgg.push_back(p2pTorToTor.Install(torSwitches.Get(i), aggSwitches.Get(j)));
        }
        torToAggNetDevices.push_back(torToAgg);        
    }

    // connecting the agg switches to the core switches
    vector<vector<NetDeviceContainer>> aggToCoreNetDevices;
    PointToPointHelper p2pAggToCore;
    p2pAggToCore.SetDeviceAttribute("DataRate", StringValue(aggToCoreLinkRate));
    p2pAggToCore.SetChannelAttribute("Delay", StringValue(aggToCoreLinkDelay));

    for (int i = 0; i < nAggSwitches; i++) {
        vector<NetDeviceContainer> aggToCore;
        for (int j = 0; j < nCoreSwitches; j++) {
            aggToCore.push_back(p2pAggToCore.Install(aggSwitches.Get(i), coreSwitches.Get(j)));
        }
        aggToCoreNetDevices.push_back(aggToCore);
    }
    
    // Install the network stack on the nodes
    InternetStackHelper stack;
    stack.InstallAll();

    // Install RED Queue Discs on the ToR switches, on ToR to Host links
    TrafficControlHelper torToHostTCH;
    torToHostTCH.SetRootQueueDisc("ns3::RedQueueDisc", 
                                  "LinkBandwidth", StringValue(hostToTorLinkRate),
                                  "LinkDelay", StringValue(hostToTorLinkDelay), 
                                  "MinTh", DoubleValue(minTh),
                                  "MaxTh", DoubleValue(maxTh));
    vector<vector<QueueDiscContainer>> torToHostQueueDiscs;
    for (int i = 0; i < nRacks; i++) {
        vector<QueueDiscContainer> qdiscs;
        for (int j = 0; j < nHosts; j++) {
            qdiscs.push_back(torToHostTCH.Install(hostsToTorsNetDevices[i][j].Get(1)));
        }
        torToHostQueueDiscs.push_back(qdiscs);
    }

    // Install RED Queue Discs on the ToR switches, on ToR to Agg links and Agg to ToR links
    TrafficControlHelper torToAggTCH;
    torToAggTCH.SetRootQueueDisc("ns3::RedQueueDisc", 
                                  "LinkBandwidth", StringValue(torToAggLinkRate),
                                  "LinkDelay", StringValue(torToAggLinkDelay), 
                                  "MinTh", DoubleValue(minTh),
                                  "MaxTh", DoubleValue(maxTh));
    vector<vector<QueueDiscContainer>> torToAggQueueDiscs;
    for (int i = 0; i < nRacks; i++) {
        vector<QueueDiscContainer> qdiscs;
        for (int j = 0; j < nAggSwitches; j++) {
            qdiscs.push_back(torToAggTCH.Install(torToAggNetDevices[i][j]));
        }
        torToAggQueueDiscs.push_back(qdiscs);
    }

    // Install RED Queue Discs on the Agg switches, on Agg to Core links and Core to Agg links
    TrafficControlHelper aggToCoreTCH;
    aggToCoreTCH.SetRootQueueDisc("ns3::RedQueueDisc", 
                                  "LinkBandwidth", StringValue(aggToCoreLinkRate),
                                  "LinkDelay", StringValue(aggToCoreLinkDelay), 
                                  "MinTh", DoubleValue(minTh),
                                  "MaxTh", DoubleValue(maxTh));
    vector<vector<QueueDiscContainer>> aggToCoreQueueDiscs;
    for (int i = 0; i < nAggSwitches; i++) {
        vector<QueueDiscContainer> qdiscs;
        for (int j = 0; j < nCoreSwitches; j++) {
            qdiscs.push_back(aggToCoreTCH.Install(aggToCoreNetDevices[i][j]));
        }
        aggToCoreQueueDiscs.push_back(qdiscs);
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

    // set the ips between the ToR switches and the Agg switches
    vector<vector<Ipv4InterfaceContainer>> ipsTorToAgg;
    address.SetBase(("10." + to_string(++nbSubnet) + ".1.0").c_str(), "255.255.255.0");
    for (int i = 0; i < nRacks; i++) {
        vector<Ipv4InterfaceContainer> ips;
        for (int j = 0; j < nAggSwitches; j++) {
            ips.push_back(address.Assign(torToAggNetDevices[i][j]));
            address.NewNetwork();
        }
        ipsTorToAgg.push_back(ips);
    }
    
    // set the ips between the Agg switches and the Core switches
    vector<vector<Ipv4InterfaceContainer>> ipsAggToCore;
    address.SetBase(("10." + to_string(++nbSubnet) + ".1.0").c_str(), "255.255.255.0");
    for (int i = 0; i < nAggSwitches; i++) {
        vector<Ipv4InterfaceContainer> ips;
        for (int j = 0; j < nCoreSwitches; j++) {
            ips.push_back(address.Assign(aggToCoreNetDevices[i][j]));
            address.NewNetwork();
        }
        ipsAggToCore.push_back(ips);
    }

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    /* Erro Model Setup for Silent packet drops*/
    // Ptr<RateErrorModel> em_R0H0T0 = CreateObject<RateErrorModel>();
    // em_R0H0T0->SetAttribute("ErrorRate", DoubleValue(errorRate));
    // em_R0H0T0->SetUnit(RateErrorModel::ErrorUnit::ERROR_UNIT_PACKET);
    // hostsToTorsNetDevices[0][0].Get(1)->SetAttribute("ReceiveErrorModel", PointerValue(em_R0H0T0));

    // Ptr<RateErrorModel> em_R0H1T0 = CreateObject<RateErrorModel>();
    // em_R0H1T0->SetAttribute("ErrorRate", DoubleValue(errorRate));
    // em_R0H1T0->SetUnit(RateErrorModel::ErrorUnit::ERROR_UNIT_PACKET);
    // hostsToTorsNetDevices[0][1].Get(1)->SetAttribute("ReceiveErrorModel", PointerValue(em_R0H1T0));
    /* ########## END: Ceating the topology ########## */



    /* ########## START: Application Setup ########## */
    // CAIDA trace replay
    // Each host in R0 sends a flow to the corresponding host in R2
    for (int i = 0; i < nHosts; i++) {
        auto* caidaTrafficGenerator = new BackgroundReplay(racks[0].Get(i), racks[2].Get(i));
        caidaTrafficGenerator->SetPctOfPacedTcps(pctPacedBack);
        // string tracesPath = "/home/mahdi/Documents/NAL/Data/chicago_2010_traffic_10min_2paths/path" + to_string(i % 2);
        // string tracesPath = "/home/mahdi/Documents/Data/chicago_2010_traffic_10min_2paths/path" + to_string(i % 2);
        string tracesPath = "/home/shossein/chicago_2010_traffic_10min_2paths/path" + to_string(i % 2);
        if (std::filesystem::exists(tracesPath)) {
            caidaTrafficGenerator->RunAllTCPTraces(tracesPath, 0);
        } else {
            cout << "requested Background Directory does not exist" << endl;
        }
    }

    // each host in R1 sends a flow to the corresponding host in R3
    for (int i = 0; i < nHosts; i++) {
        auto* caidaTrafficGenerator = new BackgroundReplay(racks[1].Get(i), racks[3].Get(i));
        caidaTrafficGenerator->SetPctOfPacedTcps(pctPacedBack);
        // string tracesPath = "/home/mahdi/Documents/NAL/Data/chicago_2010_traffic_10min_2paths/path" + to_string(i % 2);
        // string tracesPath = "/home/mahdi/Documents/Data/chicago_2010_traffic_10min_2paths/path" + to_string(i % 2);
        string tracesPath = "/home/shossein/chicago_2010_traffic_10min_2paths/path" + to_string(i % 2);
        if (std::filesystem::exists(tracesPath)) {
            caidaTrafficGenerator->RunAllTCPTraces(tracesPath, 0);
        } else {
            cout << "requested Background Directory does not exist" << endl;
        }
    }

    // each host in R2 sends a flow to the corresponding host in R1
    // for (int i = 0; i < nHosts; i++) {
    //     auto* caidaTrafficGenerator = new BackgroundReplay(racks[2].Get(i), racks[1].Get(i));
    //     caidaTrafficGenerator->SetPctOfPacedTcps(pctPacedBack);
    //     string tracesPath = "/home/mahdi/Documents/NAL/Data/chicago_2010_traffic_10min_2paths/path" + to_string(i % 2);
    //     if (std::filesystem::exists(tracesPath)) {
    //         caidaTrafficGenerator->RunAllTCPTraces(tracesPath, 0);
    //     } else {
    //         cout << "requested Background Directory does not exist" << endl;
    //     }
    // }

    // // each host in R3 sends a flow to the corresponding host in R0
    // for (int i = 0; i < nHosts; i++) {
    //     auto* caidaTrafficGenerator = new BackgroundReplay(racks[3].Get(i), racks[0].Get(i));
    //     caidaTrafficGenerator->SetPctOfPacedTcps(pctPacedBack);
    //     string tracesPath = "/home/mahdi/Documents/NAL/Data/chicago_2010_traffic_10min_2paths/path" + to_string(i % 2);
    //     if (std::filesystem::exists(tracesPath)) {
    //         caidaTrafficGenerator->RunAllTCPTraces(tracesPath, 0);
    //     } else {
    //         cout << "requested Background Directory does not exist" << endl;
    //     }
    // }

    // NS3 application
    // // Each host in R0 sends a flow to the corresponding host in R2
    // vector<Ptr<PacketSink>> R2Sinks;
    // for (int j = 0; j < nHosts; j++) {
    //     uint16_t port = 50000 + j;
    //     Address sinkLocalAddress(InetSocketAddress(Ipv4Address::GetAny(), port));
    //     PacketSinkHelper sinkHelper("ns3::TcpSocketFactory", sinkLocalAddress);
    //     ApplicationContainer sinkApp = sinkHelper.Install(racks[2].Get(j));
    //     Ptr<PacketSink> sink = sinkApp.Get(0)->GetObject<PacketSink>();
    //     sinkApp.Start(startTime);
    //     sinkApp.Stop(stopTime);
    //     R2Sinks.push_back(sink);

    //     OnOffHelper clientHelper("ns3::TcpSocketFactory", Address());
    //     clientHelper.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
    //     clientHelper.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
    //     clientHelper.SetAttribute("DataRate", DataRateValue(DataRate(appDataRate)));
    //     clientHelper.SetAttribute("PacketSize", UintegerValue(1000));
    //     ApplicationContainer clientApp;
    //     AddressValue remoteAddress(InetSocketAddress(ipsRacks[2][j].GetAddress(0), port));
    //     clientHelper.SetAttribute("Remote", remoteAddress);
    //     clientApp.Add(clientHelper.Install(racks[0].Get(j)));
    //     clientApp.Start(startTime);
    //     clientApp.Stop(stopTime);
    // }

    // // Each host in R1 sends a flow to the corresponding host in R3
    // vector<Ptr<PacketSink>> R3Sinks;
    // for (int j = 0; j < nHosts; j++) {
    //     uint16_t port = 50000 + j;
    //     Address sinkLocalAddress(InetSocketAddress(Ipv4Address::GetAny(), port));
    //     PacketSinkHelper sinkHelper("ns3::TcpSocketFactory", sinkLocalAddress);
    //     ApplicationContainer sinkApp = sinkHelper.Install(racks[3].Get(j));
    //     Ptr<PacketSink> sink = sinkApp.Get(0)->GetObject<PacketSink>();
    //     sinkApp.Start(startTime);
    //     sinkApp.Stop(stopTime);
    //     R3Sinks.push_back(sink);

    //     OnOffHelper clientHelper("ns3::TcpSocketFactory", Address());
    //     clientHelper.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
    //     clientHelper.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
    //     clientHelper.SetAttribute("DataRate", DataRateValue(DataRate(appDataRate)));
    //     clientHelper.SetAttribute("PacketSize", UintegerValue(1000));
    //     ApplicationContainer clientApp;
    //     AddressValue remoteAddress(InetSocketAddress(ipsRacks[3][j].GetAddress(0), port));
    //     clientHelper.SetAttribute("Remote", remoteAddress);
    //     clientApp.Add(clientHelper.Install(racks[1].Get(j)));
    //     clientApp.Start(startTime);
    //     clientApp.Stop(stopTime);
    // }
    /* ########## END: Application Setup ########## */



    /* ########## START: Monitoring ########## */
    // p2pHostToTor.EnablePcapAll("N4_datacenter_switch_");
    ns3::PacketMetadata::Enable(); // Enable packet metadata for debugging

    // End to End Monitors
    vector<PacketMonitor *> endToendMonitors;
    // Monitor the packets between each pair of hosts in R0 and R2
    for (int i = 0; i < nHosts; i++) {
        auto *R0R2Monitor = new PacketMonitor(startTime, stopTime + convergenceTime, racks[0].Get(i), racks[2].Get(i), "R0H" + to_string(i) + "R2H" + to_string(i), errorRate);
        R0R2Monitor->AddAppKey(AppKey(ipsRacks[0][i].GetAddress(0), ipsRacks[2][i].GetAddress(0), 0, 0));
        endToendMonitors.push_back(R0R2Monitor);
    }

    // Monitor the packets between each pair of hosts in R1 and R3
    for (int i = 0; i < nHosts; i++) {
        auto *R1R3Monitor = new PacketMonitor(startTime, stopTime + convergenceTime, racks[1].Get(i), racks[3].Get(i), "R1H" + to_string(i) + "R3H" + to_string(i), errorRate);
        R1R3Monitor->AddAppKey(AppKey(ipsRacks[1][i].GetAddress(0), ipsRacks[3][i].GetAddress(0), 0, 0));
        endToendMonitors.push_back(R1R3Monitor);
    }

    // // Monitor the packets between each pair of hosts in R2 and R1
    // for (int i = 0; i < nHosts; i++) {
    //     auto *R2R1Monitor = new PacketMonitor(startTime, stopTime + convergenceTime, racks[2].Get(i), racks[1].Get(i), "R2H" + to_string(i) + "R1H" + to_string(i));
    //     R2R1Monitor->AddAppKey(AppKey(ipsRacks[2][i].GetAddress(0), ipsRacks[1][i].GetAddress(0), 0, 0));
    //     endToendMonitors.push_back(R2R1Monitor);
    // }

    // // Monitor the packets between each pair of hosts in R3 and R0
    // for (int i = 0; i < nHosts; i++) {
    //     auto *R3R0Monitor = new PacketMonitor(startTime, stopTime + convergenceTime, racks[3].Get(i), racks[0].Get(i), "R3H" + to_string(i) + "R0H" + to_string(i));
    //     R3R0Monitor->AddAppKey(AppKey(ipsRacks[3][i].GetAddress(0), ipsRacks[0][i].GetAddress(0), 0, 0));
    //     endToendMonitors.push_back(R3R0Monitor);
    // }

    // NetDevice monitors on the hosts
    vector<NetDeviceMonitor *> hostNetDeviceMonitors;
    for (int i = 0; i < nRacks / 2; i++) {
        for (int j = 0; j < nHosts; j++) {
            auto *hostNetDeviceMonitor = new NetDeviceMonitor(startTime, stopTime + convergenceTime, DynamicCast<PointToPointNetDevice>(hostsToTorsNetDevices[i][j].Get(0)), "R" + to_string(i) + "H" + to_string(j));
            hostNetDeviceMonitor->AddAppKey(AppKey(ipsRacks[i][j].GetAddress(0), ipsRacks[i + 2][j].GetAddress(0), 0, 0));
            hostNetDeviceMonitors.push_back(hostNetDeviceMonitor);
        }
    }

    // switch monitors on the ToR switches
    vector<SwitchMonitor *> torSwitchMonitors;
    for (int i = 0; i < nRacks; i++) {
        auto *torSwitchMonitor = new SwitchMonitor(startTime, stopTime + convergenceTime, torSwitches.Get(i), "T" + to_string(i));
        for (int j = 0; j < nHosts; j++) {
            if (i < nRacks / 2) {
                torSwitchMonitor->AddAppKey(AppKey(ipsRacks[i][j].GetAddress(0), ipsRacks[i + 2][j].GetAddress(0), 0, 0));
            } else {
                torSwitchMonitor->AddAppKey(AppKey(ipsRacks[i - 2][j].GetAddress(0), ipsRacks[i][j].GetAddress(0), 0, 0));
            }
        }
        torSwitchMonitors.push_back(torSwitchMonitor);
    }

    // switch monitors on the Agg switches
    vector<SwitchMonitor *> aggSwitchMonitors;
    for (int i = 0; i < nAggSwitches; i++) {
        auto *aggSwitchMonitor = new SwitchMonitor(startTime, stopTime + convergenceTime, aggSwitches.Get(i), "A" + to_string(i));
        for (int j = 0; j < nRacks / 2; j++) {
            for (int k = 0; k < nHosts; k++) {
                aggSwitchMonitor->AddAppKey(AppKey(ipsRacks[j][k].GetAddress(0), ipsRacks[j + 2][k].GetAddress(0), 0, 0));
            }
        }
        aggSwitchMonitors.push_back(aggSwitchMonitor);
    }

    // switch monitors on the Core switches
    vector<SwitchMonitor *> coreSwitchMonitors;
    for (int i = 0; i < nCoreSwitches; i++) {
        auto *coreSwitchMonitor = new SwitchMonitor(startTime, stopTime + convergenceTime, coreSwitches.Get(i), "C" + to_string(i));
        for (int j = 0; j < nRacks / 2; j++) {
            for (int k = 0; k < nHosts; k++) {
                coreSwitchMonitor->AddAppKey(AppKey(ipsRacks[j][k].GetAddress(0), ipsRacks[j + 2][k].GetAddress(0), 0, 0));
            }
        }
        coreSwitchMonitors.push_back(coreSwitchMonitor);
    }

    // PoissonSampler on the ToR switches, Agg switches and Core switches
    vector<PoissonSampler *> PoissonSamplers;
    // PoissonSampler on the Hosts
    for (int i = 0; i < nRacks / 2; i++) {
        for (int j = 0; j < nHosts; j++) {
            Ptr<PointToPointNetDevice> hostToTorNetDevice = DynamicCast<PointToPointNetDevice>(hostsToTorsNetDevices[i][j].Get(0));
            auto *hostToTorSampler = new PoissonSampler(startTime, stopTime, nullptr, hostToTorNetDevice->GetQueue(), hostToTorNetDevice, "R" + to_string(i) + "H" + to_string(j), sampleRate);
            PoissonSamplers.push_back(hostToTorSampler);
        }
    }
    // PoissonSampler on the tor to agg links
    for (int i = 0; i < nRacks; i++) {
        for (int j = 0; j < nAggSwitches; j++) {
            Ptr<PointToPointNetDevice> torToAggNetDevice = DynamicCast<PointToPointNetDevice>(torToAggNetDevices[i][j].Get(0));            
            auto *torToAggSampler = new PoissonSampler(startTime, stopTime, DynamicCast<RedQueueDisc>(torToAggQueueDiscs[i][j].Get(0)), torToAggNetDevice->GetQueue(), torToAggNetDevice, "T" + to_string(i) + "A" + to_string(j), sampleRate);
            PoissonSamplers.push_back(torToAggSampler);
        }
    }
    // PoissonSampler on the Tor to Host links
    for (int i = 0; i < nRacks; i++) {
        for (int j = 0; j < nHosts; j++) {
            Ptr<PointToPointNetDevice> hostToTorNetDevice = DynamicCast<PointToPointNetDevice>(hostsToTorsNetDevices[i][j].Get(1));
            auto *hostToTorSampler = new PoissonSampler(startTime, stopTime, DynamicCast<RedQueueDisc>(torToHostQueueDiscs[i][j].Get(0)), hostToTorNetDevice->GetQueue(), hostToTorNetDevice, "T" + to_string(i) + "H" + to_string(j), sampleRate);
            PoissonSamplers.push_back(hostToTorSampler);
        }
    }
    // PoissonSampler on the Agg to Tor links
    for (int i = 0; i < nRacks; i++) {
        for (int j = 0; j < nAggSwitches; j++) {
            Ptr<PointToPointNetDevice> aggToTorNetDevice = DynamicCast<PointToPointNetDevice>(torToAggNetDevices[i][j].Get(1));
            auto *aggToTorSampler = new PoissonSampler(startTime, stopTime, DynamicCast<RedQueueDisc>(torToAggQueueDiscs[i][j].Get(1)), aggToTorNetDevice->GetQueue(), aggToTorNetDevice, "A" + to_string(j) + "T" + to_string(i), sampleRate);
            PoissonSamplers.push_back(aggToTorSampler);
        }
    }
    // PoissonSampler on the Agg to Core links
    for (int i = 0; i < nAggSwitches; i++) {
        for (int j = 0; j < nCoreSwitches; j++) {
            Ptr<PointToPointNetDevice> aggToCoreNetDevice = DynamicCast<PointToPointNetDevice>(aggToCoreNetDevices[i][j].Get(0));
            auto *aggToCoreSampler = new PoissonSampler(startTime, stopTime, DynamicCast<RedQueueDisc>(aggToCoreQueueDiscs[i][j].Get(0)), aggToCoreNetDevice->GetQueue(), aggToCoreNetDevice, "A" + to_string(i) + "C" + to_string(j), sampleRate);
            PoissonSamplers.push_back(aggToCoreSampler);
        }
    }
    // PoissonSampler on the Core to Agg links
    for (int i = 0; i < nCoreSwitches; i++) {
        for (int j = 0; j < nAggSwitches; j++) {
            Ptr<PointToPointNetDevice> coreToAggNetDevice = DynamicCast<PointToPointNetDevice>(aggToCoreNetDevices[j][i].Get(1));
            auto *coreToAggSampler = new PoissonSampler(startTime, stopTime, DynamicCast<RedQueueDisc>(aggToCoreQueueDiscs[j][i].Get(1)), coreToAggNetDevice->GetQueue(), coreToAggNetDevice, "C" + to_string(i) + "A" + to_string(j), sampleRate);
            PoissonSamplers.push_back(coreToAggSampler);
        }
    }
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
    cout << "errorRate: " << errorRate << endl;
    cout << "T0 id " << torSwitches.Get(0)->GetId() << endl;
    /* ########## END: Check Config ########## */


    /* ########## START: Scheduling and  Running ########## */
    // DynamicCast<RedQueueDisc>(torToAggQueueDiscs[0][0].Get(0))->TraceConnectWithoutContext("BytesInQueue", MakeCallback(&queueDiscSize));
    // DynamicCast<PointToPointNetDevice>(torToAggNetDevices[0][0].Get(0))->GetQueue()->TraceConnectWithoutContext("PacketsInQueue", MakeCallback(&queueSize));

    // DynamicCast<RedQueueDisc>(hostToTorQueueDiscs[1][0].Get(0))->TraceConnectWithoutContext("Enqueue", MakeCallback(&enqueueDisc));
    // DynamicCast<PointToPointNetDevice>(hostsToTorsNetDevices[1][0].Get(1))->GetQueue()->TraceConnectWithoutContext("PacketsInQueue", MakeCallback(&queueSize));
    // DynamicCast<PointToPointNetDevice>(hostsToTorsNetDevices[1][2].Get(1))->GetQueue()->TraceConnectWithoutContext("PacketsInQueue", MakeCallback(&queueSize2));
    // DynamicCast<PointToPointNetDevice>(torToTorNetDevices[0].Get(0))->GetQueue()->TraceConnectWithoutContext("Dequeue", MakeCallback(&dequeue));
    // DynamicCast<PointToPointNetDevice>(hostsToTorsNetDevices[0][0].Get(0))->GetQueue()->TraceConnectWithoutContext("PacketsInQueue", MakeCallback(&queueSize));
    // DynamicCast<PointToPointNetDevice>(hostsToTorsNetDevices[0][0].Get(0))->GetQueue()->TraceConnectWithoutContext("Enqueue", MakeCallback(&enqueue));
    // DynamicCast<PointToPointNetDevice>(hostsToTorsNetDevices[0][0].Get(0))->GetQueue()->TraceConnectWithoutContext("Dequeue", MakeCallback(&dequeue));

    Simulator::Stop(stopTime_1);
    Simulator::Run();
    Simulator::Destroy();

    for (auto monitor: endToendMonitors) {
        monitor->SavePacketRecords((string) (getenv("PWD")) + "/results/" + to_string(experiment)  + "/" + monitor->GetMonitorTag() + "_EndToEnd.csv");
    }
    for (auto monitor: hostNetDeviceMonitors) {
        monitor->SavePacketRecords((string) (getenv("PWD")) + "/results/" + to_string(experiment)  + "/" + monitor->GetMonitorTag() + "_start.csv");
    }
    for (auto monitor: torSwitchMonitors) {
        monitor->SavePacketRecords((string) (getenv("PWD")) + "/results/" + to_string(experiment)  + "/" + monitor->GetMonitorTag() + "_Switch.csv");
    }
    for (auto monitor: aggSwitchMonitors) {
        monitor->SavePacketRecords((string) (getenv("PWD")) + "/results/" + to_string(experiment)  + "/" + monitor->GetMonitorTag() + "_Switch.csv");
    }
    for (auto monitor: coreSwitchMonitors) {
        monitor->SavePacketRecords((string) (getenv("PWD")) + "/results/" + to_string(experiment)  + "/" + monitor->GetMonitorTag() + "_Switch.csv");
    }
    for (auto monitor: PoissonSamplers) {
        monitor->SaveSamples((string) (getenv("PWD")) + "/results/" + to_string(experiment)  + "/" + monitor->GetSampleTag() + "_PoissonSampler.csv");
    }
    /* ########## END: Scheduling and  Running ########## */



    cout << "Done " << endl;

    auto stop = std::chrono::high_resolution_clock::now();
    cout << "Total execution time = " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " microsecond" << endl;
    return 0;
}