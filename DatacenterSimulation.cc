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
#include "monitors_module/E2EMonitor.h"
#include "monitors_module/AggregatedE2EMonitor.h"
#include "monitors_module/SwitchMonitor.h"
#include "monitors_module/PoissonSampler.h"
#include "monitors_module/RegularSampler.h"
#include "monitors_module/NetDeviceMonitor.h"
#include "monitors_module/BurstMonitor.h"
#include "traffic_generator_module/background_replay/BackgroundReplay.h"
#include "traffic_generator_module/DC_traffic_generator/DCWorkloadGenerator.h"
#include "traffic_generator_module/DC_traffic_generator/ProbeGenerator.h"
#include "traffic_generator_module/DC_traffic_generator/IncastGenerator.h"
#include "queue_discs/ShapingTrafficControlLayer.h"
#include <iomanip>
#include <iostream>
#include <string>
#include <cstdlib>

using namespace ns3;
using namespace std;

void netDevicePackets(uint32_t oldValue, uint32_t newValue) {
    std::cout << Simulator::Now().GetNanoSeconds() << ": netDevicePackets: " << newValue << endl;
}

void netDeviceBytes(uint32_t oldValue, uint32_t newValue) {
    std::cout << Simulator::Now().GetNanoSeconds() << ": Bytes in NetDevice Queue: " << newValue << endl;
}

void REDQueueBytes(uint32_t oldValue, uint32_t newValue) {
    std::cout << Simulator::Now().GetNanoSeconds() << ": Bytes in RED Queue: " << newValue << endl;
}

void queueSize(uint32_t oldValue, uint32_t newValue) {
    std::cout << Simulator::Now().GetNanoSeconds() << ": Queue Size Measure: " << newValue << endl;
}

void queueSize2(uint32_t oldValue, uint32_t newValue) {
    std::cout << Simulator::Now().GetNanoSeconds() << ": Queue Size Cross: " << newValue << endl;
}

void dequeue(Ptr< const Packet > packet){
    std::cout << Simulator::Now().GetNanoSeconds() << " Packet dequeued: ";
    packet->Print(std::cout);
    std::cout << endl;
}

void depart(Ptr< const Packet > packet){
    const Ptr<Packet> &pktCopy = packet->Copy();
    PppHeader pppHeader;
    pktCopy->RemoveHeader(pppHeader);
    Ipv4Header ipHeader;
    pktCopy->RemoveHeader(ipHeader);
    if (ipHeader.GetSource() != Ipv4Address("10.3.1.1")) {
        std::cout << Simulator::Now().GetNanoSeconds() << ": Packet Departed: ";
        packet->Print(std::cout);
        std::cout << endl;
    }
}

void drop(Ptr< const Packet > packet){
    if (Simulator::Now().GetNanoSeconds() >= 300000000 && Simulator::Now().GetNanoSeconds() <= 800000000) {
        std::cout << Simulator::Now().GetNanoSeconds() << " Packet dropped: ";
        packet->Print(std::cout);
        std::cout << endl;
    }
}

void MacTxDrop(Ptr< const Packet > packet){
    std::cout << Simulator::Now().GetNanoSeconds() << " Packet MacTxDrop: ";
    packet->Print(std::cout);
    std::cout << endl;
}
void PhyTxDrop(Ptr< const Packet > packet){
    std::cout << Simulator::Now().GetNanoSeconds() << " Packet PhyTxDrop: ";
    packet->Print(std::cout);
    std::cout << endl;
}
void PhyRxDrop(Ptr< const Packet > packet){
    std::cout << Simulator::Now().GetNanoSeconds() << " Packet PhyRxDrop: ";
    packet->Print(std::cout);
    std::cout << endl;
}

void enqueue(Ptr< const Packet > packet){
    std::cout << Simulator::Now().GetNanoSeconds() << " Packet enqueued: ";
    packet->Print(std::cout);
    std::cout << endl;
}

void enqueueDiscA0T2(Ptr< const QueueDiscItem > item){

    Ipv4Header ipHeader = DynamicCast<const Ipv4QueueDiscItem>(item)->GetHeader();
    if (ipHeader.GetSource() != Ipv4Address("10.1.1.1") || ipHeader.GetDestination() != Ipv4Address("10.3.1.1")) {
        return;
    }
    const Ptr<Packet> &pktCopy = item->GetPacket()->Copy();
    TcpHeader tcpHeader;
    pktCopy->PeekHeader(tcpHeader);
    if (tcpHeader.GetSequenceNumber() == SequenceNumber32(1) && tcpHeader.GetAckNumber() != SequenceNumber32(1)) {
        return;
    }
    if (Simulator::Now().GetNanoSeconds() >= 500000000 && Simulator::Now().GetNanoSeconds() <= 800000000) {
        std::cout << Simulator::Now().GetNanoSeconds() << ": Packet enqueued Disc A0T2: ";
        item->Print(std::cout);
        item->GetPacket()->Print(std::cout);
        std::cout << endl;
    }
}

void enqueueDiscA1T2(Ptr< const QueueDiscItem > item){

    Ipv4Header ipHeader = DynamicCast<const Ipv4QueueDiscItem>(item)->GetHeader();
    if (ipHeader.GetSource() != Ipv4Address("10.1.1.1") || ipHeader.GetDestination() != Ipv4Address("10.3.1.1")) {
        return;
    }
    const Ptr<Packet> &pktCopy = item->GetPacket()->Copy();
    TcpHeader tcpHeader;
    pktCopy->PeekHeader(tcpHeader);
    if (tcpHeader.GetSequenceNumber() == SequenceNumber32(1) && tcpHeader.GetAckNumber() != SequenceNumber32(1)) {
        return;
    }
    if (Simulator::Now().GetNanoSeconds() >= 500000000 && Simulator::Now().GetNanoSeconds() <= 800000000) {
        std::cout << Simulator::Now().GetNanoSeconds() << ": Packet enqueued Disc A1T2: ";
        item->Print(std::cout);
        item->GetPacket()->Print(std::cout);
        std::cout << endl;
    }
}

static void
CwndTracer(Ptr<OutputStreamWrapper> stream, uint32_t oldval, uint32_t newval)
{
    *stream->GetStream() << Simulator::Now().GetNanoSeconds() << "," << newval << '\n';
}


void
TraceCwnd(uint32_t nodeId, uint32_t socketId, Ptr<OutputStreamWrapper> stream)
{
    Config::ConnectWithoutContext("/NodeList/" + std::to_string(nodeId) +
                                      "/$ns3::TcpL4Protocol/SocketList/" +
                                      std::to_string(socketId) + "/CongestionWindow",
                                  MakeBoundCallback(&CwndTracer, stream));
}

void QueueSizeTracer(Ptr<RedQueueDisc> redQueue, Ptr<PointToPointNetDevice> netDevice, string name) {
    Simulator::Schedule(Seconds(0.00002), &QueueSizeTracer, redQueue, netDevice, name);
    if (Simulator::Now().GetSeconds() < 0.3) {
        return;
    }
    cout << Simulator::Now().GetNanoSeconds() << " " << name << " redQueue Size + NetDevice Queue Size: " << redQueue->GetNBytes() << " + " << netDevice->GetQueue()->GetNBytes() << " = " << redQueue->GetNBytes() + netDevice->GetQueue()->GetNBytes() << endl;
}

void SetAppMaxSize(Ptr<BulkSendApplication> app) {
    app->SetMaxBytes(20000);
}

double readAvgMsgSize(string traffic) {
    string cdfFile  = "scratch/ECNMC/DCWorkloads/" + traffic + ".txt";
    string line;
    ifstream file(cdfFile);
    if (!file.is_open()) {
        cerr << "Error opening file: " << cdfFile << endl;
        return 0;
    }
    // the first line is the average message size
    getline(file, line);
    istringstream iss(line);
    double avgMsgSize;
    iss >> avgMsgSize;
    file.close();
    return avgMsgSize;
}

double computeTraffciRate(double load, DataRate linkRate, uint32_t avgMsgSize) {
    return load * linkRate.GetBitRate() / 8 / avgMsgSize; // in packets
}

void ScheduleDumpingPackets(Time steadyStartTime,
                            Time steadyStopTime,
                            const vector<E2EMonitor *>& endToendMonitors,
                            AggregatedE2EMonitor* aggregatedE2EMonitor,
                            const vector<PoissonSampler *>& PoissonSamplers,
                            const string& dirName,
                            int experiment) {
    int intervals  = (int)((steadyStopTime - steadyStartTime).GetNanoSeconds() / 100000);
    string outputDirectory = (string)(getenv("PWD")) + "/Results/results_" + dirName +
                             "/" + to_string(experiment);
    for (int i = 1; i < intervals; i++) {
        Time scheduleTime = steadyStartTime + NanoSeconds(i * 100000);
        for (auto monitor: endToendMonitors) {
            string filename = (string) (getenv("PWD")) + "/Results/results_" + dirName + "/" + to_string(experiment)  + "/" + monitor->GetMonitorTag() + "_EndToEnd.csv";
            Simulator::Schedule(scheduleTime, &E2EMonitor::SaveMonitorRecords, monitor, filename);
        }
        for (auto monitor: PoissonSamplers) {
            string filename = (string) (getenv("PWD")) + "/Results/results_" + dirName + "/" + to_string(experiment)  + "/" + monitor->GetMonitorTag() + "_PoissonSampler.csv";
            Simulator::Schedule(scheduleTime, &PoissonSampler::SaveMonitorRecords, monitor, filename);
        }
        if (aggregatedE2EMonitor != nullptr) {
            Simulator::Schedule(scheduleTime,
                                &AggregatedE2EMonitor::SaveMonitorRecords,
                                aggregatedE2EMonitor,
                                outputDirectory);
        }
    }
}

void run_single_queue_simulation(int argc, char* argv[]) {
    auto start = std::chrono::high_resolution_clock::now();
    cout << endl<< "Start Single Queue Simulation" << endl;

    string srcHostToSwitchLinkRate = "53Mbps";         // Links bandwith between src host and switch
    string hostToSwitchLinkDelay = "10us";             // Links delay between src host and switch
    string ctHostToSwitchLinkRate = "53Mbps";          // Links bandwith between cross traffic host and switch
    string bottleneckLinkRate = "10Mbps";              // Links bandwith between switches and dst host
    string duration = "20";                            // Duration of the simulation
    string trafficStartTime = "0";                     // Start time of the traffic
    string trafficStopTime = "20";                     // Stop time of the traffic
    string steadyStartTime = "3";                      // Start time of the steady state
    string steadyStopTime = "10";                      // Stop time of the steady state
    string dirName = "";                               // Directory name for the output files
    string senderTxMaxSize = "1p";                    // Maximum size of the sender's TX buffer
    string switchTXMaxSize = "1p";                     // Maximum size of the switch's TX buffer
    string swtichDstREDQueueDiscMaxSize = "10KB";      // Maximum size of the RED Queue Disc between the switch and the dst host
    string switchSrcREDQueueDiscMaxSize = "6KB";       // Maximum size of the RED Queue Disc between the switch and the src host
    string traffic = "chicago_2010_traffic_10min_2paths/path";  // If the is CAIDA, Merged CAIDA or BulkSend                            
    string probeInterval = "100us";                    // Probe interval for the probe clock at TCP socket 
    double pctPacedBack = 0.0;                         // the percentage of tcp flows of the CAIDA trace to be paced
    bool enableSwitchECN = true;                       // Enable ECN on the switches
    bool enableECMP = true;                            // Enable ECMP on the switches
    double sampleRate = 10;                            // Sample rate for the PoissonSampler
    double minTh = 0.15;                               // RED Queue Disc MinTh in % of maxSize
    double maxTh = 0.45;                               // RED Queue Disc MaxTh in % of maxSize
    int experiment = 1;                                // Experiment number
    double errorRate = 0.005;                          // Silent Packet Drop Error rate
    bool isDifferentating = false;                     // If the simulation is differentating
    double differentiationDelay = 0.35;                // Extra delay for the differentiation
    bool silentPacketDrop = false;                     // If the switch should drop packets silently
    bool Nagle = false;                                // If the Nagle algorithm should be used
    bool activeProbe = false;                          // If the active probe should be used
    bool passiveProbe = true;                          // If the passive probe should be used
    double load = 0.9;                                 // The load on the buttleneck link
    uint16_t poolSize = 30;                            // The size of the connection pool
    double avgMsgSize = 1448.0;                        // The average message size
    double hostTrafficRate = 1000.0;                   // The traffic rate of the measurement traffic
    double ctTrafficRate = 1000.0;                     // The traffic rate of the cross traffic
    int seed = 1;                                      // The seed for the random number generator

    /*command line input*/
    CommandLine cmd;
    cmd.AddValue("srcHostToSwitchLinkRate", "Links bandwith between src host and switch", srcHostToSwitchLinkRate);
    cmd.AddValue("hostToSwitchLinkDelay", "Links delay between src host and switch", hostToSwitchLinkDelay);
    cmd.AddValue("ctHostToSwitchLinkRate", "Links bandwith between cross traffic host and switch", ctHostToSwitchLinkRate);
    cmd.AddValue("bottleneckLinkRate", "Links bandwith between switches and dst host", bottleneckLinkRate);
    cmd.AddValue("enableSwichECN", "Enable ECN on the switches", enableSwitchECN);
    cmd.AddValue("enableECMP", "Enable ECMP on the switches", enableECMP);
    cmd.AddValue("duration", "Duration of the simulation", duration);
    cmd.AddValue("trafficStartTime", "Start time of the traffic", trafficStartTime);
    cmd.AddValue("trafficStopTime", "Stop time of the traffic", trafficStopTime);
    cmd.AddValue("steadyStartTime", "Start time of the steady state for measuring", steadyStartTime);
    cmd.AddValue("steadyStopTime", "Stop time of the steady state for measuring", steadyStopTime);
    cmd.AddValue("pctPacedBack", "the percentage of tcp flows of the CAIDA trace to be paced", pctPacedBack);
    cmd.AddValue("sampleRate", "Sample rate for the PoissonSampler", sampleRate);
    cmd.AddValue("minTh", "RED Queue Disc MinTh in % of maxSize", minTh);
    cmd.AddValue("maxTh", "RED Queue Disc MaxTh in % of maxSize", maxTh);
    cmd.AddValue("experiment", "Experiment number", experiment);
    cmd.AddValue("errorRate", "Silent Packet Drop Error rate", errorRate);
    cmd.AddValue("dirName", "Directory name for the output files", dirName);
    cmd.AddValue("senderTxMaxSize", "Maximum size of the sender's TX buffer", senderTxMaxSize);
    cmd.AddValue("switchTXMaxSize", "Maximum size of the switch's TX buffer", switchTXMaxSize);
    cmd.AddValue("swtichDstREDQueueDiscMaxSize", "Maximum size of the RED Queue Disc between the switch and the dst host", swtichDstREDQueueDiscMaxSize);
    cmd.AddValue("switchSrcREDQueueDiscMaxSize", "Maximum size of the RED Queue Disc between the switch and the src host", switchSrcREDQueueDiscMaxSize);
    cmd.AddValue("traffic", "If the is CAIDA, Merged CAIDA or BulkSend", traffic);
    cmd.AddValue("probeInterval", "Probe interval for the probe clock at TCP socket", probeInterval);
    cmd.AddValue("isDifferentating", "If the simulation is differentating", isDifferentating);
    cmd.AddValue("differentiationDelay", "Extra delay for the differentiation", differentiationDelay); 
    cmd.AddValue("silentPacketDrop", "If the switch should drop packets silently", silentPacketDrop);
    cmd.AddValue("load", "The load on the buttleneck link", load);
    cmd.AddValue("seed", "The seed for the random number generator", seed);
    cmd.AddValue("Nagle", "If the Nagle algorithm should be used", Nagle);
    cmd.AddValue("ActiveProbe", "If the active probe should be used", activeProbe);
    cmd.AddValue("PassiveProbe", "If the passive probe should be used", passiveProbe);
    cmd.Parse(argc, argv);

    /*set default values*/
    ns3::RngSeedManager::SetSeed(seed);
    Time startTime = Seconds(0);
    Time stopTime = Seconds(stof(duration));
    Time convergenceTime = Seconds(0.2);

    Config::SetDefault("ns3::TcpL4Protocol::SocketType", StringValue("ns3::TcpDctcp"));
    Config::SetDefault("ns3::Ipv4GlobalRouting::RandomEcmpRouting", BooleanValue(enableECMP));
    Config::SetDefault("ns3::RedQueueDisc::UseEcn", BooleanValue(enableSwitchECN));
    Config::SetDefault("ns3::CoDelQueueDisc::UseEcn", BooleanValue(false));
    Config::SetDefault("ns3::FqCoDelQueueDisc::UseEcn", BooleanValue(false));
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(1448));
    Config::SetDefault("ns3::TcpSocket::DelAckCount", UintegerValue(1));
    Config::SetDefault("ns3::TcpSocket::SndBufSize", UintegerValue(25000000));
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(25000000));
    Config::SetDefault("ns3::TcpSocket::TcpNoDelay", BooleanValue(!Nagle));
    GlobalValue::Bind("ChecksumEnabled", BooleanValue(false));
    Config::SetDefault("ns3::RedQueueDisc::UseHardDrop", BooleanValue(false));
    Config::SetDefault("ns3::RedQueueDisc::MeanPktSize", UintegerValue(1000));
    Config::SetDefault("ns3::TcpSocketBase::ProbeClockInterval", StringValue(probeInterval));
    // Config::SetDefault("ns3::DropTailQueue<Packet>::MaxSize", QueueSizeValue(QueueSize("10KB")));
    // Config::SetDefault("ns3::RedQueueDisc::MaxSize", QueueSizeValue(QueueSize("1.8MB")));
    // DCTCP tracks instantaneous queue length only; so set QW = 1
    Config::SetDefault("ns3::RedQueueDisc::QW", DoubleValue(1));
    Config::SetDefault("ns3::RedQueueDisc::Gentle", BooleanValue(false));
    Config::SetDefault("ns3::RedQueueDisc::Wait", BooleanValue(false));
    Config::SetDefault("ns3::RedQueueDisc::LInterm", DoubleValue(1));
    Config::SetDefault("ns3::WorkloadApp::ProbeStartTime", TimeValue(Seconds(stof(steadyStartTime))));
    Config::SetDefault("ns3::WorkloadApp::ProbeStopTime", TimeValue(Seconds(stof(steadyStopTime))));
    // Config::SetDefault("ns3::PointToPointNetDevice::Mtu", UintegerValue(300));
    Config::SetDefault("ns3::PointToPointNetDevice::ProbeTrsh", UintegerValue(56));
    if (isDifferentating) {
        Config::SetDefault("ns3::PrioQueueDisc::ErrorRate", DoubleValue(errorRate));
    }
    // DCTCP uses K > 1/7(C * RTT) and minTh = maxTh = K
    // maxTh = minTh = 0.15;
    // Config::SetDefault("ns3::RedQueueDisc::MinTh", DoubleValue(minTh));
    // Config::SetDefault("ns3::RedQueueDisc::MaxTh", DoubleValue(maxTh));

    int nSrcHosts = 6;
    if (activeProbe) {
        nSrcHosts += 1; // one more host for the probe traffic
    }
    int nDstHosts = 1;
    int nSwitches = 1;

    NodeContainer srcHosts;
    NodeContainer dstHosts;
    NodeContainer switches;
    srcHosts.Create(nSrcHosts);
    dstHosts.Create(nDstHosts);
    switches.Create(nSwitches);

    // connecting the hosts to the ToR switches
    vector<NetDeviceContainer> srcHostsToSwitchNetDevices;
    PointToPointHelper p2pSrcHostToSwitch;
    // p2pSrcHostToSwitch.DisableFlowControl();
    p2pSrcHostToSwitch.SetDeviceAttribute("DataRate", StringValue(srcHostToSwitchLinkRate));
    p2pSrcHostToSwitch.SetChannelAttribute("Delay", StringValue(hostToSwitchLinkDelay));
    p2pSrcHostToSwitch.SetQueue("ns3::DropTailQueue<Packet>", "MaxSize", QueueSizeValue(QueueSize(senderTxMaxSize)));

    srcHostsToSwitchNetDevices.push_back(p2pSrcHostToSwitch.Install(srcHosts.Get(0), switches.Get(0)));
    DynamicCast<PointToPointNetDevice>(srcHostsToSwitchNetDevices[0].Get(1))->GetQueue()->SetMaxSize(QueueSize(switchTXMaxSize));

    

    vector<NetDeviceContainer> ctHostsToSwitchNetDevices;
    PointToPointHelper p2pCtHostToSwitch;
    // p2pCtHostToSwitch.DisableFlowControl();
    p2pCtHostToSwitch.SetDeviceAttribute("DataRate", StringValue(ctHostToSwitchLinkRate));
    p2pCtHostToSwitch.SetChannelAttribute("Delay", StringValue(hostToSwitchLinkDelay));
    p2pCtHostToSwitch.SetQueue("ns3::DropTailQueue<Packet>", "MaxSize", QueueSizeValue(QueueSize(senderTxMaxSize)));

    // ctHostsToSwitchNetDevices.push_back(p2pCtHostToSwitch.Install(srcHosts.Get(1), switches.Get(0)));
    // DynamicCast<PointToPointNetDevice>(ctHostsToSwitchNetDevices[0].Get(1))->GetQueue()->SetMaxSize(QueueSize(switchTXMaxSize));
    for (int i = 1; i < nSrcHosts; i++) {
        ctHostsToSwitchNetDevices.push_back(p2pCtHostToSwitch.Install(srcHosts.Get(i), switches.Get(0)));
        DynamicCast<PointToPointNetDevice>(ctHostsToSwitchNetDevices[i - 1].Get(1))->GetQueue()->SetMaxSize(QueueSize(switchTXMaxSize));
    }
    // connecting the probe host to the ToR switch
    PointToPointHelper p2pProbeHostToSwitch;
    NetDeviceContainer probeHostToSwitchNetDevices;
    if (activeProbe) {
        // p2pProbeHostToSwitch.DisableFlowControl();
        p2pProbeHostToSwitch.SetDeviceAttribute("DataRate", StringValue(srcHostToSwitchLinkRate));
        p2pProbeHostToSwitch.SetChannelAttribute("Delay", StringValue(hostToSwitchLinkDelay));
        p2pProbeHostToSwitch.SetQueue("ns3::DropTailQueue<Packet>", "MaxSize", QueueSizeValue(QueueSize(senderTxMaxSize)));

        probeHostToSwitchNetDevices = p2pProbeHostToSwitch.Install(srcHosts.Get(2), switches.Get(0));
        DynamicCast<PointToPointNetDevice>(probeHostToSwitchNetDevices.Get(1))->GetQueue()->SetMaxSize(QueueSize(switchTXMaxSize));
    }

    NetDeviceContainer dstHostsToSwitchNetDevices;
    PointToPointHelper p2pDstHostToSwitch;
    // p2pDstHostToSwitch.DisableFlowControl();
    p2pDstHostToSwitch.SetDeviceAttribute("DataRate", StringValue(bottleneckLinkRate));
    p2pDstHostToSwitch.SetChannelAttribute("Delay", StringValue(hostToSwitchLinkDelay));
    p2pDstHostToSwitch.SetQueue("ns3::DropTailQueue<Packet>", "MaxSize", QueueSizeValue(QueueSize(senderTxMaxSize)));

    dstHostsToSwitchNetDevices = p2pDstHostToSwitch.Install(dstHosts.Get(0), switches.Get(0));
    DynamicCast<PointToPointNetDevice>(dstHostsToSwitchNetDevices.Get(1))->GetQueue()->SetMaxSize(QueueSize(switchTXMaxSize));

    // Install the network stack on the nodes
    InternetStackHelper stack;
    stack.InstallAll();
    // // // Install FqCoDelQueueDisc on the src to switch link
    // TrafficControlHelper srcToSwitchTCH;
    // srcToSwitchTCH.SetRootQueueDisc("ns3::FqCoDelQueueDisc");
    // srcToSwitchTCH.Install(srcHostsToSwitchNetDevices[0].Get(0));
    // Install FifoQueueDisc on all srcs to switch link
    TrafficControlHelper srcToSwitchTCH;
    srcToSwitchTCH.SetRootQueueDisc("ns3::FifoQueueDisc",
                                  "MaxSize", StringValue("10000p"));
    srcToSwitchTCH.Install(srcHostsToSwitchNetDevices[0].Get(0));
    for (int i = 1; i < nSrcHosts; i++) {
        srcToSwitchTCH.Install(ctHostsToSwitchNetDevices[i - 1].Get(0));
    }
    // // Install RED Queue Discs on the switche to src hosts links
    TrafficControlHelper switchToSrcHostTCH;
    switchToSrcHostTCH.SetRootQueueDisc("ns3::RedQueueDisc", 
                                  "LinkBandwidth", StringValue(srcHostToSwitchLinkRate),
                                  "LinkDelay", StringValue(hostToSwitchLinkDelay), 
                                  "MaxSize", StringValue(switchSrcREDQueueDiscMaxSize),
                                  "MinTh", DoubleValue(minTh * QueueSize(switchSrcREDQueueDiscMaxSize).GetValue()),
                                  "MaxTh", DoubleValue(maxTh * QueueSize(switchSrcREDQueueDiscMaxSize).GetValue()));
    vector<QueueDiscContainer> switchToSrcHostQueueDiscs;
    switchToSrcHostQueueDiscs.push_back(switchToSrcHostTCH.Install(srcHostsToSwitchNetDevices[0].Get(1)));
    // switchToSrcHostQueueDiscs.push_back(switchToSrcHostTCH.Install(srcHostsToSwitchNetDevices[0]));
    
    QueueDiscContainer switchToProbeHostQueueDisc;
    if (activeProbe) {
        // Install RED Queue Discs on the switche to probe host links
        TrafficControlHelper switchToProbeHostTCH;
        switchToProbeHostTCH.SetRootQueueDisc("ns3::RedQueueDisc", 
                                      "LinkBandwidth", StringValue(srcHostToSwitchLinkRate),
                                      "LinkDelay", StringValue(hostToSwitchLinkDelay), 
                                      "MaxSize", StringValue(switchSrcREDQueueDiscMaxSize),
                                      "MinTh", DoubleValue(minTh * QueueSize(switchSrcREDQueueDiscMaxSize).GetValue()),
                                      "MaxTh", DoubleValue(maxTh * QueueSize(switchSrcREDQueueDiscMaxSize).GetValue()));
        switchToProbeHostQueueDisc = switchToProbeHostTCH.Install(probeHostToSwitchNetDevices.Get(1));
    }

    // //Install RED Queue Discs on the switches to cross traffic hosts links
    TrafficControlHelper switchToCtHostTCH;
    switchToCtHostTCH.SetRootQueueDisc("ns3::RedQueueDisc", 
                                  "LinkBandwidth", StringValue(ctHostToSwitchLinkRate),
                                  "LinkDelay", StringValue(hostToSwitchLinkDelay), 
                                  "MaxSize", StringValue(switchSrcREDQueueDiscMaxSize),
                                  "MinTh", DoubleValue(minTh * QueueSize(switchSrcREDQueueDiscMaxSize).GetValue()),
                                  "MaxTh", DoubleValue(maxTh * QueueSize(switchSrcREDQueueDiscMaxSize).GetValue()));
    vector<QueueDiscContainer> switchToCtHostQueueDiscs;
    // switchToCtHostQueueDiscs.push_back(switchToCtHostTCH.Install(ctHostsToSwitchNetDevices[0].Get(1)));
    for (int i = 1; i < nSrcHosts; i++) {
        switchToCtHostQueueDiscs.push_back(switchToCtHostTCH.Install(ctHostsToSwitchNetDevices[i - 1].Get(1)));
    }
    // switchToCtHostQueueDiscs.push_back(switchToCtHostTCH.Install(ctHostsToSwitchNetDevices[0]));

    // Install RED Queue Discs on the switches to dst hosts links
    TrafficControlHelper switchToDstHostTCH;
    uint16_t handle = switchToDstHostTCH.SetRootQueueDisc("ns3::PrioQueueDisc", "Priomap", StringValue("0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0"));
    TrafficControlHelper::ClassIdList cid = switchToDstHostTCH.AddQueueDiscClasses(handle, 2, "ns3::QueueDiscClass");
    switchToDstHostTCH.AddChildQueueDisc(handle, cid[0], "ns3::RedQueueDisc", 
                                  "LinkBandwidth", StringValue(bottleneckLinkRate),
                                  "LinkDelay", StringValue(hostToSwitchLinkDelay), 
                                  "MaxSize", StringValue(swtichDstREDQueueDiscMaxSize),
                                  "MinTh", DoubleValue(minTh * QueueSize(swtichDstREDQueueDiscMaxSize).GetValue()),
                                  "MaxTh", DoubleValue(maxTh * QueueSize(swtichDstREDQueueDiscMaxSize).GetValue()));
    switchToDstHostTCH.AddChildQueueDisc(handle, cid[1], "ns3::TbfQueueDisc",
                                    "MaxSize", QueueSizeValue(QueueSize("20p")),
                                    "Burst", UintegerValue(QueueSize("1504B").GetValue()),
                                    "Mtu", UintegerValue(0),
                                    "Rate", DataRateValue(DataRate(bottleneckLinkRate) * 0.5),
                                    "PeakRate", DataRateValue(DataRate("0KBps")));
    // switchToDstHostTCH.SetRootQueueDisc("ns3::RedQueueDisc", 
    //                               "LinkBandwidth", StringValue(bottleneckLinkRate),
    //                               "LinkDelay", StringValue(hostToSwitchLinkDelay), 
    //                               "MaxSize", StringValue(swtichDstREDQueueDiscMaxSize),
    //                               "MinTh", DoubleValue(minTh * QueueSize(swtichDstREDQueueDiscMaxSize).GetValue()),
    //                               "MaxTh", DoubleValue(maxTh * QueueSize(swtichDstREDQueueDiscMaxSize).GetValue()));
    QueueDiscContainer switchToDstHostQueueDisc = switchToDstHostTCH.Install(dstHostsToSwitchNetDevices.Get(1));
    // QueueDiscContainer switchToDstHostQueueDisc = switchToDstHostTCH.Install(dstHostsToSwitchNetDevices);

    // Assign IP addresses
    uint16_t nbSubnet = 0;
    Ipv4AddressHelper address;

    // set the ips between the src hosts and the switch
    vector<Ipv4InterfaceContainer> srcHostsToSwitchIps;
    srcHostsToSwitchIps.reserve(1);
    address.SetBase(("10." + to_string(++nbSubnet) + ".1.0").c_str(), "255.255.255.0");
    srcHostsToSwitchIps.push_back(address.Assign(srcHostsToSwitchNetDevices[0]));
    address.NewNetwork();
    
    // set the ips between the cross traffic hosts and the switch
    vector<Ipv4InterfaceContainer> ctHostsToSwitchIps;
    ctHostsToSwitchIps.reserve(1);
    // address.SetBase(("10." + to_string(++nbSubnet) + ".1.0").c_str(), "255.255.255.0");
    // ctHostsToSwitchIps.push_back(address.Assign(ctHostsToSwitchNetDevices[0]));
    // address.NewNetwork();
    for (int i = 1; i < nSrcHosts; i++) {
        address.SetBase(("10." + to_string(++nbSubnet) + ".1.0").c_str(), "255.255.255.0");
        ctHostsToSwitchIps.push_back(address.Assign(ctHostsToSwitchNetDevices[i - 1]));
        address.NewNetwork();
    }

    // set the ips between the switche and the dst hosts
    address.SetBase(("10." + to_string(++nbSubnet) + ".1.0").c_str(), "255.255.255.0");
    Ipv4InterfaceContainer dstHostsToSwitchIps = address.Assign(dstHostsToSwitchNetDevices);
    
    Ipv4InterfaceContainer probeHostToSwitchIps;
    if (activeProbe) {
        // set the ips between the probe host and the switch
        address.SetBase(("10." + to_string(++nbSubnet) + ".1.0").c_str(), "255.255.255.0");
        probeHostToSwitchIps = address.Assign(probeHostToSwitchNetDevices);
    }

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    // /* Erro Model Setup for Silent packet drops*/
    if (silentPacketDrop) {
        Ptr<RateErrorModel> em_srcToSwtich = CreateObject<RateErrorModel>();
        em_srcToSwtich->SetAttribute("ErrorRate", DoubleValue(errorRate));
        em_srcToSwtich->SetUnit(RateErrorModel::ErrorUnit::ERROR_UNIT_PACKET);
        srcHostsToSwitchNetDevices[0].Get(1)->SetAttribute("ReceiveErrorModel", PointerValue(em_srcToSwtich));   
    }

    // Each src host sends a flow to the dst host
    // for (int i = 0; i < nSrcHosts; i++) {
    //     auto* caidaTrafficGenerator = new BackgroundReplay(srcHosts.Get(i), dstHosts.Get(0), Seconds(stof(trafficStartTime)), Seconds(stof(trafficStopTime)));
    //     caidaTrafficGenerator->SetPctOfPacedTcps(pctPacedBack);
    //     string tracesPath = "/media/experiments/" + traffic + to_string(i % 2);
    //     if (std::filesystem::exists(tracesPath)) {
    //         caidaTrafficGenerator->RunAllTCPTraces(tracesPath, 0);
    //     } else {
    //         cout << "requested Background Directory does not exist" << endl;
    //     }
    // }
    avgMsgSize = readAvgMsgSize(traffic);
    hostTrafficRate = computeTraffciRate(load, DataRate(srcHostToSwitchLinkRate), avgMsgSize);
    ctTrafficRate = computeTraffciRate(load, DataRate(ctHostToSwitchLinkRate), avgMsgSize);
    vector<Ptr<Node>> receivers;
    receivers.push_back(dstHosts.Get(0));
    auto* dcTrafficGenerator = new DCWorkloadGenerator(srcHosts.Get(0), receivers, hostTrafficRate, poolSize, "scratch/ECNMC/DCWorkloads/" + traffic, "ns3::TcpSocketFactory", Time(Seconds(0)), stopTime - Seconds(0.002));
    dcTrafficGenerator->GenrateTraffic(pctPacedBack, passiveProbe, Time(probeInterval), Seconds(0));

    // auto* dcTrafficGeneratorCross = new DCWorkloadGenerator(srcHosts.Get(1), receivers, ctTrafficRate, poolSize, "scratch/ECNMC/DCWorkloads/" + traffic, "ns3::TcpSocketFactory", Time(Seconds(0)), stopTime - Seconds(0.002));
    // dcTrafficGeneratorCross->GenrateTraffic(pctPacedBack, false, Time(probeInterval));
    for (int i = 1; i < nSrcHosts; i++) {
        auto* dcTrafficGeneratorCross = new DCWorkloadGenerator(srcHosts.Get(i), receivers, ctTrafficRate, poolSize, "scratch/ECNMC/DCWorkloads/" + traffic, "ns3::TcpSocketFactory", Time(Seconds(0)), stopTime - Seconds(0.002));
        dcTrafficGeneratorCross->GenrateTraffic(pctPacedBack, false, Time(probeInterval), Seconds(0));
    }


    // Install Probe application
    if (activeProbe) {
        auto* probeGenerator = new ProbeGenerator(srcHosts.Get(2), dstHosts.Get(0), 1 / Time(probeInterval).GetSeconds(), Seconds(stof(steadyStartTime)), Seconds(stof(steadyStopTime)));
        probeGenerator->GenrateTraffic();
    }
    
    // ObjectFactory factory;
    // factory.SetTypeId(NodeAppsHandler::GetTypeId());
    // factory.Set("StartTime", TimeValue(Seconds(0)));
    // factory.Set("StopTime", TimeValue(Seconds(0.2)));
    // Ptr<NodeAppsHandler> nodeAppsHandler = factory.Create<NodeAppsHandler>();
    // srcHosts.Get(0)->AddApplication(nodeAppsHandler);
    // uint16_t portsrc = 50001;
    // BulkSendHelper host("ns3::TcpSocketFactory", InetSocketAddress(dstHostsToSwitchIps.GetAddress(0), portsrc));
    // host.SetAttribute("MaxBytes", UintegerValue(0));
    // // host.SetAttribute("MaxBytes", UintegerValue(10000));
    // // host.SetAttribute("SendSize", UintegerValue(10000));
    // ApplicationContainer sourceApps = host.Install(srcHosts.Get(0));
    // sourceApps.Start(startTime);
    // sourceApps.Stop(stopTime);
    // PacketSinkHelper sinkSrc("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), portsrc));
    // ApplicationContainer sinkSrcApps = sinkSrc.Install(dstHosts.Get(0));
    // sinkSrcApps.Start(startTime);
    // sinkSrcApps.Stop(stopTime);
    
    // AsciiTraceHelper asciiTraceHelper;
    // Ptr<OutputStreamWrapper> stream = asciiTraceHelper.CreateFileStream((string) (getenv("PWD")) + "/Results/results_" + dirName + "/" + to_string(experiment)  + "/50001_cwnd.csv");
    
    // Simulator::Schedule(Seconds(0.0001), &TraceCwnd, 0, 0, stream);
    // // for (int i = 0; i < 2000; i++) {
    // //     Simulator::Schedule(NanoSeconds(i * 500000), &SetAppMaxSize, sourceApps.Get(0)->GetObject<BulkSendApplication>());
    // // }
    // // ct
    // uint16_t portCt = 50005;
    // BulkSendHelper ctHost("ns3::TcpSocketFactory", InetSocketAddress(dstHostsToSwitchIps.GetAddress(0), portCt));
    // // ctHost.SetAttribute("MaxBytes", UintegerValue(10000));
    // ctHost.SetAttribute("MaxBytes", UintegerValue(0));

    // ApplicationContainer ctApps = ctHost.Install(srcHosts.Get(1));
    // ctApps.Start(startTime);
    // ctApps.Stop(stopTime);
    // PacketSinkHelper sinkCt("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), portCt));
    // ApplicationContainer sinkCtApps = sinkCt.Install(dstHosts.Get(0));
    // sinkCtApps.Start(startTime);
    // sinkCtApps.Stop(stopTime);
    // for (int i = 0; i < 2000; i++) {
    //     Simulator::Schedule(NanoSeconds(i * 500000), &SetAppMaxSize, ctApps.Get(0)->GetObject<BulkSendApplication>());
    // }
    // ns3::PacketMetadata::Enable();
    // Monitor the packets between src Host 0 and dst Host 0
    auto *S0D0Monitor = new E2EMonitor(startTime, Seconds(stof(steadyStopTime)) + convergenceTime, Seconds(stof(steadyStartTime)), Seconds(stof(steadyStopTime)), DynamicCast<PointToPointNetDevice>(srcHostsToSwitchNetDevices[0].Get(0)), dstHosts.Get(0), srcHosts.Get(0), "A0D0", errorRate, DataRate(srcHostToSwitchLinkRate), DataRate(bottleneckLinkRate), Time(hostToSwitchLinkDelay), 1, 1, QueueSize(swtichDstREDQueueDiscMaxSize).GetValue(), false, 0);
    S0D0Monitor->AddAppKey(AppKey(srcHostsToSwitchIps[0].GetAddress(0), dstHostsToSwitchIps.GetAddress(0), 0, 0));

    E2EMonitor *P0D0Monitor = nullptr;
    if (activeProbe) {
        // Monitor the packets between probe Host 0 and dst Host 0
        P0D0Monitor = new E2EMonitor(startTime, Seconds(stof(steadyStopTime)) + convergenceTime, Seconds(stof(steadyStartTime)), Seconds(stof(steadyStopTime)), DynamicCast<PointToPointNetDevice>(probeHostToSwitchNetDevices.Get(0)), dstHosts.Get(0), srcHosts.Get(2), "P0D0", errorRate, DataRate(srcHostToSwitchLinkRate), DataRate(bottleneckLinkRate), Time(hostToSwitchLinkDelay), 1, 1, QueueSize(swtichDstREDQueueDiscMaxSize).GetValue(), false, 0);
        P0D0Monitor->AddAppKey(AppKey(probeHostToSwitchIps.GetAddress(0), dstHostsToSwitchIps.GetAddress(0), 0, 0));
    }
    // auto *C0D0Monitor = new E2EMonitor(startTime, Seconds(stof(steadyStopTime)) + convergenceTime, Seconds(stof(steadyStartTime)), Seconds(stof(steadyStopTime)), DynamicCast<PointToPointNetDevice>(ctHostsToSwitchNetDevices[0].Get(0)), dstHosts.Get(0), srcHosts.Get(1), "C0D0", errorRate, DataRate(ctHostToSwitchLinkRate), DataRate(bottleneckLinkRate), Time(hostToSwitchLinkDelay), 1, 1, QueueSize(swtichDstREDQueueDiscMaxSize).GetValue(), isDifferentating, differentiationDelay);
    // C0D0Monitor->AddAppKey(AppKey(ctHostsToSwitchIps[0].GetAddress(0), dstHostsToSwitchIps.GetAddress(0), 0, 0));
    // Ptr<PointToPointNetDevice> hostToSwitchrNetDevice = DynamicCast<PointToPointNetDevice>(srcHostsToSwitchNetDevices[0].Get(0));
    // auto *hostToSwitchrSampler = new PoissonSampler(Seconds(stof(steadyStartTime)), Seconds(stof(steadyStopTime)), nullptr, hostToSwitchrNetDevice->GetQueue(), hostToSwitchrNetDevice, "H", sampleRate);

    Ptr<PointToPointNetDevice> switchToDstNetDevice = DynamicCast<PointToPointNetDevice>(dstHostsToSwitchNetDevices.Get(1));
    Ptr<PointToPointNetDevice> incomingNetDevice = DynamicCast<PointToPointNetDevice>(srcHostsToSwitchNetDevices[0].Get(1));
    Ptr<PointToPointNetDevice> incomingNetDevice_1 = DynamicCast<PointToPointNetDevice>(ctHostsToSwitchNetDevices[0].Get(1));
    auto *switchToDstSampler = new PoissonSampler(Seconds(stof(steadyStartTime)), Seconds(stof(steadyStopTime)), DynamicCast<RedQueueDisc>(switchToDstHostQueueDisc.Get(0)->GetQueueDiscClass(0)->GetQueueDisc()), switchToDstNetDevice->GetQueue(), switchToDstNetDevice, "SD0", sampleRate, incomingNetDevice, incomingNetDevice_1, traffic);
    // auto *switchToDstSampler = new PoissonSampler(Seconds(stof(steadyStartTime)), Seconds(stof(steadyStopTime)), DynamicCast<RedQueueDisc>(switchToDstHostQueueDisc.Get(0)), switchToDstNetDevice->GetQueue(), switchToDstNetDevice, "SD0", sampleRate);
    // auto *switchToDstSampler = new PoissonSampler(Seconds(stof(steadyStartTime)), Seconds(stof(steadyStopTime)), nullptr, switchToDstNetDevice->GetQueue(), switchToDstNetDevice, "SD0", sampleRate);

    // auto *switchMonitor = new SwitchMonitor(startTime, Seconds(stof(steadyStopTime)) + convergenceTime, Seconds(stof(steadyStartTime)), Seconds(stof(steadyStopTime)), switches.Get(0), "S0");
    // switchMonitor->AddAppKey(AppKey(srcHostsToSwitchIps[0].GetAddress(0), dstHostsToSwitchIps.GetAddress(0), 0, 0));
    // switchMonitor->AddAppKey(AppKey(ctHostsToSwitchIps[0].GetAddress(0), dstHostsToSwitchIps.GetAddress(0), 0, 0));

    // Simulator::Schedule(Seconds(0.00002), &QueueSizeTracer, DynamicCast<RedQueueDisc>(switchToDstHostQueueDisc.Get(0)), switchToDstNetDevice, "Switch");
    // Simulator::Schedule(Seconds(0.00002), &QueueSizeTracer, DynamicCast<RedQueueDisc>(switchToSrcHostQueueDiscs[0].Get(0)), DynamicCast<PointToPointNetDevice>(srcHostsToSwitchNetDevices[0].Get(0)), "Sender");
    cout << "Sender Tx Queue Size: " << DynamicCast<PointToPointNetDevice>(srcHostsToSwitchNetDevices[0].Get(0))->GetQueue()->GetMaxSize().GetValue() << endl;
    cout << "Switch Tx Queue Size: " << DynamicCast<PointToPointNetDevice>(srcHostsToSwitchNetDevices[0].Get(1))->GetQueue()->GetMaxSize().GetValue() << endl;

    cout << "Hosts and Switches IP addresses" << endl;
    cout << "Src: " << 0 << " Id:" << srcHosts.Get(0)->GetId() << " IP: " << srcHostsToSwitchIps[0].GetAddress(0) << endl;
    cout << "Src CTs: " << endl;
    for (int i = 1; i < nSrcHosts; i++) {
        cout << "Src CT: " << i << " Id:" << srcHosts.Get(i)->GetId() << " IP: " << ctHostsToSwitchIps[i - 1].GetAddress(0) << endl;
    }
    if (activeProbe) {
        cout << "Probe: " << 2 << " Id:" << srcHosts.Get(2)->GetId() << " IP: " << probeHostToSwitchIps.GetAddress(0) << endl;
    }
    cout << "Dst: " << 0 << " Id:" << dstHosts.Get(0)->GetId() << " IP: " << dstHostsToSwitchIps.GetAddress(0) << endl;
    //print config parameters
    auto t = std::chrono::high_resolution_clock::now();
    cout << "Total preparing time = " << std::chrono::duration_cast<std::chrono::microseconds>(t - start).count() << " microsecond" << endl;
    cout << "Config Parameters" << endl;
    cout << "srcHostToSwitchLinkRate: " << srcHostToSwitchLinkRate << endl;
    cout << "ctHostToSwitchLinkRate: " << ctHostToSwitchLinkRate << endl;
    cout << "hostToSwitchLinkDelay: " << hostToSwitchLinkDelay << endl;
    cout << "bottleneckLinkRate: " << bottleneckLinkRate << endl;
    cout << "pctPacedBack: " << pctPacedBack << endl;
    cout << "probeInterval: " << probeInterval << endl;
    cout << "enableSwitchECN: " << enableSwitchECN << endl;
    cout << "enableECMP: " << enableECMP << endl;
    cout << "sampleRate: " << sampleRate << endl;
    cout << "errorRate: " << errorRate << endl;
    cout << "dirName: " << dirName << endl;
    cout << "experiment: " << experiment << endl;
    cout << "trafficStartTime: " << trafficStartTime << endl;
    cout << "trafficStopTime: " << trafficStopTime << endl;
    cout << "steadyStartTime: " << steadyStartTime << endl;
    cout << "steadyEndTime: " << steadyStopTime << endl;
    cout << "duration: " << duration << endl;
    cout << "swtichDstREDQueueDiscMaxSize: " << swtichDstREDQueueDiscMaxSize << endl;
    cout << "switchSrcREDQueueDiscMaxSize: " << switchSrcREDQueueDiscMaxSize << endl;
    cout << "minTh: " << minTh << endl;
    cout << "maxTh: " << maxTh << endl;
    cout << "traffic: " << traffic << endl;
    cout << "isDifferentating: " << isDifferentating << endl;
    cout << "differentiationDelay: " << differentiationDelay << endl;
    cout << "silentPacketDrop: " << silentPacketDrop << endl;
    cout << "load: " << load << endl;
    cout << "Average Message Size: " << avgMsgSize << endl;
    cout << "Measurement Traffic Rate: " << hostTrafficRate << endl;
    cout << "Cross Traffic Rate: " << ctTrafficRate << endl;
    cout << "Sender Nagle: " << Nagle << endl;
    cout << "Sender ActiveProbe: " << activeProbe << endl;
    cout << "Sender PassiveProbe: " << passiveProbe << endl;
    cout << "Seed: " << seed << endl;
    // /* ########## END: Check Config ########## */


    // /* ########## START: Scheduling and  Running ########## */

    Simulator::Stop(stopTime);
    Simulator::Run();
    Simulator::Destroy();

    S0D0Monitor->SaveMonitorRecords((string) (getenv("PWD")) + "/Results/results_" + dirName + "/" + to_string(experiment)  + "/" + S0D0Monitor->GetMonitorTag() + "_EndToEnd.csv");
    if (activeProbe) {
        P0D0Monitor->SaveMonitorRecords((string) (getenv("PWD")) + "/Results/results_" + dirName + "/" + to_string(experiment)  + "/" + P0D0Monitor->GetMonitorTag() + "_EndToEnd.csv");
    }
    // C0D0Monitor->SaveMonitorRecords((string) (getenv("PWD")) + "/Results/results_" + dirName + "/" + to_string(experiment)  + "/" + C0D0Monitor->GetMonitorTag() + "_EndToEnd.csv");
    // switchMonitor->SavePacketRecords((string) (getenv("PWD")) + "/Results/results_" + dirName + "/" + to_string(experiment)  + "/" + switchMonitor->GetMonitorTag() + "_Switch.csv");
    // hostToSwitchrSampler->SaveMonitorRecords((string) (getenv("PWD")) + "/Results/results_" + dirName + "/" + to_string(experiment)  + "/" + hostToSwitchrSampler->GetMonitorTag() + "_PoissonSampler.csv");
    switchToDstSampler->SaveMonitorRecords((string) (getenv("PWD")) + "/Results/results_" + dirName + "/" + to_string(experiment)  + "/" + switchToDstSampler->GetMonitorTag() + "_PoissonSampler.csv");
    
    /* ########## END: Scheduling and  Running ########## */

    cout << "Done " << endl;
    auto stop = std::chrono::high_resolution_clock::now();
    cout << "Total execution time = " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " microsecond" << endl;
}

void run_DC_simulation(int argc, char* argv[]){
    auto start = std::chrono::high_resolution_clock::now();
    cout << endl<< "Start Two Tier Datacenter Simulation" << endl;
    /* ########## START: Config ########## */
    string hostToTorLinkRate = "53Mbps";               // Links bandwith between hosts and ToR switches
    string hostToTorLinkRateCrossTraffic = "53Mbps";   // Links bandwith between hosts and ToR switches for the cross traffic
    string hostToTorLinkDelay = "10us";                // Links delay between hosts and ToR switches
    string torToAggLinkRate = "10Mbps";                // Links bandwith between ToR and Agg switches
    string torToAggLinkDelay = "10us";                 // Links delay between ToR and Agg switches
    string aggToCoreLinkRate = "10Mbps";               // Links bandwith between Agg and Core switches
    string aggToCoreLinkDelay = "10us";                // Links delay between Agg and Core switches
    string duration = "20";                            // Duration of the simulation
    string trafficStartTime = "0";                     // Start time of the traffic
    string trafficStopTime = "20";                     // Stop time of the traffic
    string steadyStartTime = "3";                      // Start time of the steady state
    string steadyStopTime = "10";                      // Stop time of the steady state
    string dirName = "";                               // Directory name for the output files
    string senderTxMaxSize = "1p";                     // Maximum size of the sender's TX buffer
    string switchTXMaxSize = "1p";                     // Maximum size of the switch's TX buffer
    string switchSrcREDQueueDiscMaxSize = "15KB";      // Maximum size of the switch's RED queue disc to the src hosts
    string switchREDQueueDiscMaxSize = "90KB";         // Maximum size of the switch's RED queue disc to the dst hosts
    string traffic = "chicago_2010_traffic_10min_2paths/path";  // If the is CAIDA, Merged CAIDA or BulkSend                            
    string probeInterval = "100us";                    // Probe interval for the probe clock at TCP socket 
    string incastperiod = "500us";                     // Incast period
    double pctPacedBack = 0.0;                         // the percentage of tcp flows of the CAIDA trace to be paced
    bool enableSwitchECN = true;                       // Enable ECN on the switches
    bool enableECMP = true;                            // Enable ECMP on the switches
    double sampleRate = 10;                            // Sample rate for the PoissonSampler
    double minTh = 0.15;                               // RED Queue Disc MinTh in % of maxSize
    double maxTh = 0.45;                               // RED Queue Disc MaxTh in % of maxSize
    int experiment = 1;                                // Experiment number
    double errorRate = 0.005;                          // Silent Packet Drop Error rate
    bool isDifferentating = false;                     // If the simulation is differentating
    string differentiationDelay = "35ns";                // Extra delay for the differentiation
    bool silentPacketDrop = false;                     // If the switch should drop packets silently
    bool Nagle = false;                                // If the Nagle algorithm should be used
    bool activeProbe = false;                          // If the active probe should be used
    bool passiveProbe = true;                          // If the passive probe should be used
    bool monitorAllFlows = false;                      // Capture all source-destination host pairs using aggregated monitors
    double load = 0.9;                                 // The load on the buttleneck link
    uint16_t poolSize = 20;                            // The size of the connection pool
    double avgMsgSize = 1448.0;                        // The average message size
    double hostTrafficRate = 1000.0;                   // The traffic rate of the measurement traffic
    uint32_t incastMessageSize = 10000;                // The size of the incast messages
    uint16_t incastFactor = 10;                        // The incast factor
    int seed = 1;                                      // The seed for the random number generator
    int nHosts = 6;                                   // Hosts per rack
    int nRacks = 4;                                    // Number of ToR racks
    int nAggSwitches = 2;                              // Number of aggregation switches
    int nCoreSwitches = 1;                             // Number of core switches
    int tbfSrcRack = 0;                                // Rack of the source host whose flows are eligible for shaping at T0's ingress
    int tbfSrcHost = 0;                                // Host (within tbfSrcRack) whose flows are eligible for shaping at T0's ingress
    int tbfDstRack = 2;                                // Rack of the destination host whose flows are eligible for shaping at T0's ingress
    int tbfDstHost = 3;                                // Host (within tbfDstRack) whose flows are eligible for shaping at T0's ingress
    double tbfFlowRedirectFraction = 0.0;              // Fraction of the tbfSrcHost->tbfDstHost TCP flows delayed by the token bucket at T0's ingress before reaching RED
    string tbfRate = "5Mbps";                          // Token bucket fill rate for shaped flows at T0's ingress
    string tbfBurst = "1504B";                         // Token bucket burst size for shaped flows at T0's ingress

    /*command line input*/
    CommandLine cmd;
    cmd.AddValue("hostToTorLinkRate", "Links bandwith between hosts and ToR switches", hostToTorLinkRate);
    cmd.AddValue("hostToTorLinkDelay", "Links delay between hosts and ToR switches", hostToTorLinkDelay);
    cmd.AddValue("torToAggLinkRate", "Links bandwith between ToR and Agg switches", torToAggLinkRate);
    cmd.AddValue("torToAggLinkDelay", "Links delay between ToR and Agg switches", torToAggLinkDelay);
    cmd.AddValue("aggToCoreLinkRate", "Links bandwith between Agg and Core switches", aggToCoreLinkRate);
    cmd.AddValue("aggToCoreLinkDelay", "Links delay between Agg and Core switches", aggToCoreLinkDelay);
    cmd.AddValue("enableSwichECN", "Enable ECN on the switches", enableSwitchECN);
    cmd.AddValue("enableECMP", "Enable ECMP on the switches", enableECMP);
    cmd.AddValue("duration", "Duration of the simulation", duration);
    cmd.AddValue("trafficStartTime", "Start time of the traffic", trafficStartTime);
    cmd.AddValue("trafficStopTime", "Stop time of the traffic", trafficStopTime);
    cmd.AddValue("steadyStartTime", "Start time of the steady state for measuring", steadyStartTime);
    cmd.AddValue("steadyStopTime", "Stop time of the steady state for measuring", steadyStopTime);
    cmd.AddValue("pctPacedBack", "the percentage of tcp flows of the CAIDA trace to be paced", pctPacedBack);
    cmd.AddValue("sampleRate", "Sample rate for the PoissonSampler", sampleRate);
    cmd.AddValue("hostToTorLinkRateCrossTraffic", "Links bandwith between hosts and ToR switches for the cross traffic", hostToTorLinkRateCrossTraffic);
    cmd.AddValue("minTh", "RED Queue Disc MinTh in % of maxSize", minTh);
    cmd.AddValue("maxTh", "RED Queue Disc MaxTh in % of maxSize", maxTh);
    cmd.AddValue("experiment", "Experiment number", experiment);
    cmd.AddValue("errorRate", "Silent Packet Drop Error rate", errorRate);
    cmd.AddValue("dirName", "Directory name for the output files", dirName);
    cmd.AddValue("senderTxMaxSize", "Maximum size of the sender's TX buffer", senderTxMaxSize);
    cmd.AddValue("switchTXMaxSize", "Maximum size of the switch's TX buffer", switchTXMaxSize);
    cmd.AddValue("switchREDQueueDiscMaxSize", "Maximum size of the switch's RED queue disc", switchREDQueueDiscMaxSize);
    cmd.AddValue("switchSrcREDQueueDiscMaxSize", "Maximum size of the switch's RED queue disc to the src hosts", switchSrcREDQueueDiscMaxSize);
    cmd.AddValue("traffic", "If the is CAIDA, Merged CAIDA or BulkSend", traffic);
    cmd.AddValue("probeInterval", "Probe interval for the probe clock at TCP socket", probeInterval);
    cmd.AddValue("isDifferentating", "If the simulation is differentating", isDifferentating);
    cmd.AddValue("differentiationDelay", "Extra delay for the differentiation", differentiationDelay); 
    cmd.AddValue("silentPacketDrop", "If the switch should drop packets silently", silentPacketDrop);
    cmd.AddValue("load", "The load on the buttleneck link", load);
    cmd.AddValue("seed", "The seed for the random number generator", seed);
    cmd.AddValue("Nagle", "If the Nagle algorithm should be used", Nagle);
    cmd.AddValue("ActiveProbe", "If the active probe should be used", activeProbe);
    cmd.AddValue("PassiveProbe", "If the passive probe should be used", passiveProbe);
    cmd.AddValue("monitorAllFlows", "Capture all source-destination host pairs using aggregated monitors", monitorAllFlows);
    cmd.AddValue("incastperiod", "Incast period", incastperiod);
    cmd.AddValue("incastMessageSize", "The size of the incast messages", incastMessageSize);
    cmd.AddValue("incastFactor", "The incast factor", incastFactor);
    cmd.AddValue("nHosts", "Number of hosts per rack", nHosts);
    cmd.AddValue("nRacks", "Number of racks", nRacks);
    cmd.AddValue("nAggSwitches", "Number of aggregation switches", nAggSwitches);
    cmd.AddValue("nCoreSwitches", "Number of core switches", nCoreSwitches);
    cmd.AddValue("tbfSrcRack", "Rack of the source host whose flows are eligible for shaping at T0's ingress", tbfSrcRack);
    cmd.AddValue("tbfSrcHost", "Host (within tbfSrcRack) whose flows are eligible for shaping at T0's ingress", tbfSrcHost);
    cmd.AddValue("tbfDstRack", "Rack of the destination host whose flows are eligible for shaping at T0's ingress", tbfDstRack);
    cmd.AddValue("tbfDstHost", "Host (within tbfDstRack) whose flows are eligible for shaping at T0's ingress", tbfDstHost);
    cmd.AddValue("tbfFlowRedirectFraction", "Fraction of the tbfSrcHost->tbfDstHost TCP flows delayed by the token bucket at T0's ingress before reaching RED", tbfFlowRedirectFraction);
    cmd.AddValue("tbfRate", "Token bucket fill rate for shaped flows at T0's ingress", tbfRate);
    cmd.AddValue("tbfBurst", "Token bucket burst size for shaped flows at T0's ingress", tbfBurst);
    cmd.Parse(argc, argv);

    /*set default values*/
    ns3::RngSeedManager::SetSeed(experiment);
    Time startTime = Seconds(0);
    Time stopTime = Seconds(stof(duration));
    Time convergenceTime = Seconds(0.0002);

    Config::SetDefault("ns3::TcpL4Protocol::SocketType", StringValue("ns3::TcpDctcp"));
    Config::SetDefault("ns3::Ipv4GlobalRouting::RandomEcmpRouting", BooleanValue(enableECMP));
    Config::SetDefault("ns3::RedQueueDisc::UseEcn", BooleanValue(enableSwitchECN));
    Config::SetDefault("ns3::CoDelQueueDisc::UseEcn", BooleanValue(false));
    Config::SetDefault("ns3::FqCoDelQueueDisc::UseEcn", BooleanValue(false));
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(1448));
    Config::SetDefault("ns3::TcpSocket::DelAckCount", UintegerValue(1));
    Config::SetDefault("ns3::TcpSocket::SndBufSize", UintegerValue(25000000));
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(25000000));
    Config::SetDefault("ns3::TcpSocket::TcpNoDelay", BooleanValue(!Nagle));
    Config::SetDefault("ns3::TcpSocket::InitialCwnd", UintegerValue(66));
    Config::SetDefault("ns3::TcpSocket::ConnTimeout", TimeValue(Seconds(0.0002)));
    Config::SetDefault("ns3::TcpSocketBase::MinRto", TimeValue(Seconds(0.0002)));
    GlobalValue::Bind("ChecksumEnabled", BooleanValue(false));
    Config::SetDefault("ns3::RedQueueDisc::UseHardDrop", BooleanValue(false));
    Config::SetDefault("ns3::RedQueueDisc::MeanPktSize", UintegerValue(1500));
    Config::SetDefault("ns3::TcpSocketBase::ProbeClockInterval", StringValue(probeInterval));
    // Config::SetDefault("ns3::RedQueueDisc::MaxSize", QueueSizeValue(QueueSize("37.5KB")));
    // Config::SetDefault("ns3::RedQueueDisc::MaxSize", QueueSizeValue(QueueSize("1.8MB")));
    // Config::SetDefault("ns3::RedQueueDisc::MaxSize", QueueSizeValue(QueueSize("250KB")));
    Config::SetDefault("ns3::RedQueueDisc::QW", DoubleValue(1));
    Config::SetDefault("ns3::RedQueueDisc::Gentle", BooleanValue(false));
    Config::SetDefault("ns3::RedQueueDisc::Wait", BooleanValue(false));
    Config::SetDefault("ns3::RedQueueDisc::LInterm", DoubleValue(1));
    Config::SetDefault("ns3::WorkloadApp::ProbeStartTime", TimeValue(Seconds(stof(steadyStartTime))));
    Config::SetDefault("ns3::WorkloadApp::ProbeStopTime", TimeValue(Seconds(stof(steadyStopTime))));
    Config::SetDefault("ns3::PointToPointNetDevice::ProbeTrsh", UintegerValue(56));
    // if (isDifferentating) {
    //     Config::SetDefault("ns3::PrioQueueDisc::ErrorRate", DoubleValue(errorRate));
    // }
    /* ########## END: Config ########## */



    /* ########## START: Ceating the topology ########## */
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
    NodeContainer activeProbeHost;
    if (activeProbe) {
        activeProbeHost.Create(1);
    }
    // connecting the hosts to the ToR switches
    vector<vector<NetDeviceContainer>> hostsToTorsNetDevices;

    PointToPointHelper p2pHostToTor;
    p2pHostToTor.SetDeviceAttribute("DataRate", StringValue(hostToTorLinkRate));
    p2pHostToTor.SetChannelAttribute("Delay", StringValue(hostToTorLinkDelay));
    p2pHostToTor.SetQueue("ns3::DropTailQueue<Packet>", "MaxSize", QueueSizeValue(QueueSize(senderTxMaxSize)));

    for (int i = 0; i < nRacks; i++) {
        vector<NetDeviceContainer> hostsToTors;
        for (int j = 0; j < nHosts; j++) {
            hostsToTors.push_back(p2pHostToTor.Install(racks[i].Get(j), torSwitches.Get(i)));
            DynamicCast<PointToPointNetDevice>(hostsToTors.back().Get(1))->GetQueue()->SetMaxSize(QueueSize(switchTXMaxSize));
            // if we are on the first rack and active probe is enabled, connect the active probe host to the ToR switch
            if (i == 0 && j == nHosts - 1 && activeProbe) {
                hostsToTors.push_back(p2pHostToTor.Install(activeProbeHost.Get(0), torSwitches.Get(i)));
                DynamicCast<PointToPointNetDevice>(hostsToTors.back().Get(1))->GetQueue()->SetMaxSize(QueueSize(switchTXMaxSize));
            }
        }
        hostsToTorsNetDevices.push_back(hostsToTors);
    }

    // connecting the Tor Switches to the Agg Switches
    vector<vector<NetDeviceContainer>> torToAggNetDevices;
    PointToPointHelper p2pTorToAgg;
    p2pTorToAgg.SetDeviceAttribute("DataRate", StringValue(torToAggLinkRate));
    p2pTorToAgg.SetChannelAttribute("Delay", StringValue(torToAggLinkDelay));
    p2pTorToAgg.SetQueue("ns3::DropTailQueue<Packet>", "MaxSize", QueueSizeValue(QueueSize(switchTXMaxSize)));

    for (int i = 0; i < nRacks; i++) {
        vector<NetDeviceContainer> torToAgg;
        for (int j = 0; j < nAggSwitches; j++) {
            torToAgg.push_back(p2pTorToAgg.Install(torSwitches.Get(i), aggSwitches.Get(j)));
        }
        torToAggNetDevices.push_back(torToAgg);        
    }

    // connecting the agg switches to the core switches
    vector<vector<NetDeviceContainer>> aggToCoreNetDevices;
    PointToPointHelper p2pAggToCore;
    p2pAggToCore.SetDeviceAttribute("DataRate", StringValue(aggToCoreLinkRate));
    p2pAggToCore.SetChannelAttribute("Delay", StringValue(aggToCoreLinkDelay));
    p2pAggToCore.SetQueue("ns3::DropTailQueue<Packet>", "MaxSize", QueueSizeValue(QueueSize(switchTXMaxSize)));

    for (int i = 0; i < nAggSwitches; i++) {
        vector<NetDeviceContainer> aggToCore;
        for (int j = 0; j < nCoreSwitches; j++) {
            aggToCore.push_back(p2pAggToCore.Install(aggSwitches.Get(i), coreSwitches.Get(j)));
        }
        aggToCoreNetDevices.push_back(aggToCore);
    }

    // Pre-aggregate a ShapingTrafficControlLayer onto T0 (in place of the plain
    // TrafficControlLayer InternetStackHelper would otherwise create for it) so that a
    // configurable fraction of one specific source-destination pair's TCP flows are delayed by a
    // token-bucket shaper at T0's ingress -- before routing/TrafficControl ever sees them --
    // while every other flow (and node) is unaffected. See queue_discs/ShapingTrafficControlLayer.h.
    Ptr<ShapingTrafficControlLayer> t0ShapingLayer = CreateObject<ShapingTrafficControlLayer>();
    t0ShapingLayer->SetAttribute("ShapingRate", DataRateValue(DataRate(tbfRate)));
    t0ShapingLayer->SetAttribute("ShapingBurst", UintegerValue(QueueSize(tbfBurst).GetValue()));
    t0ShapingLayer->SetAttribute("ShapedPacketsLogFile", StringValue((string) (getenv("PWD")) + "/Results/results_" + dirName + "/" + to_string(experiment) + "/T0_TBF_shaped_packets.csv"));
    torSwitches.Get(0)->AggregateObject(t0ShapingLayer);

    // Install the network stack on the nodes
    InternetStackHelper stack;
    stack.InstallAll();

    // // Install FifoQueueDisc on all srcs to switch link
    // TrafficControlHelper srcToSwitchTCH;
    // srcToSwitchTCH.SetRootQueueDisc("ns3::FifoQueueDisc",
    //                               "MaxSize", StringValue("10000p"));
    // for (int i = 0; i < nRacks; i++) {
    //     for (int j = 0; j < nHosts; j++) {
    //         srcToSwitchTCH.Install(hostsToTorsNetDevices[i][j].Get(0));
    //     }
    // }
    // Install RED Queue Discs on the ToR switches, on ToR to Host links
    TrafficControlHelper torToHostTCH;
    torToHostTCH.SetRootQueueDisc("ns3::RedQueueDisc", 
                                  "LinkBandwidth", StringValue(hostToTorLinkRate),
                                  "LinkDelay", StringValue(hostToTorLinkDelay), 
                                  "MaxSize", QueueSizeValue(QueueSize(switchSrcREDQueueDiscMaxSize)),
                                  "MinTh", DoubleValue(minTh * QueueSize(switchSrcREDQueueDiscMaxSize).GetValue()),
                                  "MaxTh", DoubleValue(maxTh * QueueSize(switchSrcREDQueueDiscMaxSize).GetValue()));
    vector<vector<QueueDiscContainer>> torToHostQueueDiscs;
    for (int i = 0; i < nRacks; i++) {
        vector<QueueDiscContainer> qdiscs;
        for (int j = 0; j < nHosts; j++) {
            qdiscs.push_back(torToHostTCH.Install(hostsToTorsNetDevices[i][j].Get(1)));
            if (i == 0 && j == nHosts - 1 && activeProbe) {
                qdiscs.push_back(torToHostTCH.Install(hostsToTorsNetDevices[i][j + 1].Get(1)));
            }
        }
        torToHostQueueDiscs.push_back(qdiscs);
    }

    // Install RED Queue Discs on the ToR switches, on ToR to Agg links and Agg to ToR links
    TrafficControlHelper torToAggTCH;
    torToAggTCH.SetRootQueueDisc("ns3::RedQueueDisc", 
                                  "LinkBandwidth", StringValue(torToAggLinkRate),
                                  "LinkDelay", StringValue(torToAggLinkDelay), 
                                  "MaxSize", QueueSizeValue(QueueSize(switchREDQueueDiscMaxSize)),
                                  "MinTh", DoubleValue(minTh * QueueSize(switchREDQueueDiscMaxSize).GetValue()),
                                  "MaxTh", DoubleValue(maxTh * QueueSize(switchREDQueueDiscMaxSize).GetValue()));
    // Every ToR<->Agg link (including T0<->A0) uses the plain RED queue disc, unmodified. Traffic
    // differentiation for the T0->A0 path is applied earlier, at T0's ingress from the source
    // host (see ShapingTrafficControlLayer below), not here -- so that shaped and unshaped
    // packets genuinely share this same RED instance's state.
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
                                  "MaxSize", QueueSizeValue(QueueSize(switchREDQueueDiscMaxSize)),
                                  "MinTh", DoubleValue(minTh * QueueSize(switchREDQueueDiscMaxSize).GetValue()),
                                  "MaxTh", DoubleValue(maxTh * QueueSize(switchREDQueueDiscMaxSize).GetValue()));
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
            if (i == 0 && j == nHosts - 1 && activeProbe) {
                ips.push_back(address.Assign(hostsToTorsNetDevices[i][j + 1]));
                address.NewNetwork();
            }
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

    // Configure the T0->A0 flow-redirect queue disc now that host addresses are assigned.
    t0ShapingLayer->SetAttribute("FlowRedirectSrcAddress", Ipv4AddressValue(ipsRacks[tbfSrcRack][tbfSrcHost].GetAddress(0)));
    t0ShapingLayer->SetAttribute("FlowRedirectDstAddress", Ipv4AddressValue(ipsRacks[tbfDstRack][tbfDstHost].GetAddress(0)));
    // isDifferentating is the master switch for traffic differentiation; tbfFlowRedirectFraction
    // only takes effect when it is enabled (mirrors the old isDifferentating-gated mechanism above).
    t0ShapingLayer->SetAttribute("FlowRedirectFraction", DoubleValue(isDifferentating ? tbfFlowRedirectFraction : 0.0));

    // /* Erro Model Setup for Silent packet drops*/
    if (silentPacketDrop) {
        Ptr<RateErrorModel> em_R0H0T0 = CreateObject<RateErrorModel>();
        em_R0H0T0->SetAttribute("ErrorRate", DoubleValue(errorRate));
        em_R0H0T0->SetUnit(RateErrorModel::ErrorUnit::ERROR_UNIT_PACKET);
        hostsToTorsNetDevices[0][0].Get(1)->SetAttribute("ReceiveErrorModel", PointerValue(em_R0H0T0));

        Ptr<RateErrorModel> em_R0H1T0 = CreateObject<RateErrorModel>();
        em_R0H1T0->SetAttribute("ErrorRate", DoubleValue(errorRate));
        em_R0H1T0->SetUnit(RateErrorModel::ErrorUnit::ERROR_UNIT_PACKET);
        hostsToTorsNetDevices[0][1].Get(1)->SetAttribute("ReceiveErrorModel", PointerValue(em_R0H1T0));

        cout << "Silent Packet Drop Error Models are set up." << endl;
    }
    // ****** Mahdi Change (flow redirect) ***** (START) ***** //
    // isDifferentating now gates the TBF-based traffic differentiation at T0's ingress
    // (ShapingTrafficControlLayer, configured below) instead of this old per-device interframe-gap
    // mechanism. Kept here, commented out, only as a record of the previous implementation.
    // if (isDifferentating) {
    //     // Set the interframe gap mean for point-to-point net device of R2H3
    //     cout << "Differentation is enabled. Setting extra delay of " << Time(differentiationDelay).GetNanoSeconds() << "ns on R2H3" << endl;
    //     DynamicCast<PointToPointNetDevice>(hostsToTorsNetDevices[2][3].Get(1))->SetInterframeGapMean(Time(differentiationDelay));
    // }
    // ****** Mahdi Change (flow redirect) ***** (END) ***** //
    /* ########## END: Ceating the topology ########## */



    /* ########## START: Application Setup ########## */
    // CAIDA trace replay
    // Each host in R0 sends a flow to the corresponding host in R2
    // for (int i = 0; i < nHosts; i++) {
    //     auto* caidaTrafficGenerator = new BackgroundReplay(racks[0].Get(i), racks[2].Get(i), Seconds(stof(trafficStartTime)), Seconds(stof(trafficStopTime)));
    //     caidaTrafficGenerator->SetPctOfPacedTcps(pctPacedBack);
    //     string tracesPath = "/media/experiments/chicago_2010_traffic_10min_2paths/path" + to_string(i % 2);
    //     // string tracesPath = "/media/experiments/flow_csv_files/path_group_" + to_string(i % 4 + 1);
    //     // string tracesPath = "/media/experiments/flow_csv_files_2009_new/path_group_1";
    //     // string tracesPath = "/media/experiments/chicago_2010_traffic_10min_2paths/path0";
    //     if (std::filesystem::exists(tracesPath)) {
    //         caidaTrafficGenerator->RunAllTCPTraces(tracesPath, 0);
    //     } else {
    //         cout << "requested Background Directory does not exist" << endl;
    //     }
    // }

    // // each host in R1 sends a flow to the corresponding host in R3
    // for (int i = 0; i < nHosts; i++) {
    //     auto* caidaTrafficGenerator = new BackgroundReplay(racks[1].Get(i), racks[3].Get(i), Seconds(stof(trafficStartTime)), Seconds(stof(trafficStopTime)));
    //     caidaTrafficGenerator->SetPctOfPacedTcps(pctPacedBack);
    //     string tracesPath = "/media/experiments/chicago_2010_traffic_10min_2paths/path" + to_string(i % 2);
    //     // string tracesPath = "/media/experiments/flow_csv_files/path_group_" + to_string(i % 4 + 1);
    //     // string tracesPath = "/media/experiments/flow_csv_files_2009_new/path_group_1";
    //     // string tracesPath = "/media/experiments/chicago_2010_traffic_10min_2paths/path0";
    //     if (std::filesystem::exists(tracesPath)) {
    //         caidaTrafficGenerator->RunAllTCPTraces(tracesPath, 0);
    //     } else {
    //         cout << "requested Background Directory does not exist" << endl;
    //     }
    // }

    // DC Workload traffic
    avgMsgSize = readAvgMsgSize(traffic);
    hostTrafficRate = computeTraffciRate(load, DataRate(hostToTorLinkRate), avgMsgSize);
    // vector<Ptr<Node>> dstNodes;
    // dstNodes.push_back(racks[1].Get(0));
    // auto* dcTrafficGenerator = new DCWorkloadGenerator(racks[0].Get(0), dstNodes, hostTrafficRate, poolSize, "scratch/ECNMC/DCWorkloads/" + traffic, "ns3::TcpSocketFactory", Time(Seconds(0)), stopTime - Seconds(0.002));
    // dcTrafficGenerator->GenrateTraffic(pctPacedBack, passiveProbe, Time(probeInterval));
    for (int i = 0; i < nRacks; i++) {
        for (int j = 0; j < nHosts; j++) {
            vector<Ptr<Node>> dstNodes;
            // all nodes except this node are destination nodes
            for (int k = 0; k < nRacks; k++) {
                // if (i == 1 && k == 3)
                //     continue;
                // if (i == 3 && k == 1)
                //     continue;
                for (int l = 0; l < nHosts; l++) {
                    if (k != i || l != j) {
                        dstNodes.push_back(racks[k].Get(l));
                    }
                }
            }
            auto* dcTrafficGenerator = new DCWorkloadGenerator(racks[i].Get(j), dstNodes, hostTrafficRate, poolSize, "scratch/ECNMC/DCWorkloads/" + traffic, "ns3::TcpSocketFactory", Time(Seconds(0)), stopTime - Seconds(0.00002));
            // if this is the the traffic from R0H0, activate the passiveProbing
            if (i == 0 && j == 0) {
                dcTrafficGenerator->GenrateTraffic(pctPacedBack, passiveProbe, Time(probeInterval), Seconds(stof(trafficStartTime)));
            }
            else {
                dcTrafficGenerator->GenrateTraffic(pctPacedBack, false, Time(probeInterval), Seconds(stof(trafficStartTime)));
            }
        }
    }
    // Incast Traffic 
    // auto* incastTrafficGenerator = new IncastGenerator(racks, incastFactor, incastMessageSize, Time(incastperiod), Seconds(stof(trafficStartTime)), Seconds(stof(trafficStopTime)));
    // incastTrafficGenerator->Start();
    
    if (activeProbe) {
        auto* probeGenerator = new ProbeGenerator(activeProbeHost.Get(0), racks[2].Get(3), 1 / Time(probeInterval).GetSeconds(), Seconds(stof(steadyStartTime)), Seconds(stof(steadyStopTime)));
        probeGenerator->GenrateTraffic();
    }
    /* ########## END: Application Setup ########## */



    /* ########## START: Monitoring ########## */
    // p2pHostToTor.EnablePcapAll("N4_datacenter_switch_");
    // ns3::PacketMetadata::Enable(); // Enable packet metadata for debugging

    // End to End Monitors
    vector<E2EMonitor *> endToendMonitors;
    AggregatedE2EMonitor* aggregatedE2EMonitor = nullptr;
    // // monitor the packets between host 0 in R0 and host 0 in R2
    // auto *R0H0Monitor = new E2EMonitor(startTime, Seconds(stof(steadyStopTime)) + convergenceTime, Seconds(stof(steadyStartTime)), Seconds(stof(steadyStopTime)), DynamicCast<PointToPointNetDevice>(hostsToTorsNetDevices[0][0].Get(0)), racks[2].Get(0), racks[0].Get(0), "R0H0R2H0", errorRate, DataRate(hostToTorLinkRate), DataRate(torToAggLinkRate), Time(hostToTorLinkDelay), 2, 3, QueueSize(switchREDQueueDiscMaxSize).GetValue(), false, 0);
    // for (int i = 0; i < nRacks; i++) {
    //     for (int j = 0; j < nHosts; j++) {
    //         R0H0Monitor->AddAppKey(AppKey(ipsRacks[0][0].GetAddress(0), ipsRacks[i][j].GetAddress(0), 0, 0));
    //         cout << "Monitoring AppKey: " << ipsRacks[0][0].GetAddress(0) << " to " << ipsRacks[i][j].GetAddress(0) << endl;
    //     }
    // }
    // R0H0Monitor->AddAppKey(AppKey(ipsRacks[0][0].GetAddress(0), ipsRacks[2][0].GetAddress(0), 0, 0));
    // endToendMonitors.push_back(R0H0Monitor);
    if (monitorAllFlows) {
        // One global Rx callback dispatches packets to per-source files.
        aggregatedE2EMonitor =
            new AggregatedE2EMonitor(startTime,
                                    Seconds(stof(steadyStopTime)) + convergenceTime,
                                    Seconds(stof(steadyStartTime)),
                                    Seconds(stof(steadyStopTime)),
                                    DataRate(hostToTorLinkRate),
                                    DataRate(torToAggLinkRate),
                                    Time(hostToTorLinkDelay),
                                    nAggSwitches,
                                    3);
        for (int i = 0; i < nRacks; i++) {
            for (int j = 0; j < nHosts; j++) {
                aggregatedE2EMonitor->AddSource(
                    DynamicCast<PointToPointNetDevice>(hostsToTorsNetDevices[i][j].Get(0)),
                    ipsRacks[i][j].GetAddress(0),
                    "R" + to_string(i) + "H" + to_string(j) + "_ALL");
            }
        }
    }
    else {
        // Default lightweight monitoring mode.
        for (int j = 0; j < nHosts; j++) {
            auto *R0toR2H3Monitor = new E2EMonitor(startTime, Seconds(stof(steadyStopTime)) + convergenceTime, Seconds(stof(steadyStartTime)), Seconds(stof(steadyStopTime)), DynamicCast<PointToPointNetDevice>(hostsToTorsNetDevices[0][j].Get(0)), racks[2].Get(3), racks[0].Get(j), "R0H" + to_string(j) + "R2H3", errorRate, DataRate(hostToTorLinkRate), DataRate(torToAggLinkRate), Time(hostToTorLinkDelay), 2, 3, QueueSize(switchREDQueueDiscMaxSize).GetValue(), false, 0);
            R0toR2H3Monitor->AddAppKey(AppKey(ipsRacks[0][j].GetAddress(0), ipsRacks[2][3].GetAddress(0), 0, 0));
            endToendMonitors.push_back(R0toR2H3Monitor);
        }
    }

    // // we want to monitor all the flows
    // for (int i = 0; i < nRacks; i++) {
    //     for (int j = 0; j < nHosts; j++) {
    //         for (int k = 0; k < nRacks; k++) {
    //             for (int l = 0; l < nHosts; l++) {
    //                 if (i == k && j == l) {
    //                     continue;
    //                 }
    //                 auto *monitor = new E2EMonitor(startTime, Seconds(stof(steadyStopTime)) + convergenceTime, Seconds(stof(steadyStartTime)), Seconds(stof(steadyStopTime)), DynamicCast<PointToPointNetDevice>(hostsToTorsNetDevices[i][j].Get(0)), racks[k].Get(l), racks[i].Get(j), "R" + to_string(i) + "H" + to_string(j) + "R" + to_string(k) + "H" + to_string(l), errorRate, DataRate(hostToTorLinkRate), DataRate(torToAggLinkRate), Time(hostToTorLinkDelay), 2, 3, QueueSize(switchREDQueueDiscMaxSize).GetValue(), false, 0);
    //                 monitor->AddAppKey(AppKey(ipsRacks[i][j].GetAddress(0), ipsRacks[k][l].GetAddress(0), 0, 0));
    //                 endToendMonitors.push_back(monitor);
    //             }
    //         }
    //     }
    // }

    // if (activeProbe) {
    //     auto *ActiveProbeMonitor = new E2EMonitor(startTime, Seconds(stof(steadyStopTime)) + convergenceTime, Seconds(stof(steadyStartTime)), Seconds(stof(steadyStopTime)), DynamicCast<PointToPointNetDevice>(hostsToTorsNetDevices[0][nHosts].Get(0)), racks[2].Get(3), activeProbeHost.Get(0), "R0P_R2H3", errorRate, DataRate(hostToTorLinkRate), DataRate(torToAggLinkRate), Time(hostToTorLinkDelay), 2, 3, QueueSize(switchREDQueueDiscMaxSize).GetValue(), false, 0);
    //     ActiveProbeMonitor->AddAppKey(AppKey(ipsRacks[0][nHosts].GetAddress(0), ipsRacks[2][3].GetAddress(0), 0, 0));
    //     endToendMonitors.push_back(ActiveProbeMonitor);
    // }

    // PoissonSampler on the ToR switches, Agg switches and Core switches
    vector<PoissonSampler *> PoissonSamplers;
    // // PoissonSampler on the Hosts
    // for (int i = 0; i < nRacks / 2; i++) {
    //     for (int j = 0; j < nHosts; j++) {
    //         Ptr<PointToPointNetDevice> hostToTorNetDevice = DynamicCast<PointToPointNetDevice>(hostsToTorsNetDevices[i][j].Get(0));
    //         auto *hostToTorSampler = new PoissonSampler(Seconds(stof(steadyStartTime)), Seconds(stof(steadyStopTime)), nullptr, hostToTorNetDevice->GetQueue(), hostToTorNetDevice, "R" + to_string(i) + "H" + to_string(j), sampleRate);
    //         PoissonSamplers.push_back(hostToTorSampler);
    //     }
    // }
    // PoissonSampler on the tor to agg links
    // Only T0 to A0
    Ptr<PointToPointNetDevice> torToAggNetDeviceT0A0 = DynamicCast<PointToPointNetDevice>(torToAggNetDevices[0][0].Get(0));
    auto *torToAggSampler = new PoissonSampler(Seconds(stof(steadyStartTime)), Seconds(stof(steadyStopTime)), DynamicCast<RedQueueDisc>(torToAggQueueDiscs[0][0].Get(0)), torToAggNetDeviceT0A0->GetQueue(), torToAggNetDeviceT0A0, "T0A0", sampleRate, nullptr, nullptr, traffic);
    PoissonSamplers.push_back(torToAggSampler);
    // all tor to agg links
    // for (int i = 0; i < nRacks; i++) {
    //     for (int j = 0; j < nAggSwitches; j++) {
    //         Ptr<PointToPointNetDevice> torToAggNetDevice = DynamicCast<PointToPointNetDevice>(torToAggNetDevices[i][j].Get(0));            
    //         auto *torToAggSampler = new PoissonSampler(Seconds(stof(steadyStartTime)), Seconds(stof(steadyStopTime)), DynamicCast<RedQueueDisc>(torToAggQueueDiscs[i][j].Get(0)), torToAggNetDevice->GetQueue(), torToAggNetDevice, "T" + to_string(i) + "A" + to_string(j), sampleRate, nullptr, nullptr, traffic);
    //         PoissonSamplers.push_back(torToAggSampler);
    //     }
    // }
    // // PoissonSampler on the Tor to Host links
    // Only T2 to H3
    Ptr<PointToPointNetDevice> hostToTorNetDeviceT2H3 = DynamicCast<PointToPointNetDevice>(hostsToTorsNetDevices[2][3].Get(1));
    auto *hostToTorSamplerT2H3 = new PoissonSampler(Seconds(stof(steadyStartTime)), Seconds(stof(steadyStopTime)), DynamicCast<RedQueueDisc>(torToHostQueueDiscs[2][3].Get(0)), hostToTorNetDeviceT2H3->GetQueue(), hostToTorNetDeviceT2H3, "T2H3", sampleRate, nullptr, nullptr, traffic);
    PoissonSamplers.push_back(hostToTorSamplerT2H3);
    // all Tor to Host links
    // for (int i = 0; i < nRacks; i++) {
    //     for (int j = 0; j < nHosts; j++) {
    //         Ptr<PointToPointNetDevice> hostToTorNetDevice = DynamicCast<PointToPointNetDevice>(hostsToTorsNetDevices[i][j].Get(1));
    //         auto *hostToTorSampler = new PoissonSampler(Seconds(stof(steadyStartTime)), Seconds(stof(steadyStopTime)), DynamicCast<RedQueueDisc>(torToHostQueueDiscs[i][j].Get(0)), hostToTorNetDevice->GetQueue(), hostToTorNetDevice, "T" + to_string(i) + "H" + to_string(j), sampleRate, nullptr, nullptr, traffic);
    //         PoissonSamplers.push_back(hostToTorSampler);
    //     }
    // }
    // Only T0 to H5
    // Ptr<PointToPointNetDevice> hostToTorNetDeviceT0H5 = DynamicCast<PointToPointNetDevice>(hostsToTorsNetDevices[0][5].Get(1));
    // auto *hostToTorSamplerT0H5 = new PoissonSampler(Seconds(stof(steadyStartTime)), Seconds(stof(steadyStopTime)), DynamicCast<RedQueueDisc>(torToHostQueueDiscs[0][5].Get(0)), hostToTorNetDeviceT0H5->GetQueue(), hostToTorNetDeviceT0H5, "T0H5", sampleRate, nullptr, nullptr, traffic);
    // PoissonSamplers.push_back(hostToTorSamplerT0H5);
    // // PoissonSampler on the Agg to Tor links
    // Only A0 to T2
    Ptr<PointToPointNetDevice> aggToTorNetDeviceA0T2 = DynamicCast<PointToPointNetDevice>(torToAggNetDevices[2][0].Get(1));
    auto *aggToTorSamplerA0T2 = new PoissonSampler(Seconds(stof(steadyStartTime)), Seconds(stof(steadyStopTime)), DynamicCast<RedQueueDisc>(torToAggQueueDiscs[2][0].Get(1)), aggToTorNetDeviceA0T2->GetQueue(), aggToTorNetDeviceA0T2, "A0T2", sampleRate, nullptr, nullptr, traffic);
    PoissonSamplers.push_back(aggToTorSamplerA0T2);
    // all Agg to Tor links
    // for (int i = 0; i < nRacks; i++) {
    //     for (int j = 0; j < nAggSwitches; j++) {
    //         Ptr<PointToPointNetDevice> aggToTorNetDevice = DynamicCast<PointToPointNetDevice>(torToAggNetDevices[i][j].Get(1));
    //         auto *aggToTorSampler = new PoissonSampler(Seconds(stof(steadyStartTime)), Seconds(stof(steadyStopTime)), DynamicCast<RedQueueDisc>(torToAggQueueDiscs[i][j].Get(1)), aggToTorNetDevice->GetQueue(), aggToTorNetDevice, "A" + to_string(j) + "T" + to_string(i), sampleRate, nullptr, nullptr, traffic);
    //         PoissonSamplers.push_back(aggToTorSampler);
    //     }
    // }

    // vector<SwitchMonitor *> switchMonitors;
    // //  Switch Monitor on A0
    // auto *aggSwitchMonitor = new SwitchMonitor(startTime, Seconds(stof(steadyStopTime)) + convergenceTime, Seconds(stof(steadyStartTime)), Seconds(stof(steadyStopTime)), aggSwitches.Get(0), "A0");
    // aggSwitchMonitor->AddAppKey(AppKey(ipsRacks[0][0].GetAddress(0), ipsRacks[2][3].GetAddress(0), 0, 0));
    // switchMonitors.push_back(aggSwitchMonitor);

    // // switch monitor on A1
    // auto *aggSwitchMonitor1 = new SwitchMonitor(startTime, Seconds(stof(steadyStopTime)) + convergenceTime, Seconds(stof(steadyStartTime)), Seconds(stof(steadyStopTime)), aggSwitches.Get(1), "A1");
    // aggSwitchMonitor1->AddAppKey(AppKey(ipsRacks[0][0].GetAddress(0), ipsRacks[2][3].GetAddress(0), 0, 0));
    // switchMonitors.push_back(aggSwitchMonitor1);    

    /* ########## END: Monitoring ########## */



    /* ########## START: Check Config ########## */ 
    // print hosts and switches IP addresses
    cout << "Hosts and Switches IP addresses" << endl;
    for (int i = 0; i < nRacks; i++) {
        for (int j = 0; j < nHosts; j++) {
            cout << "Rack: " << i << " Host: "<< j << " Id:" << racks[i].Get(j)->GetId() << " IP: " << ipsRacks[i][j].GetAddress(0) << endl;
        }
    }
    if (activeProbe) {
        cout << "Active Probe Host Id: " << activeProbeHost.Get(0)->GetId() << " IP: " << ipsRacks[0][nHosts].GetAddress(0) << endl;
    }
    //print config parameters

    auto t = std::chrono::high_resolution_clock::now();
    cout << "Total preparing time = " << std::chrono::duration_cast<std::chrono::microseconds>(t - start).count() << " microsecond" << endl;

    cout << "Config Parameters" << endl;
    cout << "hostToTorLinkRate: " << hostToTorLinkRate << endl;
    cout << "hostToTorLinkRateCrossTraffic: " << hostToTorLinkRateCrossTraffic << endl;
    cout << "hostToTorLinkDelay: " << hostToTorLinkDelay << endl;
    cout << "torToAggLinkRate: " << torToAggLinkRate << endl;
    cout << "torToAggLinkDelay: " << torToAggLinkDelay << endl;
    cout << "aggToCoreLinkRate: " << aggToCoreLinkRate << endl;
    cout << "aggToCoreLinkDelay: " << aggToCoreLinkDelay << endl;
    cout << "probeInterval: " << probeInterval << endl;
    cout << "switchSrcREDQueueDiscMaxSize: " << switchSrcREDQueueDiscMaxSize << endl;
    cout << "switchREDQueueDiscMaxSize: " << switchREDQueueDiscMaxSize << endl;
    cout << "minTh: " << minTh << endl;
    cout << "maxTh: " << maxTh << endl;
    cout << "traffic: " << traffic << endl;
    cout << "isDifferentating: " << isDifferentating << endl;
    cout << "differentiationDelay: " << differentiationDelay << endl;
    cout << "silentPacketDrop: " << silentPacketDrop << endl;
    cout << "load: " << load << endl;
    cout << "tbfSrcRack: " << tbfSrcRack << " tbfSrcHost: " << tbfSrcHost << " tbfDstRack: " << tbfDstRack << " tbfDstHost: " << tbfDstHost << endl;
    cout << "tbfFlowRedirectFraction: " << tbfFlowRedirectFraction << endl;
    cout << "tbfRate: " << tbfRate << " tbfBurst: " << tbfBurst << endl;
    cout << "Average Message Size: " << avgMsgSize << endl;
    cout << "hostTrafficRate: " << hostTrafficRate << endl;
    cout << "duration: " << duration << endl;
    cout << "pctPacedBack: " << pctPacedBack << endl;
    cout << "enableSwitchECN: " << enableSwitchECN << endl;
    cout << "enableECMP: " << enableECMP << endl;
    cout << "sampleRate: " << sampleRate << endl;
    cout << "errorRate: " << errorRate << endl;
    cout << "dirName: " << dirName << endl;
    cout << "experiment: " << experiment << endl;
    cout << "trafficStartTime: " << trafficStartTime << endl;
    cout << "trafficStopTime: " << trafficStopTime << endl;
    cout << "steadyStartTime: " << steadyStartTime << endl;
    cout << "steadyEndTime: " << steadyStopTime << endl;
    cout << "Sender Nagle: " << Nagle << endl;
    cout << "Sender ActiveProbe: " << activeProbe << endl;
    cout << "Sender PassiveProbe: " << passiveProbe << endl;
    cout << "Incast Factor: " << incastFactor << endl;
    cout << "Incast Message Size: " << incastMessageSize << endl;
    cout << "Incast Period: " << incastperiod << endl;
    cout << "Seed: " << seed << endl;
    /* ########## END: Check Config ########## */


    /* ########## START: Scheduling and  Running ########## */
    ScheduleDumpingPackets(Seconds(stof(steadyStartTime)),
                           Seconds(stof(steadyStopTime)),
                           endToendMonitors,
                           aggregatedE2EMonitor,
                           PoissonSamplers,
                           dirName,
                           experiment);

    Simulator::Stop(stopTime);
    Simulator::Run();
    Simulator::Destroy();

    for (auto monitor: endToendMonitors) {
        monitor->SaveMonitorRecords((string) (getenv("PWD")) + "/Results/results_" + dirName + "/" + to_string(experiment)  + "/" + monitor->GetMonitorTag() + "_" + to_string(Seconds(stof(steadyStopTime)).GetNanoSeconds()) + "_EndToEnd.csv");
    }
    if (aggregatedE2EMonitor != nullptr) {
        string outputDirectory = (string)(getenv("PWD")) + "/Results/results_" + dirName +
                                 "/" + to_string(experiment);
        aggregatedE2EMonitor->SaveMonitorRecords(outputDirectory);
        aggregatedE2EMonitor->FlushStreams();
    }

    for (auto monitor: PoissonSamplers) {
        monitor->SaveMonitorRecords((string) (getenv("PWD")) + "/Results/results_" + dirName + "/" + to_string(experiment)  + "/" + monitor->GetMonitorTag() + "_" + to_string(Seconds(stof(steadyStopTime)).GetNanoSeconds()) + "_PoissonSampler.csv");
    }

    // for (auto monitor: switchMonitors) {
    //     monitor->SavePacketRecords((string) (getenv("PWD")) + "/Results/results_" + dirName + "/" + to_string(experiment)  + "/" + monitor->GetMonitorTag() + "_" + to_string(Seconds(stof(steadyStopTime)).GetNanoSeconds()) + "_SwitchMonitor.csv");
    // }
    /* ########## END: Scheduling and  Running ########## */


    cout << "Done " << endl;

    auto stop = std::chrono::high_resolution_clock::now();
    cout << "Total execution time = " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " microsecond" << endl;
}

int main(int argc, char* argv[])
{
    if (strcmp(argv[1], "True") == 0) {
        run_single_queue_simulation(argc, argv);
    } else {
        run_DC_simulation(argc, argv);  
    }
    return 0;
}
