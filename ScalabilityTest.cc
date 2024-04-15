// // The nework Topology is as follows:
// //  
// //  s1                s3
// //  |                  |
// //  T1 -----(10Gbps)-- T2 -(1Gbps)-R2
// //  |                  |    
// //  s2                R1

// #include "ns3/applications-module.h"
// #include "ns3/core-module.h"
// #include "ns3/internet-module.h"
// #include "ns3/network-module.h"
// #include "ns3/point-to-point-module.h"
// #include "ns3/traffic-control-module.h"
// #include "ns3/flow-monitor-module.h"
// #include "monitors_module/PacketMonitor.h"
// #include "monitors_module/SwitchMonitor.h"
// #include "traffic_generator_module/background_replay/BackgroundReplay.h"
// #include <iomanip>
// #include <iostream>
// #include <string>

// using namespace ns3;
// using namespace std;

// int main(int argc, char* argv[])
// {
//     auto start = std::chrono::high_resolution_clock::now();
//     cout << endl<< "Start" << endl;
//     /* ########## START: Config ########## */
//     string edgeLinkRate = "0.2Mbps";             // Links bandwith on The Edge Layer
//     string edgeLinkDelay = "10us";               // Links delay on The Edge Layer
//     string spineLinkRate = "1Mbps";              // Links bandwith on The Spine Layer
//     string spineLinkDelay = "10us";              // Links delay on The Spine Layer
//     string S2R2DataRate = "1Mbps";               // Data rate between S2 and R2
//     string S1R1DataRate = "1Mbps";               // Data rate between S1 and R1
//     string S3R1DataRate = "1Mbps";               // Data rate between S3 and R1
//     double pctPacedBack = 0.8;                   // the percentage of tcp flows of the CAIDA trace to be paced

//     Time startTime = Seconds(0);
//     Time stopTime = Seconds(0.5);
//     bool enableSwitchECN = true;
//     bool enableECMP = false;

//     /*command line input*/
//     CommandLine cmd;
//     cmd.AddValue("edgeLinkRate", "Links bandwith on The Edge Layer", edgeLinkRate);
//     cmd.AddValue("edgeLinkDelay", "Links delay on The Edge Layer", edgeLinkDelay);
//     cmd.AddValue("spineLinkRate", "Links bandwith on The Spine Layer", spineLinkRate);
//     cmd.AddValue("spineLinkDelay", "Links delay on The Spine Layer", spineLinkDelay);
//     cmd.AddValue("S2R2DataRate", "Data rate between S2 and R2", S2R2DataRate);
//     cmd.AddValue("S1R1DataRate", "Data rate between S1 and R1", S1R1DataRate);
//     cmd.AddValue("S3R1DataRate", "Data rate between S3 and R1", S3R1DataRate);
//     cmd.AddValue("enableSwichECN", "Enable ECN on the switches", enableSwitchECN);
//     cmd.AddValue("enableECMP", "Enable ECMP on the switches", enableECMP);
//     cmd.AddValue("pctPacedTcpBack", "the percentage of tcp flows of the CAIDA trace to be paced", pctPacedBack);
//     cmd.Parse(argc, argv);

//     /*set default values*/
//     Config::SetDefault("ns3::TcpL4Protocol::SocketType", StringValue("ns3::TcpDctcp"));
//     Config::SetDefault("ns3::Ipv4GlobalRouting::RandomEcmpRouting", BooleanValue(enableECMP));
//     Config::SetDefault("ns3::RedQueueDisc::UseEcn", BooleanValue(enableSwitchECN));
//     Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(1448));
//     Config::SetDefault("ns3::TcpSocket::DelAckCount", UintegerValue(2));
//     GlobalValue::Bind("ChecksumEnabled", BooleanValue(false));
//     Config::SetDefault("ns3::RedQueueDisc::UseHardDrop", BooleanValue(false));
//     Config::SetDefault("ns3::RedQueueDisc::MeanPktSize", UintegerValue(1500));
//     // Triumph and Scorpion switches used in DCTCP Paper have 4 MB of buffer
//     // If every packet is 1500 bytes, 2666 packets can be stored in 4 MB
//     Config::SetDefault("ns3::RedQueueDisc::MaxSize", QueueSizeValue(QueueSize("2666p")));
//     Config::SetDefault("ns3::RedQueueDisc::QW", DoubleValue(1));
//     Config::SetDefault("ns3::RedQueueDisc::MinTh", DoubleValue(20));
//     Config::SetDefault("ns3::RedQueueDisc::MaxTh", DoubleValue(60));
//     /* ########## END: Config ########## */



//     /* ########## START: Ceating the topology ########## */
//     Ptr<Node> S1 = CreateObject<Node>();
//     Ptr<Node> S2 = CreateObject<Node>();
//     Ptr<Node> S3 = CreateObject<Node>();
//     Ptr<Node> T1 = CreateObject<Node>();
//     Ptr<Node> T2 = CreateObject<Node>();
//     Ptr<Node> R1 = CreateObject<Node>();
//     Ptr<Node> R2 = CreateObject<Node>();

//     PointToPointHelper p2pSpine;
//     p2pSpine.SetDeviceAttribute("DataRate", StringValue(spineLinkRate));
//     p2pSpine.SetChannelAttribute("Delay", StringValue(spineLinkDelay));

//     PointToPointHelper p2pEdge;
//     p2pEdge.SetDeviceAttribute("DataRate", StringValue(edgeLinkRate));
//     p2pEdge.SetChannelAttribute("Delay", StringValue(edgeLinkDelay));
    

//     NetDeviceContainer S1T1 = p2pSpine.Install(S1, T1);
//     NetDeviceContainer S2T1 = p2pSpine.Install(S2, T1);
//     NetDeviceContainer S3T2 = p2pSpine.Install(S3, T2);
//     NetDeviceContainer R1T2 = p2pSpine.Install(R1, T2);
//     NetDeviceContainer R2T2 = p2pSpine.Install(R2, T2);
//     NetDeviceContainer T1T2 = p2pEdge.Install(T1, T2);

//     InternetStackHelper stack;
//     stack.InstallAll();

//     TrafficControlHelper tchRED10Gbps;
//     tchRED10Gbps.SetRootQueueDisc("ns3::RedQueueDisc", 
//                                   "LinkBandwidth", StringValue(edgeLinkRate), 
//                                   "LinkDelay", StringValue(edgeLinkDelay), 
//                                   "MinTh", DoubleValue(50),
//                                   "MaxTh", DoubleValue(150));
//     QueueDiscContainer queueDisc1 = tchRED10Gbps.Install(T1T2);

//     TrafficControlHelper tchRED1Gbps;
//     tchRED1Gbps.SetRootQueueDisc("ns3::RedQueueDisc", 
//                                  "LinkBandwidth", StringValue(edgeLinkRate), 
//                                  "LinkDelay", StringValue(spineLinkDelay), 
//                                  "MinTh", DoubleValue(20),
//                                  "MaxTh", DoubleValue(60));
//     QueueDiscContainer queueDisc2 = tchRED1Gbps.Install(S1T1.Get(1));
//     QueueDiscContainer queueDisc3 = tchRED1Gbps.Install(S2T1.Get(1));
//     tchRED1Gbps.Install(S3T2.Get(1));
//     QueueDiscContainer queueDisc4 = tchRED1Gbps.Install(R1T2.Get(1));
//     tchRED1Gbps.Install(R2T2.Get(1));
    
//     uint16_t nbSubnet = 0;
//     Ipv4AddressHelper address;
//     address.SetBase(("10.1." + to_string(++nbSubnet) + ".0").c_str(), "255.255.255.0");
//     Ipv4InterfaceContainer ipR1T2 = address.Assign(R1T2);
//     address.SetBase(("10.1." + to_string(++nbSubnet) + ".0").c_str(), "255.255.255.0");
//     Ipv4InterfaceContainer ipR2T2 = address.Assign(R2T2);
//     address.SetBase(("10.1." + to_string(++nbSubnet) + ".0").c_str(), "255.255.255.0");
//     Ipv4InterfaceContainer ipS1T1 = address.Assign(S1T1);
//     address.SetBase(("10.1." + to_string(++nbSubnet) + ".0").c_str(), "255.255.255.0");
//     Ipv4InterfaceContainer ipS2T1 = address.Assign(S2T1);
//     address.SetBase(("10.1." + to_string(++nbSubnet) + ".0").c_str(), "255.255.255.0");
//     Ipv4InterfaceContainer ipS3T2 = address.Assign(S3T2);
//     address.SetBase(("10.1." + to_string(++nbSubnet) + ".0").c_str(), "255.255.255.0");
//     Ipv4InterfaceContainer ipT1T2 = address.Assign(T1T2);
//     Ipv4GlobalRoutingHelper::PopulateRoutingTables();
//     /* ########## END: Ceating the topology ########## */



//     /* ########## START: Application Setup ########## */
//     // S1 -> R1
//     auto* appTraffic = new BackgroundReplay(S1, R1);
//     appTraffic->SetPctOfPacedTcps(pctPacedBack);
//     string tracesPath = "/home/mahdi/Documents/NAL/Data/chicago_2010_traffic_10min_2paths/path0";
//     if (std::filesystem::exists(tracesPath)) {
//         appTraffic->RunAllTraces(tracesPath, 0);
//     } else {
//         cout << "requested Background Directory does not exist" << endl;
//     }

//     // // S2 -> R2
//     // auto* appTraffic2 = new BackgroundReplay(S2, R2);
//     // appTraffic2->SetPctOfPacedTcps(pctPacedBack);
//     // string tracesPath2 = "/home/mahdi/Documents/NAL/Data/chicago_2010_traffic_10min_2paths/path1";
//     // if (std::filesystem::exists(tracesPath2)) {
//     //     appTraffic2->RunAllTraces(tracesPath2, 0);
//     // } else {
//     //     cout << "requested Background Directory does not exist" << endl;
//     // }

//     // S3 -> R1
//     auto* appTraffic3 = new BackgroundReplay(S3, R1);
//     appTraffic3->SetPctOfPacedTcps(pctPacedBack);
//     string tracesPath3 = "/home/mahdi/Documents/NAL/Data/chicago_2010_traffic_10min_2paths/path0";
//     if (std::filesystem::exists(tracesPath3)) {
//         appTraffic3->RunAllTraces(tracesPath3, 0);
//     } else {
//         cout << "requested Background Directory does not exist" << endl;
//     }

//     // // S2 -> R2
//     // uint16_t port = 50000;
//     // Address sinkLocalAddress(InetSocketAddress(Ipv4Address::GetAny(), port));
//     // PacketSinkHelper sinkHelper("ns3::TcpSocketFactory", sinkLocalAddress);
//     // ApplicationContainer sinkApp = sinkHelper.Install(R2);
//     // Ptr<PacketSink> s2r2PacketSink = sinkApp.Get(0)->GetObject<PacketSink>();
//     // sinkApp.Start(startTime);
//     // sinkApp.Stop(stopTime);
//     // OnOffHelper S2ClientHelper("ns3::TcpSocketFactory", Address());
//     // S2ClientHelper.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
//     // S2ClientHelper.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
//     // S2ClientHelper.SetAttribute("DataRate", DataRateValue(DataRate(S2R2DataRate)));
//     // S2ClientHelper.SetAttribute("PacketSize", UintegerValue(1000));
//     // ApplicationContainer S2ClientApp;
//     // AddressValue remoteAddress(InetSocketAddress(ipR2T2.GetAddress(0), port));
//     // S2ClientHelper.SetAttribute("Remote", remoteAddress);
//     // S2ClientApp.Add(S2ClientHelper.Install(S2));
//     // S2ClientApp.Start(startTime);
//     // S2ClientApp.Stop(stopTime);

//     // // S1 -> R1
//     // uint16_t port2 = 50001;
//     // Address sinkLocalAddress2 = Address(InetSocketAddress(Ipv4Address::GetAny(), port2));
//     // PacketSinkHelper sinkHelper2 = PacketSinkHelper("ns3::TcpSocketFactory", sinkLocalAddress2);
//     // ApplicationContainer sinkApp2 = sinkHelper2.Install(R1);
//     // Ptr<PacketSink> s1r1PacketSink = sinkApp2.Get(0)->GetObject<PacketSink>();
//     // sinkApp2.Start(startTime);
//     // sinkApp2.Stop(stopTime);
//     // OnOffHelper S1ClientHelper("ns3::TcpSocketFactory", Address());
//     // S1ClientHelper.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
//     // S1ClientHelper.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
//     // S1ClientHelper.SetAttribute("DataRate", DataRateValue(DataRate(S1R1DataRate)));
//     // S1ClientHelper.SetAttribute("PacketSize", UintegerValue(1000));
//     // ApplicationContainer S1ClientApp;
//     // AddressValue remoteAddress2 = AddressValue(InetSocketAddress(ipR1T2.GetAddress(0), port2));
//     // S1ClientHelper.SetAttribute("Remote", remoteAddress2);
//     // S1ClientApp.Add(S1ClientHelper.Install(S1));
//     // S1ClientApp.Start(startTime);
//     // S1ClientApp.Stop(stopTime);

//     // // S3 -> R1
//     // uint16_t port3 = 50002;
//     // Address sinkLocalAddress3 = Address(InetSocketAddress(Ipv4Address::GetAny(), port3));
//     // PacketSinkHelper sinkHelper3 = PacketSinkHelper("ns3::TcpSocketFactory", sinkLocalAddress3);
//     // ApplicationContainer sinkApp3 = sinkHelper3.Install(R1);
//     // Ptr<PacketSink> s3r1PacketSink = sinkApp3.Get(0)->GetObject<PacketSink>();
//     // sinkApp3.Start(startTime);
//     // sinkApp3.Stop(stopTime);
//     // OnOffHelper S3ClientHelper("ns3::TcpSocketFactory", Address());
//     // S3ClientHelper.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
//     // S3ClientHelper.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
//     // S3ClientHelper.SetAttribute("DataRate", DataRateValue(DataRate(S3R1DataRate)));
//     // S3ClientHelper.SetAttribute("PacketSize", UintegerValue(1000));
//     // ApplicationContainer S3ClientApp;
//     // AddressValue remoteAddress3 = AddressValue(InetSocketAddress(ipR1T2.GetAddress(0), port3));
//     // S3ClientHelper.SetAttribute("Remote", remoteAddress3);
//     // S3ClientApp.Add(S3ClientHelper.Install(S3));
//     // S3ClientApp.Start(startTime);
//     // S3ClientApp.Stop(stopTime);
//     /* ########## END: Application Setup ########## */



//     /* ########## START: Monitoring ########## */
//     // p2pSpine.EnablePcapAll("N4_datacenter_switch_");

//     // End to End Monitors
//     vector<PacketMonitor *> endToendMonitors;
//     // S1 -> R1 Monitor
//     auto *S1R1Monitor = new PacketMonitor(startTime, stopTime + MilliSeconds(10), S1, R1, "S1R1");
//     S1R1Monitor->AddAppKey(AppKey(ipS1T1.GetAddress(0), ipR1T2.GetAddress(0), 0, 0));
//     endToendMonitors.push_back(S1R1Monitor);

//     // S2 -> R2 Monitor
//     auto *S2R2Monitor = new PacketMonitor(startTime, stopTime + MilliSeconds(10), S2, R2, "S2R2");
//     S2R2Monitor->AddAppKey(AppKey(ipS2T1.GetAddress(0), ipR2T2.GetAddress(0), 0, 0));
//     endToendMonitors.push_back(S2R2Monitor);

//     // S3 -> R1 Monitor
//     auto *S3R1Monitor = new PacketMonitor(startTime, stopTime + MilliSeconds(10), S3, R1, "S3R1");
//     S3R1Monitor->AddAppKey(AppKey(ipS3T2.GetAddress(0), ipR1T2.GetAddress(0), 0, 0));
//     endToendMonitors.push_back(S3R1Monitor);


//     // Switch Monitors
//     vector<SwitchMonitor *> switchMonitors;
//     // T1 Switch Monitor
//     auto *T1SwitchMonitor = new SwitchMonitor(startTime, stopTime, T1, "T1");
//     switchMonitors.push_back(T1SwitchMonitor);

//     // T2 Switch Monitor
//     auto *T2SwitchMonitor = new SwitchMonitor(startTime, stopTime, T2, "T2");
//     switchMonitors.push_back(T2SwitchMonitor);

//     /* ########## END: Monitoring ########## */



//     /* ########## START: Check Config ########## */ 
//     cout << "S1 ID: " << S1->GetId() << "\tS1 IP: " << ipS1T1.GetAddress(0) << endl;
//     cout << "S2 ID: " << S2->GetId() << "\tS2 IP: " << ipS2T1.GetAddress(0) << endl;
//     cout << "S3 ID: " << S3->GetId() << "\tS3 IP: " << ipS3T2.GetAddress(0) << endl;
//     cout << "T1 ID: " << T1->GetId() << "\tT1 IP: " << ipT1T2.GetAddress(0) << " " << ipT1T2.GetAddress(0) << endl;
//     cout << "T2 ID: " << T2->GetId() << "\tT2 IP: " << ipT1T2.GetAddress(1) << " " << ipT1T2.GetAddress(1) << endl;
//     cout << "R1 ID: " << R1->GetId() << "\tR1 IP: " << ipR1T2.GetAddress(0) << endl;
//     cout << "R2 ID: " << R2->GetId() << "\tR2 IP: " << ipR2T2.GetAddress(0) << endl;
//     /* ########## END: Check Config ########## */


//     /* ########## START: Scheduling and  Running ########## */
//     Simulator::Stop(stopTime);
//     Simulator::Run();
//     Simulator::Destroy();

//     for (auto monitor: endToendMonitors) {
//         monitor->SavePacketRecords("N4_datacenter_switch_" + monitor->GetMonitorTag() + "_endtoend.csv");
//     }
//     for (auto monitor: switchMonitors) {
//         monitor->SavePacketRecords("N4_datacenter_switch_" + monitor->GetMonitorTag() + "_switch.csv");
//     }
//     /* ########## END: Scheduling and  Running ########## */



//     cout << "Done" << endl;

//     auto stop = std::chrono::high_resolution_clock::now();
//     cout << "Total execution time = " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " microsecond" << endl;
//     return 0;
// }