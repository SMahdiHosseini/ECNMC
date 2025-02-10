//
// Created by Zeinab Shmeis on 28.05.20.
//

#ifndef E2EMONITOR_H
#define E2EMONITOR_H

#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/random-variable-stream.h"
#include "Monitor.h"
#include "PacketKey.h"
#include "AppKey.h"
#include "PacketCDF.h"

#include <ostream>
#include <unordered_map>

using namespace ns3;
using namespace std;

struct E2EMonitorEvent : MonitorEvent {

public:
    explicit E2EMonitorEvent(PacketKey *key);

    [[nodiscard]] bool GetEcn() const;
    [[nodiscard]] int GetPath() const;
    void SetEcn(bool ecn);
    void SetPath(int path);
};

class E2EMonitor : public Monitor {

private:
    double _errorRate;
    std::vector<uint32_t> sumOfPacketSizes;
    std::vector<uint32_t> sentPackets;
    std::vector<uint32_t> sentPackets_onlink;
    std::vector<uint32_t> markedPackets;
    std::vector<Time> timeAverageIntegral;
    std::vector<Time> integralStartTime;
    std::vector<Time> integralEndTime;
    DataRate hostToTorLinkRate;
    DataRate torToAggLinkRate;
    Time hostToTorLinkDelay;
    Hasher hasher;
    Ptr<UniformRandomVariable> rand;
    int numOfPaths = 2;
    int numOfSegmetns = 3;
    PacketCDF packetCDF;
    double GTDropMean;
    Time lastItemTime;
    Time firstItemTime;
    std::unordered_map<PacketKey, E2EMonitorEvent*, PacketKeyHash> _recordedPackets;
    
    void Connect(Ptr<PointToPointNetDevice> netDevice, uint32_t rxNodeId);
    void Disconnect(Ptr<PointToPointNetDevice> netDevice, uint32_t rxNodeId);

    void Enqueue(Ptr< const Packet > packet);
    void Capture(Ptr< const Packet > packet);
    void RecordIpv4PacketReceived(Ptr<const Packet> packet, Ptr<Ipv4> ipv4, uint32_t interface);
    uint64_t GetHashValue(const Ipv4Address src, const Ipv4Address dst, const uint16_t srcPort, const uint16_t dstPort, const uint8_t protocol);
    void updateTimeAverageIntegral(uint32_t path, Time delay, Time endTime);
    double calculateUnbiasedGTDrop();
public:
    E2EMonitor(const Time &startTime, const Time &duration, const Time &steadyStartTime, const Time &steadyStopTime, const Ptr<PointToPointNetDevice> netDevice, const Ptr<Node> &rxNode, const string &monitorTag, const double errorRate, 
    const DataRate &hostToTorLinkRate, const DataRate &torToAggLinkRate, const Time &hostToTorLinkDelay);
    E2EMonitor(const Time &startTime, const Time &duration, const Time &steadyStartTime, const Time &steadyStopTime, const Ptr<PointToPointNetDevice> netDevice, const Ptr<Node> &rxNode, const string &monitorTag, const double errorRate, 
    const DataRate &hostToTorLinkRate, const DataRate &torToAggLinkRate, const Time &hostToTorLinkDelay, const int numOfPaths, const int numOfSegmetns);
    void SaveMonitorRecords(const string &filename);
};

#endif //E2EMONITOR_H
