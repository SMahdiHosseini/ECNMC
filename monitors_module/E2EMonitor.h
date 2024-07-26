//
// Created by Zeinab Shmeis on 28.05.20.
//

#ifndef E2EMONITOR_H
#define E2EMONITOR_H

#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "Monitor.h"
#include "PacketKey.h"
#include "AppKey.h"

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
    ns3::Time sampleMean;
    ns3::Time unbiasedSmapleVariance;
    uint32_t sumOfPacketSizes;
    uint32_t receivedPackets;
    uint32_t sentPackets;
    uint32_t markedPackets;

    std::unordered_map<PacketKey, E2EMonitorEvent*, PacketKeyHash> _recordedPackets;

    void Connect(Ptr<PointToPointNetDevice> netDevice, uint32_t rxNodeId);
    void Disconnect(Ptr<PointToPointNetDevice> netDevice, uint32_t rxNodeId);

    void Enqueue(Ptr< const Packet > packet);
    void RecordIpv4PacketReceived(Ptr<const Packet> packet, Ptr<Ipv4> ipv4, uint32_t interface);

public:
    E2EMonitor(const Time &startTime, const Time &duration, const Time &steadyStartTime, const Time &steadyStopTime, const Ptr<PointToPointNetDevice> netDevice, const Ptr<Node> &rxNode, const string &monitorTag, const double errorRate);
    void SaveMonitorRecords(const string &filename);
};

#endif //E2EMONITOR_H
