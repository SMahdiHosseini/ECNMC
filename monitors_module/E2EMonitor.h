//
// Created by Zeinab Shmeis on 28.05.20.
//

#ifndef E2EMONITOR_H
#define E2EMONITOR_H

#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"

#include "PacketKey.h"
#include "AppKey.h"

#include <ostream>
#include <unordered_map>

using namespace ns3;
using namespace std;

struct E2EMonitorEvent {

private:
    PacketKey* _key;
    ns3::Time _sentTime;
    ns3::Time _receivedTime = Time(-1);

public:
    explicit E2EMonitorEvent(PacketKey *key);

    [[nodiscard]] PacketKey* GetPacketKey() const;
    [[nodiscard]] Time GetSentTime() const;
    [[nodiscard]] Time GetReceivedTime() const;
    [[nodiscard]] bool IsReceived() const;
    [[nodiscard]] bool GetEcn() const;
    [[nodiscard]] int GetPath() const;

    void SetSent();
    void SetReceived();
    void SetReceived(Time t);
    void SetEcn(bool ecn);
    void SetPath(int path);

    friend ostream &operator<<(ostream &os, const E2EMonitorEvent &event);
};

class E2EMonitor {

private:
    ns3::Time _startTime = Seconds(0);
    ns3::Time _duration = Seconds(0);
    double _errorRate = 0.0;
    std::string _monitorTag;
    set<AppKey> _appsKey;
    ns3::Time averageDelay = Seconds(0);
    uint32_t _receivedPackets = 0;

    std::unordered_map<PacketKey, E2EMonitorEvent*, PacketKeyHash> _recordedPackets;

    void Connect(Ptr<PointToPointNetDevice> netDevice, uint32_t rxNodeId);
    void Disconnect(Ptr<PointToPointNetDevice> netDevice, uint32_t rxNodeId);

    void Enqueue(Ptr< const Packet > packet);
    void RecordIpv4PacketReceived(Ptr<const Packet> packet, Ptr<Ipv4> ipv4, uint32_t interface);

    ns3::Time GetRelativeTime(const Time &time);

public:
    E2EMonitor(const Time &startTime, const Time &duration, const Ptr<PointToPointNetDevice> netDevice, const Ptr<Node> &rxNode, const string &monitorTag, const double errorRate);
    void AddAppKey(AppKey appKey);
    std::string GetMonitorTag() const;
    void SavePacketRecords(const string &filename);
};

#endif //E2EMONITOR_H
