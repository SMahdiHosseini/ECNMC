//
// Created by Mahdi Hosseini on 13.06.2024.
//

#ifndef ECNMC_NETDEVICEMONITOR_H
#define ECNMC_NETDEVICEMONITOR_H

#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"

#include "PacketKey.h"
#include "AppKey.h"

#include <ostream>
#include <unordered_map>

using namespace ns3;
using namespace std;

struct NetDeviceMonitorEvent {

private:
    PacketKey* _key;
    ns3::Time _sentTime;
    ns3::Time _receivedTime = Time(-1);

public:
    explicit NetDeviceMonitorEvent(PacketKey *key);

    [[nodiscard]] PacketKey* GetPacketKey() const;
    [[nodiscard]] Time GetSentTime() const;
    [[nodiscard]] Time GetReceivedTime() const;
    [[nodiscard]] bool IsSent() const;

    void SetSent();
    void SetReceived();

    friend ostream &operator<<(ostream &os, const NetDeviceMonitorEvent &event);
};

class NetDeviceMonitor {

private:
    ns3::Time _startTime = Seconds(0);
    ns3::Time _duration = Seconds(0);

    std::string _monitorTag;
    set<AppKey> _appsKey;

    std::unordered_map<PacketKey, NetDeviceMonitorEvent*, PacketKeyHash> _recordedPackets;

    void Connect(Ptr<PointToPointNetDevice> netDevice);
    void Disconnect(Ptr<PointToPointNetDevice> netDevice);

    void Enqueue(Ptr< const Packet > packet);
    void Dequeue(Ptr< const Packet > packet);

    ns3::Time GetRelativeTime(const Time &time);

public:
    NetDeviceMonitor(const Time &startTime, const Time &duration, Ptr<PointToPointNetDevice> netDevice, const string &monitorTag);
    void AddAppKey(AppKey appKey);
    std::string GetMonitorTag() const;
    void SavePacketRecords(const string &filename);
};

#endif //ECNMC_NETDEVICEMONITOR_H
