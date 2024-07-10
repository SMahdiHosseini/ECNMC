//
// Created by Mahdi Hosseini on 10.07.24.
//

#ifndef MONITOR_H
#define MONITOR_H

#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"

#include "PacketKey.h"
#include "AppKey.h"

#include <ostream>
#include <unordered_map>

using namespace ns3;
using namespace std;

struct MonitorEvent {

private:
    PacketKey* _key;
    ns3::Time _sentTime = Time(-1);
    ns3::Time _receivedTime = Time(-1);

public:
    explicit MonitorEvent(PacketKey *key);

    [[nodiscard]] PacketKey* GetPacketKey() const;
    [[nodiscard]] Time GetSentTime() const;
    [[nodiscard]] Time GetReceivedTime() const;
    [[nodiscard]] bool IsReceived() const;
    [[nodiscard]] bool IsSent() const;

    void SetSent();
    void SetSent(Time t);
    void SetReceived();
    void SetReceived(Time t);
};

class E2EMonitor {

private:
    ns3::Time _startTime = Seconds(0);
    ns3::Time _duration = Seconds(0);
    // ns3::Time _startTime = Seconds(0); steady
    // ns3::Time _duration = Seconds(0);
    double _errorRate = 0.0;
    std::string _monitorTag;
    set<AppKey> _appsKey;

    ns3::Time GetRelativeTime(const Time &time);

public:
    E2EMonitor(const Time &startTime, const Time &duration, const Ptr<PointToPointNetDevice> netDevice, const Ptr<Node> &rxNode, const string &monitorTag, const double errorRate);
    void AddAppKey(AppKey appKey);
    std::string GetMonitorTag() const;
    void SavePacketRecords(const string &filename);
};

#endif //E2EMONITOR_H
