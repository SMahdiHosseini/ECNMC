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
    ns3::Time _sentTime = Time(-1);
    ns3::Time _receivedTime = Time(-1);

protected:
    PacketKey* _key;

public:
    explicit MonitorEvent();

    [[nodiscard]] PacketKey* GetPacketKey() const;
    [[nodiscard]] Time GetSentTime() const;
    [[nodiscard]] Time GetReceivedTime() const;
    [[nodiscard]] bool IsReceived() const;
    [[nodiscard]] bool IsSent() const;

    void SetPacketKey(PacketKey *key);
    void SetSent();
    void SetSent(Time t);
    void SetReceived();
    void SetReceived(Time t);
};

class Monitor {

protected:
    ns3::Time _startTime = Seconds(0);
    ns3::Time _duration = Seconds(0);
    ns3::Time _steadyStartTime = Seconds(0);
    ns3::Time _steadyStopTime = Seconds(0);

    std::string _monitorTag;
    set<AppKey> _appsKey;

    ns3::Time GetRelativeTime(const Time &time);

public:
    Monitor(const Time &startTime, const Time &duration, const Time &steadyStartTime, const Time &steadyStopTime, const string &monitorTag);
    void AddAppKey(AppKey appKey);
    std::string GetMonitorTag() const;
    virtual void SaveMonitorRecords(const string &filename) = 0;
};

#endif //MONITOR_H
