//
// Created by Mahdi Hosseini on 19.03.24.
//

#ifndef ECC_SWITCHMONITOR_H
#define ECC_SWITCHMONITOR_H

#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/queue-item.h"
#include "ns3/traffic-control-module.h"
#include "ns3/point-to-point-module.h"

#include "PacketKey.h"
#include "AppKey.h"

#include <ostream>
#include <unordered_map>

using namespace ns3;
using namespace std;

struct SwitchMonitorEvent {

private:
    PacketKey* _key;
    ns3::Time _sentTime = Time(-1);
    ns3::Time _receivedTime;

public:
    explicit SwitchMonitorEvent(PacketKey *key);

    [[nodiscard]] PacketKey* GetPacketKey() const;
    [[nodiscard]] Time GetSentTime() const;
    [[nodiscard]] Time GetReceivedTime() const;
    [[nodiscard]] bool IsSent() const;

    void SetSent();
    void SetReceived();

    friend ostream &operator<<(ostream &os, const SwitchMonitorEvent &event);
};

class SwitchMonitor {

private:
    ns3::Time _startTime = Seconds(0);
    ns3::Time _duration = Seconds(0);

    std::string _monitorTag;
    set<AppKey> _appsKey;
    Hasher hasher;
    std::unordered_map<PacketKey, SwitchMonitorEvent*, PacketKeyHash> _recordedPackets;

    void Connect(const Ptr<Node> &node);
    void Disconnect(const Ptr<Node> &node);

    void RecordPacket(Ptr<const Packet> packet);
    
    ns3::Time GetRelativeTime(const Time &time);
    uint64_t GetHashValue(const Ipv4Address src, const Ipv4Address dst, const uint16_t srcPort, const uint16_t dstPort, const uint8_t protocol);
public:
    SwitchMonitor(const Time &startTime, const Time &duration, const Ptr<Node> &txNode, const string &monitorTag);
    void AddAppKey(AppKey appKey);
    std::string GetMonitorTag() const;
    void SavePacketRecords(const string &filename);
};

#endif //ECC_SWITCHMONITOR_H
