//
// Created by Mahdi Hosseini on 16.10.2024
//

#ifndef ECC_BURSTMONITOR_H
#define ECC_BURSTMONITOR_H

#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/queue-item.h"
#include "ns3/point-to-point-module.h"
#include "ns3/red-queue-disc.h"
#include "Monitor.h"
#include "PacketKey.h"

#include <ostream>
#include <unordered_map>

using namespace ns3;
using namespace std;

struct BurstSamplingEvent {
private:
    Time _sampleTime;
    bool _isHotThroughputUtilization;
    uint32_t queueSize;

public:
    [[nodiscard]] Time GetSampleTime() const;
    [[nodiscard]] bool IsHotThroughputUtilization() const;
    [[nodiscard]] BurstSamplingEvent();
    [[nodiscard]] BurstSamplingEvent(Time sampleTime, bool isHotThroughputUtilization, uint32_t queueSize);
    [[nodiscard]] uint32_t GetQueueSize() const;

    void SetSampleTime();
    void SetSampleTime(Time t);
    void SetHotThroughputUtilization(bool isHotThroughputUtilization);
};

class BurstMonitor {

private:
    std::unordered_map<PacketKey, bool, PacketKeyHash> _recordedPacketKeys;
    std::vector<BurstSamplingEvent> _recordedSamples;
    Time sampleInterval;
    uint64_t byteCount;
    uint64_t lastByteCount;
    std::string _sampleTag;
    DataRate linkRate;
    Ptr<RedQueueDisc> REDQueueDisc;
    Ptr<PointToPointNetDevice> outgoingNetDevice;
    void Connect(Ptr<PointToPointNetDevice> outgoingNetDevice);
    void Disconnect(Ptr<PointToPointNetDevice> outgoingNetDevice);
    void EventHandler();
    void RecordPacket(Ptr<const Packet> packet);
    void EnqueueNetDeviceQueue(Ptr<const Packet> packet);

public:
    BurstMonitor(Time steadyStopTime, Ptr<PointToPointNetDevice> outgoingNetDevice, Ptr<RedQueueDisc> _REDQueueDisc, const string &sampleTag, Time sampleInterval, const DataRate &_linkRate);
    void SaveRecords(const string &filename);
    string GetSampleTag() const;
};

#endif //ECC_BURSTMONITOR_H
