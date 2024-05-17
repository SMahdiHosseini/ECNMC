//
// Created by Mahdi Hosseini on 24.04.24.
//

#ifndef ECC_POISSONSAMPLER_H
#define ECC_POISSONSAMPLER_H

#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/queue-item.h"
#include "ns3/point-to-point-module.h"
#include "ns3/red-queue-disc.h"

#include "PacketKey.h"

#include <ostream>
#include <unordered_map>

using namespace ns3;
using namespace std;

struct samplingEvent {

private:
    PacketKey* _key;
    ns3::Time _sampleTime = Time(-1);
    ns3::Time _departureTime = Time(-1);

public:
    explicit samplingEvent(PacketKey *key);

    [[nodiscard]] PacketKey* GetPacketKey() const;
    [[nodiscard]] Time GetSampleTime() const;
    [[nodiscard]] Time GetDepartureTime() const;
    [[nodiscard]] bool IsDeparted() const;

    void SetSampleTime();
    void SetDepartureTime();

    friend ostream &operator<<(ostream &os, const samplingEvent &event);
};

class PoissonSampler {

private:
    ns3::Time _startTime = Seconds(0);
    ns3::Time _duration = Seconds(0);

    Ptr<ExponentialRandomVariable> m_var;
    Ptr<RedQueueDisc> REDQueueDisc;
    Ptr<Queue<Packet>> NetDeviceQueue;
    Ptr<PointToPointNetDevice> outgoingNetDevice;
    std::string _sampleTag;
    double _sampleRate;
    std::unordered_map<PacketKey, samplingEvent*, PacketKeyHash> _recordedSamples;
    int zeroDelayPort;
    uint32_t droppedPackets;

    Ptr<const QueueDiscItem> lastItem;
    Ptr<const Packet> lastPacket;

    void Connect(Ptr<PointToPointNetDevice> outgoingNetDevice);
    void Disconnect(Ptr<PointToPointNetDevice> outgoingNetDevice);
    void EnqueueQueueDisc(Ptr<const QueueDiscItem> item);
    void EnqueueNetDeviceQueue(Ptr< const Packet > packet);
    void EventHandler();
    void RecordPacket(Ptr<const Packet> packet);
    ns3::Time GetRelativeTime(const Time &time);

public:
    PoissonSampler(const Time &startTime, const Time &duration, Ptr<RedQueueDisc> queueDisc, Ptr<Queue<Packet>> queue, Ptr<PointToPointNetDevice> outgoingNetDevice, const string &sampleTag, double sampleRate);
    std::string GetSampleTag() const;
    void SaveSamples(const string &filename);
};

#endif //ECC_SWITCHMONITOR_H
