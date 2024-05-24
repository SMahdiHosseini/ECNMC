//
// Created by Mahdi Hosseini on 24.04.24.
//

#ifndef ECC_REGULARSAMPLER_H
#define ECC_REGULARSAMPLER_H

#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/queue-item.h"
#include "ns3/point-to-point-module.h"
#include "ns3/red-queue-disc.h"
#include "PoissonSampler.h"

#include "PacketKey.h"

#include <ostream>
#include <unordered_map>

using namespace ns3;
using namespace std;

class RegularSampler {

private:
    ns3::Time _startTime = Seconds(0);
    ns3::Time _duration = Seconds(0);

    Ptr<RedQueueDisc> REDQueueDisc;
    Ptr<Queue<Packet>> NetDeviceQueue;
    Ptr<PointToPointNetDevice> outgoingNetDevice;
    std::string _sampleTag;
    ns3::Time _samplePeriod;
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
    RegularSampler(const Time &startTime, const Time &duration, Ptr<RedQueueDisc> queueDisc, Ptr<Queue<Packet>> queue, Ptr<PointToPointNetDevice> outgoingNetDevice, const string &sampleTag, const Time &samplePeriod);

    [[nodiscard]]
    std::string GetSampleTag() const;
    void SaveSamples(const string &filename);
};

#endif //ECC_SWITCHMONITOR_H
