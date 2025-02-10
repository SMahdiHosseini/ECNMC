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
#include "Monitor.h"
#include "PacketKey.h"
#include "PacketCDF.h"

#include <ostream>
#include <unordered_map>

using namespace ns3;
using namespace std;

struct samplingEvent : MonitorEvent{

private:
    double _markingProb = 0;

public:
    explicit samplingEvent(PacketKey *key);

    [[nodiscard]] PacketKey* GetPacketKey() const;
    [[nodiscard]] Time GetSampleTime() const;
    [[nodiscard]] Time GetDepartureTime() const;
    [[nodiscard]] double GetMarkingProb() const;
    [[nodiscard]] bool IsDeparted() const;

    void SetSampleTime();
    void SetSampleTime(Time t);
    void SetDepartureTime();
    void SetDepartureTime(Time t);
    void SetMarkingProb(double markingProb);
};

class PoissonSampler : public Monitor{

private:

    Ptr<ExponentialRandomVariable> m_var;
    Ptr<RedQueueDisc> REDQueueDisc;
    Ptr<Queue<Packet>> NetDeviceQueue;
    Ptr<PointToPointNetDevice> outgoingNetDevice;
    double _sampleRate;
    std::unordered_map<PacketKey, samplingEvent*, PacketKeyHash> _recordedSamples;
    int zeroDelayPort;
    double samplesDropMean;
    double samplesDropVariance;
    Ptr<const QueueDiscItem> lastItem;
    Time lastItemTime;
    Ptr<const Packet> lastPacket;
    Time lastPacketTime;
    uint32_t numOfGTSamples;
    double GTPacketSizeMean;
    double GTDropMean;
    double GTQueuingDelay;
    std::vector<std::tuple<Time, uint32_t>> queueSizeProcess;
    Time firstItemTime;
    PacketCDF packetCDF;
    Time lastLeftTime;
    uint32_t lastLeftSize;
    DataRate outgoingDataRate;
    
    void Connect(Ptr<PointToPointNetDevice> outgoingNetDevice);
    void Disconnect(Ptr<PointToPointNetDevice> outgoingNetDevice);
    void EnqueueQueueDisc(Ptr<const QueueDiscItem> item);
    void EnqueueNetDeviceQueue(Ptr< const Packet > packet);
    void EventHandler();
    void RecordPacket(Ptr<const Packet> packet);
    void updateCounters(samplingEvent* event);
    void loadCDFData(const std::string& filename);
    uint32_t ComputeQueueSize();
public:
    PoissonSampler(const Time &steadyStartTime, const Time &steadyStopTime, Ptr<RedQueueDisc> queueDisc, Ptr<Queue<Packet>> queue, Ptr<PointToPointNetDevice> outgoingNetDevice, const string &sampleTag, double sampleRate);
    void SaveMonitorRecords(const string &filename);
};

#endif //ECC_SWITCHMONITOR_H
