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
    double _lastMarkingProb = 0;
    double _lossProb = 0;
    double _lastLossProb = 0;
    uint32_t _queueSize = 0;
    uint32_t _lastQueueSize = 0;
    uint32_t _totalQueueSize = 0;
    uint32_t _lastTotalQueueSize = 0;
    string _label;
    string _eventAction;
public:
    explicit samplingEvent(PacketKey *key);
    explicit samplingEvent();
    [[nodiscard]] PacketKey* GetPacketKey() const;
    [[nodiscard]] Time GetSampleTime() const;
    [[nodiscard]] Time GetDepartureTime() const;
    [[nodiscard]] double GetMarkingProb() const;
    [[nodiscard]] double GetLossProb() const;
    [[nodiscard]] uint32_t GetQueueSize() const;
    [[nodiscard]] uint32_t GetTotalQueueSize() const;
    [[nodiscard]] double GetLastMarkingProb() const;
    [[nodiscard]] string GetLabel() const;
    [[nodiscard]] string GetEventAction() const; 
    [[nodiscard]] bool IsDeparted() const;
    [[nodiscard]] double GetLastDropProb() const;
    [[nodiscard]] uint32_t GetLastQueueSize() const;
    [[nodiscard]] uint32_t GetLastTotalQueueSize() const;

    void SetSampleTime();
    void SetSampleTime(Time t);
    void SetDepartureTime();
    void SetDepartureTime(Time t);
    void SetMarkingProb(double markingProb);
    void SetLossProb(double lossProb);
    void SetQueueSize(uint32_t size);
    void SetTotalQueueSize(uint32_t size);
    void SetLastMarkingProb(double markingProb);
    void SetLabel(const string label);
    void SetEventAction(const string action);
    void SetLastDropProb(double lossProb);
    void SetLastQueueSize(uint32_t size);
    void SetLastTotalQueueSize(uint32_t size);
};

class PoissonSampler : public Monitor{

private:

    Ptr<ExponentialRandomVariable> m_var;
    Ptr<RedQueueDisc> REDQueueDisc;
    Ptr<Queue<Packet>> NetDeviceQueue;
    Ptr<PointToPointNetDevice> outgoingNetDevice;
    Ptr<PointToPointNetDevice> incomingNetDevice;
    Ptr<PointToPointNetDevice> incomingNetDevice_1;
    double _sampleRate;
    std::unordered_map<PacketKey, samplingEvent*, PacketKeyHash> _recordedSamples;
    int zeroDelayPort;
    double samplesMarkingProbMean;
    double samplesMarkingProbVariance;
    double samplesLossProbMean;
    double samplesLossProbVariance;
    Ptr<const QueueDiscItem> lastItem;
    Time lastItemTime;
    Ptr<const Packet> lastPacket;
    Time lastPacketTime;
    uint32_t numOfGTSamples;
    double GTPacketSizeMean;
    double GTDropMean;
    double GTQueuingDelay;
    double GTMarkingProbMean;
    std::vector<std::tuple<Time, samplingEvent>> queueSizeProcess;
    std::vector<std::tuple<Time, samplingEvent>> queueSizeProcessByPackets;
    Time firstItemTime;
    PacketCDF packetCDF;
    Time lastLeftTime;
    uint32_t lastLeftSize;
    DataRate outgoingDataRate;
    double _lastDropProb;
    uint32_t _lastQueueSize;
    uint32_t _lastTotalQueueSize;
    
    void Connect(Ptr<PointToPointNetDevice> outgoingNetDevice);
    void Disconnect(Ptr<PointToPointNetDevice> outgoingNetDevice);
    void EnqueueQueueDisc(Ptr<const QueueDiscItem> item);
    void DequeueQueueDisc(Ptr<const QueueDiscItem> item);
    void EnqueueNetDeviceQueue(Ptr< const Packet > packet);
    void EventHandler();
    void RecordPacket(Ptr<const Packet> packet);
    void RecordIncomingPacket(Ptr<const Packet> packet);
    void updateCounters(samplingEvent* event);
    void loadCDFData(const std::string& filename);
    void updateGTCounters();
    uint32_t ComputeQueueSize();
public:
    PoissonSampler(const Time &steadyStartTime, const Time &steadyStopTime, Ptr<RedQueueDisc> queueDisc, Ptr<Queue<Packet>> queue, Ptr<PointToPointNetDevice> outgoingNetDevice, const string &sampleTag, double sampleRate);
    PoissonSampler(const Time &steadyStartTime, const Time &steadyStopTime, Ptr<RedQueueDisc> queueDisc, Ptr<Queue<Packet>> queue, Ptr<PointToPointNetDevice> outgoingNetDevice, const string &sampleTag, double sampleRate, Ptr<PointToPointNetDevice> incomingNetDevice, Ptr<PointToPointNetDevice> incomingNetDevice_1);
    void SaveMonitorRecords(const string &filename);
};

#endif //ECC_SWITCHMONITOR_H
