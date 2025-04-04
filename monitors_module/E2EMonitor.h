//
// Created by Zeinab Shmeis on 28.05.20.
//

#ifndef E2EMONITOR_H
#define E2EMONITOR_H

#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/random-variable-stream.h"
#include "Monitor.h"
#include "PacketKey.h"
#include "AppKey.h"
#include "PacketCDF.h"

#include <ostream>
#include <unordered_map>

using namespace ns3;
using namespace std;

struct E2EMonitorEvent : MonitorEvent {

public:
    explicit E2EMonitorEvent(PacketKey *key);

    [[nodiscard]] bool GetEcn() const;
    [[nodiscard]] int GetPath() const;
    [[nodiscard]] Time GetTxEnqueueTime() const;
    [[nodiscard]] Time GetTxDequeueTime() const;
    [[nodiscard]] Time GetTxIpTime() const;
    void SetEcn(bool ecn);
    void SetPath(int path);
    void SetTxEnqueueTime(Time time);
    void SetTxDequeueTime(Time time);
    void SetTxIpTime(Time time);
private:
    ns3::Time _TxEnqueueTime = Time(-1);
    ns3::Time _TxDequeuTime = Time(-1);
    ns3::Time _TxIpTime = Time(-1);
};

class E2EMonitor : public Monitor {

private:
    double _errorRate;
    std::vector<uint32_t> sumOfPacketSizes;
    std::vector<uint32_t> sentPackets;
    std::vector<uint32_t> sentPackets_onlink;
    std::vector<uint32_t> markedPackets;
    std::vector<Time> timeAverageIntegral;
    std::vector<Time> integralStartTime;
    std::vector<Time> integralEndTime;
    std::vector<Time> lastDelay;
    uint32_t QueueCapacity;
    DataRate hostToTorLinkRate;
    DataRate torToAggLinkRate;
    Time hostToTorLinkDelay;
    Hasher hasher;
    Ptr<UniformRandomVariable> rand;
    int numOfPaths = 2;
    int numOfSegmetns = 3;
    PacketCDF packetCDF;
    double GTDropMean;
    Time lastItemTime;
    Time firstItemTime;
    set<AppKey> _observedAppsKey;
    Ptr<Node> _txNode;
    bool _isDifferentiate;
    std::vector<std::tuple<Time, uint32_t, double, Time>> markingProbProcess;
    std::unordered_map<uint64_t, Ptr<TcpSocketBase>> tracesSockets;
    std::unordered_map<PacketKey, E2EMonitorEvent*, PacketKeyHash> _recordedPackets;
    
    void Connect(Ptr<PointToPointNetDevice> netDevice, uint32_t rxNodeId, uint32_t txNodeId);
    void Disconnect(Ptr<PointToPointNetDevice> netDevice, uint32_t rxNodeId, uint32_t txNodeId);

    void Enqueue(Ptr< const Packet > packet);
    void Capture(Ptr< const Packet > packet);
    void RecordIpv4PacketReceived(Ptr<const Packet> packet, Ptr<Ipv4> ipv4, uint32_t interface);
    void RecordIpv4PacketSent(Ptr<const Packet> packet, Ptr<Ipv4> ipv4, uint32_t interface);
    uint64_t GetHashValue(const Ipv4Address src, const Ipv4Address dst, const uint16_t srcPort, const uint16_t dstPort, const uint8_t protocol);
    void updateTimeAverageIntegral(uint32_t path, Time delay, Time endTime);
    double calculateUnbiasedGTDrop();
    void traceNewSockets();
    void markingProbUpdate(uint32_t bytesMarked, uint32_t bytesAcked, double alpha, Time rtt);
    // void NewAck(SequenceNumber32 sqn);
public:
    E2EMonitor(const Time &startTime, const Time &duration, const Time &steadyStartTime, const Time &steadyStopTime, const Ptr<PointToPointNetDevice> netDevice, const Ptr<Node> &rxNode, const string &monitorTag, const double errorRate, 
    const DataRate &hostToTorLinkRate, const DataRate &torToAggLinkRate, const Time &hostToTorLinkDelay);
    E2EMonitor(const Time &startTime, const Time &duration, const Time &steadyStartTime, const Time &steadyStopTime, const Ptr<PointToPointNetDevice> netDevice, const Ptr<Node> &rxNode, const Ptr<Node> &txNode, const string &monitorTag, const double errorRate, 
    const DataRate &hostToTorLinkRate, const DataRate &torToAggLinkRate, const Time &hostToTorLinkDelay, const int numOfPaths, const int numOfSegmetns, uint32_t queueCapacity, const bool isDifferentiate);
    void SaveMonitorRecords(const string &filename);
    void RecordPacket(Ptr<const Packet> packet);
};

#endif //E2EMONITOR_H
