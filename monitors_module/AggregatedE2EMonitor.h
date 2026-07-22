#ifndef AGGREGATED_E2E_MONITOR_H
#define AGGREGATED_E2E_MONITOR_H

#include "PacketKey.h"

#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ns3;

/**
 * A single end-to-end monitor for an all-to-all workload.
 *
 * The previous implementation created one E2EMonitor per source. Every one
 * subscribed to every node's IPv4 Rx trace, so a received packet was parsed
 * once per source host. This class subscribes once and dispatches completed
 * packets to source-specific output streams.
 */
class AggregatedE2EMonitor
{
  public:
    AggregatedE2EMonitor(const Time& startTime,
                         const Time& duration,
                         const Time& steadyStartTime,
                         const Time& steadyStopTime,
                         const DataRate& hostToTorLinkRate,
                         const DataRate& torToAggLinkRate,
                         const Time& hostToTorLinkDelay,
                         uint32_t numOfPaths,
                         uint32_t numOfSegments);
    ~AggregatedE2EMonitor();

    void AddSource(Ptr<PointToPointNetDevice> netDevice,
                   Ipv4Address sourceAddress,
                   const std::string& monitorTag);
    void SaveMonitorRecords(const std::string& outputDirectory);
    void FlushStreams();

  private:
    struct PacketEvent
    {
        explicit PacketEvent(const PacketKey& packetKey, uint32_t owner)
            : key(packetKey), sourceIndex(owner)
        {
        }

        PacketKey key;
        uint32_t sourceIndex;
        Time sentTime = Time(-1);
        Time receivedTime = Time(-1);
        int path = 0;
        bool ecn = false;
    };

    struct SourceState
    {
        Ptr<PointToPointNetDevice> netDevice;
        Ipv4Address address;
        std::string monitorTag;
        std::ofstream packetsFileStream;
        bool packetsFileInitialized = false;
    };

    void Connect();
    void Disconnect();
    static void CaptureTrampoline(AggregatedE2EMonitor* monitor,
                                  uint32_t sourceIndex,
                                  Ptr<const Packet> packet);
    void Capture(uint32_t sourceIndex, Ptr<const Packet> packet);
    void RecordIpv4PacketReceived(Ptr<const Packet> packet, Ptr<Ipv4> ipv4, uint32_t interface);
    void InitializePacketsFile(SourceState& source, const std::string& outputDirectory);
    uint64_t GetHashValue(Ipv4Address src,
                          Ipv4Address dst,
                          uint16_t srcPort,
                          uint16_t dstPort,
                          uint8_t protocol);
    Time CalculateTransmissionDelay(uint32_t packetSize) const;

    Time m_startTime;
    Time m_duration;
    Time m_steadyStartTime;
    Time m_steadyStopTime;
    Time m_stalePacketTimeout = MilliSeconds(50);
    DataRate m_hostToTorLinkRate;
    DataRate m_torToAggLinkRate;
    Time m_hostToTorLinkDelay;
    uint32_t m_numOfPaths;
    uint32_t m_numOfSegments;
    Hasher m_hasher;

    std::vector<SourceState> m_sources;
    std::unordered_map<uint32_t, uint32_t> m_sourceByAddress;
    std::unordered_map<PacketKey, PacketEvent, PacketKeyHash> m_recordedPackets;
};

#endif // AGGREGATED_E2E_MONITOR_H
