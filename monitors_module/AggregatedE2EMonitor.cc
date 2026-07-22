#include "AggregatedE2EMonitor.h"

#include "ns3/ipv4-header.h"

#include <sstream>

AggregatedE2EMonitor::AggregatedE2EMonitor(const Time& startTime,
                                           const Time& duration,
                                           const Time& steadyStartTime,
                                           const Time& steadyStopTime,
                                           const DataRate& hostToTorLinkRate,
                                           const DataRate& torToAggLinkRate,
                                           const Time& hostToTorLinkDelay,
                                           uint32_t numOfPaths,
                                           uint32_t numOfSegments)
    : m_startTime(startTime),
      m_duration(duration),
      m_steadyStartTime(steadyStartTime),
      m_steadyStopTime(steadyStopTime),
      m_hostToTorLinkRate(hostToTorLinkRate),
      m_torToAggLinkRate(torToAggLinkRate),
      m_hostToTorLinkDelay(hostToTorLinkDelay),
      m_numOfPaths(numOfPaths),
      m_numOfSegments(numOfSegments)
{
    Simulator::Schedule(m_startTime, &AggregatedE2EMonitor::Connect, this);
    Simulator::Schedule(m_startTime + m_duration, &AggregatedE2EMonitor::Disconnect, this);
}

AggregatedE2EMonitor::~AggregatedE2EMonitor()
{
    for (auto& source : m_sources)
    {
        if (source.packetsFileStream.is_open())
        {
            source.packetsFileStream.close();
        }
    }
}

void
AggregatedE2EMonitor::AddSource(Ptr<PointToPointNetDevice> netDevice,
                                Ipv4Address sourceAddress,
                                const std::string& monitorTag)
{
    NS_ABORT_MSG_IF(m_sourceByAddress.count(sourceAddress.Get()) != 0,
                    "Duplicate source address in AggregatedE2EMonitor: " << sourceAddress);

    uint32_t sourceIndex = m_sources.size();
    m_sourceByAddress.emplace(sourceAddress.Get(), sourceIndex);
    SourceState source;
    source.netDevice = netDevice;
    source.address = sourceAddress;
    source.monitorTag = monitorTag;
    m_sources.push_back(std::move(source));
}

void
AggregatedE2EMonitor::Connect()
{
    for (uint32_t sourceIndex = 0; sourceIndex < m_sources.size(); ++sourceIndex)
    {
        m_sources[sourceIndex].netDevice->TraceConnectWithoutContext(
            "PromiscSniffer",
            MakeBoundCallback(&AggregatedE2EMonitor::CaptureTrampoline, this, sourceIndex));
    }
    Config::ConnectWithoutContext("/NodeList/*/$ns3::Ipv4L3Protocol/Rx",
                                  MakeCallback(&AggregatedE2EMonitor::RecordIpv4PacketReceived,
                                               this));
}

void
AggregatedE2EMonitor::Disconnect()
{
    for (uint32_t sourceIndex = 0; sourceIndex < m_sources.size(); ++sourceIndex)
    {
        m_sources[sourceIndex].netDevice->TraceDisconnectWithoutContext(
            "PromiscSniffer",
            MakeBoundCallback(&AggregatedE2EMonitor::CaptureTrampoline, this, sourceIndex));
    }
    Config::DisconnectWithoutContext("/NodeList/*/$ns3::Ipv4L3Protocol/Rx",
                                     MakeCallback(&AggregatedE2EMonitor::RecordIpv4PacketReceived,
                                                  this));
}

void
AggregatedE2EMonitor::CaptureTrampoline(AggregatedE2EMonitor* monitor,
                                        uint32_t sourceIndex,
                                        Ptr<const Packet> packet)
{
    monitor->Capture(sourceIndex, packet);
}

void
AggregatedE2EMonitor::Capture(uint32_t sourceIndex, Ptr<const Packet> packet)
{
    if (Simulator::Now() < m_steadyStartTime || Simulator::Now() > m_steadyStopTime)
    {
        return;
    }

    PacketKey* parsedKey = PacketKey::Packet2PacketKey(packet, FIRST_HEADER_PPP);
    if (parsedKey->GetSrcIp() != m_sources[sourceIndex].address ||
        m_sourceByAddress.count(parsedKey->GetDstIp().Get()) == 0)
    {
        delete parsedKey;
        return;
    }

    parsedKey->SetPacketSize(packet->GetSize());
    auto [it, inserted] =
        m_recordedPackets.try_emplace(*parsedKey, *parsedKey, sourceIndex);
    PacketEvent& event = it->second;
    if (inserted)
    {
        event.sentTime = Simulator::Now();
    }
    event.key.SetPacketSize(packet->GetSize());
    event.key.SetTagged(parsedKey->IsTagged());
    delete parsedKey;
}

void
AggregatedE2EMonitor::RecordIpv4PacketReceived(Ptr<const Packet> packet,
                                                Ptr<Ipv4> ipv4,
                                                uint32_t interface)
{
    if (Simulator::Now() < m_steadyStartTime || Simulator::Now() > m_steadyStopTime)
    {
        return;
    }

    // Ipv4L3Protocol::Rx also fires at routers. Finalize only at the node
    // which owns the packet's destination address.
    Ipv4Header header;
    packet->PeekHeader(header);
    if (ipv4->GetInterfaceForAddress(header.GetDestination()) < 0 ||
        m_sourceByAddress.count(header.GetSource().Get()) == 0 ||
        m_sourceByAddress.count(header.GetDestination().Get()) == 0)
    {
        return;
    }

    PacketKey* parsedKey = PacketKey::Packet2PacketKey(packet, FIRST_HEADER_IPV4);
    auto eventIt = m_recordedPackets.find(*parsedKey);
    delete parsedKey;
    if (eventIt == m_recordedPackets.end())
    {
        return;
    }

    PacketEvent& event = eventIt->second;
    event.receivedTime = Simulator::Now();
    event.ecn = header.GetEcn() == Ipv4Header::ECN_CE;
    event.path = GetHashValue(event.key.GetSrcIp(),
                             event.key.GetDstIp(),
                             event.key.GetSrcPort(),
                             event.key.GetDstPort(),
                             header.GetProtocol()) %
                 m_numOfPaths;
}

uint64_t
AggregatedE2EMonitor::GetHashValue(Ipv4Address src,
                                   Ipv4Address dst,
                                   uint16_t srcPort,
                                   uint16_t dstPort,
                                   uint8_t protocol)
{
    m_hasher.clear();
    std::ostringstream data;
    data << src << dst << protocol << dstPort << srcPort;
    return m_hasher.GetHash32(data.str());
}

Time
AggregatedE2EMonitor::CalculateTransmissionDelay(uint32_t packetSize) const
{
    if (m_numOfSegments == 3)
    {
        return m_hostToTorLinkRate.CalculateBytesTxTime(packetSize) * 2 +
               m_torToAggLinkRate.CalculateBytesTxTime(packetSize) * 2 +
               m_hostToTorLinkDelay * 4;
    }
    if (m_numOfSegments == 1)
    {
        return m_hostToTorLinkRate.CalculateBytesTxTime(packetSize) +
               m_torToAggLinkRate.CalculateBytesTxTime(packetSize) +
               m_hostToTorLinkDelay * 2;
    }
    return Seconds(0);
}

void
AggregatedE2EMonitor::InitializePacketsFile(SourceState& source,
                                             const std::string& outputDirectory)
{
    if (source.packetsFileInitialized)
    {
        return;
    }
    const std::string filename = outputDirectory + "/" + source.monitorTag +
                                 "_EndToEnd_packets.csv";
    source.packetsFileStream.open(filename, std::ios::out | std::ios::trunc);
    NS_ABORT_MSG_IF(!source.packetsFileStream.is_open(),
                    "Could not open aggregate E2E output file: " << filename);
    source.packetsFileStream
        << "SourceIp,SourcePort,DestinationIp,DestinationPort,SequenceNb,ACKNb,Id,"
           "PayloadSize,Path,SentTime,IsReceived,ReceiveTime,transmissionDelay,ECN,Tagged\n";
    source.packetsFileInitialized = true;
}

void
AggregatedE2EMonitor::SaveMonitorRecords(const std::string& outputDirectory)
{
    for (auto& source : m_sources)
    {
        InitializePacketsFile(source, outputDirectory);
    }

    for (auto it = m_recordedPackets.begin(); it != m_recordedPackets.end();)
    {
        PacketEvent& event = it->second;
        const bool received = event.receivedTime != Time(-1);
        const bool stale = event.sentTime != Time(-1) &&
                           Simulator::Now() - event.sentTime > m_stalePacketTimeout;
        if (!received && !stale)
        {
            ++it;
            continue;
        }

        SourceState& source = m_sources[event.sourceIndex];
        const PacketKey& key = event.key;
        source.packetsFileStream << key.GetSrcIp() << ',' << key.GetSrcPort() << ','
                                 << key.GetDstIp() << ',' << key.GetDstPort() << ','
                                 << key.GetSeqNb() << ',' << key.GetAckNb() << ','
                                 << key.GetId() << ',' << key.GetPacketSize() << ','
                                 << event.path << ',' << event.sentTime.GetNanoSeconds() << ','
                                 << received << ',' << event.receivedTime.GetNanoSeconds() << ',';
        if (received)
        {
            source.packetsFileStream
                << CalculateTransmissionDelay(key.GetPacketSize()).GetNanoSeconds() << ','
                << event.ecn << ',' << key.IsTagged() << '\n';
        }
        else
        {
            source.packetsFileStream << "-1,-1,-1\n";
        }
        it = m_recordedPackets.erase(it);
    }
}

void
AggregatedE2EMonitor::FlushStreams()
{
    for (auto& source : m_sources)
    {
        if (source.packetsFileStream.is_open())
        {
            source.packetsFileStream.flush();
        }
    }
}
