#include "ShapingTrafficControlLayer.h"

#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/ipv4-header.h"
#include "ns3/tcp-header.h"
#include "ns3/double.h"
#include "ns3/uinteger.h"
#include "ns3/string.h"
#include "ns3/packet.h"

#include <cmath>
#include <cstdlib>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("ShapingTrafficControlLayer");

NS_OBJECT_ENSURE_REGISTERED(ShapingTrafficControlLayer);

// EtherType used for IPv4 packets (ns3::Ipv4L3Protocol::PROT_NUMBER); duplicated here as a
// literal to avoid an extra include, matching how the rest of this project hardcodes protocol
// numbers (e.g. TCP's protocol number 6) rather than pulling in the defining header.
static constexpr uint16_t IPV4_PROT_NUMBER = 0x0800;

TypeId
ShapingTrafficControlLayer::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::ShapingTrafficControlLayer")
            .SetParent<TrafficControlLayer>()
            .SetGroupName("TrafficControl")
            .AddConstructor<ShapingTrafficControlLayer>()
            .AddAttribute("FlowRedirectSrcAddress",
                          "Source address of TCP flows eligible for shaping "
                          "(0.0.0.0 disables the feature).",
                          Ipv4AddressValue(Ipv4Address("0.0.0.0")),
                          MakeIpv4AddressAccessor(&ShapingTrafficControlLayer::m_flowRedirectSrc),
                          MakeIpv4AddressChecker())
            .AddAttribute("FlowRedirectDstAddress",
                          "Destination address of TCP flows eligible for shaping.",
                          Ipv4AddressValue(Ipv4Address("0.0.0.0")),
                          MakeIpv4AddressAccessor(&ShapingTrafficControlLayer::m_flowRedirectDst),
                          MakeIpv4AddressChecker())
            .AddAttribute("FlowRedirectFraction",
                          "Fraction (0-1) of matching TCP flows (identified by their 4-tuple, "
                          "with the decision made once per flow and reused for every packet of "
                          "that flow) that are delayed by the token-bucket shaper before being "
                          "handed up to routing/TrafficControl on this node.",
                          DoubleValue(0.0),
                          MakeDoubleAccessor(&ShapingTrafficControlLayer::m_flowRedirectFraction),
                          MakeDoubleChecker<double>(0.0, 1.0))
            .AddAttribute("ShapingRate",
                          "Rate at which tokens enter the shaping bucket.",
                          DataRateValue(DataRate("125KB/s")),
                          MakeDataRateAccessor(&ShapingTrafficControlLayer::m_shapingRate),
                          MakeDataRateChecker())
            .AddAttribute("ShapingBurst",
                          "Size of the shaping token bucket, in bytes.",
                          UintegerValue(125000),
                          MakeUintegerAccessor(&ShapingTrafficControlLayer::m_shapingBurst),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("ShapedPacketsLogFile",
                          "If non-empty, path to a CSV file recording every packet classified as "
                          "eligible for shaping (whether or not it actually incurred any extra "
                          "delay, e.g. because tokens happened to be available), along with the "
                          "extra delay the token bucket added to it.",
                          StringValue(""),
                          MakeStringAccessor(&ShapingTrafficControlLayer::SetShapedPacketsLogFile,
                                             &ShapingTrafficControlLayer::GetShapedPacketsLogFile),
                          MakeStringChecker());
    return tid;
}

ShapingTrafficControlLayer::ShapingTrafficControlLayer()
    : TrafficControlLayer()
{
    NS_LOG_FUNCTION(this);
    m_random = CreateObject<UniformRandomVariable>();
    m_random->SetAttribute ("Min", DoubleValue (0.0));
    m_random->SetAttribute ("Max", DoubleValue (1.0));
}

TypeId
ShapingTrafficControlLayer::GetInstanceTypeId() const
{
    // TrafficControlLayer::GetInstanceTypeId() always returns TrafficControlLayer::GetTypeId(),
    // which would make SetAttribute() for the attributes added above (looked up via
    // GetInstanceTypeId()) silently fail to find them on an instance of this subclass.
    return GetTypeId();
}

void
ShapingTrafficControlLayer::SetShapedPacketsLogFile(std::string filepath)
{
    m_shapedPacketsLogPath = filepath;
    if (filepath.empty())
    {
        return;
    }
    m_shapedPacketsLogStream.open(filepath, std::ios::out | std::ios::trunc);
    m_shapedPacketsLogStream << "SourceIp,SourcePort,DestinationIp,DestinationPort,SequenceNb,"
                                "ACKNb,Id,PayloadSize,ArrivalTime,ExtraDelay,ReleaseTime\n";
    m_shapedPacketsLogInitialized = true;
}

std::string
ShapingTrafficControlLayer::GetShapedPacketsLogFile() const
{
    return m_shapedPacketsLogPath;
}

void
ShapingTrafficControlLayer::LogShapedPacket(const Ipv4Header& ipHeader,
                                            const TcpHeader& tcpHeader,
                                            Ptr<const Packet> p,
                                            Time arrivalTime,
                                            Time extraDelay)
{
    if (!m_shapedPacketsLogInitialized)
    {
        return;
    }
    uint32_t headerBytes = ipHeader.GetSerializedSize() + tcpHeader.GetSerializedSize();
    uint32_t payloadSize = p->GetSize() > headerBytes ? p->GetSize() - headerBytes : 0;
    m_shapedPacketsLogStream << ipHeader.GetSource() << "," << tcpHeader.GetSourcePort() << ","
                             << ipHeader.GetDestination() << "," << tcpHeader.GetDestinationPort()
                             << "," << tcpHeader.GetSequenceNumber() << ","
                             << tcpHeader.GetAckNumber() << "," << ipHeader.GetIdentification() << ","
                             << payloadSize << "," << arrivalTime.GetNanoSeconds() << ","
                             << extraDelay.GetNanoSeconds() << ","
                             << (arrivalTime + extraDelay).GetNanoSeconds() << "\n";
}

void
ShapingTrafficControlLayer::DeliverNow(Ptr<NetDevice> device,
                                       Ptr<const Packet> p,
                                       uint16_t protocol,
                                       Address from,
                                       Address to,
                                       NetDevice::PacketType packetType)
{
    // Qualified call: always the base class's normal dispatch, regardless of virtual overrides,
    // so a delayed/re-delivered packet is never reclassified or re-buffered by Receive() below.
    TrafficControlLayer::Receive(device, p, protocol, from, to, packetType);
}

void
ShapingTrafficControlLayer::Receive(Ptr<NetDevice> device,
                                    Ptr<const Packet> p,
                                    uint16_t protocol,
                                    const Address& from,
                                    const Address& to,
                                    NetDevice::PacketType packetType)
{
    NS_LOG_FUNCTION(this << device << p << protocol << from << to << packetType);

    bool shape = false;
    Ipv4Header ipHeader;
    TcpHeader tcpHeader;

    if (m_flowRedirectFraction > 0.0 && protocol == IPV4_PROT_NUMBER)
    {
        Ptr<Packet> pktCopy = p->Copy();
        pktCopy->RemoveHeader(ipHeader);

        if (ipHeader.GetProtocol() == 6 && ipHeader.GetSource() == m_flowRedirectSrc &&
            ipHeader.GetDestination() == m_flowRedirectDst)
        {
            pktCopy->PeekHeader(tcpHeader);
            uint32_t key = (static_cast<uint32_t>(tcpHeader.GetSourcePort()) << 16) |
                           tcpHeader.GetDestinationPort();

            auto it = m_flowRedirectDecisions.find(key);
            if (it == m_flowRedirectDecisions.end())
            {
                shape = m_random->GetValue() < m_flowRedirectFraction;
                m_flowRedirectDecisions[key] = shape;
            }
            else
            {
                shape = it->second;
            }
        }
    }

    if (!shape)
    {
        DeliverNow(device, p, protocol, from, to, packetType);
        return;
    }

    if (!m_tokenBucketInitialized)
    {
        m_btokens = m_shapingBurst;
        m_timeCheckpoint = Simulator::Now();
        m_tokenBucketInitialized = true;
    }

    uint32_t pktSize = p->GetSize();
    Time now = Simulator::Now();
    double delta = (now - m_timeCheckpoint).GetSeconds();

    int64_t btoks = m_btokens + std::llround(delta * (m_shapingRate.GetBitRate() / 8.0));
    if (btoks > static_cast<int64_t>(m_shapingBurst))
    {
        btoks = m_shapingBurst;
    }
    btoks -= pktSize;

    Time releaseTime;
    if (btoks >= 0)
    {
        releaseTime = now;
    }
    else
    {
        releaseTime = now + m_shapingRate.CalculateBytesTxTime(-btoks);
    }
    m_timeCheckpoint = now;
    m_btokens = btoks;

    // Preserve per-flow packet ordering: never let a later-arriving packet of this flow be
    // delivered before an earlier one that is still waiting on its own release time.
    if (releaseTime < m_nextAvailableTime)
    {
        releaseTime = m_nextAvailableTime;
    }
    m_nextAvailableTime = releaseTime;

    Time extraDelay = releaseTime > now ? (releaseTime - now) : Time(0);
    LogShapedPacket(ipHeader, tcpHeader, p, now, extraDelay);

    if (releaseTime <= now)
    {
        NS_LOG_LOGIC("Shaped packet delivered immediately (tokens available)");
        DeliverNow(device, p, protocol, from, to, packetType);
    }
    else
    {
        NS_LOG_LOGIC("Shaped packet delayed by " << (releaseTime - now).As(Time::US));
        Simulator::Schedule(releaseTime - now,
                            &ShapingTrafficControlLayer::DeliverNow,
                            this,
                            device,
                            p,
                            protocol,
                            from,
                            to,
                            packetType);
    }
}

} // namespace ns3
