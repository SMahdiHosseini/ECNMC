// Created by Mahdi Hosseini
//
// ShapingTrafficControlLayer implements "traffic differentiation" at ingress: a configurable
// fraction of one specific source->destination pair's TCP flows are delayed by a token-bucket
// filter *before* they are handed up to routing/TrafficControl on this node, while every other
// packet passes through unmodified. By the time a (possibly delayed) packet reaches the node's
// normal Ipv4L3Protocol -> routing -> TrafficControlLayer::Send() -> RedQueueDisc path, it is
// indistinguishable from any other freshly-arrived packet, so RedQueueDisc itself stays entirely
// vanilla and every flow (shaped or not) genuinely shares the same RED queue/AQM state.
//
// This is implemented as an ingress hook rather than as a second queue-disc stage because
// ns3::QueueDisc::Enqueue() synchronously asserts that every accepted packet is immediately
// reflected in the (private, not subclass-accessible) enqueued-packet counters -- there is no
// supported way for a queue disc to accept a packet now and only really place it into a shared
// child queue disc later. TrafficControlLayer::Receive() is virtual specifically to let
// subclasses intervene on the receive path, which is what is used here: a fraction of one flow's
// packets are held in a small per-flow, order-preserving buffer and re-delivered later via
// Simulator::Schedule, calling TrafficControlLayer::Receive() directly (bypassing virtual
// dispatch) so the delayed delivery cannot be reclassified/re-buffered a second time.
//
// To install: aggregate an instance of this class onto the target node with
// node->AggregateObject(...) *before* InternetStackHelper::Install()/InstallAll() runs for that
// node -- InternetStackHelper only creates a plain ns3::TrafficControlLayer if the node doesn't
// already have one aggregated (see InternetStackHelper::CreateAndAggregateObjectFromTypeId()).

#ifndef SHAPING_TRAFFIC_CONTROL_LAYER_H
#define SHAPING_TRAFFIC_CONTROL_LAYER_H

#include "ns3/traffic-control-layer.h"
#include "ns3/ipv4-address.h"
#include "ns3/ipv4-header.h"
#include "ns3/tcp-header.h"
#include "ns3/data-rate.h"
#include "ns3/nstime.h"
#include "ns3/random-variable-stream.h"

#include <fstream>
#include <map>
#include <string>

namespace ns3
{

class ShapingTrafficControlLayer : public TrafficControlLayer
{
  public:
    static TypeId GetTypeId();

    ShapingTrafficControlLayer();

    TypeId GetInstanceTypeId() const override;

    void Receive(Ptr<NetDevice> device,
                 Ptr<const Packet> p,
                 uint16_t protocol,
                 const Address& from,
                 const Address& to,
                 NetDevice::PacketType packetType) override;

    /**
     * Every packet classified as eligible for shaping (i.e. matching the configured flow, whether
     * or not it actually incurs any extra delay because tokens happened to be available) is
     * appended as one row to this CSV file, alongside the extra delay TBF added to it.
     * \param filepath path to the CSV file to (re)create.
     */
    void SetShapedPacketsLogFile(std::string filepath);
    std::string GetShapedPacketsLogFile() const;

  private:
    void LogShapedPacket(const Ipv4Header& ipHeader,
                         const TcpHeader& tcpHeader,
                         Ptr<const Packet> p,
                         Time arrivalTime,
                         Time extraDelay);

    // Delivers a packet to the normal (base class) receive path. Used both for the immediate,
    // unshaped case and as the target of the delayed Simulator::Schedule call for shaped
    // packets, so that a re-delivered packet is never re-classified/re-buffered.
    void DeliverNow(Ptr<NetDevice> device,
                    Ptr<const Packet> p,
                    uint16_t protocol,
                    Address from,
                    Address to,
                    NetDevice::PacketType packetType);

    Ipv4Address m_flowRedirectSrc; //!< Source address of flows eligible for shaping (0.0.0.0 = disabled)
    Ipv4Address m_flowRedirectDst; //!< Destination address of flows eligible for shaping
    double m_flowRedirectFraction = 0.0; //!< Fraction of matching TCP flows shaped
    std::map<uint32_t, bool> m_flowRedirectDecisions; //!< Sticky per-flow decision, keyed by (srcPort << 16) | dstPort

    DataRate m_shapingRate;      //!< Token bucket fill rate
    uint32_t m_shapingBurst = 0; //!< Token bucket size, in bytes

    bool m_tokenBucketInitialized = false;
    int64_t m_btokens = 0; //!< Tokens currently available, in bytes
    Time m_timeCheckpoint;  //!< Last time the token bucket was updated
    Time m_nextAvailableTime = Time(0); //!< Earliest delivery time of the next shaped packet (preserves per-flow order)
    Ptr<UniformRandomVariable> m_random; //!< Random variable for per-flow shaping decisions
    std::string m_shapedPacketsLogPath;
    std::ofstream m_shapedPacketsLogStream;
    bool m_shapedPacketsLogInitialized = false;
};

} // namespace ns3

#endif /* SHAPING_TRAFFIC_CONTROL_LAYER_H */
