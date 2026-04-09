/*
 * Copyright (c) 2007, 2008 University of Washington
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#include "point-to-point-net-device.h"

#include "point-to-point-channel.h"
#include "ppp-header.h"

#include "ns3/error-model.h"
#include "ns3/llc-snap-header.h"
#include "ns3/log.h"
#include "ns3/mac48-address.h"
#include "ns3/pointer.h"
#include "ns3/queue.h"
#include "ns3/simulator.h"
#include "ns3/trace-source-accessor.h"
#include "ns3/uinteger.h"
#include "ns3/ipv4.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("PointToPointNetDevice");

NS_OBJECT_ENSURE_REGISTERED(PointToPointNetDevice);
// ****** Mahdi Change ***** (START) ***** //
// void
// PointToPointNetDevice::DoFragmentation(Ptr<Packet> packet, Ipv4Header& ipv4Header, uint32_t _firstFragmentSize, std::list<Ipv4PayloadHeaderPair>& listFragments)
// {
//     // BEWARE: here we do assume that the header options are not present.
//     // a much more complex handling is necessary in case there are options.
//     // If (when) IPv4 option headers will be implemented, the following code shall be changed.
//     // Of course also the reassemby code shall be changed as well.

//     NS_LOG_FUNCTION(this << *packet << _firstFragmentSize << &listFragments);

//     Ptr<Packet> p = packet->Copy();

//     NS_ASSERT_MSG((ipv4Header.GetSerializedSize() == 5 * 4),
//                   "IPv4 fragmentation implementation only works without option headers.");

//     uint16_t offset = 0;
//     uint16_t originalOffset = ipv4Header.GetFragmentOffset();
//     uint32_t currentFragmentablePartSize = 0;

//     // IPv4 fragments are all 8 bytes aligned but the last.
//     // The IP payload size is:
//     // floor( ( outIfaceMtu - ipv4Header.GetSerializedSize() ) /8 ) *8
//     uint32_t firstFragmentSize = (_firstFragmentSize - ipv4Header.GetSerializedSize()) & ~uint32_t(0x7);
//     TcpHeader tcpHeader;
//     p->PeekHeader(tcpHeader);

//     NS_LOG_LOGIC("Fragmenting - Target Size: " << firstFragmentSize);

//     Ipv4Header firstFragmentHeader = ipv4Header;
//     currentFragmentablePartSize = firstFragmentSize;
//     firstFragmentHeader.SetMoreFragments();
//     firstFragmentHeader.SetFragmentOffset(offset + originalOffset);
//     firstFragmentHeader.SetPayloadSize(currentFragmentablePartSize);
//     if (Node::ChecksumEnabled())
//     {
//         firstFragmentHeader.EnableChecksum();
//     }
//     Ptr<Packet> firstFragment = p->CreateFragment(offset, currentFragmentablePartSize);
//     listFragments.emplace_back(firstFragment, firstFragmentHeader);

//     // The rest of packet goes to the next fragment
//     offset += currentFragmentablePartSize;
//     Ipv4Header secondFragmentHeader = ipv4Header;
//     currentFragmentablePartSize = p->GetSize() - offset;
//     secondFragmentHeader.SetFragmentOffset(offset + originalOffset);
//     secondFragmentHeader.SetPayloadSize(currentFragmentablePartSize + tcpHeader.GetSerializedSize()); // add the TCP header size to the packet
//     secondFragmentHeader.SetLastFragment();
//     if (Node::ChecksumEnabled())
//     {
//         secondFragmentHeader.EnableChecksum();
//     }
//     Ptr<Packet> secondFragment = p->CreateFragment(offset, currentFragmentablePartSize);
//     secondFragment->AddHeader(tcpHeader); // add TCP header to the second fragment
//     listFragments.emplace_back(secondFragment, secondFragmentHeader);
// }
// ****** Mahdi Change ***** (END) ***** //

TypeId
PointToPointNetDevice::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::PointToPointNetDevice")
            .SetParent<NetDevice>()
            .SetGroupName("PointToPoint")
            .AddConstructor<PointToPointNetDevice>()
            .AddAttribute("Mtu",
                          "The MAC-level Maximum Transmission Unit",
                          UintegerValue(DEFAULT_MTU),
                          MakeUintegerAccessor(&PointToPointNetDevice::SetMtu,
                                               &PointToPointNetDevice::GetMtu),
                          MakeUintegerChecker<uint16_t>())
            .AddAttribute("Address",
                          "The MAC address of this device.",
                          Mac48AddressValue(Mac48Address("ff:ff:ff:ff:ff:ff")),
                          MakeMac48AddressAccessor(&PointToPointNetDevice::m_address),
                          MakeMac48AddressChecker())
            .AddAttribute("DataRate",
                          "The default data rate for point to point links",
                          DataRateValue(DataRate("32768b/s")),
                          MakeDataRateAccessor(&PointToPointNetDevice::m_bps),
                          MakeDataRateChecker())
            .AddAttribute("ReceiveErrorModel",
                          "The receiver error model used to simulate packet loss",
                          PointerValue(),
                          MakePointerAccessor(&PointToPointNetDevice::m_receiveErrorModel),
                          MakePointerChecker<ErrorModel>())
            .AddAttribute("InterframeGap",
                          "The time to wait between packet (frame) transmissions",
                          TimeValue(Seconds(0.0)),
                          MakeTimeAccessor(&PointToPointNetDevice::m_tInterframeGap),
                          MakeTimeChecker())

            //
            // Transmit queueing discipline for the device which includes its own set
            // of trace hooks.
            //
            .AddAttribute("TxQueue",
                          "A queue to use as the transmit queue in the device.",
                          PointerValue(),
                          MakePointerAccessor(&PointToPointNetDevice::m_queue),
                          MakePointerChecker<Queue<Packet>>())
            // ****** Mahdi Change ***** (START) ***** //
            .AddAttribute("ProbeTrsh",
                            "The threshold for probing the channel",
                            UintegerValue(100),
                            MakeUintegerAccessor(&PointToPointNetDevice::m_probeThreshold),
                            MakeUintegerChecker<uint32_t>())
            .AddAttribute("InterframeGapMean",
                            "The mean interframe gap to use in faulty device",
                            TimeValue(Seconds(0.0)),
                            MakeTimeAccessor(&PointToPointNetDevice::m_tInterframeGapMean),
                            MakeTimeChecker())
            // ****** Mahdi Change ***** (END) ***** //

            //
            // Trace sources at the "top" of the net device, where packets transition
            // to/from higher layers.
            //
            .AddTraceSource("MacTx",
                            "Trace source indicating a packet has arrived "
                            "for transmission by this device",
                            MakeTraceSourceAccessor(&PointToPointNetDevice::m_macTxTrace),
                            "ns3::Packet::TracedCallback")
            .AddTraceSource("MacTxDrop",
                            "Trace source indicating a packet has been dropped "
                            "by the device before transmission",
                            MakeTraceSourceAccessor(&PointToPointNetDevice::m_macTxDropTrace),
                            "ns3::Packet::TracedCallback")
            .AddTraceSource("MacPromiscRx",
                            "A packet has been received by this device, "
                            "has been passed up from the physical layer "
                            "and is being forwarded up the local protocol stack.  "
                            "This is a promiscuous trace,",
                            MakeTraceSourceAccessor(&PointToPointNetDevice::m_macPromiscRxTrace),
                            "ns3::Packet::TracedCallback")
            .AddTraceSource("MacRx",
                            "A packet has been received by this device, "
                            "has been passed up from the physical layer "
                            "and is being forwarded up the local protocol stack.  "
                            "This is a non-promiscuous trace,",
                            MakeTraceSourceAccessor(&PointToPointNetDevice::m_macRxTrace),
                            "ns3::Packet::TracedCallback")
#if 0
    // Not currently implemented for this device
    .AddTraceSource ("MacRxDrop",
                     "Trace source indicating a packet was dropped "
                     "before being forwarded up the stack",
                     MakeTraceSourceAccessor (&PointToPointNetDevice::m_macRxDropTrace),
                     "ns3::Packet::TracedCallback")
#endif
            //
            // Trace sources at the "bottom" of the net device, where packets transition
            // to/from the channel.
            //
            .AddTraceSource("PhyTxBegin",
                            "Trace source indicating a packet has begun "
                            "transmitting over the channel",
                            MakeTraceSourceAccessor(&PointToPointNetDevice::m_phyTxBeginTrace),
                            "ns3::Packet::TracedCallback")
            .AddTraceSource("PhyTxEnd",
                            "Trace source indicating a packet has been "
                            "completely transmitted over the channel",
                            MakeTraceSourceAccessor(&PointToPointNetDevice::m_phyTxEndTrace),
                            "ns3::Packet::TracedCallback")
            .AddTraceSource("PhyTxDrop",
                            "Trace source indicating a packet has been "
                            "dropped by the device during transmission",
                            MakeTraceSourceAccessor(&PointToPointNetDevice::m_phyTxDropTrace),
                            "ns3::Packet::TracedCallback")
#if 0
    // Not currently implemented for this device
    .AddTraceSource ("PhyRxBegin",
                     "Trace source indicating a packet has begun "
                     "being received by the device",
                     MakeTraceSourceAccessor (&PointToPointNetDevice::m_phyRxBeginTrace),
                     "ns3::Packet::TracedCallback")
#endif
            .AddTraceSource("PhyRxEnd",
                            "Trace source indicating a packet has been "
                            "completely received by the device",
                            MakeTraceSourceAccessor(&PointToPointNetDevice::m_phyRxEndTrace),
                            "ns3::Packet::TracedCallback")
            .AddTraceSource("PhyRxDrop",
                            "Trace source indicating a packet has been "
                            "dropped by the device during reception",
                            MakeTraceSourceAccessor(&PointToPointNetDevice::m_phyRxDropTrace),
                            "ns3::Packet::TracedCallback")

            //
            // Trace sources designed to simulate a packet sniffer facility (tcpdump).
            // Note that there is really no difference between promiscuous and
            // non-promiscuous traces in a point-to-point link.
            //
            .AddTraceSource("Sniffer",
                            "Trace source simulating a non-promiscuous packet sniffer "
                            "attached to the device",
                            MakeTraceSourceAccessor(&PointToPointNetDevice::m_snifferTrace),
                            "ns3::Packet::TracedCallback")
            .AddTraceSource("PromiscSniffer",
                            "Trace source simulating a promiscuous packet sniffer "
                            "attached to the device",
                            MakeTraceSourceAccessor(&PointToPointNetDevice::m_promiscSnifferTrace),
                            "ns3::Packet::TracedCallback")
            // ****** Mahdi Change ***** (START) ***** //
            .AddTraceSource("StartTxOut",
                            "Trace source indicating the start of packet transmission",
                            MakeTraceSourceAccessor(&PointToPointNetDevice::m_startTxOutTrace),
                            "ns3::Packet::TracedCallback");
            // ****** Mahdi Change ***** (END) ***** //
    return tid;
}

PointToPointNetDevice::PointToPointNetDevice()
    : m_txMachineState(READY),
      m_channel(nullptr),
      m_linkUp(false),
      m_currentPkt(nullptr)
{
    NS_LOG_FUNCTION(this);
}

PointToPointNetDevice::~PointToPointNetDevice()
{
    NS_LOG_FUNCTION(this);
}

void
PointToPointNetDevice::AddHeader(Ptr<Packet> p, uint16_t protocolNumber)
{
    NS_LOG_FUNCTION(this << p << protocolNumber);
    PppHeader ppp;
    ppp.SetProtocol(EtherToPpp(protocolNumber));
    p->AddHeader(ppp);
}

bool
PointToPointNetDevice::ProcessHeader(Ptr<Packet> p, uint16_t& param)
{
    NS_LOG_FUNCTION(this << p << param);
    PppHeader ppp;
    p->RemoveHeader(ppp);
    param = PppToEther(ppp.GetProtocol());
    return true;
}

void
PointToPointNetDevice::DoDispose()
{
    NS_LOG_FUNCTION(this);
    m_node = nullptr;
    m_channel = nullptr;
    m_receiveErrorModel = nullptr;
    m_currentPkt = nullptr;
    m_queue = nullptr;
    NetDevice::DoDispose();
}

// ****** Mahdi Change ***** (START) ***** // 
uint32_t
PointToPointNetDevice::GetNBytesTotal()
{
    NS_LOG_FUNCTION(this);
    uint32_t remainedBytes = 0;
    if(m_currentPkt)
    {
        remainedBytes = m_currentPkt->GetSize() - ((Simulator::Now() - m_lastTxStart).GetSeconds() * m_bps.GetBitRate() / 8);
    }
    // std::cout << " ### PointToPointNetDevice ### Remained Bytes: " << remainedBytes << " BytesInQueue: " << m_queue->GetNBytes() << " at time: " << Simulator::Now().GetNanoSeconds() << std::endl;
    return remainedBytes + m_queue->GetNBytes();
}

DataRate 
PointToPointNetDevice::GetDataRate()
{
    
    return m_bps;
}

bool
PointToPointNetDevice::IsIdle()
{
    NS_LOG_FUNCTION(this);
    return (m_txMachineState == READY && m_queue->IsEmpty());
}

// void
// PointToPointNetDevice::TagCurrPacket()
// {   
//     NS_LOG_FUNCTION(this);
//     NS_ASSERT_MSG(m_currentPkt, "PointToPointNetDevice::TagCurrPacket(): m_currentPkt zero");
//     NS_ASSERT_MSG(m_lastTxStart != Seconds(0), "PointToPointNetDevice::TagCurrPacket(): m_lastTxStart zero");
//     MeasurementProbeTagWithBits tag;
//     uint32_t transmittedBits = m_bps.GetBitRate() * (Simulator::Now() - m_lastTxStart).GetSeconds();
    
//     Ptr<Packet> pktCopy = m_currentPkt->Copy();
//     PppHeader ppp;
//     pktCopy->RemoveHeader(ppp);
//     Ipv4Header ipv4;
//     pktCopy->RemoveHeader(ipv4);
//     // std::cout << " ### PointToPointNetDevice ### Tagging current packet: " << ipv4.GetIdentification() << std::endl;

//     // check if the current packet already has a tag
//     if (m_currentPkt->PeekPacketTag(tag))
//     {
//         MeasurementProbeTagWithBits newTag = tag; // Copy existing tag
//         newTag.SetBitFlag(transmittedBits);
//         m_currentPkt->RemovePacketTag(tag); // Remove old tag
//         // Add the new tag with updated bits
//         m_currentPkt->AddPacketTag(newTag);
//     }
//     else
//     {
//         tag.SetFlag(true);
//         tag.SetBitFlag(transmittedBits);
//         m_currentPkt->AddPacketTag(tag);
//     }
// }

// void
// PointToPointNetDevice::TagClosestPacket()
// {
//     NS_LOG_FUNCTION(this);
//     NS_ASSERT_MSG(m_currentPkt, "PointToPointNetDevice::TagCurrPacket(): m_currentPkt zero");
//     NS_ASSERT_MSG(m_lastTxStart != Seconds(0), "PointToPointNetDevice::TagCurrPacket(): m_lastTxStart zero");

//     MeasurementProbeTagWithBits tag;
//     Time fromTheLast = Simulator::Now() - m_lastTxEnd;
//     Time toTheNext = m_bps.CalculateBytesTxTime(m_currentPkt->GetSize()) - (Simulator::Now() - m_lastTxStart);
//     // std::cout << "fromTheLast: " << fromTheLast.GetNanoSeconds() << ", toTheNext: " << toTheNext.GetNanoSeconds() << std::endl;
//     if (fromTheLast > toTheNext)
//     {
//         // std::cout << "Tagging current packet: ";
//         // m_currentPkt->Print(std::cout);
//         // std::cout << std::endl;
//         if (m_currentPkt->PeekPacketTag(tag))
//         {
//             MeasurementProbeTagWithBits newTag = tag; // Copy existing tag
//             newTag.SetBitFlag(1);
//             m_currentPkt->RemovePacketTag(tag); // Remove old tag
//             // Add the new tag with updated bits
//             m_currentPkt->AddPacketTag(newTag);
//         }
//         else
//         {
//             tag.SetFlag(true);
//             tag.SetBitFlag(1);
//             m_currentPkt->AddPacketTag(tag);
//         }
//     }
//     else
//     {
//         // std::cout << "Tagging last packet: ";
//         // m_lastPkt->Print(std::cout);
//         // std::cout << std::endl;
//         if (m_lastPkt->PeekPacketTag(tag))
//         {
//             MeasurementProbeTagWithBits newTag = tag; // Copy existing tag
//             newTag.SetBitFlag(1);
//             m_lastPkt->RemovePacketTag(tag); // Remove old tag
//             // Add the new tag with updated bits
//             m_lastPkt->AddPacketTag(newTag);
//         }
//         else
//         {
//             tag.SetFlag(true);
//             tag.SetBitFlag(1);
//             m_lastPkt->AddPacketTag(tag);
//         }
//         // overriwting the tag
//         m_phyTxEndTrace(m_lastPkt);
//     }
// }
// void
// PointToPointNetDevice::ManageNextSend(uint32_t mss)
// {
//     if (!m_isHalted) 
//     {
//         m_remainedHaltTime = m_bps.CalculateBytesTxTime(mss) + m_channel->GetDelay();
//         m_haltStartTime = Simulator::Now();
//         m_isHalted = true;
//         m_tagNext = true;
//         // std::cout << " ### PointToPointNetDevice ### Device is halted for " << m_remainedHaltTime.GetNanoSeconds() << " at time: " << Simulator::Now().GetNanoSeconds() << std::endl;
//         Simulator::Schedule(m_bps.CalculateBytesTxTime(mss / 2), &PointToPointNetDevice::ResumeTransmission, this);
//     }
// }

// void
// PointToPointNetDevice::ResumeTransmission()
// {
//     NS_LOG_FUNCTION(this);
//     Ptr<const Packet> p = m_queue->Peek();
//     if (p)
//     {
//         m_remainedHaltTime -= (Simulator::Now() - m_haltStartTime);
//         m_remainedHaltTime -= (m_bps.CalculateBytesTxTime(p->GetSize()) + m_channel->GetDelay());
//         if (m_remainedHaltTime <= Seconds(0))
//         {
//             m_isHalted = false;
//             m_haltStartTime = Seconds(0);
//             m_remainedHaltTime = Seconds(0);
//             // std::cout << " ### PointToPointNetDevice ### Device is resumed at time: " << Simulator::Now().GetNanoSeconds() << std::endl;
//             Ptr<Packet> pkt = m_queue->Dequeue();
//             if (m_tagNext)
//             {
//                 MeasurementProbeTagWithBits tag;
//                 tag.SetFlag(true);
//                 tag.SetBitFlag(0);
//                 pkt->AddPacketTag(tag);
//                 m_tagNext = false;
//             }
//             m_snifferTrace(pkt);
//             m_promiscSnifferTrace(pkt);
//             // ****** Mahdi Change ***** (START) ***** //
//             m_startTxOutTrace(pkt);
//             // ****** Mahdi Change ***** (END) ***** //
//             TransmitStart(pkt);
//         }
//         else
//         {
//             // std::cout << " ### PointToPointNetDevice ### Device is still halted, remaining time: " << m_remainedHaltTime.GetNanoSeconds() << " at time: " << Simulator::Now().GetNanoSeconds() << std::endl;
//             Simulator::Schedule(m_remainedHaltTime, &PointToPointNetDevice::ResumeTransmission, this);
//         }
//     }
//     else
//     {
//         m_isHalted = false;
//         m_haltStartTime = Seconds(0);
//         m_remainedHaltTime = Seconds(0);
//         // std::cout << " ### PointToPointNetDevice ### No packet to resume transmission at time: " << Simulator::Now().GetNanoSeconds() << std::endl;
//     }
// }

// void
// PointToPointNetDevice::TagNextPacket()
// {
//     NS_LOG_FUNCTION(this);
//     m_tagNext = true;
// }

// bool
// PointToPointNetDevice::IsProbeNeeded()
// {
//     NS_LOG_FUNCTION(this);
//     // Check if the current packet is not null
//     if (m_currentPkt)
//     {
//         uint32_t remainedBytes = m_currentPkt->GetSize() - (m_bps.GetBitRate() * (Simulator::Now() - m_lastTxStart).GetSeconds() / 8);
//         // std::cout << "Remained bytes in current packet: " << remainedBytes << std::endl;
//         if (remainedBytes > m_probeThreshold)
//         {
//             // std::cout << "Probe needed: " << remainedBytes << " bytes remaining." << std::endl;
//             return true;
//         }
//         // std::cout << "No probe needed, current packet is sufficient." << std::endl;
//         TagCurrPacket(); // Tag the current packet if it is not needed for probing
//     }
//     // std::cout << "no packet is sent, probe needed." << std::endl;
//     return true;
// }

void
PointToPointNetDevice::FragmentPacket(Ptr<Packet> p, uint32_t firstFragmentSize)
{
    NS_LOG_FUNCTION(this);
    // // std::cout << "Fragmenting packet: " << std::endl;
    // // p->Print(std::cout);
    // // std::cout << std::endl;
    // std::list<Ipv4PayloadHeaderPair> listFragments;

    // // std::cout << "before PPP remove: "; 
    // // p->Print(std::cout);


    // PppHeader pppHeader;
    // p->RemoveHeader(pppHeader);

    // // std::cout << "\nAfter PPP remove: ";
    // // p->Print(std::cout);
    // // std::cout << std::endl;

    // Ipv4Header ipv4Header;
    // p->RemoveHeader(ipv4Header);

    // // std::cout << "After IPv4 remove: ";
    // // p->Print(std::cout);
    // // std::cout << std::endl;

    // DoFragmentation(p, ipv4Header, firstFragmentSize, listFragments);

    // // std::cout << "After fragmentation: " << std::endl;
    // for (auto it = listFragments.begin(); it != listFragments.end(); it++)
    // {
    //     // std::cout << "before AddHeader: ";
    //     // it->first->Print(std::cout);
    //     // std::cout << std::endl;
    //     it->first->AddHeader(it->second);
    //     // std::cout << "after AddHeader: ";
    //     // it->first->Print(std::cout);
    //     // std::cout << std::endl;
    //     it->first->AddHeader(pppHeader);
    //     // std::cout << "after AddHeader: ";
    //     // it->first->Print(std::cout);
    //     // std::cout << std::endl;
    //     PrioPackets.push_back(it->first);
    // }
    // // std::cout << "Fragmented packet into " << listFragments.size() << " fragments." << std::endl;

}

// void
// PointToPointNetDevice::SetNextPoissonTick(Time nextTick)
// {
//     NS_LOG_FUNCTION(this << nextTick.As(Time::S));
//     m_nextPoissonTick = nextTick + Simulator::Now();
// }

Ptr<Packet>
PointToPointNetDevice::CheckForFragmentation(Ptr<Packet> p)
{
    NS_LOG_FUNCTION(this << p);
    if (m_nextPoissonTick == Seconds(0) || m_probeThreshold == 0)
    {
        return p;
    }

    Time txTime = m_bps.CalculateBytesTxTime(p->GetSize());
    // std::cout << "Checking for fragmentation, txTime: " << txTime.GetNanoSeconds() << " nextPoissonTick: " << m_nextPoissonTick.GetNanoSeconds() << " now: " << Simulator::Now().GetNanoSeconds() << std::endl;
    if ((Simulator::Now() < m_nextPoissonTick) && txTime > m_nextPoissonTick - Simulator::Now())
    {
        uint32_t firstFragmentSize = ((m_nextPoissonTick - Simulator::Now()).GetSeconds() * m_bps.GetBitRate() / 8) + m_probeThreshold;
        uint32_t remained = ((txTime - (m_nextPoissonTick - Simulator::Now())).GetSeconds() * m_bps.GetBitRate()) / 8;
        if (remained > m_probeThreshold)
        {   
            // std::cout << "Fragmentation needed, remained size: " << remained << std::endl;
            FragmentPacket(p, firstFragmentSize);
            p = PrioPackets.front();
            PrioPackets.erase(PrioPackets.begin());
            return p;
        }
        // else
        // {
        //     std::cout << "No fragmentation needed, remained size: " << remained << std::endl;
        // }
    }
    // else
    // {
    //     std::cout << "No fragmentation needed. tx before poisson" << std::endl;
    // }
    return p;
}

// ****** Mahdi Change ***** (END) ***** //

void
PointToPointNetDevice::SetDataRate(DataRate bps)
{
    NS_LOG_FUNCTION(this);
    //mahdi
    std::cout << " GOT HERE " << bps.GetBitRate() << std::endl;
    //mahdi
    m_bps = bps;
}

void
PointToPointNetDevice::SetInterframeGap(Time t)
{
    NS_LOG_FUNCTION(this << t.As(Time::S));
    m_tInterframeGap = t;
}

// ****** Mahdi Change ***** (START) ***** //
void
PointToPointNetDevice::SetInterframeGapMean(Time t)
{
    NS_LOG_FUNCTION(this << t.As(Time::S));

    m_tInterframeGapMean = t;
    m_varInterframeGap->SetAttribute("Mean", DoubleValue(t.GetSeconds()));
}
// ****** Mahdi Change ***** (END) ***** //

bool
PointToPointNetDevice::TransmitStart(Ptr<Packet> p)
{
    NS_LOG_FUNCTION(this << p);
    NS_LOG_LOGIC("UID is " << p->GetUid() << ")"); 

    //
    // This function is called to start the process of transmitting a packet.
    // We need to tell the channel that we've started wiggling the wire and
    // schedule an event that will be executed when the transmission is complete.
    //
    NS_ASSERT_MSG(m_txMachineState == READY, "Must be READY to transmit");
    m_txMachineState = BUSY;
    m_currentPkt = p;
    m_phyTxBeginTrace(m_currentPkt);

    Time txTime = m_bps.CalculateBytesTxTime(p->GetSize());
    Time txCompleteTime = txTime + m_tInterframeGap;

    // ****** Mahdi Change ***** (START) ***** //
    if (m_tInterframeGapMean > Seconds(0))
    {
        Time randomIFG = Seconds(m_varInterframeGap->GetValue());
        // std::cout << " ### PointToPointNetDevice ### Using random interframe gap of " << randomIFG.GetNanoSeconds() << " at time: " << Simulator::Now().GetNanoSeconds();
        // Ptr<Ipv4> ipv4 = m_node->GetObject<Ipv4>();
        // int32_t ifIndex = ipv4->GetInterfaceForDevice(this);
        // std::cout << " on Device with IP: " << ipv4->GetAddress(ifIndex, 0).GetLocal() << std::endl;
        txCompleteTime = txTime + randomIFG;
    }
    // if (m_node->GetObject<ns3::Ipv4>()->GetAddress(1, 0).GetLocal() == Ipv4Address("10.1.1.1"))
    // {
    //     Ipv4Header ipv4Header;
    //     PppHeader pppHeader;
    //     Ptr<Packet> pktCopy = p->Copy();
    //     pktCopy->RemoveHeader(pppHeader);
    //     pktCopy->RemoveHeader(ipv4Header);
    //     std::cout << "Start transmitting packet with ID: " << ipv4Header.GetIdentification() << " and " <<  pktCopy->GetUid() << " at time: " << Simulator::Now().GetNanoSeconds() << " with size: " << m_currentPkt->GetSize() << std::endl;
    // }
    m_lastTxStart = Simulator::Now();
    m_currTxEnd = txCompleteTime + Simulator::Now();
    // ****** Mahdi Change ***** (END) ***** //

    NS_LOG_LOGIC("Schedule TransmitCompleteEvent in " << txCompleteTime.As(Time::S));
    Simulator::Schedule(txCompleteTime, &PointToPointNetDevice::TransmitComplete, this);

    bool result = m_channel->TransmitStart(p, this, txTime);
    if (!result)
    {
        m_phyTxDropTrace(p);
    }
    return result;
}

void
PointToPointNetDevice::TransmitComplete()
{
    NS_LOG_FUNCTION(this);

    //
    // This function is called to when we're all done transmitting a packet.
    // We try and pull another packet off of the transmit queue.  If the queue
    // is empty, we are done, otherwise we need to start transmitting the
    // next packet.
    //
    NS_ASSERT_MSG(m_txMachineState == BUSY, "Must be BUSY if transmitting");
    m_txMachineState = READY;

    NS_ASSERT_MSG(m_currentPkt, "PointToPointNetDevice::TransmitComplete(): m_currentPkt zero");

    // ****** Mahdi Change ***** (START) ***** //
    m_lastPkt = m_currentPkt;
    m_lastTxEnd = Simulator::Now();
    // if (m_node->GetObject<ns3::Ipv4>()->GetAddress(1, 0).GetLocal() == Ipv4Address("10.1.1.1"))
    // {
    //     Ipv4Header ipv4Header;
    //     PppHeader pppHeader;
    //     Ptr<Packet> pktCopy = m_currentPkt->Copy();
    //     pktCopy->RemoveHeader(pppHeader);
    //     pktCopy->RemoveHeader(ipv4Header);
    //     std::cout << "Finished transmitting packet: " << ipv4Header.GetIdentification() << " and "  << pktCopy->GetUid() << " at time: " << m_lastTxEnd.GetNanoSeconds() << " with size: " << m_currentPkt->GetSize() << std::endl;
    // }
    // ****** Mahdi Change ***** (END) ***** //

    m_phyTxEndTrace(m_currentPkt);
    m_currentPkt = nullptr;
    
    // ****** Mahdi Change ***** (START) ***** // 
    m_lastTxStart = Seconds(0); // Reset the last transmission start time
    Ptr<Packet> p;
    if (PrioPackets.size() > 0) // check if there are prioritized packets
    {
        p = PrioPackets.front();
        PrioPackets.erase(PrioPackets.begin());
        // std::cout << "Transmitting prioritized packet: " << p->GetUid() << " Remained " << PrioPackets.size() << " packets" << std::endl;
    }
    else
    {
        p = m_queue->Dequeue();
    }
    if (!m_isHalted) // start transmitting the next packet, if there is any, if we are not halted
    {
        if (!p)
        {
            NS_LOG_LOGIC("No pending packets in device queue after tx complete");
            return;
        }

        //
        // Got another packet off of the queue, so start the transmit process again.
        //
        p = CheckForFragmentation(p);
        m_snifferTrace(p);
        m_promiscSnifferTrace(p);
        // ****** Mahdi Change ***** (START) ***** //
        m_startTxOutTrace(p);
        // ****** Mahdi Change ***** (END) ***** //
        TransmitStart(p);
    }
    // ****** Mahdi Change ***** (END) ***** //
}

bool
PointToPointNetDevice::Attach(Ptr<PointToPointChannel> ch)
{
    NS_LOG_FUNCTION(this << &ch);

    m_channel = ch;

    m_channel->Attach(this);

    //
    // This device is up whenever it is attached to a channel.  A better plan
    // would be to have the link come up when both devices are attached, but this
    // is not done for now.
    //
    NotifyLinkUp();
    return true;
}

void
PointToPointNetDevice::SetQueue(Ptr<Queue<Packet>> q)
{
    NS_LOG_FUNCTION(this << q);
    m_queue = q;
}

void
PointToPointNetDevice::SetReceiveErrorModel(Ptr<ErrorModel> em)
{
    NS_LOG_FUNCTION(this << em);
    m_receiveErrorModel = em;
}

void
PointToPointNetDevice::Receive(Ptr<Packet> packet)
{
    NS_LOG_FUNCTION(this << packet);
    uint16_t protocol = 0;

    if (m_receiveErrorModel && m_receiveErrorModel->IsCorrupt(packet))
    {
        //
        // If we have an error model and it indicates that it is time to lose a
        // corrupted packet, don't forward this packet up, let it go.
        //
        m_phyRxDropTrace(packet);
    }
    else
    {
        //
        // Hit the trace hooks.  All of these hooks are in the same place in this
        // device because it is so simple, but this is not usually the case in
        // more complicated devices.
        //
        m_snifferTrace(packet);
        m_promiscSnifferTrace(packet);
        m_phyRxEndTrace(packet);

        //
        // Trace sinks will expect complete packets, not packets without some of the
        // headers.
        //
        Ptr<Packet> originalPacket = packet->Copy();

        //
        // Strip off the point-to-point protocol header and forward this packet
        // up the protocol stack.  Since this is a simple point-to-point link,
        // there is no difference in what the promisc callback sees and what the
        // normal receive callback sees.
        //
        ProcessHeader(packet, protocol);

        if (!m_promiscCallback.IsNull())
        {
            m_macPromiscRxTrace(originalPacket);
            m_promiscCallback(this,
                              packet,
                              protocol,
                              GetRemote(),
                              GetAddress(),
                              NetDevice::PACKET_HOST);
        }

        m_macRxTrace(originalPacket);
        m_rxCallback(this, packet, protocol, GetRemote());
    }
}

Ptr<Queue<Packet>>
PointToPointNetDevice::GetQueue() const
{
    NS_LOG_FUNCTION(this);
    return m_queue;
}

void
PointToPointNetDevice::NotifyLinkUp()
{
    NS_LOG_FUNCTION(this);
    m_linkUp = true;
    m_linkChangeCallbacks();
}

void
PointToPointNetDevice::SetIfIndex(const uint32_t index)
{
    NS_LOG_FUNCTION(this);
    m_ifIndex = index;
}

uint32_t
PointToPointNetDevice::GetIfIndex() const
{
    return m_ifIndex;
}

Ptr<Channel>
PointToPointNetDevice::GetChannel() const
{
    return m_channel;
}

//
// This is a point-to-point device, so we really don't need any kind of address
// information.  However, the base class NetDevice wants us to define the
// methods to get and set the address.  Rather than be rude and assert, we let
// clients get and set the address, but simply ignore them.

void
PointToPointNetDevice::SetAddress(Address address)
{
    NS_LOG_FUNCTION(this << address);
    m_address = Mac48Address::ConvertFrom(address);
}

Address
PointToPointNetDevice::GetAddress() const
{
    return m_address;
}

bool
PointToPointNetDevice::IsLinkUp() const
{
    NS_LOG_FUNCTION(this);
    return m_linkUp;
}

void
PointToPointNetDevice::AddLinkChangeCallback(Callback<void> callback)
{
    NS_LOG_FUNCTION(this);
    m_linkChangeCallbacks.ConnectWithoutContext(callback);
}

//
// This is a point-to-point device, so every transmission is a broadcast to
// all of the devices on the network.
//
bool
PointToPointNetDevice::IsBroadcast() const
{
    NS_LOG_FUNCTION(this);
    return true;
}

//
// We don't really need any addressing information since this is a
// point-to-point device.  The base class NetDevice wants us to return a
// broadcast address, so we make up something reasonable.
//
Address
PointToPointNetDevice::GetBroadcast() const
{
    NS_LOG_FUNCTION(this);
    return Mac48Address("ff:ff:ff:ff:ff:ff");
}

bool
PointToPointNetDevice::IsMulticast() const
{
    NS_LOG_FUNCTION(this);
    return true;
}

Address
PointToPointNetDevice::GetMulticast(Ipv4Address multicastGroup) const
{
    NS_LOG_FUNCTION(this);
    return Mac48Address("01:00:5e:00:00:00");
}

Address
PointToPointNetDevice::GetMulticast(Ipv6Address addr) const
{
    NS_LOG_FUNCTION(this << addr);
    return Mac48Address("33:33:00:00:00:00");
}

bool
PointToPointNetDevice::IsPointToPoint() const
{
    NS_LOG_FUNCTION(this);
    return true;
}

bool
PointToPointNetDevice::IsBridge() const
{
    NS_LOG_FUNCTION(this);
    return false;
}

bool
PointToPointNetDevice::Send(Ptr<Packet> packet, const Address& dest, uint16_t protocolNumber)
{
    NS_LOG_FUNCTION(this << packet << dest << protocolNumber);
    NS_LOG_LOGIC("p=" << packet << ", dest=" << &dest);
    NS_LOG_LOGIC("UID is " << packet->GetUid());

    //
    // If IsLinkUp() is false it means there is no channel to send any packet
    // over so we just hit the drop trace on the packet and return an error.
    //
    if (!IsLinkUp())
    {
        m_macTxDropTrace(packet);
        return false;
    }

    //
    // Stick a point to point protocol header on the packet in preparation for
    // shoving it out the door.
    //
    AddHeader(packet, protocolNumber);

    m_macTxTrace(packet);

    //
    // We should enqueue and dequeue the packet to hit the tracing hooks.
    //
    
    if (m_queue->Enqueue(packet))
    {
        //
        // If the channel is ready for transition we send the packet right now
        //
        if (m_txMachineState == READY && (!m_isHalted) && (PrioPackets.size() == 0))   // mahdi
        {
            packet = m_queue->Dequeue();
            packet = CheckForFragmentation(packet);
            m_snifferTrace(packet);
            m_promiscSnifferTrace(packet);
            // ****** Mahdi Change ***** (START) ***** //
            m_startTxOutTrace(packet);
            // ****** Mahdi Change ***** (END) ***** //
            bool ret = TransmitStart(packet);
            return ret;
        }
        return true;
    }

    // Enqueue may fail (overflow)

    m_macTxDropTrace(packet);
    return false;
}

bool
PointToPointNetDevice::SendFrom(Ptr<Packet> packet,
                                const Address& source,
                                const Address& dest,
                                uint16_t protocolNumber)
{
    NS_LOG_FUNCTION(this << packet << source << dest << protocolNumber);
    return false;
}

Ptr<Node>
PointToPointNetDevice::GetNode() const
{
    return m_node;
}

void
PointToPointNetDevice::SetNode(Ptr<Node> node)
{
    NS_LOG_FUNCTION(this);
    m_node = node;
}

bool
PointToPointNetDevice::NeedsArp() const
{
    NS_LOG_FUNCTION(this);
    return false;
}

void
PointToPointNetDevice::SetReceiveCallback(NetDevice::ReceiveCallback cb)
{
    m_rxCallback = cb;
}

void
PointToPointNetDevice::SetPromiscReceiveCallback(NetDevice::PromiscReceiveCallback cb)
{
    m_promiscCallback = cb;
}

bool
PointToPointNetDevice::SupportsSendFrom() const
{
    NS_LOG_FUNCTION(this);
    return false;
}

void
PointToPointNetDevice::DoMpiReceive(Ptr<Packet> p)
{
    NS_LOG_FUNCTION(this << p);
    Receive(p);
}

Address
PointToPointNetDevice::GetRemote() const
{
    NS_LOG_FUNCTION(this);
    NS_ASSERT(m_channel->GetNDevices() == 2);
    for (std::size_t i = 0; i < m_channel->GetNDevices(); ++i)
    {
        Ptr<NetDevice> tmp = m_channel->GetDevice(i);
        if (tmp != this)
        {
            return tmp->GetAddress();
        }
    }
    NS_ASSERT(false);
    // quiet compiler.
    return Address();
}

bool
PointToPointNetDevice::SetMtu(uint16_t mtu)
{
    NS_LOG_FUNCTION(this << mtu);
    m_mtu = mtu;
    return true;
}

uint16_t
PointToPointNetDevice::GetMtu() const
{
    NS_LOG_FUNCTION(this);
    return m_mtu;
}

uint16_t
PointToPointNetDevice::PppToEther(uint16_t proto)
{
    NS_LOG_FUNCTION_NOARGS();
    switch (proto)
    {
    case 0x0021:
        return 0x0800; // IPv4
    case 0x0057:
        return 0x86DD; // IPv6
    default:
        NS_ASSERT_MSG(false, "PPP Protocol number not defined!");
    }
    return 0;
}

uint16_t
PointToPointNetDevice::EtherToPpp(uint16_t proto)
{
    NS_LOG_FUNCTION_NOARGS();
    switch (proto)
    {
    case 0x0800:
        return 0x0021; // IPv4
    case 0x86DD:
        return 0x0057; // IPv6
    default:
        NS_ASSERT_MSG(false, "PPP Protocol number not defined!");
    }
    return 0;
}

} // namespace ns3
