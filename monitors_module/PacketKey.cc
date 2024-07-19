//
// Created by Zeinab Shmeis on 29.05.20.
//

#include "PacketKey.h"

#include "ns3/applications-module.h"
#include "ns3/ppp-header.h"

#include <boost/functional/hash.hpp>


PacketKey::PacketKey(const Ipv4Address &srcIp, const Ipv4Address &dstIp, uint16_t id,
                     uint16_t srcPort, uint16_t dstPort, const SequenceNumber32& seqNb, const SequenceNumber32& ackNb,
                     uint32_t size, size_t payloadHash) : _srcIp(srcIp), _dstIp(dstIp), _id(id),
                        _srcPort(srcPort), _dstPort(dstPort), _seqNb(seqNb), _ackNb(ackNb),
                        _size(size), _payloadHash(payloadHash), _packetSize(0), records(0), path(0), ecn(false) {}

bool PacketKey::operator<(const PacketKey &rhs) const {
    if (_srcIp < rhs._srcIp) return true;
    if (rhs._srcIp < _srcIp) return false;
    if (_dstIp < rhs._dstIp) return true;
    if (rhs._dstIp < _dstIp) return false;
    if (_id < rhs._id) return true;
    if (rhs._id < _id) return false;
    if (_srcPort < rhs._srcPort) return true;
    if (rhs._srcPort < _srcPort) return false;
    if (_dstPort < rhs._dstPort) return true;
    if (rhs._dstPort < _dstPort) return false;
    if (_seqNb < rhs._seqNb) return true;
    if (rhs._seqNb < _seqNb) return false;
    if (_ackNb < rhs._ackNb) return true;
    if (rhs._ackNb < _ackNb) return false;
    return _payloadHash < rhs._payloadHash;
}
bool PacketKey::operator>(const PacketKey &rhs) const { return rhs < *this; }
bool PacketKey::operator<=(const PacketKey &rhs) const { return !(rhs < *this); }
bool PacketKey::operator>=(const PacketKey &rhs) const { return !(*this < rhs); }

bool PacketKey::operator==(const PacketKey &rhs) const {
    return _srcIp == rhs._srcIp &&
           _dstIp == rhs._dstIp &&
            _id == rhs._id &&
           _srcPort == rhs._srcPort &&
           _dstPort == rhs._dstPort &&
           _seqNb == rhs._seqNb &&
           _ackNb == rhs._ackNb &&
           _size == rhs._size &&
           _payloadHash == rhs._payloadHash;
}
bool PacketKey::operator!=(const PacketKey &rhs) const { return !(rhs == *this); }

ostream &operator<<(ostream &os, const PacketKey &key) {
    os << "Packet: [ ";
    os << "SrcIP = " << key._srcIp << ", DstIP = " << key._dstIp << ", Identification = " << key._id << ", ";
    os << "SrcPort = " << key._srcPort << ", DstPort = " << key._dstPort << ", Seq = " << key._seqNb << ", Ack = " << key._ackNb << ", ";
    os << "Size = " << key._size << ", PayloadHash = " << key._payloadHash << ", ";
    os << "]";
    return os;
}

Ipv4Address PacketKey::GetSrcIp() const { return _srcIp; }
Ipv4Address PacketKey::GetDstIp() const { return _dstIp; }
uint16_t PacketKey::GetId() const { return _id; }
uint16_t PacketKey::GetSrcPort() const { return _srcPort; }
uint16_t PacketKey::GetDstPort() const { return _dstPort; }
const SequenceNumber32 &PacketKey::GetSeqNb() const { return _seqNb; }
const SequenceNumber32 &PacketKey::GetAckNb() const { return _ackNb; }
uint32_t PacketKey::GetSize() const { return _size; }
size_t PacketKey::GetPayloadHash() const { return _payloadHash; }
uint32_t PacketKey::GetPacketSize() const { return _packetSize; }
int PacketKey::GetRecords() const { return records; }
int PacketKey::GetPath() const { return path; }
bool PacketKey::GetEcn() const { return ecn; }

PacketKey* PacketKey::Packet2PacketKey(Ptr<const Packet> packet, uint8_t firstHeaderType) {
    const Ptr<Packet> &pktCopy = packet->Copy();

    // extract ppp information
    PppHeader pppHeader;
    if(firstHeaderType == FIRST_HEADER_PPP) {
        pktCopy->RemoveHeader(pppHeader);
    }

    // extract ipv4 information
    Ipv4Header ipHeader;
    Ipv4Address srcIp;
    Ipv4Address dstIp;
    uint16_t id = 0;
    if (firstHeaderType <= FIRST_HEADER_IPV4) {
        pktCopy->RemoveHeader(ipHeader);
        srcIp = ipHeader.GetSource();
        dstIp = ipHeader.GetDestination();
        id = ipHeader.GetIdentification();
    }

    // extract transport layer info
    uint16_t srcPort = 0, dstPort = 0;
    auto seqNb = SequenceNumber32(0), ackNb = SequenceNumber32(0);
    uint32_t size = 0;
    hash<string> hasher;
    size_t payloadHash = 0;
    if(ipHeader.GetProtocol() == 6 || firstHeaderType == FIRST_HEADER_TCP){
        TcpHeader tcpHeader;
        pktCopy->RemoveHeader(tcpHeader);
        srcPort = tcpHeader.GetSourcePort();
        dstPort = tcpHeader.GetDestinationPort();
        seqNb = tcpHeader.GetSequenceNumber();
        ackNb = tcpHeader.GetAckNumber();
        size = pktCopy->GetSize();
        payloadHash = hasher(pktCopy->ToString());
        pktCopy->AddHeader(tcpHeader);
    }
    else if (ipHeader.GetProtocol() == 17 || firstHeaderType == FIRST_HEADER_UDP) {
        UdpHeader udpHeader;
        pktCopy->RemoveHeader(udpHeader);
        srcPort = udpHeader.GetSourcePort();
        dstPort = udpHeader.GetDestinationPort();
        size = pktCopy->GetSize();
        payloadHash = hasher(pktCopy->ToString());
        pktCopy->AddHeader(udpHeader);
    }

    if (firstHeaderType <= FIRST_HEADER_IPV4){
        pktCopy->AddHeader(ipHeader);
    }

    if(firstHeaderType == FIRST_HEADER_PPP) {
        pktCopy->AddHeader(pppHeader);
    }

    return new PacketKey(srcIp, dstIp, id, srcPort, dstPort, seqNb, ackNb, size, payloadHash);
}

std::size_t PacketKeyHash::operator()(PacketKey const &packetKey) const noexcept {
    std::size_t seed = 0x315E4139;
    Ipv4AddressHash ipHash;
    boost::hash_combine(seed, ipHash(packetKey.GetSrcIp()));
    boost::hash_combine(seed, ipHash(packetKey.GetDstIp()));
    boost::hash_combine(seed, static_cast<std::size_t>(packetKey.GetId()));
    boost::hash_combine(seed, static_cast<std::size_t>(packetKey.GetSrcPort()));
    boost::hash_combine(seed, static_cast<std::size_t>(packetKey.GetDstPort()));
    boost::hash_combine(seed, packetKey.GetSeqNb().GetValue());
    boost::hash_combine(seed, packetKey.GetAckNb().GetValue());
    boost::hash_combine(seed, packetKey.GetPayloadHash());
    boost::hash_combine(seed, packetKey.GetRecords());
    return seed;
}

void PacketKey::SetPayloadHash(size_t payloadHash) { _payloadHash = payloadHash; }
void PacketKey::SetPacketSize(uint32_t packetSize) { _packetSize = packetSize; }
void PacketKey::SetSize(uint32_t size) { _size = size; }
void PacketKey::SetAckNb(const SequenceNumber32 &ackNb) { _ackNb = ackNb; }
void PacketKey::SetSeqNb(const SequenceNumber32 &seqNb) { _seqNb = seqNb; }
void PacketKey::SetDstPort(uint16_t dstPort) { _dstPort = dstPort; }
void PacketKey::SetSrcPort(uint16_t srcPort) { _srcPort = srcPort; }
void PacketKey::SetSrcIp(const Ipv4Address &srcIp) { _srcIp = srcIp; }
void PacketKey::SetDstIp(const Ipv4Address &dstIp) { _dstIp = dstIp; }
void PacketKey::SetId(uint16_t id) { _id = id; }
void PacketKey::SetRecords(int records) { PacketKey::records = records; }
void PacketKey::SetPath(int path) { PacketKey::path = path; }
void PacketKey::SetEcn(bool ecn) { PacketKey::ecn = ecn; }