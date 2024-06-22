//
// Created by Zeinab Shmeis on 29.05.20.
//

#ifndef NEUTRALITY_PACKETKEY_H
#define NEUTRALITY_PACKETKEY_H

#include "ns3/core-module.h"
#include "ns3/internet-module.h"

#define FIRST_HEADER_PPP 0
#define FIRST_HEADER_IPV4 1
#define FIRST_HEADER_TCP 2
#define FIRST_HEADER_UDP 3

using namespace ns3;
using namespace std;

class PacketKey {

private:
    Ipv4Address _srcIp, _dstIp;
    uint16_t _id;
    uint16_t _srcPort, _dstPort;
    SequenceNumber32 _seqNb, _ackNb;
    uint32_t _size;
    size_t _payloadHash;
    int records;
    int path;
    bool ecn;

public:
    PacketKey(const Ipv4Address &srcIp, const Ipv4Address &dstIp, uint16_t id, uint16_t srcPort, uint16_t dstPort,
              const SequenceNumber32& seqNb, const SequenceNumber32& ackNb, uint32_t size, size_t payloadHash);

    [[nodiscard]] Ipv4Address GetSrcIp() const;
    [[nodiscard]] Ipv4Address GetDstIp() const;
    [[nodiscard]] uint16_t GetId() const;
    [[nodiscard]] uint16_t GetSrcPort() const;
    [[nodiscard]] uint16_t GetDstPort() const;
    [[nodiscard]] const SequenceNumber32 &GetSeqNb() const;
    [[nodiscard]] const SequenceNumber32 &GetAckNb() const;
    [[nodiscard]] uint32_t GetSize() const;
    [[nodiscard]] size_t GetPayloadHash() const;
    [[nodiscard]] int GetRecords() const;
    [[nodiscard]] int GetPath() const;
    [[nodiscard]] bool GetEcn() const;

    bool operator<(const PacketKey &rhs) const;
    bool operator>(const PacketKey &rhs) const;
    bool operator<=(const PacketKey &rhs) const;
    bool operator>=(const PacketKey &rhs) const;
    bool operator==(const PacketKey &rhs) const;
    bool operator!=(const PacketKey &rhs) const;

    friend ostream &operator<<(ostream &os, const PacketKey &event);
    static PacketKey* Packet2PacketKey(Ptr<const Packet> packet, uint8_t firstHeaderType);

    void SetPayloadHash(size_t payloadHash);
    void SetSize(uint32_t size);
    void SetAckNb(const SequenceNumber32 &ackNb);
    void SetSeqNb(const SequenceNumber32 &seqNb);
    void SetDstPort(uint16_t dstPort);
    void SetSrcPort(uint16_t srcPort);
    void SetSrcIp(const Ipv4Address &srcIp);
    void SetDstIp(const Ipv4Address &dstIp);
    void SetId(uint16_t id);
    void SetRecords(int records);
    void SetPath(int path);
    void SetEcn(bool ecn);
};

struct PacketKeyHash {
    std::size_t operator()(PacketKey const& packetKey) const noexcept;
};


#endif //NEUTRALITY_PACKETKEY_H
