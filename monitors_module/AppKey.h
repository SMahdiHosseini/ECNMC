//
// Created by nal on 22.03.21.
//

#ifndef WEHE_P_TOMOGRAPHY_APPKEY_H
#define WEHE_P_TOMOGRAPHY_APPKEY_H

#include "ns3/core-module.h"
#include "ns3/internet-module.h"

#include "PacketKey.h"

using namespace ns3;
using namespace std;

class AppKey {

private:
    Ipv4Address _srcIp, _dstIp;
    uint16_t _srcPort, _dstPort;

public:
    AppKey(const Ipv4Address &srcIp, const Ipv4Address &dstIp, uint16_t srcPort, uint16_t dstPort);
    AppKey(const Ipv4Address &srcIp, const Ipv4Address &dstIp);

    [[nodiscard]] Ipv4Address GetSrcIp() const;
    [[nodiscard]] Ipv4Address GetDstIp() const;
    [[nodiscard]] uint16_t GetSrcPort() const;
    [[nodiscard]] uint16_t GetDstPort() const;

    bool operator<(const AppKey &rhs) const;
    bool operator>(const AppKey &rhs) const;
    bool operator<=(const AppKey &rhs) const;
    bool operator>=(const AppKey &rhs) const;
    bool operator==(const AppKey &rhs) const;
    bool operator!=(const AppKey &rhs) const;

    friend ostream &operator<<(ostream &os, const AppKey &event);

    bool belongsToApp(const PacketKey &packetKey);

    static AppKey PacketKey2AppKey(const PacketKey &packetKey);

};

struct AppKeyHash {
    std::size_t operator()(AppKey const& appKey) const noexcept;
};


#endif //WEHE_P_TOMOGRAPHY_APPKEY_H
