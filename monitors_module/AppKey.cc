//
// Created by nal on 22.03.21.
//

#include <boost/functional/hash.hpp>
#include "AppKey.h"

// Construct an AppKey. use 0 for ports when you don't want it to be used
AppKey::AppKey(const Ipv4Address &srcIp, const Ipv4Address &dstIp, uint16_t srcPort, uint16_t dstPort) : _srcIp(srcIp), _dstIp(dstIp), _srcPort(srcPort), _dstPort(dstPort){}

AppKey::AppKey(const Ipv4Address &srcIp, const Ipv4Address &dstIp) : AppKey(srcIp, dstIp, 0, 0){}


bool AppKey::operator<(const AppKey &rhs) const {
    if (_srcPort != 0 &&  rhs._srcPort != 0) {
        if (_srcPort < rhs._srcPort) return true;
        if (rhs._srcPort < _srcPort) return false;
    }

    if (_dstPort != 0 && rhs._dstPort != 0) {
        if (_dstPort < rhs._dstPort) return true;
        if (rhs._dstPort < _dstPort) return false;
    }

    if (_srcIp < rhs._srcIp) return true;
    if (rhs._srcIp < _srcIp) return false;
    return _dstIp < rhs._dstIp;
}

bool AppKey::operator>(const AppKey &rhs) const {
    return rhs < *this;
}

bool AppKey::operator<=(const AppKey &rhs) const {
    return !(rhs < *this);
}

bool AppKey::operator>=(const AppKey &rhs) const {
    return !(*this < rhs);
}

bool AppKey::operator==(const AppKey &rhs) const {
    if (_srcPort == 0 && _dstPort == 0) {
        return _srcIp == rhs.GetSrcIp() && _dstIp == rhs.GetDstIp();
    }
    if (_srcPort == 0) {
        return _srcIp == rhs.GetSrcIp() && _dstIp == rhs.GetDstIp() && _dstPort == rhs.GetDstPort();
    }
    if (_dstPort == 0) {
        return _srcIp == rhs.GetSrcIp() && _dstIp == rhs.GetDstIp() && _srcPort == rhs.GetSrcPort();
    }
    return _srcIp == rhs._srcIp && _dstIp == rhs._dstIp && _srcPort == rhs.GetSrcPort() && _dstPort == rhs._dstPort;
}

bool AppKey::operator!=(const AppKey &rhs) const {
    return !(rhs == *this);
}

ostream &operator<<(ostream &os, const AppKey &key) {
    os << "_srcIp: " << key._srcIp << " _dstIp: " << key._dstIp << " _srcPort: " << key._srcPort << " _dstPort: " << key._dstPort;
    return os;
}

Ipv4Address AppKey::GetSrcIp() const { return _srcIp; }

Ipv4Address AppKey::GetDstIp() const { return _dstIp; }

uint16_t AppKey::GetSrcPort() const { return _srcPort; }

uint16_t AppKey::GetDstPort() const { return _dstPort; }

bool AppKey::belongsToApp(const PacketKey &packetKey) {
    if (_srcPort == 0 && _dstPort == 0) {
        return _srcIp == packetKey.GetSrcIp() && _dstIp == packetKey.GetDstIp();
    }
    if (_srcPort == 0) {
        return _srcIp == packetKey.GetSrcIp() && _dstIp == packetKey.GetDstIp() && _dstPort == packetKey.GetDstPort();
    }
    if (_dstPort == 0) {
        return _srcIp == packetKey.GetSrcIp() && _dstIp == packetKey.GetDstIp() && _srcPort == packetKey.GetSrcPort();
    }
    return _srcIp == packetKey.GetSrcIp() && _dstIp == packetKey.GetDstIp() && _srcPort == packetKey.GetSrcPort() && _dstPort == packetKey.GetDstPort();
}

AppKey AppKey::PacketKey2AppKey(const PacketKey &packetKey) {
    return {packetKey.GetSrcIp(), packetKey.GetDstIp(), packetKey.GetSrcPort(), packetKey.GetDstPort()};
}


std::size_t AppKeyHash::operator()(AppKey const &AppKey) const noexcept {
    std::size_t seed = 0x315E4138;
    Ipv4AddressHash ipHash;
    boost::hash_combine(seed, ipHash(AppKey.GetSrcIp()));
    boost::hash_combine(seed, ipHash(AppKey.GetDstIp()));
    boost::hash_combine(seed, static_cast<std::size_t>(AppKey.GetSrcPort()));
    boost::hash_combine(seed, static_cast<std::size_t>(AppKey.GetDstPort()));
    return seed;
}
