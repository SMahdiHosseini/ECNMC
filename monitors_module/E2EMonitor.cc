//
// Created by Zeinab Shmeis on 28.05.20.
//

#include "E2EMonitor.h"

E2EMonitorEvent::E2EMonitorEvent(PacketKey *key) { SetPacketKey(key); }

void E2EMonitorEvent::SetEcn(bool ecn) { _key->SetEcn(ecn); }
void E2EMonitorEvent::SetPath(int path) { _key->SetPath(path);  }
bool E2EMonitorEvent::GetEcn() const { return _key->GetEcn(); } 
int E2EMonitorEvent::GetPath() const { return _key->GetPath(); }


E2EMonitor::E2EMonitor(const Time &startTime, const Time &duration, const Time &steadyStartTime, const Time &steadyStopTime, const Ptr<PointToPointNetDevice> netDevice, const Ptr<Node> &rxNode, const string &monitorTag, double errorRate,
                       const DataRate &_hostToTorLinkRate, const DataRate &_torToAggLinkRate, const Time &_hostToTorLinkDelay) 
: Monitor(startTime, duration, steadyStartTime, steadyStopTime, monitorTag) {
    _errorRate = errorRate;
    hostToTorLinkRate = _hostToTorLinkRate;
    torToAggLinkRate = _torToAggLinkRate;
    hostToTorLinkDelay = _hostToTorLinkDelay;
    for (int i = 0; i < 2; i++) {
        sampleMean.push_back(Seconds(0));
        unbiasedSmapleVariance.push_back(Seconds(0));
        sampleSize.push_back(0);
        sumOfPacketSizes.push_back(0);
        sentPackets.push_back(0);
        markedPackets.push_back(0);
    }
    hasher = Hasher();
    Simulator::Schedule(_startTime, &E2EMonitor::Connect, this, netDevice, rxNode->GetId());
    Simulator::Schedule(_startTime + _duration, &E2EMonitor::Disconnect, this, netDevice, rxNode->GetId());
}

uint64_t E2EMonitor::GetHashValue(const Ipv4Address src, const Ipv4Address dst, const uint16_t srcPort, const uint16_t dstPort, const uint8_t protocol) {
    hasher.clear();
    std::ostringstream oss;
    oss << src
        << dst
        << protocol
        << dstPort
        << srcPort;
    std::string data = oss.str();
    uint32_t hash = hasher.GetHash32(data);
    oss.str("");
    return hash;
}

void E2EMonitor::Connect(const Ptr<PointToPointNetDevice> netDevice, uint32_t rxNodeId) {
    netDevice->GetQueue()->TraceConnectWithoutContext("Enqueue", MakeCallback(&E2EMonitor::Enqueue, this));
    Config::ConnectWithoutContext("/NodeList/" + to_string(rxNodeId) + "/$ns3::Ipv4L3Protocol/Rx", MakeCallback(
            &E2EMonitor::RecordIpv4PacketReceived, this));
}

void E2EMonitor::Disconnect(const Ptr<PointToPointNetDevice> netDevice, uint32_t rxNodeId) {
    netDevice->GetQueue()->TraceDisconnectWithoutContext("Enqueue", MakeCallback(&E2EMonitor::Enqueue, this));
}

void E2EMonitor::Enqueue(Ptr<const Packet> packet) {
    if (Simulator::Now() < _steadyStartTime || Simulator::Now() > _steadyStopTime) {
        return;
    }
    PacketKey* packetKey = PacketKey::Packet2PacketKey(packet, FIRST_HEADER_PPP);
    if(_appsKey.count(AppKey::PacketKey2AppKey(*packetKey))) {
        packetKey->SetPacketSize(packet->GetSize());
        auto *packetEvent = new E2EMonitorEvent(packetKey);
        packetEvent->SetSent();
        _recordedPackets[*packetKey] = packetEvent;

        const Ptr<Packet> &pktCopy = packet->Copy();
        PppHeader pppHeader;
        pktCopy->RemoveHeader(pppHeader);
        Ipv4Header IPHeader;
        pktCopy->RemoveHeader(IPHeader);
        uint64_t hash = GetHashValue(packetKey->GetSrcIp(), packetKey->GetDstIp(), packetKey->GetSrcPort(), packetKey->GetDstPort(), IPHeader.GetProtocol());
        pktCopy->AddHeader(IPHeader);
        pktCopy->AddHeader(pppHeader);

        sentPackets[hash % 2] += 1;
    }
}

void E2EMonitor::RecordIpv4PacketReceived(Ptr<const Packet> packet, Ptr<Ipv4> ipv4, uint32_t interface) {
    PacketKey* packetKey = PacketKey::Packet2PacketKey(packet, FIRST_HEADER_IPV4);
    if(_appsKey.count(AppKey::PacketKey2AppKey(*packetKey))) {
        auto packetKeyEventPair = _recordedPackets.find(*packetKey);
        if (packetKeyEventPair != _recordedPackets.end()) {
            Ipv4Header header;
            packet->PeekHeader(header);
            uint64_t hash = GetHashValue(packetKey->GetSrcIp(), packetKey->GetDstIp(), packetKey->GetSrcPort(), packetKey->GetDstPort(), header.GetProtocol());
            int path = hash % 2;
            if (header.EcnTypeToString(header.GetEcn()) == "CE") {
                packetKeyEventPair->second->SetEcn(true);
                markedPackets[path] += 1;
            }
            packetKeyEventPair->second->SetPath(path);
            packetKeyEventPair->second->SetReceived();
            sumOfPacketSizes[path] += packetKeyEventPair->first.GetPacketSize();
            Time transmissionDelay = hostToTorLinkRate.CalculateBytesTxTime(packetKeyEventPair->first.GetPacketSize()) * 2
                                    + torToAggLinkRate.CalculateBytesTxTime(packetKeyEventPair->first.GetPacketSize()) * 2
                                    + hostToTorLinkDelay * 4;
            // TODO: Implement delay de-prioritization
            updateBasicCounters(packetKeyEventPair->second->GetSentTime(), packetKeyEventPair->second->GetReceivedTime() - transmissionDelay, path);
            // remove the packet from the map to reduce the memory usage of the simulation
            _recordedPackets.erase(packetKeyEventPair);
        }
    }
}

void E2EMonitor::SaveMonitorRecords(const string& filename) {
    ofstream outfile;
    outfile.open(filename);
    outfile << "path,sampleDelayMean,unbiasedSmapleDelayVariance,averagePacketSize,receivedPackets,sentPackets,markedPackets" << endl;
    outfile << 0 << "," << sampleMean[0].GetNanoSeconds() << "," << unbiasedSmapleVariance[0].GetNanoSeconds() << "," << sumOfPacketSizes[0] / sampleSize[0] << "," << sampleSize[0] << "," << sentPackets[0] << "," << markedPackets[0] << endl;
    outfile << 1 << "," << sampleMean[1].GetNanoSeconds() << "," << unbiasedSmapleVariance[1].GetNanoSeconds() << "," << sumOfPacketSizes[1] / sampleSize[1] << "," << sampleSize[1] << "," << sentPackets[1] << "," << markedPackets[1] << endl;
    outfile.close();
}
