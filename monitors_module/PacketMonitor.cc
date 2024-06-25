//
// Created by Zeinab Shmeis on 28.05.20.
//

#include "PacketMonitor.h"

PacketMonitorEvent::PacketMonitorEvent(PacketKey *key) : _key(key) {}

void PacketMonitorEvent::SetSent() { _sentTime = ns3::Simulator::Now(); }
void PacketMonitorEvent::SetReceived() { _receivedTime = ns3::Simulator::Now(); }
void PacketMonitorEvent::SetReceived(Time t) { _receivedTime = t; }
void PacketMonitorEvent::SetEcn(bool ecn) { _key->SetEcn(ecn); }
PacketKey *PacketMonitorEvent::GetPacketKey() const { return _key; }
Time PacketMonitorEvent::GetSentTime() const { return _sentTime; }
Time PacketMonitorEvent::GetReceivedTime() const {  return _receivedTime; }
bool PacketMonitorEvent::GetEcn() const { return _key->GetEcn(); }
bool PacketMonitorEvent::IsReceived() const { return _receivedTime != Time(-1); }

ostream &operator<<(ostream &os, const PacketMonitorEvent &event) {
    os << "PacketMonitorEvent: [ ";
    os << "Key = " << *(event._key) << ", SentTime = " << event._sentTime << ", ReceiveTime = " << event._receivedTime;
    os << "]";
    return os;
}


PacketMonitor::PacketMonitor(const Time &startTime, const Time &duration, const Ptr<Node> &txNode, const Ptr<Node> &rxNode, const string &monitorTag, double errorRate) {
    _startTime = startTime;
    _duration = duration;
    _monitorTag = monitorTag;
    _errorRate = errorRate;

    Simulator::Schedule(_startTime, &PacketMonitor::Connect, this, txNode->GetId(), rxNode->GetId());
    Simulator::Schedule(_startTime + _duration, &PacketMonitor::Disconnect, this, txNode->GetId(), rxNode->GetId());
}

void PacketMonitor::Connect(uint32_t txNodeId, uint32_t rxNodeId) {
    Config::ConnectWithoutContext("/NodeList/" + to_string(txNodeId) + "/$ns3::Ipv4L3Protocol/Tx", MakeCallback(
            &PacketMonitor::RecordIpv4PacketSent, this));
    Config::ConnectWithoutContext("/NodeList/" + to_string(rxNodeId) + "/$ns3::Ipv4L3Protocol/Rx", MakeCallback(
            &PacketMonitor::RecordIpv4PacketReceived, this));
}

void PacketMonitor::Disconnect(uint32_t txNodeId, uint32_t rxNodeId) {
    Config::DisconnectWithoutContext("/NodeList/" + to_string(txNodeId) + "/$ns3::Ipv4L3Protocol/Tx", MakeCallback(
            &PacketMonitor::RecordIpv4PacketSent, this));

}

void PacketMonitor::AddAppKey(AppKey appKey) {
    _appsKey.insert(appKey);
}

void PacketMonitor::RecordIpv4PacketSent(Ptr<const Packet> packet, Ptr<Ipv4> ipv4, uint32_t interface) {
    PacketKey* packetKey = PacketKey::Packet2PacketKey(packet, FIRST_HEADER_IPV4);
    if(_appsKey.count(AppKey::PacketKey2AppKey(*packetKey))) {
        Ipv4Header header;
        packet->PeekHeader(header);
        uint64_t hash = DynamicCast<Ipv4L3Protocol>(ipv4)->GetHashValue_out(packetKey->GetSrcIp(), packetKey->GetDstIp(), packetKey->GetSrcPort(), packetKey->GetDstPort(), header.GetProtocol());
        packetKey->SetPath(hash % 2);
        auto *packetEvent = new PacketMonitorEvent(packetKey);
        packetEvent->SetSent();
        _recordedPackets[*packetKey] = packetEvent;
    }
}

void PacketMonitor::RecordIpv4PacketReceived(Ptr<const Packet> packet, Ptr<Ipv4> ipv4, uint32_t interface) {
    PacketKey* packetKey = PacketKey::Packet2PacketKey(packet, FIRST_HEADER_IPV4);
    if(_appsKey.count(AppKey::PacketKey2AppKey(*packetKey))) {
        auto packetKeyEventPair = _recordedPackets.find(*packetKey);
        if (packetKeyEventPair != _recordedPackets.end()) {
            Ipv4Header header;
            packet->PeekHeader(header);
            if (header.EcnTypeToString(header.GetEcn()) == "CE") {
                packetKeyEventPair->second->SetEcn(true);
            }
            // if (_monitorTag == "R0H0R2H0" || _monitorTag == "R0H1R2H1") {
            //     Ptr<UniformRandomVariable> m_rand = CreateObject<UniformRandomVariable>();
            //     uint32_t t = m_rand->GetInteger(0, 1);
            //     if (t < _errorRate){
            //         // Time time = packetKeyEventPair->second->GetSentTime() + Time((ns3::Simulator::Now() - packetKeyEventPair->second->GetSentTime()).GetNanoSeconds() * 1.25);
            //         // packetKeyEventPair->second->SetReceived(time);
            //         packetKeyEventPair->second->SetReceived();
            //     }
            //     else{
            //         packetKeyEventPair->second->SetReceived();
            //     }
            // }
            // else{
                packetKeyEventPair->second->SetReceived();
            // }
        }
    }
}

void PacketMonitor::SavePacketRecords(const string& filename) {
    ofstream outfile;
    outfile.open(filename);
    outfile << "SourceIp,SourcePort,DestinationIp,DestinationPort,SequenceNb,Id,PayloadSize,Path,SentTime,IsReceived,ReceiveTime,ECN" << endl;
    for (auto& packetKeyEventPair: _recordedPackets) {
        PacketKey key = packetKeyEventPair.first;
        PacketMonitorEvent* event = packetKeyEventPair.second;

        outfile << key.GetSrcIp() << "," << key.GetSrcPort() << ",";
        outfile << key.GetDstIp() << "," << key.GetDstPort() << "," << key.GetSeqNb() << "," << key.GetId()  << "," << key.GetSize() << ",";
        outfile << key.GetPath() << ",";
        outfile << GetRelativeTime(event->GetSentTime()).GetNanoSeconds() << ",";
        outfile << event->IsReceived() << "," << GetRelativeTime(event->GetReceivedTime()).GetNanoSeconds() << "," << event->GetEcn() << endl;
    }
    outfile.close();
}

string PacketMonitor::GetMonitorTag() const { return _monitorTag; }
ns3::Time PacketMonitor::GetRelativeTime(const Time &time){ return time - _startTime; }

