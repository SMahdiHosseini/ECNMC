//
// Created by Zeinab Shmeis on 28.05.20.
//

#include "E2EMonitor.h"

E2EMonitorEvent::E2EMonitorEvent(PacketKey *key) : _key(key) {}

void E2EMonitorEvent::SetSent() { _sentTime = ns3::Simulator::Now(); }
void E2EMonitorEvent::SetReceived() { _receivedTime = ns3::Simulator::Now(); }
void E2EMonitorEvent::SetReceived(Time t) { _receivedTime = t; }
void E2EMonitorEvent::SetEcn(bool ecn) { _key->SetEcn(ecn); }
void E2EMonitorEvent::SetPath(int path) { _key->SetPath(path);  }
PacketKey *E2EMonitorEvent::GetPacketKey() const { return _key; }
Time E2EMonitorEvent::GetSentTime() const { return _sentTime; }
Time E2EMonitorEvent::GetReceivedTime() const {  return _receivedTime; }
bool E2EMonitorEvent::GetEcn() const { return _key->GetEcn(); }
int E2EMonitorEvent::GetPath() const { return _key->GetPath(); }
bool E2EMonitorEvent::IsReceived() const { return _receivedTime != Time(-1); }

ostream &operator<<(ostream &os, const E2EMonitorEvent &event) {
    os << "E2EMonitorEvent: [ ";
    os << "Key = " << *(event._key) << ", SentTime = " << event._sentTime << ", ReceiveTime = " << event._receivedTime;
    os << "]";
    return os;
}


E2EMonitor::E2EMonitor(const Time &startTime, const Time &duration, const Ptr<PointToPointNetDevice> netDevice, const Ptr<Node> &rxNode, const string &monitorTag, double errorRate) {
    _startTime = startTime;
    _duration = duration;
    _monitorTag = monitorTag;
    _errorRate = errorRate;

    Simulator::Schedule(_startTime, &E2EMonitor::Connect, this, netDevice, rxNode->GetId());
    Simulator::Schedule(_startTime + _duration, &E2EMonitor::Disconnect, this, netDevice, rxNode->GetId());
}

void E2EMonitor::Connect(const Ptr<PointToPointNetDevice> netDevice, uint32_t rxNodeId) {
    netDevice->GetQueue()->TraceConnectWithoutContext("Enqueue", MakeCallback(&E2EMonitor::Enqueue, this));
    Config::ConnectWithoutContext("/NodeList/" + to_string(rxNodeId) + "/$ns3::Ipv4L3Protocol/Rx", MakeCallback(
            &E2EMonitor::RecordIpv4PacketReceived, this));
}

void E2EMonitor::Disconnect(const Ptr<PointToPointNetDevice> netDevice, uint32_t rxNodeId) {
    netDevice->GetQueue()->TraceDisconnectWithoutContext("Enqueue", MakeCallback(&E2EMonitor::Enqueue, this));
}

void E2EMonitor::AddAppKey(AppKey appKey) {
    _appsKey.insert(appKey);
}

void E2EMonitor::Enqueue(Ptr<const Packet> packet) {
    PacketKey* packetKey = PacketKey::Packet2PacketKey(packet, FIRST_HEADER_PPP);
    if(_appsKey.count(AppKey::PacketKey2AppKey(*packetKey))) {
        packetKey->SetPacketSize(packet->GetSize());
        auto *packetEvent = new E2EMonitorEvent(packetKey);
        packetEvent->SetSent();
        _recordedPackets[*packetKey] = packetEvent;
    }
}

void E2EMonitor::RecordIpv4PacketReceived(Ptr<const Packet> packet, Ptr<Ipv4> ipv4, uint32_t interface) {
    PacketKey* packetKey = PacketKey::Packet2PacketKey(packet, FIRST_HEADER_IPV4);
    if(_appsKey.count(AppKey::PacketKey2AppKey(*packetKey))) {
        auto packetKeyEventPair = _recordedPackets.find(*packetKey);
        if (packetKeyEventPair != _recordedPackets.end()) {
            Ipv4Header header;
            packet->PeekHeader(header);
            if (header.EcnTypeToString(header.GetEcn()) == "CE") {
                packetKeyEventPair->second->SetEcn(true);
            }
            uint64_t hash = DynamicCast<Ipv4L3Protocol>(ipv4)->GetHashValue_out(packetKey->GetSrcIp(), packetKey->GetDstIp(), packetKey->GetSrcPort(), packetKey->GetDstPort(), header.GetProtocol());
            packetKeyEventPair->second->SetPath(hash % 2);
            
            // if (_monitorTag == "R0H0R2H0" || _monitorTag == "R0H1R2H1") {
            //     Ptr<UniformRandomVariable> m_rand = CreateObject<UniformRandomVariable>();
            //     uint32_t t = m_rand->GetInteger(1, 100);
            //     if (t < _errorRate * 100){
            //         Time time = packetKeyEventPair->second->GetSentTime() + Time((ns3::Simulator::Now() - packetKeyEventPair->second->GetSentTime()).GetNanoSeconds() * 1.25);
            //         packetKeyEventPair->second->SetReceived(time);
            //         // std::cout << ns3::Simulator::Now().GetNanoSeconds() << "," << packetKey->GetSrcIp() << "," << packetKey->GetSrcPort() << "," << packetKey->GetDstIp() << "," << 
            //         // packetKey->GetDstPort() << "," << packetKey->GetSeqNb() << "," << packetKey->GetId() << std::endl;
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

void E2EMonitor::SavePacketRecords(const string& filename) {
    ofstream outfile;
    outfile.open(filename);
    outfile << "SourceIp,SourcePort,DestinationIp,DestinationPort,SequenceNb,Id,PayloadSize,PacketSize,Path,SentTime,IsReceived,ReceiveTime,ECN" << endl;
    for (auto& packetKeyEventPair: _recordedPackets) {
        PacketKey key = packetKeyEventPair.first;
        E2EMonitorEvent* event = packetKeyEventPair.second;

        outfile << key.GetSrcIp() << "," << key.GetSrcPort() << ",";
        outfile << key.GetDstIp() << "," << key.GetDstPort() << "," << key.GetSeqNb() << "," << key.GetId()  << "," << key.GetSize() << "," << key.GetPacketSize() << ",";
        outfile << event->GetPath() << ",";
        outfile << GetRelativeTime(event->GetSentTime()).GetNanoSeconds() << ",";
        outfile << event->IsReceived() << "," << GetRelativeTime(event->GetReceivedTime()).GetNanoSeconds() << "," << event->GetEcn() << endl;
    }
    outfile.close();
}

string E2EMonitor::GetMonitorTag() const { return _monitorTag; }
ns3::Time E2EMonitor::GetRelativeTime(const Time &time){ return time - _startTime; }

