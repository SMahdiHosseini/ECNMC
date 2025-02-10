//
// Created by Mahdi Hosseini on 19.03.24.
//

#include "SwitchMonitor.h"

SwitchMonitorEvent::SwitchMonitorEvent(PacketKey *key) : _key(key) {}

void SwitchMonitorEvent::SetSent() { _sentTime = ns3::Simulator::Now(); }
void SwitchMonitorEvent::SetReceived() { _receivedTime = ns3::Simulator::Now(); }

PacketKey *SwitchMonitorEvent::GetPacketKey() const { return _key; }
Time SwitchMonitorEvent::GetSentTime() const { return _sentTime; }
Time SwitchMonitorEvent::GetReceivedTime() const {  return _receivedTime; }
bool SwitchMonitorEvent::IsSent() const { return _sentTime != Time(-1); }

ostream &operator<<(ostream &os, const SwitchMonitorEvent &event) {
    os << "SwitchMonitorEvent: [ ";
    os << "Key = " << *(event._key) << ", SentTime = " << event._sentTime << ", ReceiveTime = " << event._receivedTime;
    os << "]";
    return os;
}

SwitchMonitor::SwitchMonitor(const Time &startTime, const Time &duration, const Time &steadyStartTime, const Time  &steadyStopTime, const Ptr<Node> &node, const string &monitorTag) {
    _startTime = startTime;
    _duration = duration;
    _monitorTag = monitorTag;
    _steadyStartTime = steadyStartTime;
    _steadyStopTime = steadyStopTime;
    hasher = Hasher();
    Simulator::Schedule(_startTime, &SwitchMonitor::Connect, this, node);
    Simulator::Schedule(_startTime + _duration, &SwitchMonitor::Disconnect, this, node);
}

void SwitchMonitor::Connect(const Ptr<Node> &node) {
    // iterate over all net devices of the node
    for (uint32_t i = 0; i < node->GetNDevices(); i++) {
        Ptr<NetDevice> netDevice = node->GetDevice(i);
        if (netDevice->GetInstanceTypeId().GetName() == "ns3::PointToPointNetDevice") {
            Ptr<PointToPointNetDevice> p2pNetDevice = DynamicCast<PointToPointNetDevice>(netDevice);
            p2pNetDevice->TraceConnectWithoutContext("PromiscSniffer", MakeCallback(&SwitchMonitor::RecordPacket, this));
            // // check if the queue is a RED queue
            // if (p2pNetDevice->GetNode()->GetObject<TrafficControlLayer>()->GetRootQueueDiscOnDevice(p2pNetDevice)->GetInstanceTypeId() == ns3::RedQueueDisc::GetTypeId()) {
            //     std::cout << "RED queue detected" << std::endl;
            // }
            // else {
            //     std::cout << "RED queue not detected" << std::endl;
            // }
        }
    }
}

void SwitchMonitor::Disconnect(const Ptr<Node> &node) {
    for (uint32_t i = 0; i < node->GetNDevices(); i++) {
        Ptr<NetDevice> netDevice = node->GetDevice(i);
        if (netDevice->GetInstanceTypeId().GetName() == "ns3::PointToPointNetDevice") {
            Ptr<PointToPointNetDevice> p2pNetDevice = DynamicCast<PointToPointNetDevice>(netDevice);
            p2pNetDevice->TraceDisconnectWithoutContext("PromiscSniffer", MakeCallback(&SwitchMonitor::RecordPacket, this));
        }
    }
}

void SwitchMonitor::RecordPacket(Ptr<const Packet> packet) {
    if (Simulator::Now() < _steadyStartTime || Simulator::Now() > _steadyStopTime) {
        return;
    }
    PacketKey* packetKey = PacketKey::Packet2PacketKey(packet, FIRST_HEADER_PPP);
    if(_appsKey.count(AppKey::PacketKey2AppKey(*packetKey))) {
        auto packetKeyEventPair = _recordedPackets.find(*packetKey);
        if (packetKeyEventPair != _recordedPackets.end()) {
            packetKeyEventPair->second->SetSent();
        } 
        else {
            auto *packetEvent = new SwitchMonitorEvent(packetKey);
            _recordedPackets[*packetKey] = packetEvent;
            packetEvent->SetReceived();
        }
    }
}

void SwitchMonitor::AddAppKey(AppKey appKey) {
    _appsKey.insert(appKey);
}

uint64_t SwitchMonitor::GetHashValue(const Ipv4Address src, const Ipv4Address dst, const uint16_t srcPort, const uint16_t dstPort, const uint8_t protocol) {
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

void SwitchMonitor::SavePacketRecords(const string& filename) {
    ofstream outfile;
    outfile.open(filename);
    outfile << "SourceIp,SourcePort,DestinationIp,DestinationPort,SequenceNb,Id,PayloadSize,ReceiveTime,IsSent,SentTime,path" << endl;
    for (auto& packetKeyEventPair: _recordedPackets) {
        PacketKey key = packetKeyEventPair.first;
        SwitchMonitorEvent* event = packetKeyEventPair.second;
        uint64_t hash = GetHashValue(key.GetSrcIp(), key.GetDstIp(), key.GetSrcPort(), key.GetDstPort(), 6);
        int path = hash % 2;
        outfile << key.GetSrcIp() << "," << key.GetSrcPort() << ",";
        outfile << key.GetDstIp() << "," << key.GetDstPort() << "," << key.GetSeqNb() << "," << key.GetId()  << "," << key.GetSize() << ",";
        outfile << GetRelativeTime(event->GetReceivedTime()).GetNanoSeconds() << ",";
        outfile << event->IsSent() << "," << GetRelativeTime(event->GetSentTime()).GetNanoSeconds() << "," << path << endl;
    }
    outfile.close();
}

string SwitchMonitor::GetMonitorTag() const { return _monitorTag; }
ns3::Time SwitchMonitor::GetRelativeTime(const Time &time){ return time - _startTime; }

