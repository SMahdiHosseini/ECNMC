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

SwitchMonitor::SwitchMonitor(const Time &startTime, const Time &duration, const Ptr<Node> &node, const string &monitorTag){
    _startTime = startTime;
    _duration = duration;
    _monitorTag = monitorTag;

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
    PacketKey* packetKey = PacketKey::Packet2PacketKey(packet, false);
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

void SwitchMonitor::SavePacketRecords(const string& filename) {
    ofstream outfile;
    outfile.open(filename);
    outfile << "SourceIp,SourcePort,DestinationIp,DestinationPort,SequenceNb,PayloadSize,ReceiveTime,IsSent,SentTime" << endl;
    for (auto& packetKeyEventPair: _recordedPackets) {
        PacketKey key = packetKeyEventPair.first;
        SwitchMonitorEvent* event = packetKeyEventPair.second;

        outfile << key.GetSrcIp() << "," << key.GetSrcPort() << ",";
        outfile << key.GetDstIp() << "," << key.GetDstPort() << "," << key.GetSeqNb() << "," << key.GetSize() << ",";
        outfile << GetRelativeTime(event->GetReceivedTime()).GetNanoSeconds() << ",";
        outfile << event->IsSent() << "," << GetRelativeTime(event->GetSentTime()).GetNanoSeconds() << endl;
    }
    outfile.close();
}

string SwitchMonitor::GetMonitorTag() const { return _monitorTag; }
ns3::Time SwitchMonitor::GetRelativeTime(const Time &time){ return time - _startTime; }

