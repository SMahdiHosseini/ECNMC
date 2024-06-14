//
// Created by Zeinab Shmeis on 13.06.2024.
//

#include "NetDeviceMonitor.h"

NetDeviceMonitorEvent::NetDeviceMonitorEvent(PacketKey *key) : _key(key) {}

void NetDeviceMonitorEvent::SetSent() { _sentTime = ns3::Simulator::Now(); }
void NetDeviceMonitorEvent::SetReceived() { _receivedTime = ns3::Simulator::Now(); }

PacketKey *NetDeviceMonitorEvent::GetPacketKey() const { return _key; }
Time NetDeviceMonitorEvent::GetSentTime() const { return _sentTime; }
Time NetDeviceMonitorEvent::GetReceivedTime() const {  return _receivedTime; }
bool NetDeviceMonitorEvent::IsSent() const { return _sentTime != Time(-1); }

ostream &operator<<(ostream &os, const NetDeviceMonitorEvent &event) {
    os << "NetDeviceMonitorEvent: [ ";
    os << "Key = " << *(event._key) << ", SentTime = " << event._sentTime << ", ReceiveTime = " << event._receivedTime;
    os << "]";
    return os;
}


NetDeviceMonitor::NetDeviceMonitor(const Time &startTime, const Time &duration, Ptr<PointToPointNetDevice> netDevice, const string &monitorTag) {
    _startTime = startTime;
    _duration = duration;
    _monitorTag = monitorTag;

    Simulator::Schedule(_startTime, &NetDeviceMonitor::Connect, this, netDevice);
    Simulator::Schedule(_startTime + _duration, &NetDeviceMonitor::Disconnect, this, netDevice);
}

void NetDeviceMonitor::Connect(Ptr<PointToPointNetDevice> netDevice) {
    netDevice->GetQueue()->TraceConnectWithoutContext("Enqueue", MakeCallback(&NetDeviceMonitor::Enqueue, this));
    netDevice->GetQueue()->TraceConnectWithoutContext("Dequeue", MakeCallback(&NetDeviceMonitor::Dequeue, this));
}

void NetDeviceMonitor::Disconnect(Ptr<PointToPointNetDevice> netDevice) {
    netDevice->GetQueue()->TraceDisconnectWithoutContext("Enqueue", MakeCallback(&NetDeviceMonitor::Enqueue, this));
    netDevice->GetQueue()->TraceDisconnectWithoutContext("Dequeue", MakeCallback(&NetDeviceMonitor::Dequeue, this));
}

void NetDeviceMonitor::AddAppKey(AppKey appKey) {
    _appsKey.insert(appKey);
}

void NetDeviceMonitor::Enqueue(Ptr<const Packet> packet) {
    PacketKey* packetKey = PacketKey::Packet2PacketKey(packet, FIRST_HEADER_PPP);
    if(_appsKey.count(AppKey::PacketKey2AppKey(*packetKey))) {
        auto *packetEvent = new NetDeviceMonitorEvent(packetKey);
        packetEvent->SetReceived();
        _recordedPackets[*packetKey] = packetEvent;
    }
}

void NetDeviceMonitor::Dequeue(Ptr<const Packet> packet) {
    PacketKey* packetKey = PacketKey::Packet2PacketKey(packet, FIRST_HEADER_PPP);
    if(_appsKey.count(AppKey::PacketKey2AppKey(*packetKey))) {
        auto packetKeyEventPair = _recordedPackets.find(*packetKey);
        if (packetKeyEventPair != _recordedPackets.end()) {
            packetKeyEventPair->second->SetSent();
        }
    }
}

void NetDeviceMonitor::SavePacketRecords(const string &filename) {
    ofstream outfile;
    outfile.open(filename);
    outfile << "SourceIp,SourcePort,DestinationIp,DestinationPort,SequenceNb,Id,PayloadSize,ReceiveTime,IsSent,SentTime" << endl;
    for (auto& packetKeyEventPair: _recordedPackets) {
        PacketKey key = packetKeyEventPair.first;
        NetDeviceMonitorEvent* event = packetKeyEventPair.second;

        outfile << key.GetSrcIp() << "," << key.GetSrcPort() << ",";
        outfile << key.GetDstIp() << "," << key.GetDstPort() << "," << key.GetSeqNb() << "," << key.GetId()  << "," << key.GetSize() << ",";
        outfile << GetRelativeTime(event->GetReceivedTime()).GetNanoSeconds() << ",";
        outfile << event->IsSent() << "," << GetRelativeTime(event->GetSentTime()).GetNanoSeconds() << endl;
    }
    outfile.close();
}
string NetDeviceMonitor::GetMonitorTag() const { return _monitorTag; }
ns3::Time NetDeviceMonitor::GetRelativeTime(const Time &time){ return time - _startTime; }

