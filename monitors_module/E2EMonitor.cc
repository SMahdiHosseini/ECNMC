//
// Created by Zeinab Shmeis on 28.05.20.
//

#include "E2EMonitor.h"

E2EMonitorEvent::E2EMonitorEvent(PacketKey *key) { SetPacketKey(key); }

void E2EMonitorEvent::SetEcn(bool ecn) { _key->SetEcn(ecn); }
void E2EMonitorEvent::SetPath(int path) { _key->SetPath(path);  }
bool E2EMonitorEvent::GetEcn() const { return _key->GetEcn(); } 
int E2EMonitorEvent::GetPath() const { return _key->GetPath(); }


E2EMonitor::E2EMonitor(const Time &startTime, const Time &duration, const Time &steadyStartTime, const Time &steadyStopTime, const Ptr<PointToPointNetDevice> netDevice, const Ptr<Node> &rxNode, const string &monitorTag, double errorRate) 
: Monitor(startTime, duration, steadyStartTime, steadyStopTime, monitorTag) {
    _errorRate = errorRate;
    sampleMean = Seconds(0);
    unbiasedSmapleVariance = Seconds(0);
    sumOfPacketSizes = 0;
    receivedPackets = 0;
    sentPackets = 0;
    markedPackets = 0;

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
        sentPackets++;
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
                markedPackets++;
            }
            uint64_t hash = DynamicCast<Ipv4L3Protocol>(ipv4)->GetHashValue_out(packetKey->GetSrcIp(), packetKey->GetDstIp(), packetKey->GetSrcPort(), packetKey->GetDstPort(), header.GetProtocol());
            packetKeyEventPair->second->SetPath(hash % 2);
            packetKeyEventPair->second->SetReceived();
            sumOfPacketSizes += packetKeyEventPair->first.GetPacketSize();
            receivedPackets++;
            Time delta = (packetKeyEventPair->second->GetReceivedTime() - packetKeyEventPair->second->GetSentTime() - sampleMean);
            sampleMean = sampleMean + Time(delta.GetNanoSeconds() / receivedPackets);
            if (receivedPackets <= 1) {
                unbiasedSmapleVariance = Time(0);
            }
            else {
                unbiasedSmapleVariance = unbiasedSmapleVariance + Time((delta.GetNanoSeconds() * delta.GetNanoSeconds()) / receivedPackets) - Time(unbiasedSmapleVariance.GetNanoSeconds() / (receivedPackets - 1));
            }
            // remove the packet from the map
            _recordedPackets.erase(packetKeyEventPair);
        }
    }
}

void E2EMonitor::SaveMonitorRecords(const string& filename) {
    ofstream outfile;
    outfile.open(filename);
    outfile << "sampleDelayMean,unbiasedSmapleDelayVariance,averagePacketSize,receivedPackets,sentPackets,markedPackets" << endl;
    outfile << sampleMean.GetNanoSeconds() << "," << unbiasedSmapleVariance.GetNanoSeconds() << "," << sumOfPacketSizes / receivedPackets << "," << receivedPackets << "," << sentPackets << "," << markedPackets << endl;
    // outfile << "SourceIp,SourcePort,DestinationIp,DestinationPort,SequenceNb,Id,PayloadSize,PacketSize,Path,SentTime,IsReceived,ReceiveTime,ECN" << endl;
    // for (auto& packetKeyEventPair: _recordedPackets) {
    //     PacketKey key = packetKeyEventPair.first;
    //     E2EMonitorEvent* event = packetKeyEventPair.second;

    //     outfile << key.GetSrcIp() << "," << key.GetSrcPort() << ",";
    //     outfile << key.GetDstIp() << "," << key.GetDstPort() << "," << key.GetSeqNb() << "," << key.GetId()  << "," << key.GetSize() << "," << key.GetPacketSize() << ",";
    //     outfile << event->GetPath() << ",";
    //     outfile << GetRelativeTime(event->GetSentTime()).GetNanoSeconds() << ",";
    //     outfile << event->IsReceived() << "," << GetRelativeTime(event->GetReceivedTime()).GetNanoSeconds() << "," << event->GetEcn() << endl;
    // }
    outfile.close();
}
