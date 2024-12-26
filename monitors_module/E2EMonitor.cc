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
: E2EMonitor(startTime, duration, steadyStartTime, steadyStopTime, netDevice, rxNode, monitorTag, errorRate, _hostToTorLinkRate, _torToAggLinkRate, _hostToTorLinkDelay, 2, 3) {
    
}

E2EMonitor::E2EMonitor(const Time &startTime, const Time &duration, const Time &steadyStartTime, const Time &steadyStopTime, const Ptr<PointToPointNetDevice> netDevice, const Ptr<Node> &rxNode, const string &monitorTag, double errorRate,
                       const DataRate &_hostToTorLinkRate, const DataRate &_torToAggLinkRate, const Time &_hostToTorLinkDelay, const int _numOfPaths, const int _numOfSegmetns) 
: Monitor(startTime, duration, steadyStartTime, steadyStopTime, monitorTag) {
    numOfPaths = _numOfPaths;
    numOfSegmetns = _numOfSegmetns;
    _errorRate = errorRate;
    hostToTorLinkRate = _hostToTorLinkRate;
    torToAggLinkRate = _torToAggLinkRate;
    hostToTorLinkDelay = _hostToTorLinkDelay;
    for (int i = 0; i < numOfPaths; i++) {
        sampleMean.push_back(Seconds(0));
        unbiasedSmapleVariance.push_back(Seconds(0));
        sampleSize.push_back(0);
        sumOfPacketSizes.push_back(0);
        sentPackets.push_back(0);
        markedPackets.push_back(0);
        timeAverageIntegral.push_back(Seconds(0));
        integralStartTime.push_back(Time(-1));
        integralEndTime.push_back(Time(-1));
        sentPackets_onlink.push_back(0);
    }
    packetCDF.loadCDFData("/media/experiments/ns-allinone-3.41/ns-3.41/scratch/ECNMC/Helpers/packet_size_cdf_singleQueue.csv");
    GTDropMean = 0;
    lastItemTime = Time(0);
    firstItemTime = Time(-1);
    hasher = Hasher();
    rand = CreateObject<UniformRandomVariable>();
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
    netDevice->TraceConnectWithoutContext("PromiscSniffer", MakeCallback(&E2EMonitor::Capture, this));
    Config::ConnectWithoutContext("/NodeList/" + to_string(rxNodeId) + "/$ns3::Ipv4L3Protocol/Rx", MakeCallback(
            &E2EMonitor::RecordIpv4PacketReceived, this));
}

void E2EMonitor::Disconnect(const Ptr<PointToPointNetDevice> netDevice, uint32_t rxNodeId) {
    netDevice->GetQueue()->TraceDisconnectWithoutContext("Enqueue", MakeCallback(&E2EMonitor::Enqueue, this));
}

void E2EMonitor::Capture(Ptr< const Packet > packet) {
    if (Simulator::Now() < _steadyStartTime || Simulator::Now() > _steadyStopTime) {
        return;
    }
    PacketKey* packetKey = PacketKey::Packet2PacketKey(packet, FIRST_HEADER_PPP);
    if(_appsKey.count(AppKey::PacketKey2AppKey(*packetKey))) {
        packetKey->SetPacketSize(packet->GetSize());
        auto *packetEvent = new E2EMonitorEvent(packetKey);
        packetEvent->SetSent();
        // uncomment the following line to record the packets when they are sent over the link
        _recordedPackets[*packetKey] = packetEvent;

        const Ptr<Packet> &pktCopy = packet->Copy();
        PppHeader pppHeader;
        pktCopy->RemoveHeader(pppHeader);
        Ipv4Header IPHeader;
        pktCopy->RemoveHeader(IPHeader);
        uint64_t hash = GetHashValue(packetKey->GetSrcIp(), packetKey->GetDstIp(), packetKey->GetSrcPort(), packetKey->GetDstPort(), IPHeader.GetProtocol());
        pktCopy->AddHeader(IPHeader);
        pktCopy->AddHeader(pppHeader);
        packetKey->SetPath(hash % numOfPaths);
        sentPackets_onlink[hash % numOfPaths] += 1;
    }
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
        // uncomment the following line to record the packets when they are enqueued
        // _recordedPackets[*packetKey] = packetEvent;

        const Ptr<Packet> &pktCopy = packet->Copy();
        PppHeader pppHeader;
        pktCopy->RemoveHeader(pppHeader);
        Ipv4Header IPHeader;
        pktCopy->RemoveHeader(IPHeader);
        uint64_t hash = GetHashValue(packetKey->GetSrcIp(), packetKey->GetDstIp(), packetKey->GetSrcPort(), packetKey->GetDstPort(), IPHeader.GetProtocol());
        pktCopy->AddHeader(IPHeader);
        pktCopy->AddHeader(pppHeader);
        packetKey->SetPath(hash % numOfPaths);
        sentPackets[hash % numOfPaths] += 1;
    }
}

void E2EMonitor::updateTimeAverageIntegral(uint32_t path, Time delay, Time endTime) {
    if (integralStartTime[path] != Time(-1)) {
        timeAverageIntegral[path] += Time(delay.GetNanoSeconds() * (endTime - integralEndTime[path]).GetNanoSeconds());
    }

    if (integralStartTime[path] == Time(-1)) {
        integralStartTime[path] = endTime;
    }
    integralEndTime[path] = endTime;
}

void E2EMonitor::RecordIpv4PacketReceived(Ptr<const Packet> packet, Ptr<Ipv4> ipv4, uint32_t interface) {
    PacketKey* packetKey = PacketKey::Packet2PacketKey(packet, FIRST_HEADER_IPV4);
    if(_appsKey.count(AppKey::PacketKey2AppKey(*packetKey))) {
        auto packetKeyEventPair = _recordedPackets.find(*packetKey);
        if (packetKeyEventPair != _recordedPackets.end()) {
            Ipv4Header header;
            packet->PeekHeader(header);
            uint64_t hash = GetHashValue(packetKey->GetSrcIp(), packetKey->GetDstIp(), packetKey->GetSrcPort(), packetKey->GetDstPort(), header.GetProtocol());
            int path = hash % numOfPaths;
            if (header.EcnTypeToString(header.GetEcn()) == "CE") {
                packetKeyEventPair->second->SetEcn(true);
                markedPackets[path] += 1;
            }
            packetKeyEventPair->second->SetPath(path);
            packetKeyEventPair->second->SetReceived();
            sumOfPacketSizes[path] += packetKeyEventPair->first.GetPacketSize();
            Time transmissionDelay = Seconds(0);
            if (numOfSegmetns == 3) {
                transmissionDelay = hostToTorLinkRate.CalculateBytesTxTime(packetKeyEventPair->first.GetPacketSize()) * 2
                                    + torToAggLinkRate.CalculateBytesTxTime(packetKeyEventPair->first.GetPacketSize()) * 2
                                    + hostToTorLinkDelay * 4;
            } else if (numOfSegmetns == 1) {
                transmissionDelay = hostToTorLinkRate.CalculateBytesTxTime(packetKeyEventPair->first.GetPacketSize())
                                    + torToAggLinkRate.CalculateBytesTxTime(packetKeyEventPair->first.GetPacketSize())
                                    + hostToTorLinkDelay * 2;
            }
            // Implementation of delay de-prioritization
            Time additionalDeprioritizationDelay = Seconds(0);
            // uint32_t temp = rand->GetInteger(0, 1 / _errorRate);
            // if (temp == 0 && (_monitorTag == "R0H0R2H0" || _monitorTag == "R0H1R2H1")) {
            //     additionalDeprioritizationDelay = packetKeyEventPair->second->GetReceivedTime() - transmissionDelay - packetKeyEventPair->second->GetSentTime();
            //     additionalDeprioritizationDelay = Time(additionalDeprioritizationDelay.GetNanoSeconds() * 0.35);
            // }


            updateBasicCounters(packetKeyEventPair->second->GetSentTime(), packetKeyEventPair->second->GetReceivedTime() + additionalDeprioritizationDelay - transmissionDelay, path);
            updateTimeAverageIntegral(path, packetKeyEventPair->second->GetReceivedTime() + additionalDeprioritizationDelay - transmissionDelay - packetKeyEventPair->second->GetSentTime(), packetKeyEventPair->second->GetReceivedTime());
            // remove the packet from the map to reduce the memory usage of the simulation
            // _recordedPackets.erase(packetKeyEventPair);
            Time prev = lastItemTime;
            lastItemTime = packetKeyEventPair->second->GetSentTime();
            if (firstItemTime == Time(-1)) {
                firstItemTime =  packetKeyEventPair->second->GetSentTime();
                cout << "First Item Time: " << firstItemTime.GetNanoSeconds() << endl;
                cout << "Last Item Time: " << lastItemTime.GetNanoSeconds() << endl;
                cout << "Prev: " << prev.GetNanoSeconds() << endl;        
                return;
            }
            uint32_t queueSize = ((packetKeyEventPair->second->GetReceivedTime() - transmissionDelay - packetKeyEventPair->second->GetSentTime()) * torToAggLinkRate) / 8;
            std::cout << 
            " Queue Size Measure: " << queueSize << 
            " Equeue: " << (packetKeyEventPair->second->GetSentTime() + hostToTorLinkDelay + hostToTorLinkRate.CalculateBytesTxTime(packetKeyEventPair->first.GetPacketSize())).GetNanoSeconds() << 
            " Dequeue: " << (packetKeyEventPair->second->GetReceivedTime() - hostToTorLinkDelay - torToAggLinkRate.CalculateBytesTxTime(packetKeyEventPair->first.GetPacketSize())).GetNanoSeconds() << endl;
            packet->Print(std::cout);
            std::cout << endl;

            double dropProbDynamicCDF = 0;
            if (queueSize > 100000) {
                queueSize = queueSize - 100000;
                dropProbDynamicCDF = packetCDF.calculateProbabilityGreaterThan(QueueSize("37.5KB").GetValue() - queueSize);
            }
            GTDropMean = (GTDropMean * (prev - firstItemTime).GetNanoSeconds() + dropProbDynamicCDF * (lastItemTime - prev).GetNanoSeconds()) / (lastItemTime - firstItemTime).GetNanoSeconds();
            // if (dropProbDynamicCDF > 0) {

            //     std::cout << "E2E:     " << (packetKeyEventPair->second->GetSentTime() + hostToTorLinkDelay + hostToTorLinkRate.CalculateBytesTxTime(packetKeyEventPair->first.GetPacketSize())).GetNanoSeconds() << "    GTDropMean: " << GTDropMean << endl;
            //     std::cout << "prev: " << prev.GetNanoSeconds() << " first: " << firstItemTime.GetNanoSeconds() << " last: " << lastItemTime.GetNanoSeconds() << " dropProbDynamicCDF: " << dropProbDynamicCDF << endl;
            //     std::cout << 
            //     "prev_enq: " << (prev + hostToTorLinkDelay + hostToTorLinkRate.CalculateBytesTxTime(packetKeyEventPair->first.GetPacketSize())).GetNanoSeconds() << 
            //     " first_enq: " << (firstItemTime + hostToTorLinkDelay + hostToTorLinkRate.CalculateBytesTxTime(packetKeyEventPair->first.GetPacketSize())).GetNanoSeconds() << 
            //     " last: " << (lastItemTime + hostToTorLinkDelay + hostToTorLinkRate.CalculateBytesTxTime(packetKeyEventPair->first.GetPacketSize())).GetNanoSeconds() << " dropProbDynamicCDF: " << dropProbDynamicCDF << endl;
            // }
        }
    }
}

void E2EMonitor::SaveMonitorRecords(const string& filename) {
    ofstream outfile;
    outfile.open(filename);
    outfile << "path,sampleDelayMean,unbiasedSmapleDelayVariance,averagePacketSize,receivedPackets,sentPackets,markedPackets,timeAverage,sentPacketsOnLink,GTDropMean" << endl;
    for (int i = 0; i < numOfPaths; i++) {
        outfile << i << "," << sampleMean[i].GetNanoSeconds() << "," << unbiasedSmapleVariance[i].GetNanoSeconds() << "," << sumOfPacketSizes[i] / sampleSize[i] << "," << sampleSize[i] << "," << sentPackets[i] << "," << markedPackets[i] 
        << "," << timeAverageIntegral[i].GetNanoSeconds() / (integralEndTime[i] - integralStartTime[i]).GetNanoSeconds() << "," << sentPackets_onlink[i]
        << "," << GTDropMean << endl;
    }
    outfile << firstItemTime.GetNanoSeconds() << "," << lastItemTime.GetNanoSeconds() << endl;
    outfile.close();
    // ofstream packetsFile;
    // packetsFile.open(filename.substr(0, filename.size() - 4) + "_packets.csv");
    // packetsFile << "sentTime,receivedTime,size,path" << endl;
    // for (auto &recordedPacket : _recordedPackets) {
    //     packetsFile << recordedPacket.second->GetSentTime().GetNanoSeconds() << "," << recordedPacket.second->GetReceivedTime().GetNanoSeconds() << "," << recordedPacket.first.GetPacketSize() << "," << recordedPacket.second->GetPath() << endl;
    // }
    // packetsFile.close();
    ofstream packetsFile;
    packetsFile.open(filename.substr(0, filename.size() - 4) + "_packets.csv");
    packetsFile << "SourceIp,SourcePort,DestinationIp,DestinationPort,SequenceNb,Id,PayloadSize,Path,SentTime,IsReceived,ReceiveTime,ECN" << endl;
    for (auto& packetKeyEventPair: _recordedPackets) {
        PacketKey key = packetKeyEventPair.first;
        E2EMonitorEvent* event = packetKeyEventPair.second;

        packetsFile << key.GetSrcIp() << "," << key.GetSrcPort() << ",";
        packetsFile << key.GetDstIp() << "," << key.GetDstPort() << "," << key.GetSeqNb() << "," << key.GetId()  << "," << key.GetSize() << ",";
        packetsFile << event->GetPath() << ",";
        packetsFile << GetRelativeTime(event->GetSentTime()).GetNanoSeconds() << ",";
        packetsFile << event->IsReceived() << "," << GetRelativeTime(event->GetReceivedTime()).GetNanoSeconds() << "," << event->GetEcn() << endl;
    }
    packetsFile.close();
}
