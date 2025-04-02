//
// Created by Zeinab Shmeis on 28.05.20.
//

#include "E2EMonitor.h"

E2EMonitorEvent::E2EMonitorEvent(PacketKey *key) { SetPacketKey(key); }

void E2EMonitorEvent::SetEcn(bool ecn) { _key->SetEcn(ecn); }
void E2EMonitorEvent::SetPath(int path) { _key->SetPath(path);  }
void E2EMonitorEvent::SetTxEnqueueTime(Time time) { _TxEnqueueTime = time; }
void E2EMonitorEvent::SetTxDequeueTime(Time time) { _TxDequeuTime = time; }
void E2EMonitorEvent::SetTxIpTime(Time time) { _TxIpTime = time; }
bool E2EMonitorEvent::GetEcn() const { return _key->GetEcn(); } 
int E2EMonitorEvent::GetPath() const { return _key->GetPath(); }
Time E2EMonitorEvent::GetTxEnqueueTime() const { return _TxEnqueueTime; }
Time E2EMonitorEvent::GetTxDequeueTime() const { return _TxDequeuTime; }
Time E2EMonitorEvent::GetTxIpTime() const { return _TxIpTime; }


E2EMonitor::E2EMonitor(const Time &startTime, const Time &duration, const Time &steadyStartTime, const Time &steadyStopTime, const Ptr<PointToPointNetDevice> netDevice, const Ptr<Node> &rxNode, const string &monitorTag, double errorRate,
                       const DataRate &_hostToTorLinkRate, const DataRate &_torToAggLinkRate, const Time &_hostToTorLinkDelay) 
: E2EMonitor(startTime, duration, steadyStartTime, steadyStopTime, netDevice, rxNode, nullptr, monitorTag, errorRate, _hostToTorLinkRate, _torToAggLinkRate, _hostToTorLinkDelay, 2, 3, 10000) {
    
}

E2EMonitor::E2EMonitor(const Time &startTime, const Time &duration, const Time &steadyStartTime, const Time &steadyStopTime, const Ptr<PointToPointNetDevice> netDevice, const Ptr<Node> &rxNode, const Ptr<Node> &txNode, const string &monitorTag, double errorRate,
                       const DataRate &_hostToTorLinkRate, const DataRate &_torToAggLinkRate, const Time &_hostToTorLinkDelay, const int _numOfPaths, const int _numOfSegmetns, uint32_t queueCapacity) 
: Monitor(startTime, duration, steadyStartTime, steadyStopTime, monitorTag) {
    _txNode = txNode;
    numOfPaths = _numOfPaths;
    numOfSegmetns = _numOfSegmetns;
    _errorRate = errorRate;
    hostToTorLinkRate = _hostToTorLinkRate;
    torToAggLinkRate = _torToAggLinkRate;
    hostToTorLinkDelay = _hostToTorLinkDelay;
    for (int i = 0; i < numOfPaths; i++) {
        sampleMean.push_back(0);
        unbiasedSmapleVariance.push_back(0.0);
        sampleSize.push_back(0);
        sumOfPacketSizes.push_back(0);
        sentPackets.push_back(0);
        markedPackets.push_back(0);
        timeAverageIntegral.push_back(Seconds(0));
        integralStartTime.push_back(Time(-1));
        integralEndTime.push_back(Time(-1));
        lastDelay.push_back(Time(0));
        sentPackets_onlink.push_back(0);
    }
    packetCDF.loadCDFData("/media/experiments/ns-allinone-3.41/ns-3.41/scratch/ECNMC/Helpers/packet_size_cdf_singleQueue.csv");
    GTDropMean = 0;
    lastItemTime = Time(0);
    firstItemTime = Time(-1);
    hasher = Hasher();
    rand = CreateObject<UniformRandomVariable>();
    QueueCapacity = queueCapacity;
    Simulator::Schedule(_startTime, &E2EMonitor::Connect, this, netDevice, rxNode->GetId(), txNode->GetId());
    Simulator::Schedule(_startTime + _duration, &E2EMonitor::Disconnect, this, netDevice, rxNode->GetId(), txNode->GetId());
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

void E2EMonitor::Connect(const Ptr<PointToPointNetDevice> netDevice, uint32_t rxNodeId, uint32_t txNodeId) {
    netDevice->GetQueue()->TraceConnectWithoutContext("Enqueue", MakeCallback(&E2EMonitor::Enqueue, this));
    netDevice->TraceConnectWithoutContext("PromiscSniffer", MakeCallback(&E2EMonitor::Capture, this));
    Config::ConnectWithoutContext("/NodeList/" + to_string(rxNodeId) + "/$ns3::Ipv4L3Protocol/Rx", MakeCallback(
            &E2EMonitor::RecordIpv4PacketReceived, this));
    // Config::ConnectWithoutContext("/NodeList/" + to_string(txNodeId) + "/$ns3::Ipv4L3Protocol/Tx", MakeCallback(
    //         &E2EMonitor::RecordIpv4PacketSent, this));
}

void E2EMonitor::Disconnect(const Ptr<PointToPointNetDevice> netDevice, uint32_t rxNodeId, uint32_t txNodeId) {
    netDevice->GetQueue()->TraceDisconnectWithoutContext("Enqueue", MakeCallback(&E2EMonitor::Enqueue, this));
    Config::DisconnectWithoutContext("/NodeList/" + to_string(rxNodeId) + "/$ns3::Ipv4L3Protocol/Rx", MakeCallback(
            &E2EMonitor::RecordIpv4PacketReceived, this));
    // Config::DisconnectWithoutContext("/NodeList/" + to_string(txNodeId) + "/$ns3::Ipv4L3Protocol/Tx", MakeCallback(
    //         &E2EMonitor::RecordIpv4PacketSent, this));
}

void E2EMonitor::Capture(Ptr< const Packet > packet) {
    if (Simulator::Now() < _steadyStartTime || Simulator::Now() > _steadyStopTime) {
        return;
    }
    PacketKey* packetKey = PacketKey::Packet2PacketKey(packet, FIRST_HEADER_PPP);
    if(_appsKey.count(AppKey::PacketKey2AppKey(*packetKey))) {
        // if (!_observedAppsKey.count(AppKey::PacketKey2AppKey(*packetKey))) {
        //     _observedAppsKey.insert(AppKey::PacketKey2AppKey(*packetKey));
            // traceNewSockets();
        // }

        packetKey->SetPacketSize(packet->GetSize());
        E2EMonitorEvent* packetEvent;
        auto packetKeyEventPair = _recordedPackets.find(*packetKey);
        if (packetKeyEventPair == _recordedPackets.end()) {
            packetEvent = new E2EMonitorEvent(packetKey);
            _recordedPackets[*packetKey] = packetEvent;
        }
        else {
            packetEvent = packetKeyEventPair->second;
        }
        packetEvent->SetSent();
        packetEvent->SetTxDequeueTime(Simulator::Now());
        // uncomment the following line to record the packets when they are sent over the link
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
        sentPackets_onlink[hash % numOfPaths] += 1;
    }
}

void E2EMonitor::Enqueue(Ptr<const Packet> packet) {
    if (Simulator::Now() < _steadyStartTime || Simulator::Now() > _steadyStopTime) {
        return;
    }
    PacketKey* packetKey = PacketKey::Packet2PacketKey(packet, FIRST_HEADER_PPP);
    if(_appsKey.count(AppKey::PacketKey2AppKey(*packetKey))) {
        // E2EMonitorEvent* packetEvent;
        // auto packetKeyEventPair = _recordedPackets.find(*packetKey);
        // if (packetKeyEventPair == _recordedPackets.end()) {
        //     packetEvent = new E2EMonitorEvent(packetKey);
        //     _recordedPackets[*packetKey] = packetEvent;
        // }
        // else {
        //     packetEvent = packetKeyEventPair->second;
        // }
        packetKey->SetPacketSize(packet->GetSize());
        auto *packetEvent = new E2EMonitorEvent(packetKey);
        packetEvent->SetTxEnqueueTime(Simulator::Now());
        // uncomment the following line to record the packets when they are enqueued
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
        sentPackets[hash % numOfPaths] += 1;
    }
}

void E2EMonitor::markingProbUpdate(uint32_t bytesMarked, uint32_t bytesAcked, double alpha, Time rtt) {
    double F = 0.0;
    if (bytesAcked > 0) {
        F = bytesMarked * 1.0 / bytesAcked;
    }
    // cout << "Now: " << Simulator::Now().GetNanoSeconds() << " Bytes Acked: " << bytesAcked << " Bytes Marked: " << bytesMarked << " F: " <<  F << endl;
    markingProbProcess.push_back(std::make_tuple(Simulator::Now(), bytesAcked, F, rtt));
}

// void E2EMonitor::NewAck(SequenceNumber32 sqn) {
    
// }
void E2EMonitor::traceNewSockets() {
    Ptr<TcpL4Protocol> tcp = _txNode->GetObject<TcpL4Protocol>();
    ObjectMapValue sockets;
    tcp->GetAttribute("SocketList", sockets);
    for (auto it = sockets.Begin(); it != sockets.End(); ++it) {
       if (tracesSockets.find(it->first) == tracesSockets.end()) {
            tracesSockets[it->first] = DynamicCast<TcpSocketBase>(it->second);
            // tracesSockets[it->first]->TraceConnectWithoutContext("HighestRxAck", MakeCallback(&E2EMonitor::NewAck, this));
            tracesSockets[it->first]->GetCongestionControlAlgorithm()->GetObject<TcpDctcp>()->TraceConnectWithoutContext("CongestionEstimate", MakeCallback(&E2EMonitor::markingProbUpdate, this));
       }
    }
}

void E2EMonitor::updateTimeAverageIntegral(uint32_t path, Time delay, Time endTime) {
    if (integralStartTime[path] != Time(-1)) {
        // timeAverageIntegral[path] += Time(((delay + lastDelay[path]) / 2).GetNanoSeconds() * (endTime - integralEndTime[path]).GetNanoSeconds());
        timeAverageIntegral[path] += Time(delay.GetNanoSeconds() * (endTime - integralEndTime[path]).GetNanoSeconds());
    }

    if (integralStartTime[path] == Time(-1)) {
        integralStartTime[path] = endTime;
        lastDelay[path] = delay;
    }
    integralEndTime[path] = endTime;
    lastDelay[path] = delay;
}

void E2EMonitor::RecordIpv4PacketSent(Ptr<const Packet> packet, Ptr<Ipv4> ipv4, uint32_t interface) {   
    if (Simulator::Now() < _steadyStartTime || Simulator::Now() > _steadyStopTime) {
        return;
    } 
    PacketKey* packetKey = PacketKey::Packet2PacketKey(packet, FIRST_HEADER_IPV4);
    if(_appsKey.count(AppKey::PacketKey2AppKey(*packetKey))) {
        auto *packetEvent = new E2EMonitorEvent(packetKey);
        packetEvent->SetTxIpTime(Simulator::Now());
        _recordedPackets[*packetKey] = packetEvent;
    }
    
}

void E2EMonitor::RecordIpv4PacketReceived(Ptr<const Packet> packet, Ptr<Ipv4> ipv4, uint32_t interface) {    
    PacketKey* packetKey = PacketKey::Packet2PacketKey(packet, FIRST_HEADER_IPV4);
    if(_appsKey.count(AppKey::PacketKey2AppKey(*packetKey))) {
        auto packetKeyEventPair = _recordedPackets.find(*packetKey);
        if (packetKeyEventPair != _recordedPackets.end()) {
            if (packetKeyEventPair->second->GetTxDequeueTime() == Time(-1)) {
                _recordedPackets.erase(packetKeyEventPair);
                return;
            }
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

            // updateBasicCounters(packetKeyEventPair->second->GetTxEnqueueTime(), packetKeyEventPair->second->GetReceivedTime() + additionalDeprioritizationDelay - transmissionDelay, path);
            // updateTimeAverageIntegral(path, packetKeyEventPair->second->GetReceivedTime() + additionalDeprioritizationDelay - transmissionDelay - packetKeyEventPair->second->GetTxEnqueueTime(), packetKeyEventPair->second->GetTxEnqueueTime());
            updateBasicCounters(packetKeyEventPair->second->GetSentTime(), packetKeyEventPair->second->GetReceivedTime() + additionalDeprioritizationDelay - transmissionDelay, path);
            updateTimeAverageIntegral(path, packetKeyEventPair->second->GetReceivedTime() + additionalDeprioritizationDelay - transmissionDelay - packetKeyEventPair->second->GetSentTime(), packetKeyEventPair->second->GetSentTime() + hostToTorLinkRate.CalculateBytesTxTime(packetKeyEventPair->first.GetPacketSize()) + hostToTorLinkDelay);

            // remove the packet from the map to reduce the memory usage of the simulation
            // _recordedPackets.erase(packetKeyEventPair);
            Time prev = lastItemTime;
            lastItemTime = packetKeyEventPair->second->GetSentTime() + hostToTorLinkRate.CalculateBytesTxTime(packetKeyEventPair->first.GetPacketSize()) + hostToTorLinkDelay;
            if (firstItemTime == Time(-1)) {
                firstItemTime = packetKeyEventPair->second->GetSentTime() + hostToTorLinkRate.CalculateBytesTxTime(packetKeyEventPair->first.GetPacketSize()) + hostToTorLinkDelay;
                return;
            }
            int availableCapacity = max((int) packetKeyEventPair->first.GetPacketSize(), (int) (QueueCapacity - (((packetKeyEventPair->second->GetReceivedTime() - transmissionDelay - packetKeyEventPair->second->GetSentTime()) * torToAggLinkRate) / 8)));
            double dropProbDynamicCDF = packetCDF.calculateProbabilityGreaterThan(availableCapacity);
            
            GTDropMean = (GTDropMean * (prev - firstItemTime).GetNanoSeconds() + dropProbDynamicCDF * (lastItemTime - prev).GetNanoSeconds()) / (lastItemTime - firstItemTime).GetNanoSeconds();
            // cout << "### E2E ### Enqueue Time: " << lastItemTime.GetNanoSeconds() << " Queue Size: " << (((packetKeyEventPair->second->GetReceivedTime() - transmissionDelay - packetKeyEventPair->second->GetSentTime()) * torToAggLinkRate) / 8) << endl;
        }
    }
}

double E2EMonitor::calculateUnbiasedGTDrop() {
    std::vector<E2EMonitorEvent*> sortedSamples;
    for (auto &sample : _recordedPackets) {
        if (sample.second->GetSentTime() != Time(-1)) {
            sortedSamples.push_back(sample.second);
        }
    }
    std::sort(sortedSamples.begin(), sortedSamples.end(), [](E2EMonitorEvent* a, E2EMonitorEvent* b) {
        return a->GetSentTime() < b->GetSentTime();
    });
    double GTDropMean = 0;
    Time lastItemTime = Time(0);
    Time firstItemTime = Time(-1);
    for (auto &sample : sortedSamples) {
        Time prev = lastItemTime;
        uint32_t packteSize = _recordedPackets.find(*sample->GetPacketKey())->first.GetPacketSize();
        lastItemTime = sample->GetSentTime() + hostToTorLinkRate.CalculateBytesTxTime(packteSize) + hostToTorLinkDelay;
        if (firstItemTime == Time(-1)) {
            firstItemTime = lastItemTime;
            continue;
        }
        double dropProbDynamicCDF = 0;
        if (sample->GetReceivedTime() == Time(-1)) {
            // dropProbDynamicCDF = packetCDF.calculateProbabilityGreaterThan(packteSize);
            dropProbDynamicCDF = 1.0;
        }
        else {
            Time transmissionDelay = hostToTorLinkRate.CalculateBytesTxTime(packteSize)
                                    + torToAggLinkRate.CalculateBytesTxTime(packteSize)
                                    + hostToTorLinkDelay * 2;

            int availableCapacity = max((int) packteSize, (int)(QueueCapacity - (((sample->GetReceivedTime() - transmissionDelay - sample->GetSentTime()) * torToAggLinkRate) / 8)));
            dropProbDynamicCDF = packetCDF.calculateProbabilityGreaterThan(availableCapacity);   
        }
        GTDropMean = (GTDropMean * (prev - firstItemTime).GetNanoSeconds() + dropProbDynamicCDF * (lastItemTime - prev).GetNanoSeconds()) / (lastItemTime - firstItemTime).GetNanoSeconds();
    }
    return GTDropMean;

}
void E2EMonitor::SaveMonitorRecords(const string& filename) {
    ofstream outfile;
    outfile.open(filename);
    outfile << "path,sampleDelayMean,unbiasedSmapleDelayVariance,averagePacketSize,receivedPackets,sentPackets,markedPackets,timeAverage,sentPacketsOnLink,GTDropMean,UnbiasedGTDropMean,OWAQsize" << endl;
    for (int i = 0; i < numOfPaths; i++) {
        outfile << i << "," << sampleMean[i] << "," << unbiasedSmapleVariance[i] << "," << sumOfPacketSizes[i] / sampleSize[i] << "," << sampleSize[i] << "," << sentPackets[i] << "," << markedPackets[i] 
        << "," << timeAverageIntegral[i].GetNanoSeconds() / (integralEndTime[i] - integralStartTime[i]).GetNanoSeconds() << "," << sentPackets_onlink[i]
        << "," << GTDropMean << "," << calculateUnbiasedGTDrop() 
        << "," << (Time(timeAverageIntegral[i].GetNanoSeconds() / (integralEndTime[i] - integralStartTime[i]).GetNanoSeconds()) * torToAggLinkRate) / 8 << endl;

    }
    outfile.close();

    ofstream packetsFile;
    packetsFile.open(filename.substr(0, filename.size() - 4) + "_packets.csv");
    packetsFile << "SourceIp,SourcePort,DestinationIp,DestinationPort,SequenceNb,Id,PayloadSize,Path,TxEnqueueTime,TxDequeueTime,SentTime,IsReceived,ReceiveTime,transmissionDelay,ECN" << endl;
    for (auto& packetKeyEventPair: _recordedPackets) {
        PacketKey key = packetKeyEventPair.first;
        E2EMonitorEvent* event = packetKeyEventPair.second;
        Time transmissionDelay = Seconds(0);
        if (numOfSegmetns == 3) {
            transmissionDelay = hostToTorLinkRate.CalculateBytesTxTime(key.GetPacketSize()) * 2
                                + torToAggLinkRate.CalculateBytesTxTime(key.GetPacketSize()) * 2
                                + hostToTorLinkDelay * 4;
        } else if (numOfSegmetns == 1) {
            transmissionDelay = hostToTorLinkRate.CalculateBytesTxTime(key.GetPacketSize())
                                + torToAggLinkRate.CalculateBytesTxTime(key.GetPacketSize())
                                + hostToTorLinkDelay * 2;
        }
        packetsFile << key.GetSrcIp() << "," << key.GetSrcPort() << ",";
        packetsFile << key.GetDstIp() << "," << key.GetDstPort() << "," << key.GetSeqNb() << "," << key.GetId()  << "," << key.GetPacketSize() << ",";
        packetsFile << event->GetPath() << ",";
        packetsFile << event->GetTxEnqueueTime().GetNanoSeconds() << "," << event->GetTxDequeueTime().GetNanoSeconds() << ",";
        packetsFile << event->GetSentTime().GetNanoSeconds() << ",";
        packetsFile << event->IsReceived() << "," << event->GetReceivedTime().GetNanoSeconds() << "," << transmissionDelay.GetNanoSeconds() << "," << event->GetEcn() << endl;
    }
    packetsFile.close();

    ofstream markingsFile;
    markingsFile.open(filename.substr(0, filename.size() - 4) + "_markings.csv");
    markingsFile << "Time,BytesAcked,MarkingProb,rtt" << endl;
    for (auto &item : markingProbProcess) {
        markingsFile << std::get<0>(item).GetNanoSeconds() << "," << std::get<1>(item) << "," << std::get<2>(item) << "," << std::get<3>(item).GetNanoSeconds() << endl;
    }
    markingsFile.close();
}
