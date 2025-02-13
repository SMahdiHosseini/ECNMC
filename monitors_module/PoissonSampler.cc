//
// Created by Mahdi Hosseini on 24.04.24.
//
#include "PoissonSampler.h"
#include <iomanip>

samplingEvent::samplingEvent(PacketKey *key) { SetPacketKey(key); }

void samplingEvent::SetSampleTime() { SetSent(ns3::Simulator::Now()); }
void samplingEvent::SetSampleTime(Time t) { SetSent(t); }
void samplingEvent::SetDepartureTime(Time t) { SetReceived(t); }
void samplingEvent::SetDepartureTime() { SetReceived(ns3::Simulator::Now()); }
void samplingEvent::SetMarkingProb(double markingProb) { _markingProb = markingProb; }
PacketKey *samplingEvent::GetPacketKey() const { return _key; }
Time samplingEvent::GetSampleTime() const { return GetSentTime(); }
Time samplingEvent::GetDepartureTime() const { return GetReceivedTime(); }
double samplingEvent::GetMarkingProb() const { return _markingProb; }
bool samplingEvent::IsDeparted() const { return GetReceivedTime() != Time(-1); }

PoissonSampler::PoissonSampler(const Time &steadyStartTime, const Time &steadyStopTime, Ptr<RedQueueDisc> queueDisc, Ptr<Queue<Packet>> queue, Ptr<PointToPointNetDevice>  outgoingNetDevice, const string &sampleTag, double sampleRate) 
: Monitor(Seconds(0), steadyStopTime, steadyStartTime, steadyStopTime, sampleTag) {
    REDQueueDisc = queueDisc;
    NetDeviceQueue = queue;
    m_var = CreateObject<ExponentialRandomVariable>();
    m_var->SetAttribute("Mean", DoubleValue(1/sampleRate));
    _sampleRate = sampleRate;
    zeroDelayPort = 0;
    samplesDropMean = 0;
    samplesDropVariance = 0;
    sampleMean.push_back(Seconds(0));
    unbiasedSmapleVariance.push_back(Seconds(0));
    sampleSize.push_back(0);
    // packetCDF.loadCDFData("/media/experiments/ns-allinone-3.41/ns-3.41/scratch/ECNMC/Helpers/packet_size_cdf.csv");
    packetCDF.loadCDFData("/media/experiments/ns-allinone-3.41/ns-3.41/scratch/ECNMC/Helpers/packet_size_cdf_singleQueue.csv");
    numOfGTSamples = 0;
    GTPacketSizeMean = 0;
    GTDropMean = 0;
    GTQueuingDelay = 0;
    firstItemTime = Time(-1);
    lastItemTime = Time(0);
    outgoingDataRate = outgoingNetDevice->GetDataRate();
    Simulator::Schedule(Seconds(0), &PoissonSampler::Connect, this, outgoingNetDevice);
    Simulator::Schedule(steadyStopTime, &PoissonSampler::Disconnect, this, outgoingNetDevice);
}

uint32_t PoissonSampler::ComputeQueueSize() {
    uint32_t TXedBytes = (outgoingDataRate * (Simulator::Now() - lastLeftTime)) / 8;
    uint32_t remainedBytes = (lastLeftSize > TXedBytes) ? lastLeftSize - TXedBytes : 0;
    if (REDQueueDisc != nullptr) {
        return REDQueueDisc->GetCurrentSize().GetValue() + NetDeviceQueue->GetNBytes() + (REDQueueDisc->GetNPackets() - 1) * 2 + remainedBytes - 2;
    }
    return NetDeviceQueue->GetNBytes() + remainedBytes;
}

void PoissonSampler::EnqueueQueueDisc(Ptr<const QueueDiscItem> item) {
    lastItem = item;
    // packetCDF.addPacket(item->GetSize());
    lastItemTime = Simulator::Now();
}

void PoissonSampler::EnqueueNetDeviceQueue(Ptr<const Packet> packet) {
    lastPacket = packet;
    if (Simulator::Now() < _steadyStartTime || Simulator::Now() > _steadyStopTime) {
        return;
    }
    // uncomment the fllowing lines to calculate the Ground Truth Mean Drop probability
    // const Ptr<Packet> &pktCopy = packet->Copy();
    // PppHeader pppHeader;
    // pktCopy->RemoveHeader(pppHeader);
    // Ipv4Header ipHeader;
    // pktCopy->RemoveHeader(ipHeader);
    // if (ipHeader.GetSource() != Ipv4Address("10.1.1.1")) {
    //     return;
    // }
    // Time prev = lastPacketTime;
    // lastPacketTime = Simulator::Now();
    // if (firstItemTime == Time(-1)) {
    //     firstItemTime = lastPacketTime;
    //     return;
    // }
    // Time queuingDelay = outgoingDataRate.CalculateBytesTxTime(ComputeQueueSize() - packet->GetSize());
    // GTQueuingDelay = ((GTQueuingDelay * (prev - firstItemTime).GetNanoSeconds()) + (queuingDelay.GetNanoSeconds() * (lastPacketTime - prev).GetNanoSeconds())) / (lastPacketTime - firstItemTime).GetNanoSeconds();
    // queueSizeProcess.push_back(std::make_tuple(Simulator::Now(), ComputeQueueSize() - packet->GetSize()));
    // cout << "### POISSON ### Enqueue Time: " << Simulator::Now().GetNanoSeconds() << " Queueing Delay: " << queuingDelay.GetNanoSeconds() << " Queue Size: " << ComputeQueueSize() - packet->GetSize() << " GTQueuingDelay: " << GTQueuingDelay << endl;

    // double dropProbDynamicCDF = 0;
    // // if (QueueSize("100KB").GetValue() <= ComputeQueueSize()){
    // //     dropProbDynamicCDF = 1.0;
    // // }
    // // else {
    //     dropProbDynamicCDF = packetCDF.calculateProbabilityGreaterThan(NetDeviceQueue->GetMaxSize().GetValue() - NetDeviceQueue->GetCurrentSize().GetValue());
    // // }
    // GTDropMean = (GTDropMean * (prev - firstItemTime).GetNanoSeconds() + dropProbDynamicCDF * (lastPacketTime - prev).GetNanoSeconds()) / (Simulator::Now() - firstItemTime).GetNanoSeconds();
    // cout << "### POISSON ### Enqueue Time: " << Simulator::Now().GetNanoSeconds() << " Queue Size: " << NetDeviceQueue->GetCurrentSize().GetValue() << " Drop Prob: " << dropProbDynamicCDF << " GTDropMean: " << GTDropMean << endl;
    // cout << "Vars: " << " firstItemTime: " << firstItemTime.GetNanoSeconds() << " lastItemTime: " << lastPacketTime.GetNanoSeconds() << " prev: " << prev.GetNanoSeconds() << endl;
}

void PoissonSampler::Connect(Ptr<PointToPointNetDevice> outgoingNetDevice) {
    if (REDQueueDisc != nullptr) {
        REDQueueDisc->TraceConnectWithoutContext("Enqueue", MakeCallback(&PoissonSampler::EnqueueQueueDisc, this));
    }
    NetDeviceQueue->TraceConnectWithoutContext("Enqueue", MakeCallback(&PoissonSampler::EnqueueNetDeviceQueue, this));
    outgoingNetDevice->TraceConnectWithoutContext("PromiscSniffer", MakeCallback(&PoissonSampler::RecordPacket, this));
    // generate the first event
    double nextEvent = m_var->GetValue();
    Simulator::Schedule(Seconds(nextEvent), &PoissonSampler::EventHandler, this);
}

void PoissonSampler::Disconnect(Ptr<PointToPointNetDevice> outgoingNetDevice) {
    outgoingNetDevice->TraceDisconnectWithoutContext("PromiscSniffer", MakeCallback(&PoissonSampler::RecordPacket, this));
    if (REDQueueDisc != nullptr) {
        REDQueueDisc->TraceDisconnectWithoutContext("Enqueue", MakeCallback(&PoissonSampler::EnqueueQueueDisc, this));
    }
    NetDeviceQueue->TraceDisconnectWithoutContext("Enqueue", MakeCallback(&PoissonSampler::EnqueueNetDeviceQueue, this));
}

void PoissonSampler::EventHandler() {
    // Generate a new event
    double nextEvent = m_var->GetValue();
    if (Simulator::Now() > _steadyStopTime) {
        return;
    }
    Simulator::Schedule(Seconds(nextEvent), &PoissonSampler::EventHandler, this);
    if (Simulator::Now() < _steadyStartTime) {
        return;
    }
    
    double dropProbDynamicCDF = 0;
    uint32_t queueSize;
    if (REDQueueDisc != nullptr) {
        queueSize = REDQueueDisc->GetNBytes() + NetDeviceQueue->GetNBytes();
        dropProbDynamicCDF = packetCDF.calculateProbabilityGreaterThan(REDQueueDisc->GetMaxSize().GetValue() - REDQueueDisc->GetNBytes());
    }
    else {
        queueSize = NetDeviceQueue->GetNBytes();
        //TODO: the following line has to be fixed. It is not correct if the queue max size is in packets
        dropProbDynamicCDF = packetCDF.calculateProbabilityGreaterThan(NetDeviceQueue->GetMaxSize().GetValue() - NetDeviceQueue->GetCurrentSize().GetValue());
    }
    // queueSize = ComputeQueueSize();
    PacketKey* packetKey = new PacketKey(ns3::Ipv4Address("0.0.0.0"), ns3::Ipv4Address("0.0.0.1"), 0, zeroDelayPort++, zeroDelayPort++, ns3::SequenceNumber32(0), ns3::SequenceNumber32(0), 0, 0);
    Time queuingDelay = outgoingDataRate.CalculateBytesTxTime(queueSize);
    samplingEvent* event = new samplingEvent(packetKey);
    event->SetSampleTime(Simulator::Now());
    event->SetDepartureTime(Simulator::Now() + queuingDelay);
    event->SetMarkingProb(dropProbDynamicCDF);
    _recordedSamples[*packetKey] = event;
    // cout << "### EVENT ### " << "Time: " << Simulator::Now().GetNanoSeconds() << " Queuing delay: " << queuingDelay.GetNanoSeconds() << " Queue Size: " << queueSize << " Drop Prob: " << dropProbDynamicCDF << endl;
    updateCounters(event);
}

void PoissonSampler::updateCounters(samplingEvent* event) {
    updateBasicCounters(event->GetSampleTime(), event->GetDepartureTime(), 0);
    
    double delta = (event->GetMarkingProb() - samplesDropMean);
    samplesDropMean = samplesDropMean + (delta / sampleSize[0]);
    if (sampleSize[0] <= 1) {
        samplesDropVariance = 0;
    }
    else {
        samplesDropVariance = samplesDropVariance + ((delta* delta) / sampleSize[0]) - (samplesDropVariance / (sampleSize[0] - 1));
    }
}

void PoissonSampler::RecordPacket(Ptr<const Packet> packet) {
    const Ptr<Packet> &pktCopy = packet->Copy();
    PppHeader pppHeader;
    pktCopy->RemoveHeader(pppHeader);
    Ipv4Header ipHeader;
    pktCopy->RemoveHeader(ipHeader);
    if (ipHeader.GetSource() != Ipv4Address("10.3.1.1")) {
        lastLeftTime = Simulator::Now();
        lastLeftSize = packet->GetSize();
    }
    PacketKey* packetKey = PacketKey::Packet2PacketKey(packet, FIRST_HEADER_PPP);
    if (_recordedSamples.find(*packetKey) != _recordedSamples.end()) {
        _recordedSamples[*packetKey]->SetDepartureTime();
        updateCounters(_recordedSamples[*packetKey]);
        // remove the packet from the map to reduce the memory usage of the simulation
        _recordedSamples.erase(*packetKey);
        // bool ECNFlag = false;
        // if (REDQueueDisc == nullptr) {
        //     const Ptr<Packet> &pktCopy = packet->Copy();
        //     PppHeader pppHeader;
        //     Ipv4Header header;
        //     pktCopy->RemoveHeader(pppHeader);
        //     pktCopy->RemoveHeader(header);
        //     if (header.EcnTypeToString(header.GetEcn()) == "CE") {
        //         _recordedSamples[*packetKey]->SetMarkingProb(1.0);
        //         ECNFlag = true;
        //     }
        // }

        // check if there exists a packet with the same key but different record field
        packetKey->SetRecords(packetKey->GetRecords() + 1);
        while (_recordedSamples.find(*packetKey) != _recordedSamples.end())
        {
            _recordedSamples[*packetKey]->SetDepartureTime();
            updateCounters(_recordedSamples[*packetKey]);
            // remove the packet from the map to reduce the memory usage of the simulation
            _recordedSamples.erase(*packetKey);
            // if (ECNFlag) {
            //     _recordedSamples[*packetKey]->SetMarkingProb(1.0);
            // }
            packetKey->SetRecords(packetKey->GetRecords() + 1);
        }
    }
}


void PoissonSampler::SaveMonitorRecords(const string& filename) {
    ofstream outfile;
    outfile.open(filename);
    outfile << "sampleDelayMean,unbiasedSmapleDelayVariance,sampleSize,samplesDropMean,samplesDropVariance,GTSampleSize,GTPacketSizeMean,GTDropMean,GTQueuingDelay" << endl;
    outfile << sampleMean[0].GetNanoSeconds() << "," << unbiasedSmapleVariance[0].GetNanoSeconds() << "," << sampleSize[0] << "," << samplesDropMean << "," << samplesDropVariance << "," << numOfGTSamples << "," << GTPacketSizeMean << "," << GTDropMean << "," << GTQueuingDelay << endl;
    outfile << "Time,QueuingDelay,DropProb" << endl;
    for (auto &item : _recordedSamples) {
        outfile << item.second->GetSampleTime().GetNanoSeconds() << "," << (item.second->GetDepartureTime() - item.second->GetSampleTime()).GetNanoSeconds() << "," << item.second->GetMarkingProb() << endl;
    }
    outfile.close();
    // if (_monitorTag == "SD0") {
    //     packetCDF.printCDF();
    // }
}
