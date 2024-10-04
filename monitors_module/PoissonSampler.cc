//
// Created by Mahdi Hosseini on 24.04.24.
//
#include "PoissonSampler.h"
#include <iomanip>

samplingEvent::samplingEvent(PacketKey *key) { SetPacketKey(key); }

void samplingEvent::SetSampleTime() { SetSent(ns3::Simulator::Now()); }
void samplingEvent::SetSampleTime(Time t) { SetSent(t); }
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

    Simulator::Schedule(Seconds(0), &PoissonSampler::Connect, this, outgoingNetDevice);
    Simulator::Schedule(steadyStopTime, &PoissonSampler::Disconnect, this, outgoingNetDevice);
}

void PoissonSampler::EnqueueQueueDisc(Ptr<const QueueDiscItem> item) {
    lastItem = item;
    lastItemTime = Simulator::Now();
}

void PoissonSampler::EnqueueNetDeviceQueue(Ptr<const Packet> packet) {
    lastPacket = packet;
    lastPacketTime = Simulator::Now();
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
    Simulator::Schedule(Seconds(nextEvent), &PoissonSampler::EventHandler, this);
    if (Simulator::Now() < _steadyStartTime || Simulator::Now() > _steadyStopTime) {
        return;
    }
    PacketKey* packetKey;
    bool zeroDelay = false;
    bool drop = false;
    Time sampleTime = Simulator::Now();
    if (REDQueueDisc != nullptr && REDQueueDisc->GetNPackets() > 0) {
        // check the quque disc size
        if (REDQueueDisc->GetNPackets() > 0) {
            // extract transport layer info
            Ipv4Header ipHeader = DynamicCast<const Ipv4QueueDiscItem>(lastItem)->GetHeader();
            if (ipHeader.GetProtocol() == 6){
                packetKey = PacketKey::Packet2PacketKey(lastItem->GetPacket(), FIRST_HEADER_TCP);
            }
            else {
                packetKey = PacketKey::Packet2PacketKey(lastItem->GetPacket(), FIRST_HEADER_UDP);
            }
            packetKey->SetId(ipHeader.GetIdentification());
            packetKey->SetSrcIp(ipHeader.GetSource());
            packetKey->SetDstIp(ipHeader.GetDestination());
            sampleTime = lastItemTime;
            if ((QueueSize("37.5KB").GetValue() - REDQueueDisc->GetCurrentSize().GetValue()) < QueueSize("700B").GetValue()) {
                drop = true;
            }

        }
    }
    else if (NetDeviceQueue->GetNPackets() > 0) {
        packetKey = PacketKey::Packet2PacketKey(lastPacket, FIRST_HEADER_PPP);
        sampleTime = lastPacketTime;
        if ((REDQueueDisc == nullptr) && (NetDeviceQueue->GetCurrentSize() >= QueueSize("100p"))) {
            drop = true;
        }
    }
    else {
        packetKey = new PacketKey(ns3::Ipv4Address("0.0.0.0"), ns3::Ipv4Address("0.0.0.1"), 0, zeroDelayPort++, zeroDelayPort++, ns3::SequenceNumber32(0), ns3::SequenceNumber32(0), 0, 0);
        zeroDelay = true;
        sampleTime = Simulator::Now();
    }
    
    // add the event to the recorded samples
    samplingEvent* event = new samplingEvent(packetKey);
    // check if the packet is already recorded, add 1 to the record field of the packet and add the event to the recorded samples
    while (_recordedSamples.find(*packetKey) != _recordedSamples.end()) {
        packetKey->SetRecords(packetKey->GetRecords() + 1);
    }
    event->SetSampleTime(sampleTime);
    _recordedSamples[*packetKey] = event;
    // if there is no packet in the queue, then add the event pair with zero delay
    if (zeroDelay) {
        event->SetDepartureTime();
        updateCounters(event);
    }
    // set the drop information
    // if (REDQueueDisc != nullptr) {
    //     event->SetMarkingProb(REDQueueDisc->GetMarkingProbability());
    // }
    if (drop) {
        event->SetMarkingProb(1.0);
    }
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
    outfile << "sampleDelayMean,unbiasedSmapleDelayVariance,sampleSize,samplesDropMean,samplesDropVariance" << endl;
    outfile << sampleMean[0].GetNanoSeconds() << "," << unbiasedSmapleVariance[0].GetNanoSeconds() << "," << sampleSize[0] << "," << samplesDropMean << "," << samplesDropVariance << endl;
    outfile.close();
}
