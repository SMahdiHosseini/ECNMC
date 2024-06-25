//
// Created by Mahdi Hosseini on 24.04.24.
//
#include "PoissonSampler.h"
#include <iomanip>

samplingEvent::samplingEvent(PacketKey *key) : _key(key) {}

void samplingEvent::SetSampleTime() { _sampleTime = ns3::Simulator::Now(); }
void samplingEvent::SetDepartureTime() { _departureTime = ns3::Simulator::Now(); }
void samplingEvent::SetMarkingProb(double markingProb) { _markingProb = markingProb; }
PacketKey *samplingEvent::GetPacketKey() const { return _key; }
Time samplingEvent::GetSampleTime() const { return _sampleTime; }
Time samplingEvent::GetDepartureTime() const { return _departureTime; }
double samplingEvent::GetMarkingProb() const { return _markingProb; }
bool samplingEvent::IsDeparted() const { return _departureTime != Time(-1); }

ostream &operator<<(ostream &os, const samplingEvent &event) {
    os << "SamplingEvent: [ ";
    os << "Key = " << *(event._key) << ", SampleTime = " << event._sampleTime << ", DepartureTime = " << event._departureTime;
    os << "]";
    return os;
}

PoissonSampler::PoissonSampler(const Time &startTime, const Time &duration, Ptr<RedQueueDisc> queueDisc, Ptr<Queue<Packet>> queue, Ptr<PointToPointNetDevice>  outgoingNetDevice, const string &sampleTag, double sampleRate) {
    _startTime = startTime;
    _duration = duration;
    _sampleTag = sampleTag;
    REDQueueDisc = queueDisc;
    NetDeviceQueue = queue;
    m_var = CreateObject<ExponentialRandomVariable>();
    m_var->SetAttribute("Mean", DoubleValue(1/sampleRate));
    _sampleRate = sampleRate;
    zeroDelayPort = 0;

    Simulator::Schedule(_startTime, &PoissonSampler::Connect, this, outgoingNetDevice);
    Simulator::Schedule(_startTime + _duration, &PoissonSampler::Disconnect, this, outgoingNetDevice);
}

void PoissonSampler::EnqueueQueueDisc(Ptr<const QueueDiscItem> item) {
    lastItem = item;
    // PacketKey* packetKey = PacketKey::Packet2PacketKey(lastItem->GetPacket(), FIRST_HEADER_TCP);
    // Ipv4Header ipHeader = DynamicCast<const Ipv4QueueDiscItem>(lastItem)->GetHeader();
    // packetKey->SetId(ipHeader.GetIdentification());
    // packetKey->SetSrcIp(ipHeader.GetSource());
    // packetKey->SetDstIp(ipHeader.GetDestination());
    // std::cout << Simulator::Now().GetNanoSeconds() << " : " << packetKey->GetSrcIp() << "," << packetKey->GetSrcPort() << "," << packetKey->GetDstIp() << "," << packetKey->GetDstPort() << "," << packetKey->GetSeqNb() << std::endl;
}

void PoissonSampler::EnqueueNetDeviceQueue(Ptr<const Packet> packet) {
    lastPacket = packet;
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
    PacketKey* packetKey;
    bool zeroDelay = false;
    bool drop = false;
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

            if ((QueueSize("37.5KB").GetValue() - REDQueueDisc->GetCurrentSize().GetValue()) < QueueSize("1448B").GetValue()) {
                drop = true;
            }

        }
    }
    else if (NetDeviceQueue->GetNPackets() > 0) {
        packetKey = PacketKey::Packet2PacketKey(lastPacket, FIRST_HEADER_PPP);
        if ((REDQueueDisc == nullptr) && (NetDeviceQueue->GetCurrentSize() >= QueueSize("100p"))) {
            drop = true;
        }
    }
    else {
        packetKey = new PacketKey(ns3::Ipv4Address("0.0.0.0"), ns3::Ipv4Address("0.0.0.1"), 0, zeroDelayPort++, zeroDelayPort++, ns3::SequenceNumber32(0), ns3::SequenceNumber32(0), 0, 0);
        zeroDelay = true;
    }
    
    // add the event to the recorded samples
    samplingEvent* event = new samplingEvent(packetKey);
    // check if the packet is already recorded, add 1 to the record field of the packet and add the event to the recorded samples
    while (_recordedSamples.find(*packetKey) != _recordedSamples.end()) {
        packetKey->SetRecords(packetKey->GetRecords() + 1);
    }
    event->SetSampleTime();
    _recordedSamples[*packetKey] = event;
    // if there is no packet in the queue, then add the event pair with zero delay
    if (zeroDelay) {
        event->SetDepartureTime();
    }
    // set the drop information
    // if (REDQueueDisc != nullptr) {
    //     event->SetMarkingProb(REDQueueDisc->GetMarkingProbability());
    // }
    if (drop) {
        event->SetMarkingProb(1.0);
    }
    // Generate a new event
    double nextEvent = m_var->GetValue();
    Simulator::Schedule(Seconds(nextEvent), &PoissonSampler::EventHandler, this);
}

void PoissonSampler::RecordPacket(Ptr<const Packet> packet) {
    PacketKey* packetKey = PacketKey::Packet2PacketKey(packet, FIRST_HEADER_PPP);
    if (_recordedSamples.find(*packetKey) != _recordedSamples.end()) {
        _recordedSamples[*packetKey]->SetDepartureTime();
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
            // if (ECNFlag) {
            //     _recordedSamples[*packetKey]->SetMarkingProb(1.0);
            // }
            packetKey->SetRecords(packetKey->GetRecords() + 1);
        }
    }
}


void PoissonSampler::SaveSamples(const string& filename) {
    ofstream outfile;
    outfile.open(filename);
    outfile << "SourceIp,SourcePort,DestinationIp,DestinationPort,SequenceNb,Id,PayloadSize,SampleTime,IsDeparted,DepartTime,MarkingProb" << endl;
    for (auto& packetKeyEventPair: _recordedSamples) {
        PacketKey key = packetKeyEventPair.first;
        samplingEvent* event = packetKeyEventPair.second;

        outfile << key.GetSrcIp() << "," << key.GetSrcPort() << ",";
        outfile << key.GetDstIp() << "," << key.GetDstPort() << "," << key.GetSeqNb() << "," << key.GetId()  << "," << key.GetSize() << ",";
        outfile << GetRelativeTime(event->GetSampleTime()).GetNanoSeconds() << ",";
        outfile << event->IsDeparted() << "," << GetRelativeTime(event->GetDepartureTime()).GetNanoSeconds() << ",";
        outfile << std::fixed << std::setprecision(10);
        outfile << event->GetMarkingProb() << endl;
    }
    outfile.close();
}

string PoissonSampler::GetSampleTag() const { return _sampleTag; }
ns3::Time PoissonSampler::GetRelativeTime(const Time &time){ return time - _startTime; }
