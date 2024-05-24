//
// Created by Mahdi Hosseini on 23.05.24.
//
#include "RegularSampler.h"


RegularSampler::RegularSampler(const Time &startTime, const Time &duration, Ptr<RedQueueDisc> queueDisc, Ptr<Queue<Packet>> queue, Ptr<PointToPointNetDevice>  outgoingNetDevice, const string &sampleTag, const Time &samplePeriod) {
    _startTime = startTime;
    _duration = duration;
    _sampleTag = sampleTag;
    REDQueueDisc = queueDisc;
    NetDeviceQueue = queue;
    _samplePeriod = samplePeriod;
    zeroDelayPort = 0;
    droppedPackets = 0;

    Simulator::Schedule(_startTime, &RegularSampler::Connect, this, outgoingNetDevice);
    Simulator::Schedule(_startTime + _duration, &RegularSampler::Disconnect, this, outgoingNetDevice);
}

void RegularSampler::EnqueueQueueDisc(Ptr<const QueueDiscItem> item) {
    lastItem = item;
}

void RegularSampler::EnqueueNetDeviceQueue(Ptr<const Packet> packet) {
    lastPacket = packet;
}

void RegularSampler::Connect(Ptr<PointToPointNetDevice> outgoingNetDevice) {
    REDQueueDisc->TraceConnectWithoutContext("Enqueue", MakeCallback(&RegularSampler::EnqueueQueueDisc, this));
    NetDeviceQueue->TraceConnectWithoutContext("Enqueue", MakeCallback(&RegularSampler::EnqueueNetDeviceQueue, this));
    outgoingNetDevice->TraceConnectWithoutContext("PromiscSniffer", MakeCallback(&RegularSampler::RecordPacket, this));
    // generate the first event
    Simulator::Schedule(_samplePeriod, &RegularSampler::EventHandler, this);
}

void RegularSampler::Disconnect(Ptr<PointToPointNetDevice> outgoingNetDevice) {
    outgoingNetDevice->TraceDisconnectWithoutContext("PromiscSniffer", MakeCallback(&RegularSampler::RecordPacket, this));
    REDQueueDisc->TraceDisconnectWithoutContext("Enqueue", MakeCallback(&RegularSampler::EnqueueQueueDisc, this));
    NetDeviceQueue->TraceDisconnectWithoutContext("Enqueue", MakeCallback(&RegularSampler::EnqueueNetDeviceQueue, this));
}

void RegularSampler::EventHandler() {
    QueueDisc::Stats st = REDQueueDisc->GetStats();
    droppedPackets = st.GetNDroppedPackets(RedQueueDisc::UNFORCED_DROP) + st.GetNDroppedPackets(RedQueueDisc::FORCED_DROP) + st.GetNDroppedPackets(QueueDisc::INTERNAL_QUEUE_DROP);
    PacketKey* packetKey;
    bool zeroDelay = false;
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
    }
    else if (NetDeviceQueue->GetNPackets() > 0) {
        packetKey = PacketKey::Packet2PacketKey(lastPacket, FIRST_HEADER_PPP);
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
    // Generate a new event
    Simulator::Schedule(_samplePeriod, &RegularSampler::EventHandler, this);
}

void RegularSampler::RecordPacket(Ptr<const Packet> packet) {
    PacketKey* packetKey = PacketKey::Packet2PacketKey(packet, FIRST_HEADER_PPP);
    if (_recordedSamples.find(*packetKey) != _recordedSamples.end()) {
        _recordedSamples[*packetKey]->SetDepartureTime();
        // check if there exists a packet with the same key but different record field
        packetKey->SetRecords(packetKey->GetRecords() + 1);
        while (_recordedSamples.find(*packetKey) != _recordedSamples.end())
        {
            _recordedSamples[*packetKey]->SetDepartureTime();
            packetKey->SetRecords(packetKey->GetRecords() + 1);
        }
    }
}


void RegularSampler::SaveSamples(const string& filename) {
    ofstream outfile;
    outfile.open(filename);
    outfile << "SourceIp,SourcePort,DestinationIp,DestinationPort,SequenceNb,PayloadSize,SampleTime,IsDeparted,DepartTime" << endl;
    for (auto& packetKeyEventPair: _recordedSamples) {
        PacketKey key = packetKeyEventPair.first;
        samplingEvent* event = packetKeyEventPair.second;

        outfile << key.GetSrcIp() << "," << key.GetSrcPort() << ",";
        outfile << key.GetDstIp() << "," << key.GetDstPort() << "," << key.GetSeqNb() << "," << key.GetSize() << ",";
        outfile << GetRelativeTime(event->GetSampleTime()).GetNanoSeconds() << ",";
        outfile << event->IsDeparted() << "," << GetRelativeTime(event->GetDepartureTime()).GetNanoSeconds() << endl;
    }
    outfile << "DroppedPackets: " << droppedPackets << endl;
    outfile.close();
}

string RegularSampler::GetSampleTag() const { return _sampleTag; }
ns3::Time RegularSampler::GetRelativeTime(const Time &time){ return time - _startTime; }
