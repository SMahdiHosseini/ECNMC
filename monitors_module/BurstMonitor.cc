//
// Created by Mahdi Hosseini on 16.10.2024
//
#include "BurstMonitor.h"
#include <iomanip>

BurstSamplingEvent::BurstSamplingEvent(Time sampleTime, bool isHotThroughputUtilization, uint32_t queueSize) {
    _sampleTime = sampleTime;
    _isHotThroughputUtilization = isHotThroughputUtilization;
    this->queueSize = queueSize;
}
void BurstSamplingEvent::SetSampleTime() { _sampleTime = ns3::Simulator::Now(); }
void BurstSamplingEvent::SetSampleTime(Time t) { _sampleTime = t; }
Time BurstSamplingEvent::GetSampleTime() const { return _sampleTime; }
uint32_t BurstSamplingEvent::GetQueueSize() const { return queueSize; }
bool BurstSamplingEvent::IsHotThroughputUtilization () const { return _isHotThroughputUtilization; }
void BurstSamplingEvent::SetHotThroughputUtilization(bool isHotThroughputUtilization) { _isHotThroughputUtilization = isHotThroughputUtilization; }

BurstMonitor::BurstMonitor(Time steadyStopTime, Ptr<PointToPointNetDevice> _outgoingNetDevice, Ptr<RedQueueDisc> _REDQueueDisc, const string &sampleTag, Time sampleInterval, const DataRate &_linkRate) {
    _sampleTag = sampleTag;
    this->sampleInterval = sampleInterval;
    byteCount = 0;
    lastByteCount = 0;
    linkRate = _linkRate;
    _recordedSamples = std::vector<BurstSamplingEvent>();
    REDQueueDisc = _REDQueueDisc;
    outgoingNetDevice = _outgoingNetDevice;
    Simulator::Schedule(Seconds(0), &BurstMonitor::Connect, this, outgoingNetDevice);
    Simulator::Schedule(steadyStopTime, &BurstMonitor::Disconnect, this, outgoingNetDevice);
}

void BurstMonitor::Connect(Ptr<PointToPointNetDevice> outgoingNetDevice) {
    outgoingNetDevice->TraceConnectWithoutContext("PromiscSniffer", MakeCallback(&BurstMonitor::RecordPacket, this));
    outgoingNetDevice->GetQueue()->TraceConnectWithoutContext("Enqueue", MakeCallback(&BurstMonitor::EnqueueNetDeviceQueue, this));
    // generate the first event
    Simulator::Schedule(sampleInterval, &BurstMonitor::EventHandler, this);
}

void BurstMonitor::Disconnect(Ptr<PointToPointNetDevice> outgoingNetDevice) {
    outgoingNetDevice->TraceDisconnectWithoutContext("PromiscSniffer", MakeCallback(&BurstMonitor::RecordPacket, this));
}

void BurstMonitor::EnqueueNetDeviceQueue(Ptr<const Packet> packet) {
    PacketKey* packetKey = PacketKey::Packet2PacketKey(packet, FIRST_HEADER_PPP);
    _recordedPacketKeys[*packetKey] = true;
}

void BurstMonitor::EventHandler() {
    // Generate a new event
    Simulator::Schedule(sampleInterval, &BurstMonitor::EventHandler, this);
   
    Time sampleTime = Simulator::Now();
    // uint64_t utilization = (byteCount - lastByteCount) * 8 / sampleInterval.GetSeconds();
    // // if (_sampleTag == "T0A0") {
    // //     std::cout << "Time: " << sampleTime.GetSeconds() << "s, Utilization: " << utilization / 1000000 << " Mbps, byteCount: " << byteCount << std::endl;
    // // }
    // if (utilization > linkRate.GetBitRate() / 2) {
    //     _recordedSamples.push_back(BurstSamplingEvent(sampleTime, true));
    // }
    // else {
    //     _recordedSamples.push_back(BurstSamplingEvent(sampleTime, false));
    // }
    // lastByteCount = byteCount;
    uint32_t queueSize = REDQueueDisc->GetStats().GetTotalDroppedBytes();
    _recordedSamples.push_back(BurstSamplingEvent(sampleTime, 0, queueSize));
}


void BurstMonitor::RecordPacket(Ptr<const Packet> packet) {
    PacketKey* packetKey = PacketKey::Packet2PacketKey(packet, FIRST_HEADER_PPP);
    if (_recordedPacketKeys.find(*packetKey) != _recordedPacketKeys.end()) {
        byteCount += packetKey->GetSize();
    }
}


void BurstMonitor::SaveRecords(const string& filename) {
    ofstream outfile;
    outfile.open(filename);
    outfile << "sampleTime, isHotThroughputUtilization,queueSize" << endl;
    for (auto &event : _recordedSamples) {
        outfile << event.GetSampleTime() << ", " << event.IsHotThroughputUtilization() << "," << event.GetQueueSize() << endl;
    }
    outfile.close();
}

string BurstMonitor::GetSampleTag() const {
    return _sampleTag;
}