//
// Created by Mahdi Hosseini on 24.04.24.
//
#include "PoissonSampler.h"
#include <iomanip>

samplingEvent::samplingEvent(PacketKey *key) { SetPacketKey(key); }
samplingEvent::samplingEvent() { _key = nullptr; }
void samplingEvent::SetSampleTime() { SetSent(ns3::Simulator::Now()); }
void samplingEvent::SetSampleTime(Time t) { SetSent(t); }
void samplingEvent::SetDepartureTime(Time t) { SetReceived(t); }
void samplingEvent::SetDepartureTime() { SetReceived(ns3::Simulator::Now()); }
void samplingEvent::SetMarkingProb(double markingProb) { _markingProb = markingProb; }
void samplingEvent::SetLossProb(double lossProb) { _lossProb = lossProb; }
void samplingEvent::SetQueueSize(uint32_t size) { _queueSize = size; }
void samplingEvent::SetTotalQueueSize(uint32_t size) { _totalQueueSize = size; }
void samplingEvent::SetLastMarkingProb(double markingProb) { _lastMarkingProb = markingProb; }
void samplingEvent::SetLabel(const string label) { _label = label; }
void samplingEvent::SetEventAction(const string action) { _eventAction = action; }
void samplingEvent::SetLastDropProb(double lossProb) { _lastLossProb = lossProb; }
void samplingEvent::SetLastQueueSize(uint32_t size) { _lastQueueSize = size; }
void samplingEvent::SetLastTotalQueueSize(uint32_t size) { _lastTotalQueueSize = size; }
PacketKey *samplingEvent::GetPacketKey() const { return _key; }
Time samplingEvent::GetSampleTime() const { return GetSentTime(); }
Time samplingEvent::GetDepartureTime() const { return GetReceivedTime(); }
double samplingEvent::GetMarkingProb() const { return _markingProb; }
double samplingEvent::GetLossProb() const { return _lossProb; }
bool samplingEvent::IsDeparted() const { return GetReceivedTime() != Time(-1); }
uint32_t samplingEvent::GetQueueSize() const { return _queueSize; }
uint32_t samplingEvent::GetTotalQueueSize() const { return _totalQueueSize; }
double samplingEvent::GetLastMarkingProb() const { return _lastMarkingProb; }
string samplingEvent::GetLabel() const { return _label; }
string samplingEvent::GetEventAction() const { return _eventAction; }
double samplingEvent::GetLastDropProb() const { return _lastLossProb; }
uint32_t samplingEvent::GetLastQueueSize() const { return _lastQueueSize; }
uint32_t samplingEvent::GetLastTotalQueueSize() const { return _lastTotalQueueSize; }
PoissonSampler::PoissonSampler(const Time &steadyStartTime, const Time &steadyStopTime, Ptr<RedQueueDisc> queueDisc, Ptr<Queue<Packet>> queue, Ptr<PointToPointNetDevice>  outgoingNetDevice, const string &sampleTag, double sampleRate) 
: PoissonSampler(steadyStartTime, steadyStopTime, queueDisc, queue, outgoingNetDevice, sampleTag, sampleRate, nullptr, nullptr) {

}

PoissonSampler::PoissonSampler(const Time &steadyStartTime, const Time &steadyStopTime, Ptr<RedQueueDisc> queueDisc, Ptr<Queue<Packet>> queue, Ptr<PointToPointNetDevice>  outgoingNetDevice, const string &sampleTag, double sampleRate, Ptr<PointToPointNetDevice> _incomingNetDevice, Ptr<PointToPointNetDevice> _incomingNetDevice_1)
: Monitor(Seconds(0), steadyStopTime, steadyStartTime, steadyStopTime, sampleTag) {
    REDQueueDisc = queueDisc;
    NetDeviceQueue = queue;
    m_var = CreateObject<ExponentialRandomVariable>();
    m_var->SetAttribute("Mean", DoubleValue(1/sampleRate));
    _sampleRate = sampleRate;
    zeroDelayPort = 0;
    samplesMarkingProbMean = 0.0;
    samplesMarkingProbVariance = 0.0;
    samplesLossProbMean = 0.0;
    samplesLossProbVariance = 0.0;
    sampleMean.push_back(0);
    unbiasedSmapleVariance.push_back(0.0);
    sampleSize.push_back(0);
    // packetCDF.loadCDFData("/media/experiments/ns-allinone-3.41/ns-3.41/scratch/ECNMC/Helpers/packet_size_cdf.csv");
    packetCDF.loadCDFData("/media/experiments/ns-allinone-3.41/ns-3.41/scratch/ECNMC/Helpers/packet_size_cdf_singleQueue.csv");
    numOfGTSamples = 0;
    GTPacketSizeMean = 0;
    GTDropMean = 0;
    GTQueuingDelay = 0;
    GTMarkingProbMean = 0;
    firstItemTime = Time(-1);
    lastItemTime = Time(0);
    outgoingDataRate = outgoingNetDevice->GetDataRate();
    incomingNetDevice = _incomingNetDevice;
    incomingNetDevice_1 = _incomingNetDevice_1;
    Simulator::Schedule(Seconds(0), &PoissonSampler::Connect, this, outgoingNetDevice);
    Simulator::Schedule(steadyStopTime, &PoissonSampler::Disconnect, this, outgoingNetDevice);
}

uint32_t PoissonSampler::ComputeQueueSize() {
    uint32_t TXedBytes = (outgoingDataRate * (Simulator::Now() - lastLeftTime)) / 8;
    uint32_t remainedBytes = (lastLeftSize > TXedBytes) ? lastLeftSize - TXedBytes : 0;
    if (REDQueueDisc != nullptr) {
        return REDQueueDisc->GetNBytes() + NetDeviceQueue->GetNBytes() + remainedBytes;
    }
    return NetDeviceQueue->GetNBytes() + remainedBytes;
}

void PoissonSampler::RecordIncomingPacket(Ptr<const Packet> packet) {
    // uncomment the fllowing lines to calculate the Ground Truth Mean Drop probability
    if (Simulator::Now() < _steadyStartTime || Simulator::Now() > _steadyStopTime) {
        return;
    }
    const Ptr<Packet> &pktCopy = packet->Copy();
    PppHeader pppHeader;
    pktCopy->RemoveHeader(pppHeader);
    Ipv4Header ipHeader;
    pktCopy->RemoveHeader(ipHeader);
    // if (ipHeader.GetSource() != Ipv4Address("10.1.1.1")) {
    //     return;
    // }
    if (ipHeader.GetSource() == Ipv4Address("10.3.1.1")) {
        return;
    }
    TcpHeader tcpHeader;
    pktCopy->PeekHeader(tcpHeader);
    // Time prev = lastPacketTime;
    // lastPacketTime = Simulator::Now();
    // if (firstItemTime == Time(-1)) {
    //     firstItemTime = lastPacketTime;
    //     return;
    // }
    // Time queuingDelay = outgoingDataRate.CalculateBytesTxTime(ComputeQueueSize());
    Time queuingDelay = outgoingDataRate.CalculateBytesTxTime(REDQueueDisc->GetNBytes() + NetDeviceQueue->GetNBytes());
    double dropProbDynamicCDF = packetCDF.calculateProbabilityGreaterThan(REDQueueDisc->GetMaxSize().GetValue() - REDQueueDisc->GetNBytes());
    double markingProbDynamic = REDQueueDisc->GetMarkingProbability();

    samplingEvent event = samplingEvent();
    event.SetSampleTime(Simulator::Now());
    event.SetDepartureTime(Simulator::Now() + queuingDelay);
    event.SetLossProb(dropProbDynamicCDF);
    event.SetMarkingProb(markingProbDynamic);
    event.SetQueueSize(REDQueueDisc->GetNBytes());
    event.SetTotalQueueSize(ComputeQueueSize());
    event.SetLastMarkingProb(REDQueueDisc->_lastMarkingProb);
    std::ostringstream oss;
    ipHeader.GetSource().Print(oss);
    std::string headerString = oss.str();
    string label = headerString + ":" + to_string(tcpHeader.GetSourcePort());
    event.SetLabel(label);
    queueSizeProcessByPackets.push_back(std::make_tuple(Simulator::Now(), event));

    _lastDropProb = dropProbDynamicCDF;
    _lastQueueSize = REDQueueDisc->GetNBytes();
    _lastTotalQueueSize = ComputeQueueSize();
    // updateGTCounters();
    // cout << "### POISSON ### Enqueue Time: " << Simulator::Now().GetNanoSeconds() << " Queueing Delay: " << queuingDelay.GetNanoSeconds() << " REDQueue Size: " << REDQueueDisc->GetNBytes() << " GTQueuingDelay: " << GTQueuingDelay << " ECN >>> " << markingProbDynamic << endl;
}

void PoissonSampler::updateGTCounters() {
    Time prev = lastPacketTime;
    lastPacketTime = Simulator::Now();
    if (firstItemTime == Time(-1)) {
        firstItemTime = lastPacketTime;
        return;
    }
    if (lastPacketTime == firstItemTime){
        return;
    }
    // Time queuingDelay = outgoingDataRate.CalculateBytesTxTime(ComputeQueueSize());
    Time queuingDelay = outgoingDataRate.CalculateBytesTxTime(REDQueueDisc->GetNBytes() + NetDeviceQueue->GetNBytes());
    GTQueuingDelay = ((GTQueuingDelay * (prev - firstItemTime).GetNanoSeconds()) + (queuingDelay.GetNanoSeconds() * (lastPacketTime - prev).GetNanoSeconds())) / (lastPacketTime - firstItemTime).GetNanoSeconds();

    double dropProbDynamicCDF = 0;
    dropProbDynamicCDF = packetCDF.calculateProbabilityGreaterThan(REDQueueDisc->GetMaxSize().GetValue() - REDQueueDisc->GetNBytes());
    GTDropMean = (GTDropMean * (prev - firstItemTime).GetNanoSeconds() + dropProbDynamicCDF * (lastPacketTime - prev).GetNanoSeconds()) / (lastPacketTime - firstItemTime).GetNanoSeconds();

    double markingProbDynamic = 0;
    markingProbDynamic = REDQueueDisc->GetMarkingProbability();
    GTMarkingProbMean = (GTMarkingProbMean * (prev - firstItemTime).GetNanoSeconds() + markingProbDynamic * (lastPacketTime - prev).GetNanoSeconds()) / (lastPacketTime - firstItemTime).GetNanoSeconds();
}

void PoissonSampler::EnqueueQueueDisc(Ptr<const QueueDiscItem> item) {
    lastItem = item;
    // packetCDF.addPacket(item->GetSize());
    lastItemTime = Simulator::Now();
    if (Simulator::Now() < _steadyStartTime || Simulator::Now() > _steadyStopTime) {
        return;
    }
    // Time queuingDelay = outgoingDataRate.CalculateBytesTxTime(ComputeQueueSize());
    Time queuingDelay = outgoingDataRate.CalculateBytesTxTime(REDQueueDisc->GetNBytes() + NetDeviceQueue->GetNBytes());
    double dropProbDynamicCDF = packetCDF.calculateProbabilityGreaterThan(REDQueueDisc->GetMaxSize().GetValue() - REDQueueDisc->GetNBytes());
    double markingProbDynamic = REDQueueDisc->GetMarkingProbability();

    samplingEvent event = samplingEvent();
    event.SetSampleTime(Simulator::Now());
    event.SetDepartureTime(Simulator::Now() + queuingDelay);
    event.SetLossProb(dropProbDynamicCDF);
    event.SetMarkingProb(markingProbDynamic);
    event.SetQueueSize(REDQueueDisc->GetNBytes());
    event.SetTotalQueueSize(ComputeQueueSize());
    event.SetLastMarkingProb(REDQueueDisc->_lastMarkingProb);
    TcpHeader tcpHeader;
    item->GetPacket()->PeekHeader(tcpHeader);
    std::ostringstream oss;
    DynamicCast<const Ipv4QueueDiscItem>(item)->GetHeader().GetSource().Print(oss);
    std::string headerString = oss.str();
    string label = headerString + ":" + to_string(tcpHeader.GetSourcePort()); 
    event.SetLabel(label);
    event.SetEventAction("E");
    queueSizeProcess.push_back(std::make_tuple(Simulator::Now(), event));
    // cout << "### POISSON ### Time: " << Simulator::Now().GetNanoSeconds() << " *** Enqueue *** " << " Queue Size: " << REDQueueDisc->GetNBytes() + NetDeviceQueue->GetNBytes() << " Total Queue Size: " << ComputeQueueSize() << " Queuing Delay: " << queuingDelay.GetNanoSeconds() << " packet size: " << item->GetSize() << endl;
    updateGTCounters();
}

void PoissonSampler::DequeueQueueDisc(Ptr<const QueueDiscItem> item) {
    if (Simulator::Now() < _steadyStartTime || Simulator::Now() > _steadyStopTime) {
        return;
    }
    // Time queuingDelay = outgoingDataRate.CalculateBytesTxTime(ComputeQueueSize());
    uint32_t queueSize;
    if (ComputeQueueSize() == 0) {
        queueSize = 0;
    }
    else {
        queueSize = REDQueueDisc->GetNBytes() + NetDeviceQueue->GetNBytes() + item->GetSize() + 2;
    }
    Time queuingDelay = outgoingDataRate.CalculateBytesTxTime(queueSize);
    double dropProbDynamicCDF = packetCDF.calculateProbabilityGreaterThan(REDQueueDisc->GetMaxSize().GetValue() - REDQueueDisc->GetNBytes());
    double markingProbDynamic = REDQueueDisc->GetMarkingProbability();

    samplingEvent event = samplingEvent();
    event.SetSampleTime(Simulator::Now());
    event.SetDepartureTime(Simulator::Now() + queuingDelay);
    event.SetLossProb(dropProbDynamicCDF);
    event.SetMarkingProb(markingProbDynamic);
    event.SetQueueSize(REDQueueDisc->GetNBytes());
    event.SetTotalQueueSize(ComputeQueueSize() + item->GetSize() + 2);
    event.SetLastMarkingProb(REDQueueDisc->_lastMarkingProb);
    TcpHeader tcpHeader;
    item->GetPacket()->PeekHeader(tcpHeader);
    std::ostringstream oss;
    DynamicCast<const Ipv4QueueDiscItem>(item)->GetHeader().GetSource().Print(oss);
    std::string headerString = oss.str();
    string label = headerString + ":" + to_string(tcpHeader.GetSourcePort()); 
    event.SetLabel(label);
    event.SetEventAction("D");
    queueSizeProcess.push_back(std::make_tuple(Simulator::Now(), event));
    // cout << "### POISSON ### Time: " << Simulator::Now().GetNanoSeconds() << " *** Dequeue *** " << " Queue Size: " << queueSize << " Total Queue Size: " << ComputeQueueSize() + item->GetSize() + 2 << " Queuing Delay: " << queuingDelay.GetNanoSeconds() << " packet size: " << item->GetSize() << endl;
    updateGTCounters();
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
    // // Time queuingDelay = outgoingDataRate.CalculateBytesTxTime(REDQueueDisc->GetNBytes() + NetDeviceQueue->GetNBytes());
    // GTQueuingDelay = ((GTQueuingDelay * (prev - firstItemTime).GetNanoSeconds()) + (queuingDelay.GetNanoSeconds() * (lastPacketTime - prev).GetNanoSeconds())) / (lastPacketTime - firstItemTime).GetNanoSeconds();
    // cout << "### POISSON ### Enqueue Time: " << Simulator::Now().GetNanoSeconds() << " Queueing Delay: " << queuingDelay.GetNanoSeconds() << " Queue Size: " << ComputeQueueSize() - packet->GetSize() << " GTQueuingDelay: " << GTQueuingDelay << endl;

    // double dropProbDynamicCDF = 0;
    // // if (QueueSize("100KB").GetValue() <= ComputeQueueSize()){
    // //     dropProbDynamicCDF = 1.0;
    // // }
    // // else {
        // // dropProbDynamicCDF = packetCDF.calculateProbabilityGreaterThan(NetDeviceQueue->GetMaxSize().GetValue() - NetDeviceQueue->GetCurrentSize().GetValue());
        // dropProbDynamicCDF = packetCDF.calculateProbabilityGreaterThan(REDQueueDisc->GetMaxSize().GetValue() - REDQueueDisc->GetNBytes());
    // // }
    // GTDropMean = (GTDropMean * (prev - firstItemTime).GetNanoSeconds() + dropProbDynamicCDF * (lastPacketTime - prev).GetNanoSeconds()) / (Simulator::Now() - firstItemTime).GetNanoSeconds();
    // GTDropMean = (GTDropMean * (prev - firstItemTime).GetNanoSeconds() + ((dropProbDynamicCDF + lastProb) / 2.0) * (lastPacketTime - prev).GetNanoSeconds()) / (Simulator::Now() - firstItemTime).GetNanoSeconds();
    // cout << "### POISSON ### Enqueue Time: " << Simulator::Now().GetNanoSeconds() << " Queue Size: " << NetDeviceQueue->GetCurrentSize().GetValue() << " Drop Prob: " << dropProbDynamicCDF << " GTDropMean: " << GTDropMean << endl;
    // cout << "Vars: " << " firstItemTime: " << firstItemTime.GetNanoSeconds() << " lastItemTime: " << lastPacketTime.GetNanoSeconds() << " prev: " << prev.GetNanoSeconds() << endl;
}

void PoissonSampler::Connect(Ptr<PointToPointNetDevice> outgoingNetDevice) {
    if (REDQueueDisc != nullptr) {
        REDQueueDisc->TraceConnectWithoutContext("Enqueue", MakeCallback(&PoissonSampler::EnqueueQueueDisc, this));
        REDQueueDisc->TraceConnectWithoutContext("Dequeue", MakeCallback(&PoissonSampler::DequeueQueueDisc, this));
    }
    NetDeviceQueue->TraceConnectWithoutContext("Enqueue", MakeCallback(&PoissonSampler::EnqueueNetDeviceQueue, this));
    incomingNetDevice->TraceConnectWithoutContext("PromiscSniffer", MakeCallback(&PoissonSampler::RecordIncomingPacket, this));
    incomingNetDevice_1->TraceConnectWithoutContext("PromiscSniffer", MakeCallback(&PoissonSampler::RecordIncomingPacket, this));
    outgoingNetDevice->TraceConnectWithoutContext("PromiscSniffer", MakeCallback(&PoissonSampler::RecordPacket, this));
    // generate the first event
    double nextEvent = m_var->GetValue();
    Simulator::Schedule(Seconds(nextEvent), &PoissonSampler::EventHandler, this);
}

void PoissonSampler::Disconnect(Ptr<PointToPointNetDevice> outgoingNetDevice) {
    outgoingNetDevice->TraceDisconnectWithoutContext("PromiscSniffer", MakeCallback(&PoissonSampler::RecordPacket, this));
    incomingNetDevice->TraceDisconnectWithoutContext("PromiscSniffer", MakeCallback(&PoissonSampler::RecordIncomingPacket, this));
    incomingNetDevice_1->TraceDisconnectWithoutContext("PromiscSniffer", MakeCallback(&PoissonSampler::RecordIncomingPacket, this));
    if (REDQueueDisc != nullptr) {
        REDQueueDisc->TraceDisconnectWithoutContext("Enqueue", MakeCallback(&PoissonSampler::EnqueueQueueDisc, this));
        REDQueueDisc->TraceDisconnectWithoutContext("Dequeue", MakeCallback(&PoissonSampler::DequeueQueueDisc, this));
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
    
    double dropProbDynamicCDF = 0.0;
    double markingProbDynamic = 0.0;
    uint32_t queueSize;
    if (REDQueueDisc != nullptr) {
        queueSize = REDQueueDisc->GetNBytes() + NetDeviceQueue->GetNBytes();
        // queueSize = ComputeQueueSize();
        dropProbDynamicCDF = packetCDF.calculateProbabilityGreaterThan(REDQueueDisc->GetMaxSize().GetValue() - REDQueueDisc->GetNBytes());
        markingProbDynamic = REDQueueDisc->GetMarkingProbability();
    }
    else {
        queueSize = NetDeviceQueue->GetNBytes();
        //TODO: the following line has to be fixed. It is not correct if the queue max size is in packets
        dropProbDynamicCDF = packetCDF.calculateProbabilityGreaterThan(NetDeviceQueue->GetMaxSize().GetValue() - NetDeviceQueue->GetCurrentSize().GetValue());
    }
    PacketKey* packetKey = new PacketKey(ns3::Ipv4Address("0.0.0.0"), ns3::Ipv4Address("0.0.0.1"), 0, zeroDelayPort++, zeroDelayPort++, ns3::SequenceNumber32(0), ns3::SequenceNumber32(0), 0, 0);
    Time queuingDelay = outgoingDataRate.CalculateBytesTxTime(queueSize);
    samplingEvent* event = new samplingEvent(packetKey);
    event->SetSampleTime(Simulator::Now());
    event->SetDepartureTime(Simulator::Now() + queuingDelay);
    event->SetLossProb(dropProbDynamicCDF);
    event->SetMarkingProb(markingProbDynamic);
    event->SetQueueSize(REDQueueDisc->GetNBytes());
    event->SetTotalQueueSize(ComputeQueueSize());
    event->SetLastMarkingProb(REDQueueDisc->_lastMarkingProb);
    event->SetLastDropProb(_lastDropProb);
    event->SetLastQueueSize(_lastQueueSize);
    event->SetLastTotalQueueSize(_lastTotalQueueSize);
    _recordedSamples[*packetKey] = event;
    // cout << "### EVENT ### " << "Time: " << Simulator::Now().GetNanoSeconds() << " Queue Size: " << queueSize << " Total Queue Size: " << ComputeQueueSize() << " Queuing Delay: " << queuingDelay.GetNanoSeconds() << endl;
    updateCounters(event);
}

void PoissonSampler::updateCounters(samplingEvent* event) {
    updateBasicCounters(event->GetSampleTime(), event->GetDepartureTime(), 0);
    
    double delta = (event->GetLossProb() - samplesLossProbMean);
    samplesLossProbMean = samplesLossProbMean + (delta / sampleSize[0]);
    if (sampleSize[0] <= 1) {
        samplesLossProbVariance = 0.0;
    }
    else {
        samplesLossProbVariance = samplesLossProbVariance + ((delta * delta) / (double) sampleSize[0]) - (samplesLossProbVariance / (double) (sampleSize[0] - 1));
    }

    delta = (event->GetMarkingProb() - samplesMarkingProbMean);
    samplesMarkingProbMean = samplesMarkingProbMean + (delta / sampleSize[0]);
    if (sampleSize[0] <= 1) {
        samplesMarkingProbVariance = 0.0;
    }
    else {
        samplesMarkingProbVariance = samplesMarkingProbVariance + ((delta * delta) / (double) sampleSize[0]) - (samplesMarkingProbVariance / (double) (sampleSize[0] - 1));
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
    outfile << "sampleDelayMean,unbiasedSmapleDelayVariance,sampleSize,samplesDropMean,samplesDropVariance,samplesMarkingProbMean,samplesMarkingProbVariance,GTSampleSize,GTPacketSizeMean,GTDropMean,GTQueuingDelay,GTMarkingProbMean" << endl;
    outfile << sampleMean[0] << "," << unbiasedSmapleVariance[0] << "," << sampleSize[0] << "," << samplesLossProbMean << "," << samplesLossProbVariance << "," << samplesMarkingProbMean << "," << samplesMarkingProbVariance
    << "," << numOfGTSamples << "," << GTPacketSizeMean << "," << GTDropMean << "," << GTQueuingDelay << "," << GTMarkingProbMean << endl;
    outfile.close();

    ofstream eventsFile;
    eventsFile.open(filename.substr(0, filename.size() - 4) + "_events.csv");

    eventsFile << "Time,QueuingDelay,DropProb,MarkingProb,QueueSize,TotalQueueSize,LastMarkingProb,LastDropProb,LastQueueSize,LastTotalQueueSize" << endl;
    for (auto &item : _recordedSamples) {
        eventsFile << item.second->GetSampleTime().GetNanoSeconds() << "," << (item.second->GetDepartureTime() - item.second->GetSampleTime()).GetNanoSeconds() << "," << item.second->GetLossProb() << "," << item.second->GetMarkingProb() << "," << item.second->GetQueueSize() << 
        "," << item.second->GetTotalQueueSize() << "," << item.second->GetLastMarkingProb() << "," << item.second->GetLastDropProb() << "," << item.second->GetLastQueueSize() << "," << item.second->GetLastTotalQueueSize() << endl;
    }
    eventsFile.close();

    ofstream queueSizeFile;
    queueSizeFile.open(filename.substr(0, filename.size() - 4) + "_queueSize.csv");
    queueSizeFile << "Time,QueuingDelay,DropProb,MarkingProb,QueueSize,TotalQueueSize,LastMarkingProb,Label,Action" << endl;
    for (auto &item : queueSizeProcess) {
        samplingEvent event = std::get<1>(item);
        queueSizeFile << std::get<0>(item).GetNanoSeconds() << "," << (event.GetDepartureTime() - event.GetSampleTime()).GetNanoSeconds() << "," << event.GetLossProb() << "," << event.GetMarkingProb() << "," << event.GetQueueSize() << 
        "," << event.GetTotalQueueSize() << "," << event.GetLastMarkingProb() << "," << event.GetLabel() << "," << event.GetEventAction() << endl;
    }
    queueSizeFile.close();

    ofstream queueSizeByPacketsFile;
    queueSizeByPacketsFile.open(filename.substr(0, filename.size() - 4) + "_queueSizeByPackets.csv");
    queueSizeByPacketsFile << "Time,QueuingDelay,DropProb,MarkingProb,QueueSize,TotalQueueSize,LastMarkingProb,Label,Action" << endl;
    for (auto &item : queueSizeProcessByPackets) {
        samplingEvent event = std::get<1>(item);
        queueSizeByPacketsFile << std::get<0>(item).GetNanoSeconds() << "," << (event.GetDepartureTime() - event.GetSampleTime()).GetNanoSeconds() << "," << event.GetLossProb() << "," << event.GetMarkingProb() << "," << event.GetQueueSize() << 
        "," << event.GetTotalQueueSize() << "," << event.GetLastMarkingProb() << "," << event.GetLabel() << "," << event.GetEventAction() << endl;
    }
    queueSizeByPacketsFile.close();

    // if (_monitorTag == "SD0") {
    //     packetCDF.printCDF();
    // }
}
