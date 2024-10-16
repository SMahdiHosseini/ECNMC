//
// Created by nal on 31.08.20.
//
#include "TraceReplaySender.h"

NS_LOG_COMPONENT_DEFINE ("TraceReplaySender");

NS_OBJECT_ENSURE_REGISTERED (TraceReplaySender);

TypeId TraceReplaySender::GetTypeId() {

    static TypeId tid = TypeId ("ns3::TraceReplaySender")
            .SetParent<Application> ()
            .SetGroupName("Applications")
            .AddConstructor<TraceReplaySender> ()
            .AddAttribute ("RemoteAddress",
                           "The destination Address of the outbound packets",
                           AddressValue (),
                           MakeAddressAccessor (&TraceReplaySender::_receiverAddress),
                           MakeAddressChecker ())
            .AddAttribute ("Protocol", "the name of the protocol to use to send traffic by the applications",
                           StringValue ("ns3::UdpSocketFactory"),
                           MakeStringAccessor (&TraceReplaySender::_protocol),
                           MakeStringChecker())
            .AddAttribute ("TraceFile", "the name of the file to read the traces from",
                           StringValue (""),
                           MakeStringAccessor (&TraceReplaySender::_traceFilename),
                           MakeStringChecker())
            .AddAttribute ("EnablePacing", "enable pacing of tcp packets",
                           BooleanValue (true),
                           MakeBooleanAccessor(&TraceReplaySender::_enablePacing),
                           MakeBooleanChecker())
            .AddAttribute ("TrafficStartTime", "the time at which the traffic starts",
                           TimeValue (Seconds(0)),
                           MakeTimeAccessor (&TraceReplaySender::_trafficStartTime),
                           MakeTimeChecker())
            .AddAttribute ("TrafficEndTime", "the time at which the traffic ends",
                            TimeValue (Seconds(0)),
                            MakeTimeAccessor (&TraceReplaySender::_trafficEndTime),
                            MakeTimeChecker())
    ;
    return tid;
}

TraceReplaySender::TraceReplaySender() {
    NS_LOG_FUNCTION (this);

    _traceItemIdx = 0;
    _socket = nullptr;
    _sendEvent = EventId ();
    _enablePacing = true;
    _isSecondPhase = false;
}

TraceReplaySender::~TraceReplaySender() {
    NS_LOG_FUNCTION (this);
}


void TraceReplaySender::LoadTrace(const string& traceFile) {
    NS_LOG_FUNCTION(this);
    /* check if the trace file exists*/
    if (!std::filesystem::exists(traceFile)) {
        NS_LOG_ERROR("Trace file " << traceFile << " does not exist");
        return;
    }
    ifstream traceInput(traceFile);
    string line;
    Time previousTime = Seconds(0);
    while(getline(traceInput, line)) {
        vector<string> pkt_attributes = SplitStr(line, ',');

        uint32_t frameNb = stoi(pkt_attributes[0]);
        Time timestamp = Seconds(stod(pkt_attributes[1]));
        uint32_t payload_size = stoi(pkt_attributes[2]);
        // if (timestamp < _trafficStartTime || (timestamp > _trafficEndTime && _trafficEndTime != Seconds(0))) { continue; }
        // timestamp = timestamp - _trafficStartTime;
        // _traceItems.push_back({frameNb, timestamp, payload_size});
        _traceItems.push_back({frameNb, (timestamp - previousTime) / 30 , payload_size});
        previousTime = timestamp;
    }
    traceInput.close();
    // std::cout << "Loaded " << _traceItems.size() << " trace items from " << traceFile << std::endl;
    if (!_traceItems.empty()) {
        _startEvent = Simulator::Schedule(_traceItems.front().timestamp, &TraceReplaySender::PrepareSocket, this);
        Simulator::Schedule(_traceItems.back().timestamp + Seconds(1), &TraceReplaySender::StopApplication, this);
    }
}

void TraceReplaySender::DoDispose() {
    NS_LOG_FUNCTION (this);

    Application::DoDispose();
}

void TraceReplaySender::StartApplication() {
    NS_LOG_FUNCTION(this);

    LoadTrace(_traceFilename);
}

void TraceReplaySender::PrepareSocket() {
    NS_LOG_FUNCTION (this);

    if (!_socket)     {
        TypeId tid = TypeId::LookupByName (_protocol);
        _socket = Socket::CreateSocket (GetNode (), tid);
        if (_socket->Bind () == -1) {
            NS_FATAL_ERROR ("Failed to bind socket");
        }
        _socket->Connect(_receiverAddress);
    }

    // part to change starts from here
    _socket->SetRecvCallback (MakeNullCallback<void, Ptr<Socket> > ());
    _socket->SetAllowBroadcast (true);

    // to enable/disable pacing for the measurement traffic
    if(_protocol == "ns3::TcpSocketFactory") {
        Ptr<TcpSocketBase> tcpSocket = _socket->GetObject<TcpSocketBase>();
        tcpSocket->SetPacingStatus(_enablePacing);
    }
    // std::cout << "Socket prepared for " << _traceFilename << std::endl;
    _sendEvent = Simulator::Schedule(Seconds(0), &TraceReplaySender::ScheduleNextSend, this);
}

void TraceReplaySender::StopApplication() {
    NS_LOG_FUNCTION (this);

    _socket->Dispose();
    Simulator::Cancel (_sendEvent);
    Simulator::Cancel (_startEvent);
    DoDispose();
}

void TraceReplaySender::Send(const TraceReplayItem& item) {
    NS_LOG_FUNCTION(this);

    SeqTsHeader seqTs;
    seqTs.SetSeq (item.frameNb);
    uint32_t segmentSize = item.payloadSize > (8+4) ? item.payloadSize - (8+4) : item.payloadSize; // 8+4 : the size of the seqTs header
    Ptr<Packet> p = Create<Packet>(segmentSize);
    if(item.payloadSize > (8+4)) { p->AddHeader(seqTs); }

    if ((_socket->Send (p)) < 0) {
        NS_LOG_INFO ("Error while sending " << item.payloadSize << " bytes to "
                                            << Ipv4Address::ConvertFrom (_receiverAddress));
    }
}

void TraceReplaySender::ScheduleNextSend() {
    NS_LOG_FUNCTION(this);
    // if(_traceItemIdx == 0) {
    //     std::cout << "TraceReplaySender: Started sending packets from trace: " << _traceFilename << " at: " << Simulator::Now() << std::endl;
    // }
    if(_traceItemIdx >= _traceItems.size()) {
        // std::cout << "TraceReplaySender: Finished sending all packets trace: " << _traceFilename << " at: " << Simulator::Now() << std::endl;
        // return;
        _traceItemIdx = 0;
    }

    Send(_traceItems[_traceItemIdx]);
    _traceItemIdx++;
    if(_traceItemIdx >= _traceItems.size()) {
        // std::cout << "TraceReplaySender: Finished sending all packets trace: " << _traceFilename << " at: " << Simulator::Now() << std::endl;
        // return;
        _traceItemIdx = 0;
    }

    TraceReplayItem nextItem = _traceItems[_traceItemIdx];
    // Time currTime = Simulator::Now();
    // Time remainingTime = (nextItem.timestamp > currTime) ? nextItem.timestamp - currTime : Seconds(0);
    // remainingTime += MicroSeconds(helper_methods::GetRandomNumber(0, 100)); // to add randomness
    Time remainingTime = nextItem.timestamp + MicroSeconds(helper_methods::GetRandomNumber(0, 7)); // to add randomness
    _sendEvent = Simulator::Schedule(remainingTime, &TraceReplaySender::ScheduleNextSend, this);
}

