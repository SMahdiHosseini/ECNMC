//
// Created by nal on 25.03.25
//
#include "PoissonReplaySender.h"

NS_LOG_COMPONENT_DEFINE ("PoissonReplaySender");

NS_OBJECT_ENSURE_REGISTERED (PoissonReplaySender);

TypeId PoissonReplaySender::GetTypeId() {

    static TypeId tid = TypeId ("ns3::PoissonReplaySender")
            .SetParent<Application> ()
            .SetGroupName("Applications")
            .AddConstructor<PoissonReplaySender> ()
            .AddAttribute ("RemoteAddress",
                           "The destination Address of the outbound packets",
                           AddressValue (),
                           MakeAddressAccessor (&PoissonReplaySender::_receiverAddress),
                           MakeAddressChecker ())
            .AddAttribute ("Protocol", "the name of the protocol to use to send traffic by the applications",
                           StringValue ("ns3::UdpSocketFactory"),
                           MakeStringAccessor (&PoissonReplaySender::_protocol),
                           MakeStringChecker())
            .AddAttribute("Rate", "The rate of the Poisson process (samples per second)",
                          DoubleValue(5000.0),
                          MakeDoubleAccessor(&PoissonReplaySender::_rate),
                          MakeDoubleChecker<double>())
    ;
    return tid;
}

PoissonReplaySender::PoissonReplaySender() {
    NS_LOG_FUNCTION (this);
    _frameNb = 0;
    _socket = nullptr;
    m_var = CreateObject<ExponentialRandomVariable>();
    m_var->SetAttribute("Mean", DoubleValue(1 / _rate));
}

void PoissonReplaySender::DoDispose() {
    NS_LOG_FUNCTION (this);

    Application::DoDispose();
}

void PoissonReplaySender::StartApplication() {
    NS_LOG_FUNCTION(this);
    PrepareSocket();
}

void PoissonReplaySender::PrepareSocket() {
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
        tcpSocket->SetPacingStatus(0);
    }
    _sendEvent = Simulator::Schedule(Seconds(0), &PoissonReplaySender::ScheduleNextSend, this);
}

void PoissonReplaySender::StopApplication() {
    NS_LOG_FUNCTION (this);

    _socket->Dispose();
    Simulator::Cancel (_sendEvent);
    DoDispose();
}

void PoissonReplaySender::Send() {
    NS_LOG_FUNCTION(this);

    SeqTsHeader seqTs;
    seqTs.SetSeq (_frameNb++);
    uint32_t segmentSize = 1448;
    Ptr<Packet> p = Create<Packet>(segmentSize);
    p->AddHeader(seqTs);

    if ((_socket->Send (p)) < 0) {
        NS_LOG_INFO ("Error while sending " << segmentSize << " bytes to "
                                            << Ipv4Address::ConvertFrom (_receiverAddress));
    }
}

void PoissonReplaySender::ScheduleNextSend() {
    Send();
    double nextEvent = m_var->GetValue();
    _sendEvent = Simulator::Schedule(Seconds(nextEvent), &PoissonReplaySender::ScheduleNextSend, this);
}

