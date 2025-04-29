//
// Created by nal on 24.04.25
//
#include "NodeAppsHandler.h"

NS_LOG_COMPONENT_DEFINE ("NodeAppsHandler");

NS_OBJECT_ENSURE_REGISTERED (NodeAppsHandler);

TypeId PoissonReplaySender::GetTypeId() {

    static TypeId tid = TypeId ("ns3::NodeAppsHandler")
            .SetParent<Application> ()
            .SetGroupName("Applications")
            .AddConstructor<NodeAppsHandler> ()
            .AddAttribute("ConnectionSize",
                           "The number of connections to be created",
                           UintegerValue(1),
                           MakeUintegerAccessor(&NodeAppsHandler::_connectionSize),
                           MakeUintegerChecker<uint32_t>())
            .AddAttribute ("RemoteAddresses",
                           "The destination Address of othe nodes to send packets to",
                           vector<AddressValue()>(),
                           MakeAddressAccessor (&NodeAppsHandler::_receiverAddress),
                           MakeAddressChecker())
            .AddAttribute ("Protocol", "the name of the protocol to use to send traffic by the applications",
                           StringValue ("ns3::TcpSocketFactory"),
                           MakeStringAccessor (&NodeAppsHandler::_protocol),
                           MakeStringChecker())
            .AddAttribute("Rate", "The rate of the Poisson process (request per second)",
                          DoubleValue(5000.0),
                          MakeDoubleAccessor(&PoissonReplaySender::_rate),
                          MakeDoubleChecker<double>())
    ;
    return tid;
}

NodeAppsHandler::NodeAppsHandler() {
    NS_LOG_FUNCTION (this);
    m_var = CreateObject<ExponentialRandomVariable>();
    m_var->SetAttribute("Mean", DoubleValue(1 / _rate));
}

void NodeAppsHandler::DoDispose() {
    NS_LOG_FUNCTION (this);

    Application::DoDispose();
}

void NodeAppsHandler::StartApplication() {
    NS_LOG_FUNCTION(this);
    PrepareSockets();
}

void NodeAppsHandler::PrepareSockets() {
    NS_LOG_FUNCTION (this);

    for (auto &address : _receiverAddress) {
        NS_LOG_INFO("Receiver address: " << Ipv4Address::ConvertFrom(address));
        Ptr<Socket> socket = Socket::CreateSocket(GetNode(), TypeId::LookupByName(_protocol));
        if socket->Bind();
        socket->Connect(address);
        socket->SetAllowBroadcast (true);
        socket->SetRecvCallback (MakeNullCallback<void, Ptr<Socket> > ());
        _connectionPool[address].push_back(socket);
    }
    // if (!_socket)     {
    //     TypeId tid = TypeId::LookupByName (_protocol);
    //     _socket = Socket::CreateSocket (GetNode (), tid);
    //     if (_socket->Bind () == -1) {
    //         NS_FATAL_ERROR ("Failed to bind socket");
    //     }
    //     _socket->Connect(_receiverAddress);
    // }

    // // part to change starts from here
    // _socket->SetRecvCallback (MakeNullCallback<void, Ptr<Socket> > ());
    // _socket->SetAllowBroadcast (true);

    // // to enable/disable pacing for the measurement traffic
    // if(_protocol == "ns3::TcpSocketFactory") {
    //     Ptr<TcpSocketBase> tcpSocket = _socket->GetObject<TcpSocketBase>();
    //     tcpSocket->SetPacingStatus(0);
    // }
    // _sendEvent = Simulator::Schedule(Seconds(0), &PoissonReplaySender::ScheduleNextSend, this);
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

