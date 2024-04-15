//
// Created by nal on 27.04.21.
//

#include "TraceReplayReceiver.h"

NS_LOG_COMPONENT_DEFINE ("TraceReplayReceiver");

NS_OBJECT_ENSURE_REGISTERED (TraceReplayReceiver);

TypeId TraceReplayReceiver::GetTypeId() {

    static TypeId tid = TypeId ("ns3::TraceReplayReceiver")
            .SetParent<Application> ()
            .SetGroupName("Applications")
            .AddConstructor<TraceReplayReceiver> ()
            .AddAttribute ("ServerAddress",
                           "The Address of the created server socket",
                           AddressValue (),
                           MakeAddressAccessor (&TraceReplayReceiver::_serverAddress),
                           MakeAddressChecker ())
            .AddAttribute ("Protocol", "the name of the protocol to use to send traffic by the applications",
                           StringValue ("ns3::UdpSocketFactory"),
                           MakeStringAccessor (&TraceReplayReceiver::_protocol),
                           MakeStringChecker())
    ;
    return tid;
}

TraceReplayReceiver::TraceReplayReceiver() {
    NS_LOG_FUNCTION (this);

    _socket = nullptr;
}

TraceReplayReceiver::~TraceReplayReceiver() {
    NS_LOG_FUNCTION (this);
}

void TraceReplayReceiver::DoDispose() {
    NS_LOG_FUNCTION (this);

    Application::DoDispose();
}

void TraceReplayReceiver::StartApplication() {
    NS_LOG_FUNCTION(this);

    if (!_socket)     {
        TypeId tid = TypeId::LookupByName (_protocol);
        _socket = Socket::CreateSocket (GetNode (), tid);
        if (_socket->Bind(_serverAddress) == -1) {
            NS_FATAL_ERROR ("Failed to bind socket");
        }
        _socket->Listen ();
        _socket->ShutdownSend ();

        if(tid == UdpSocketFactory::GetTypeId ()) {
            _socket->SetRecvCallback (MakeCallback (&TraceReplayReceiver::Recv, this));
        }
        else if(tid == TcpSocketFactory::GetTypeId ()) {
            _socket->SetAcceptCallback( MakeNullCallback<bool, Ptr<Socket>, const Address &>(),
                                         MakeCallback(&TraceReplayReceiver::HandleTCPAccept, this));
        }

    }
}

void TraceReplayReceiver::StopApplication() {
    NS_LOG_FUNCTION (this);

    _socket->Dispose();
}

void TraceReplayReceiver::HandleTCPAccept (Ptr<Socket> socket, const Address& from) {
    NS_LOG_FUNCTION(this);

    _socket = socket;
    _socket->SetRecvCallback (MakeCallback (&TraceReplayReceiver::Recv, this));
}

void TraceReplayReceiver::Recv(Ptr<Socket> socket) {
    NS_LOG_FUNCTION(this);

    Ptr<Packet> packet = socket->Recv();
    if(packet->GetSize() == 0) { return; }
}

