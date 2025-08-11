//
// Created by nal on 30.06.25
//
#include "ProbeApp.h"

NS_LOG_COMPONENT_DEFINE ("ProbeApp");

NS_OBJECT_ENSURE_REGISTERED (ProbeApp);

TypeId ProbeApp::GetTypeId() {

    static TypeId tid = TypeId ("ns3::ProbeApp")
            .SetParent<Application> ()
            .SetGroupName("Applications")
            .AddConstructor<ProbeApp> ()
            .AddAttribute ("Protocol", "the name of the protocol to use to send traffic by the applications",
                           StringValue ("ns3::TcpSocketFactory"),
                           MakeStringAccessor (&ProbeApp::_protocol),
                           MakeStringChecker())
            .AddAttribute("Rate", "The rate of the Poisson process (request per second)",
                          DoubleValue(5000.0),
                          MakeDoubleAccessor(&ProbeApp::_rate),
                          MakeDoubleChecker<double>())
    ;
    return tid;
}

ProbeApp::ProbeApp() {
    NS_LOG_FUNCTION (this);
    m_var = CreateObject<ExponentialRandomVariable>();
}

ProbeApp::~ProbeApp() {
    NS_LOG_FUNCTION (this);
}

void ProbeApp::SetReceiverAddress(Address receiverAddress){
    _receiverAddress = receiverAddress;
}

void ProbeApp::DoDispose() {
    NS_LOG_FUNCTION (this);
    Application::DoDispose();
}

void ProbeApp::StartApplication() {
    NS_LOG_FUNCTION(this);
    cout << "Node " << GetNodeIP(GetNode(), 1) << " ProbeApp started at: " << Simulator::Now().GetSeconds() << " Will end at: " << this->m_stopTime.GetNanoSeconds() << endl;
    m_var->SetAttribute("Mean", DoubleValue(1/_rate));
    PrepareConnection();
    double nextEventTime = m_var->GetValue();
    // cout << "Next event scheduled at: " << (Simulator::Now() + Seconds(nextEventTime)).GetNanoSeconds() << " seconds" << endl;
    socket->Send(Create<Packet>(0)); // Send an empty packet to establish the connection
    _sendEvent = Simulator::Schedule(Seconds(nextEventTime) + Seconds(0.001), &ProbeApp::ScheduleNextSend, this);
}

void ProbeApp::PrepareConnection() {
    NS_LOG_FUNCTION (this);
    // Create a socket
    TypeId tid = TypeId::LookupByName (_protocol);
    socket = Socket::CreateSocket (GetNode(), tid);
    if (socket->Bind () == -1) {
        cout << "Failed to bind socket" << endl;
    }
    if (socket->Connect(_receiverAddress) == -1) {
        cout << "Failed to connect socket to " << InetSocketAddress::ConvertFrom(_receiverAddress).GetIpv4() << " on port " << InetSocketAddress::ConvertFrom(_receiverAddress).GetPort() << endl;
    }
    socket->SetRecvCallback (MakeNullCallback<void, Ptr<Socket> > ());
    socket->SetAllowBroadcast (true);
}

void ProbeApp::StopApplication() {
    NS_LOG_FUNCTION (this);
    cout << "Node " << GetNodeIP(GetNode(), 1) << " ProbeApp stopped at: " << Simulator::Now().GetSeconds() << endl;
    //close connection
    if (socket) {
        socket->Close();
    }
    Simulator::Cancel (_sendEvent);
}

void ProbeApp::Send() {
    NS_LOG_FUNCTION(this);
    // cout << "Node " << GetNodeIP(GetNode(), 1) << " ProbeApp sending packet at: " << Simulator::Now().GetSeconds() << endl;
    DynamicCast<TcpSocketBase>(socket)->SendProbe();
    // socket->SendProbe();
}

void ProbeApp::ScheduleNextSend() {
    Send();
    double nextEvent = m_var->GetValue();
    _sendEvent = Simulator::Schedule(Seconds(nextEvent), &ProbeApp::ScheduleNextSend, this);
}

