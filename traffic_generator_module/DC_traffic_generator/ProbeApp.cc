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
                           StringValue ("ns3::UdpSocketFactory"),
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
    _sendEvent = Simulator::Schedule(Seconds(nextEventTime), &ProbeApp::ScheduleNextSend, this);
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
    //close connection
    if (socket) {
        socket->Close();
    }
    Simulator::Cancel (_sendEvent);
}

void ProbeApp::Send() {
    NS_LOG_FUNCTION(this);
    // send a packet
    Ptr<Packet> packet = Create<Packet>(1); // Create a packet of size 1 bytes
    if (socket->Send(packet) < 0) {
        // cout << "Error sending packet from " << GetNodeIP(GetNode(), 1) << " to " << InetSocketAddress::ConvertFrom(_receiverAddress).GetIpv4() << endl;
        NS_LOG_INFO ("Error while sending packet to " << InetSocketAddress::ConvertFrom(_receiverAddress).GetIpv4());
    } else {
        // cout << "Packet sent from " << GetNodeIP(GetNode(), 1) << " to " << InetSocketAddress::ConvertFrom(_receiverAddress).GetIpv4() << endl;
        NS_LOG_INFO ("Packet sent to " << InetSocketAddress::ConvertFrom(_receiverAddress).GetIpv4());
    }

}

void ProbeApp::ScheduleNextSend() {
    Send();
    double nextEvent = m_var->GetValue();
    _sendEvent = Simulator::Schedule(Seconds(nextEvent), &ProbeApp::ScheduleNextSend, this);
}

