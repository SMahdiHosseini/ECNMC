//
// Created by Mahdi on 26.11.25.
//

#include "IncastGenerator.h"

uint32_t IncastGenerator::SOCKET_COUNT = 0;

IncastGenerator::IncastGenerator(const vector<NodeContainer>& nodes, int incastFactor, uint32_t messageSize, Time incastperiod, Time trafficStartTime, Time trafficEndTime) :
        incastFactor(incastFactor), messageSize(messageSize), incastperiod(incastperiod), trafficStartTime(trafficStartTime), trafficEndTime(trafficEndTime) {
    for (const auto& nc : nodes) {
        for (NodeContainer::Iterator it = nc.Begin(); it != nc.End(); ++it) {
            this->nodes.push_back(*it);
        }
    }
    m_uv = CreateObject<UniformRandomVariable>();
}

void 
IncastGenerator::GenerateTraffic() {
    Simulator::Schedule(incastperiod, &IncastGenerator::GenerateTraffic, this);

    std::set<uint32_t> senderIndices;
    while (senderIndices.size() < static_cast<size_t>(incastFactor)) {
        uint32_t index = m_uv->GetInteger(0, nodes.size() - 1);
        senderIndices.insert(index);
    }

    uint32_t receiverIndex;
    do {
        receiverIndex = m_uv->GetInteger(0, nodes.size() - 1);
    } while (senderIndices.find(receiverIndex) != senderIndices.end());

    Ptr<Node> receiverNode = nodes[receiverIndex];
    for (const auto& senderIndex : senderIndices) {
        InetSocketAddress receiverAddress = InetSocketAddress(GetNodeIP(receiverNode, 1), 20000 + ++SOCKET_COUNT);
        // create sink at receiver
        IncastReceiverHelper incastHelperServer(receiverAddress);
        incastHelperServer.SetAttribute("Protocol", StringValue("ns3::TcpSocketFactory"));
        incastHelperServer.SetAttribute("MessageSize", UintegerValue(messageSize));
        ApplicationContainer incastAppServer = incastHelperServer.Install(receiverNode);
        incastAppServer.Start(Seconds(0));

        // create a socket at sender and send data
        Ptr<Node> senderNode = nodes[senderIndex];
        Ptr<Socket> socket;
        TypeId tid = TypeId::LookupByName ("ns3::TcpSocketFactory");
        socket = Socket::CreateSocket(senderNode, tid);
        if (socket->Bind () == -1) {
            NS_FATAL_ERROR ("Failed to bind socket");
        }
        socket->Connect(receiverAddress);
        socket->SetConnectCallback(MakeCallback(&IncastGenerator::ConnectionSucceeded, this), MakeNullCallback<void, Ptr<Socket> > ());
        socket->SetCloseCallbacks(MakeCallback(&IncastGenerator::ConnectionClosed, this), MakeNullCallback<void, Ptr<Socket> > ());
        // socket->SetRecvCallback (MakeCallback(&IncastGenerator::Recv, this));
        socket->SetAllowBroadcast (true);
        Ptr<TcpSocketBase> tcpSocket = socket->GetObject<TcpSocketBase>();
        tcpSocket->SetPacingStatus(true); 
        // cout << "IncastGenerator: Created socket at sender node " << senderNode->GetId() << " and receiver node: " << receiverNode->GetId() << " at " << Simulator::Now().GetNanoSeconds() << " seconds." << endl;
    }   
}
void IncastGenerator::ConnectionSucceeded(Ptr<Socket> socket) {
    socket->Send(Create<Packet>(messageSize));
}

void IncastGenerator::ConnectionClosed(Ptr<Socket> socket) {
    // cout << "IncastGenerator: Connection closed at " << Simulator::Now().GetSeconds() << " seconds." << endl;
    return;
}

void IncastGenerator::Recv(Ptr<Socket> socket) {
    // check if what is written in the packet is "DONE"
    Ptr<Packet> packet = socket->Recv();
    uint32_t packetSize = packet->GetSize();
    cout << "IncastGenerator: Received packet of size " << packetSize << " bytes at " << Simulator::Now().GetSeconds() << " seconds." << endl;
    if (packetSize == 5) {
        uint8_t buffer[4];
        packet->CopyData(buffer, 4);
        if (buffer[0] == 'D' && buffer[1] == 'O' && buffer[2] == 'N' && buffer[3] == 'E') {
            // close the socket
            cout << "IncastGenerator: Received DONE message at " << Simulator::Now().GetSeconds() << " seconds." << endl;
        }
    }
}

void 
IncastGenerator::Start() {
    Simulator::Schedule(trafficStartTime, &IncastGenerator::GenerateTraffic, this);
}

