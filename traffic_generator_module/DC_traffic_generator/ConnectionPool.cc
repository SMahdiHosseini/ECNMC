
#include "ConnectionPool.h"

NS_LOG_COMPONENT_DEFINE ("ConnectionPool");

ConnectionPool::ConnectionPool(const Address& address, const string& protocol, Ptr<Node> senderNode)
    : remoteAddress(address), protocol(protocol), senderNode(senderNode) {
    NS_LOG_FUNCTION(this);
    m_uniform = CreateObject<UniformRandomVariable>();
}

ConnectionPool::~ConnectionPool() {
    NS_LOG_FUNCTION(this);
}

void
ConnectionPool::CloseConnections() {
    NS_LOG_FUNCTION(this);
    for (auto& socket : sockets) {
        if (socket) {
            socket->Close();
        }
    }
    sockets.clear();
    socketStates.clear();
}

void 
ConnectionPool::CreateSockets(vector<Address> receiverAddresses, bool enablePacing) {
    NS_LOG_FUNCTION(this);
    for (const auto& receiverAddress : receiverAddresses) {
        NS_LOG_FUNCTION (this);
        Ptr<Socket> socket;
        TypeId tid = TypeId::LookupByName (protocol);
        socket = Socket::CreateSocket (senderNode, tid);
        if (socket->Bind () == -1) {
            NS_FATAL_ERROR ("Failed to bind socket");
        }
        socket->Connect(receiverAddress);
        socket->SetRecvCallback (MakeNullCallback<void, Ptr<Socket> > ());
        socket->SetAllowBroadcast (true);
        if(protocol == "ns3::TcpSocketFactory") {
            Ptr<TcpSocketBase> tcpSocket = socket->GetObject<TcpSocketBase>();
            tcpSocket->SetPacingStatus(enablePacing);
            // tcpSocket->StartMeasurementProbeClock();
        }
        sockets.push_back(socket);
        socketStates.push_back(false);
        cout << "Socket created for " << InetSocketAddress::ConvertFrom(receiverAddress).GetIpv4() << " On port " << InetSocketAddress::ConvertFrom(receiverAddress).GetPort() << endl;
    }
}
Ptr<Socket> ConnectionPool::findIdleSocket() {
    uint32_t socketIndex = m_uniform->GetInteger(0, sockets.size() - 1);
    if (DynamicCast<TcpSocketBase>(sockets[socketIndex])->GetTxBuffer()->Size() > 0) {
        // cout << "Socket " << socketIndex << " is full, trying another to find the first none empty.\n";
        for (uint32_t i = 0; i < sockets.size(); ++i) {
            if (DynamicCast<TcpSocketBase>(sockets[i])->GetTxBuffer()->Size() == 0) {
                // cout << "Found available socket at index: " << i << " Nagle's algo: " << DynamicCast<TcpSocketBase>(sockets[i])->GetTcpNoDelay() << endl;
                // cout << "Socket Size: " << DynamicCast<TcpSocketBase>(sockets[i])->GetTxBuffer()->Size() << endl;
                return sockets[i];
            }
        }
        // cout << "All sockets are full, returning the first one.\n";
        // cout << "Socket Size: " << DynamicCast<TcpSocketBase>(sockets[socketIndex])->GetTxBuffer()->Size() << endl;
        return sockets[socketIndex];
    } else {
        // cout << "Socket " << socketIndex << " is available for sending.\n";
        // cout << "Socket Size: " << DynamicCast<TcpSocketBase>(sockets[socketIndex])->GetTxBuffer()->Size() << endl;
        return sockets[socketIndex];
    }
    
}
void 
ConnectionPool::SendData(const Ptr<Packet>& packet) {
    NS_LOG_FUNCTION(this);
    if (findIdleSocket()->Send(packet) < 0) {
        // cout << "Error sending packet from " << GetNodeIP(senderNode, 1) << " to " << InetSocketAddress::ConvertFrom(remoteAddress).GetIpv4() << endl;
        NS_LOG_INFO ("Error while sending packet to " << InetSocketAddress::ConvertFrom(remoteAddress).GetIpv4());
    } else {
        // cout << "Packet sent from " << GetNodeIP(senderNode, 1) << " to " << InetSocketAddress::ConvertFrom(remoteAddress).GetIpv4() << endl;
        NS_LOG_INFO ("Packet sent to " << InetSocketAddress::ConvertFrom(remoteAddress).GetIpv4());
    }
}