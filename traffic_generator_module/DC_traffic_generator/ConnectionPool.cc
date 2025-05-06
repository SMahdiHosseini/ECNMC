
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
ConnectionPool::CreateSockets(vector<Address> receiverAddresses) {
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
            tcpSocket->SetPacingStatus(false);
        }
        sockets.push_back(socket);
        socketStates.push_back(false);
        cout << "Socket created for " << InetSocketAddress::ConvertFrom(receiverAddress).GetIpv4() << " On port " << InetSocketAddress::ConvertFrom(receiverAddress).GetPort() << endl;
    }
}

void 
ConnectionPool::SendData(const Ptr<Packet>& packet) {
    NS_LOG_FUNCTION(this);
    uint32_t socketIndex = m_uniform->GetInteger(0, sockets.size() - 1);
    if (sockets[socketIndex]->Send(packet) < 0) {
        NS_LOG_INFO ("Error while sending packet to " << InetSocketAddress::ConvertFrom(remoteAddress).GetIpv4());
    } else {
        NS_LOG_INFO ("Packet sent to " << InetSocketAddress::ConvertFrom(remoteAddress).GetIpv4());
    }
}