//
// Created by Mahdi on 25.05.25
//

#ifndef CONNECTIONPOOL_H
#define CONNECTIONPOOL_H

#include "ns3/core-module.h"
#include "ns3/applications-module.h"
#include "ns3/internet-module.h"

#include "../../helper_classes/HelperMethods.h"

using namespace ns3;
using namespace std;
using namespace helper_methods;

class ConnectionPool {
public:
    ConnectionPool(const Address& address, const string& protocol, Ptr<Node> senderNode);
    ~ConnectionPool();
    void CreateSockets(vector<Address> receiverAddresses);
    void CloseConnections();
    void SendData(const Ptr<Packet>& packet);
    void SetSocketState(uint32_t socketId, bool state);
    bool GetSocketState(uint32_t socketId) const;
private:
    vector<Ptr<Socket>> sockets;
    vector<bool> socketStates;
    Address remoteAddress;
    string protocol;
    Ptr<Node> senderNode;
    Ptr<UniformRandomVariable> m_uniform;
};

#endif //CONNECTIONPOOL_H