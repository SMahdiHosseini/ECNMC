//
// Created by Mahdi on 25.05.25
//

#ifndef CONNECTIONPOOL_H
#define CONNECTIONPOOL_H

#include "ns3/core-module.h"
#include "ns3/applications-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-net-device.h"
#include "ns3/queue.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/drop-tail-queue.h"

#include "../../helper_classes/HelperMethods.h"

using namespace ns3;
using namespace std;
using namespace helper_methods;

class ConnectionPool {
public:
    ConnectionPool(const Address& address, const string& protocol, Ptr<Node> senderNode, double probeInterval);
    ~ConnectionPool();
    void CreateSockets(vector<Address> receiverAddresses, bool enablePacing, bool enableProbe);
    void CloseConnections();
    void SendData(const Ptr<Packet>& packet);
    void SetSocketState(uint32_t socketId, bool state);
    bool GetSocketState(uint32_t socketId) const;
    void ProbeNetwork();
private:
    vector<Ptr<Socket>> sockets;
    vector<bool> socketStates;
    Address remoteAddress;
    string protocol;
    Ptr<Node> senderNode;
    Ptr<UniformRandomVariable> m_uniform;
    Ptr<ExponentialRandomVariable> m_varProbe;
    Ptr<Socket> findIdleSocket();
    EventId _probeEvent;
    double _probeInterval;
    void ScheduleNextProbe();
};

#endif //CONNECTIONPOOL_H