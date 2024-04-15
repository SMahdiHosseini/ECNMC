//
// Created by nal on 27.04.21.
//

#ifndef WEHE_P_TOMOGRAPHY_TRACEREPLAYRECEIVER_H
#define WEHE_P_TOMOGRAPHY_TRACEREPLAYRECEIVER_H

#include "ns3/core-module.h"
#include "ns3/applications-module.h"
#include "ns3/internet-module.h"

using namespace ns3;
using namespace std;

class TraceReplayReceiver : public Application {
private:

    void StartApplication() override;
    void StopApplication() override;

    void HandleTCPAccept (Ptr<Socket> socket, const Address& from);
    void Recv(Ptr<Socket> socket);

    Address _serverAddress;
    string _protocol;
    Ptr<Socket> _socket;

protected:
    void DoDispose() override;

public:
    static TypeId GetTypeId();

    TraceReplayReceiver();
    ~TraceReplayReceiver() override;
};


#endif //WEHE_P_TOMOGRAPHY_TRACEREPLAYRECEIVER_H
