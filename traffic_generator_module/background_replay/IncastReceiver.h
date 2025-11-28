//
// Created by nal on 28.11.25.
//

#ifndef INCASTRECEIVER_H
#define INCASTRECEIVER_H

#include "ns3/core-module.h"
#include "ns3/applications-module.h"
#include "ns3/internet-module.h"
#include "TraceReplayReceiver.h"

using namespace ns3;
using namespace std;

class IncastReceiver : public TraceReplayReceiver {
private:
    void Recv(Ptr<Socket> socket) override;
    uint32_t totalBytesReceived = 0;
    uint32_t messageSize = 100;

protected:
    void DoDispose() override;

public:
    static TypeId GetTypeId();

    IncastReceiver();
    ~IncastReceiver() override;
};


#endif //INCASTRECEIVER_H