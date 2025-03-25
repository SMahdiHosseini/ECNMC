//
// Created by nal on 25.03.25
//

#ifndef POISSONREPLAYSENDER_H
#define POISSONREPLAYSENDER_H

#include "ns3/core-module.h"
#include "ns3/applications-module.h"
#include "ns3/internet-module.h"

#include "../../helper_classes/HelperMethods.h"

using namespace ns3;
using namespace std;
using namespace helper_methods;

class PoissonReplaySender : public Application {

private:
    void StartApplication() override;
    void PrepareSocket();
    void StopApplication() override;
    void Send();
    void ScheduleNextSend();

    Ptr<Socket> _socket;
    Address _receiverAddress;
    string _protocol;
    uint32_t _frameNb;
    EventId _sendEvent;
    double _rate;
    Ptr<ExponentialRandomVariable> m_var;;

protected:
    void DoDispose() override;

public:
    static TypeId GetTypeId();

    PoissonReplaySender();
    ~PoissonReplaySender() override;

};


#endif //POISSONREPLAYSENDER_H
