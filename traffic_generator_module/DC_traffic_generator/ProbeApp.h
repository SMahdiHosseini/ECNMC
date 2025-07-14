//
// Created by nal on 30.06.25
//

#ifndef PROBEAPP_H
#define PROBEAPP_H

#include "ns3/core-module.h"
#include "ns3/applications-module.h"
#include "ns3/internet-module.h"

#include "../../helper_classes/HelperMethods.h"

using namespace ns3;
using namespace std;
using namespace helper_methods;

class ProbeApp : public Application {

private:
    void StartApplication() override;
    void PrepareConnection();
    void StopApplication() override;
    void Send();
    void ScheduleNextSend();
    void ReadWorkloadFile();

    Address _receiverAddress;
    string _protocol;
    double _rate;;
    Ptr<ExponentialRandomVariable> m_var;;
    EventId _sendEvent;
    Ptr<Socket> socket;
protected:
    void DoDispose() override;

public:
    static TypeId GetTypeId();
    void SetReceiverAddress(Address receiverAddress);
    ProbeApp();
    ~ProbeApp() override;

};


#endif //PROBEAPP_H
