//
// Created by nal on 24.04.25
//

#ifndef NODEAPPSHANDLER_H
#define NODEAPPSHANDLER_H

#include "ns3/core-module.h"
#include "ns3/applications-module.h"
#include "ns3/internet-module.h"

#include "../../helper_classes/HelperMethods.h"

using namespace ns3;
using namespace std;
using namespace helper_methods;

class NodeAppsHandler : public Application {

private:
    void StartApplication() override;
    void PrepareSockets();
    void StopApplication() override;
    void Send();
    void ScheduleNextSend();
    void ReadWorkloadFile();

    // std::unordered_map<Address, vector<Ptr<Socket>>> _connectionPool;
    std::vector<Address> _receiverAddress;
    string _protocol;
    double _rate;
    uint32_t _connectionPoolSize;
    Ptr<ExponentialRandomVariable> m_var;;
    EventId _sendEvent;
    std::string workloadFile;
    Ptr<EmpiricalRandomVariable> m_erv;
protected:
    void DoDispose() override;

public:
    static TypeId GetTypeId();
    void addReceiverAddress(Address address);
    NodeAppsHandler();
    ~NodeAppsHandler() override;

};


#endif //NODEAPPSHANDLER_H
