//
// Created by nal on 24.04.25
//

#ifndef WORKLOAD_APP_H
#define WORKLOAD_APP_H

#include "ns3/core-module.h"
#include "ns3/applications-module.h"
#include "ns3/internet-module.h"

#include "../../helper_classes/HelperMethods.h"
#include "ConnectionPool.h"

using namespace ns3;
using namespace std;
using namespace helper_methods;

class WorkloadApp : public Application {

private:
    void StartApplication() override;
    void PrepareConnections();
    void StopApplication() override;
    void Send();
    void ScheduleNextSend();
    void ReadWorkloadFile();

    vector<vector<Address>> _receiverAddress;
    vector<ConnectionPool> _connectionPools;
    string _protocol;
    double _rate;
    uint32_t _receiversNumber;
    Ptr<ExponentialRandomVariable> m_var;;
    EventId _sendEvent;
    std::string workloadPath;
    Ptr<EmpiricalRandomVariable> m_erv;
    Ptr<UniformRandomVariable> m_uniform;
protected:
    void DoDispose() override;

public:
    static TypeId GetTypeId();
    void SetReceiverAddress(vector<vector<Address>> receiversAddresses);
    WorkloadApp();
    ~WorkloadApp() override;

};


#endif //WORKLOAD_APP_H
