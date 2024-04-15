//
// Created by nal on 27.04.21.
//

#ifndef WEHE_P_TOMOGRAPHY_TRACEREPLAYRECEIVERHELPER_H
#define WEHE_P_TOMOGRAPHY_TRACEREPLAYRECEIVERHELPER_H

#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/application-container.h"

using namespace ns3;
using namespace std;

class TraceReplayReceiverHelper {

private:
    ObjectFactory _factory;

public:
    TraceReplayReceiverHelper();

    TraceReplayReceiverHelper(Address addr);

    void SetAttribute(std::string name, const AttributeValue &value);

    ApplicationContainer Install(NodeContainer c);
};


#endif //WEHE_P_TOMOGRAPHY_TRACEREPLAYRECEIVERHELPER_H
