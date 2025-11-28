//
// Created by nal on 28.11.25.
//

#ifndef INCASTRECEIVERHELPER_H
#define INCASTRECEIVERHELPER_H

#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/application-container.h"

using namespace ns3;
using namespace std;

class IncastReceiverHelper {

private:
    ObjectFactory _factory;

public:
    IncastReceiverHelper();

    IncastReceiverHelper(Address addr);
    void SetAttribute(std::string name, const AttributeValue &value);

    ApplicationContainer Install(NodeContainer c);
};


#endif //INCASTRECEIVERHELPER_H