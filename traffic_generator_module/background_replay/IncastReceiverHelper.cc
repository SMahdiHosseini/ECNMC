//
// Created by nal on 28.11.25.
//

#include "IncastReceiver.h"
#include "IncastReceiverHelper.h"
IncastReceiverHelper::IncastReceiverHelper() {
    _factory.SetTypeId (IncastReceiver::GetTypeId ());
}

IncastReceiverHelper::IncastReceiverHelper(Address address) {
    _factory.SetTypeId (IncastReceiver::GetTypeId ());
    SetAttribute ("ServerAddress", AddressValue (address));
}

void IncastReceiverHelper::SetAttribute(std::string name, const AttributeValue &value) {
    _factory.Set (name, value);
}

ApplicationContainer IncastReceiverHelper::Install(NodeContainer c) {
    ApplicationContainer apps;
    for (NodeContainer::Iterator i = c.Begin (); i != c.End (); ++i) {
        Ptr<Node> node = *i;
        Ptr<IncastReceiver> server = _factory.Create<IncastReceiver> ();
        node->AddApplication(server);
        apps.Add (server);
    }
    return apps;
}