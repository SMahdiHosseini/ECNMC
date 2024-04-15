//
// Created by nal on 27.04.21.
//

#include "TraceReplayReceiver.h"
#include "TraceReplayReceiverHelper.h"

TraceReplayReceiverHelper::TraceReplayReceiverHelper() {
    _factory.SetTypeId (TraceReplayReceiver::GetTypeId ());
}

TraceReplayReceiverHelper::TraceReplayReceiverHelper(Address address) {
    _factory.SetTypeId (TraceReplayReceiver::GetTypeId ());
    SetAttribute ("ServerAddress", AddressValue (address));
}

void TraceReplayReceiverHelper::SetAttribute(std::string name, const AttributeValue &value) {
    _factory.Set (name, value);
}

ApplicationContainer TraceReplayReceiverHelper::Install(NodeContainer c) {
    ApplicationContainer apps;
    for (NodeContainer::Iterator i = c.Begin (); i != c.End (); ++i) {
        Ptr<Node> node = *i;
        Ptr<TraceReplayReceiver> server = _factory.Create<TraceReplayReceiver> ();
        node->AddApplication(server);
        apps.Add (server);
    }
    return apps;
}