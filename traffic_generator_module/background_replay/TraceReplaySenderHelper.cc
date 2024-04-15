//
// Created by nal on 31.08.20.
//

#include "TraceReplaySenderHelper.h"
#include "TraceReplaySender.h"


TraceReplaySenderHelper::TraceReplaySenderHelper() {
    _factory.SetTypeId (TraceReplaySender::GetTypeId ());
}

TraceReplaySenderHelper::TraceReplaySenderHelper(Address address, uint16_t port) {
    _factory.SetTypeId (TraceReplaySender::GetTypeId ());
    SetAttribute ("RemoteAddress", AddressValue(address));
    SetAttribute ("RemotePort", UintegerValue(port));
}

TraceReplaySenderHelper::TraceReplaySenderHelper(Address address) {
    _factory.SetTypeId (TraceReplaySender::GetTypeId ());
    SetAttribute ("RemoteAddress", AddressValue (address));
}

void TraceReplaySenderHelper::SetAttribute(std::string name, const AttributeValue &value) {
    _factory.Set (name, value);
}

ApplicationContainer TraceReplaySenderHelper::Install(NodeContainer c) {
    ApplicationContainer apps;
    for (NodeContainer::Iterator i = c.Begin (); i != c.End (); ++i) {
        Ptr<Node> node = *i;
        Ptr<TraceReplaySender> client = _factory.Create<TraceReplaySender> ();
        node->AddApplication(client);
        apps.Add (client);
    }
    return apps;
}