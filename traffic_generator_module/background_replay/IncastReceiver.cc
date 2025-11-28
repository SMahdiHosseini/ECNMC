//
// Created by nal on 28.11.25.
//

#include "IncastReceiver.h"

NS_LOG_COMPONENT_DEFINE ("IncastReceiver");

NS_OBJECT_ENSURE_REGISTERED (IncastReceiver);
TypeId IncastReceiver::GetTypeId() {

    static TypeId tid = TypeId ("ns3::IncastReceiver")
            .SetParent<TraceReplayReceiver> ()
            .SetGroupName("Applications")
            .AddConstructor<IncastReceiver> ()
            .AddAttribute ("MessageSize",
                           "The size of the message to be received before sending DONE",
                           UintegerValue (100),
                           MakeUintegerAccessor (&IncastReceiver::messageSize),
                           MakeUintegerChecker<uint32_t> (1))
    ;
    return tid;
}

IncastReceiver::IncastReceiver() {
    NS_LOG_FUNCTION (this);
    totalBytesReceived = 0;
}

IncastReceiver::~IncastReceiver() {
    NS_LOG_FUNCTION (this);
}

void IncastReceiver::DoDispose() {
    NS_LOG_FUNCTION (this);

    TraceReplayReceiver::DoDispose();
}

void IncastReceiver::Recv(Ptr<Socket> socket) {
    NS_LOG_FUNCTION(this);

    Ptr<Packet> packet = socket->Recv();
    // check if the total number of bytes received is 100B send back "DONE" message to sender
    // and close the socket
    // else do nothing
    totalBytesReceived += packet->GetSize();
    if(totalBytesReceived >= 100) {
        const char doneMessage[] = "DONE";
        Ptr<Packet> donePacket = Create<Packet> ((const uint8_t*)"DONE", sizeof(doneMessage));
        socket->Send(donePacket);
        socket->Close();
    }
}

