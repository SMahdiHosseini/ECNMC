//
// Created by nal on 31.08.20.
//

#ifndef WEHE_PLUS_TOMOGRAPHY_TRACEREPLAYSENDER_H
#define WEHE_PLUS_TOMOGRAPHY_TRACEREPLAYSENDER_H

#include "ns3/core-module.h"
#include "ns3/applications-module.h"
#include "ns3/internet-module.h"

#include "../../helper_classes/HelperMethods.h"

using namespace ns3;
using namespace std;
using namespace helper_methods;

struct TraceReplayItem {
    uint32_t frameNb;
    Time timestamp;
    uint32_t payloadSize;
};

class TraceReplaySender : public Application {

private:
    void LoadTrace(const string& traceFile);
    void StartApplication() override;
    void PrepareSocket();
    void StopApplication() override;
    void Send(const TraceReplayItem& item);
    void ScheduleNextSend();
    void dctcpCallBack(uint32_t bytesAcked, uint32_t bytesMarked, double alpha);

    Ptr<Socket> _socket;
    Address _receiverAddress;
    string _protocol;
    string _traceFilename;
    vector<TraceReplayItem> _traceItems;
    uint32_t _traceItemIdx;
    bool _enablePacing;
    EventId _sendEvent, _startEvent;
    Time _trafficStartTime;
    Time _trafficEndTime;
    bool _isSecondPhase;

protected:
    void DoDispose() override;

public:
    static TypeId GetTypeId();

    TraceReplaySender();
    ~TraceReplaySender() override;

};


#endif //WEHE_PLUS_TOMOGRAPHY_TRACEREPLAYSENDER_H
