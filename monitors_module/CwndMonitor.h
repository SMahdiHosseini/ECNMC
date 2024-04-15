//
// Created by Zeinab Shmeis on 25.03.20.
//

#ifndef DUMBBELL_TOPOLOGY_CWNDMONITOR_H
#define DUMBBELL_TOPOLOGY_CWNDMONITOR_H


#include <cstdint>

#include "ns3/core-module.h"
#include "ns3/internet-module.h"

using namespace ns3;
using namespace std;

template<typename T>

struct ChangeTuple {
    T oldVal;
    T newVal;
    ns3::Time time;

    [[nodiscard]] string to_string() const;
};


class CwndMonitor {

    private:
        Ptr<Socket> _socket;

        ns3::Time _startTime;
        string _outputFolderPath;

        vector<ChangeTuple<uint32_t>> cwndChanges;
        vector<ChangeTuple<uint32_t>> cwndInfChanges;
        vector<ChangeTuple<TcpSocketState::TcpCongState_t>> congStateChanges;
        vector<ChangeTuple<ns3::Time>> rtoChanges;
        vector<ChangeTuple<ns3::Time>> rttChanges;
        vector<ChangeTuple<ns3::DataRate>> pacingfRateChanges;

        void ConnectTraces();
        void RecordCwndChange(uint32_t oldVal, uint32_t newVal);
        void RecordCongStateChange(TcpSocketState::TcpCongState_t oldVal, TcpSocketState::TcpCongState_t newVal);
        void RecordRTO(Time oldVal, Time newVal);
        void RecordRTT(Time oldVal, Time newVal);
        void RecordPacingRate(DataRate oldVal, DataRate newVal);

        string GetDisplayTime(const Time& time);

    public:
        CwndMonitor(const Ptr<Socket>& socket, string outputFolderPath);
        void SaveCwndChanges();

        void SaveCongStateChanges();
        void SaveRtoChanges();
        void SaveRttChanges();
        void SavePacingRateChanges();

};

#endif //DUMBBELL_TOPOLOGY_CWNDMONITOR_H
