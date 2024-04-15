//
// Created by Zeinab Shmeis on 25.03.20.
//

#include "CwndMonitor.h"

#include <utility>

template<typename T>
string ChangeTuple<T>::to_string() const {
    std::ostringstream outStr;
    outStr << oldVal << "\t" << newVal << "\t" << time;
    return outStr.str();
}

CwndMonitor::CwndMonitor(const Ptr<Socket>& socket, string outputFolderPath) {
    _socket = socket;
    _startTime = ns3::Now();
    _outputFolderPath = std::move(outputFolderPath);
    system(("mkdir -p " + _outputFolderPath).c_str());

    Simulator::Schedule(Seconds(0), &CwndMonitor::ConnectTraces, this);
}

void CwndMonitor::ConnectTraces() {
    _socket->TraceConnectWithoutContext("CongestionWindow", MakeCallback(&CwndMonitor::RecordCwndChange, this));
    _socket->TraceConnectWithoutContext("CongState", MakeCallback(&CwndMonitor::RecordCongStateChange, this));
    _socket->TraceConnectWithoutContext("RTO", MakeCallback(&CwndMonitor::RecordRTO, this));
    _socket->TraceConnectWithoutContext("RTT", MakeCallback(&CwndMonitor::RecordRTT, this));
    _socket->TraceConnectWithoutContext("PacingRate", MakeCallback(&CwndMonitor::RecordPacingRate, this));
}

void CwndMonitor::RecordCwndChange(uint32_t oldVal, uint32_t newVal) {
    cwndChanges.push_back({oldVal, newVal, Simulator::Now()});
}

void CwndMonitor::RecordCongStateChange(TcpSocketState::TcpCongState_t oldVal, TcpSocketState::TcpCongState_t newVal) {
    congStateChanges.push_back({oldVal, newVal, Simulator::Now()});
}

void CwndMonitor::RecordRTO(Time oldVal, Time newVal) {
    rtoChanges.push_back({std::move(oldVal), std::move(newVal), Simulator::Now()});
}

void CwndMonitor::RecordRTT(Time oldVal, Time newVal) {
    rttChanges.push_back({std::move(oldVal), std::move(newVal), Simulator::Now()});
}

void CwndMonitor::RecordPacingRate(DataRate oldVal, DataRate newVal) {
    pacingfRateChanges.push_back({oldVal, newVal, Simulator::Now()});
}

string CwndMonitor::GetDisplayTime(const Time& time) {
    return to_string((time - _startTime).GetNanoSeconds() / 1e9);
}

void CwndMonitor::SaveCwndChanges() {
    ofstream outfile;
    outfile.open(_outputFolderPath + "/cwnd_changes.csv");
    for(const auto& cwndChange : cwndChanges) {
        outfile << cwndChange.oldVal << "," << cwndChange.newVal << "," << GetDisplayTime(cwndChange.time) << endl;
    }
    outfile.close();
}

void CwndMonitor::SaveCongStateChanges() {
    ofstream outfile;
    outfile.open(_outputFolderPath + "/cong_state_changes.csv");
    for(const auto& congStateChange : congStateChanges) {
        outfile << TcpSocketState::TcpCongStateName[congStateChange.oldVal] << "," <<
                TcpSocketState::TcpCongStateName[congStateChange.newVal] << "," <<
                GetDisplayTime(congStateChange.time) << endl;
    }
    outfile.close();
}

void CwndMonitor::SaveRtoChanges() {
    ofstream outfile;
    outfile.open(_outputFolderPath + "/rto_changes.csv");
    for(const auto& rtoChange : rtoChanges) {
        outfile << to_string(rtoChange.oldVal.GetNanoSeconds()/1e9) << "," <<
                to_string(rtoChange.newVal.GetNanoSeconds()/1e9) << "," <<
                GetDisplayTime(rtoChange.time) << endl;
    }
    outfile.close();
}

void CwndMonitor::SaveRttChanges() {
    ofstream outfile;
    outfile.open(_outputFolderPath + "/rtt_changes.csv");
    for(const auto& rttChange : rttChanges) {
        outfile << to_string(rttChange.oldVal.GetNanoSeconds() / 1e9) << "," <<
                to_string(rttChange.newVal.GetNanoSeconds() / 1e9) << "," <<
                GetDisplayTime(rttChange.time) << endl;
    }
    outfile.close();
}

void CwndMonitor::SavePacingRateChanges() {
    ofstream outfile;
    outfile.open(_outputFolderPath + "/pacing_rate_changes.csv");
    for(const auto& pacingRateChange : pacingfRateChanges) {
        outfile << to_string(pacingRateChange.oldVal.GetBitRate() * 1e-9) << "," <<
                to_string(pacingRateChange.newVal.GetBitRate() * 1e-9) << "," <<
                GetDisplayTime(pacingRateChange.time) << endl;
    }
    outfile.close();
}



