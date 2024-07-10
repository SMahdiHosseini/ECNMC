//
// Created by Mahdi Hosseini on 10.07.24.
//

#include "Monitor.h"

MonitorEvent::MonitorEvent() {}
void MonitorEvent::SetPacketKey(PacketKey *key) { _key = key; }
void MonitorEvent::SetSent() { _sentTime = ns3::Simulator::Now(); }
void MonitorEvent::SetSent(Time t) { _sentTime = t; }
void MonitorEvent::SetReceived() { _receivedTime = ns3::Simulator::Now(); }
void MonitorEvent::SetReceived(Time t) { _receivedTime = t; }
PacketKey *MonitorEvent::GetPacketKey() const { return _key; }
Time MonitorEvent::GetSentTime() const { return _sentTime; }
Time MonitorEvent::GetReceivedTime() const {  return _receivedTime; }
bool MonitorEvent::IsReceived() const { return _receivedTime != Time(-1); }
bool MonitorEvent::IsSent() const { return _sentTime != Time(-1); }


Monitor::Monitor(const Time &startTime, const Time &duration, const Time &steadyStartTime, const Time &steadyStopTime, const string &monitorTag) {
    _startTime = startTime;
    _duration = duration;
    _monitorTag = monitorTag;
    _steadyStartTime = steadyStartTime;
    _steadyStopTime = steadyStopTime;
}

void Monitor::AddAppKey(AppKey appKey) {
    _appsKey.insert(appKey);
}

string Monitor::GetMonitorTag() const { return _monitorTag; }
Time Monitor::GetRelativeTime(const Time &time){ return time - _startTime; }

