//
// Created by nal on 08.02.21.
//

#include "ns3/applications-module.h"

#include <random>

#include "TraceReplaySenderHelper.h"
#include "TraceReplayReceiverHelper.h"
#include "BackgroundReplay.h"

uint32_t BackgroundReplay::SOCKET_COUNT = 0;

BackgroundReplay::BackgroundReplay(const Ptr<Node>& sender, const Ptr<Node>& receiver) : _sender(sender), _receiver(receiver) {}

void BackgroundReplay::SetPctOfPacedTcps(double pct) {
    _pctOfPacedTcp = pct;
}

void BackgroundReplay::RunAllTraces(const string& tracesPath, uint32_t nbTCPFlows, uint32_t nbUDPFlows, uint8_t dscp) {
    uint32_t j = 0;
    for(uint32_t i = 0; i < nbUDPFlows; i++) {
        while (!std::filesystem::exists(tracesPath + "/UDP/trace_" + to_string(j) + ".csv")) {
            j++;
        }
        string tracePath = tracesPath + "/UDP/trace_" + to_string(j) + ".csv";
        RunSingleTrace(tracePath, "ns3::UdpSocketFactory", dscp);
        j++;
    }

    j = 0;
    for(uint32_t i = 0; i < nbTCPFlows; i++) {
        while (!std::filesystem::exists(tracesPath + "/TCP/trace_" + to_string(j) + ".csv")) {
            j++;
        }
        string tracePath = tracesPath + "/TCP/trace_" + to_string(j) + ".csv";
        RunSingleTrace(tracePath, "ns3::TcpSocketFactory", dscp);
        j++;
    }
}

void BackgroundReplay::RunAllTraces(const string& tracesPath, uint8_t dscp) {
    uint32_t nbTCPFlows = GetSubDirCount(tracesPath + "/TCP");
    uint32_t nbUDPFlows = GetSubDirCount(tracesPath + "/UDP");
    this->RunAllTraces(tracesPath, nbTCPFlows, nbUDPFlows, dscp);
}

void BackgroundReplay::RunAllTCPTraces(const string& tracesPath, uint8_t dscp) {
    uint32_t nbTCPFlows = GetSubDirCount(tracesPath + "/TCP");
    uint32_t nbUDPFlows = GetSubDirCount(tracesPath + "/UDP");
    // in datacenter setting we do not have UDP flows
    nbUDPFlows = 0;
    this->RunAllTraces(tracesPath, nbTCPFlows, nbUDPFlows, dscp);
}

void BackgroundReplay::RunSpecificTraces(const vector<string>& tcpTracesPath, const vector<string>& udpTracesPath, uint8_t dscp) {
    for(const string& tracePath: udpTracesPath) {
        RunSingleTrace(tracePath, "ns3::UdpSocketFactory", dscp);
    }
    for(const string& tracePath: tcpTracesPath) {
        RunSingleTrace(tracePath, "ns3::TcpSocketFactory", dscp);
    }
}

vector<string>
BackgroundReplay::RunTracesWithRandomThrottledTCPFlows(const string& tracesPath, double throttledProb, uint8_t thottledDscp) {
    uint32_t nbTCPFlows = GetSubDirCount(tracesPath + "/TCP");
    vector<string> tcpTracesPathNeutral, tcpTracesPathThrottled;

    random_device rd;
    mt19937 mt(rd());
    uniform_real_distribution<double> dist(0.0, 1.0);

    uint32_t countThrottledTCP = 0;
    for(uint32_t i = 0; i < nbTCPFlows; i++) {
        string tracePath = tracesPath + "/TCP/trace_" + to_string(i) + ".csv";
        if (dist(mt) < throttledProb) {
            countThrottledTCP++;
            tcpTracesPathThrottled.push_back(tracePath);
        }
        else {
            tcpTracesPathNeutral.push_back(tracePath);
        }
    }
    this->RunSpecificTraces(tcpTracesPathNeutral, {tracesPath + "/UDP/trace_0.csv"}, 0);
    this->RunSpecificTraces(tcpTracesPathThrottled, {}, thottledDscp);
    return tcpTracesPathThrottled;
}

vector<string>
BackgroundReplay::RunTracesWithRandomThrottledUDPFlows(const string& tracesPath, double throttledProb, uint8_t thottledDscp) {
    uint32_t nbTCPFlows = GetSubDirCount(tracesPath + "/TCP");
    vector<string> tcpTracesPathNeutral, udpTracesPathThrottled;

    random_device rd;
    mt19937 mt(rd());
    uniform_real_distribution<double> dist(0.0, 1.0);

    uint32_t countThrottledUDP = 0;
    for(uint32_t i = 0; i < nbTCPFlows; i++) {
        string tracePath = tracesPath + "/TCP/trace_" + to_string(i) + ".csv";
        if (dist(mt) < throttledProb) {
            countThrottledUDP++;
            udpTracesPathThrottled.push_back(tracePath);
        }
        else {
            tcpTracesPathNeutral.push_back(tracePath);
        }
    }
    this->RunSpecificTraces(tcpTracesPathNeutral, {tracesPath + "/UDP/trace_0.csv"}, 0);
    this->RunSpecificTraces({}, udpTracesPathThrottled, thottledDscp);
    return udpTracesPathThrottled;
}

void BackgroundReplay::RunSingleTrace(const string& tracePath, const string& protocol, uint8_t dscp = 0) {
    uint32_t traceId = ++SOCKET_COUNT;

    // randomly select whether to pace this sender
    random_device rd;
    mt19937 mt(rd());
    uniform_real_distribution<double> dist(0.0, 1.0);
    bool enablePacing = dist(mt) < _pctOfPacedTcp;

    InetSocketAddress receiverAddress = InetSocketAddress(GetNodeIP(_receiver, 1), 4000 + traceId);
    receiverAddress.SetTos(Dscp2Tos(dscp)); // for traffic differentiation

    // create sink at server
    TraceReplayReceiverHelper replayHelperServer(receiverAddress);
    replayHelperServer.SetAttribute("Protocol", StringValue(protocol));
    ApplicationContainer replayAppServer = replayHelperServer.Install(_receiver);
    replayAppServer.Start(Simulator::Now());

    // create the replay application at client
    TraceReplaySenderHelper replayHelperClient(receiverAddress);
    replayHelperClient.SetAttribute("Protocol", StringValue(protocol));
    replayHelperClient.SetAttribute("TraceFile", StringValue(tracePath));
    replayHelperClient.SetAttribute("EnablePacing", BooleanValue(enablePacing));
    ApplicationContainer replayAppClient = replayHelperClient.Install(_sender);
    replayAppClient.Start(Simulator::Now());
}
