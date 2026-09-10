//
// Created by nal on 24.04.25
//
#include "WorkloadApp.h"

NS_LOG_COMPONENT_DEFINE ("WorkloadApp");

NS_OBJECT_ENSURE_REGISTERED (WorkloadApp);

TypeId WorkloadApp::GetTypeId() {

    static TypeId tid = TypeId ("ns3::WorkloadApp")
            .SetParent<Application> ()
            .SetGroupName("Applications")
            .AddConstructor<WorkloadApp> ()
            .AddAttribute ("Protocol", "the name of the protocol to use to send traffic by the applications",
                           StringValue ("ns3::TcpSocketFactory"),
                           MakeStringAccessor (&WorkloadApp::_protocol),
                           MakeStringChecker())
            .AddAttribute("Rate", "The rate of the Poisson process (request per second)",
                          DoubleValue(5000.0),
                          MakeDoubleAccessor(&WorkloadApp::_rate),
                          MakeDoubleChecker<double>())
            .AddAttribute("WorkloadPath", "The workload file to be used",
                          StringValue("scratch/ECNMC/DCWorkloads/Google_AllRPC.txt"),
                          MakeStringAccessor(&WorkloadApp::workloadPath),
                          MakeStringChecker())
            .AddAttribute("EnablePacing", "Enable pacing for the application",
                          BooleanValue(false),
                          MakeBooleanAccessor(&WorkloadApp::_enablePacing),
                          MakeBooleanChecker())
            .AddAttribute("Probe", "Enable probe traffic generation",
                            BooleanValue(false),
                            MakeBooleanAccessor(&WorkloadApp::_probe),
                            MakeBooleanChecker())
            .AddAttribute("ProbeInterval", "Interval for probe traffic generation",
                            DoubleValue(0.1),
                            MakeDoubleAccessor(&WorkloadApp::_probeInterval),
                            MakeDoubleChecker<double>())
            .AddAttribute("ProbeStartTime", "Start time for probe traffic generation",
                            TimeValue(Seconds(0)),
                            MakeTimeAccessor(&WorkloadApp::_probeStartTime),
                            MakeTimeChecker())
            .AddAttribute("ProbeStopTime", "Stop time for probe traffic generation",
                            TimeValue(Seconds(1)),
                            MakeTimeAccessor(&WorkloadApp::_probeStopTime),
                            MakeTimeChecker())
            .AddAttribute("ArrivalProcess",
                            "Message arrival process: \"Poisson\" (exponential inter-message times, the "
                            "original background DC workload) or \"Periodic\" (a deterministic 1/Rate "
                            "inter-message time, so that senders configured alike fire together)",
                            StringValue("Poisson"),
                            MakeStringAccessor(&WorkloadApp::_arrivalProcess),
                            MakeStringChecker())
            .AddAttribute("JitterNs",
                            "Per-message dither, drawn uniformly in [0, JitterNs) nanoseconds and "
                            "applied around an exact periodic grid rather than added to the "
                            "interval, so it never accumulates into a drift. A few nanoseconds is "
                            "far below one packet's serialisation time, so the senders still land "
                            "in the same burst -- it only randomises which of them the switch "
                            "serves first, which is otherwise fixed by ns-3's deterministic "
                            "tie-break and pins the monitored flow's position in every batch",
                            DoubleValue(0.0),
                            MakeDoubleAccessor(&WorkloadApp::_jitterNs),
                            MakeDoubleChecker<double>(0.0))
            .AddAttribute("StartPhase",
                            "Offset of this application's first message, in units of one period. "
                            "Senders given different phases fire staggered rather than together, "
                            "which is what turns one synchronized batch into a smooth stream",
                            DoubleValue(0.0),
                            MakeDoubleAccessor(&WorkloadApp::_startPhase),
                            MakeDoubleChecker<double>(0.0, 1.0))
            .AddAttribute("FixedMessageSize",
                            "If non-zero, every message is exactly this many bytes instead of a draw "
                            "from the workload's message-size CDF",
                            UintegerValue(0),
                            MakeUintegerAccessor(&WorkloadApp::_fixedMsgSize),
                            MakeUintegerChecker<uint32_t>())
    ;
    return tid;
}

WorkloadApp::WorkloadApp() {
    NS_LOG_FUNCTION (this);
    m_var = CreateObject<ExponentialRandomVariable>();
    m_erv = CreateObject<EmpiricalRandomVariable>();
    m_uniform = CreateObject<UniformRandomVariable>();
    _trafficStartTime = Seconds(0);
}

void WorkloadApp::SetTrafficStartTime(Time startTime) {
    NS_LOG_FUNCTION (this);
    _trafficStartTime = startTime;
}

WorkloadApp::~WorkloadApp() {
    NS_LOG_FUNCTION (this);
}

void WorkloadApp::ReadWorkloadFile() {
    NS_LOG_FUNCTION (this);
    std::ifstream file(workloadPath);
    if (!file.is_open()) {
        NS_FATAL_ERROR("Could not open workload file: " << workloadPath);
    }

    std::string line;
    std::getline(file, line); // Skip the header line
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        double value;
        double cdf;
        if (!(iss >> value >> cdf)) {
            NS_FATAL_ERROR("Error reading line from workload file: " << line);
        }
        // Assuming the workload file contains pairs of (value, cdf)
        // Add the value and cdf to the empirical random variable
        m_erv->CDF(value, cdf);
    }
    file.close();
}

void WorkloadApp::SetReceiverAddress(vector<vector<Address>> receiversAddresses){
    NS_LOG_FUNCTION (this);
    for (auto &addresses : receiversAddresses) {
        vector<Address> addressList;
        for (auto &address : addresses) {
            addressList.push_back(address);
        }
        _receiverAddress.push_back(addressList);
    }
    _receiversNumber = _receiverAddress.size();
    cout << "Number of receivers: " << _receiversNumber << endl;
}

void WorkloadApp::DoDispose() {
    NS_LOG_FUNCTION (this);
    Application::DoDispose();
}

void WorkloadApp::StartApplication() {
    NS_LOG_FUNCTION(this);
    cout << "Node " << GetNodeIP(GetNode(), 1) << " WorkloadApp started at: " << Simulator::Now().GetSeconds() << " Will end at: " << this->m_stopTime.GetNanoSeconds() << endl;
    m_var->SetAttribute("Mean", DoubleValue(1/_rate));
    if (_arrivalProcess == "Periodic") {
        cout << "    Periodic arrival process: period " << (1e9 / _rate) << " ns, message size "
             << (_fixedMsgSize > 0 ? to_string(_fixedMsgSize) + " B" : string("from the workload CDF"))
             << ", start phase " << _startPhase << " period(s) = " << (_startPhase * 1e9 / _rate)
             << " ns, per-message jitter [0, " << _jitterNs << ") ns" << endl;
    }
    ReadWorkloadFile();
    PrepareConnections();
    if (_arrivalProcess == "Periodic") {
        _nominalNext = _trafficStartTime + Seconds((_startPhase + 1.0) / _rate);
        ScheduleAtNominal();
    }
    else {
        double nextEventTime = m_var->GetValue();
        _sendEvent = Simulator::Schedule(_trafficStartTime + Seconds(nextEventTime), &WorkloadApp::ScheduleNextSend, this);
    }
}

void WorkloadApp::PrepareConnections() {
    NS_LOG_FUNCTION (this);

    for (auto &addresses : _receiverAddress) {
        ConnectionPool* connectionPool = new ConnectionPool(addresses[0], _protocol, GetNode(), _probeInterval);
        connectionPool->CreateSockets(addresses, _enablePacing, _probe, _probeStartTime, _probeStopTime);
        _connectionPools.push_back(connectionPool);
        // cout << "Connection Pool of: Sender Address: " << GetNode()->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal() << " Receiver Address: " << InetSocketAddress::ConvertFrom(addresses[0]).GetIpv4() << " created!" << endl;
    }
    cout << "Connection Pool of: Sender Address: " << GetNode()->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal() << " created!" << endl;
}

void WorkloadApp::StopApplication() {
    NS_LOG_FUNCTION (this);
    for (auto &connectionPool : _connectionPools) {
        connectionPool->CloseConnections();
    }
    cout << "Node " << GetNodeIP(GetNode(), 1) << " Connection Pools closed at " << Simulator::Now().GetNanoSeconds() << endl;
    Simulator::Cancel (_sendEvent);
}

void WorkloadApp::Send() {
    NS_LOG_FUNCTION(this);
    uint32_t segmentSize = _fixedMsgSize > 0 ? _fixedMsgSize : (uint32_t) m_erv->GetValue();
    // segmentSize *= 1442; // for DCTCP workload
    uint32_t selectedReceiver = m_uniform->GetInteger(0, _receiversNumber - 1);
    // cout << "Node " << GetNodeIP(GetNode(), 1) << " WorkloadApp sending to receiver size of: " << segmentSize << " at: " << Simulator::Now().GetNanoSeconds() << endl;
    // check if the node IP address contains "10.2."
    stringstream ss;
    ss << GetNodeIP(GetNode(), 1);
    string nodeIp = ss.str();
    // Poisson (all-to-all background) mode only: suppress rack1<->rack3 traffic. Periodic senders
    // are given an explicit destination, so this filter must not silently discard their messages
    // -- it would do exactly that for a rack-1 sender aimed at a rack-3 receiver.
    if (_arrivalProcess != "Periodic" &&
        (nodeIp.find("10.2.") != string::npos || nodeIp.find("10.4.") != string::npos)) {
        stringstream receiverSs;
        receiverSs << InetSocketAddress::ConvertFrom(_receiverAddress[selectedReceiver][0]).GetIpv4();
        string receiverIp = receiverSs.str();
        if (receiverIp.find("10.2.") != string::npos || receiverIp.find("10.4.") != string::npos) {
    //         // cout << "Not Sending Message of size " << segmentSize << "from : " << GetNodeIP(GetNode(), 1) << " to receiver " << InetSocketAddress::ConvertFrom(_receiverAddress[selectedReceiver][0]).GetIpv4() << endl;
            return;
        }
    }
    // cout << "Node " << GetNodeIP(GetNode(), 1) << " WorkloadApp sending to receiver " << InetSocketAddress::ConvertFrom(_receiverAddress[selectedReceiver][0]).GetIpv4() << " size of: " << segmentSize << " at: " << Simulator::Now().GetNanoSeconds() << endl;
    _connectionPools[selectedReceiver]->SendData(Create<Packet>(segmentSize));
}

void WorkloadApp::ScheduleAtNominal() {
    // Fire on the exact grid `_nominalNext`, dithered by [0, JitterNs). The dither is applied to
    // the grid point and NOT to the inter-message interval: adding it to the interval would make
    // it a random walk, and over 60000 periods a few-ns step would drift senders hundreds of ns
    // apart, destroying the synchronisation the incast depends on.
    Time jitter = _jitterNs > 0 ? Time::FromDouble(m_uniform->GetValue(0.0, _jitterNs), Time::NS)
                                : Time(0);
    Time when = _nominalNext + jitter;
    Time now = Simulator::Now();
    _sendEvent = Simulator::Schedule(when > now ? when - now : Time(0),
                                     &WorkloadApp::ScheduleNextSend, this);
}

void WorkloadApp::ScheduleNextSend() {
    // cout << "Node " << GetNodeIP(GetNode(), 1) << " WorkloadApp sending at: " << Simulator::Now().GetNanoSeconds() << endl;
    Send();
    if (_arrivalProcess == "Periodic") {
        _nominalNext += Seconds(1.0 / _rate);
        ScheduleAtNominal();
    }
    else {
        _sendEvent = Simulator::Schedule(Seconds(m_var->GetValue()), &WorkloadApp::ScheduleNextSend, this);
    }
    // cout << "Node " << GetNodeIP(GetNode(), 1) << " WorkloadApp next event at: " << (Simulator::Now() + Seconds(nextEvent)).GetNanoSeconds() << " Event: " << _sendEvent.GetUid() << endl;
}

