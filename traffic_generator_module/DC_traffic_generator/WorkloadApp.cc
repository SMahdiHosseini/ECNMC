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
    ;
    return tid;
}

WorkloadApp::WorkloadApp() {
    NS_LOG_FUNCTION (this);
    m_var = CreateObject<ExponentialRandomVariable>();
    m_erv = CreateObject<EmpiricalRandomVariable>();
    m_uniform = CreateObject<UniformRandomVariable>();
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
    ReadWorkloadFile();
    PrepareConnections();
    double nextEventTime = m_var->GetValue();
    _sendEvent = Simulator::Schedule(Seconds(nextEventTime), &WorkloadApp::ScheduleNextSend, this);
}

void WorkloadApp::PrepareConnections() {
    NS_LOG_FUNCTION (this);

    for (auto &addresses : _receiverAddress) {
        cout << "Connection Pool of: Sender Address: " << GetNode()->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal() << " Receiver Address: " << InetSocketAddress::ConvertFrom(addresses[0]).GetIpv4() << endl;
        ConnectionPool connectionPool(addresses[0], _protocol, GetNode());
        connectionPool.CreateSockets(addresses);
        _connectionPools.push_back(connectionPool);
    }
}

void WorkloadApp::StopApplication() {
    NS_LOG_FUNCTION (this);
    for (auto &connectionPool : _connectionPools) {
        connectionPool.CloseConnections();
    }
    cout << "Node " << GetNodeIP(GetNode(), 1) << " Connection Pools closed at " << Simulator::Now().GetNanoSeconds() << endl;
    Simulator::Cancel (_sendEvent);
}

void WorkloadApp::Send() {
    NS_LOG_FUNCTION(this);
    uint32_t segmentSize = m_erv->GetValue();
    // segmentSize *= 1442; // for DCTCP workload
    uint32_t selectedReceiver = m_uniform->GetInteger(0, _receiversNumber - 1);
    _connectionPools[selectedReceiver].SendData(Create<Packet>(segmentSize));
}

void WorkloadApp::ScheduleNextSend() {
    Send();
    double nextEvent = m_var->GetValue();
    _sendEvent = Simulator::Schedule(Seconds(nextEvent), &WorkloadApp::ScheduleNextSend, this);
}

