//
// Created by nal on 24.04.25
//
#include "NodeAppsHandler.h"

NS_LOG_COMPONENT_DEFINE ("NodeAppsHandler");

NS_OBJECT_ENSURE_REGISTERED (NodeAppsHandler);

TypeId NodeAppsHandler::GetTypeId() {

    static TypeId tid = TypeId ("ns3::NodeAppsHandler")
            .SetParent<Application> ()
            .SetGroupName("Applications")
            .AddConstructor<NodeAppsHandler> ()
            .AddAttribute("ConnectionPoolSize",
                           "The number of connections to be created",
                           UintegerValue(1),
                           MakeUintegerAccessor(&NodeAppsHandler::_connectionPoolSize),
                           MakeUintegerChecker<uint32_t>())
            .AddAttribute ("Protocol", "the name of the protocol to use to send traffic by the applications",
                           StringValue ("ns3::TcpSocketFactory"),
                           MakeStringAccessor (&NodeAppsHandler::_protocol),
                           MakeStringChecker())
            .AddAttribute("Rate", "The rate of the Poisson process (request per second)",
                          DoubleValue(5000.0),
                          MakeDoubleAccessor(&NodeAppsHandler::_rate),
                          MakeDoubleChecker<double>())
            .AddAttribute("WorkloadFile", "The workload file to be used",
                          StringValue("scratch/ECNMC/DCWorkloads/Google_AllRPC.txt"),
                          MakeStringAccessor(&NodeAppsHandler::workloadFile),
                          MakeStringChecker())
    ;
    return tid;
}

NodeAppsHandler::NodeAppsHandler() {
    NS_LOG_FUNCTION (this);
    m_var = CreateObject<ExponentialRandomVariable>();
    m_erv = CreateObject<EmpiricalRandomVariable>();
}

NodeAppsHandler::~NodeAppsHandler() {
    NS_LOG_FUNCTION (this);
    Simulator::Cancel(_sendEvent);
    DoDispose();
}

void NodeAppsHandler::ReadWorkloadFile() {
    NS_LOG_FUNCTION (this);
    std::ifstream file(workloadFile);
    if (!file.is_open()) {
        NS_FATAL_ERROR("Could not open workload file: " << workloadFile);
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

void NodeAppsHandler::addReceiverAddress(Address address) {
    NS_LOG_FUNCTION (this);
    _receiverAddress.push_back(address);
}

void NodeAppsHandler::DoDispose() {
    NS_LOG_FUNCTION (this);

    Application::DoDispose();
}

void NodeAppsHandler::StartApplication() {
    NS_LOG_FUNCTION(this);
    m_var->SetAttribute("Mean", DoubleValue(1/_rate));
    ReadWorkloadFile();
    PrepareSockets();
}

void NodeAppsHandler::PrepareSockets() {
    NS_LOG_FUNCTION (this);

    // for (auto &address : _receiverAddress) {
    //     NS_LOG_INFO("Preparing connections for receiver address: " << Ipv4Address::ConvertFrom(address));
    //     for (int i = 0; i < _connectionPoolSize; ++i) {
    //         NS_LOG_INFO("Creating connection " << i + 1 << " for receiver address: " << Ipv4Address::ConvertFrom(address));
        
    //         Ptr<Socket> socket = Socket::CreateSocket(GetNode(), TypeId::LookupByName(_protocol));
    //         if (socket->Bind () == -1) {
    //             NS_FATAL_ERROR ("Failed to bind socket");
    //         }
    //         socket->Connect(address);
    //         socket->SetAllowBroadcast (true);
    //         socket->SetRecvCallback (MakeNullCallback<void, Ptr<Socket> > ());
    //         if(_protocol == "ns3::TcpSocketFactory") {
    //             Ptr<TcpSocketBase> tcpSocket = _socket->GetObject<TcpSocketBase>();
    //             tcpSocket->SetPacingStatus(0);
    //         }
    //         _connectionPool[address].push_back(socket);
    //     }
    // }
    double nextEventTime = m_var->GetValue();
    _sendEvent = Simulator::Schedule(Seconds(nextEventTime), &NodeAppsHandler::ScheduleNextSend, this);
}

void NodeAppsHandler::StopApplication() {
    NS_LOG_FUNCTION (this);

    // _socket->Dispose();
    Simulator::Cancel (_sendEvent);
    DoDispose();
}

void NodeAppsHandler::Send() {
    NS_LOG_FUNCTION(this);
    uint32_t segmentSize = m_erv->GetValue();
    // segmentSize *= 1442; // for DCTCP workload
    cout << Simulator::Now().GetNanoSeconds() << "," << segmentSize << endl;
    // if ((_socket->Send (p)) < 0) {
    //     NS_LOG_INFO ("Error while sending " << segmentSize << " bytes to "
    //                                         << Ipv4Address::ConvertFrom (_receiverAddress));
    // }
}

void NodeAppsHandler::ScheduleNextSend() {
    Send();
    double nextEvent = m_var->GetValue();
    _sendEvent = Simulator::Schedule(Seconds(nextEvent), &NodeAppsHandler::ScheduleNextSend, this);
}

