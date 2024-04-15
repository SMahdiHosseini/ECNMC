//
// Created by nal on 18.01.22.
//

#include "HelperMethods.h"

namespace helper_methods {
    vector<string> SplitStr(const string &s, char delim) {
        vector<string> result;
        stringstream ss(s);
        string item;

        while (getline(ss, item, delim)) {
            result.push_back(item);
        }

        return result;
    }

    template<class T>
    string VectorToString(vector<T> vector, string separator) {
        if (vector.empty()) return "";

        stringstream ss;
        ss << vector[0];
        auto aggregate = [&ss, &separator](const T &s) { ss << separator << s; };
        for_each(vector.begin() + 1, vector.end(), aggregate);

        return ss.str();
    }

    uint32_t GetRandomNumber(uint32_t min, uint32_t max) {
        std::random_device rd;
        std::mt19937 mt (rd());
        std::uniform_int_distribution<uint32_t> dist(min, max);
        return dist(mt);
    }

    uint32_t GetSubDirCount(const string &dirPath) {
        auto dirIter = std::filesystem::directory_iterator(dirPath);
        return count_if(
                begin(dirIter),
                end(dirIter),
                [](auto &entry) { return entry.is_regular_file(); }
        );
    }

    string ComputeQueueSize(const string &linkRate, const vector<string> &linksDelay) { // RTT * link_rate
        regex reg("([0-9]*)ms");
        double minRTT = 0.005; // in sec
        for (const string &delay: linksDelay) {
            minRTT += 2 * stoi(regex_replace(delay, reg, "$1")) * 1e-3;
        }
        return to_string(int(minRTT * (ns3::DataRate(linkRate).GetBitRate() * 0.125))) + "B";
    }

    ns3::Ipv4Address GetNodeIP(ns3::Ptr<ns3::Node> node, uint32_t interface) {
        return node->GetObject<ns3::Ipv4>()->GetAddress(interface, 0).GetLocal();
    }

    bool DoesPolicerLocationMatch(const string &linkLocation, const string &expectedPolicerLocation) {
        if (expectedPolicerLocation == linkLocation) return true;
        if (expectedPolicerLocation == "nc" && linkLocation.find("nc") != std::string::npos) return true;
        return false;
    }

    bool IsPolicerTypePerFlowPolicer(int policerType) {
        return policerType == 1;
    }

    uint8_t Dscp2Tos(uint8_t dscp) {
        return dscp << 2;
    }

    string GetSocketFactory(bool isTCP) {
        return (isTCP == 1) ? "ns3::TcpSocketFactory" : "ns3::UdpSocketFactory";
    }
}

