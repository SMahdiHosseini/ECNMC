#ifndef PACKETCDF_H
#define PACKETCDF_H

#include <ostream>
#include <unordered_map>
#include "ns3/core-module.h"

using namespace ns3;
using namespace std;

class PacketCDF {
public:
    void SetCDFFile(const std::string& filename) {
        fileName = filename;
    }

    void loadCDFData() {
        std::ifstream file(fileName);
        std::string line;

        // Skip the header
        std::getline(file, line);

        // Read each line of the CSV file
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string packet_size_str, cdf_str;
            if (std::getline(ss, packet_size_str, ',') && std::getline(ss, cdf_str, ',')) {
                packet_cdf[std::stoi(packet_size_str)] = std::stod(cdf_str);
            }
        }
    }
    // Function to add a new packet size and update the CDF
    void addPacket(uint32_t packet_size) {
        // Increment the count for the packet size
        packet_count[packet_size]++;
        total_packets++;

        // Update the cumulative distribution function (CDF)
        updateCDF();
    }

    // Function to calculate the probability of packet size being greater than a threshold
    double calculateProbabilityGreaterThan(uint32_t threshold) {
        // Iterate through the map to find the first entry greater than the threshold
        auto it = packet_cdf.upper_bound(threshold);
        if (it == packet_cdf.end()) {
            // std::cout << "Error: Packet size CDF not updated correctly " << it->first << " " << threshold << std::endl;
            return 0.0;  // No packet size greater than threshold
        }

        return 1.0 - it->second;  // Probability is 1 - CDF for the first packet size > threshold
    }

    // Print the CDF for debugging or verification
    void printCDF() const {
        std::cout << "packet_size,cdf" << std::endl;
        for (const auto& entry : packet_cdf) {
            std::cout << entry.first << "," << entry.second << std::endl;
        }
    }

    // Save the CDF data to a CSV file
    void saveCDFData() const {
        std::ofstream
        file(fileName);
        file << "packet_size,cdf" << std::endl;
        for (const auto& entry : packet_cdf) {
            file << entry.first << "," << entry.second << std::endl;
        }
        file.close();
    }

private:
    std::map<uint32_t, uint32_t> packet_count;  // Stores count of each packet size
    std::map<uint32_t, double> packet_cdf; // Stores CDF values for each packet size
    uint32_t total_packets = 0;               // Total number of packets observed
    string fileName;                     // File name for saving CDF data

    // Function to update the CDF values after adding a packet
    void updateCDF() {
        double cumulative_probability = 0.0;

        // Iterate over packet sizes in ascending order to compute the cumulative probability
        for (const auto& entry : packet_count) {
            uint32_t packet_size = entry.first;
            uint32_t count = entry.second;

            // Update cumulative probability for this packet size
            cumulative_probability += static_cast<double>(count) / total_packets;
            packet_cdf[packet_size] = cumulative_probability;
        }

        // Ensure the last CDF value is exactly 1.0
        if (!packet_cdf.empty()) {
            packet_cdf.rbegin()->second = 1.0;
        }
    }
};
#endif //PACKETCDF_H