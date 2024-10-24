import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import random

# Function to read TCP packet data from CSV and calculate traffic rate
def plot_traffic_rate_by_groups(csv_file, max_rows=100000, num_groups=3):
    # Initialize a dictionary to store packets by flow
    flows = defaultdict(list)

    # Read the CSV file
    with open(csv_file, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row

        # Row counter
        row_count = 0

        for row in csv_reader:
            if row_count >= max_rows:
                break  # Stop after reading the maximum number of rows

            # Extract relevant data from each row
            relative_time = float(row[0])  # Relative time (in seconds)
            src_ip = row[1]                # Source IP
            dst_ip = row[2]                # Destination IP
            src_port = row[3]              # Source port
            dst_port = row[4]              # Destination port
            packet_size = int(row[5])      # Packet size (in bytes)

            # Create a flow identifier based on src/dst IPs and ports
            flow_id = (src_ip, dst_ip, src_port, dst_port)
            
            # Append the relative time and packet size to the respective flow
            flows[flow_id].append((relative_time, packet_size))

            row_count += 1  # Increment the row counter
        
     # Shuffle the flows randomly
    sorted_flows = list(flows.items())
    random.shuffle(sorted_flows)  # Randomize the order of flows

    # Split the flows into groups randomly
    group_size = len(sorted_flows) // num_groups
    flow_groups = [sorted_flows[i:i + group_size] for i in range(0, len(sorted_flows), group_size)]
    # Plot traffic rate for each group
    for group_index, group in enumerate(flow_groups[:num_groups]):
        plot_group_traffic_rate(group, group_index)

def plot_group_traffic_rate(flow_group, group_index):
    # Initialize lists to store total traffic per time interval
    combined_times = []
    combined_packet_sizes = []

    # Process each flow in the group
    for flow_id, packets in flow_group:
        relative_times, packet_sizes = zip(*packets)
        combined_times.extend(relative_times)
        combined_packet_sizes.extend(packet_sizes)

    # Determine the time interval (1 second)
    interval = 1.0
    max_time = max(combined_times)
    num_intervals = int(max_time // interval) + 1
    traffic_rate_Mbps = np.zeros(num_intervals)

    # Accumulate packet sizes in each interval
    for time, size in zip(combined_times, combined_packet_sizes):
        index = int(time // interval)
        if index < num_intervals:
            traffic_rate_Mbps[index] += size * 8  # Convert packet size to bits (1 byte = 8 bits)

    # Convert the traffic rate to Mbps (Megabits per second)
    traffic_rate_Mbps /= (1024 * 1024)  # Convert bits to megabits (1024*1024 = 1 megabit)

    # Prepare the time axis for plotting
    time_axis = np.arange(0, num_intervals) * interval

    # Plot the traffic rate for the group
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, traffic_rate_Mbps, marker='o', linestyle='-', color=np.random.rand(3,))
    plt.title(f'TCP Traffic Rate for Group {group_index + 1} (Mbps)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Traffic Rate (Mbps)')
    plt.xticks(np.arange(0, max_time + 1, step=5))
    plt.grid()
    plt.xlim(0, max_time)
    plt.show()
    plt.savefig(f'traffic_rate_group_{group_index + 1}.png')

# Specify the input CSV file
csv_file = "equinix-nyc.dirA.20190117-125910.UTC.anon.pcap.csv"  # Replace with the path to your CSV file

# Plot the traffic rate for 3 different groups (read only the first 100,000 rows)
plot_traffic_rate_by_groups(csv_file, max_rows=10000000, num_groups=2)


