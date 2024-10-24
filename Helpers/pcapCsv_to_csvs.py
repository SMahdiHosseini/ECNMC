import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import random
import os

# Function to read TCP packet data from CSV and calculate traffic rate
def plot_traffic_rate_by_random_groups(csv_file, max_rows=100000, num_groups=3, max_flows=2000):
    # Initialize a dictionary to store packets by flow
    flows = defaultdict(list)

    # Read the CSV file
    with open(csv_file, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row

        # Row counter
        row_count = 0

        for row in csv_reader:
            # if row_count >= max_rows:
            #     break  # Stop after reading the maximum number of rows

            # Extract relevant data from each row
            relative_time = float(row[0]) / 10 # Relative time (in seconds)
            # relative_time = float(row[0]) # Relative time (in seconds)
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

    # Merge flows together to ensure no more than max_flows (2000)
    # merged_flows = merge_flows_randomly(sorted_flows, max_flows)
    merged_flows = sorted_flows

    # Split the merged flows into groups
    group_size = len(merged_flows) // num_groups
    flow_groups = [merged_flows[i:i + group_size] for i in range(0, len(merged_flows), group_size)]

    # filter the flows with less than 150 packets or that last less than 1.3 seconds, not the group
    pruned_flow_groups = []
    for group_index, group in enumerate(flow_groups[:num_groups]):
        pruned_flows = []
        for i, (flow_id, packets) in enumerate(group):
            # Sort packets by their relative time before writing
            sorted_packets = sorted(packets, key=lambda packet: packet[0])  # Sort by relative_time (index 0)
            if sorted_packets[-1][0] - sorted_packets[0][0] < 1.7:
                continue
            if sorted_packets[0][0] > 0.15:
                continue
            if len(sorted_packets) < 4200:
                continue
            pruned_flows.append((flow_id, sorted_packets))
        pruned_flow_groups.append(pruned_flows)
    flow_groups = pruned_flow_groups

    # Plot traffic rate for each random group and save flows
    for group_index, group in enumerate(flow_groups[:num_groups]):
        # Plot the traffic rate for the group
        plot_group_traffic_rate(group, group_index)

        # Save each merged flow in the group to a separate CSV file in the respective folder
        save_flows_to_csv(group, group_index)

def merge_flows_randomly(sorted_flows, max_flows):
    # If there are fewer flows than the limit, no merging is needed
    if len(sorted_flows) <= max_flows:
        return sorted_flows
    
    # Randomly group flows until we have no more than max_flows
    merged_flows = []
    random.shuffle(sorted_flows)

    # Calculate how many flows should be merged together
    merge_size = len(sorted_flows) // max_flows

    # Merge flows into groups
    i = 0
    while i < len(sorted_flows):
        # Group flows together, ensuring they are merged into no more than max_flows sets
        current_group = sorted_flows[i:i + merge_size]

        # Combine packets from the grouped flows
        merged_packets = []
        for flow_id, packets in current_group:
            merged_packets.extend(packets)

        # Create a new flow with the combined packets
        # For simplicity, we'll use the flow ID of the first flow in the group
        merged_flows.append((current_group[0][0], merged_packets))

        i += merge_size

    return merged_flows[:max_flows]

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
    interval = 0.005
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
    traffic_rate_Mbps /= interval  # Divide by the interval to get the rate per second
    # Prepare the time axis for plotting
    time_axis = np.arange(0, num_intervals) * interval

    # Plot the traffic rate for the group
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, traffic_rate_Mbps, marker='o', linestyle='-', color=np.random.rand(3,))
    plt.title(f'TCP Traffic Rate for Random Group {group_index + 1} (Mbps)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Traffic Rate (Mbps)')
    # plot the average traffic rate
    plt.axhline(y=np.mean(traffic_rate_Mbps), color='r', linestyle='--', label='Average Traffic Rate')
    # plt.xticks(np.arange(0, max_time + 1, step=5))
    plt.grid()
    plt.xlim(0, max_time)
    plt.savefig(f'traffic_rate_group_{group_index + 1}.png')

def save_flows_to_csv(flow_group, group_index):
    # Create a folder for the group if it doesn't exist
    folder_name = f"flow_csv_files_2009_new_new/path_group_{group_index + 1}/TCP"
    os.makedirs(folder_name, exist_ok=True)
    os.makedirs(f"flow_csv_files_2009_new_new/path_group_{group_index + 1}/UDP", exist_ok=True)
    longFlows = 0
    # Save each flow in the group to a separate CSV file
    for i, (flow_id, packets) in enumerate(flow_group):
        file_name = os.path.join(folder_name, f"trace_{i + 1}.csv")

        # Sort packets by their relative time before writing
        sorted_packets = sorted(packets, key=lambda packet: packet[0])  # Sort by relative_time (index 0)

        # continue if the first packet relative time is grater than 3
        # if sorted_packets[0][0] < 1.3:
        # if len(sorted_packets) < 150:
        #     continue
        longFlows += 1
        with open(file_name, mode='w', newline='') as file:
            csv_writer = csv.writer(file)
            # Write each packet in the flow to the CSV
            j = 0
            for packet in sorted_packets:
                relative_time, packet_size = packet
                csv_writer.writerow([j, relative_time, packet_size])
                j += 1

        print(f"Saved merged flow {i + 1} of group {group_index + 1} to {file_name}")
    print(f"long flows in group {group_index + 1} is {longFlows}")
# Specify the input CSV file
csv_file = "equinix-chicago.dirA.20090820-125904.UTC.anon.pcap.csv"  # Replace with the path to your CSV file

# Plot the traffic rate for 3 randomly split groups, merge flows, and save them (read only the first 100,000 rows)
plot_traffic_rate_by_random_groups(csv_file, max_rows=500000, num_groups=1, max_flows=30000)
