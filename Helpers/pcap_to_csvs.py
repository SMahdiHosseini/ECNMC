from scapy.all import rdpcap
import pandas as pd
import os

# Function to extract the flow key (src_ip, dst_ip, src_port, dst_port)
def get_flow_key(packet):
    if packet.haslayer('IP') and packet.haslayer('TCP'):
        src_ip = packet['IP'].src
        dst_ip = packet['IP'].dst
        src_port = packet['TCP'].sport
        dst_port = packet['TCP'].dport
        return (src_ip, dst_ip, src_port, dst_port)
    return None

# Read pcap file
pcap_file = "equinix-chicago.dirA.20101029-125904.UTC.anon.pcap"  # Replace with the path to your .pcap file
packets = rdpcap(pcap_file, count=100)  # Read first 1000 packets

# Create a dictionary to hold data for each flow
flows = {}

# Process each packet
for i, packet in enumerate(packets):
    flow_key = get_flow_key(packet)
    if flow_key:
        packet_size = packet['IP'].len - 58
        timestamp = packet.time
        packet.show()
        print(packet.time)
        break
        
        
#         # Create a new flow if it's not already in the dictionary
#         if flow_key not in flows:
#             # print(f"New flow: {len(flows) + 1}")
#             flows[flow_key] = []
        
#         # Append the packet information (packet number, size, timestamp) to the flow
#         flows[flow_key].append([i + 1, packet_size, timestamp])

# # Save each flow to a separate CSV file
# output_dir = "flow_csv_files"
# os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

# i = 0
# for flow_key, flow_data in flows.items():
#     src_ip, dst_ip, src_port, dst_port = flow_key
#     csv_filename = f"trace_{i}.csv"
#     csv_filepath = os.path.join(output_dir, csv_filename)
    
#     # Convert to DataFrame and save as CSV
#     df = pd.DataFrame(flow_data, columns=['Packet Number', 'Packet Size', 'Timestamp'])
#     df.to_csv(csv_filepath, index=False)
#     i += 1

# print(f"CSV files saved to {output_dir}")
