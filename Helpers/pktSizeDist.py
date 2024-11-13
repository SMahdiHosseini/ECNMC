import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np

def calculate_probability_greater_than(threshold):
    # Find the rows where packet size is greater than the threshold
    greater_than_threshold = cdf_data[cdf_data['packet_size'] > threshold]
    
    if greater_than_threshold.empty:
        return 0.0  # If no packet is greater than the threshold, probability is 0

    # Calculate the probability as 1 - CDF value at the first packet greater than the threshold
    probability = 1 - greater_than_threshold['cdf'].iloc[0]
    return probability

# Path to the directory containing your CSV files
directories = ['/media/experiments/chicago_2010_traffic_10min_2paths/path0/TCP/', '/media/experiments/chicago_2010_traffic_10min_2paths/path1/TCP/']

# Use glob to find all CSV files in the directory
all_files = []
for directory in directories:
    all_files += glob.glob(directory + "*.csv")


# List to store packet sizes from all files
all_packet_sizes = []

# Loop through each file and append packet sizes to the list
for file in all_files:
    df = pd.read_csv(file)
    # Assuming packet sizes are on 3rd column
    df.iloc[:, 2] = df.iloc[:, 2].apply(lambda x: x + 58)
    all_packet_sizes.extend(df.iloc[:, 2].tolist())

plt.figure(figsize=(10, 6))
plt.hist(all_packet_sizes, bins=50, color='skyblue', edgecolor='black')
plt.xlabel('Packet Size (bytes)')
plt.ylabel('Frequency')
plt.title('Distribution of Packet Sizes')
plt.grid(True)
plt.savefig('packet_size_distribution_hist.png')
plt.close()

all_packet_sizes = np.sort(all_packet_sizes)
cdf = np.arange(1, len(all_packet_sizes) + 1) / len(all_packet_sizes)

# Save the packet sizes and their CDF values to a CSV file
cdf_data = pd.DataFrame({'packet_size': all_packet_sizes, 'cdf': cdf})
cdf_data = cdf_data.groupby('packet_size')['cdf'].max().reset_index()
cdf_data.to_csv('packet_size_cdf.csv', index=False)

# print the mean and median
print('Mean:', np.mean(all_packet_sizes))
print('Median:', np.median(all_packet_sizes))
# Plotting the CDF
plt.figure(figsize=(10, 6))
plt.plot(cdf_data['packet_size'], cdf_data['cdf'], marker='.', linestyle='none', color='skyblue')
plt.xlabel('Packet Size (bytes)')
plt.ylabel('CDF')
plt.title('Cumulative Distribution Function (CDF) of Packet Sizes')
plt.grid(True)
plt.savefig('packet_size_distribution.png')
plt.close()

print('Probability of packet size greater than 1500 bytes:', calculate_probability_greater_than(1499))