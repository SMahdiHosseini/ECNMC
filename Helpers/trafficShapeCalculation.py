import os
import dask.dataframe as dd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

def read_csv_files_dask(directory):
    # set the header of the csv files
    header = ['seqNb', 'timestamp', 'payload_size']
    # Get all csv files in the directory
    csv_files = glob(os.path.join(directory, '*.csv'))
    # Read all csv files into a Dask DataFrame
    ddf = dd.read_csv(csv_files)
    # Set the column names
    ddf.columns = header
    return ddf

def compute_traffic_over_time_dask(ddf, path):
    # ddf['timestamp'] = dd.to_datetime(ddf['timestamp'], unit='s')
    # ddf = ddf[ddf['timestamp'] < '1970-01-01 00:00:01']
    ddf = ddf[ddf['timestamp'] < 100]
    ddf = ddf.sort_values(by='timestamp')
    # convert to pandas dataframe
    ddf = ddf.compute()
    ddf = ddf.reset_index(drop=True)
    # set seqNb 
    ddf['seqNb'] = ddf.index
    ddf.set_index('seqNb', inplace=True)
    ddf.to_csv('traffic_over_time_{}.csv'.format(path))

def read_data_ready(path):
    if path == -1:
        ddf_t_0 = pd.read_csv('traffic_over_time_{}.csv'.format(0)).drop(['seqNb'], axis=1)
        ddf_t_1 = pd.read_csv('traffic_over_time_{}.csv'.format(1)).drop(['seqNb'], axis=1)
        ddf_t_2 = pd.read_csv('traffic_over_time_{}.csv'.format(2)).drop(['seqNb'], axis=1)
        ddf_t_3 = pd.read_csv('traffic_over_time_{}.csv'.format(2)).drop(['seqNb'], axis=1)

        
    if path != -1:
        if path == 2 or path == 3:
            ddf = pd.read_csv('traffic_over_time_{}.csv'.format(2))
        else:
            ddf = pd.read_csv('traffic_over_time_{}.csv'.format(path))

    if path == -1:
        ddf = pd.merge(ddf_t_0, ddf_t_1, on='timestamp', how='outer')
        ddf['payload_size'] = ddf[['payload_size_x', 'payload_size_y']].apply(lambda x: x['payload_size_x'] if pd.isna(x['payload_size_y']) else (x['payload_size_y'] if pd.isna(x['payload_size_x']) else x['payload_size_x'] + x['payload_size_y']), axis=1)
        ddf = ddf.drop(['payload_size_x', 'payload_size_y'], axis=1)

        
        ddf = pd.merge(ddf, ddf_t_2, on='timestamp', how='outer')
        ddf['payload_size'] = ddf[['payload_size_x', 'payload_size_y']].apply(lambda x: x['payload_size_x'] if pd.isna(x['payload_size_y']) else (x['payload_size_y'] if pd.isna(x['payload_size_x']) else x['payload_size_x'] + x['payload_size_y']), axis=1)
        ddf = ddf.drop(['payload_size_x', 'payload_size_y',], axis=1)


        ddf = pd.merge(ddf, ddf_t_3, on='timestamp', how='outer')
        ddf['payload_size'] = ddf[['payload_size_x', 'payload_size_y']].apply(lambda x: x['payload_size_x'] if pd.isna(x['payload_size_y']) else (x['payload_size_y'] if pd.isna(x['payload_size_x']) else x['payload_size_x'] + x['payload_size_y']), axis=1)
        ddf = ddf.drop(['payload_size_x', 'payload_size_y'], axis=1)


    ddf['timestamp'] = pd.to_datetime(ddf['timestamp'])
    ddf = ddf.set_index('timestamp')

    # Resample data to 0.05 second intervals and sum the payload sizes
    traffic_over_time = ddf.resample('0.02s').sum()
    # # Convert payload size to Megabits (Mb)
    traffic_over_time['payload_size'] = traffic_over_time['payload_size'] * 8 / (1024 * 1024) / 0.02
    
    # creat a new column for the time and set to seconds
    traffic_over_time['time'] = traffic_over_time.index
    traffic_over_time['time'] = traffic_over_time['time'].dt.time
    traffic_over_time['time'] = traffic_over_time['time'].apply(lambda x: x.hour*3600 + x.minute*60 + x.second)
    traffic_over_time['time'] = traffic_over_time['time'] - traffic_over_time['time'][0]

    return traffic_over_time

def plot_traffic(traffic_over_time):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    plt.figure(figsize=(10, 6))
    for i in range(len(traffic_over_time)):
        # also show the average line for each traffic
        avg = np.median(traffic_over_time[i]['payload_size'])
        plt.axhline(y=avg, color=colors[i], linestyle='--', label='Median of Traffic {}'.format(i))
        plt.plot(np.array(traffic_over_time[i]['time']), np.array(traffic_over_time[i]['payload_size']), label='Traffic {}'.format(i), color=colors[i])

    # avg = np.median(traffic_over_time[-1]['payload_size'])
    # plt.axhline(y=avg, color=colors[-1], linestyle='--', label='Median of Traffic {}'.format("All"))
    # plt.plot(np.array(traffic_over_time[-1]['time']), np.array(traffic_over_time[-1]['payload_size']), label='Traffic {}'.format(i), color=colors[-1])

    plt.xlabel('Time')
    plt.ylabel('Traffic (Mbps)')
    plt.title('Network Traffic Over Time')
    plt.grid(True)
    plt.legend()
    plt.savefig('traffic_over_time.png')

def main(directories):
    for i in range(len(directories)):
        ddf = read_csv_files_dask(directory=directories[i])
        compute_traffic_over_time_dask(ddf, i)

    traffics_over_time = []
    for i in range(len(directories)):
        traffics_over_time.append(read_data_ready(i))
    # traffics_over_time.append(read_data_ready(-1))
    plot_traffic(traffics_over_time)

# Directory containing the CSV files
paths = [0]
# directories = ['/home/mahdi/Documents/NAL/Data/chicago_2010_traffic_10min_2paths/path{}/TCP'.format(i) for i in paths]
directories = ['/media/experiments/chicago_2010_traffic_10min_2paths/path{}/TCP'.format(i) for i in paths]
# directories = ['/media/experiments/flow_csv_files_2009_new/path_group_1/TCP']

if __name__ == "__main__":
    main(directories)