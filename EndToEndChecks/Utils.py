import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import anderson
from scipy.stats import f_oneway, kruskal

import csv
from collections import defaultdict
estimation_gain = 0.0625
init_alpha = 1

class PacketCDF:
    def __init__(self):
        self.packet_count = defaultdict(int)  # Stores count of each packet size
        self.packet_cdf = {}  # Stores CDF values for each packet size
        self.total_packets = 0  # Total number of packets observed

    def load_cdf_data(self, filename):
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header
            for row in reader:
                if len(row) >= 2:
                    packet_size = int(row[0])
                    cdf_value = float(row[1])
                    self.packet_cdf[packet_size] = cdf_value
    
    def add_packet(self, packet_size):
        """ Adds a new packet size and updates the CDF."""
        self.packet_count[packet_size] += 1
        self.total_packets += 1
        self._update_cdf()

    def calculate_probability_greater_than(self, threshold):
        """ Computes the probability of a packet size being greater than the given threshold."""
        for size in sorted(self.packet_cdf.keys()):
            if size > threshold:
                return 1.0 - self.packet_cdf[size]
        return 0.0

    def calculate_probability_less_equal_than(self, threshold):
        """ Computes the probability of a packet size being less than or equal to the given threshold."""
        for size in sorted(self.packet_cdf.keys())[::-1]:
            if size <= threshold:
                return self.packet_cdf[size]
        return 0.0
    
    def compute_conditional_probability(self, A, B, num_samples=100000):
        # Generate random samples from X using inverse transform sampling
        X_samples = np.interp(np.random.rand(num_samples), list(self.packet_cdf.values()), list(self.packet_cdf.keys())).astype(int)
        Y_samples = np.interp(np.random.rand(num_samples), list(self.packet_cdf.values()), list(self.packet_cdf.keys())).astype(int)
        
        # Filter samples where X >= B
        valid_X = X_samples[X_samples >= B]
        valid_Y = Y_samples[:len(valid_X)]  # Match the sample size
        
        # Compute probability P(Y > X + A | X >= B)
        count_Y_greater = np.sum(valid_Y > (valid_X + A))
        probability = count_Y_greater / len(valid_X) if len(valid_X) > 0 else 0
        
        return probability

    def print_cdf(self):
        """ Prints the CDF values for debugging or verification."""
        print("packet_size,cdf")
        for size, cdf in self.packet_cdf.items():
            print(f"{size},{cdf}")

    def _update_cdf(self):
        """ Updates the cumulative distribution function (CDF) after adding a packet."""
        cumulative_probability = 0.0
        sorted_sizes = sorted(self.packet_count.keys())
        
        for size in sorted_sizes:
            count = self.packet_count[size]
            cumulative_probability += count / self.total_packets
            self.packet_cdf[size] = cumulative_probability
        
        # Ensure the last CDF value is exactly 1.0
        if self.packet_cdf:
            last_key = sorted_sizes[-1]
            self.packet_cdf[last_key] = 1.0
            
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (40, 20)
plt.rcParams.update({
    "lines.color": "black",
    "patch.edgecolor": "black",
    "text.color": "black",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "grid.color": "gray",
    "figure.facecolor": "white",
    "figure.edgecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "font.size": 30,
    "xtick.labelsize":30,
    "ytick.labelsize":30,
    "lines.linewidth":1.,
    "legend.fontsize": 10,
    })

def calculate_drop_rate(__ns3_path, steadyStart, steadyEnd, rate, segments, checkColumn, projectColumn, experiment, results_folder):
    loss_sum = 0
    counts = 0
    for segment in segments:
        file_paths = glob.glob('{}/scratch/{}/{}/{}/*_{}.csv'.format(__ns3_path, results_folder, rate, experiment, segment))
        for file_path in file_paths:
            df_name = file_path.split('/')[-1].split('_')[0]
            if 'C' in df_name:
                continue
            df = pd.read_csv(file_path)
            df = df[df[projectColumn] > steadyStart * 1000000000]
            df = df[df[projectColumn] < steadyEnd * 1000000000]
            # calculate the drop rate by dividing the some of the payload of dropped packets by the total payload of the sent packets
            # total_payload = df['PayloadSize'].sum()
            # dropped_payload = df[df[checkColumn] == 0]['PayloadSize'].sum()
            # if total_payload == 0:
            #     swtiches_dropRates[df_name] = 0
            # else:
            #     swtiches_dropRates[df_name] = dropped_payload / total_payload
            loss_sum += len(df[df[checkColumn] == 0]) / len(df)
            counts += 1
    return loss_sum / counts
    # if len([value for value in swtiches_dropRates.values() if value != 0]) == 0:
    #     return 0
    # return sum([value for value in swtiches_dropRates.values() if value != 0]) / len([value for value in swtiches_dropRates.values() if value != 0])

def calculate_avgDrop_rate_offline(endToEnd_dfs, paths):
    return 1 - sum([endToEnd_dfs[flow]['successProbMean'][p] * endToEnd_dfs[flow]['sampleSize'][p] for p in range(len(paths)) for flow in endToEnd_dfs.keys()]) / sum([endToEnd_dfs[flow]['sampleSize'][p] for p in range(len(paths)) for flow in endToEnd_dfs.keys()])

def calculate_drop_rate_online(endToEnd_dfs, paths):
    loss_sum = 0
    counts = 0
    for flow in endToEnd_dfs.keys():
        for p in range(len(paths)):
            loss_sum += endToEnd_dfs[flow]['sentPacketsOnLink'][p] - endToEnd_dfs[flow]['receivedPackets'][p]
            counts += endToEnd_dfs[flow]['sentPacketsOnLink'][p]
    return loss_sum / counts

def read_burst_samples(__ns3_path, rate, segment, experiment, results_folder):
    file_paths = glob.glob('{}/scratch/{}/{}/{}/*_{}.csv'.format(__ns3_path, results_folder, rate, experiment, segment))
    dfs = {}
    for file_path in file_paths:
        df_name = file_path.split('/')[-1].split('_')[0]
        df = pd.read_csv(file_path)
        df = df.rename(columns={' isHotThroughputUtilization': 'isHot'})
        dfs[df_name] = df
    return dfs

def read_queuingDelay(__ns3_path, rate, segment, experiment, results_folder, linkDelay, incomingLinkRate, outgoingLinkRate):
    file_paths = glob.glob('{}/scratch/{}/{}/{}/*_{}.csv'.format(__ns3_path, results_folder, rate, experiment, segment))
    dfs = {}
    for file_path in file_paths:
        df_name = file_path.split('/')[-1].split('_')[0]
        full_df = pd.read_csv(file_path)
        dfs[df_name] = {}
        # first rename the columns Path to path, SentTime to sentTime, ReceiveTime to receivedTime
        full_df = full_df.rename(columns={'Path': 'path', 'SentTime': 'sentTime', 'ReceiveTime': 'receivedTime'})
        for path in full_df['path'].unique():
            # sort data by 'sentTime' column
            df = full_df[full_df['path'] == path]
            df = df.sort_values(by='sentTime').reset_index(drop=True)
            # add 54 bytes to the packet size to account for the ethernet header
            df['PayloadSize'] = df['PayloadSize'] + 54
            # add a nre columns "enqueueTime" which is the packet sentTime + linkDelay + (packetSize * 8) / incomingLinkRate
            df['enqueueTime'] = df['sentTime'] + linkDelay + (df['PayloadSize'] * 8) / incomingLinkRate
            # add a new columns "dequeueTime" which is the packet receivedTime - linkDelay - (packetSize * 8) / outgoingLinkRate
            df['dequeueTime'] = df['receivedTime'] - linkDelay - (df['PayloadSize'] * 8) / outgoingLinkRate
            # add a new columns "queuelength" which is the (dequeueTime - enqueueTime) * outgoingLinkRate
            df['queuelength'] = (df['dequeueTime'] - df['enqueueTime']) * outgoingLinkRate / 8
            # remove all columns other than path, enqueueTime, dequeueTime, queuelength, size
            df = df[['path', 'enqueueTime', 'dequeueTime', 'queuelength', 'PayloadSize']]
            dfs[df_name]['A' + str(path)] = df
    return dfs

def read_lossProb(__ns3_path, rate, segment, experiment, results_folder):
    file_paths = glob.glob('{}/scratch/{}/{}/{}/*_{}.csv'.format(__ns3_path, results_folder, rate, experiment, segment))
    dfs = {}
    for file_path in file_paths:
        df_name = file_path.split('/')[-1].split('_')[0]
        full_df = pd.read_csv(file_path)
        dfs[df_name] = {}
        dfs[df_name]['timeAvgSuccessProb'] = {}
        # remove all columns other than path, sentTime, receivedTime
        # first rename the columns Path to path, SentTime to sentTime, ReceiveTime to receivedTime
        full_df = full_df.rename(columns={'Path': 'path', 'SentTime': 'sentTime', 'ReceiveTime': 'receivedTime'})
        full_df = full_df[['path', 'sentTime', 'receivedTime']]
        for path in full_df['path'].unique():
            # sort data by 'sentTime' column
            df = full_df[full_df['path'] == path]
            df = df.sort_values(by='sentTime').reset_index(drop=True)
            df['lossProb'] = 0
            df.loc[df['receivedTime'] < 0, 'lossProb'] = 1
            df['time_diff'] = df['sentTime'].shift(-1) - df['sentTime']
            df['time_diff'] = df['time_diff'].fillna(0)
            integral_lossProb = (df['lossProb'] * df['time_diff']).sum()
            total_duration = df['sentTime'].iloc[-1] - df['sentTime'].iloc[0]
            time_average_lossProb = integral_lossProb / total_duration
            dfs[df_name]['timeAvgSuccessProb']['A' + str(path)] = 1.0 - time_average_lossProb
            print(dfs[df_name]['timeAvgSuccessProb']['A' + str(path)])
    return dfs
            
def plot_queueSize_time(__ns3_path, rate, segment, experiment, results_folder):
    file_paths = glob.glob('{}/scratch/{}/{}/{}/*_{}.csv'.format(__ns3_path, results_folder, rate, experiment, segment))
    dfs = {}
    for file_path in file_paths:
        df_name = file_path.split('/')[-1].split('_')[0]
        df = pd.read_csv(file_path)
        print(df)

def read_online_computations(__ns3_path, rate, segment, experiment, results_folder):
    file_paths = glob.glob('{}/scratch/{}/{}/{}/*_{}.csv'.format(__ns3_path, results_folder, rate, experiment, segment))
    dfs = {}
    for file_path in file_paths:
        df_name = file_path.split('/')[-1].split('_')[0]
        df = pd.read_csv(file_path)
        df = df.rename(columns={'sampleDelayMean': 'DelayMean', 'unbiasedSmapleDelayVariance': 'DelayStd'})
        if segment == 'PoissonSampler':
            df = df.loc[:0]
            # change the all columns type to double
            df = df.astype(float)
            df = df.rename(columns={'samplesDropMean': 'successProbMean', 'samplesDropVariance': 'successProbStd'})
            # df = df.rename(columns={'GTDropMean': 'successProbMean', 'samplesDropVariance': 'successProbStd'})
            df['DelayStd'] = np.sqrt(df['DelayStd'])
            df['successProbStd'] = np.sqrt(df['successProbStd'])
            # convert the success probability to loss probability
            df['successProbMean'] = 1 - df['successProbMean']
            # convert df to a dictionary
            dfs[df_name] = df.iloc[0].to_dict()
        else:
            df = df.rename(columns={'UnbiasedGTDropMean': 'enqueueTimeAvgSuccessProb'})
            df['successProbMean'] = df['receivedPackets'] / df['sentPacketsOnLink']
            df['enqueueTimeAvgSuccessProb'] = 1 - df['enqueueTimeAvgSuccessProb']
            dfs[df_name] = df.to_dict()
    return dfs

def calculate_offline_switch_congestionEstimation(full_df_, df_res):
    full_df = full_df_.copy()
    congestionEst = np.zeros(len(full_df))
    congestionEst[0] = init_alpha
    for i in range(1, len(full_df)):
        congestionEst[i] = congestionEst[i-1] * (1 - estimation_gain) + full_df.loc[i, "MarkingProb"] * estimation_gain
    full_df["congestionEst"] = congestionEst
    df_res['congestionEstMean'] = full_df['congestionEst'].mean()
    df_res['congestionEstStd'] = full_df['congestionEst'].std()
    return df_res

def calculate_offline_E2E_markingFraction(full_df_, paths, df_res):
    full_df = full_df_.copy()
    full_df['MarkingProb'] = full_df.apply(lambda x: x['MarkingProb'] if x['BytesAcked'] != 0 else 1, axis=1)
    for path in paths:
        full_df = full_df.sort_values(by='Time').reset_index(drop=True)
        time = full_df['Time'].values
        values = full_df['MarkingProb'].values
        time_average_right = np.sum(values[:-1] * np.diff(time)) / (time[-1] - time[0])
        df_res['enqueueTimeAvgNonMarkingFractionProb'][path] = 1 - (time_average_right)
    return df_res

def calculate_offline_E2E_congestionEstimation(full_df_, paths, df_res):
    full_df = full_df_.copy()
    for path in paths:
        congestionEst = np.zeros(len(full_df))
        congestionEst[0] = init_alpha
        for i in range(1, len(full_df)):
            congestionEst[i] = congestionEst[i-1] * (1 - estimation_gain) + full_df.loc[i, "MarkingProb"] * estimation_gain
        full_df["congestionEst"] = congestionEst
        full_df = full_df.sort_values(by='Time').reset_index(drop=True)
        time = full_df['Time'].values
        values = full_df['congestionEst'].values
        time_average_right = np.sum(values[:-1] * np.diff(time)) / (time[-1] - time[0])
        df_res['congestionEst'][path] = time_average_right
    return df_res

def calculate_offline_E2E_workload(full_df, df_res, steadyStart, steadyEnd):
    full_df_ = full_df.copy()
    for path in full_df_['Path'].unique():
        df = full_df_[full_df_['Path'] == path]
        df_res['first'][path] = df['SentTime'].iloc[0]
        df_res['last'][path] = df['SentTime'].iloc[-1]
        df_res['workload'][path] = df['PayloadSize'].sum() * 8 / (steadyEnd - steadyStart)
        df = None
    full_df_ = None
    return df_res

def calculate_offline_E2E_lossRates(__ns3_path, full_df, df_res, checkColumn, linksRate, swtichDstREDQueueDiscMaxSize):
    df_res['successProb'] = {}
    for var in ['event', 'probability']:
        for method in ['rightCont_timeAvg', 'leftCont_timeAvg', 'linearInterp_timeAvg']:
            df_res['successProb'][var + '_' + method] = {}

    packets_cfd = PacketCDF()
    packets_cfd.load_cdf_data('{}/scratch/ECNMC/Helpers/packet_size_cdf_singleQueue.csv'.format(__ns3_path))
    full_df_ = full_df[full_df['SentTime'] != -1].copy()
    for path in full_df_['Path'].unique():
        df = full_df_[full_df_['Path'] == path]
        df_res['sampleSize'][path] = len(df)
        df = df.sort_values(by='SentTime').reset_index(drop=True)

        df['nonDropEvent'] = df.apply(lambda x: 1.0 if x[checkColumn] != 0 else 0.0, axis=1)
        df_res['successProbMean'][path] = df['nonDropEvent'].mean()

        time = df['SentTime'].values
        values = df['nonDropEvent'].values

        rightCont_time_average = np.sum(values[:-1] * np.diff(time)) / (time[-1] - time[0])
        df_res['successProb']['event_rightCont_timeAvg'][path] = rightCont_time_average

        leftCont_time_average = np.sum(values[1:] * np.diff(time)) / (time[-1] - time[0])
        df_res['successProb']['event_leftCont_timeAvg'][path] = leftCont_time_average

        linearInterp_time_average = np.sum(((values[:-1] + values[1:]) / 2) * np.diff(time)) / (time[-1] - time[0])
        df_res['successProb']['event_linearInterp_timeAvg'][path] = linearInterp_time_average

        # avg, std = e2e_poisson_sampling(time, values)
        # df_res['successProb']['event_poisson_eventAvg'][path] = (avg, std)

        df['nonDropProb'] = df.apply(lambda x: 1.0 - packets_cfd.calculate_probability_greater_than(max(swtichDstREDQueueDiscMaxSize - (x['Delay'] * linksRate / 8), x['PayloadSize'])) if x[checkColumn] != 0 else 0.0, axis=1)

        time = df['SentTime'].values
        values = df['nonDropProb'].values

        rightCont_time_average = np.sum(values[:-1] * np.diff(time)) / (time[-1] - time[0])
        df_res['successProb']['probability_rightCont_timeAvg'][path] = rightCont_time_average

        leftCont_time_average = np.sum(values[1:] * np.diff(time)) / (time[-1] - time[0])
        df_res['successProb']['probability_leftCont_timeAvg'][path] = leftCont_time_average

        linearInterp_time_average = np.sum(((values[:-1] + values[1:]) / 2) * np.diff(time)) / (time[-1] - time[0])
        df_res['successProb']['probability_linearInterp_timeAvg'][path] = linearInterp_time_average

        # avg, std = e2e_poisson_sampling(time, values)
        # df_res['successProb']['probability_poisson_eventAvg'][path] = (avg, std)

    full_df_ = None
    return df_res

def calculate_offline_markingProbMean_at_receiver(df, swtichDstREDQueueDiscMaxSize, linkRate):
    T = ((swtichDstREDQueueDiscMaxSize * 8) / linkRate) * 0.30
    df['SentTime'] = df['SentTime'] - df['SentTime'].iloc[0]
    ecn_df = pd.DataFrame(columns=['time', 'F'])

    group_id = 0
    start_time = df.iloc[0]["SentTime"]
    end_time = df.iloc[0]["SentTime"]
    total = 0
    marked = 0
    for i in range(len(df)):
        if df.iloc[i]["SentTime"] - start_time <= T:
            total += 1
            end_time = df.iloc[i]["SentTime"]
            if df.iloc[i]["ECN"] == 1:
                marked += 1
        else:
            ecn_df = pd.concat([pd.DataFrame([[end_time, marked / total]], columns=ecn_df.columns), ecn_df], ignore_index=True)
            total = 0
            marked = 0
            start_time = df.iloc[i]["SentTime"]
            end_time = df.iloc[i]["SentTime"]
    ecn_df = ecn_df.sort_values(by='time').reset_index(drop=True)

    temp = ecn_df.iloc[0]['time']
    ecn_df['InterArrivalTime'] = ecn_df['time'].diff().fillna(temp)
    ecn_df['F'] = ecn_df['F'] * ecn_df['InterArrivalTime']
    return 1 - (ecn_df['F'].sum() / ecn_df['InterArrivalTime'].sum())

def e2e_poisson_sampling(time, values):
    rate = 12 * 1e-6
    duration = time[-1] - time[0]
    bound = 1080

    inter_arrival_times = np.random.exponential(scale=1/rate, size=int(duration * rate))
    poisson_times = 3 * 1e8 + np.cumsum(inter_arrival_times)
    
    poisson_times = poisson_times[poisson_times <= time[-1]]

    selected = []
    for t in poisson_times:
        idx = np.searchsorted(time, t)
        candidates = []
        if idx > 0:
            candidates.append(idx - 1)
        if idx < len(time):
            candidates.append(idx)

        # Find the closest valid one
        closest = None
        min_diff = float('inf')
        for i in candidates:
            diff = abs(time[i] - t)
            if diff <= bound and diff < min_diff:
                closest = i
                min_diff = diff

        if closest is not None:
            selected.append(values[closest])
            # if time[closest] <= t:
            #     selected.append(values[closest] + sizes[closest] - (t - time[closest]))
            # else:
            #     selected.append(values[closest] + (time[closest] - t))

    if selected:
        avg = np.mean(selected)
        return avg, np.std(selected) / np.sqrt(len(selected))
        # print(f"Average Delay from Poisson-sampled SentTimes (within {bound}): {avg} and std/Rn:{np.std(selected) / np.sqrt(len(selected)) + 0.01201685}")
    else:
        print("No matches found within the specified bound.")
        return 0, 0

def calculate_offline_markingProbMean_at_receiver_poisson(df, swtichDstREDQueueDiscMaxSize, linkRate):
    df['SentTime'] = df['SentTime'] - df['SentTime'].iloc[0]
    T = ((swtichDstREDQueueDiscMaxSize * 8) / linkRate) * 0.15
    sample_times = np.cumsum(np.random.exponential(T, int(df['SentTime'].max() / T)))
    markingProbs = []
    for sample_time in sample_times:
        if sample_time > df['SentTime'].max():
            break
        df_sample = df[(df['SentTime'] - sample_time).abs() <= T / 2]
        if len(df_sample) == 0:
            continue
        markingProbs.append(1 - (df_sample['ECN'].sum() / len(df_sample)))
    return np.mean(markingProbs)

def calculate_offline_E2E_markingProb(full_df, df_res, checkColumn, swtichDstREDQueueDiscMaxSize, linkRate, __ns3_path, tsh):
    # timeAvg_methods = ['rightCont_timeAvg', 'leftCont_timeAvg', 'linearInterp_timeAvg']
    # nonMarkingProb_timeAvg_vars = ['event_currentProb', 'event_lastProb']
    df_res['nonMarkingProb'] = {}
    for var in ['event']:
        for method in ['rightCont_timeAvg', 'leftCont_timeAvg', 'linearInterp_timeAvg']:
            df_res['nonMarkingProb'][var + '_' + method] = {}
    
    full_df_ = full_df[full_df['SentTime'] != -1].copy()
    for path in full_df_['Path'].unique():
        df = full_df_[full_df_['Path'] == path]
        df['ECN'] = df.apply(lambda x: x['ECN'] if x[checkColumn] != 0 else 1, axis=1)
        df['nonMarking'] = 1.0 - df['ECN']
        df = df.sort_values(by='SentTime').reset_index(drop=True)

        time = df['SentTime'].values
        values = df['nonMarking'].values

        rightCont_time_average = np.sum(values[:-1] * np.diff(time)) / (time[-1] - time[0])
        df_res['nonMarkingProb']['event_rightCont_timeAvg'][path] = rightCont_time_average

        leftCont_time_average = np.sum(values[1:] * np.diff(time)) / (time[-1] - time[0])
        df_res['nonMarkingProb']['event_leftCont_timeAvg'][path] = leftCont_time_average

        linearInterp_time_average = np.sum(((values[:-1] + values[1:]) / 2) * np.diff(time)) / (time[-1] - time[0])
        df_res['nonMarkingProb']['event_linearInterp_timeAvg'][path] = linearInterp_time_average

        # avg, std = e2e_poisson_sampling(time, values)
        # df_res['nonMarkingProb']['event_poisson_eventAvg'][path] = (avg, std)

    full_df_ = None
    return df_res

def calculate_offline_E2E_delays(full_df, removeDrops, checkColumn, df_res):
    df_res['delay'] = {}
    for var in ['event']:
        for method in ['rightCont_timeAvg', 'leftCont_timeAvg', 'linearInterp_timeAvg', ]:
            df_res['delay'][var + '_' + method] = {}
    
    full_df_ = full_df.copy()
    if removeDrops:
        full_df_ = full_df_[full_df_[checkColumn] == 1]
    for path in full_df_['Path'].unique():
        df = full_df_[full_df_['Path'] == path]
        df = df.sort_values(by='SentTime').reset_index(drop=True)
        time = df['SentTime'].values
        values = df['Delay'].values

        rightCont_time_average = np.sum(values[:-1] * np.diff(time)) / (time[-1] - time[0])
        df_res['delay']['event_rightCont_timeAvg'][path] = rightCont_time_average

        leftCont_time_average = np.sum(values[1:] * np.diff(time)) / (time[-1] - time[0])
        df_res['delay']['event_leftCont_timeAvg'][path] = leftCont_time_average

        linearInterp_time_average = np.sum(((values[:-1] + values[1:]) / 2) * np.diff(time)) / (time[-1] - time[0])
        df_res['delay']['event_linearInterp_timeAvg'][path] = linearInterp_time_average

        # avg, std = e2e_poisson_sampling(time, values)
        # df_res['delay']['event_poisson_eventAvg'][path] = (avg, std)

        df = None
    full_df_ = None
    return df_res

def prune_data(full_df, projectColumn, steadyStart, steadyEnd):
    full_df = full_df[full_df[projectColumn] >= steadyStart]
    full_df = full_df[full_df[projectColumn] <= steadyEnd]
    full_df = full_df.sort_values(by=[projectColumn], ignore_index=True)
    return full_df

def addRemoveTransmission_data(full_df, linkDelays, linksRates):
    full_df['Delay'] = abs(full_df['ReceiveTime'] - full_df['SentTime'] - full_df['transmissionDelay'])
    full_df['SentTime'] = full_df['SentTime'] + linkDelays[0] + (full_df['PayloadSize'] * 8) / linksRates[0]
    return full_df

def timeShift(full_df, timeColumn, sizeColumn, linkDelays, linksRates):
    full_df[timeColumn] = full_df[timeColumn] - full_df['rtt'] / 2
    # full_df[timeColumn] = full_df[timeColumn] - (linkDelays[0] * 2 + (full_df[sizeColumn] * 8) / linksRates[0] + linkDelays[1] * 2 + (full_df[sizeColumn] * 8) / linksRates[1])
    return full_df

def compare_with_poison(full_df, linkRate, experiment, rate, results_folder, __ns3_path):
    poisson_df = pd.read_csv('{}/scratch/{}/{}/{}/SD0_PoissonSampler_events.csv'.format(__ns3_path, results_folder, rate, experiment))
    poisson_df['Label'] = 'Poisson'
    full_df = pd.concat([full_df, poisson_df], ignore_index=True)
    full_df = full_df.sort_values(by=['Time', 'TotalQueueSize'], ascending=[True, True]).reset_index(drop=True)
    changed_rows = full_df[(full_df['Label'] == 'Poisson') & (full_df['QueuingDelay'] != full_df['QueuingDelay'].shift(1))]
    previous_rows = full_df.loc[changed_rows.index - 1]
    # print(pd.concat([previous_rows, changed_rows]).sort_index())
    print(full_df[(full_df['Action'] == 'I') & (full_df['Time'] <= 300993713)])
    print(changed_rows)
    print("**********")
    print(previous_rows)
    

def manipulate_for_delay_Q(full_df, linkRate):
    mask = (full_df['Action'] == 'D') & (full_df['Action'].shift(-1) == 'E') & (full_df['Time'] != full_df['Time'].shift(-1)) & (full_df['TotalQueueSize'] != 0)
    time_diff = ((full_df['Time'].shift(-1) - full_df['Time']) * linkRate) / 8
    
    # Filter rows where the condition is met
    insert_rows = full_df[mask & (time_diff > full_df['TotalQueueSize'])].copy()
    if not insert_rows.empty:
        insert_rows['Time'] = insert_rows['Time'] + (insert_rows['TotalQueueSize']  * 8 / linkRate).astype(int)
        insert_rows['TotalQueueSize'] = 0
        insert_rows['QueuingDelay'] = 0
        insert_rows['MarkingProb'] = 0
        insert_rows['DropProb'] = 0
        insert_rows['Action'] = 'I'  # Marking as 'I' for intermediate
        
        full_df = pd.concat([full_df, insert_rows], ignore_index=True).sort_values(by='Time').reset_index(drop=True)
    full_df = full_df.sort_values(by=['Time', 'TotalQueueSize'], ascending=[True, True]).reset_index(drop=True)
    
    full_df['Delay'] = ((full_df['TotalQueueSize'] * 8) / linkRate).astype(int)
    time = full_df['Time'].values
    actions = full_df['Action'].values
    # values = full_df['QueuingDelay'].values
    values = full_df['Delay'].values

    linear_sum = 0
    for i in range(len(values[:-1])):
        x_1 = values[i]
        dt = time[i + 1] - time[i]
        if actions[i + 1] == 'E':
            if x_1 > 0:
                x_2 = x_1 - dt
            else:
                x_2 = 0
        else:
            x_2 = values[i + 1]
        linear_sum += (x_1 + x_2) / 2 * dt
    linearInterp_time_average = linear_sum / (time[-1] - time[0])
    return full_df, linearInterp_time_average

def manipulate_for_delay_Q_m(full_df, linkRate):
    mask = (full_df['Time'] != full_df['Time'].shift(-1)) & (full_df['TotalQueueSize'] != 0)
    time_diff = ((full_df['Time'].shift(-1) - full_df['Time']) * linkRate) / 8
    
    # Filter rows where the condition is met
    insert_rows = full_df[mask & (time_diff > full_df['TotalQueueSize'])].copy()
    if not insert_rows.empty:
        insert_rows['Time'] = insert_rows['Time'] + (insert_rows['TotalQueueSize']  * 8 / linkRate).astype(int)
        insert_rows['TotalQueueSize'] = 0
        insert_rows['QueuingDelay'] = 0
        insert_rows['MarkingProb'] = 0
        insert_rows['DropProb'] = 0
        insert_rows['Action'] = 'I'  # Marking as 'I' for intermediate
        
        full_df = pd.concat([full_df, insert_rows], ignore_index=True).sort_values(by='Time').reset_index(drop=True)
    full_df = full_df.sort_values(by=['Time', 'TotalQueueSize'], ascending=[True, True]).reset_index(drop=True)


    full_df['Delay'] = ((full_df['TotalQueueSize'] * 8) / linkRate).astype(int)
    time = full_df['Time'].values
    actions = full_df['Action'].values
    # values = full_df['QueuingDelay'].values
    values = full_df['Delay'].values

    linear_sum = 0
    for i in range(len(values[:-1])):
        x_1 = values[i]
        dt = time[i + 1] - time[i]
        if actions[i + 1] == 'D':
            x_2 = values[i + 1]
        elif (actions[i + 1] == 'E' or actions[i + 1] == 'I'):
            if actions[i] == 'I':
                x_2 = values[i + 1]
                # continue
            else:
                x_2 = x_1 - dt
        linear_sum += (x_1 + x_2) / 2 * dt
    linearInterp_time_average = linear_sum / (time[-1] - time[0])
    return full_df, linearInterp_time_average  

def calculate_offline_computations_on_switch(__ns3_path, results_folder, rate, experiment, segment, steadyStart, steadyEnd, paths, linkRate):
    file_paths = glob.glob('{}/scratch/{}/{}/{}/*_{}.csv'.format(__ns3_path, results_folder, rate, experiment, segment))
    dfs = {}
    for file_path in file_paths:
        df_res = {}
        df_name = 'A0D0'
        full_df = pd.read_csv(file_path)
        df_res['first'] = {}
        df_res['last'] = {}
        df_res['workload'] = {}
        df_res['sampleSize'] = {}
        df_res['successProbMean'] = {}
        full_df = prune_data(full_df, 'Time', steadyStart, steadyEnd)
        full_df = full_df[full_df['Label'].str.contains('10.1.1.1', na=False)]
        full_df = full_df.sort_values(by=['Time', 'TotalQueueSize'], ascending=[True, True]).reset_index(drop=True)
        full_df, delay_linearInterp_time_average = manipulate_for_delay_Q_m(full_df, linkRate)
        # full_df, delay_linearInterp_time_average = manipulate_for_delay_Q(full_df, linkRate)
        # compare_with_poison(full_df.copy(), linkRate, experiment, rate, results_folder, __ns3_path)
        df_res['delay'] = {}
        for var in ['event']:
            for method in ['rightCont_timeAvg', 'leftCont_timeAvg', 'linearInterp_timeAvg']:
                df_res['delay'][var + '_' + method] = {}

        df_res['successProb'] = {}
        for var in ['probability']:
            for method in ['rightCont_timeAvg', 'leftCont_timeAvg', 'linearInterp_timeAvg']:
                df_res['successProb'][var + '_' + method] = {}

        df_res['nonMarkingProb'] = {}
        for var in ['event']:
            for method in ['rightCont_timeAvg', 'leftCont_timeAvg', 'linearInterp_timeAvg']:
                df_res['nonMarkingProb'][var + '_' + method] = {}
        
        for path in paths:
            df_res['sampleSize'][path] = len(full_df)
            
            full_df['nonDropProb'] = 1.0 - full_df['DropProb']
            time = full_df['Time'].values
            values = full_df['nonDropProb'].values
            rightCont_time_average = np.sum(values[:-1] * np.diff(time)) / (time[-1] - time[0])
            df_res['successProb']['probability_rightCont_timeAvg'][path] = rightCont_time_average
            leftCont_time_average = np.sum(values[1:] * np.diff(time)) / (time[-1] - time[0])
            df_res['successProb']['probability_leftCont_timeAvg'][path] = leftCont_time_average
            linearInterp_time_average = np.sum(((values[:-1] + values[1:]) / 2) * np.diff(time)) / (time[-1] - time[0])
            df_res['successProb']['probability_linearInterp_timeAvg'][path] = linearInterp_time_average
            df_res['successProbMean'][path] = full_df['nonDropProb'].mean()

            values = full_df['Delay'].values
            # values = full_df['QueuingDelay'].values
            time = full_df['Time'].values
            rightCont_time_average = np.sum(values[:-1] * np.diff(time)) / (time[-1] - time[0])
            df_res['delay']['event_rightCont_timeAvg'][path] = rightCont_time_average
            leftCont_time_average = np.sum(values[1:] * np.diff(time)) / (time[-1] - time[0])
            df_res['delay']['event_leftCont_timeAvg'][path] = leftCont_time_average
            # df_res['delay']['event_linearInterp_timeAvg'][path] = np.sum((values[:-1] * np.diff(time)) - ((np.diff(time) * np.diff(time)) / 2)) / (time[-1] - time[0])
            df_res['delay']['event_linearInterp_timeAvg'][path] = delay_linearInterp_time_average

            df_res['first'][path] = full_df['Time'].iloc[0]
            df_res['last'][path] = full_df['Time'].iloc[-1]
            df_res['workload'][path] = 0

            full_df['nonMarkingProb'] = 1.0 - full_df['MarkingProb']
            time = full_df['Time'].values
            values = full_df['nonMarkingProb'].values
            rightCont_time_average = np.sum(values[:-1] * np.diff(time)) / (time[-1] - time[0])
            df_res['nonMarkingProb']['event_rightCont_timeAvg'][path] = rightCont_time_average
            leftCont_time_average = np.sum(values[1:] * np.diff(time)) / (time[-1] - time[0])
            df_res['nonMarkingProb']['event_leftCont_timeAvg'][path] = leftCont_time_average
            linearInterp_time_average = np.sum(((values[:-1] + values[1:]) / 2) * np.diff(time)) / (time[-1] - time[0])
            df_res['nonMarkingProb']['event_linearInterp_timeAvg'][path] = linearInterp_time_average

        dfs[df_name] = df_res
    return dfs
def calculate_offline_computations(__ns3_path, rate, segment, experiment, results_folder, steadyStart, steadyEnd, projectColumn, removeDrops=True, checkColumn="", linksRates=[], linkDelays=[], swtichDstREDQueueDiscMaxSize=0, stats=None, tsh=0.15):
    file_paths = glob.glob('{}/scratch/{}/{}/{}/*_{}.csv'.format(__ns3_path, results_folder, rate, experiment, segment))
    dfs = {}
    for file_path in file_paths:
        df_res = {}
        df_name = file_path.split('/')[-1].split('_')[0]
        full_df = pd.read_csv(file_path)
        if 'EndToEnd_markings' in segment:
            df_res = stats[df_name]
            df_res['enqueueTimeAvgNonMarkingFractionProb'] = {}
            df_res['congestionEst'] = {}
            full_df = timeShift(full_df, 'Time', 'BytesAcked', linkDelays, linksRates)
            full_df = prune_data(full_df, projectColumn, steadyStart, steadyEnd)
            df_res = calculate_offline_E2E_markingFraction(full_df, stats[df_name]['DelayMean'].keys(), df_res)
            df_res = calculate_offline_E2E_congestionEstimation(full_df, stats[df_name]['DelayMean'].keys(), df_res)
        if 'EndToEnd_packets' in segment:
            df_res['first'] = {}
            df_res['last'] = {}
            df_res['workload'] = {}
            df_res['sampleSize'] = {}
            df_res['successProbMean'] = {}
            full_df = addRemoveTransmission_data(full_df, linkDelays, linksRates)
            full_df = prune_data(full_df, projectColumn, steadyStart, steadyEnd)
            df_res = calculate_offline_E2E_lossRates(__ns3_path, full_df, df_res, checkColumn, linksRates[1], swtichDstREDQueueDiscMaxSize)
            df_res = calculate_offline_E2E_delays(full_df, removeDrops, checkColumn, df_res)
            df_res = calculate_offline_E2E_workload(full_df, df_res, steadyStart, steadyEnd)
            df_res = calculate_offline_E2E_markingProb(full_df, df_res, checkColumn, swtichDstREDQueueDiscMaxSize, linksRates[1], __ns3_path, tsh)
        if 'Poisson' in segment:
            full_df = prune_data(full_df, projectColumn, steadyStart, steadyEnd)
            # df_res = calculate_offline_switch_congestionEstimation(full_df, df_res)
            full_df['Delay'] = (full_df['TotalQueueSize'] * 8) / linksRates[0]
            df_res['DelayMean'] = full_df['Delay'].mean()
            df_res['DelayStd'] = full_df['Delay'].std()
            # df_res['DelayMean'] = full_df['QueuingDelay'].mean()
            # df_res['DelayStd'] = full_df['QueuingDelay'].std()
            df_res['first'] = full_df['Time'].iloc[0]
            df_res['last'] = full_df['Time'].iloc[-1]
            df_res['sampleSize'] = len(full_df)
            df_res['successProbMean'] = 1 - full_df['DropProb'].mean()
            df_res['successProbStd'] = full_df['DropProb'].std()
            df_res['nonMarkingProbMean'] = 1 - full_df['MarkingProb'].mean()
            df_res['nonMarkingProbStd'] = full_df['MarkingProb'].std()
            df_res['lastNonMarkingProbMean'] = 1 - full_df['LastMarkingProb'].mean()
            df_res['lastNonMarkingProbStd'] = full_df['LastMarkingProb'].std()
        dfs[df_name] = df_res
    return dfs

def read_data(__ns3_path, steadyStart, steadyEnd, rate, segment, checkColumn, projectColumn, experiment, remove_duplicates, results_folder, removeDrops=True):
    file_paths = glob.glob('{}/scratch/{}/{}/{}/*_{}.csv'.format(__ns3_path, results_folder, rate, experiment, segment))
    dfs = {}
    for file_path in file_paths:
        df_name = file_path.split('/')[-1].split('_')[0]
        df = pd.read_csv(file_path)
        if removeDrops:
            df = df[df[checkColumn] == 1]
        df = df.reset_index(drop=True)
        df = df[df[projectColumn] > steadyStart * 1000000000]
        df = df[df[projectColumn] < steadyEnd * 1000000000]
        df = df.sort_values(by=[projectColumn], ignore_index=True)
        if removeDrops:
            df = df.drop(columns=[checkColumn])
        if segment == 'EndToEnd' or segment == 'EndToEnd_crossTraffic':
            df['Delay'] = abs(df['ReceiveTime'] - df['SentTime'])
        if remove_duplicates:
            df = df.drop_duplicates(subset=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb', 'Id'], keep='first', ignore_index=True)
        dfs[df_name] = df
    return dfs

def read_data_flowIndicator(__ns3_path, rate, results_folder):
    flows_name = []
    file_paths = glob.glob('{}/scratch/{}/{}/0/*_EndToEnd.csv'.format(__ns3_path, results_folder, rate))
    for file_path in file_paths:
        flows_name.append(file_path.split('/')[-1].split('_')[0])
    return flows_name

def read_queues_indicators(__ns3_path, rate, results_folder):
    flows_name = []
    file_paths = glob.glob('{}/scratch/{}/{}/0/*_PoissonSampler.csv'.format(__ns3_path, results_folder, rate))
    for file_path in file_paths:
        if 'C' not in file_path.split('/')[-1].split('_')[0]:
            flows_name.append(file_path.split('/')[-1].split('_')[0])
    return flows_name

def convert_to_float(x):
    if 'Mbps' in x:
        return float(x[:-4])
    elif 'Kbps' in x:
        return float(x[:-4]) / 1000
    elif 'Gbps' in x:
        return float(x[:-4]) * 1000
    elif 'ms' in x:
        return float(x[:-2])
    elif 'us' in x:
        return float(x[:-2]) / 1000
    elif 'KB'in x:
        return float(x[:-2]) * 1000
    else:
        return float(x)

def calc_epsilon_with_bias(confidenceValue, segement_statistics, bias):
    return (calc_epsilon(confidenceValue, segement_statistics) + (bias / segement_statistics['DelayMean']))

def calc_epsilon(confidenceValue, segement_statistics):
    return (confidenceValue * segement_statistics['DelayStd']) / (np.sqrt(segement_statistics['sampleSize']) * segement_statistics['DelayMean'])

def calc_epsilon_loss_with_bias(confidenceValue, segement_statistics, bias):
    return (calc_epsilon_loss(confidenceValue, segement_statistics) + (bias / segement_statistics['successProbMean']))

def calc_epsilon_loss(confidenceValue, segement_statistics):
    return (confidenceValue * segement_statistics['successProbStd']) / (np.sqrt(segement_statistics['sampleSize']) * segement_statistics['successProbMean'])

def calc_epsilon_last_marking(confidenceValue, segement_statistics):
    return (confidenceValue * segement_statistics['lastNonMarkingProbStd']) / (np.sqrt(segement_statistics['sampleSize']) * segement_statistics['lastNonMarkingProbMean'])

def calc_epsilon_last_marking_with_bias(confidenceValue, segement_statistics, bias):
    return (calc_epsilon_last_marking(confidenceValue, segement_statistics) + (bias / segement_statistics['lastNonMarkingProbMean']))

def calc_epsilon_marking_with_bias(confidenceValue, segement_statistics, bias):
    return (calc_epsilon_marking(confidenceValue, segement_statistics) + (bias / segement_statistics['nonMarkingProbMean']))

def calc_epsilon_marking(confidenceValue, segement_statistics):
    return (confidenceValue * segement_statistics['nonMarkingProbStd']) / (np.sqrt(segement_statistics['sampleSize']) * segement_statistics['nonMarkingProbMean'])

def calc_epsilon_loss_2(confidenceValue, segement_statistics):
    return (confidenceValue * segement_statistics['successProbStd_2']) / (np.sqrt(segement_statistics['sampleSize']) * segement_statistics['successProbMean_2'])

def calc_error(confidenceValue, segement_statistics):
    return (confidenceValue * segement_statistics['DelayStd']) / np.sqrt(segement_statistics['sampleSize'])

def sample_data(data, sample_column):
    exit = False
    while not exit:
        # option 1: sample data with a fixed rate
        data_copy = data.sample(frac=0.05).sort_values(by=[sample_column])
        
        # option 2: sample data with a poisson process. Pick the closest packet to the arrival time
        # interArrivals = np.random.exponential(1/poisson_sample_rate, int(duration * poisson_sample_rate)) * 1000000000
        # interArrivals = np.cumsum(interArrivals)
        # interArrivals  = interArrivals + steadyStart * 1000000000
        # interArrivals = interArrivals[interArrivals > steadyStart * 1000000000]
        # interArrivals = interArrivals[interArrivals < steadyEnd * 1000000000]
        # data_copy = pd.DataFrame()
        # for i in range(len(interArrivals)):
        #     data_copy = pd.concat([data_copy, data.iloc[(data[sample_column] - interArrivals[i]).abs().argsort()[:1]]])

        # option 3: sample data with a poisson process. Pick the packets based on the exp distribution not the arrival time
        # exps = np.random.exponential(1/poisson_sample_rate, len(data))
        # c = np.abs(exps - 1/poisson_sample_rate) / (1/poisson_sample_rate) < 0.01
        # data_copy = data.copy()
        # data_copy['IsSample'] = c
        # data_copy = data_copy[data_copy['IsSample'] == True]
        # data_copy = data_copy.sort_values(by=[sample_column])

        data_copy['InterArrivalTime'] = data_copy[sample_column].diff()
        data_copy = data_copy.dropna().reset_index(drop=True)
        anderson_statistic, anderson_critical_values, _ = anderson(data_copy['InterArrivalTime'], 'expon')
        if anderson_statistic < anderson_critical_values[2]:
            # print('Anderson-Darling test passed')
            exit = True
    return data_copy.drop(columns=['InterArrivalTime'])

def get_switch_samples_delays(flowIndicatorDf, switchDf):
    l_df = flowIndicatorDf.copy()
    l_df = pd.merge(l_df, switchDf, on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb', 'Id'], how='right')
    l_df['SentTime'] = l_df['SentTime'].mask(l_df['SourceIp'] == '0.0.0.0', l_df['SampleTime'])
    l_df['ReceiveTime'] = l_df['ReceiveTime'].mask(l_df['SourceIp'] == '0.0.0.0' , l_df['SampleTime'])
    l_df = l_df.dropna(subset=['SentTime', 'ReceiveTime'])
    return l_df

def switch_data(flowIndicatorDf, switchDf, sampling):
    l_df = flowIndicatorDf.copy()
    l_df = pd.merge(l_df, switchDf, on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb', 'Id'], how='inner')
    if sampling:
        l_df = sample_data(l_df, 'ReceiveTime')
    return l_df

def switch_different_traffics_delaymean(switchDf):
    l_df = switchDf.copy()
    l_df['Delay'] = abs(l_df['ReceiveTime'] - l_df['SentTime'])
    l_df = l_df.groupby(['SourceIp', 'DestinationIp']).mean().reset_index()
    # l_df = l_df.groupby(['SourceIp', 'DestinationIp']).count().reset_index()
    print(l_df)

def intermediateLink_transmission(flowIndicatorDf, source, dest, linkNum):
    l_df = flowIndicatorDf.copy()
    l_df = pd.merge(l_df, source.drop(columns=['ReceiveTime']), on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb', 'Id'], how='inner')
    l_df = pd.merge(l_df, dest.drop(columns=['SentTime']), on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb', 'Id'], how='inner')
    l_df['Delay_' + str(linkNum)] = abs(l_df['ReceiveTime'] - l_df['SentTime'])
    return l_df.drop(columns=['ReceiveTime', 'SentTime'])

def interSwitch_queuing(flowIndicatorDf, switchDf, segNum):
    l_df = flowIndicatorDf.copy()
    l_df = pd.merge(l_df, switchDf, on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb', 'Id'], how='inner')
    l_df['Delay_' + str(segNum)] = abs(l_df['ReceiveTime'] - l_df['SentTime'])
    return l_df.drop(columns=['ReceiveTime', 'SentTime'])

def addDelay(data):
    if 'Delay' not in data.columns:
        data['Delay'] = abs(data['ReceiveTime'] - data['SentTime'])
    return data

def get_timeAvg(data):
    # sort the data based on the receive time
    # data = data.sort_values(by=['ReceiveTime'])
    data = data.sort_values(by=['SentTime'])
    # calculate the interarrival time
    # data['InterArrivalTime'] = data['ReceiveTime'].diff().fillna(0)
    data['InterArrivalTime'] = data['SentTime'].diff().fillna(0)
    # calculate the time average: sum(InterArrivalTime * Delay) / sum(InterArrivalTime)
    timeAvg = (data['InterArrivalTime'] * data['Delay']).sum() / data['InterArrivalTime'].sum()
    return timeAvg

def get_endToEd_loss_statistics(data):
    statistics = {}
    data_copy = data.copy()
    # statistics['successProbMeanPackets'] = 1 - (len(data_copy[(data_copy['ECN'] == 1) | (data_copy['IsReceived'] == 0)]) / len(data_copy))
    statistics['successProbMeanPackets'] = 1 - (len(data_copy[data_copy['IsReceived'] == 0]) / len(data_copy))
    statistics['successProbMeanBytes'] = 1 - (data_copy[(data_copy['ECN'] == 1) | (data_copy['IsReceived'] == 0)]['PayloadSize'].sum() / data_copy['PayloadSize'].sum())
    return statistics

def get_loss_statistics(data):
    statistics = {}
    data_copy = data.copy()
    statistics['successProbMean'] = 1 - data_copy['MarkingProb'].mean()
    statistics['successProbStd'] = data_copy['MarkingProb'].std()
    statistics['successProbMean_2'] = 1 - data_copy['MarkingProb_2'].mean()
    statistics['successProbStd_2'] = data_copy['MarkingProb_2'].std()
    statistics['sampleSize'] = len(data_copy)
    return statistics

def get_statistics(data, removeZeroes=False, timeAvg=False):
    statistics = {}
    data_copy = addDelay(data.copy())
    if removeZeroes:
        data_copy = data_copy[data_copy['Delay'] > 0]
    statistics['DelayMean'] = data_copy['Delay'].mean()
    statistics['DelayStd'] = data_copy['Delay'].std()
    statistics['sampleSize'] = len(data_copy)
    statistics['DelaySkew'] = data_copy['Delay'].skew()
    statistics['sizeStd'] = data_copy['PayloadSize'].std()
    if timeAvg:
        statistics['timeAvg'] = get_timeAvg(data_copy)
    return statistics

def print_traffic_rate(endToEnd_dfs):
    endToEnd_dataRates = {}
    for flow in endToEnd_dfs.keys():
        endToEnd_dataRates[flow] = endToEnd_dfs[flow]['PayloadSize'].sum() * 8 / (10) / 1000000
    print([(key, value / sum(endToEnd_dataRates.values()) * 100) for key, value in endToEnd_dataRates.items()])

def clear_data_from_outliers_in_time(endToEnd_dfs, switches_dfs, start_dfs):
    for switch in switches_dfs.keys():
        per_traffic_data = []
        for flow in endToEnd_dfs.keys():
            per_traffic_data.append(pd.merge(endToEnd_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay', 'Path', 'ECN', 'PacketSize']), switches_dfs[switch], on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb', 'Id'], how='inner'))
        switches_dfs[switch] = pd.concat(per_traffic_data)

    for queue in start_dfs.keys():
        per_traffic_data = []
        for flow in endToEnd_dfs.keys():
            per_traffic_data.append(pd.merge(endToEnd_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay', 'Path', 'ECN', 'PacketSize']), start_dfs[queue], on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb', 'Id'], how='inner'))
        start_dfs[queue] = pd.concat(per_traffic_data)
    

def read_paths_flows(switches_dfs, test):
    # ecah path flows are a dataframe of unique sourceIp, sourcePort, destinationIp, destinationPort 
    paths = {}
    for switch in switches_dfs:
        # get the unique sourceIp, sourcePort, destinationIp, destinationPort
        if not test:
            paths[switch] = switches_dfs[switch].drop_duplicates(subset=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort'], keep='first', ignore_index=True).drop(columns=['SentTime', 'ReceiveTime', 'Id', 'SequenceNb', 'PayloadSize'])
        else:
            paths[switch] = switches_dfs[switch].drop_duplicates(subset=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'SequenceNb', 'Id'], keep='first', ignore_index=True).drop(columns=['SentTime', 'ReceiveTime', 'PayloadSize'])
    # print(paths)
    return paths

def delayProcess_consistency_check(flows_sampled, rounds_results):
    for q in flows_sampled.keys():
        for flow_on_switch in flows_sampled[q]:
            flow_on_switch['Delay'] = abs(flow_on_switch['ReceiveTime'] - flow_on_switch['SentTime'])

        anova_res  = f_oneway(*[flows_sampled[q][i]['Delay'] for i in range(len(flows_sampled[q]))])
        kruskal_res = kruskal(*[flows_sampled[q][i]['Delay'] for i in range(len(flows_sampled[q]))])
        if anova_res.pvalue > 0.05:
            rounds_results['ANOVA'][q] += 1
        if kruskal_res.pvalue > 0.05:
            rounds_results['Kruskal'][q] += 1
        # # plot the delay distribution of each flow on each switch
        # for i in range(len(flows_sampled[q])):
        #     plt.hist(flows_sampled[q][i]['Delay'], bins=100)
        #     plt.title('Switch {}'.format(i))
        #     plt.xlabel('Delay (ns)')

        # plt.legend(['Flow {}'.format(i) for i in range(len(flows_sampled[q]))])    
        # plt.savefig('../results/{}_delayDist.png'.format(q))
        # plt.close()

def plot_overall_delay_distribution(rate, common_switch_sample_df, queue):
    # plot the delay distribution of SWitch T0 and Sample T0
    fig, ax = plt.subplots(1, 1)
    sns.histplot(common_switch_sample_df['SentTime'] - common_switch_sample_df['ReceiveTime'], bins=100)
    ax.set_title('Sample T0')
    ax.set_xlabel('Delay (ns)')
    plt.savefig('../results/{}/{}_{}_overall_delayDist.png'.format(rate, rate, queue))
    plt.close()

def plot_delay_over_time(endToEnd_dfs, paths, rate, results_folder):
    for flow in endToEnd_dfs.keys():
        if flow == "R0H0R2H0" or flow == "R0H1R2H1":
            for path in paths:
                path_flow = endToEnd_dfs[flow][endToEnd_dfs[flow]['Path'] == int(path[1])]
                path_flow = path_flow.sort_values(by=['ReceiveTime'])
                plt.plot(path_flow['ReceiveTime'], path_flow['Delay'], label='path {}'.format(path))
            plt.legend()
            plt.xlabel('Time (ns)')
            plt.ylabel('Delay (ns)')
            plt.title('Flow {}'.format(flow))
            plt.savefig('../results_postProcessing_reverse_delay_2/{}/{}_delayOverTime_{}.png'.format(rate, flow, results_folder))
            # plt.savefig('../results_postProcessing/{}/{}_{}_delayOverTime_{}.png'.format(1.0, rate, flow, results_folder))
            plt.close()
            plt.clf()