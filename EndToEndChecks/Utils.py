import psutil, os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from enum import Enum
import seaborn as sns
import numpy as np
from scipy.stats import anderson
from scipy.stats import f_oneway, kruskal
from scipy.stats import bernoulli, ks_2samp
from math import factorial, exp
import csv
from collections import defaultdict
import pprint

estimation_gain = 0.0625
init_alpha = 1

class SubSamplingError(str, Enum):
    NoError = 'NoError'
    MinDGTMaxD = 'MinDGTMaxD'
    NotEnoughPackets = 'NotEnoughPackets'
    NotEnoughSamples = 'NotEnoughSamples'
    IDCITrsh = 'IDCITrsh'
    NotPoisson = 'NotPoisson'

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
    
    def compute_average_packet_size_from_cdf(self):
        """
        Computes the average packet size using the CDF via finite difference approximation.
        Assumes self.packet_cdf is sorted and well-formed.
        """
        if not self.packet_cdf:
            return 0.0

        sorted_sizes = sorted(self.packet_cdf.keys())
        avg_size = 0.0
        prev_cdf = 0.0

        for size in sorted_sizes:
            cdf = self.packet_cdf[size]
            prob_mass = cdf - prev_cdf
            avg_size += size * prob_mass
            prev_cdf = cdf

        return avg_size

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
    return 1 - np.average([endToEnd_dfs[flow]['successProbMean'][p] for p in range(len(paths)) for flow in endToEnd_dfs.keys()])

def calculate_drop_rate_online(endToEnd_dfs, paths):
    loss_sum = 0
    counts = 0
    for flow in endToEnd_dfs.keys():
        for p in range(len(paths)):
            loss_sum += endToEnd_dfs[flow]['sentPacketsOnLink'][p] - endToEnd_dfs[flow]['receivedPackets'][p]
            counts += endToEnd_dfs[flow]['sentPacketsOnLink'][p]
    return loss_sum / counts

def calculate_drop_rate_DC(samples_dfs):
    successRates = []
    for queue in samples_dfs.keys():
        successRates.append(samples_dfs[queue]['SuccessProbMean'])
    return 1 - np.prod(successRates)

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
    if len(full_df_) <= 1:
        df_res['first'][0] = steadyStart
        df_res['last'][0] = steadyEnd
        df_res['workload'][0] = 0       
    for path in full_df_['Path'].unique():
        df = full_df_[full_df_['Path'] == path]
        if len(df) <= 1:
            df_res['first'][path] = steadyStart
            df_res['last'][path] = steadyEnd
            df_res['workload'][path] = 0
        else:
            df_res['first'][path] = df['SentTime'].iloc[0]
            df_res['last'][path] = df['SentTime'].iloc[-1]
            df_res['workload'][path] = df['PayloadSize'].sum() * 8 / (steadyEnd - steadyStart)
            # print("Path: {}, total packets: {}, workload: {} bps".format(path, len(df), df_res['workload'][path]))
        df = None
    full_df_ = None
    return df_res

def calculate_offline_E2E_lossRates_DC(full_df, df_res, checkColumn, txDelay, df_name, passiveProbe, samplingMethod, steadyStart, steadyEnd):
    df_res['successProb'] = {}
    for var in ['event']:
        for method in ['rightCont_timeAvg', 'leftCont_timeAvg', 'linearInterp_timeAvg', 'poisson_eventAvg', 'eventAvg']:
            df_res['successProb'][var + '_' + method] = {}

    df_res['sampleSize']['successProb'] = {}
    df_res['bias']['successProb'] = {}
    full_df_ = full_df[full_df['SentTime'] != -1].copy()
    for path in full_df_['Path'].unique():
        df_res['bias']['successProb'][path] = 0
        df = full_df_[full_df_['Path'] == path]
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

        df_res['successProb']['event_eventAvg'][path] = (np.mean(values), np.std(values) / np.sqrt(len(values)))

        if passiveProbe:
            interarrival = np.diff(time)
            anderson_statistic, anderson_critical_values, _ = anderson(interarrival, 'expon')
            if anderson_statistic <= anderson_critical_values[2]:
                samples_times = time
            else:
                print("Sample times are 'NOT' exponentially distributed.")
                samples_times = []
        else:
            # samples_times = find_samples_path(time, txDelay, df_res['RTT'][path], df_name, samplingMethod)
            samples_times = find_samples_path_new(time, txDelay, df_res['RTT'][path], df_name, samplingMethod, steadyStart, steadyEnd, steps=1)

        df_res['sampleSize']['successProb'][path] = len(samples_times)
        samples_values = df[df['SentTime'].isin(samples_times)]['nonDropEvent'].values
        if df_res['sampleSize']['successProb'][path] == 0:
            avg, std = 0, 0
        else:
            avg, std = np.mean(samples_values), np.std(samples_values) / np.sqrt(len(samples_values))
        df_res['successProb']['event_poisson_eventAvg'][path] = (avg, std)
    
    full_df_ = None
    return df_res

def calculate_offline_E2E_lossRates(__ns3_path, full_df, df_res, checkColumn, txDelay, linksRate, swtichDstREDQueueDiscMaxSize, df_name, passiveProbe, samplingMethod):
    df_res['successProb'] = {}
    for var in ['event', 'probability']:
        for method in ['rightCont_timeAvg', 'leftCont_timeAvg', 'linearInterp_timeAvg', 'poisson_eventAvg', 'eventAvg']:
            df_res['successProb'][var + '_' + method] = {}

    packets_cfd = PacketCDF()
    packets_cfd.load_cdf_data('{}/scratch/ECNMC/Helpers/packet_size_cdf_singleQueue.csv'.format(__ns3_path))
    df_res['sampleSize']['successProb'] = {}
    df_res['bias']['successProb'] = {}
    full_df_ = full_df[full_df['SentTime'] != -1].copy()
    for path in full_df_['Path'].unique():
        df_res['bias']['successProb'][path] = 0
        df = full_df_[full_df_['Path'] == path]
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

        df_res['successProb']['event_eventAvg'][path] = (np.mean(values), np.std(values) / np.sqrt(len(values)))

        if passiveProbe:
            interarrival = np.diff(time)
            anderson_statistic, anderson_critical_values, _ = anderson(interarrival, 'expon')
            if anderson_statistic <= anderson_critical_values[2]:
                samples_times = time
            else:
                print("Sample times are 'NOT' exponentially distributed.")
                samples_times = []
        else:
            samples_times = find_samples_path(time, txDelay, df_res['RTT'][path], df_name, samplingMethod)
        df_res['sampleSize']['successProb'][path] = len(samples_times)
        samples_values = df[df['SentTime'].isin(samples_times)]['nonDropEvent'].values
        if df_res['sampleSize']['successProb'][path] == 0:
            avg, std = 0, 0
        else:
            avg, std = np.mean(samples_values), np.std(samples_values) / np.sqrt(len(samples_values))
        df_res['successProb']['event_poisson_eventAvg'][path] = (avg, std)

        df['nonDropProb'] = df.apply(lambda x: 1.0 - packets_cfd.calculate_probability_greater_than(max(swtichDstREDQueueDiscMaxSize - (x['Delay'] * linksRate / 8), x['PayloadSize'])) if x[checkColumn] != 0 else 0.0, axis=1)

        time = df['SentTime'].values
        values = df['nonDropProb'].values

        rightCont_time_average = np.sum(values[:-1] * np.diff(time)) / (time[-1] - time[0])
        df_res['successProb']['probability_rightCont_timeAvg'][path] = rightCont_time_average

        leftCont_time_average = np.sum(values[1:] * np.diff(time)) / (time[-1] - time[0])
        df_res['successProb']['probability_leftCont_timeAvg'][path] = leftCont_time_average

        linearInterp_time_average = np.sum(((values[:-1] + values[1:]) / 2) * np.diff(time)) / (time[-1] - time[0])
        df_res['successProb']['probability_linearInterp_timeAvg'][path] = linearInterp_time_average

        df_res['successProb']['probability_eventAvg'][path] = (np.mean(values), np.std(values) / np.sqrt(len(values)))

        samples_values = df[df['SentTime'].isin(samples_times)]['nonDropProb'].values
        if df_res['sampleSize']['successProb'][path] == 0:
            avg, std = 0, 0
        else:
            avg, std = np.mean(samples_values), np.std(samples_values) / np.sqrt(len(samples_values))

        df_res['successProb']['probability_poisson_eventAvg'][path] = (avg, std)

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

def calc_RTT(avgQueueDelay, linksPropDelay, linksRate, avgPacketSize):
    totalPropDelay = np.sum([2 * prop for prop in linksPropDelay])
    totalTxDelay = np.sum([avgPacketSize * 8 / rate for rate in linksRate])
    return totalPropDelay + totalTxDelay + avgQueueDelay

def calc_RTT_per_path(full_df, df_res, checkColumn, linkDelays):
    full_df_ = full_df.copy()
    full_df_ = full_df_[full_df_[checkColumn] == 1]
    for path in full_df_['Path'].unique():
        df = full_df_[full_df_['Path'] == path]
        df = df.sort_values(by='SentTime').reset_index(drop=True)
        # df_res['RTT'][path] = np.mean(abs(df['ReceiveTime'] - df['TxDequeueTime'])) + np.sum(linkDelays)
        df_res['RTT'][path] = 2 * np.sum(linkDelays)
        df = None
    full_df_ = None
    return df_res

def find_samples_path_ccf(arrival_times, steadyStart, steadyEnd, queue_names, file_path, linkDelays, linkRates, queue_size_trshs, MinimumNumberOfSamples):
    result = {}
    subSamplingError = SubSamplingError.NoError

    # times = np.arange(steadyStart, steadyEnd, 90)
    # times = np.cumsum(np.random.exponential(90, size=(steadyEnd - steadyStart) // 90)) + steadyStart
    # T = 8000 * 16
    # times, queue_size_samples, _, _ = sample_total_queue_size(times, queue_names, file_path, linkDelays, linkRates, queue_size_trshs)
    # arrival_increments = sample_increments_of_arrivals(arrival_times, T, times)
    # res = crosscorr_qsize_vs_arrival_increments(arrival_increments, queue_size_samples, times)
    # band = 1.96 / np.sqrt(len(arrival_increments))
    # ccf = res['crosscorr']
    # result['e2eVsSwitchCCFpercntg'] = len(np.where((ccf < -band) | (ccf > band))[0]) / len(ccf) * 100
    # result['e2eVsSwitchMaxCCF'] = np.max(np.abs(ccf))
    result['e2eVsSwitchCCFpercntg'] = np.nan
    result['e2eVsSwitchMaxCCF'] = np.nan
    lags, chi_squared_statistic = chi_squared_test(arrival_times, steadyStart, steadyEnd) 
    out_of_band = [lag for lag, r in zip(lags, chi_squared_statistic) if r]
    result['e2eCorrArrivals'] = (min(out_of_band), len(out_of_band), len(lags))
    # visualize_crosscorr_result(res, file_path)
    # res_auto = autocorr_arrival_increments(arrival_increments)
    # res_auto["times"] = times
    # visualize_autocorr_result(res_auto, file_path, T)
    if len(arrival_times) < MinimumNumberOfSamples:
        print ("Warning: Not enough e2e packets!")
        subSamplingError = SubSamplingError.NotEnoughPackets + "+" + subSamplingError.value
        return [], subSamplingError, result

    return arrival_times, SubSamplingError.NoError, result

def remove_randomly_within_lag(arrival_times, T, steadyStart, steadyEnd, lag, initial_p):
    # Randomly pick one arrival within each lag window, and remove the others
    # print("Removing randomly within lag", lag, "with initial probability", initial_p)
    times = np.arange(steadyStart, steadyEnd, lag * T)
    arrivals_idxs_per_bin = sample_increments_of_arrivals(arrival_times, lag * T, times, event_type="idx")
    round = 0
    keep_idx = []
    while True:
        keep_idx = []
        for idxs in arrivals_idxs_per_bin:
            if len(idxs) > 1:
                keep_idx.append(np.random.choice(idxs, size=1, replace=False)[0])
                # keep_idx.append(idxs[0])
            elif len(idxs) == 1:
                keep_idx.append(idxs[0])
        keep_idx = np.array(keep_idx, dtype=int)
        # run a random thinning with probability p to further reduce the number of samples
        keep_idx = keep_idx[np.random.rand(len(keep_idx)) < initial_p]
        mask = np.zeros(len(arrival_times), dtype=bool)
        mask[keep_idx] = True
        lags, res, chi2_res = chi_squared_test(arrival_times[mask], steadyStart, steadyEnd, lags=[lag])
        if res[0] == False:
            # print("No significant dependence at lag", lag, ". Stopping iteration.")
            break
        if len(keep_idx) < 100:
            # print("Not enough samples left after thinning. Stopping iteration.")
            break
        round += 1
        initial_p *= 0.95
        # print("Round", round, "thinning with probability", initial_p, "number of samples left:", len(keep_idx), "out of total", len(arrival_times))

    mask = np.zeros(len(arrival_times), dtype=bool)
    mask[keep_idx] = True
    return mask

def find_samples_path_chi_squared_test(arrival_times, steadyStart, steadyEnd, MinimumNumberOfSamples):
    result = {}
    subSamplingError = SubSamplingError.NoError.value
    result['e2eVsSwitchCCFpercntg'] = np.nan
    result['e2eVsSwitchMaxCCF'] = np.nan

    correlated_lags = []
    round = 0
    initial_p = 1.0
    while True:
        # print("\n############# after round", round, "#############")
        # print("Arrivals: ", len(arrival_times))
        # rel, w1, lam_hat = rel_w1_to_exp_fit(arrival_times)
        # print("***** Exponential Fit *****")
        # print("Relative Error:", rel)
        # print("Wasserstein-1 Distance:", w1)
        # print("Estimated Lambda:", lam_hat)
        lags, res, chi2 = chi_squared_test(arrival_times, steadyStart, steadyEnd)
        correlated_lags = [lag for lag, r in zip(lags, res) if r]
        # print("Significant dependence at lags:", correlated_lags[:10])
        upper_band = 0.05 + ((1.96 * np.sqrt(0.95*0.05)) / np.sqrt(len(lags)))
        lower_band = 0.05 - ((1.96 * np.sqrt(0.95*0.05)) / np.sqrt(len(lags)))
        out_of_band = [lag for lag, r in zip(lags, res) if r]
        result['e2eCorrArrivals'] = (min(out_of_band), len(out_of_band), len(lags))

        # print("band for chi-squared test:", lower_band, ",", upper_band, "lags with significant dependence:", len(correlated_lags) / len(lags))
        if len(correlated_lags) / len(lags) < upper_band:
            # print("The proportion of lags with significant dependence is within the expected band. Stopping iteration.")
            subSamplingError = SubSamplingError.NotPoisson + "+" + subSamplingError
            break
        if len(arrival_times) < 100:
            # print("Too few arrival times left. Stopping iteration.")
            break
        lag = correlated_lags[0]
        mask = remove_randomly_within_lag(arrival_times, T=120, steadyStart=steadyStart, steadyEnd=steadyEnd, lag=lag, initial_p=initial_p)
        arrival_times = arrival_times[mask]
        round += 1
    
    if len(arrival_times) < MinimumNumberOfSamples:
        # print ("Warning: Not enough e2e packets!")
        subSamplingError = SubSamplingError.NotEnoughPackets + "+" + subSamplingError
        return [], subSamplingError, result

    return arrival_times, SubSamplingError.NoError, result


def find_samples_path_new(time, txDelay, avg_interarrival_=None, df_name=None, samplingMethod='Orig', steadyStart=0, steadyEnd=1, steps=1, MinimumNumberOfSamples=0):
    subSamplingError = SubSamplingError.NoError
    # state 0: find the minimum Δ that has more than 95% non-empty intervals and the maximum Δ that gives the minumum number of samples
    minD, _ = find_delta_for_empty_prob(time, p0_max=0.05)
    maxD = (steadyEnd - steadyStart) / MinimumNumberOfSamples
    
    # stage 1: if minD is larger than maxD, we cannot do subsampling
    if minD >= maxD:
        print ("Warning: Minimum Δ is larger than maximum Δ, cannot do subsampling! MinD: {}, MaxD: {}".format(minD, maxD))
        subSamplingError = SubSamplingError.MinDGTMaxD
        minD = maxD
        # stage 2: plot IDC over Δ for minD to steadyEnd - steadyStart) / MinimumNumberOfSamples to see if we can do subsampling
        # plot_idc_over_delta(time, d_max=maxD, t_start=steadyStart, duration=steadyEnd - steadyStart, label_prefix=f"{df_name}bfore_trimming_")
        # return [], subSamplingError
    avgD = (minD + maxD) * 0.5

    # stage 2: plot IDC over Δ for minD to steadyEnd - steadyStart) / MinimumNumberOfSamples to see if we can do subsampling
    deltas_valid, idc_values = plot_idc_over_delta(time, d_max=maxD, t_start=steadyStart, duration=steadyEnd - steadyStart, label_prefix=f"{df_name}test_bfore_trimming_")

    # stage 2.1: print out the first derivative of IDC at Δ = avgD
    # deriv, info = idc_derivative_at_delta(deltas_valid, idc_values, d1=avgD)
    # deriv, info = idc_derivative_by_local_averaging(deltas_valid, idc_values, d1=avgD)
    # print("Estimated derivative of exp:", df_name, "is", deriv)
    # stage 2.1: print out the first derivative of IDC at Δ = minD
    # deriv, info = idc_derivative_at_delta(deltas_valid, idc_values, d1=minD)
    deriv, info = idc_derivative_by_local_averaging(deltas_valid, idc_values, d1=minD)
    # print("Estimated derivative of exp:", df_name, "is", deriv)

    # stage 3: see if we have enough packets to sample form
    if len(time) < MinimumNumberOfSamples:
        print ("Warning: Not enough e2e packets!")
        subSamplingError = SubSamplingError.NotEnoughPackets + "+" + subSamplingError.value
        return [], subSamplingError
    
    if deriv > 8e-6:
        print ("Warning: IDC is increasing at Δ = {}, cannot do subsampling! Derivative: {}".format(minD, deriv))
        subSamplingError = SubSamplingError.IDCITrsh + "+" + subSamplingError.value
        return [], subSamplingError

    # stage 4: distance-aware sampling to get the samples for estimation
    # samples = distanceAwareSampling(time, 1.2 / maxD)
    # samples = distanceAwareSampling(time, 1.0 / avgD)
    samples = distanceAwareSampling(time, 1.0 / minD)

    # stage 5: see if we have enough samples after distance-aware sampling
    if len(samples) < (MinimumNumberOfSamples * 0.95):
        print ("Warning: Not enough samples after distance-aware sampling!", "Got {}, expected {}".format(len(samples), MinimumNumberOfSamples))
        subSamplingError = SubSamplingError.NotEnoughSamples + "+" + subSamplingError.value
        return [], subSamplingError

    # stage 6: plot IDC over Δ for the samples to see if the subsamplig went well
    # plot_idc_over_delta(samples, d_min=minD, d_max=maxD, t_start=steadyStart, duration=steadyEnd - steadyStart, label_prefix=f"{df_name}after_trimming_withDA(woADtest)_maxD_")

    return samples, subSamplingError

    
    #################################################
    # aggregated_samples = []
    # steps = 1
    # for step in range(steps):
    #     intervalStart = steadyStart + (steadyEnd - steadyStart) / steps * step
    #     intervalEnd = steadyStart + (steadyEnd - steadyStart) / steps * (step + 1)
    #     interval_times = time[(time >= intervalStart) & (time < intervalEnd)]
    # #     # print("Interval from {} to {} has {} packets".format(intervalStart, intervalEnd, len(interval_times)))
    # #     t_sel_, report = e2e_poisson_like_sampler(interval_times, N_min=len(interval_times) * 0.7, max_delta_for_idc=avg_interarrival_ * 5, df_name=df_name)
    # #     # print("Poisson-like sampler selected {} packets".format(len(t_sel_)))
    # #     # t_sel, intervalBrnval = find_samples_path(interval_times, txDelay, avg_interarrival_, df_name, samplingMethod, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    # #     # print("Bernoulli sampler selected {} packets".format(len(t_sel)))
    # #     if len(t_sel_) == 0:
    # #         return []
    # #     aggregated_samples.extend(t_sel_)

    #     plot_idc_over_delta(interval_times, t_start=steadyStart, duration=steadyEnd - steadyStart, label_prefix=f"{df_name}bfore_trimming_")
    #     t_sel_, info = trim_counts_round_robin_J_two_scales(interval_times)
    #     if len(t_sel_) == 0:
    #         return []
    #     plot_idc_over_delta(t_sel_, t_start=steadyStart, duration=steadyEnd - steadyStart, label_prefix=f"{df_name}after_trimming_")
    #     # X = info["X_counts"]   # original counts per Δ
    #     # Y = info["Y_counts"]   # trimmed counts per Δ
    #     # plot_bin_count_distributions(X, Y, title_suffix=" (Δ-bin)")
    #     # t_sel_lambda =  info["Delta"] / float(Y.mean())
    #     # plot_iat_distribution(interval_times, t_sel_, t_sel_lambda, title_suffix="")
    #     # # for round in info["trace"]:
    #     # #     print(round)
    #     # print(info["trace"][-1])
    #     aggregated_samples.extend(t_sel_)
    # return aggregated_samples
    # # print("Delta:", info["Delta"])
    # # print("IDC(Δ)  :", info["initial_idc_by_factor"][1], "->", info["final_idc_by_factor"][1])
    # # print("IDC(2Δ) :", info["initial_idc_by_factor"].get(2), "->", info["final_idc_by_factor"].get(2))
    # # print("kept:", info["final_total"])
    #################################################
    # aggregated_samples = []
    # max_sample_size_brnval = []
    # for step in range(steps):
    #     intervalStart = steadyStart + (steadyEnd - steadyStart) / steps * step
    #     intervalEnd = steadyStart + (steadyEnd - steadyStart) / steps * (step + 1)
    #     interval_times = time[(time >= intervalStart) & (time < intervalEnd)]
    #     intervalSamples, intervalBrnval = find_samples_path(interval_times, txDelay, avg_interarrival_, df_name, samplingMethod, [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.18])
    #     if len(intervalSamples) == 0:
    #         return []
    #     aggregated_samples.extend(intervalSamples)
    #     max_sample_size_brnval.append(intervalBrnval)
 
    # if len(set(max_sample_size_brnval)) == 1:
    #     return aggregated_samples
    # else:
    #     min_brnval = min(max_sample_size_brnval)
    #     for step in range(steps):
    #         intervalStart = steadyStart + (steadyEnd - steadyStart) / steps * step
    #         intervalEnd = steadyStart + (steadyEnd - steadyStart) / steps * (step + 1)
    #         interval_times = time[(time >= intervalStart) & (time < intervalEnd)]
    #         intervalSamples, _ = find_samples_path(interval_times, txDelay, avg_interarrival_, df_name, samplingMethod, [min_brnval])
    #         if len(intervalSamples) == 0:
    #             return []
    #         aggregated_samples.extend(intervalSamples)
    #     return aggregated_samples
    #################################################
    # # Step 1: Compute interarrival times
    # interarrival = np.diff(time)

    # # Step 2: Check if the entire sequence is exponential
    # anderson_statistic, anderson_critical_values, _ = anderson(interarrival, 'expon')
    # if anderson_statistic <= anderson_critical_values[2]:
    #     print("Interarrival times are exponentially distributed.")
    #     return time

    # # Step 3: Custom binning by txDelay and maxLength
    # bins = []
    # i = 0
    # n = len(time)

    # while i < n:
    #     bin_start = i
    #     bin_times = [time[i]]
    #     i += 1
    #     while i < n:
    #         gap = time[i] - time[i - 1]
    #         span = time[i] - bin_times[0]
    #         if gap > txDelay:
    #             break
    #         bin_times.append(time[i])
    #         i += 1
    #     bins.append(np.array(bin_times))

    # # Step 4: Try random sampling from bins + exponential test
    # max_sample_size = 0
    # best_sample = None

    # for brnval in [0.1, 0.15, 0.2]:
    #     tries = 20
    #     while tries > 0:
    #         selected_indices = []
    #         for bin_times in bins:
    #             if len(bin_times) > 0:
    #                 chosen = np.random.choice(bin_times)
    #                 selected_indices.append(chosen)
    #         selected_times = np.array(sorted(selected_indices))

    #         if len(selected_times) <= 1:
    #             tries -= 1
    #             continue

    #         # First check: directly test selected times
    #         selected_interarrival = np.diff(selected_times)
    #         anderson_statistic, anderson_critical_values, _ = anderson(selected_interarrival, 'expon')
    #         if anderson_statistic <= anderson_critical_values[2]:
    #             if len(selected_times) > max_sample_size:
    #                 max_sample_size = len(selected_times)
    #                 best_sample = selected_times
    #                 break

    #         # Second check: apply Bernoulli sampling
    #         keep_mask = bernoulli.rvs(brnval, size=len(selected_times))
    #         final_times = selected_times[keep_mask == 1]

    #         if len(final_times) <= 1:
    #             tries -= 1
    #             continue

    #         anderson_statistic, anderson_critical_values, _ = anderson(np.diff(final_times), 'expon')
    #         if anderson_statistic <= anderson_critical_values[2]:
    #             if len(final_times) > max_sample_size:
    #                 max_sample_size = len(final_times)
    #                 best_sample = final_times
    #                 break
    #         tries -= 1

    # if best_sample is not None:
    #     return best_sample

    # print("Failed to find exponentially distributed interarrival times after 20 tries.")
    # return []

# def calculate_Poisson_bias
# f = lambda rate: 1 - (np.sum(-np.expm1(-rate * interarrivals)) / np.sum(rate * interarrivals))
def distanceAwareSampling(time, rate):
    interarrival = np.diff(time)
    probabilities = -np.expm1(-interarrival * rate)
    selected_mask = bernoulli.rvs(probabilities, size=len(probabilities))
    selected_times = time[1:][selected_mask == 1]
    if len(selected_times) > 1:
            return selected_times
    return []

    # interarrival = np.diff(time)
    # probabilities = -np.expm1(-interarrival * rate)
    # tries = 20
    # while tries > 0:
    #     selected_mask = bernoulli.rvs(probabilities, size=len(probabilities))
    #     selected_times = time[1:][selected_mask == 1]
    #     # check if the interarrival of selected times is exponential
    #     if len(selected_times) > 1:
    #         anderson_statistic, anderson_critical_values, _ = anderson(np.diff(selected_times), 'expon')
    #         if anderson_statistic <= anderson_critical_values[2]:
    #             return selected_times
    #     tries -= 1
    # print("Failed to find exponentially distributed interarrival times after 20 tries.")
    # return []

def poissonLikeSampling(time, rate, trsh):
    tries = 1
    interarrival = np.diff(time)
    probabilities = -np.expm1(-interarrival * rate)
    selected_mask = bernoulli.rvs(probabilities, size=len(probabilities))
    selected_times = time[1:][selected_mask == 1]
    print("Initial probabilities and interarrivals:")
    print(np.mean(probabilities), np.mean(interarrival), len(selected_times), len(time))
    
    while tries > 0:
        samples = [time[0]]
        selected_mask = []
        probabilities = []
        interarrival = []
        for t in time[1:]:
            probabilities.append(-np.expm1(-rate * (t - samples[-1])))
            interarrival.append(t - samples[-1])
            selected_mask.append(bernoulli.rvs(probabilities[-1], size=1)[0])
            if selected_mask[-1] == 1 or t - samples[-1] > trsh:
                samples.append(t)
        print("Initial probabilities and interarrivals 22:")
        print(np.mean(probabilities), np.mean(interarrival), len(samples), len(time))
        samples = samples[1:]  # remove the first element which is always selected
        if len(samples) > 1:
            anderson_statistic, anderson_critical_values, _ = anderson(np.diff(samples), 'expon')
            if anderson_statistic <= anderson_critical_values[2]:
                return np.array(samples)
        tries -= 1
    print("Failed to find exponentially distributed interarrival times after 20 tries.")
    return []

def randomSampling(time):
    tries = 20
    while tries > 0:
        selected = np.random.choice(time, size=int(0.05 * len(time)), replace=False)
        selected = np.sort(selected)
        anderson_statistic, anderson_critical_values, _ = anderson(np.diff(selected), 'expon')
        if anderson_statistic <= anderson_critical_values[2]:
            return selected
        tries -= 1
    print("Failed to find exponentially distributed interarrival times after 20 tries.")
    return []

def find_samples_path(time, txDelay, avg_interarrival_=None, df_name=None, samplingMethod='Orig', brnval_list=[0.1, 0.2, 0.3, 0.5]):
    if "P0" in df_name:
        return time
    # check the 99 percentile of the interarrival times
    # interarrival_99 = np.percentile(np.diff(time), 99)
    # # print("99 percentile of interarrival times:", interarrival_99, "txDelay:", txDelay * 1.05, "we use DA:", interarrival_99 < txDelay * 1.05)
    # if interarrival_99 < txDelay * 1.05:
    #     return distanceAwareSampling(time, 5e-7)
    # interarrival_99 = np.percentile(np.diff(time), 99)
    # if interarrival_99 > txDelay * 1.05:
    #     return distanceAwareSampling(time, 3e-7)
    
    if samplingMethod == "DA":
    # #     # rate = find_sampling_rate(time, 0.0075)
        # sampling_rate = 1 / np.quantile(np.diff(time), 0.99)
        # sampling_rate = 1 / np.mean(np.diff(time))
        sampling_rate = 2e-6
        return distanceAwareSampling(time, sampling_rate)

    # return poissonLikeSampling(time, 3e-10, 3e6)
    # return randomSampling(time)
    # return find_samples_path_new(time, txDelay, 480*1e3)  
    # Step 1: Compute interarrival times
    interarrival = np.diff(time)

    # Step 2: Check if interarrivals follow an exponential distribution
    # selection_mask = bernoulli.rvs(0.2, size=len(time))
    # temp_times = time[selection_mask == 1]
    # return temp_times
    if len(interarrival) > 1:
        anderson_statistic, anderson_critical_values, _ = anderson(interarrival, 'expon')
        if anderson_statistic <= anderson_critical_values[4]:
            print("Interarrival times are exponentially distributed.")
            return time
    
    # print("Interarrival times are not exponentially distributed. Proceeding with sampling...")

    # Step 3: Divide into chunks of average interarrival time
    # avg_interarrival = np.mean(interarrival) * 10
    # avg_interarrival  = avg_interarrival_ * 5
    avg_interarrival = avg_interarrival_
    # avg_interarrival = np.mean(interarrival[interarrival > txDelay])
    # print("Average interarrival time:", avg_interarrival)
    start_time = time[0]
    end_time = time[-1]
    bins = np.arange(start_time, end_time, avg_interarrival)
    max_sample_size = 0
    max_sample_size_times = []
    max_sample_size_brnval = 0
    # for brnval in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    for brnval in brnval_list:
        tries = 20
        final_values = []
        while tries > 0 :
            selected_indices = []
            for i in range(len(bins) - 1):
                # Get indices in the current chunk
                mask = (time >= bins[i]) & (time < bins[i+1])
                indices = np.where(mask)[0]
                if len(indices) > 0:
                    # Randomly pick one index from the chunk
                    selected_indices.append(np.random.choice(indices))

            if len(selected_indices) <= 1:
                tries -= 1
                continue
            selected_indices = np.array(selected_indices)
            selected_times = time[selected_indices]
            # print("Selected times:", len(selected_times), "from", len(time), "total packets.")
            # Step 4: Check if interarrival of selected packets is exponential
            selected_interarrival = np.diff(selected_times)
            anderson_statistic, anderson_critical_values, _ = anderson(selected_interarrival, 'expon')
            if anderson_statistic <= anderson_critical_values[4]:
                # print("Selected First interarrival times are exponentially distributed.")
                if len(selected_times) > max_sample_size:
                    max_sample_size = len(selected_times)
                    max_sample_size_times = selected_times
                    max_sample_size_brnval = brnval
                    break
                # return selected_times

            # Step 5: Use Bernoulli sampling on selected packets
            keep_mask = bernoulli.rvs(brnval, size=len(selected_times))
            final_times = selected_times[keep_mask == 1]
            # print("Final times after Bernoulli sampling:", len(final_times), "with brnval:", brnval)
            if len(final_times) <= 1:
                tries -= 1
                continue
            anderson_statistic, anderson_critical_values, _ = anderson(np.diff(final_times), 'expon')
            if anderson_statistic <= anderson_critical_values[4]:
                # print("Selected Second interarrival times are exponentially distributed.")
                # return final_times
                if len(final_times) > max_sample_size:
                    max_sample_size = len(final_times)
                    max_sample_size_times = final_times
                    max_sample_size_brnval = brnval
                    break
            tries -= 1
            # print("Tries left:", tries, "with brnval:", brnval)
        # print("after brnval:", brnval, "max_sample_size:", max_sample_size)
    if max_sample_size:
        # print("Max sample size found:", max_sample_size, len(max_sample_size_times))
        return max_sample_size_times, max_sample_size_brnval
    print("Failed to find exponentially distributed interarrival times after 20 tries.")
    return [], 0

def e2e_poisson_sampling(time, values, delay=False, sizes=None):
    duration = time[-1] - time[0]
    rate = len(values) / duration
    bound = 500
    
    inter_arrival_times = np.random.exponential(scale=1/rate, size=int(duration * rate))
    poisson_times = 3 * 1e8 + np.cumsum(inter_arrival_times)
    
    poisson_times = poisson_times[poisson_times <= time[-1]]
    poisson_times = poisson_times[poisson_times >= time[0]]
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
            if delay is False:
                selected.append(values[closest])
            else:
                if time[closest] <= t:
                    selected.append(max(values[closest] + sizes[closest] - (t - time[closest]), 0))
                else:
                    selected.append(values[closest] + (time[closest] - t))

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

def calculate_offline_E2E_markingProb(full_df, df_res, checkColumn, txDelay, swtichDstREDQueueDiscMaxSize, linkRate, __ns3_path, tsh, df_name, passiveProbe, samplingMethod, steadyStart, steadyEnd):
    # timeAvg_methods = ['rightCont_timeAvg', 'leftCont_timeAvg', 'linearInterp_timeAvg']
    # nonMarkingProb_timeAvg_vars = ['event_currentProb', 'event_lastProb']
    df_res['nonMarkingProb'] = {}
    for var in ['event']:
        for method in ['rightCont_timeAvg', 'leftCont_timeAvg', 'linearInterp_timeAvg', 'poisson_eventAvg', 'eventAvg']:
            df_res['nonMarkingProb'][var + '_' + method] = {}
    
    full_df_ = full_df[full_df['SentTime'] != -1].copy()
    df_res['sampleSize']['nonMarkingProb'] = {}
    df_res['bias']['nonMarkingProb'] = {}
    for path in full_df_['Path'].unique():
        df = full_df_[full_df_['Path'] == path].copy()
        df_res['bias']['nonMarkingProb'][path] = 0
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

        if passiveProbe:
            interarrival = np.diff(time)
            anderson_statistic, anderson_critical_values, _ = anderson(interarrival, 'expon')
            if anderson_statistic <= anderson_critical_values[2]:
                samples_times = time
            else:
                print("Sample times are 'NOT' exponentially distributed.")
                samples_times = []
        else:
            # samples_times = find_samples_path(time, txDelay, df_res['RTT'][path], df_name, samplingMethod)
            samples_times = find_samples_path_new(time, txDelay, df_res['RTT'][path], df_name, samplingMethod, steadyStart, steadyEnd, steps=1)
        df_res['sampleSize']['nonMarkingProb'][path] = len(samples_times)
        samples_values = df[df['SentTime'].isin(samples_times)]['nonMarking'].values
        if df_res['sampleSize']['nonMarkingProb'][path] == 0:
            avg, std = 0, 0
        else:
            avg, std = np.mean(samples_values), np.std(samples_values) / np.sqrt(len(samples_values))
        df_res['nonMarkingProb']['event_poisson_eventAvg'][path] = (avg, std)

        df_res['nonMarkingProb']['event_eventAvg'][path] = (np.mean(values), np.std(values) / np.sqrt(len(values)))
    full_df_ = None
    return df_res

def calculate_offline_E2E_delays(full_df, removeDrops, checkColumn, txDelay, df_res, df_name, passiveProbe, samplingMethod, steadyStart, steadyEnd, 
                                 samples_paths_aggregated_statistics=None, queue_names=None, linkDelays=None, linkRates=None, queue_size_trshs=None):
    df_res['delay'] = {}
    for var in ['event']:
        for method in ['rightCont_timeAvg', 'leftCont_timeAvg', 'linearInterp_timeAvg', 'poisson_eventAvg', 'eventAvg']:
            df_res['delay'][var + '_' + method] = {}
    
    full_df_ = full_df.copy()
    if removeDrops:
        full_df_ = full_df_[full_df_[checkColumn] == 1]
    df_res['sampleSize']['delay'] = {}
    df_res['subSamplingError']['delay'] = {}
    df_res['bias']['delay'] = {}
    df_res['Corr'] = {}
    result = {}
    for path in full_df_['Path'].unique():
        df = full_df_[full_df_['Path'] == path]
        df = df.sort_values(by='SentTime').reset_index(drop=True)
        df_res['totalPckts'][path] = len(df)
        time = df['SentTime'].values
        values = df['Delay'].values
        subSamplingError = SubSamplingError.NoError
        rightCont_time_average = np.sum(values[:-1] * np.diff(time)) / (time[-1] - time[0])
        df_res['delay']['event_rightCont_timeAvg'][path] = rightCont_time_average

        leftCont_time_average = np.sum(values[1:] * np.diff(time)) / (time[-1] - time[0])
        df_res['delay']['event_leftCont_timeAvg'][path] = leftCont_time_average

        linearInterp_time_average = np.sum(((values[:-1] + values[1:]) / 2) * np.diff(time)) / (time[-1] - time[0])
        df_res['delay']['event_linearInterp_timeAvg'][path] = linearInterp_time_average
        # print("Calculating delay for path:", path, "with", len(time), "packets.")
        if passiveProbe:
            interarrival = np.diff(time)
            anderson_statistic, anderson_critical_values, _ = anderson(interarrival, 'expon')
            if anderson_statistic <= anderson_critical_values[2]:
                samples_times = time
            else:
                print("Sample times are 'NOT' exponentially distributed.")
                samples_times = []
        else:
            # samples_times = find_samples_path(time, txDelay, df_res['RTT'][path], df_name, samplingMethod)
            if path in samples_paths_aggregated_statistics.keys():
                if samples_paths_aggregated_statistics[path]['MinimumE2ESampleSize'] is not None:
                #    samples_times, subSamplingError = find_samples_path_new(time, txDelay, df_res['RTT'][path], df_name, samplingMethod, steadyStart, steadyEnd, steps=1, MinimumNumberOfSamples=samples_paths_aggregated_statistics[path]['MinimumE2ESampleSize'])
                    # samples_times, subSamplingError, result = find_samples_path_ccf(time, steadyStart, steadyEnd, queue_names, df_name, linkDelays, linkRates, queue_size_trshs, samples_paths_aggregated_statistics[path]['MinimumE2ESampleSize'])
                    samples_times, subSamplingError, result = find_samples_path_chi_squared_test(time, steadyStart, steadyEnd, samples_paths_aggregated_statistics[path]['MinimumE2ESampleSize'])
                else:
                    samples_times = []
            else:
                samples_times = []
        df_res['Corr'][path] = result
        samples = df[df['SentTime'].isin(samples_times)]
        df_res['bias']['delay'][path] = (samples['PayloadSize'] - (samples['BitsTag'] / 8)).mean()
        # print("Calculating delay for path:", path, "with", len(time), "packets. is done! ")
        df_res['sampleSize']['delay'][path] = len(samples_times)
        df_res['subSamplingError']['delay'][path] = subSamplingError
        samples_values = df[df['SentTime'].isin(samples_times)]['Delay'].values
        # print(df[df['SentTime'].isin(samples_times)])
        # samples_packetSizes = df[df['SentTime'].isin(samples_times)]['PayloadSize'].values
        # print("samples_times: ", len(samples_times))
        # # plot the sample values and times in a scatter plot
        # plt.scatter(samples_times, (samples_values / 8 * 0.97*600*1e-3) + samples_packetSizes)
        # plt.ylim(0, 19000)
        # plt.xlabel('Sent Time')
        # plt.ylabel('Queue size(B)')
        # plt.title(f'Queue size Samples for Path {path}')
        # plt.savefig(f'Queue_size_path_{path}.png')
        # plt.close()
        if df_res['sampleSize']['delay'][path] == 0:
            df_res['InterArrivals'][path] = np.nan
            avg, std = 0, 0
        else:
            df_res['InterArrivals'][path] = np.diff(samples_times).mean()
            avg, std = np.mean(samples_values), np.std(samples_values) / np.sqrt(len(samples_values))
        df_res['delay']['event_poisson_eventAvg'][path] = (avg, std)
        # print("Path:", path, "E2E Delay Average from Poisson-sampled SentTimes:", avg, "std/Rn:", std)
        df_res['delay']['event_eventAvg'][path] = (np.mean(values), np.std(values) / np.sqrt(len(values)))
        df = None
    full_df_ = None
    return df_res

def prune_data(full_df, projectColumn, steadyStart, steadyEnd):
    full_df = full_df[full_df[projectColumn] >= steadyStart]
    full_df = full_df[full_df[projectColumn] <= steadyEnd]
    full_df = full_df.sort_values(by=[projectColumn], ignore_index=True)
    return full_df

def addExtraDelay(full_df, differentiationDelay, errorRate):
    if differentiationDelay > 0:
        extra_delay_indices = full_df.sample(frac=errorRate).index
        full_df.loc[extra_delay_indices, 'Delay'] += np.int64(full_df.loc[extra_delay_indices, 'Delay'] * differentiationDelay)
    return full_df

def addPacketsFromOtherPaths(full_df, errorRate, fromPath, toPath):
    if errorRate > 0:
        extra_delay_indices = full_df[full_df['Path'] == fromPath].sample(frac=errorRate).index
        full_df.loc[extra_delay_indices, 'Path'] = toPath
    return full_df

def addRemoveTransmission_data(full_df, linkDelays, linksRates):
    full_df['Delay'] = abs(full_df['ReceiveTime'] - full_df['SentTime'] - full_df['transmissionDelay'])
    # full_df['Delay'] = abs(full_df['ReceiveTime'] - full_df['TxEnqueueTime'] - full_df['transmissionDelay'])
    # full_df['Time'] = full_df['SentTime']
    full_df['SentTime'] = full_df['SentTime'] + linkDelays[0] + (full_df['PayloadSize'] * 8) / linksRates[0]
    # round the SentTime to the nearest integer
    full_df['SentTime'] = full_df['SentTime'].apply(lambda x: int(round(x)))
    # full_df['SentTime'] = full_df['TxEnqueueTime']
    # if there are multiple rows with the same Id, keep only the one with IsReceived == 1
    full_df = full_df.sort_values("IsReceived", ascending=False)
    full_df = full_df.drop_duplicates(subset="Id", keep="first")
    full_df = full_df.sort_values(by=['SentTime']).reset_index(drop=True)
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
    

def manipulate_for_delay_Q(full_df, linkRate, steadyStart, steadyEnd, experiment):
    full_df = full_df.sort_values(by=['Time', 'TotalQueueSize', 'Action'], ascending=[True, True, False]).reset_index(drop=True)
    # if there is still a D action with the same time with an E action, but it is after E in the dataframe, we need to swap them
    full_df['Action_shifted'] = full_df['Action'].shift(-1)
    swap_mask = (full_df['Action'] == 'D') & (full_df['Action_shifted'] == 'E') & (full_df['Time'] == full_df['Time'].shift(-1)) & (full_df['Label'] == full_df['Label'].shift(-1))
    indices_to_swap = full_df[swap_mask].index
    for idx in indices_to_swap:
        full_df.at[idx, 'Action'], full_df.at[idx + 1, 'Action'] = full_df.at[idx + 1, 'Action'], full_df.at[idx, 'Action']
    full_df = full_df.drop(columns=['Action_shifted'])
    
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
    full_df = full_df.sort_values(by=['Time', 'TotalQueueSize', 'Action'], ascending=[True, True, False]).reset_index(drop=True)
    full_df['Delay'] = ((full_df['TotalQueueSize'] * 8) / linkRate).astype(int)
    time = full_df['Time'].values
    actions = full_df['Action'].values
    # values = full_df['QueuingDelay'].values
    values = full_df['Delay'].values

    linear_sum = 0
    temp_df = pd.DataFrame()
    for i in range(len(values[:-1])):
        x_1 = values[i]
        dt = time[i + 1] - time[i]
        if actions[i + 1] == 'E':
            if x_1 > 0:
                x_2 = x_1 - dt
                if x_2 < -1:
                    temp_df = pd.concat([temp_df, full_df.iloc[i:i+2]])
            else:
                x_2 = 0
        else:
            x_2 = values[i + 1]
        linear_sum += (x_1 + x_2) / 2 * dt
    if len(temp_df) > 0:
        print("Experiment:", experiment)
        print("temp_df", temp_df)
    # linearInterp_time_average = linear_sum / (time[-1] - time[0])
    linearInterp_time_average = linear_sum / (steadyEnd - steadyStart)
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

def plot_queuingDelay_distribution(__ns3_path, results_folder, rate, experiment, segment, steadyStart, steadyEnd, paths, linksRates, linkDelays, ks_dict):
    file_paths = glob.glob('{}/scratch/{}/{}/{}/*_{}.csv'.format(__ns3_path, results_folder, rate, experiment, segment))
    dfs = {}
    for file_path in file_paths:
        file_M = "/".join(file_path.split('/')[:-1] + ['A0D0_EndToEnd_packets.csv'])
        file_switch = "/".join(file_path.split('/')[:-1] + ['SD0_PoissonSampler_events.csv'])
        # file_switch = "/".join(file_path.split('/')[:-1] + ['SD0_PoissonSampler_queueSize.csv'])
        full_df_switch = pd.read_csv(file_switch)
        # full_df_switch = full_df_switch[full_df_switch['Action'] == 'E'].copy().reset_index(drop=True)
        full_df_switch = full_df_switch.sort_values(by=['Time', 'TotalQueueSize'], ascending=[True, True]).reset_index(drop=True)
        full_df_switch['Delay'] = ((full_df_switch['TotalQueueSize'] * 8) / linksRates[1]).astype(int)

        full_df_M = pd.read_csv(file_M)
        full_df_M = addRemoveTransmission_data(full_df_M, linkDelays, linksRates)
        full_df_M = prune_data(full_df_M, 'SentTime', steadyStart, steadyEnd)
        full_df_M = full_df_M.sort_values(by=['SentTime'], ascending=[True]).reset_index(drop=True)
        
        time = full_df_M['SentTime'].values
        # full_df_M = full_df_switch[full_df_switch['Label'].str.contains('10.1.1.1', na=False)]
        # time = full_df_M['Time'].values

        samples_times = find_samples_path(time, 0, 200000, "Poisson", samplingMethod='Orig')
        samples_values = full_df_M[full_df_M['SentTime'].isin(samples_times)]['Delay'].values
        # samples_values = full_df_M[full_df_M['Time'].isin(samples_times)]['Delay'].values
        d3 = np.asarray(samples_values, dtype=float)
        d3 = d3[np.isfinite(d3)]

        distanceAwareSampling_samples_times = find_samples_path(time, 0, 200000, "Poisson", samplingMethod='DA')
        distanceAwareSampling_samples_values = full_df_M[full_df_M['SentTime'].isin(distanceAwareSampling_samples_times)]['Delay'].values
        d4 = np.asarray(distanceAwareSampling_samples_values, dtype=float)
        d4 = d4[np.isfinite(d4)]

        d1 = np.asarray(full_df_switch['Delay'], dtype=float)
        d1 = d1[np.isfinite(d1)]
        d2 = np.asarray(full_df_M['Delay'], dtype=float)
        d2 = d2[np.isfinite(d2)]
        p_value_all = ks_2samp(d1, d2).pvalue
        p_value_sampling = ks_2samp(d1, d3).pvalue
        p_value_sampling_da = ks_2samp(d1, d4).pvalue
        # ks_dict[str(experiment) + "_da"] = int(p_value_sampling_da > 0.05)
        # ks_dict[str(experiment)] = int(p_value_sampling > 0.05)
        # print(experiment, "KS p-value (Switch vs samples):", p_value_sampling > 0.05, "KS p-value (Switch vs Distance Aware samples):", p_value_sampling_da > 0.05)
        # print(experiment, "is from the same dist: ", p_value_all > 0.05, p_value_all, " After Sampling: ", p_value > 0.05, p_value)
        # ---------------- CDF with percentile zoom + tail annotation ----------------
        plt.figure(figsize=(10, 6))

        # Build CDFs (sorted values + empirical probabilities)
        x1 = np.sort(np.asarray(full_df_switch['Delay'].values, dtype=float))
        x1 = x1[np.isfinite(x1)]
        y1 = np.arange(1, len(x1) + 1) / max(len(x1), 1)

        x2 = np.sort(np.asarray(full_df_M['Delay'].values, dtype=float))
        x2 = x2[np.isfinite(x2)]
        y2 = np.arange(1, len(x2) + 1) / max(len(x2), 1)

        x3 = np.sort(d3)
        y3 = np.arange(1, len(x3) + 1) / max(len(x3), 1)

        x4 = np.sort(d4)
        y4 = np.arange(1, len(x4) + 1) / max(len(x4), 1)

        # Plot CDFs
        plt.step(x1, y1, where='post', label="Samples at the switch", color='b', alpha=0.9)
        plt.step(x2, y2, where='post', label="Measurement Traffic", color='r', alpha=0.9)
        plt.step(x3, y3, where='post', label="Samples from Measurement", color='g', alpha=0.9)
        plt.step(x4, y4, where='post', label="Distance Aware Sampling from Measurement", color='m', alpha=0.9)
        # Percentile to show (zoom)
        p = 0.995  # 99.5th percentile; adjust (e.g., 0.99 or 0.999) as needed

        # Use a common x-limit based on both datasets so scales match
        combined = np.concatenate([x1, x2, x3]) if (len(x1) and len(x2) and len(x3)) else (x1 if len(x1) else x2)
        if combined.size:
            x_right = np.quantile(combined, p)
            plt.xlim(left=0, right=x_right)

            # Tail fractions beyond x_right for each series
            # tail1 = (x1 > x_right).sum() / max(len(x1), 1)
            # tail2 = (x2 > x_right).sum() / max(len(x2), 1)
            # tail3 = (x3 > x_right).sum() / max(len(x3), 1)
            # Annotate tails (place inside axes, bottom-right corner)
            # txt = (f"{(1-p)*100:.2f}% > {x_right:.3g} (combined cutoff)\n"
            #        f"Switch tail: {tail1*100:.2f}%\n"
            #        f"Meas. tail: {tail2*100:.2f}%\n"
            #        f"Samples tail: {tail3*100:.2f}%\n"
            #        f"KS p-val (Switch vs Meas.): {p_value_all:.3f}\n"
            #          f"KS p-val (Switch vs Samples): {p_value:.3f}")
            txt = (f"{(1-p)*100:.2f}% > {x_right:.3g} (combined cutoff)\n"
                   f"KS p-val (Switch vs Meas.): {p_value_all:.3f}\n"
                     f"KS p-val (Switch vs Samples): {p_value_sampling:.3f}"
                     f"\nKS p-val (Switch vs Distance Aware Samples): {p_value_sampling_da:.3f}")
            plt.text(0.98, 0.05, txt, ha='right', va='bottom',
                     transform=plt.gca().transAxes, fontsize=11,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, lw=0))
        else:
            # No data: keep defaults to avoid errors
            plt.xlim(auto=True)

        plt.ylim(0, 1.05)
        plt.title('Queuing Delay CDF', fontsize=16)
        plt.xlabel('Delay', fontsize=16)  # not normalized anymore; add units if known (e.g., 'Delay (s)')
        plt.ylabel('Cumulative Probability', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, which='both', axis='both', alpha=0.5)
        plt.legend(fontsize=12)
        plt.savefig('{}/scratch/{}/{}/{}/queuingDelay_distribution.png'.format(__ns3_path, results_folder, rate, experiment, segment))
        plt.close()

def plot_interarrival_distribution(__ns3_path, results_folder, rate, experiment, segment, steadyStart, steadyEnd, onlyMeasurement):
    file_paths = glob.glob('{}/scratch/{}/{}/{}/*_{}.csv'.format(__ns3_path, results_folder, rate, experiment, segment))
    dfs = {}
    for file_path in file_paths:
        full_df = pd.read_csv(file_path)
        full_df = prune_data(full_df, 'Time', steadyStart, steadyEnd)
        if onlyMeasurement:
            full_df = full_df[full_df['Label'].str.contains('10.1.1.1', na=False)]
        full_df = full_df[full_df['Action'] == 'E'].copy()
        full_df = full_df.sort_values(by=['Time', 'TotalQueueSize'], ascending=[True, True]).reset_index(drop=True)
        full_df['InterArrival'] = full_df['Time'].diff().fillna(0)
        # plot the distribution of the queuing delay
        plt.figure(figsize=(10, 6))
        plt.hist(full_df['InterArrival'], bins=200, density=True, color='g')
        # plot the mean as a vertical line with its value
        mean = full_df['InterArrival'].mean()
        plt.axvline(mean, color='r', linestyle='dashed', linewidth=1)
        plt.text(mean, 0, 'Mean: {:.2f}'.format(mean), color='r', fontsize=12)
        plt.title('Interarrivals Distribution', fontsize=16)
        plt.xlabel('Interarrivals (ns)', fontsize=16)
        plt.ylabel('Density', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid()
        if onlyMeasurement:
            plt.savefig('{}/scratch/{}/{}/{}/interarrivalsOfMeasurmentTraffic_distribution.png'.format(__ns3_path, results_folder, rate, experiment, segment))
        else:
            plt.savefig('{}/scratch/{}/{}/{}/interarrivals_distribution.png'.format(__ns3_path, results_folder, rate, experiment, segment))

def compute_mixing_poisson_e2e(switch_df, traffics_df, RTT):
    time = traffics_df['SentTime'].values
    samples_times = find_samples_path(time, 0, RTT)
    samples_times = samples_times.astype(int)
    switch_df = switch_df.copy()  # avoid modifying original
    for sample in samples_times:
        samples_times = np.append(samples_times, [sample + 1, sample - 1])
    samples_times = np.sort(samples_times)
    switch_df.loc[switch_df['Time'].isin(samples_times), 'Label'] = '10.0.0.0'
    return compute_mixing_selected_traffic(switch_df, '10.0.0.0')
    
def compute_mixing_poisson_switch(switch_df, traffics_df):
    switch_df = switch_df.copy()  # avoid modifying original
    traffics_df = traffics_df.copy()  # avoid modifying original
    switch_df = switch_df.drop(columns=['QueuingDelay', 'DropProb', 'MarkingProb', 'QueueSize', 'LastMarkingProb', 'LastDropProb', 'LastQueueSize', 'LastTotalQueueSize'])
    traffics_df = traffics_df.drop(columns=['QueuingDelay', 'DropProb', 'MarkingProb', 'QueueSize', 'LastMarkingProb', 'Action'])
    switch_df['Label'] = '10.0.0.0'
    concatenated_df = pd.concat([switch_df, traffics_df], ignore_index=True)
    concatenated_df = concatenated_df.sort_values(by=['Time', 'TotalQueueSize'], ascending=[True, True]).reset_index(drop=True)
    return compute_mixing_selected_traffic(concatenated_df, '10.0.0.0')
    
def compute_mixing_poissonEventAvg_switch(switch_df, traffics_df):
    switch_df = switch_df.copy()  # avoid modifying original
    traffics_df = traffics_df.copy()  # avoid modifying original
    switch_df = switch_df.drop(columns=['QueuingDelay', 'DropProb', 'MarkingProb', 'QueueSize', 'LastMarkingProb', 'LastDropProb', 'LastQueueSize', 'LastTotalQueueSize'])
    traffics_df = traffics_df.drop(columns=['QueuingDelay', 'DropProb', 'MarkingProb', 'QueueSize', 'LastMarkingProb', 'Action'])
    switch_df['Label'] = '10.0.0.0'
    concatenated_df = pd.concat([switch_df, traffics_df], ignore_index=True)
    concatenated_df = concatenated_df.sort_values(by=['Time', 'TotalQueueSize'], ascending=[True, True]).reset_index(drop=True)
    # where the label is '10.0.0.0', if TotalQueueSize is zero, then add -1 * Signal[-1] to Signal, else add 1 if the previous label was '10.1.1.1', -1 if it was not
    labels = concatenated_df['Label'].astype(str)
    sizes = concatenated_df['TotalQueueSize'].values
    Signal = [1 if '10.1.1.1' in labels[0] else -1]
    last_label = labels[0]
    for i in range(0, len(labels)):
        if ('10.0.0.0' in labels[i]):
            if sizes[i] == 0:
                Signal.append(-1 * Signal[-1])
            else:
                if '10.1.1.1' in last_label:
                    Signal.append(1)
                else:
                    Signal.append(-1)
        else:
            last_label = labels[i]

    Signal = np.array(Signal)
    sign_changes = Signal[1:] != Signal[:-1]
    rate = np.sum(sign_changes) / (len(Signal) - 1)
    return rate

def compute_mixing_selected_traffic(df, traffic):
    df = df.copy()  # avoid modifying original
    labels = df['Label'].astype(str)
    sizes = df['TotalQueueSize'].values
    SignalMOnly = []
    for i in range(0, len(labels)):
        if (traffic in labels[i]):
            if (len(SignalMOnly) == 0):
                SignalMOnly.append(1)
            else:
                if i > 0:
                    if (traffic in labels[i - 1]):
                        SignalMOnly.append(1)
                    else:
                        if sizes[i] >= sizes[i - 1]:
                            SignalMOnly.append(1)
                        else:
                            SignalMOnly.append(-1)
                if i < len(labels) - 1:
                    if (traffic in labels[i + 1]):
                        SignalMOnly.append(1)
                    else:
                        if sizes[i] > sizes[i + 1]:
                            SignalMOnly.append(1)
                        else:
                            SignalMOnly.append(-1)
    SignalMOnly = np.array(SignalMOnly)
    SignalMOnly_sign_changes = SignalMOnly[1:] != SignalMOnly[:-1]
    SignalMOnly_rate = np.sum(SignalMOnly_sign_changes) / (len(SignalMOnly) - 1)
    return SignalMOnly_rate

def compute_timeAverage_mixingRate(df):
    df = df.copy()  # avoid modifying original
    labels = df['Label'].astype(str)
    sizes = df['TotalQueueSize'].values
    times = df['Time'].values

    Signal = []
    Signal.append(1 if "10.1.1.1" in labels[0] else -1)

    for i in range(1, len(labels)):
        if ("10.1.1.1" in labels[i] and "10.1.1.1" in labels[i - 1]) or ("10.1.1.1" not in labels[i] and "10.1.1.1" not in labels[i - 1]):
            if ("10.1.1.1" in labels[i]):
                Signal.append(1)
            else:
                Signal.append(-1)
        else:
            if "10.1.1.1" in labels[i]:
                if sizes[i] > sizes[i - 1]:
                    Signal.append(1)
                if sizes[i] < sizes[i - 1]:
                    Signal.append(-1)
                if sizes[i] == sizes[i - 1]:
                    Signal.append((-1) * Signal[-1])
            else:
                if sizes[i] > sizes[i - 1]:
                    Signal.append(-1)
                if sizes[i] < sizes[i - 1]:
                    Signal.append(1)
                if sizes[i] == sizes[i - 1]:
                    Signal.append((-1) * Signal[-1])
    
    Signal = np.array(Signal)
    sign_changes = Signal[1:] != Signal[:-1]
    # compute the time average of the Signal changes
    time_diffs = np.diff(times)
    time_avg = np.sum(sign_changes * time_diffs) / (times[-1] - times[0])
    return time_avg

# def compute_mixingRate_intervalsAvg

def compute_S_column(df):
    df = df.copy()  # avoid modifying original
    labels = df['Label'].astype(str)
    sizes = df['TotalQueueSize'].values

    Signal = []
    Signal.append(1 if "10.1.1.1" in labels[0] else -1)

    SignalMOnly = []

    DifferenceDelay = []
    DifferenceDelay.append(0)

    for i in range(0, len(labels)):
        if ("10.1.1.1" in labels[i]):
            if (len(SignalMOnly) == 0):
                SignalMOnly.append(1)
            else:
                if i > 0:
                    if ("10.1.1.1" in labels[i - 1]):
                        SignalMOnly.append(1)
                    else:
                        if sizes[i] >= sizes[i - 1]:
                            SignalMOnly.append(1)
                        else:
                            SignalMOnly.append(-1)
                if i < len(labels) - 1:
                    if ("10.1.1.1" in labels[i + 1]):
                        SignalMOnly.append(1)
                    else:
                        if sizes[i] > sizes[i + 1]:
                            SignalMOnly.append(1)
                        else:
                            SignalMOnly.append(-1)
        if i == 0:
            continue
        if ("10.1.1.1" in labels[i] and "10.1.1.1" in labels[i - 1]) or ("10.1.1.1" not in labels[i] and "10.1.1.1" not in labels[i - 1]):
            if ("10.1.1.1" in labels[i]):
                Signal.append(1)
            else:
                Signal.append(-1)
            DifferenceDelay.append((sizes[i] + sizes[i - 1]) / 2)
        else:
            if "10.1.1.1" in labels[i]:
                if sizes[i] > sizes[i - 1]:
                    Signal.append(1)
                if sizes[i] < sizes[i - 1]:
                    Signal.append(-1)
                if sizes[i] == sizes[i - 1]:
                    Signal.append((-1) * Signal[-1])
                
                DifferenceDelay.append(sizes[i] - sizes[i - 1])
            else:
                if sizes[i] > sizes[i - 1]:
                    Signal.append(-1)
                if sizes[i] < sizes[i - 1]:
                    Signal.append(1)
                if sizes[i] == sizes[i - 1]:
                    Signal.append((-1) * Signal[-1])
                
                DifferenceDelay.append(sizes[i - 1] - sizes[i])


    if len(Signal) != len(labels):
        print("Warning: S column contains values other than 1 or -1", len(Signal), len(labels))
    Signal = np.array(Signal)
    sign_changes = Signal[1:] != Signal[:-1]
    rate = np.sum(sign_changes) / (len(Signal) - 1)

    SignalMOnly = np.array(SignalMOnly)
    SignalMOnly_sign_changes = SignalMOnly[1:] != SignalMOnly[:-1]
    SignalMOnly_rate = np.sum(SignalMOnly_sign_changes) / (len(SignalMOnly) - 1)

    #compute the time average of the Signal
    time = df['Time'].values
    SignalAvg = np.sum(Signal[1:] * np.diff(time)) / (time[-1] - time[0])

    differenceDelayAvg = np.sum(DifferenceDelay[1:] * np.diff(time)) / (time[-1] - time[0])
    # print(f"Sign Change Rate: {rate} Time Average: {time_avg}")
    return rate, SignalAvg, differenceDelayAvg, SignalMOnly_rate

def computeMixingRate(__ns3_path, results_folder, rate, experiment, segment, steadyStart, steadyEnd, linksRates=[], linkDelays=[]):
    file_paths = glob.glob('{}/scratch/{}/{}/{}/*_{}.csv'.format(__ns3_path, results_folder, rate, experiment, segment))
    dfs = {}
    for file_path in file_paths:
        full_df = pd.read_csv(file_path)
        full_df = prune_data(full_df, 'Time', steadyStart, steadyEnd)
        full_df = full_df[full_df['Action'] == 'E'].copy().reset_index(drop=True)
        SigneChangeRate, SignalAvg, DelayDiff, SignalMOnly_rate = compute_S_column(full_df)
        df_name = file_path.split('/')[-1].split('_')[0]
        dfs[df_name] = {}
        # dfs[df_name]['SigneChangeRateTimeAvg'] = compute_timeAverage_mixingRate(full_df)
        dfs[df_name]['SigneChangeRate'] = SigneChangeRate
        dfs[df_name]['SignalAvg'] = SignalAvg
        dfs[df_name]['DelayDiff'] = DelayDiff
        dfs[df_name]['SigneChangeRateMOnly'] = SignalMOnly_rate
        # dfs[df_name]['SigneChangeRatePoisson'] = compute_mixing_poisson_switch(pd.read_csv('{}/scratch/{}/{}/{}/{}_PoissonSampler_events.csv'.format(__ns3_path, results_folder, rate, experiment, df_name)), full_df)
        # dfs[df_name]['SigneChangeRatePoissonEventAvg'] = compute_mixing_poissonEventAvg_switch(pd.read_csv('{}/scratch/{}/{}/{}/{}_PoissonSampler_events.csv'.format(__ns3_path, results_folder, rate, experiment, df_name)), full_df)
        # full_df_M = pd.read_csv('{}/scratch/{}/{}/{}/A0D0_EndToEnd_packets.csv'.format(__ns3_path, results_folder, rate, experiment))
        # full_df_M = addRemoveTransmission_data(full_df_M, linkDelays, linksRates)
        # full_df_M = prune_data(full_df_M, "SentTime", steadyStart, steadyEnd)
        # dfs[df_name]['SigneChangeRateE2EPoisson'] = compute_mixing_poisson_e2e(full_df, full_df_M, 2 * np.sum(linkDelays))
    return dfs

def plot_queuingDelay_time_new(__ns3_path, results_folder, rate, experiment, segment, steadyStart, steadyEnd, paths, linkRate):
    file_paths = glob.glob('{}/scratch/{}/{}/{}/*_{}.csv'.format(__ns3_path, results_folder, rate, experiment, segment))
    dfs = {}
    for file_path in file_paths:
        full_df = pd.read_csv(file_path)
        full_df = prune_data(full_df, 'Time', steadyStart, steadyEnd)
        full_df = full_df[full_df['Action'] == 'E'].copy().reset_index(drop=True)
        full_df = full_df.sort_values(by=['Time', 'TotalQueueSize'], ascending=[True, True]).reset_index(drop=True)

        # --- plot all traffic classes (each Label) ---
        plt.figure(figsize=(10, 6))
        full_df['Label'] = full_df['Label'].str.split(':').str[0]
        labels = (
            full_df['Label']
            .dropna()
            .astype(str)
            .unique()
        )
        cmap = plt.get_cmap('tab20')
        color_map = {lab: cmap(i % 20) for i, lab in enumerate(sorted(labels))}

        for lab in sorted(labels):
            g = full_df[full_df['Label'].str.contains(lab, na=False)]
            plt.scatter(g['Time'], g['TotalQueueSize'],
                        s=3,
                        marker='o',
                        color=color_map[lab],
                        label=lab)

        # (optional) keep your sampling logic for a specific label, or do it per label
        # Example: sample on 10.1.1.1 if present
        target_label = '10.1.1.1'
        if target_label in labels:
            g = full_df[full_df['Label'].str.contains(target_label, na=False)]
            time = g['Time'].values.astype(float)
            samples_times = find_samples_path(time, 0, 200000, "Poisson")
            samples_values = g[g['Time'].isin(samples_times)]['TotalQueueSize'].values
            plt.scatter(samples_times, samples_values, color='k', marker='^', s=10, label=f'Sampled {target_label}')

        # axes/limits/grids (as before)
        plt.ylim(0, 19000)
        steadyStart_plot = 0.3 * 1e9
        steadyEnd_plot   = 0.8 * 1e9
        plt.xlim(steadyStart_plot, steadyEnd_plot)
        plt.yticks(np.linspace(0, 19000, 20))
        plt.grid(axis='y')
        plt.title('Queue Size per time', fontsize=16)
        plt.xlabel('Time (ns)', fontsize=16)
        plt.ylabel('Size (B)', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(ncol=2, fontsize=9)  # multi-column legend for many labels
        plt.savefig('{}/scratch/{}/{}/{}/queuingDelay_time_{}_{}.png'.format(__ns3_path, results_folder, rate, experiment, segment, steadyStart_plot, steadyEnd_plot))

def plot_queuingDelay_time(__ns3_path, results_folder, rate, experiment, segment, steadyStart, steadyEnd, paths, linkRates, maxQueueSize):
    # plot_queuingDelay_time_new(__ns3_path, results_folder, rate, experiment, segment, steadyStart, steadyEnd, paths, linkRate)
    # return
    file_paths = glob.glob('{}/scratch/{}/{}/{}/*_{}.csv'.format(__ns3_path, results_folder, rate, experiment, segment))
    dfs = {}
    for file_path in file_paths:
        full_df = pd.read_csv(file_path)
        full_df = prune_data(full_df, 'Time', steadyStart, steadyEnd)
        # full_df = prune_data(full_df, 'Time', 450 * 1e6, 500 * 1e6)
        full_df = full_df[full_df['Action'] == 'E'].copy().reset_index(drop=True)
        full_df_M = full_df[full_df['Label'].str.contains('10.1.1.1', na=False)]
        full_df_CT = full_df[~full_df['Label'].str.contains('10.1.1.1', na=False)]
        full_df_M = full_df_M.sort_values(by=['Time', 'TotalQueueSize'], ascending=[True, True]).reset_index(drop=True)

        time = full_df_M['Time'].values.astype(float)
        # samples_times = distanceAwareSampling(time, 3e-7)
        # samples_values = full_df_M[full_df_M['Time'].isin(samples_times)]['TotalQueueSize'].values
        # samples_times = full_df[full_df['Label'].str.contains('10.4.1.1', na=False)]['Time'].values
        # samples_values = full_df[full_df['Label'].str.contains('10.4.1.1', na=False)]['TotalQueueSize'].values

        full_df_CT = full_df_CT.sort_values(by=['Time', 'TotalQueueSize'], ascending=[True, True]).reset_index(drop=True)
        # full_df_M['Delay'] = ((full_df_M['TotalQueueSize'] * 8) / linkRate).astype(int)
        # full_df_CT['Delay'] = ((full_df_CT['TotalQueueSize'] * 8) / linkRate).astype(int)
        # plot the queueing delay over time with different colors for different labels
        # SigneChangeRate = compute_S_column(full_df)
        # print(f"Sign Change Rate for {experiment} : {SigneChangeRate} Rate")
        plt.figure(figsize=(10, 6))
        # plt.scatter(full_df_M['Time'], full_df_M['Delay'], color='r', label='Measurement Traffic', marker='o', s=3)
        # plt.scatter(full_df_CT['Time'], full_df_CT['Delay'], color='b', label='Cross Traffic', marker='x', s=1)
        plt.scatter(full_df_M['Time'], full_df_M['TotalQueueSize'], color='r', label='Measurement Traffic', marker='o', s=3)
        plt.scatter(full_df_CT['Time'], full_df_CT['TotalQueueSize'], color='b', label='Cross Traffic', marker='x', s=1)
        # plt.scatter(samples_times, samples_values, color='g', label='Sampled Traffic', marker='^', s=10)
        plt.ylim(0, maxQueueSize * 0.75)
        # add the mean and variance of all the delays
        # mean_full = full_df['TotalQueueSize'].mean()
        # std_full = full_df['TotalQueueSize'].std()
        # plt.axhline(mean_full, color='g', linestyle='dashed', linewidth=1, label='Mean: {:.2f} B'.format(mean_full))
        # plt.axhline(mean_full + std_full, color='g', linestyle='dotted', linewidth=1, label='Mean + Std: {:.2f} B'.format(mean_full + std_full))
        # plt.axhline(mean_full - std_full, color='g', linestyle='dotted', linewidth=1, label='Mean - Std: {:.2f} B'.format(mean_full - std_full))
        # steadyStart_plot = 0.505 * 1e9
        # steadyEnd_plot = 0.507 * 1e9
        # plt.xlim(steadyStart_plot, steadyEnd_plot)
        # set 100 ticks in y-axis
        plt.yticks(np.linspace(0, maxQueueSize * 0.75, 20))
        # enable grids in y-axis
        plt.grid(axis='y')
        plt.legend()
        plt.title('Queue Size per time', fontsize=16)
        plt.grid()
        plt.xlabel('Time (ns)', fontsize=16)
        # plt.ylabel('Delay (ns)', fontsize=16)
        plt.ylabel('Size (B)', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        queue_name = file_path.split('/')[-1].split('_')[0]
        plt.savefig('{}/scratch/{}/{}/{}/{}_queueSize_time_{}_{}.png'.format(__ns3_path, results_folder, rate, experiment, queue_name, steadyStart, steadyEnd))
        plt.close()
        # lags, corr = cross_correlation_delay_time_series(full_df_M['Time'].values, full_df_M['TotalQueueSize'].values, full_df_CT['Time'].values, full_df_CT['TotalQueueSize'].values, bin_width=1000000, max_lag=100000000, normalize=True, plot=False)
        # print(f"Cross-correlation lags: {lags}")
        # print(f"Cross-correlation values: {corr}")
        # max_corr = np.max(corr) 
        # lag_at_max = lags[np.argmax(corr)]
        # symmetry = np.corrcoef(corr[:len(corr)//2], corr[:len(corr)//2:-1])[0, 1]
        # print(f"Max correlation: {max_corr} at lag {lag_at_max} with symmetry {symmetry}")
        full_df = None
        full_df_M = None
        full_df_CT = None

def calculate_offline_computations_on_switch(__ns3_path, results_folder, rate, experiment, segment, steadyStart, steadyEnd, paths, linkRates, load, queues_names):
    file_paths = glob.glob('{}/scratch/{}/{}/{}/{}/*_{}.csv'.format(__ns3_path, results_folder, rate, load, experiment, segment))
    dfs = {}
    for file_path in file_paths:
        df_res = {}
        df_name = file_path.split('/')[-1].split('_')[0]
        if df_name not in queues_names:
            continue
        full_df = pd.read_csv(file_path)
        df_res['first'] = {}
        df_res['last'] = {}
        df_res['workload'] = {}
        df_res['sampleSize'] = {}
        df_res['sampleSize']['delay'] = {}
        df_res['sampleSize']['successProb'] = {}
        df_res['sampleSize']['nonMarkingProb'] = {}
        df_res['totalPckts'] = {}
        df_res['successProbMean'] = {}
        full_df = prune_data(full_df, 'Time', steadyStart, steadyEnd)
        # full_df = full_df[full_df['Label'].str.contains('10.1.1.1', na=False)]
        full_df = full_df.sort_values(by=['Time', 'TotalQueueSize'], ascending=[True, True]).reset_index(drop=True)
        if df_name[0] == 'T' and df_name[2] == "A":
            linkRate = linkRates[1]
        if df_name[0] == 'T' and df_name[2] == "H":
            linkRate = linkRates[3]
        if df_name[0] == 'A' and df_name[2] == "T":
            linkRate = linkRates[2]
        # full_df, delay_linearInterp_time_average = manipulate_for_delay_Q_m(full_df, linkRate)
        # print("Switch Name:", df_name, "Link Rate:", linkRate)
        full_df, delay_linearInterp_time_average = manipulate_for_delay_Q(full_df, linkRate, steadyStart, steadyEnd, experiment)
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
        df_res['InterArrivals'] = {}
        df_res['RTT'] = {}
        for path in paths:
            df_res['sampleSize']['delay'][path] = len(full_df)
            df_res['sampleSize']['successProb'][path] = len(full_df)
            df_res['sampleSize']['nonMarkingProb'][path] = len(full_df)
            df_res['totalPckts'][path] = len(full_df)
            
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
            # endToEndStats[flow]['InterArrivals'][path]
            df_res['InterArrivals'][path] = full_df['Time'].mean()
            df_res['RTT'][path] = full_df['Delay'].mean()
        dfs[df_name] = df_res
    return dfs

def plot_cdf(full_df, name):
    plt.figure(figsize=(10, 6))
    sns.ecdfplot(data=full_df, x='PayloadSize')
    plt.title('cdf', fontsize=16)
    plt.xlabel('size', fontsize=16)
    plt.ylabel('CDF', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid()
    plt.show()
    plt.savefig('{}packetSize_cdf.png'.format(name))

def calculate_offline_mixing(__ns3_path, rate, segment, experiment, results_folder, steadyStart, steadyEnd, projectColumn, removeDrops=True, checkColumn="", linksRates=[], linkDelays=[], swtichDstREDQueueDiscMaxSize=0, stats=None, tsh=0.15, differentiationDelay=None, errorRate=None, load=None):
    if differentiationDelay is not None and errorRate is not None:
        file_paths = glob.glob('{}/scratch/{}/{}/{}/D_{}/f_{}/{}/*_{}.csv'.format(__ns3_path, results_folder, rate, load, differentiationDelay, errorRate, experiment, segment))
    else:
        file_paths = glob.glob('{}/scratch/{}/{}/{}/{}/*_{}.csv'.format(__ns3_path, results_folder, rate, load, experiment, segment))
    dfs = {}
    for file_path in file_paths:
        df_res = {}
        df_name = file_path.split('/')[-1].split('_')[0]
        full_df = pd.read_csv(file_path)
        full_df = prune_data(full_df, projectColumn, steadyStart, steadyEnd)
        full_df = full_df[full_df['Action'] == 'E'].copy()
        full_df['Delay'] = (full_df['TotalQueueSize'] * 8) / linksRates[0]
        full_df_M = full_df[full_df['Label'].str.contains('10.1.1.1', na=False)]
        full_df_CT = full_df[~full_df['Label'].str.contains('10.1.1.1', na=False)]
        full_df_M = full_df_M.sort_values(by=['Time', 'TotalQueueSize'], ascending=[True, True]).reset_index(drop=True)
        full_df_CT = full_df_CT.sort_values(by=['Time', 'TotalQueueSize'], ascending=[True, True]).reset_index(drop=True)
        delay_M = full_df_M['Delay'].values
        delay_CT = full_df_CT['Delay'].values
        cdf_M = np.arange(1, len(delay_M) + 1) / len(delay_M)
        cdf_CT = np.arange(1, len(delay_CT) + 1) / len(delay_CT)
        ks_stat, p_value = ks_2samp(cdf_M, cdf_CT)
        df_res['ks_statistic'] = ks_stat

        bins = np.arange(steadyStart, steadyEnd, 5 * 2 * np.sum(linkDelays))
        KSs = []
        for i in range(len(bins) - 1):
            # Get indices in the current chunk
            selected_M = full_df_M[(full_df_M['Time'] >= bins[i]) & (full_df_M['Time'] < bins[i + 1])]['Delay'].values
            selected_CT = full_df_CT[(full_df_CT['Time'] >= bins[i]) & (full_df_CT['Time'] < bins[i + 1])]['Delay'].values
            if len(selected_M) == 0 or len(selected_CT) == 0:
                continue
            cdf_M = np.arange(1, len(selected_M) + 1) / len(selected_M)
            cdf_CT = np.arange(1, len(selected_CT) + 1) / len(selected_CT)

            # KS Test (statistic only)
            ks_stat, p_value = ks_2samp(cdf_M, cdf_CT)
            KSs.append(ks_stat)

        df_res['ks_statisticMean'] = np.mean(KSs)
        dfs[df_name] = df_res
    return dfs

def sort_queues_by_path(queue_names, linkDelays, linkRates):
    sorted_queues = [None] * len(queue_names)
    sorted_linkRates = [None] * len(queue_names)
    sorted_linkDelays = [None] * len(queue_names)
    for queue_name in queue_names:
        if queue_name[0] == 'T' and queue_name[2] == "A":
            sorted_queues[0] = queue_name
            sorted_linkRates[0] = linkRates[1]
            sorted_linkDelays[0] = linkDelays[1]

        if queue_name[0] == 'A' and queue_name[2] == "T":
            sorted_queues[1] = queue_name
            sorted_linkRates[1] = linkRates[2]
            sorted_linkDelays[1] = linkDelays[2]

        if queue_name[0] == 'T' and queue_name[2] == "H":
            sorted_queues[2] = queue_name
            sorted_linkRates[2] = linkRates[3]
            sorted_linkDelays[2] = linkDelays[3]

    return sorted_queues, sorted_linkDelays, sorted_linkRates

def find_queue_size_at_time(times, queue_sizes, target_time, link_rate):
    if times.size == 0:
        return np.full(len(target_time), np.nan)
    if times.size == 1:
        return queue_sizes[-1]
    # find the position in times where target_time would be inserted to maintain order
    positions = np.searchsorted(times, target_time, side='right') - 1
    # invalid positions are those that are out of bounds (before the first time or after the last time)
    invalid = positions < 0
    invalid |= positions >= times.size - 1

    positions = np.clip(positions, 0, times.size - 1)
    
    # Get queue sizes and times at matched positions
    matched_queue_sizes = queue_sizes[positions]
    matched_times = times[positions]
    # print(f"Target times: {target_time}", f"Matched times: {matched_times}")
    # l = [(t, m) for t, m in zip(target_time, matched_times) if abs(t - m) > 2]
    # print(f"not matched times: {l[:10]}")
    # Apply draining logic: the queue size should decrease over time.
    # Drained bytes = time_difference * link_rate / 8 (convert bits/ns to bytes/ns)
    time_differences = np.asarray(target_time, dtype=float) - matched_times
    drained_bytes = (time_differences * link_rate) / 8
    
    # Final queue size = original size - drained amount, but not less than zero
    final_queue_sizes = np.maximum(0, matched_queue_sizes - drained_bytes)
    final_queue_sizes[invalid] = np.nan  
    return final_queue_sizes

def remove_nan_samples(times, queue_sizes, queue_ECN_samples, queue_delay_samples):
    valid_indices = ~np.isnan(queue_sizes)
    return times[valid_indices], queue_sizes[valid_indices], queue_ECN_samples[valid_indices], queue_delay_samples[valid_indices]

def sample_queue_size(times, file_path, link_rate):
    # print(f"Sampling total queue size from {file_path} with link rate {link_rate} bpns")
    full_df = pd.read_csv(file_path)
    full_df = full_df.sort_values(by=['Time', 'TotalQueueSize'], ascending=[True, False]).reset_index(drop=True)
    # print(f"file_path: {file_path}\n")
    sample_times = np.asarray(times, dtype=float)
    df_times = full_df['Time'].to_numpy(dtype=float)
    df_queue_sizes = full_df['TotalQueueSize'].to_numpy(dtype=float)
    return find_queue_size_at_time(df_times, df_queue_sizes, sample_times, link_rate)

def sample_ECN_marking(queue_size_samples, queue_size_trsh):
    return (queue_size_samples >= queue_size_trsh).astype(int)

def sample_queueing_delay(queue_size_samples, link_rate):
    return (queue_size_samples * 8) / link_rate

def sample_total_queue_size_with_size(times, sizes, queue_names, dir_prefix, linkDelays, linkRates, queue_size_trshs):
    queue_names, linkDelays, linkRates = sort_queues_by_path(queue_names, linkDelays, linkRates)
    queue_size_samples = np.zeros((len(queue_names), len(times)))
    queue_ECN_samples = np.zeros((len(queue_names), len(times)), dtype=int)
    queue_delay_samples = np.zeros((len(queue_names), len(times)))
    sample_times = np.asarray(times, dtype=float)
    invalid_indices = np.zeros(len(times), dtype=bool)
    for queue_name in queue_names:
        file_path = dir_prefix + queue_name + '_PoissonSampler_queueSize.csv'
        idx = queue_names.index(queue_name)
        # print(f"Arrival time at Queue {queue_name}: {sample_times[:10]}")
        queue_size_sample = sample_queue_size(sample_times, file_path, linkRates[idx])
        queue_size_samples[idx][~invalid_indices] = queue_size_sample - sizes[~invalid_indices]
        queue_size_samples[idx][invalid_indices] = np.nan
        new_invalid_indices = np.isnan(queue_size_sample)

        invalid_indices = np.isnan(queue_size_samples[idx])
        # print(f"Queue {queue_name} - Sampled queue sizes: {queue_size_samples[idx][:10]}")
        # shift sampling times for next queue for valid indices only (where we have valid samples), and round to integer nanoseconds
        sample_times = sample_times[~new_invalid_indices] + linkDelays[idx] + ((sizes[~invalid_indices] + queue_size_samples[idx][~invalid_indices]) * 8 / linkRates[idx]).astype(int) + 1
        queue_ECN_samples[idx][~invalid_indices] = sample_ECN_marking(queue_size_samples[idx][~invalid_indices], queue_size_trshs[idx])
        queue_ECN_samples[idx][invalid_indices] = 0
        queue_delay_samples[idx][~invalid_indices] = sample_queueing_delay(queue_size_samples[idx][~invalid_indices], linkRates[idx])
        queue_delay_samples[idx][invalid_indices] = np.nan

    return remove_nan_samples(times, np.sum(queue_size_samples, axis=0), np.any(queue_ECN_samples, axis=0).astype(int), np.sum(queue_delay_samples, axis=0))

def qqplot_queue_vs_arrivals(
    queue_values,
    arrival_increments,
    file_path=None,
    title="Q-Q plot: Q(t) vs arrival increments",
):
    """
    Generate a Q-Q plot comparing Q(t) and arrival increments.

    Parameters
    ----------
    queue_values : array-like
        1D array of queue samples Q(t).
    arrival_increments : array-like
        1D array of sampled arrival increments.
    num_quantiles : int
        Number of quantile points to use in the Q-Q plot.
    file_path : str or None
        If provided, save the figure to this path.
    title : str
        Plot title.

    Returns
    -------
    dict
        {
            "queue_quantiles": ...,
            "arrival_quantiles": ...,
            "quantile_levels": ...
        }
    """
    import statsmodels.api as sm
    from statsmodels.graphics.gofplots import qqplot_2samples

    queue_values = np.asarray(queue_values, dtype=float)
    arrival_increments = np.asarray(arrival_increments, dtype=float)

    if queue_values.ndim != 1 or arrival_increments.ndim != 1:
        raise ValueError("queue_values and arrival_increments must be 1D arrays.")
    if len(queue_values) == 0 or len(arrival_increments) == 0:
        raise ValueError("Inputs must be non-empty.")

    # Remove NaNs / infs
    queue_values = queue_values[np.isfinite(queue_values)]
    arrival_increments = arrival_increments[np.isfinite(arrival_increments)]
    # normalize the data to have zero mean and unit variance
    queue_values = (queue_values - np.mean(queue_values)) / np.std(queue_values)
    arrival_increments = (arrival_increments - np.mean(arrival_increments)) / np.std(arrival_increments)

    if len(queue_values) == 0 or len(arrival_increments) == 0:
        raise ValueError("Inputs must contain at least one finite value.")

    plt.figure(figsize=(30, 10))
    pp_x = sm.ProbPlot(queue_values)
    pp_y = sm.ProbPlot(arrival_increments)
    fig = qqplot_2samples(pp_x, pp_y, line='45')
    ax = fig.axes[0]
    # set the color and size of the points
    for line in ax.get_lines():
        line.set_marker('o')
        line.set_markersize(10)
        line.set_alpha(0.7)
        line.set_color('blue')
    plt.xlabel("Quantiles of Q(t)")
    plt.ylabel("Quantiles of arrival increments")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    # plt.legend()
    plt.tight_layout()

    if file_path is not None:
        plt.savefig(file_path)
        plt.close()
    else:
        plt.show()

def boxes_test(
    arrival_increments,
    lags
):
    """
    Returns Ljung-Box and Box-Pierce test results for autocorrelation of a time series.

    Parameters
    ----------
    arrival_increments : array-like
        1D array of sampled arrival increments.
    Returns
    -------
    dict
        {
            "ljung_box_statistic": ...,
            "ljung_box_pvalue": ...,
            "box_pierce_statistic": ...,
            "box_pierce_pvalue": ...
        }
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    arrival_increments = np.asarray(arrival_increments, dtype=float)
    if arrival_increments.ndim != 1:
        raise ValueError("arrival_increments must be a 1D array.")
    if len(arrival_increments) == 0:
        raise ValueError("arrival_increments must be non-empty.")

    # Remove NaNs / infs
    arrival_increments = arrival_increments[np.isfinite(arrival_increments)]
    if len(arrival_increments) == 0:
        raise ValueError("arrival_increments must contain at least one finite value.")

    # Perform Ljung-Box test
    ljung_box_result = acorr_ljungbox(arrival_increments, boxpierce=True, lags=lags)
    print(f"Ljung-Box test statistic: {ljung_box_result['lb_stat']}, p-value: {ljung_box_result['lb_pvalue']}")
    print(f"Box-Pierce test statistic: {ljung_box_result['bp_stat']}, p-value: {ljung_box_result['bp_pvalue']}")
    # return {
    #     "ljung_box_statistic": ljung_box_statistic,
    #     "ljung_box_pvalue": ljung_box_pvalue,
    #     "box_pierce_statistic": box_pierce_statistic,
    #     "box_pierce_pvalue": box_pierce_pvalue
    # }

def chi_squared_test(
    arrival_times, 
    steadyStart, 
    steadyEnd, 
    lags=None
):
    """
    Perform a chi-squared test for independence between arrival events at time t and time t - lag.

    Parameters
    ----------
    arrival_times : array-like
        1D array of arrival times (in the same time units as steadyStart and steadyEnd).
    steadyStart : float
        Start time of the steady state period.
    steadyEnd : float
        End time of the steady state period.
    lag : float or None
        Time lag to test for independence. If None, tests multiple lags (1, 2, 4, ..., 8192 samples) and prints results for each.
    """
    from scipy.stats import chi2_contingency

    times = np.arange(steadyStart, steadyEnd, 120)
    T = 120
    if lags is None:
        lags = [i for i in range(0, 512)]
        lags = lags[1:]

    res = [False] * len(lags)
    chi2_res = [0] * len(lags)
    arrival_increments = sample_increments_of_arrivals(arrival_times, T, times, event_type='binary')
    for j, lag in enumerate(lags):
        if lag >= len(arrival_increments):
            res[j] = False
            continue

        A = arrival_increments[:-lag]
        B = arrival_increments[lag:]

        A1B1 = np.sum(A & B)
        A1B0 = np.sum(A & (~B))
        A0B1 = np.sum((~A) & B)
        A0B0 = np.sum((~A) & (~B))

        table = np.array([[A1B1, A1B0], [A0B1, A0B0]])
        chi2, p, _, _ = chi2_contingency(table)
        chi2_res[j] = chi2
        res[j] = p < 0.05


    return lags, res, chi2_res

def sample_total_queue_size_non_combined(res, times, queue_names, dir_prefix, linkDelays, linkRates, queue_size_trshs, 
                                     steadyStart=0.01e9, steadyEnd=0.1e9, intervals=10000, path_observation=False, sampling_factor=None):
    queue_names, linkDelays, linkRates = sort_queues_by_path(queue_names, linkDelays, linkRates)
    tag = 'poisson'
    if path_observation:
        tag = 'e2e'
    for queue_name in queue_names:
        res[queue_name+ tag + '_samples_queue_delay_mean'] = 0
        res[queue_name+ tag + '_samples_queue_delay_std'] = 0
        res[queue_name+ tag + '_samples_queue_delay_count'] = 0
        res[queue_name+'poisson_prob_non_empty'] = 0
        res[queue_name+'poisson_prev_queue_non_empty_prob_percentile'] = 0
        res[queue_name+'bias'] = 0
    
    iterations = 1 if sampling_factor is None else 100
    for itr in range(iterations):
        sample_times_itr = np.asarray(times, dtype=float)
        if sampling_factor is not None:
            sample_mask = np.random.rand(len(sample_times_itr)) < sampling_factor
            sample_times_itr = sample_times_itr[sample_mask]
        # TODO: check why we have the same thing over different experiments
        # print(f"Iteration {itr+1}/{iterations} - Sampling total {len(sample_times_itr)} with mean arrivlas: {np.mean(times)} first 10 times: {times[:10]}")
        queue_size_samples = np.zeros((len(queue_names), len(sample_times_itr)))
        queue_ECN_samples = np.zeros((len(queue_names), len(sample_times_itr)), dtype=int)
        queue_delay_samples = np.zeros((len(queue_names), len(sample_times_itr)))
        invalid_indices = np.zeros(len(sample_times_itr), dtype=bool)
        sample_times = sample_times_itr

        for queue_name in queue_names:
            file_path = dir_prefix + queue_name + '_PoissonSampler_queueSize.csv'
            idx = queue_names.index(queue_name)
            queue_size_sample = sample_queue_size(sample_times, file_path, linkRates[idx])
            queue_size_samples[idx][~invalid_indices] = queue_size_sample
            queue_size_samples[idx][invalid_indices] = np.nan
            new_invalid_indices = np.isnan(queue_size_sample)
            invalid_indices = np.isnan(queue_size_samples[idx])
            queue_ECN_samples[idx][~invalid_indices] = sample_ECN_marking(queue_size_samples[idx][~invalid_indices], queue_size_trshs[idx])
            queue_ECN_samples[idx][invalid_indices] = 0
            queue_delay_samples[idx][~invalid_indices] = sample_queueing_delay(queue_size_samples[idx][~invalid_indices], linkRates[idx])

            prob_non_empty = queue_size_samples[idx][~invalid_indices] > 0
            prob_non_empty = np.sum(prob_non_empty) / len(prob_non_empty)

            res[queue_name+ tag + '_samples_queue_delay_mean'] = (res[queue_name+ tag + '_samples_queue_delay_mean'] * itr + np.nanmean(queue_delay_samples[idx])) / (itr + 1)
            res[queue_name+ tag + '_samples_queue_delay_std'] = (res[queue_name+ tag + '_samples_queue_delay_std'] * itr + np.nanstd(queue_delay_samples[idx])) / (itr + 1)
            res[queue_name+ tag + '_samples_queue_delay_count'] = (res[queue_name+ tag + '_samples_queue_delay_count'] * itr + len(queue_delay_samples[idx][~invalid_indices])) / (itr + 1)
            if not path_observation:
                res[queue_name+'poisson_prob_non_empty'] = prob_non_empty
                if idx > 0:
                    res[queue_name+'poisson_prev_queue_non_empty_prob_percentile'] = np.nanpercentile(queue_delay_samples[idx], res[queue_names[idx - 1]+'poisson_prob_non_empty'] * 100)
                    bias = res[queue_name+'poisson_prev_queue_non_empty_prob_percentile'] * res[queue_names[idx - 1]+'poisson_prob_non_empty']
                    res[queue_name+'bias'] = bias
                else:
                    res[queue_name+'poisson_prev_queue_non_empty_prob_percentile'] = np.nan
                    bias = 0
                    res[queue_name+'bias'] = bias
            # shift sampling times for next queue for valid indices only (where we have valid samples)
            if path_observation:
                sample_times = (sample_times[~new_invalid_indices] + queue_size_samples[idx][~invalid_indices] * 8 / linkRates[idx] + linkDelays[idx]).round()
            else:
                invalid_indices = np.zeros(len(times), dtype=bool)
                sample_times = np.asarray(times, dtype=float) + linkDelays[0] * (idx + 1)

            queue_delay_samples[idx][invalid_indices] = np.nan
    return remove_nan_samples(sample_times_itr, np.sum(queue_size_samples, axis=0), np.any(queue_ECN_samples, axis=0).astype(int), np.sum(queue_delay_samples, axis=0)), res

def combine_sampling_results(res, queue_names):
    queue_names, _, _ = sort_queues_by_path(queue_names, [None] * (len(queue_names) + 1), [None] * (len(queue_names) + 1))
    for queue_name in queue_names:
        idx = queue_names.index(queue_name)
        res[queue_name+'error_bound'] = res[queue_name+'e2e_samples_queue_delay_std'] * 1.96 / np.sqrt(res[queue_name+'e2e_samples_queue_delay_count']) + res[queue_name+'poisson_samples_queue_delay_std'] * 1.96 / np.sqrt(res[queue_name+'poisson_samples_queue_delay_count'])
        res[queue_name+'e2e_vs_poisson_consistent'] = int(abs(res[queue_name+'e2e_samples_queue_delay_mean'] - res[queue_name+'poisson_samples_queue_delay_mean']) <= res[queue_name+'error_bound'])
        if idx > 0:
            res[queue_name+'e2e_vs_poisson_consistent_with_bias'] = int(abs(res[queue_name+'e2e_samples_queue_delay_mean'] - (res[queue_name+'poisson_samples_queue_delay_mean'] + res[queue_name+'bias'])) <= res[queue_name+'error_bound'])
        else:
            res[queue_name+'e2e_vs_poisson_consistent_with_bias'] = res[queue_name+'e2e_vs_poisson_consistent']
    return res

def sample_total_queue_size(times, queue_names, dir_prefix, linkDelays, linkRates, queue_size_trshs, steadyStart=0.01e9, steadyEnd=0.1e9, intervals=10000):
    queue_names, linkDelays, linkRates = sort_queues_by_path(queue_names, linkDelays, linkRates)
    queue_size_samples = np.zeros((len(queue_names), len(times)))
    queue_ECN_samples = np.zeros((len(queue_names), len(times)), dtype=int)
    queue_delay_samples = np.zeros((len(queue_names), len(times)))
    sample_times = np.asarray(times, dtype=float)
    invalid_indices = np.zeros(len(times), dtype=bool)
    # Poisson_sample_times = np.array(np.cumsum(np.random.exponential(intervals, size=int((steadyEnd - steadyStart) // intervals))) + steadyStart, dtype=np.int64)
    res = {}
    # prev_poisson_samples = np.zeros(len(Poisson_sample_times))
    # prev_invalid_indices = None
    # curr_poisson_samples = np.zeros(len(Poisson_sample_times))
    # curr_invalid_indices = None
    for queue_name in queue_names:
        file_path = dir_prefix + queue_name + '_PoissonSampler_queueSize.csv'
        idx = queue_names.index(queue_name)
        queue_size_sample = sample_queue_size(sample_times, file_path, linkRates[idx])
        queue_size_samples[idx][~invalid_indices] = queue_size_sample
        queue_size_samples[idx][invalid_indices] = np.nan
        new_invalid_indices = np.isnan(queue_size_sample)

        invalid_indices = np.isnan(queue_size_samples[idx])

        queue_ECN_samples[idx][~invalid_indices] = sample_ECN_marking(queue_size_samples[idx][~invalid_indices], queue_size_trshs[idx])
        queue_ECN_samples[idx][invalid_indices] = 0
        queue_delay_samples[idx][~invalid_indices] = sample_queueing_delay(queue_size_samples[idx][~invalid_indices], linkRates[idx])
        # Poisson_sample_times = np.array(np.cumsum(np.random.exponential(intervals, size=int((steadyEnd - steadyStart) // intervals))) + steadyStart, dtype=np.int64)
        # Poisson_sample_times = Poisson_sample_times + linkDelays[idx] * idx
        Poisson_sample_times = np.asarray(times, dtype=float) + linkDelays[idx] * idx
        poisson_samples = sample_queue_size(Poisson_sample_times, file_path, linkRates[idx])
        poisson_invalid_indices = np.isnan(poisson_samples)
        poisson_samples_delay = sample_queueing_delay(poisson_samples[~poisson_invalid_indices], linkRates[idx])
        #########
        # if idx > 0:
        #     prev_poisson_samples = curr_poisson_samples
        #     prev_invalid_indices = curr_invalid_indices
        #     curr_invalid_indices = np.isnan(poisson_samples)
        #     curr_poisson_samples = np.zeros(len(Poisson_sample_times))
        #     curr_poisson_samples[~curr_invalid_indices] = poisson_samples_delay
        # else:
        #     curr_invalid_indices = np.isnan(poisson_samples)
        #     curr_poisson_samples[~curr_invalid_indices] = poisson_samples_delay
        ###########
        prob_non_empty = poisson_samples[~poisson_invalid_indices] > 0
        prob_non_empty = np.sum(prob_non_empty) / len(prob_non_empty)

        res[queue_name+'e2e_samples_queue_delay_mean'] = np.nanmean(queue_delay_samples[idx])
        # print(f"Queue {queue_name} - E2E samples queue delay mean: {res[queue_name+'e2e_samples_queue_delay_mean']}")
        res[queue_name+'e2e_samples_queue_delay_std'] = np.nanstd(queue_delay_samples[idx])
        res[queue_name+'e2e_samples_queue_delay_count'] = len(queue_delay_samples[idx][~invalid_indices])
        res[queue_name+'poisson_samples_queue_delay_mean'] = np.nanmean(poisson_samples_delay)
        # print(f"Queue {queue_name} - Poisson samples queue delay mean: {res[queue_name+'poisson_samples_queue_delay_mean']}")
        res[queue_name+'poisson_samples_queue_delay_std'] = np.nanstd(poisson_samples_delay)
        res[queue_name+'poisson_samples_queue_delay_count'] = len(poisson_samples_delay)
        res[queue_name+'poisson_prob_non_empty'] = prob_non_empty
        res[queue_name+'error_bound'] = res[queue_name+'e2e_samples_queue_delay_std'] * 1.96 / np.sqrt(len(queue_delay_samples[idx][~invalid_indices])) + res[queue_name+'poisson_samples_queue_delay_std'] * 1.96 / np.sqrt(len(poisson_samples_delay))
        res[queue_name+'e2e_vs_poisson_consistent'] = int(abs(res[queue_name+'e2e_samples_queue_delay_mean'] - res[queue_name+'poisson_samples_queue_delay_mean']) <= res[queue_name+'error_bound'])
        if idx > 0:
            res[queue_name+'poisson_prev_queue_non_empty_prob_percentile'] = np.nanpercentile(poisson_samples_delay, res[queue_names[idx - 1]+'poisson_prob_non_empty'] * 100)
            bias = res[queue_name+'poisson_prev_queue_non_empty_prob_percentile'] * res[queue_names[idx - 1]+'poisson_prob_non_empty']
            res[queue_name+'bias'] = bias
            res[queue_name+'e2e_vs_poisson_consistent_with_bias'] = int(abs(res[queue_name+'e2e_samples_queue_delay_mean'] - (res[queue_name+'poisson_samples_queue_delay_mean'] + bias)) <= res[queue_name+'error_bound'])
        else:
            res[queue_name+'poisson_prev_queue_non_empty_prob_percentile'] = np.nan
            bias = 0
            res[queue_name+'bias'] = bias
            res[queue_name+'e2e_vs_poisson_consistent_with_bias'] = res[queue_name+'e2e_vs_poisson_consistent']
        ###########
        # if idx > 0:
            # print(f"Correlation between queue size of {queue_names[idx - 1]} and {queue_name}")
            # x = prev_poisson_samples
            # y = curr_poisson_samples
            # x = queue_delay_samples[idx - 1]
            # y = queue_delay_samples[idx]
            # print(f"x mean: {np.nanmean(x)}, y mean: {np.nanmean(y)}")
            # corr_indices = (prev_poisson_samples > 0) & (~curr_invalid_indices) & (~prev_invalid_indices)
            # corr_indices = (queue_delay_samples[idx - 1] > 0) & (~invalid_indices)
            # x = x[corr_indices]
            # y = y[corr_indices]
            # x = x - x.mean()
            # y = y - y.mean()
            # corr = np.correlate(x, y, mode="full")
            # lags = np.arange(-len(x) + 1, len(x))
            # denom = np.sqrt(np.sum(x**2) * np.sum(y**2))
            # corr = corr / denom
            # band = 1.96 / np.sqrt(len(lags))  # 95% confidence interval for zero correlation
            # print(f"Correlation between queue delay samples of {queue_names[idx - 1]} and {queue_name} at lag 0: {corr[0]}, with Band at 95% confidence: {band}")
            # mask = lags >= 0
            # lags = lags[mask]
            # corr = corr[mask]
            # plt.figure(figsize=(30, 10))
            # plt.plot(lags, corr, marker='o', linestyle='-', markersize=4, linewidth=2)
            # plt.axhline(0, linewidth=1)
            # plt.axvline(0, linewidth=1)
            # band = 1.96 / np.sqrt(len(lags))  # 95% confidence interval for zero correlation
            # plt.axhline(band, color='black', linestyle='dashed', linewidth=3, label='95% confidence band')
            # plt.axhline(-band, color='black', linestyle='dashed', linewidth=3)
            # plt.xlabel("Lag ")
            # plt.ylabel("Cross-correlation")
            # plt.title("Cross-correlation")
            # plt.grid(True, alpha=0.5)
            # plt.set_ylim(bottom=-0.4, top=1.0)
            # plt.ylim(bottom=-1.05 * max(corr), top=1.05 * max(corr))
            # plt.set_yticks(np.arange(-0.4, 0.8, 0.2))
            # plt.set_xticks(np.arange(0, np.max(lags_time), max(lags_time) / 20), labels=[f"{float(t/1000000):.1f}" for t in np.arange(0, np.max(lags_time), max(lags_time) / 20)])
            # plt.set_xticks(np.arange(np.min(lags_time), np.max(lags_time), (np.max(lags_time) - np.min(lags_time)) / 20), labels=[f"{float(t/1000000):.1f}" for t in np.arange(np.min(lags_time), np.max(lags_time), (np.max(lags_time) - np.min(lags_time)) / 20)])
            # plt.tick_params(axis='y', labelsize=30)
            # plt.tight_layout()
            # plt.savefig(f'{dir_prefix}crosscorr_{queue_names[idx - 1]}_{queue_name}_pathObserver_Q_GT0_{res[queue_name+"e2e_samples_queue_delay_count"]}samples_withBands.png')
            # plt.close()
        ###########
        # shift sampling times for next queue for valid indices only (where we have valid samples)
        sample_times = (sample_times[~new_invalid_indices] + queue_size_samples[idx][~invalid_indices] * 8 / linkRates[idx] + linkDelays[idx]).round()
        # sample_times = sample_times[~new_invalid_indices]
        queue_delay_samples[idx][invalid_indices] = np.nan

    return remove_nan_samples(times, np.sum(queue_size_samples, axis=0), np.any(queue_ECN_samples, axis=0).astype(int), np.sum(queue_delay_samples, axis=0)), res

def sample_total_queue_size_single_queue(times, queue_name, dir_prefix, linkDelay, linkRate, queue_size_trsh):
    queue_size_samples = np.zeros((1, len(times)))
    queue_ECN_samples = np.zeros((1, len(times)), dtype=int)
    queue_delay_samples = np.zeros((1, len(times)))
    sample_times = np.asarray(times, dtype=float)
    invalid_indices = np.zeros(len(times), dtype=bool)
    file_path = dir_prefix + queue_name + '_PoissonSampler_queueSize.csv'
    queue_size_sample = sample_queue_size(sample_times, file_path, linkRate)
    queue_size_samples[0][~invalid_indices] = queue_size_sample
    queue_size_samples[0][invalid_indices] = np.nan
    new_invalid_indices = np.isnan(queue_size_sample)

    invalid_indices = np.isnan(queue_size_samples[0])
    prob_non_empty = queue_size_samples[0][~invalid_indices] > 0
    prob_non_empty = np.sum(prob_non_empty) / len(prob_non_empty)
    queue_delay_percentile = np.nanpercentile(sample_queueing_delay(queue_size_samples[0][~invalid_indices], linkRate), prob_non_empty * 100)
    print(f"Queue {queue_name} - duration: {np.nanmax(sample_times[~new_invalid_indices]) - np.nanmin(sample_times[~new_invalid_indices])} ns and length: {len(sample_times[~new_invalid_indices])} samples")
    print(f"Queue {queue_name} - min time: {np.nanmin(sample_times[~new_invalid_indices])} ns, max time: {np.nanmax(sample_times[~new_invalid_indices])} ns")
    print(f"Queue {queue_name} - probability of non-empty queue: {prob_non_empty}")
    print(f"Queue {queue_name} - non-empty percentile queue delay: {queue_delay_percentile} ns")
    print(f"Queue {queue_name} - bias : {queue_delay_percentile * prob_non_empty} ns")
    # plt.figure(figsize=(10, 6))
    # plt.scatter(sample_times[~new_invalid_indices], queue_size_samples[0][~invalid_indices], color='r', label='Sampled Queue Size', marker='o', s=3)
    # plt.ylim(0, np.nanmax(queue_size_samples[0][~invalid_indices]) * 1.5)
    # plt.grid(axis='y')
    # plt.legend()
    # plt.title(f'Queue Size per time for {queue_name}', fontsize=16)
    # plt.xlabel('Time (ns)', fontsize=16)
    # plt.ylabel('Size (B)', fontsize=16)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.savefig(f'Poisson_queue_size_time_{queue_name}.png')
    # plt.close()
    queue_ECN_samples[0][~invalid_indices] = sample_ECN_marking(queue_size_samples[0][~invalid_indices], queue_size_trsh)
    queue_ECN_samples[0][invalid_indices] = 0
    queue_delay_samples[0][~invalid_indices] = sample_queueing_delay(queue_size_samples[0][~invalid_indices], linkRate)
    queue_delay_samples[0][invalid_indices] = np.nan

    return remove_nan_samples(times, np.sum(queue_size_samples, axis=0), np.any(queue_ECN_samples, axis=0).astype(int), np.sum(queue_delay_samples, axis=0))

def visualize_autocorr_Ts(results, file_path):
    """
    Visualize the output of autocorr_arrival_increments(...)

    Parameters
    ----------
    result : dict
        Output of autocorr_arrival_increments(...)
    max_points_scatter : int
        Maximum number of points to show in the scatter plot.
        If there are more points, a random subset is used.
    """

    fig, ax = plt.subplots(1, 1, figsize=(30, 20))

    # --------------------------------------------------
    # 1) Auto-correlations
    # --------------------------------------------------
    for result in results:
        times = np.asarray(result["times"])
        lags = np.asarray(result["lags"])
        autocorr = np.asarray(result["autocorr"])
        lags_time = lags * np.mean(np.diff(times))  # convert lags from sample index to time
        ax.plot(lags_time, autocorr, marker='o', linestyle='-', markersize=4, linewidth=2, label=result['T'])
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Autocorrelation")
    ax.grid(True, alpha=0.5)
    ax.set_ylim(bottom=-0.4, top=1.0)
    ax.set_yticks(np.arange(-0.4, 0.8, 0.2))
    # ax.set_xticks(np.arange(0, np.max(lags_time), max(lags_time) / 20), labels=[f"{float(t/1000000):.1f}" for t in np.arange(0, np.max(lags_time), max(lags_time) / 20)])
    ax.tick_params(axis='y', labelsize=30)
    ax.legend(fontsize=30)
    plt.tight_layout()
    plt.savefig(file_path + 'autocorr_of_arrival_increment_diff_Ts.png')
    plt.close()

def visualize_autocorr_result(result, file_path, T):
    """
    Visualize the output of autocorr_arrival_increments(...)

    Parameters
    ----------
    result : dict
        Output of autocorr_arrival_increments(...)
    max_points_scatter : int
        Maximum number of points to show in the scatter plot.
        If there are more points, a random subset is used.
    """

    times = np.asarray(result["times"])
    arrival_increment = np.asarray(result["arrival_increment"])
    lags = np.asarray(result["lags"])
    autocorr = np.asarray(result["autocorr"])
    lags_time = lags * np.mean(np.diff(times))  # convert lags from sample index to time

    fig, axes = plt.subplots(3, 1, figsize=(30, 40))

    # --------------------------------------------------
    # 1) Time series plot
    # --------------------------------------------------
    ax1 = axes[0]
    sc1 = ax1.scatter(times, arrival_increment, alpha=1.0, s=20, marker='o', label="Queue size", color='blue')
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Arrival increments")
    ax1.grid(True, alpha=0.1)

    ax1.set_title("Arrival increments over time")

    # --------------------------------------------------
    # 3) Cross-correlation
    # --------------------------------------------------
    ax = axes[1]
    ax.plot(lags_time, autocorr, marker='o', linestyle='-', markersize=4, linewidth=2)
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.axvline(T, color='blue', linestyle='dashed', linewidth=3, label=r'$\tau$ = {} ms'.format(float(T / 1e6)))
    ax.axvline(2 * T, color='blue', linestyle='dashed', linewidth=3, label=r'$2\tau$ = {} ms'.format(float(2 * T / 1e6)))
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Autocorrelation")
    ax.grid(True, alpha=0.5)
    ax.set_ylim(bottom=-0.4, top=1.0)
    ax.set_yticks(np.arange(-0.4, 0.8, 0.2))
    ax.set_xticks(np.arange(0, np.max(lags_time), max(lags_time) / 20), labels=[f"{float(t/1000000):.1f}" for t in np.arange(0, np.max(lags_time), max(lags_time) / 20)])
    ax.tick_params(axis='y', labelsize=30)
    ax.legend(fontsize=30)

    # --------------------------------------------------
    # 3) Cross-correlation(zoomed in)
    # --------------------------------------------------
    ax = axes[2]
    # find the lag that is 10 times T
    zoomed_lags = lags_time <= 10 * T
    ax.plot(lags_time[zoomed_lags], autocorr[zoomed_lags], marker='o', linestyle='-', markersize=4, linewidth=2)
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.axvline(T, color='blue', linestyle='dashed', linewidth=3, label=r'$\tau$ = {} us'.format(float(T / 1e3)))
    ax.axvline(2 * T, color='blue', linestyle='dashed', linewidth=3, label=r'$2\tau$ = {} us'.format(float(2 * T / 1e3)))
    # ax.axvline(3 * T, color='blue', linestyle='dashed', linewidth=3, label=r'$3\tau$ = {} us'.format(float(3 * T / 1e3)))
    ax.set_xlabel("Lag (us)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Autocorrelation (zoomed in)")
    ax.grid(True, alpha=0.5)
    ax.set_ylim(bottom=-0.4, top=1.0)
    ax.set_yticks(np.arange(-0.4, 0.8, 0.2))
    ax.set_xticks(np.arange(0, np.max(lags_time[zoomed_lags]), max(lags_time[zoomed_lags]) / 20), labels=[f"{float(t/1000):.0f}" for t in np.arange(0, np.max(lags_time[zoomed_lags]), max(lags_time[zoomed_lags]) / 20)])
    ax.tick_params(axis='y', labelsize=30)
    ax.legend(fontsize=30)
    plt.tight_layout()
    plt.savefig(file_path + 'autocorr_of_arrival_increment.png')
    plt.close()

def visualize_crosscorr_result(result, file_path, max_points_scatter=10000000, suffix=""):
    """
    Visualize the output of crosscorr_qsize_vs_arrival_increment(...)

    Parameters
    ----------
    result : dict
        Output of crosscorr_qsize_vs_arrival_increment(...)
    max_points_scatter : int
        Maximum number of points to show in the scatter plot.
        If there are more points, a random subset is used.
    """

    times = np.asarray(result["times"])
    queue_sizes = np.asarray(result["queue_sizes"])
    arrival_increment = np.asarray(result["arrival_increment"])
    lags = np.asarray(result["lags"])
    crosscorr = np.asarray(result["crosscorr"])
    lags_time = lags * np.mean(np.diff(times))  # convert lags from sample index to time

    fig, axes = plt.subplots(3, 1, figsize=(30, 40))

    # --------------------------------------------------
    # 1) Time series plot
    # --------------------------------------------------
    ax1 = axes[0]

    # First axis: queue size
    sc1 = ax1.scatter(times, queue_sizes, alpha=1.0, s=20, marker='o', label="Queue size", color='blue')
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Queue size (B)")
    ax1.grid(True, alpha=0.1)
    ax2 = ax1.twinx()
    sc2 = ax2.scatter(times, arrival_increment, alpha=1.0, s=20, marker='s', label="Arrival increments", color='red')
    ax2.set_ylabel("Arrival increments")
    handles = [sc1, sc2]
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc='best', fontsize=40, fancybox=True, shadow=True)

    ax1.set_title("Aligned time series (dual axis)")
    # --------------------------------------------------
    # 2) Scatter plot
    # --------------------------------------------------
    ax = axes[1]

    n = len(queue_sizes)
    if n > max_points_scatter:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_points_scatter, replace=False)
        x = queue_sizes[idx]
        y = arrival_increment[idx]
    else:
        x = queue_sizes
        y = arrival_increment

    ax.scatter(x, y, alpha=1.0, s=20)
    ax.set_xlabel("Queue size (B)")
    ax.set_ylabel("Arrival increment")
    ax.set_title("Scatter: queueing size vs arrival increment")
    ax.grid(True, alpha=0.3)

    # --------------------------------------------------
    # 3) Cross-correlation
    # --------------------------------------------------
    ax = axes[2]
    ax.plot(lags_time, crosscorr, marker='o', linestyle='-', markersize=4, linewidth=2)
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    band = 1.96 / np.sqrt(len(lags_time))  # 95% confidence interval for zero correlation
    ax.axhline(band, color='black', linestyle='dashed', linewidth=3, label='95% confidence band')
    ax.axhline(-band, color='black', linestyle='dashed', linewidth=3)
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Cross-correlation")
    ax.set_title("Cross-correlation")
    ax.grid(True, alpha=0.5)
    # ax.set_ylim(bottom=-0.4, top=1.0)
    ax.set_ylim(bottom=-1.05 * max(crosscorr), top=1.05 * max(crosscorr))
    # ax.set_yticks(np.arange(-0.4, 0.8, 0.2))
    ax.set_xticks(np.arange(0, np.max(lags_time), max(lags_time) / 20), labels=[f"{float(t/1000000):.1f}" for t in np.arange(0, np.max(lags_time), max(lags_time) / 20)])
    # ax.set_xticks(np.arange(np.min(lags_time), np.max(lags_time), (np.max(lags_time) - np.min(lags_time)) / 20), labels=[f"{float(t/1000000):.1f}" for t in np.arange(np.min(lags_time), np.max(lags_time), (np.max(lags_time) - np.min(lags_time)) / 20)])
    ax.tick_params(axis='y', labelsize=30)
    plt.tight_layout()
    plt.savefig(file_path + 'crosscorr_qsize_vs_arrival_increment' + suffix + '.png')
    plt.close()

def sample_increments_of_arrivals_bytes(arrival_times, T, times_to_sample, arrival_sizes):
    """
    Sample arrival increments at specific times.

    Parameters
    ----------
    arrival_times : array-like
        1D array of arrival timestamps.
    T : float
        Window length used to count arrivals in [t, t+T).
    times_to_sample : array-like
        1D array of timestamps at which to sample arrival increment.
    arrival_sizes : array-like
        1D array of arrival sizes.
    Returns
    -------
    arrival_increment_samples : array
        Array of arrival increments in bytes at the specified times.
    """
    arrival_times = np.asarray(arrival_times, dtype=float)
    times_to_sample = np.asarray(times_to_sample, dtype=float)
    arrival_sizes = np.asarray(arrival_sizes, dtype=float)

    if arrival_times.ndim != 1 or times_to_sample.ndim != 1 or arrival_sizes.ndim != 1:
        raise ValueError("arrival_times, times_to_sample, and arrival_sizes must be 1D arrays.")
    if len(arrival_times) == 0:
        raise ValueError("arrival_times must be non-empty.")
    if T <= 0:
        raise ValueError("T must be positive.")

    # Sort arrival times and arrival sizes for searchsorted
    sorted_indices = np.argsort(arrival_times)
    arrival_times = arrival_times[sorted_indices]
    arrival_sizes = arrival_sizes[sorted_indices]

    left_idx = np.searchsorted(arrival_times, times_to_sample, side="left")
    right_idx = np.searchsorted(arrival_times, times_to_sample + T, side="left")
    arrival_increment_bytes_samples = np.zeros(len(times_to_sample), dtype=float)
    for i in range(len(times_to_sample)):
        arrival_increment_bytes_samples[i] = np.sum(arrival_sizes[left_idx[i]:right_idx[i]])

    return arrival_increment_bytes_samples

def sample_increments_of_arrivals(arrival_times, T, times_to_sample, event_type="count"):
    """
    Sample arrival increments at specific times.

    Parameters
    ----------
    arrival_times : array-like
        1D array of arrival timestamps.
    T : float
        Window length used to count arrivals in [t, t+T).
    times_to_sample : array-like
        1D array of timestamps at which to sample arrival increment.
    event_type : str, optional
        Type of event to sample. Default is "count".
    Returns
    -------
    arrival_increment_samples : array
        Array of arrival increments at the specified times.
    """
    arrival_times = np.asarray(arrival_times, dtype=float)
    times_to_sample = np.asarray(times_to_sample, dtype=float)

    if arrival_times.ndim != 1 or times_to_sample.ndim != 1:
        raise ValueError("arrival_times and times_to_sample must be 1D arrays.")
    if len(arrival_times) == 0:
        raise ValueError("arrival_times must be non-empty.")
    if T <= 0:
        raise ValueError("T must be positive.")

    # Sort arrival times for searchsorted
    arrival_times = np.sort(arrival_times)

    left_idx = np.searchsorted(arrival_times, times_to_sample, side="left")
    right_idx = np.searchsorted(arrival_times, times_to_sample + T, side="left")
    if event_type == "binary":
        arrival_increment_samples = (right_idx - left_idx) > 0  # convert to binary increments (1 if at least one arrival, else 0)
    elif event_type == "idx":
        arrival_increment_samples = []
        for l, r in zip(left_idx, right_idx):
            arrival_increment_samples.append([i for i in range(l, r)])
    else:
        arrival_increment_samples = right_idx - left_idx
    return arrival_increment_samples

def autocorr_arrival_increments(
    arrival_increments,
    max_lag=None,
    normalize=True,
    subtract_mean=True,
):
    """
    Compute the autocorrelation of arrival increments.

    Parameters
    ----------
    arrival_increments : array-like
        1D array of arrival increments.
    max_lag : int or None
        Maximum lag in number of arrivals. If None, returns all lags.
    normalize : bool
        If True, return normalized autocorrelation.
    subtract_mean : bool
        If True, subtract the mean from the series before computing autocorrelation.
    Returns
    -------
    result : dict
        {
            "lags": lags in sample index,
            "autocorr": autocorrelation values
        }
    """

    arrival_increments = np.asarray(arrival_increments, dtype=float)

    if arrival_increments.ndim != 1:
        raise ValueError("arrival_increments must be a 1D array.")
    if len(arrival_increments) == 0:
        raise ValueError("arrival_increments must be non-empty.")

    x = arrival_increments.astype(float).copy()
    
    if subtract_mean:
        x = x - x.mean()

    corr = np.correlate(x, x, mode="full")
    lags = np.arange(-len(x) + 1, len(x))

    if normalize:
        denom = np.sum(x**2)
        if denom == 0:
            raise ValueError("Cannot normalize because the series has zero energy.")
        corr = corr / denom

    mask = lags >= 1 # exclude lag 0 to focus on correlation between different increments
    if max_lag is not None:
        if max_lag < 0:
            raise ValueError("max_lag must be non-negative.")
        mask = mask & (lags <= max_lag)
    lags = lags[mask]
    corr = corr[mask]

    return {
        "arrival_increment": arrival_increments,
        "lags": lags,
        "autocorr": corr,
    }

def crosscorr_qsize_vs_arrival_increments(
    arrival_increments,
    queue_sizes,
    times,
    max_lag=None,
    normalize=True,
    subtract_mean=True,
):
    """
    Compute the cross-correlation between total queue size and arrival increments.

    Parameters
    ----------
    arrival_increments : array-like
        1D array of arrival increments.
        Increment at time t_i is the number of arrivals in [t_i, t_i + T).
    queue_sizes : array-like
        1D array of total queue size values, one per timestamp.
        queue_sizes[i] is the total queue size observed at times[i].
    times : array-like
        1D array of timestamps.
    max_lag : int or None
        Maximum lag in number of arrivals. If None, returns all lags.
    normalize : bool
        If True, return normalized cross-correlation.

    Returns
    -------
    result : dict
        {
            "times": times,
            "queue_sizes": aligned queue size series,
            "arrival_increment": arrival_increments,
            "lags": lags in sample index,
            "crosscorr": cross-correlation values
        }

    Notes
    -----
    For each time t_i:
        arrival_increments[i] = #{ arrivals in [t_i, t_i + T) }

    Cross-correlation is computed between:
        x[i] = queue_sizes[i]
        y[i] = arrival_increment[i]

    With numpy.correlate(x, y, mode='full'), a positive lag means:
        queue_sizes earlier are correlated with future arrival increments.
    """

    arrival_increments = np.asarray(arrival_increments, dtype=float)
    queue_sizes = np.asarray(queue_sizes, dtype=float)
    times = np.asarray(times, dtype=float)

    if arrival_increments.ndim != 1 or queue_sizes.ndim != 1 or times.ndim != 1:
        raise ValueError("arrival_increments, queue_sizes, and times must be 1D arrays.")
    if len(arrival_increments) != len(queue_sizes) or len(arrival_increments) != len(times):
        raise ValueError("arrival_increments, queue_sizes, and times must have the same length.")
    if len(arrival_increments) == 0:
        raise ValueError("Inputs must be non-empty.")

    x = queue_sizes.astype(float).copy()
    y = arrival_increments.astype(float).copy()
    
    if subtract_mean:
        x = x - x.mean()
        y = y - y.mean()

    corr = np.correlate(x, y, mode="full")
    lags = np.arange(-len(x) + 1, len(x))

    if normalize:
        denom = np.sqrt(np.sum(x**2) * np.sum(y**2))
        if denom == 0:
            raise ValueError("Cannot normalize because one series has zero energy.")
        corr = corr / denom

    mask = lags >= 0
    # mask = np.ones(len(lags), dtype=bool)  # keep all lags, including negative ones
    if max_lag is not None:
        if max_lag < 0:
            raise ValueError("max_lag must be non-negative.")
        mask = mask & (lags <= max_lag)
    lags = lags[mask]
    corr = corr[mask]

    return {
        "times": times,
        "queue_sizes": queue_sizes,
        "arrival_increment": arrival_increments,
        "lags": lags,
        "crosscorr": corr,
    }

def visualize_totalQ_and_ECN(times, queue_size_samples, queue_ECN_samples, file_path):
    """
    Visualize the total queue size and ECN marking samples over time.

    Parameters
    ----------
    times : array-like
        1D array of timestamps corresponding to the samples.
    queue_size_samples : array-like
        1D array of total queue size samples at the corresponding times.
    queue_ECN_samples : array-like
        1D array of ECN marking samples (0 or 1) at the corresponding times.
    """

    times = np.asarray(times)
    queue_sizes = np.asarray(queue_size_samples)
    queue_ECN_samples = np.asarray(queue_ECN_samples)

    fig, axes = plt.subplots(2, 1, figsize=(60, 40))

    # --------------------------------------------------
    # 1) Time series plot of queue size
    # --------------------------------------------------
    ax1 = axes[0]

    # First axis: queue size
    sc1 = ax1.scatter(times, queue_sizes, alpha=1.0, s=20, marker='o', label="Queue size", color='blue')
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Queue size (B)")
    ax1.grid(True, alpha=0.7)
    ax1.set_title("Total Queue Size over Time")
    # --------------------------------------------------
    # 2) Time series plot of ECN markings
    # --------------------------------------------------
    ax2 = axes[1]

    # Second axis: ECN markings
    sc2 = ax2.scatter(times, queue_ECN_samples, alpha=1.0, s=20, marker='s', label="ECN markings", color='red')
    ax2.set_xlabel("Time")
    ax2.set_ylabel("ECN marking (0 or 1)")
    ax2.grid(True, alpha=0.7)
    ax2.set_title("ECN Markings over Time")
    plt.tight_layout()
    plt.savefig(file_path + 'Q(t)_ECN(t).png')
    plt.close()    

def visualize_crosscorr_Ts(results, file_path):
    """
    Visualize the output of crosscorr_qsize_vs_arrival_increment(...)

    Parameters
    ----------
    result : list of dict
        Output of crosscorr_qsize_vs_arrival_increment(...)
    max_points_scatter : int
        Maximum number of points to show in the scatter plot.
        If there are more points, a random subset is used.
    """

    times = np.asarray(results[0]["times"])
    lags = np.asarray(results[0]["lags"])
    lags_time = lags * np.mean(np.diff(times))  # convert lags from sample index to time

    fig, ax = plt.subplots(1, 1, figsize=(30, 20))
    # --------------------------------------------------
    # 3) Cross-correlation for different T values
    # --------------------------------------------------
    for result in results:
        crosscorr = np.asarray(result["crosscorr"])
        ax.plot(lags_time, crosscorr, marker='o', linestyle='-', markersize=4, linewidth=2, label=f"T={result['T']} ns")
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Cross-correlation")
    ax.set_title("Cross-correlation")
    ax.grid(True, alpha=0.5)
    ax.set_ylim(bottom=-0.4, top=1.0)
    ax.set_yticks(np.arange(-0.4, 0.8, 0.2))
    ax.set_xticks(np.arange(0, np.max(lags_time), max(lags_time) / 20), labels=[f"{float(t/1000000):.1f}" for t in np.arange(0, np.max(lags_time), max(lags_time) / 20)])
    ax.tick_params(axis='y', labelsize=30)
    ax.legend(fontsize=30)
    plt.tight_layout()
    plt.savefig(file_path + 'crosscorr_qsize_vs_arrival_increment_diff_T_Sampling_90ns.png')
    plt.close()

def reconstructSignal(full_df_, linksRates, file_path):
    from pynufft import NUFFT

    full_df_copy = full_df_.copy()  # avoid modifying original
    full_df_copy['QueueSize'] = (full_df_copy['Delay'] * linksRates[1] / 8) + full_df_copy['PayloadSize']
    window = 0.0005 * 1e9
    # window = 0.01 * 1e9
    # startTime = 0.55 * 1e9
    # endTime = 0.6 * 1e9
    # full_df = full_df[(full_df['SentTime'] >= startTime) & (full_df['SentTime'] <= endTime)]
    full_df_copy = full_df_copy.sort_values(by=['SentTime', 'QueueSize'], ascending=[True, True]).reset_index(drop=True)
    reconstructed_dfs = []
    for path in full_df_copy['Path'].unique():
        full_df = full_df_copy[full_df_copy['Path'] == path]

        t_nonuniform_all = full_df['SentTime'].to_numpy()
        QueueSize_samples_all = full_df['QueueSize'].to_numpy()

        # do the NUFFT reconstruction for each window of size window
        t_nonuniform_list = []
        QueueSize_samples_list = []
        j = 0
        for i in range(0, len(t_nonuniform_all)):
            if (t_nonuniform_all[i] - t_nonuniform_all[j] > window) or (i == len(t_nonuniform_all) - 1):
                t_nonuniform_list.append(t_nonuniform_all[j:i])
                QueueSize_samples_list.append(QueueSize_samples_all[j:i])
                j = i
        # print(f"Number of windows: {len(t_nonuniform_list)}")
        # --- Define observation window ---
        t_uniform_list= []
        reconstructSignal = []
        # reconstructSignal_2 = []
        for i in range(len(t_nonuniform_list)):
            t_nonuniform = t_nonuniform_list[i]
            QueueSize_samples = QueueSize_samples_list[i]

            T = t_nonuniform.max() - t_nonuniform.min()  # Total observed time duration
            if T <= 0:
                continue
            t_shifted = t_nonuniform - t_nonuniform.min()  # Shift time to start from 0


            # --- Prepare NUFFT ---0
            nufft_obj = NUFFT()
            # Normalize time to [-0.5, 0.5) and convert to radians
            om = (t_shifted / T - 0.5) * 2 * np.pi  # Shape (M,)

            # Make sure om has shape (M, 1)
            om = om.reshape(-1, 1)

            # Grid configuration
            N = len(QueueSize_samples)  # Number of uniform frequency points (resolution)
            # N = max(1024, 4 * len(t_nonuniform))
            Kd = (int(2 * N),)  # Oversampled FFT grid (e.g., 2x of N)
            Jd = (6,)           # Kaiser-Bessel kernel size (use 6 or 8, NOT 4*N)

            # Plan NUFFT with these parameters
            nufft_obj.plan(om, (N,), Kd, Jd)

            # --- Perform NUFFT ---
            x = QueueSize_samples.astype(np.complex64)
            X_freq = nufft_obj.forward(x)
            # QueueSize_restore = nufft_obj.solve(x, solver='cg', maxiter=3)

            # --- Reconstruct delay signal on uniform time grid ---
            t_uniform_list.append(np.linspace(0, T, N, endpoint=False) + t_nonuniform.min())
            QueueSize_restore = nufft_obj.solve(X_freq,'cg', maxiter=30)
            # QueueSize_restore_2 = nufft_obj.adjoint(X_freq)
            # print(f"Number of points in window {i}: {len(QueueSize_restore)}, and {len(QueueSize_restore_2)} number of samples: {len(QueueSize_samples)}")
            reconstructSignal.append(QueueSize_restore)
            # reconstructSignal_2.append(QueueSize_restore_2)

        # --- Plotting ---
        full_time = np.concatenate(t_uniform_list)
        full_signal = np.concatenate([np.real(q) for q in reconstructSignal])
        from scipy.interpolate import interp1d
        continuous_function = interp1d(full_time, full_signal, kind='cubic', fill_value="extrapolate")
        t_query = np.linspace(full_time.min(), full_time.max(), 10000)
        q_query = continuous_function(t_query)


        plt.figure(figsize=(10, 6))
        plt.scatter(full_df['SentTime'], full_df['QueueSize'], color='b', label='Measurement Traffic', marker='x', s=1)
        # plot reconstructSignal
        for i in range(len(reconstructSignal)):
            plt.scatter(t_uniform_list[i], np.real(reconstructSignal[i]), color='r', label='Reconstructed Signal CG' if i == 0 else "", marker='o', s=1.5)
            # plt.scatter(t_uniform_list[i], np.real(reconstructSignal_2[i]), color='g', label='Reconstructed Signal adjoint' if i == 0 else "", marker='o', s=1.5)
        plt.plot(t_query, q_query, 'k-', label='Continuous Signal (Interpolated)', linewidth=0.2)
        plt.ylim(0, 19000)
        plt.legend()
        plt.title('Queue Size per time', fontsize=16)
        plt.grid()
        plt.xlabel('Time (ns)', fontsize=16)
        plt.ylabel('Size (B)', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(f'{file_path}reconstructedSignal.png')
        plt.close()
        # creat a DataFrame with the reconstructed signal
        reconstructed_df = pd.DataFrame({
            'Time': np.concatenate(t_uniform_list),
            'TotalQueueSize': np.real(np.concatenate(reconstructSignal)),
            'Path': path
        })
        reconstructed_dfs.append(reconstructed_df)

    return pd.concat(reconstructed_dfs, ignore_index=True)

def calculate_reconstructedSignal_delays(reconstructedSignal_df, df_res, linkRate):
    df_res['delay'] = {}
    df_res['delay']['event_linearInterp_timeAvg'] = {}
    df_res['sampleSize']['delay'] = {}
    for path in reconstructedSignal_df['Path'].unique():
        full_df = reconstructedSignal_df[reconstructedSignal_df['Path'] == path]
        full_df = full_df.sort_values(by='Time').reset_index(drop=True)
        df_res['totalPckts'][path] = len(full_df)

        time_diff = ((full_df['Time'].shift(-1) - full_df['Time']) * linkRate) / 8
    
        # Filter rows where the condition is met
        insert_rows = full_df[time_diff > full_df['TotalQueueSize']].copy()
        if not insert_rows.empty:
            insert_rows['Time'] = insert_rows['Time'] + (insert_rows['TotalQueueSize']  * 8 / linkRate).astype(int)
            insert_rows['TotalQueueSize'] = 0
            insert_rows['Path'] = path
            
            full_df = pd.concat([full_df, insert_rows], ignore_index=True).sort_values(by='Time').reset_index(drop=True)
        full_df = full_df.sort_values(by=['Time', 'TotalQueueSize'], ascending=[True, True]).reset_index(drop=True)
        
        full_df['Delay'] = ((full_df['TotalQueueSize'] * 8) / linkRate).astype(int)
        time = full_df['Time'].values
        values = full_df['Delay'].values

        linearInterp_time_average = np.sum(((values[:-1] + values[1:]) / 2) * np.diff(time)) / (time[-1] - time[0])
        df_res['delay']['event_linearInterp_timeAvg'][path] = linearInterp_time_average
        df_res['sampleSize']['delay'][path] = len(values)
        full_df = None
    reconstructedSignal_df = None
    return df_res

def find_sampling_rate(time, maxError):
    """Solve for rate >= 0 so 1 - sum(1 - exp(-rate * interarrivals)) / sum(rate * interarrivals) <= maxError using binary search."""
    if maxError <= 0 or time.size == 0:
        return 0.0
    
    interarrivals = np.diff(time)
    total_time = float(interarrivals.sum())
    low, high = 0.0, 1e-6
    f = lambda rate: 1 - (np.sum(-np.expm1(-rate * interarrivals)) / np.sum(rate * interarrivals))
    
    for _ in range(60):
        mid = (low + high) / 2
        if f(mid) > maxError: high = mid
        else: low = mid

    return (high + low) / 2

# -----------------------------
# Core metrics
# -----------------------------

def compute_iats(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    t = np.sort(t)
    x = np.diff(t)
    x = x[np.isfinite(x) & (x > 0)]
    return x

def cv_iat(t: np.ndarray) -> float:
    x = compute_iats(t)
    return float(np.std(x) / np.mean(x))

def idc_curve(t: np.ndarray, deltas: np.ndarray, min_bins: int = 30, max_bins: int = 5000):
    """
    Compute IDC(Δ) = Var(NΔ)/E[NΔ] for counts NΔ in bins of width Δ.
    Returns arrays (deltas_used, idc_values).
    """
    if len(t) < 2:
        return np.array([]), np.array([])
    t = np.asarray(t, dtype=float)
    t = np.sort(t)
    t0 = t[0]
    tt = t - t0
    T = tt[-1]
    if T <= 0:
        raise ValueError("Timestamp span is zero.")

    deltas = np.asarray(deltas, dtype=float)
    out_d, out_idc, out_mu, out_var = [], [], [], []
    for Delta in deltas:
        if not np.isfinite(Delta) or Delta <= 0:
            continue
        nbins = int(np.floor(T / Delta)) + 1
        # if nbins < min_bins or nbins > max_bins:
        #     continue
        edges = np.linspace(0, nbins * Delta, nbins + 1)
        counts, _ = np.histogram(tt, bins=edges)
        mu = counts.mean()
        if mu <= 0:
            continue
        var = counts.var(ddof=1) if nbins > 1 else 0.0
        out_d.append(Delta)
        out_idc.append(var / mu)
        out_mu.append(mu)
        out_var.append(var)

    if len(out_d) == 0:
        raise RuntimeError("No valid deltas produced IDC values. Adjust min_bins/max_bins or deltas.")
    out_d = np.array(out_d)
    out_idc = np.array(out_idc)
    out_mu = np.array(out_mu)
    out_var = np.array(out_var)
    order = np.argsort(out_d)
    return out_d[order], out_idc[order], out_mu[order], out_var[order]

def find_idc_plateau_delta(deltas: np.ndarray,
                           idc: np.ndarray,
                           slope_thresh: float = 0.10,
                           consec: int = 4,
                           smooth_window: int = 5) -> float:
    """
    Pick Δ* as the earliest delta where the (smoothed) log-log slope of IDC
    stays below slope_thresh for 'consec' consecutive points.
    """
    deltas = np.asarray(deltas, float)
    idc = np.asarray(idc, float)
    good = np.isfinite(deltas) & np.isfinite(idc) & (deltas > 0) & (idc > 0)
    d = deltas[good]
    y = idc[good]
    if d.size < smooth_window + consec:
        return float(np.median(d))  # fallback

    logd = np.log(d)
    logy = np.log(y)
    slope = np.abs(np.gradient(logy, logd))
    sm = np.empty_like(slope)
    for i in range(slope.size):
        lo = max(0, i - smooth_window + 1)
        sm[i] = np.median(slope[lo:i+1])

    for i in range(0, sm.size - consec + 1):
        if np.all(sm[i:i+consec] < slope_thresh):
            return float(d[i])
    return float(d[-1])

def rel_w1_to_exp_fit(t_sel: np.ndarray):
    """
    Relative Wasserstein-1 distance between empirical IATs and Exp(mean IAT).
    Uses quantile-based formula in 1D, no SciPy needed.
    Returns (relW1, W1, lambda_hat).
    """
    x = compute_iats(t_sel)
    if x.size < 5:
        return np.nan, np.nan, np.nan
    x = np.sort(x)
    n = x.size
    mu = x.mean()
    lam_hat = 1.0 / mu
    p = (np.arange(1, n + 1) - 0.5) / n
    q_exp = -np.log(1.0 - p) / lam_hat
    w1 = float(np.mean(np.abs(x - q_exp)))
    rel = float(w1 / mu)
    return rel, w1, float(lam_hat)

def idc_slope_over_region(deltas: np.ndarray, idc: np.ndarray, delta_min: float) -> float:
    """
    Fit slope of log(IDC) vs log(Δ) for Δ >= delta_min using least squares.
    Returns absolute slope; near 0 => IDC "flat" over that region.
    """
    deltas = np.asarray(deltas, float)
    idc = np.asarray(idc, float)
    good = np.isfinite(deltas) & np.isfinite(idc) & (deltas > 0) & (idc > 0) & (deltas >= delta_min)
    d = deltas[good]
    y = idc[good]
    if d.size < 4:
        return np.nan
    X = np.log(d)
    Y = np.log(y)
    Xc = X - X.mean()
    slope = float((Xc @ (Y - Y.mean())) / (Xc @ Xc))
    return abs(slope)

# -----------------------------
# Sampling building blocks
# -----------------------------

def bernoulli_thin(t: np.ndarray, q: float, rng: np.random.Generator) -> np.ndarray:
    t = np.asarray(t, float)
    if q >= 1.0:
        return t.copy()
    keep = rng.random(t.size) < q
    return t[keep]

def local_rate_equalized_thin(t: np.ndarray, lambda_target: float, Delta: float, rng: np.random.Generator) -> np.ndarray:
    """
    Time-varying thinning: q(t_i) = min(1, lambda_target / hat_lambda(t_i)),
    where hat_lambda estimated via a sliding window of width Delta centered at t_i.
    Implemented in O(n) using two pointers.
    """
    t = np.asarray(t, float)
    t = np.sort(t)
    n = t.size
    if n == 0:
        return t
    if Delta <= 0:
        raise ValueError("Delta must be positive for local rate estimation.")

    left = 0
    right = 0
    half = Delta / 2.0
    keep = np.zeros(n, dtype=bool)

    for i in range(n):
        ti = t[i]
        while left < n and t[left] < ti - half:
            left += 1
        while right < n and t[right] < ti + half:
            right += 1
        count = max(1, right - left)  # include at least itself
        lam_hat = count / Delta
        q = min(1.0, lambda_target / lam_hat) if lam_hat > 0 else 1.0
        keep[i] = (rng.random() < q)
    return t[keep]

def soft_decluster(t: np.ndarray, Delta: float, cap_c: int, rng: np.random.Generator) -> np.ndarray:
    """
    Partition into windows of length Delta and keep up to cap_c packets per window uniformly at random.
    """
    t = np.asarray(t, float)
    t = np.sort(t)
    if t.size == 0:
        return t
    if Delta <= 0:
        raise ValueError("Delta must be positive for declustering.")
    if cap_c < 0:
        raise ValueError("cap_c must be >= 0.")

    t0 = t[0]
    bins = np.floor((t - t0) / Delta).astype(int)

    kept_idx = []
    start = 0
    while start < t.size:
        b = bins[start]
        end = start + 1
        while end < t.size and bins[end] == b:
            end += 1
        idx = np.arange(start, end)
        if idx.size <= cap_c:
            kept_idx.extend(idx.tolist())
        else:
            chosen = rng.choice(idx, size=cap_c, replace=False)
            kept_idx.extend(chosen.tolist())
        start = end

    kept_idx = np.array(kept_idx, dtype=int)
    kept_idx.sort()
    return t[kept_idx]

# -----------------------------
# Full pipeline
# -----------------------------

def e2e_poisson_like_sampler(t,
                            N_min: int,
                            relW1_tol: float = 0.10,
                            idc_slope_tol: float = 0.15,
                            deltas_for_idc=None,
                            plateau_slope_thresh: float = 0.10,
                            rng_seed: int = 0,
                            caps=(1,),
                            max_delta_for_idc=50000.0, df_name=""):
    """
    Multi-stage sampling to maximize number of selected samples while aiming for Poisson-like selection times.

    Stages:
      1) Global Bernoulli thinning to reach ~N_min
      2) If needed: local-rate equalized thinning using Δ* from IDC plateau
      3) If needed: soft declustering at Δ* with cap c in 'caps', with optional final thinning

    Returns:
      t_sel, report_dict
    """
    rng = np.random.default_rng(rng_seed)
    t = np.asarray(t, float)
    t = t[np.isfinite(t)]
    t = np.sort(t)
    if t.size < 5:
        raise ValueError("Need at least 5 timestamps.")

    t = np.unique(t)  # remove duplicates (zero IATs)
    if t.size < 5:
        raise ValueError("Need at least 5 unique timestamps.")

    T = t[-1] - t[0]
    if T <= 0:
        raise ValueError("Timestamp span must be positive.")
    N = t.size
    lambda_target = N_min / T

    Delta_star, mu = find_delta_for_empty_prob(t, p0_max=0.10)
    d0, idc0, mu0, var0 = idc_curve(t, np.array([Delta_star]))
    print(f"Delta star for empty prob 0.1: {Delta_star}, mu: {mu}, idc: {idc0[0]}")
    return t, {}
    # # Choose IDC deltas if not provided
    # if deltas_for_idc is None:
    #     d_min = max(T / 750000.0, np.finfo(float).eps)
    #     d_max = max(T / 1800.0, d_min * 10.0)
    #     deltas_for_idc = np.logspace(np.log10(d_min), np.log10(d_max), 200)
    # deltas_for_idc = np.asarray(deltas_for_idc, float)

    # # Original diagnostics + Δ*
    # CV0 = cv_iat(t)
    # d0, idc0, mu0, var0 = idc_curve(t, deltas_for_idc)
    # Delta_star = find_idc_plateau_delta(d0, idc0, slope_thresh=plateau_slope_thresh)
    # plt.figure(figsize=(10, 6))
    # plt.plot(d0, idc0, marker='o', linewidth=1)
    # # print(f"Min d0: {d0.min()}, Max d0: {d0.max()}")
    # # print(f"Min idc0: {idc0.min()}, Max idc0: {idc0.max()}")
    # # plt.xscale('log'); plt.yscale('log')
    # plt.axvline(Delta_star, linestyle='--')
    # plt.title("Original IDC(Δ) with Δ* (vertical dashed)")
    # plt.xlabel("Δ"); plt.ylabel("IDC(Δ)")
    # plt.grid(True, which="both", linestyle="--", alpha=0.5)
    # plt.tight_layout()
    # plt.savefig(f"original_idc_plot_{df_name.split('/')[-1]}_{t[0]:.0f}_{t[-1]:.0f}.png")
    # # # find the d0 with the closest idc0 to 1.0
    # idx_closest_to_one = np.argmin(np.abs(idc0 - 1.0))
    # d_closest_to_one = d0[idx_closest_to_one]
    # print(f"Delta closest to IDC=1.0: {d_closest_to_one}")
    # # print(f"\lambda of the minimizer delta: { mu0[idx_closest_to_one]/ d_closest_to_one} packets/second")
    # print(f"\lambda of the minimizer delta: { mu0[idx_closest_to_one]} packets/delta")
    # print(f"average rate: {len(t) / (T * 1e-9)} packets/second")

    # # plt.figure(figsize=(10, 6))
    # # plt.plot(mu0, idc0, marker='o', linewidth=1)
    # # plt.title("Original μ curve")
    # # plt.ylabel("μ(Δ)"); plt.xlabel("Δ")
    # # plt.grid(True, which="both", linestyle="--", alpha=0.5)
    # t2 = soft_decluster(t, Delta=d_closest_to_one, cap_c=1, rng=rng)
    # return t2, {}

    # report = {
    #     "N_total": int(N),
    #     "T": float(T),
    #     "CV_original": float(CV0),
    #     "Delta_star": float(Delta_star),
    #     "stages": []
    # }

    # def validate(tsel):
    #     rel, w1, lam_hat = rel_w1_to_exp_fit(tsel)
    #     ds, idcs = idc_curve(tsel, deltas_for_idc)
    #     slope = idc_slope_over_region(ds, idcs, Delta_star)
    #     return {
    #         "N_sel": int(tsel.size),
    #         "rate_sel": float(tsel.size / T),
    #         "relW1": float(rel),
    #         "W1": float(w1),
    #         "lambda_exp_hat": float(lam_hat),
    #         "idc_slope_abs": float(slope),
    #         "d_idc": ds,
    #         "idc": idcs
    #     }

    # def passes(v):
    #     ok1 = (v["relW1"] <= relW1_tol)
    #     ok2 = (np.isfinite(v["idc_slope_abs"]) and v["idc_slope_abs"] <= idc_slope_tol)
    #     return ok1 and ok2

    # def declusting_sampling():
    #     qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #     t2_maxSize = 0
    #     t2_best = t
    #     v2_best = validate(t)
    #     c_best = None
    #     for c in caps:
    #         for q_final in qs:
    #             tries = 20
    #             while tries > 0:
    #                 base = t
    #                 t2 = soft_decluster(base, Delta=Delta_star, cap_c=int(c), rng=rng)
    #                 # print(F"After soft decluster with cap {c}, from time {t2[0]} to {t2[-1]} with size {t2.size} out of {base.size} Delta_star {Delta_star}")
    #                 t2_candidate = bernoulli_thin(t2, q_final, rng)
    #                 v2_candidate = validate(t2_candidate)
    #                 if passes(v2_candidate) and t2_maxSize < t2_candidate.size:
    #                     t2_best = t2_candidate
    #                     v2_best = v2_candidate
    #                     t2_maxSize = t2_candidate.size
    #                     c_best = c
    #                     # print(F"from time {t2[0]} to {t2[-1]} with size {t2.size} we thin with q {q_final} to size {t2_candidate.size}")
    #                     break
    #                 tries -= 1
    #     return t2_best, v2_best, c_best
    
    # # Stage 1: global thinning (max yield)
    # q0 = min(1.0, N_min / N)
    # t1 = bernoulli_thin(t, q0, rng)
    # d1, idc1, mu1, var1 = idc_curve(t1, deltas_for_idc)
    # Delta_star_1 = find_idc_plateau_delta(d1, idc1, slope_thresh=plateau_slope_thresh)
    # plt.figure(figsize=(10, 6))
    # plt.plot(d1, idc1, marker='o', linewidth=1)
    # # plt.xscale('log'); plt.yscale('log')
    # plt.axvline(Delta_star_1, linestyle='--')
    # plt.title("IDC(Δ) with Δ* (vertical dashed)")
    # plt.xlabel("Δ"); plt.ylabel("IDC(Δ)")
    # plt.grid(True, which="both", linestyle="--", alpha=0.5)
    # plt.tight_layout()
    # plt.savefig(f"after_thinning_idc_plot_{df_name.split('/')[-1]}_{t[0]:.0f}_{t[-1]:.0f}.png")
    # return t, report

    # v1 = validate(t1)
    # report["stages"].append({"name": "stage1_global_thin", "q": float(q0),
    #                          **{k: v1[k] for k in v1 if k not in ("d_idc","idc")}})

    # if passes(v1):
    #     report["selected_stage"] = "stage1_global_thin"
    #     return t1, report

    # # Stage 2: soft declustering (cap >= 1)    
    # t2_best, v2_best, c = declusting_sampling()
    # report["stages"].append({"name": "stage2_soft_decluster", "cap_c": c,
    #                             **{k: v2_best[k] for k in v2_best if k not in ("d_idc","idc")}})
    
    # if passes(v2_best):
    #     report["selected_stage"] = f"stage2_soft_decluster_cap{c}"
    #     return t2_best, report
    
    # # stage 3: declustering of max_delta_for_idc if not yet tried
    # Delta_star = max_delta_for_idc
    # t2_best, v2_best, c = declusting_sampling()
    # report["stages"].append({"name": "stage3_soft_decluster_maxDelta", "cap_c": c,
    #                             **{k: v2_best[k] for k in v2_best if k not in ("d_idc","idc")}})
    # if passes(v2_best):
    #     report["selected_stage"] = f"stage3_soft_decluster_cap{c}_maxDelta"
    #     return t2_best, report
    
    # report["selected_stage"] = "none"
    # return [], report


# def trim_counts_round_robin_to_idc_multiscale(
#     t: np.ndarray,
#     p0_max: float = 0.10,
#     target_idc: float = 1.0,
#     tol_primary: float = 0.001,
#     max_rounds: int = 300,
#     min_mean_per_bin: float = 0.2,
#     min_total_keep: int = 1000,
#     policy: str = "reduce_if_gt1",              # "closest" or "reduce_if_gt1"
#     check_factors=(2, 3, 4, 5, 6),                 # enforce safeguards at 2Δ,4Δ (can use just (2,))
#     allow_worsen: float = 0.0,            # allow tiny worsening (e.g., 1e-6)
#     rng_seed: int = 0,
#     return_debug: bool = True,
# ):
#     """
#     Your iterative, round-robin, 'remove at most 1 per interval' algorithm,
#     with a multi-scale safeguard.

#     Workflow:
#       1) Find Δ via find_delta_for_empty_prob(t, p0_max).
#       2) Bin into Δ => counts X_i.
#       3) Maintain kept counts Y_i initialized to X_i.
#       4) Iterate rounds:
#          - traverse bins; for each bin i, try Y_i -> Y_i-1
#          - accept only if:
#              (a) primary objective improves (IDCΔ closer to target, or reduces if >1), AND
#              (b) IDC at each coarser scale (e.g., 2Δ,4Δ) does NOT worsen (beyond allow_worsen).
#       5) After convergence, sample exactly Y_i packets uniformly per bin.

#     Returns:
#       t_selected, info (includes Delta, initial/final IDC at each scale, trace if requested)
#     """
#     rng = np.random.default_rng(rng_seed)

#     # --- small helpers (local) ---
#     def _sanitize_times(tt):
#         tt = np.asarray(tt, dtype=float)
#         tt = tt[np.isfinite(tt)]
#         tt = np.unique(np.sort(tt))
#         if tt.size < 2:
#             raise ValueError("Need at least 2 finite timestamps.")
#         return tt

#     def _bin_ids(tt, Delta):
#         tt = _sanitize_times(tt)
#         if Delta <= 0:
#             raise ValueError("Delta must be > 0.")
#         t0 = tt[0]
#         x = tt - t0
#         T = x[-1]
#         nb = int(np.floor(T / Delta)) + 1
#         bid = np.floor(x / Delta).astype(int)
#         bid = np.clip(bid, 0, nb - 1)
#         return tt, bid, nb, t0

#     def _idc_from_S_SS(nbins, S, SS):
#         # IDC = Var/Mean, Var = E[Y^2] - E[Y]^2, population variance (ddof=0)
#         if nbins <= 0:
#             return np.inf
#         mu = S / nbins
#         if mu <= 0:
#             return np.inf
#         ey2 = SS / nbins
#         var = ey2 - mu * mu
#         if var < 0 and var > -1e-12:
#             var = 0.0
#         return float(var / mu)

#     def _objective(val):
#         if policy == "reduce_if_gt1":
#             return max(0.0, val - target_idc)
#         return abs(val - target_idc)

#     def _select_packets_per_bin(tt, bid, y, rng_):
#         order = np.argsort(tt)
#         tt = tt[order]
#         bid = bid[order]
#         selected_idx = []
#         n = tt.size
#         i = 0
#         while i < n:
#             b0 = bid[i]
#             j = i + 1
#             while j < n and bid[j] == b0:
#                 j += 1
#             idx = np.arange(i, j)
#             k = int(y[b0])
#             if k > 0:
#                 if idx.size <= k:
#                     selected_idx.extend(idx.tolist())
#                 else:
#                     chosen = rng_.choice(idx, size=k, replace=False)
#                     selected_idx.extend(chosen.tolist())
#             i = j
#         selected_idx = np.array(selected_idx, dtype=int)
#         selected_idx.sort()
#         return tt[selected_idx]

#     # --- Step 1: choose Δ using your empty-bin rule ---
#     Delta, mu_scan = find_delta_for_empty_prob(t, p0_max=p0_max)
#     if Delta is None:
#         raise RuntimeError(f"No Δ found with empirical empty-bin prob <= {p0_max}.")

#     # --- Step 2: bin at Δ and initialize counts ---
#     t_clean, b_fine, n_fine, t0 = _bin_ids(t, Delta)
#     X = np.bincount(b_fine, minlength=n_fine).astype(int)
#     Y = X.copy()

#     # Build multi-scale structures:
#     # scale factors include 1 (fine) plus check_factors
#     factors = [1] + [int(f) for f in check_factors if int(f) >= 2]
#     # unique and sorted
#     factors = sorted(set(factors))

#     # For each factor f, define coarse bin index for each fine bin i: coarse = i // f
#     # counts_f = sum of Y over fine bins mapping to coarse bins
#     scales = {}
#     for f in factors:
#         map_f = (np.arange(n_fine) // f).astype(int)
#         n_coarse = int(map_f.max()) + 1
#         counts_f = np.bincount(map_f, weights=Y, minlength=n_coarse).astype(int)
#         S = float(counts_f.sum())
#         SS = float((counts_f * counts_f).sum())
#         idc = _idc_from_S_SS(n_coarse, S, SS)
#         scales[f] = {
#             "map": map_f,
#             "n": n_coarse,
#             "counts": counts_f,
#             "S": S,
#             "SS": SS,
#             "idc": float(idc),
#         }

#     idc0_all = {f: scales[f]["idc"] for f in factors}
#     idc_primary = scales[1]["idc"]

#     debug = []
#     if return_debug:
#         debug.append({
#             "round": 0,
#             "idc_primary": float(idc_primary),
#             "idc_by_factor": {f: float(scales[f]["idc"]) for f in factors},
#             "kept_total": int(Y.sum()),
#             "mean_per_fine_bin": float(Y.mean()),
#             "removed_this_round": 0
#         })

#     # --- Round-robin trimming with multi-scale safeguard ---
#     for r in range(1, max_rounds + 1):
#         if _objective(scales[1]["idc"]) <= tol_primary:
#             break
#         # if Y.sum() < min_total_keep:
#         #     break
#         # if Y.mean() < min_mean_per_bin:
#         #     break

#         changed = 0
#         eligible = np.where(Y > 0)[0]
#         if eligible.size == 0:
#             break

#         for i in eligible:
#             if Y[i] <= 0:
#                 continue

#             cur_idc_primary = scales[1]["idc"]
#             cur_obj = _objective(cur_idc_primary)

#             # We'll attempt decrement in fine bin i:
#             # This affects each scale f at coarse bin j = i//f
#             # We'll compute prospective IDC for each scale without committing, then commit if accepted.
#             prospective = {}

#             # First compute prospective per-scale IDC after decrement
#             for f in factors:
#                 j = i // f
#                 sc = scales[f]
#                 old = sc["counts"][j]
#                 if old <= 0:
#                     # shouldn't happen if Y[i]>0, but safe guard
#                     prospective[f] = (sc["idc"], sc["S"], sc["SS"], old)
#                     continue
#                 S2 = sc["S"] - 1.0
#                 SS2 = sc["SS"] - float(2 * old - 1)  # old^2 - (old-1)^2 = 2*old-1
#                 idc2 = _idc_from_S_SS(sc["n"], S2, SS2)
#                 prospective[f] = (idc2, S2, SS2, old)

#             new_idc_primary = prospective[1][0]
#             new_obj = _objective(new_idc_primary)

#             # Primary acceptance
#             if policy == "reduce_if_gt1":
#                 accept_primary = (cur_idc_primary > target_idc) and (new_idc_primary < cur_idc_primary)
#             else:
#                 accept_primary = (new_obj < cur_obj)

#             if not accept_primary:
#                 continue

#             # Multi-scale safeguard: do not worsen coarser IDC beyond allow_worsen
#             ok_multi = True
#             for f in factors:
#                 if f == 1:
#                     continue
#                 if prospective[f][0] > scales[f]["idc"] + allow_worsen:
#                     ok_multi = False
#                     break

#             if not ok_multi:
#                 continue

#             # Commit decrement
#             Y[i] -= 1
#             for f in factors:
#                 j = i // f
#                 idc2, S2, SS2, old = prospective[f]
#                 scales[f]["counts"][j] = old - 1
#                 scales[f]["S"] = S2
#                 scales[f]["SS"] = SS2
#                 scales[f]["idc"] = float(idc2)

#             changed += 1
#             if _objective(scales[1]["idc"]) <= tol_primary:
#                 break

#         if return_debug:
#             debug.append({
#                 "round": r,
#                 "idc_primary": float(scales[1]["idc"]),
#                 "idc_by_factor": {f: float(scales[f]["idc"]) for f in factors},
#                 "kept_total": int(Y.sum()),
#                 "mean_per_fine_bin": float(Y.mean()),
#                 "removed_this_round": int(changed)
#             })

#         if changed == 0:
#             break

#     # --- Sample actual packets per fine bin according to final Y ---
#     t_sel = _select_packets_per_bin(t_clean, b_fine, Y, rng)

#     info = {
#         "Delta": float(Delta),
#         "t0": float(t0),
#         "p0_max": float(p0_max),
#         "factors": factors,
#         "allow_worsen": float(allow_worsen),
#         "policy": policy,
#         "tol_primary": float(tol_primary),
#         "initial_idc_by_factor": {f: float(idc0_all[f]) for f in factors},
#         "final_idc_by_factor": {f: float(scales[f]["idc"]) for f in factors},
#         "initial_total": int(X.sum()),
#         "final_total": int(Y.sum()),
#         "nbins_fine": int(n_fine),
#         "empty_prob_at_Delta": float(np.mean(X == 0)),
#         "mean_count_at_Delta": float(X.mean()),
#         "X_counts": X,
#         "Y_counts": Y,
#     }
#     if return_debug:
#         info["trace"] = debug

#     return t_sel, info
def plot_iat_distribution(t_before, t_after, t_sel_lambda=None, nbins=None, title_suffix=""):
    """
    Plot inter-arrival time (IAT) distributions BEFORE and AFTER trimming
    using stem plots, with Exponential(mean) reference curves.

    Parameters
    ----------
    t_before : array-like
        Packet timestamps BEFORE trimming
    t_after : array-like
        Packet timestamps AFTER trimming
    nbins : int or None
        Number of bins used to discretize IATs.
        If None, chosen automatically (Freedman–Diaconis rule).
    title_suffix : str
        Optional suffix for plot title
    """

    def compute_iat(t):
        t = np.asarray(t, float)
        t = t[np.isfinite(t)]
        t = np.unique(np.sort(t))
        if t.size < 2:
            return np.array([])
        return np.diff(t)

    def auto_nbins(iat):
        # Freedman–Diaconis rule with safety caps
        q25, q75 = np.percentile(iat, [25, 75])
        iqr = q75 - q25
        if iqr <= 0:
            return 30
        bw = 2 * iqr / (len(iat) ** (1 / 3))
        if bw <= 0:
            return 30
        nb = int(np.ceil((iat.max() - iat.min()) / bw))
        return int(np.clip(nb, 20, 200))

    # ---- compute IATs ----
    iat_before = compute_iat(t_before)
    iat_after = compute_iat(t_after)

    if iat_before.size == 0 or iat_after.size == 0:
        raise ValueError("Not enough timestamps to compute IATs.")

    mean_iat_before = iat_before.mean()
    if t_sel_lambda is not None:
        mean_iat_after = t_sel_lambda
    else:
        mean_iat_after = iat_after.mean()

    # ---- choose nbins automatically if needed ----
    if nbins is None:
        nbins = auto_nbins(np.concatenate([iat_before, iat_after]))

    # ---- common binning ----
    xmax = max(iat_before.max(), iat_after.max())
    bins = np.linspace(0, xmax, nbins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # ---- empirical PMFs ----
    p_before, _ = np.histogram(iat_before, bins=bins)
    p_after, _ = np.histogram(iat_after, bins=bins)

    p_before = p_before / p_before.sum()
    p_after = p_after / p_after.sum()

    # ---- exponential reference (converted to PMF scale) ----
    bin_width = bins[1] - bins[0]
    exp_before = (1 / mean_iat_before) * np.exp(-bin_centers / mean_iat_before) * bin_width
    exp_after = (1 / mean_iat_after) * np.exp(-bin_centers / mean_iat_after) * bin_width

    # ---- plot ----
    plt.figure(figsize=(8, 5))

    plt.stem(
        bin_centers, p_before,
        linefmt="C0-", markerfmt="C0o", basefmt=" ",
        label=f"Before, mean IAT={mean_iat_before:.3g}"
    )

    plt.plot(
        bin_centers, exp_before,
        "C0--", linewidth=2, label="Exp(mean before)"
    )

    plt.plot(
        bin_centers, exp_after,
        "C1--", linewidth=2, label="Exp(mean after)"
    )

    plt.stem(
        bin_centers, p_after,
        linefmt="C1-", markerfmt="C1s", basefmt=" ",
        label=f"After, mean IAT={mean_iat_after:.3g}"
    )

    plt.xlabel("Inter-arrival time")
    plt.ylabel("Probability")
    plt.title("Inter-arrival time distribution" + title_suffix)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("iat_distributions.png")
    plt.close()


def plot_bin_count_distributions(X, Y, max_k=None, title_suffix=""):
    """
    Plot empirical distributions of packet counts per Δ-bin:
      - before trimming (X)
      - after trimming (Y)

    Also overlays Poisson(mean) reference curves.

    Parameters
    ----------
    X : array-like
        Original bin counts per Δ
    Y : array-like
        Kept bin counts per Δ after trimming
    max_k : int or None
        Max count to plot on x-axis (defaults to max of X,Y)
    title_suffix : str
        Optional string appended to plot title
    """

    X = np.asarray(X, dtype=int)
    Y = np.asarray(Y, dtype=int)

    if max_k is None:
        max_k = max(X.max(), Y.max())

    k = np.arange(0, max_k + 1)

    # empirical PMFs
    px = np.bincount(X, minlength=max_k + 1) / X.size
    py = np.bincount(Y, minlength=max_k + 1) / Y.size

    # Poisson references
    mu_x = X.mean()
    mu_y = Y.mean()

    def poisson_pmf(mu, k):
        return np.array([exp(-mu) * mu**i / factorial(i) for i in k])

    p_pois_x = poisson_pmf(mu_x, k)
    p_pois_y = poisson_pmf(mu_y, k)

    plt.figure(figsize=(8, 5))

    plt.stem(k, px, linefmt="C0-", markerfmt="C0o", basefmt=" ",
             label=f"Before (X), mean={mu_x:.2f}")
    plt.stem(k, py, linefmt="C1-", markerfmt="C1s", basefmt=" ",
             label=f"After (Y), mean={mu_y:.2f}")

    plt.plot(k, p_pois_x, "C0--", alpha=0.6, label="Poisson(mean(X))")
    plt.plot(k, p_pois_y, "C1--", alpha=0.6, label="Poisson(mean(Y))")

    plt.xlabel("Packets per Δ-bin")
    plt.ylabel("Probability")
    plt.title("Distribution of packets per Δ-bin" + title_suffix)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("bin_count_distributions.png")
    plt.close()


def trim_counts_round_robin_J_two_scales(
    t: np.ndarray,
    p0_max: float = 0.05,
    target_idc: float = 1.0,
    tol_J: float = 0.005,
    max_rounds: int = 400,
    # objective weights
    w_idc1: float = 1.0,
    w_idc2: float = 1.0,
    w_acf1: float = 1.0,
    w_acf2: float = 1.0,
    allow_worsen: float = 0.0, # small slack (e.g., 1e-6) to avoid floating noise
    rng_seed: int = 0,
    return_debug: bool = True,
):
    """
    Round-robin trimming over Δ-bins (remove at most 1 per bin per round),
    where acceptance is based on the combined objective:

        J = w1 * |IDC(Δ) - target| + w2 * |IDC(2Δ) - target|

    Steps:
      1) Choose Δ using find_delta_for_empty_prob(t, p0_max).
      2) Bin arrivals into Δ-bins => counts X_i, initialize kept counts Y_i = X_i.
      3) Maintain IDC(Δ) and IDC(2Δ) incrementally.
      4) Traverse bins; for each bin i, tentatively decrement Y_i by 1,
         accept iff J decreases (by at least allow_worsen).
      5) Sample actual packets to match final Y_i (uniform within each bin).

    Returns:
      t_selected, info (Delta, initial/final IDC at Δ and 2Δ, trace if requested)
    """
    # rng = np.random.default_rng(rng_seed)
    rng = None

    # ---- helpers ----
    def _sanitize_times(tt):
        tt = np.asarray(tt, dtype=float)
        tt = tt[np.isfinite(tt)]
        tt = np.unique(np.sort(tt))
        if tt.size < 2:
            raise ValueError("Need at least 2 finite timestamps.")
        return tt

    def _bin_ids(tt, Delta):
        tt = _sanitize_times(tt)
        if Delta <= 0:
            raise ValueError("Delta must be > 0.")
        t0 = tt[0]
        x = tt - t0
        T = x[-1]
        nb = int(np.floor(T / Delta)) + 1
        bid = np.floor(x / Delta).astype(int)
        bid = np.clip(bid, 0, nb - 1)
        return tt, bid, nb, t0

    def _idc_from_S_SS(nbins, S, SS):
        if nbins <= 0:
            return np.inf
        mu = S / nbins
        if mu <= 0:
            return np.inf
        ey2 = SS / nbins
        var = ey2 - mu * mu
        if var < 0 and var > -1e-12:
            var = 0.0
        return float(var / mu)

    def _select_packets_per_bin(tt, bid, y, rng_):
        order = np.argsort(tt)
        tt = tt[order]
        bid = bid[order]
        selected_idx = []
        n = tt.size
        i = 0
        while i < n:
            b0 = bid[i]
            j = i + 1
            while j < n and bid[j] == b0:
                j += 1
            idx = np.arange(i, j)
            k = int(y[b0])
            if k > 0:
                if idx.size <= k:
                    selected_idx.extend(idx.tolist())
                else:
                    chosen = np.random.choice(idx, size=k, replace=False)
                    selected_idx.extend(chosen.tolist())
                    
            i = j
        selected_idx = np.array(selected_idx, dtype=int)
        selected_idx.sort()
        return tt[selected_idx]
    
    def _stats_init(y):
        y = y.astype(int, copy=True)
        n = y.size
        S = float(y.sum())
        SS = float((y * y).sum())
        if n >= 2:
            P = float((y[:-1] * y[1:]).sum())
        else:
            P = 0.0
        return y, n, S, SS, P

    def _idc_acf_from_stats(n, S, SS, P):
        if n <= 0:
            return np.inf, 0.0
        mu = S / n
        if mu <= 0:
            return np.inf, 0.0
        var = SS / n - mu * mu
        if var < 0 and var > -1e-12:
            var = 0.0
        idc = (var / mu) if mu > 0 else np.inf

        if n < 2 or var <= 0:
            acf1 = 0.0
        else:
            cov1 = (P / (n - 1)) - (mu * mu)
            acf1 = cov1 / var
        return float(idc), float(acf1)

    # def J(idc1, idc2):
    #     return w1 * abs(idc1 - target_idc) + w2 * abs(idc2 - target_idc)
    def J(idc1, idc2, acf1_1, acf1_2):
        return (
            w_idc1 * abs(idc1 - target_idc)
            + w_idc2 * abs(idc2 - target_idc)
            + w_acf1 * abs(acf1_1)
            + w_acf2 * abs(acf1_2)
        )

    # ---- Step 1: choose Δ ----
    Delta, mu = find_delta_for_empty_prob(t, p0_max=p0_max)
    # Delta, mu = find_delta_for_closest_mean_packets_per_bin(t, target_mean=1.0)
    # print(f"Chosen Delta: {Delta} with mean packets per bin: {mu}")
    if Delta is None:
        raise RuntimeError(f"No Δ found such that P(empty bin) <= {p0_max}.")

    # ---------- 2) bin at Δ ----------
    t_clean, bid, n_fine, t0 = _bin_ids(t, Delta)
    X = np.bincount(bid, minlength=n_fine).astype(int)
    Y = X.copy()

    # scale Δ
    y1, n1, S1, SS1, P1 = _stats_init(Y)

    # scale 2Δ by summing pairs
    map2 = (np.arange(n_fine) // 2).astype(int)
    n2 = int(map2.max()) + 1
    y2 = np.bincount(map2, weights=y1, minlength=n2).astype(int)
    y2, n2, S2, SS2, P2 = _stats_init(y2)

    idc1, acf1_1 = _idc_acf_from_stats(n1, S1, SS1, P1)
    idc2, acf1_2 = _idc_acf_from_stats(n2, S2, SS2, P2)
    Jcur = J(idc1, idc2, acf1_1, acf1_2)

    trace = []
    if return_debug:
        trace.append({
            "round": 0,
            "J": Jcur,
            "idc_Delta": idc1,
            "idc_2Delta": idc2,
            "acf1_Delta": acf1_1,
            "acf1_2Delta": acf1_2,
            "kept_total": int(Y.sum()),
            "mean_per_bin": float(Y.mean()),
            "removed_this_round": 0
        })
    rel, w1, lam_hat = rel_w1_to_exp_fit(t_clean)
    min_packets_per_bin = 0
    # ---------- 3) round-robin trimming ----------
    break_reason = None
    for r in range(1, max_rounds + 1):
        if Jcur <= tol_J:
            break_reason = "tol_J"
            break

        changed = 0
        eligible = np.where(Y > min_packets_per_bin)[0]
        if eligible.size == 0:
            break_reason = "no_eligible"
            break

        for i in eligible:
            if Y[i] <= min_packets_per_bin:
                continue

            # ---- propose decrement at Δ scale: y1[i] -> y1[i]-1 ----
            a1 = y1[i]
            if a1 <= min_packets_per_bin:
                continue

            # update S1, SS1, P1 incrementally
            S1_p = S1 - 1.0
            SS1_p = SS1 - float(2 * a1 - 1)

            P1_p = P1
            if n1 >= 2:
                if i > 0:
                    P1_p -= float(y1[i - 1])        # term (i-1,i) decreases by y_{i-1}
                if i < n1 - 1:
                    P1_p -= float(y1[i + 1])        # term (i,i+1) decreases by y_{i+1}

            idc1_p, acf1_1_p = _idc_acf_from_stats(n1, S1_p, SS1_p, P1_p)

            # ---- corresponding decrement at 2Δ scale in coarse bin j=i//2 ----
            j = i // 2
            a2 = y2[j]
            if a2 <= 1:
                continue

            S2_p = S2 - 1.0
            SS2_p = SS2 - float(2 * a2 - 1)

            P2_p = P2
            if n2 >= 2:
                if j > 0:
                    P2_p -= float(y2[j - 1])
                if j < n2 - 1:
                    P2_p -= float(y2[j + 1])

            idc2_p, acf1_2_p = _idc_acf_from_stats(n2, S2_p, SS2_p, P2_p)

            Jnew = J(idc1_p, idc2_p, acf1_1_p, acf1_2_p)

            # accept iff J decreases
            if Jnew < Jcur - allow_worsen:
                # commit fine, with probability proportional to Y[i]
                p_i = 1.0 * (Y[i] / Y.max())
                if p_i > 1.0:
                    p_i = 1.0

                if np.random.random() >= p_i:
                    continue

                # commit fine
                Y[i] -= 1
                y1[i] = a1 - 1
                S1, SS1, P1 = S1_p, SS1_p, P1_p

                # commit coarse
                y2[j] = a2 - 1
                S2, SS2, P2 = S2_p, SS2_p, P2_p

                # commit derived stats
                idc1, acf1_1 = idc1_p, acf1_1_p
                idc2, acf1_2 = idc2_p, acf1_2_p
                Jcur = Jnew
                changed += 1

                if Jcur <= tol_J:
                    break_reason = "tol_J"
                    break
        t_sel = _select_packets_per_bin(t_clean, bid, Y, rng)
        rel, w1, lam_hat = rel_w1_to_exp_fit(t_sel)
        empty_prob = float(np.sum(Y == 0)) / float(Y.size)
        if return_debug:
            trace.append({
                "round": r,
                "J": Jcur,
                "idc_Delta": idc1,
                "idc_2Delta": idc2,
                "acf1_Delta": acf1_1,
                "acf1_2Delta": acf1_2,
                "kept_total": int(Y.sum()),
                "mean_per_bin": float(Y.mean()),
                "removed_this_round": int(changed),
                "relW1": float(rel),
                "empty_prob": float(empty_prob)
            })

        if changed == 0:
            break_reason = "no_change"
            break

    # ---------- 4) final sampling ----------
    if (break_reason is None) or (break_reason == "no_eligible") or (break_reason == "no_change"):
        t_sel = []
    else:
        t_sel = _select_packets_per_bin(t_clean, bid, Y, rng)
    
    rel, _, _ = rel_w1_to_exp_fit(t_sel)
    # if rel >= 0.05:
    #     t_sel = []
    
    info = {
        "Delta": float(Delta),
        "p0_max": float(p0_max),
        "weights": {"w_idc1": w_idc1, "w_idc2": w_idc2, "w_acf1": w_acf1, "w_acf2": w_acf2},
        "final": {
            "J": float(Jcur),
            "idc_Delta": float(idc1),
            "idc_2Delta": float(idc2),
            "acf1_Delta": float(acf1_1),
            "acf1_2Delta": float(acf1_2),
            "total_kept": int(Y.sum()),
            "relW1": float(rel),
        },
        "initial_total": int(X.sum()),
        "final_total": int(Y.sum()),
        "X_counts": X,
        "Y_counts": Y,
    }
    if return_debug:
        info["trace"] = trace

    return t_sel, info
def idc_derivative_by_local_averaging(
    deltas,
    idc_values,
    d1,
    half_window_points=40,
):
    """
    Estimate the derivative of IDC at delta = d1 using local averaging
    on the left and right of d1.

    The derivative is computed as:
        (x2 - x1) / (t2 - t1)
    where:
        x1 = mean IDC on the left side of d1
        x2 = mean IDC on the right side of d1
        t1 = mean delta on the left side of d1
        t2 = mean delta on the right side of d1

    Parameters
    ----------
    deltas : array-like
        1D strictly increasing array of delta values.
    idc_values : array-like
        1D array of IDC values corresponding to deltas.
    d1 : float
        Delta around which the derivative is estimated.
    half_window_points : int
        Number of points to use on each side of d1.

    Returns
    -------
    derivative : float
        Estimated derivative d(IDC)/d(delta) at d1.
    info : dict
        Diagnostic information about the computation.
    """
    deltas = np.asarray(deltas, dtype=float)
    idc_values = np.asarray(idc_values, dtype=float)

    if deltas.ndim != 1 or idc_values.ndim != 1:
        raise ValueError("deltas and idc_values must be 1D arrays")
    if len(deltas) != len(idc_values):
        raise ValueError("deltas and idc_values must have the same length")
    if len(deltas) < 2 * half_window_points + 1:
        raise ValueError("Not enough points for the requested half_window_points")
    if not np.all(np.diff(deltas) > 0):
        raise ValueError("deltas must be strictly increasing")
    if not (deltas[0] <= d1 <= deltas[-1]):
        raise ValueError("d1 must lie within the range of deltas")

    # Remove invalid points
    mask = np.isfinite(deltas) & np.isfinite(idc_values)
    deltas = deltas[mask]
    idc_values = idc_values[mask]

    # if len(deltas) < 2 * half_window_points + 1:
    #     raise ValueError("Not enough finite points after filtering")

    # Find the insertion location of d1
    idx = np.searchsorted(deltas, d1)

    # Left block: points immediately before d1
    left_start = idx - half_window_points
    left_end = idx

    # Right block: points at/after d1
    right_start = idx
    right_end = idx + half_window_points

    if left_start < 0 or right_end > len(deltas):
        # raise ValueError("d1 is too close to the boundary for the requested window size")
        left_start = max(0, left_start)
        right_end = min(len(deltas), right_end)

    left_deltas = deltas[left_start:left_end]
    left_idc = idc_values[left_start:left_end]

    right_deltas = deltas[right_start:right_end]
    right_idc = idc_values[right_start:right_end]

    t1 = np.mean(left_deltas)
    t2 = np.mean(right_deltas)
    x1 = np.mean(left_idc)
    x2 = np.mean(right_idc)

    if t2 == t1:
        raise ValueError("Mean delta values on the two sides are equal; cannot divide by zero")

    derivative = (x2 - x1) / (t2 - t1)

    return derivative, {
        "d1": d1,
        "t1": t1,
        "t2": t2,
        "x1": x1,
        "x2": x2,
        "left_count": len(left_deltas),
        "right_count": len(right_deltas),
    }

def idc_derivative_at_delta(
    deltas,
    idc_values,
    d1,
    half_window_points=2000,
    poly_order=3,
):
    """
    Estimate d(IDC)/d(delta) at delta = d1 from noisy IDC data
    using a local polynomial fit.

    Parameters
    ----------
    deltas : array-like
        1D array of delta values.
    idc_values : array-like
        1D array of IDC(delta) values.
    d1 : float
        Delta at which to estimate the derivative.
    half_window_points : int, default=40
        Number of points taken on each side of d1 for local fitting.
    poly_order : int, default=2
        Degree of local polynomial. 2 is usually a good choice.

    Returns
    -------
    derivative : float
        Estimated first derivative d(IDC)/d(delta) at d1.
    info : dict
        Extra information about the fit.
    """
    deltas = np.asarray(deltas, dtype=float)
    idc_values = np.asarray(idc_values, dtype=float)

    if deltas.ndim != 1 or idc_values.ndim != 1:
        raise ValueError("deltas and idc_values must be 1D arrays")
    if len(deltas) != len(idc_values):
        raise ValueError("deltas and idc_values must have the same length")
    if len(deltas) < 5:
        raise ValueError("Need at least 5 points")
    if not np.all(np.diff(deltas) > 0):
        raise ValueError("deltas must be strictly increasing")
    if not (deltas[0] <= d1 <= deltas[-1]):
        raise ValueError("d1 must lie within the delta range")

    # Remove NaNs/infs
    mask = np.isfinite(deltas) & np.isfinite(idc_values)
    deltas = deltas[mask]
    idc_values = idc_values[mask]

    if len(deltas) < poly_order + 2:
        raise ValueError("Not enough finite points after filtering")

    # Find nearest point to d1
    center_idx = np.argmin(np.abs(deltas - d1))

    # Local window
    # left = max(0, center_idx - half_window_points)
    # right = min(len(deltas), center_idx + half_window_points + 1)
    left = 0
    right = len(deltas)

    x = deltas[left:right]
    y = idc_values[left:right]

    if len(x) < poly_order + 2:
        raise ValueError("Window too small for requested polynomial order")

    # Center x around d1 for numerical stability
    x_shift = x - d1

    # Fit local polynomial y ≈ a0 + a1(x-d1) + a2(x-d1)^2 + ...
    coeffs = np.polyfit(x_shift, y, deg=poly_order)

    # Derivative at x=d1 corresponds to coefficient of first-order term
    # np.polyfit returns highest degree first
    derivative = np.polyder(np.poly1d(coeffs))(0.0)

    return derivative, {
        "d1": d1,
        "nearest_delta": deltas[center_idx],
        "window_left_delta": x[0],
        "window_right_delta": x[-1],
        "num_points_used": len(x),
        "poly_order": poly_order,
        "coeffs": coeffs,
    }

def plot_idc_over_delta(timestamps, d_min=30.0, d_max=5000000.0, t_start=None, duration=None, label_prefix=""):
    """
    Compute and plot IDC(delta) = Var(N_delta) / E[N_delta]
    for event timestamps over a range of delta values.

    Parameters
    ----------
    timestamps : array-like
        1D array of event timestamps.
    d_min : float
        Minimum delta value.
    d_max : float
        Maximum delta value.
    t_start : float or None
        Optional start time for analysis (defaults to min timestamp).
    duration : float or None
        Optional duration for analysis (defaults to max timestamp - min timestamp).

    Returns
    -------
    deltas_valid : np.ndarray
        Delta values used.
    idc_values : np.ndarray
        IDC for each delta.
    """
    min_windows = 100
    timestamps = np.asarray(timestamps, dtype=float)
    deltas = np.logspace(np.log10(d_min), np.log10(d_max), 3000)

    if timestamps.ndim != 1:
        raise ValueError("timestamps must be a 1D array")
    if deltas.ndim != 1:
        raise ValueError("deltas must be a 1D array")
    if len(timestamps) == 0:
        raise ValueError("timestamps is empty")
    if np.any(deltas <= 0):
        raise ValueError("all deltas must be positive")

    if t_start is not None and duration is not None:
        timestamps = timestamps[timestamps >= t_start]
        t_stop = t_start + duration
    
    else:
        duration = timestamps[-1] - timestamps[0]
        t_start = timestamps[0]
        t_stop = t_start + duration

    # keep only timestamps in the observation interval
    timestamps = timestamps[(timestamps >= t_start) & (timestamps < t_stop)]

    idc_values = []
    deltas_valid = []

    for delta in deltas:
        n_full = duration // delta

        if n_full < min_windows:
            # idc_values.append(np.nan)
            continue

        end = t_start + n_full * delta
        edges = t_start + np.arange(n_full + 1, dtype=np.int64) * delta

        ts_use = timestamps[timestamps < end]
        counts, _ = np.histogram(ts_use, bins=edges)

        mean_count = counts.mean()
        if mean_count == 0:
            # idc_values.append(np.nan)
            continue

        var_count = counts.var(ddof=1)
        idc_values.append(var_count / mean_count)
        deltas_valid.append(delta)

    idc_values = np.asarray(idc_values, dtype=float)
    deltas_valid = np.asarray(deltas_valid, dtype=float)

    fig, ax = plt.subplots(figsize=(30, 20))
    ax.plot(deltas_valid, idc_values, marker="o")
    ax.set_xlabel(r"$\Delta$(ns)")
    ax.set_ylabel(r"$IDC(\Delta)$")
    ax.set_title("Index of Dispersion for Counts vs Window Size")
    ax.grid(True, alpha=0.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=15))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=20))

    plt.tight_layout()
    if label_prefix == "":
        plt.savefig("idc_over_delta.png")
    else:
        plt.savefig(f"{label_prefix}idc_over_delta.png")
    plt.close()

    return deltas_valid, idc_values

def find_delta_for_closest_mean_packets_per_bin(t, target_mean=1.0):
    """
    Find the smallest bin width Δ such that the empirical mean number of packets per bin
    μ_hat(Δ) is closest to target_mean (default 1.0).

    Inputs:
      t: 1D array-like of packet arrival timestamps (seconds or any time unit).
      target_mean: threshold for mean packets per bin (e.g., 1.0).
    Returns:
        Delta_star (float) if found, else None.
    """
    t = np.asarray(t, dtype=float)
    t = t[np.isfinite(t)]
    t = np.unique(np.sort(t))
    if t.size < 2:
        raise ValueError("Need at least 2 timestamps.")

    t0 = t[0]
    tt = t - t0
    T = tt[-1]
    if T <= 0:
        raise ValueError("Timestamp span must be positive.")

    # Default candidate grid: from fine to coarse
    # smallest Δ: about T/max_bins, largest Δ: about T/min_bins
    d_min = 120.0 # 120 ns
    d_max = 500000.0 # 500 us
    deltas = np.logspace(np.log10(d_min), np.log10(d_max), 200)

    deltas = np.asarray(list(deltas), dtype=float)

    used_d, mu_list = [], []

    for Delta in deltas:
        if not np.isfinite(Delta) or Delta <= 0:
            continue
        nbins = int(np.floor(T / Delta)) + 1
        # bins over [0, nbins*Delta]
        edges = np.linspace(0, nbins * Delta, nbins + 1)
        counts, _ = np.histogram(tt, bins=edges)
        mu_hat = float(np.mean(counts))
        used_d.append(float(Delta))
        mu_list.append(mu_hat)

    used_d = np.asarray(used_d)
    mu_arr = np.asarray(mu_list)

    # sort by Δ increasing
    order = np.argsort(used_d)
    used_d, mu_arr = used_d[order], mu_arr[order]
    # pick Δ minimizing |mu_hat - target_mean|
    diffs = np.abs(mu_arr - target_mean)
    idx_min = np.argmin(diffs)
    Delta_star = float(used_d[idx_min])
    print("Delta:", Delta_star, "mean_count_at_Delta:", mu_arr[idx_min])

    return Delta_star, mu_arr[idx_min]

def find_delta_for_empty_prob(t, p0_max=0.10):
    """
    Find the smallest bin width Δ such that the empirical probability of an empty bin
    P_hat(X=0) is <= p0_max (default 10%).

    Inputs:
      t: 1D array-like of packet arrival timestamps (seconds or any time unit).
      p0_max: threshold for empty-bin probability (e.g., 0.10).
      deltas: optional iterable of candidate Δ values. If None, uses a log-spaced grid.

    Returns:
      Delta_star (float) if found, else None.
    """
    t = np.asarray(t, dtype=float)
    t = t[np.isfinite(t)]
    t = np.unique(np.sort(t))
    if t.size < 2:
        raise ValueError("Need at least 2 timestamps.")

    t0 = t[0]
    tt = t - t0
    T = tt[-1]
    if T <= 0:
        raise ValueError("Timestamp span must be positive.")

    # Default candidate grid: from fine to coarse
    d_min = 30.0 # 30 ns
    d_max = 1000000.0 # 1 ms
    min_bins = 100
    deltas = np.logspace(np.log10(d_min), np.log10(d_max), 3000)

    deltas = np.asarray(list(deltas), dtype=float)

    used_d, p0_list, mu_list = [], [], []

    for Delta in deltas:
        if not np.isfinite(Delta) or Delta <= 0:
            continue
        nbins = int(np.floor(T / Delta)) + 1
        if nbins < min_bins:
            continue
        # bins over [0, nbins*Delta]
        edges = np.linspace(0, nbins * Delta, nbins + 1)
        counts, _ = np.histogram(tt, bins=edges)

        p0_hat = float(np.mean(counts == 0))
        mu_hat = float(np.mean(counts))

        used_d.append(float(Delta))
        p0_list.append(p0_hat)
        mu_list.append(mu_hat)

    used_d = np.asarray(used_d)
    p0_arr = np.asarray(p0_list)
    mu_arr = np.asarray(mu_list)

    # sort by Δ increasing
    order = np.argsort(used_d)
    used_d, p0_arr, mu_arr = used_d[order], p0_arr[order], mu_arr[order]

    # pick smallest Δ meeting threshold
    ok = np.where(p0_arr <= p0_max)[0]
    Delta_star = float(used_d[ok[0]]) if ok.size > 0 else None
    
    # if no Δ meets threshold, return Δ that minimizes p0_arr
    if Delta_star is None:
        # print("Warning: no Δ found with P_hat(X=0) <= {:.4f}. Returning Δ minimizing P_hat.".format(p0_max))
        idx_min = np.argmin(p0_arr)
        Delta_star = float(used_d[idx_min])
        mu_star = float(mu_arr[idx_min])
        # print("Delta chosen:", Delta_star, "empty_prob_at_Delta:", p0_arr[idx_min], "mean_count_at_Delta:", mu_star)
        return Delta_star, mu_star
    else:
        # print("Delta:", Delta_star, "empty_prob_at_Delta:", p0_arr[ok[0]], "mean_count_at_Delta:", mu_arr[ok[0]])
        return Delta_star, mu_arr[ok[0]]
def compute_average_packet_size(file_path):
    # read all csv files in file_path ending with 'EndToEnd_packets.csv' and compute the average packet size
    sum_size = 0
    count = 0
    for file in glob.glob(file_path + '*EndToEnd_packets.csv'):
        df = pd.read_csv(file)
        if 'PayloadSize' in df.columns:
            sum_size += df['PayloadSize'].sum()
            count += df['PayloadSize'].count()
    average_packet_size = sum_size / count if count > 0 else 0
    return average_packet_size

def compute_bias_based_on_average_packet_size(sampling_results, average_packet_size, queue_names, linkRates, alternative_routes=[3, 6]):
    queue_names, _, linkRates = sort_queues_by_path(queue_names, [None, None, None, None], linkRates)
    
    for queue_name in queue_names:
        idx = queue_names.index(queue_name)
        sampling_results[queue_name+'NPkts'] = sampling_results[queue_name+'e2e_samples_queue_delay_mean'] * linkRates[idx] / (average_packet_size * 8)
        sampling_results[queue_name+'NBytes'] = sampling_results[queue_name+'NPkts'] * average_packet_size
        if idx == 0:
            continue
        # sampling_results[queue_name+'bias'] = sampling_results[queue_names[idx - 1]+'poisson_prob_non_empty'] * average_packet_size * 8 / linkRates[idx] * (1 / alternative_routes[idx - 1])
        if idx == 1:
            sampling_results[queue_name+'bias'] = sampling_results[queue_names[idx - 1]+'poisson_prob_non_empty'] * average_packet_size * 8 / linkRates[idx] * (1 / alternative_routes[idx - 1])
        if idx == 2:
            sampling_results[queue_name+'bias'] = sampling_results[queue_names[idx - 2]+'poisson_prob_non_empty'] * average_packet_size * 8 / linkRates[idx] * (1 / alternative_routes[idx - 1]) * (1 / alternative_routes[idx - 2])
            sampling_results[queue_name+'bias'] += sampling_results[queue_names[idx - 2]+'poisson_prob_non_empty'] * sampling_results[queue_names[idx - 1]+'poisson_prob_non_empty'] * average_packet_size * 8 / linkRates[idx] * (1 / alternative_routes[idx - 1]) * (1 - 1 / alternative_routes[idx - 2])
            sampling_results[queue_name+'bias'] += sampling_results[queue_names[idx - 1]+'poisson_prob_non_empty'] * (1 - sampling_results[queue_names[idx - 2]+'poisson_prob_non_empty']) * average_packet_size * 8 / linkRates[idx] * (1 / alternative_routes[idx - 1])
        sampling_results[queue_name+'e2e_vs_poisson_consistent_with_bias'] = int(abs(sampling_results[queue_name+'e2e_samples_queue_delay_mean'] - (sampling_results[queue_name+'poisson_samples_queue_delay_mean'] + sampling_results[queue_name+'bias'])) <= sampling_results[queue_name+'error_bound'])
    
    return sampling_results

def calculate_offline_delay_bias_DC(__ns3_path, rate, experiment, results_folder, steadyStart, steadyEnd, linkRates=[], linkDelays=[], 
                                    swtichDstREDQueueDiscMaxSize=[0], tsh=0.15, differentiationDelay=None, errorRate=None, load=None, 
                                    queue_names=[], flow_names=[], e2e_intervals=10000, sampling_factor=None):
    if differentiationDelay is not None and errorRate is not None:
        file_path = '{}/scratch/{}/{}/{}/D_{}/f_{}/{}/'.format(__ns3_path, results_folder, rate, load, differentiationDelay, errorRate, experiment)
    else:
        file_path = '{}/scratch/{}/{}/{}/{}/'.format(__ns3_path, results_folder, rate, load, experiment)

    times = np.array(np.cumsum(np.random.exponential(e2e_intervals, size=int((steadyEnd - steadyStart) // e2e_intervals))) + steadyStart, dtype=np.int64)
    # (_, queue_size_samples, _, queue_delay_samples_poisson_e2e), res = sample_total_queue_size(times, queue_names, file_path, linkDelays, linkRates, np.array(swtichDstREDQueueDiscMaxSize[1:], dtype=float) * tsh)
    res = {}
    (_, _, _, queue_delay_samples_poisson_e2e), res = sample_total_queue_size_non_combined(res, times, queue_names, file_path, linkDelays, linkRates, np.array(swtichDstREDQueueDiscMaxSize[1:], dtype=float) * tsh, path_observation=True, sampling_factor=sampling_factor)
    (_, _, _, _), res = sample_total_queue_size_non_combined(res, times, queue_names, file_path, linkDelays, linkRates, np.array(swtichDstREDQueueDiscMaxSize[1:], dtype=float) * tsh, path_observation=False)
    res = combine_sampling_results(res, queue_names)
    res = compute_bias_based_on_average_packet_size(res, compute_average_packet_size(file_path), queue_names, linkRates)

    res['sum_poisson_samples_queue_delay_mean'] = sum([res[queue_name+'poisson_samples_queue_delay_mean'] for queue_name in queue_names])
    res['e2e_poisson_samples_queue_delay_mean'] = np.mean(queue_delay_samples_poisson_e2e)
    res['e2e_poisson_samples_queue_delay_std'] = np.std(queue_delay_samples_poisson_e2e)
    res['e2e_vs_sum_error_bound'] = 1.96 * res['sum_poisson_samples_queue_delay_mean'] * np.max([res[queue_name+'poisson_samples_queue_delay_std'] / (np.sqrt(res[queue_name+'poisson_samples_queue_delay_count']) * res[queue_name+'poisson_samples_queue_delay_mean']) for queue_name in queue_names])
    res['e2e_vs_sum_error_bound'] += 1.96 * res['e2e_poisson_samples_queue_delay_std'] / np.sqrt(len(queue_delay_samples_poisson_e2e))
    res['e2e_vs_sum_consistent'] = int(abs(res['e2e_poisson_samples_queue_delay_mean'] - res['sum_poisson_samples_queue_delay_mean']) <= res['e2e_vs_sum_error_bound'])
    bias = sum([res[queue_name+'bias'] for queue_name in queue_names])
    res['e2e_vs_sum_consistent_with_bias'] = int(abs(res['e2e_poisson_samples_queue_delay_mean'] - (res['sum_poisson_samples_queue_delay_mean'] + bias)) <= res['e2e_vs_sum_error_bound'])
    return res
    
def calculate_offline_computations_DC(__ns3_path, rate, segment, experiment, results_folder, steadyStart, steadyEnd, projectColumn, nHosts, removeDrops=True, checkColumn="", linkRates=[], linkDelays=[], 
                                      swtichDstREDQueueDiscMaxSize=[0], stats=None, tsh=0.15, differentiationDelay=None, errorRate=None, load=None, passiveProbe=False, queue_names=[], flow_names=[],
                                      samples_paths_aggregated_statistics=None):
    if differentiationDelay is not None and errorRate is not None:
        file_paths = glob.glob('{}/scratch/{}/{}/{}/D_{}/f_{}/{}/*_{}.csv'.format(__ns3_path, results_folder, rate, load, differentiationDelay, errorRate, experiment, segment))
    else:
        file_paths = glob.glob('{}/scratch/{}/{}/{}/{}/*_{}.csv'.format(__ns3_path, results_folder, rate, load, experiment, segment))
    dfs = {}
    # if 'EndToEnd_packets' in segment:
    #     e2e_merged_df = pd.DataFrame()
    #     file_paths.append('R0R2H3')
    e2e_merged_df = pd.DataFrame()
    for file_path in file_paths:
        if file_path == 'R0R2H3':
            full_df = e2e_merged_df.copy()
            df_name = 'R0R2H3'
        else:
            df_name = file_path.split('/')[-1].split('_')[0]
            full_df = pd.read_csv(file_path)
        df_res = {}
        if 'EndToEnd_packets' in segment:
            # if "R0H" in df_name:
            #     e2e_merged_df = pd.concat([e2e_merged_df, full_df], ignore_index=True)
            #     continue
            if len(flow_names) != 0 and df_name not in flow_names:
                # print("Skipping flow not in flow_names:", df_name)
                continue
            df_res['first'] = {}
            df_res['last'] = {}
            df_res['workload'] = {}
            df_res['sampleSize'] = {}
            df_res['subSamplingError'] = {}
            df_res['successProbMean'] = {}
            df_res['sampleSize'] = {}
            df_res['totalPckts'] = {}
            df_res['RTT'] = {}
            df_res['InterArrivals'] = {}
            df_res['bias'] = {}
            df_res['ActiveFractionOfAll'] = {}
            df_res['ActiveFractionOfAll']['Packets'] = 0
            df_res['ActiveFractionOfAll']['Bytes'] = 0
            df_res['ActiveFractionOfTagged'] = {}
            df_res['ActiveFractionOfTagged']['Packets'] = 0
            df_res['ActiveFractionOfTagged']['Bytes'] = 0
            txDelay_to_firstQ = (1502 * 8 / linkRates[0])
            full_df = addRemoveTransmission_data(full_df, linkDelays, linkRates)
            
            if passiveProbe:
                # TODO: have not touches yet
                print("Passive probing not implemented yet")
                all_packets = len(full_df)
                all_bytes = full_df['PayloadSize'].sum()
                full_df = full_df[full_df['Tagged'] != "0"]
                full_df['BitsTag'] = full_df['Tagged'].apply(lambda x: x.split(':')[1:] if isinstance(x, str) else [])
                full_df = full_df.explode('BitsTag')
                full_df['BitsTag'] = full_df['BitsTag'].astype(int)

                df_res['ActiveFractionOfAll']['Packets'] = len(full_df[full_df['BitsTag'] == 0]) / all_packets
                df_res['ActiveFractionOfAll']['Bytes'] = full_df[full_df['BitsTag'] == 0]['PayloadSize'].sum() / all_bytes
                df_res['ActiveFractionOfTagged']['Packets'] = len(full_df[full_df['BitsTag'] == 0]) / len(full_df)
                df_res['ActiveFractionOfTagged']['Bytes'] = full_df[full_df['BitsTag'] == 0]['PayloadSize'].sum() / full_df['PayloadSize'].sum()
                # full_df['Delay'] = full_df['Delay'] + full_df['BitsTag'] / linksRates[1]
                # full_df['SentTime'] = full_df['SentTime'] + full_df['BitsTag'] / linksRates[0]
                full_df = full_df.sort_values(by=['SentTime'])
                # interarrival = np.diff(full_df['SentTime'].values)
                # print(full_df)
                # anderson_statistic, anderson_critical_values, _ = anderson(interarrival, 'expon')
                # if anderson_statistic <= anderson_critical_values[2]:
                #     print("Interarrival times are exponentially distributed.")
                # else:
                #     print("Interarrival times are *NOT* exponentially distributed.")
            else:
                full_df['BitsTag'] = 0
            # if errorRate is not None:
            #     full_df = addPacketsFromOtherPaths(full_df, errorRate, 1, 0)
            full_df = prune_data(full_df, projectColumn, steadyStart, steadyEnd)
            df_res = calc_RTT_per_path(full_df, df_res, checkColumn, linkDelays)
            # print(f"DC {df_name} len full df after pruning: {len(full_df)}")
            samplingMethod = "Orig"

            # plotting the queue size and ECN marking samples over time for the first queue in the path
            # times = np.cumsum(np.random.exponential(10, size=(steadyEnd - steadyStart) // 10)) + steadyStart
            # times, queue_size_samples, queue_ECN_samples = sample_total_queue_size(times, queue_names, ('/'.join(file_path.split('/')[:-1])) + '/', linkDelays, linkRates, np.array(swtichDstREDQueueDiscMaxSize, dtype=float) * tsh)
            # visualize_totalQ_and_ECN(times, queue_size_samples, queue_ECN_samples, ('/'.join(file_path.split('/')[:-1])) + '/')
            
            # plotting the cross correlation between the queue size and increment of arrivals for different windows
            # temp = full_df[full_df['Path'] == 0]
            # arrival_times = temp['SentTime'].values
            # times = np.cumsum(np.random.exponential(90, size=(steadyEnd - steadyStart) // 90)) + steadyStart
            # times, queue_size_samples, _ = sample_total_queue_size(times, queue_names, ('/'.join(file_path.split('/')[:-1])) + '/', linkDelays, linkRates, np.array(swtichDstREDQueueDiscMaxSize, dtype=float) * tsh)
            # res = []
            # for scale in [0.25, 0.5, 1, 2, 4, 16]:
            #     increments = sample_increments_of_arrivals(arrival_times, 8000 * scale, times)
            #     result = crosscorr_qsize_vs_arrival_increments(increments, queue_size_samples, times)
            #     result['T'] = 8000 * scale
            #     res.append(result)
            # visualize_crosscorr_Ts(res, ('/'.join(file_path.split('/')[:-1])) + '/')
            # df_res = calculate_offline_E2E_lossRates_DC(full_df, df_res, checkColumn, txDelay_to_firstQ, df_name, passiveProbe, samplingMethod, steadyStart, steadyEnd)
            df_res = calculate_offline_E2E_delays(full_df, removeDrops, checkColumn, txDelay_to_firstQ, df_res, 
                                                  '{}/scratch/{}/{}/{}/{}/'.format(__ns3_path, results_folder, rate, load, experiment), 
                                                  passiveProbe, samplingMethod, steadyStart, steadyEnd, samples_paths_aggregated_statistics[df_name], queue_names, linkDelays, linkRates, 
                                                  np.array(swtichDstREDQueueDiscMaxSize, dtype=float) * tsh)
            # df_res = calculate_offline_E2E_workload(full_df, df_res, steadyStart, steadyEnd)
            # df_res = calculate_offline_E2E_markingProb(full_df, df_res, checkColumn, txDelay_to_firstQ, swtichDstREDQueueDiscMaxSize, linkRates[0], __ns3_path, tsh, df_name, passiveProbe, samplingMethod, steadyStart, steadyEnd)
            # # for all values in df_res['bias'], multiply them by 1000 to convert to ms
            # # TODO: The bias term for multihop setting is different 
            # for metric in df_res['bias']:
            #     for path in df_res['bias'][metric]:
            #         df_res['bias'][metric][path] = abs(df_res['bias'][metric][path] * ((load * (nHosts - 1)) - (nHosts * rate)))
            #         if metric == 'delay':
            #             df_res['bias'][metric][path] = (df_res['bias'][metric][path] * 8) / linkRates[0]
        if 'Poisson' in segment:
            if len(queue_names) != 0 and df_name not in queue_names:
                continue
            packets_cfd = PacketCDF()
            packets_cfd.load_cdf_data('{}/scratch/ECNMC/DCWorkloads/packet_size_cdf_{}.csv'.format(__ns3_path, results_folder.split('/')[-1]))
            # packets_cfd.load_cdf_data('{}/scratch/ECNMC/Helpers/packet_size_cdf.csv'.format(__ns3_path, results_folder.split('/')[-1]))
            if df_name[0] == 'T' and df_name[2] == 'A':
                outgoingLinkRate = linkRates[1]
                switchMaxSize = swtichDstREDQueueDiscMaxSize[1]
                steadyStart = steadyStart + linkDelays[1]
                steadyEnd = steadyEnd + linkDelays[1]

            if df_name[0] == 'A' and df_name[2] == 'T':
                outgoingLinkRate = linkRates[2]
                switchMaxSize = swtichDstREDQueueDiscMaxSize[1]
                steadyStart = steadyStart + linkDelays[2] + linkDelays[1]
                steadyEnd = steadyEnd + linkDelays[2] + linkDelays[1]

            if df_name[0] == 'T' and df_name[2] == 'H':
                outgoingLinkRate = linkRates[3]
                switchMaxSize = swtichDstREDQueueDiscMaxSize[0]
                steadyStart = steadyStart + linkDelays[3] + linkDelays[2] + linkDelays[1]
                steadyEnd = steadyEnd + linkDelays[3] + linkDelays[2] + linkDelays[1]

            full_df = prune_data(full_df, projectColumn, steadyStart, steadyEnd)

            full_df['Delay'] = (full_df['TotalQueueSize'] * 8) / outgoingLinkRate
            df_res['DelayMean'] = full_df['Delay'].mean()
            df_res['DelayStd'] = full_df['Delay'].std()
            full_df['LastDelay'] = (full_df['LastTotalQueueSize'] * 8) / outgoingLinkRate
            df_res['LastDelayMean'] = full_df['LastDelay'].mean()
            df_res['LastDelayStd'] = full_df['LastDelay'].std()
            df_res['InterArrivals'] = full_df['Time'].diff().mean()
            df_res['first'] = full_df['Time'].iloc[0]
            df_res['last'] = full_df['Time'].iloc[-1]
            df_res['sampleSize'] = len(full_df)
            df_res['SuccessProbMean'] = 1 - full_df['DropProb'].mean()
            df_res['SuccessProbStd'] = full_df['DropProb'].std()
            df_res['LastSuccessProbMean'] = 1 - full_df['LastDropProb'].mean()
            df_res['LastSuccessProbStd'] = full_df['LastDropProb'].std()
            df_res['NonMarkingProbMean'] = 1 - full_df['MarkingProb'].mean()
            df_res['NonMarkingProbStd'] = full_df['MarkingProb'].std()
            df_res['LastNonMarkingProbMean'] = 1 - full_df['LastMarkingProb'].mean()
            df_res['LastNonMarkingProbStd'] = full_df['LastMarkingProb'].std()
            df_res["Occupancy"] = full_df['QueueSize'].mean() / switchMaxSize * 100
            # compute the avergae packet size from the CDF
            avgPacktSize = 1500 if "Nagle" in results_folder.split('/')[0] else packets_cfd.compute_average_packet_size_from_cdf()
            df_res["avgPacktSize"] = avgPacktSize
            df_res["PacktsInQueue"] = full_df['TotalQueueSize'].mean() / avgPacktSize
            df_res["BytesInQueue"] = full_df['TotalQueueSize'].mean()
            df_res["EmptyFrac"] = len(full_df[full_df['TotalQueueSize'] == 0]) / len(full_df) * 100
            df_res["GT1PktsFrac"] = len(full_df[full_df['TotalQueueSize'] > avgPacktSize]) / len(full_df) * 100
            # print(f"DC {df_name} Avg Delay : {df_res['DelayMean']} ns, delay Std: {df_res['DelayStd']} ns, samples: {df_res['sampleSize']}")
            # print(f"DC {df_name} Avg Success Prob: {df_res['SuccessProbMean']}, Success Prob Std: {df_res['SuccessProbStd']}, samples: {df_res['sampleSize']}")
            # print(f"DC {df_name} Avg Non-Marking Prob: {df_res['NonMarkingProbMean']}, Non-Marking Prob Std: {df_res['NonMarkingProbStd']}, samples: {df_res['sampleSize']}")
        if df_name == 'R0R2H3':
            df_name = 'R0H0R2H3'
        dfs[df_name] = df_res
    return dfs

def calculate_offline_computations(__ns3_path, rate, segment, experiment, results_folder, steadyStart, steadyEnd, projectColumn, nHosts, removeDrops=True, checkColumn="", linksRates=[], linkDelays=[], swtichDstREDQueueDiscMaxSize=0, stats=None, tsh=0.15, differentiationDelay=None, errorRate=None, load=None, passiveProbe=False, flow_names=['AD0']):
    if differentiationDelay == 0.0 and errorRate is not None:
        file_paths = glob.glob('{}/scratch/{}/{}/{}/D_{}/f_{}/{}/*_{}.csv'.format(__ns3_path, results_folder, rate, load, differentiationDelay, errorRate, experiment, segment))
    else:
        file_paths = glob.glob('{}/scratch/{}/{}/{}/{}/*_{}.csv'.format(__ns3_path, results_folder, rate, load, experiment, segment))
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
            if len(flow_names) != 0 and df_name not in flow_names:
                print("Skipping flow not in flow_names:", df_name)
                continue
            df_res['first'] = {}
            df_res['last'] = {}
            df_res['workload'] = {}
            df_res['sampleSize'] = {}
            df_res['successProbMean'] = {}
            df_res['sampleSize'] = {}
            df_res['totalPckts'] = {}
            df_res['RTT'] = {}
            df_res['InterArrivals'] = {}
            df_res['bias'] = {}
            df_res['ActiveFractionOfAll'] = {}
            df_res['ActiveFractionOfAll']['Packets'] = 0
            df_res['ActiveFractionOfAll']['Bytes'] = 0
            df_res['ActiveFractionOfTagged'] = {}
            df_res['ActiveFractionOfTagged']['Packets'] = 0
            df_res['ActiveFractionOfTagged']['Bytes'] = 0
            txDelay = (1502 * 8 / linksRates[0])
            full_df = addRemoveTransmission_data(full_df, linkDelays, linksRates)
            
            if passiveProbe:
                all_packets = len(full_df)
                all_bytes = full_df['PayloadSize'].sum()
                full_df = full_df[full_df['Tagged'] != "0"]
                full_df['BitsTag'] = full_df['Tagged'].apply(lambda x: x.split(':')[1:] if isinstance(x, str) else [])
                full_df = full_df.explode('BitsTag')
                full_df['BitsTag'] = full_df['BitsTag'].astype(int)

                df_res['ActiveFractionOfAll']['Packets'] = len(full_df[full_df['BitsTag'] == 0]) / all_packets
                df_res['ActiveFractionOfAll']['Bytes'] = full_df[full_df['BitsTag'] == 0]['PayloadSize'].sum() / all_bytes
                df_res['ActiveFractionOfTagged']['Packets'] = len(full_df[full_df['BitsTag'] == 0]) / len(full_df)
                df_res['ActiveFractionOfTagged']['Bytes'] = full_df[full_df['BitsTag'] == 0]['PayloadSize'].sum() / full_df['PayloadSize'].sum()
                # full_df['Delay'] = full_df['Delay'] + full_df['BitsTag'] / linksRates[1]
                # full_df['SentTime'] = full_df['SentTime'] + full_df['BitsTag'] / linksRates[0]
                full_df = full_df.sort_values(by=['SentTime'])
                # interarrival = np.diff(full_df['SentTime'].values)
                # print(full_df)
                # anderson_statistic, anderson_critical_values, _ = anderson(interarrival, 'expon')
                # if anderson_statistic <= anderson_critical_values[2]:
                #     print("Interarrival times are exponentially distributed.")
                # else:
                #     print("Interarrival times are *NOT* exponentially distributed.")
            else:
                full_df['BitsTag'] = 0
                
            if (differentiationDelay is not None) and (differentiationDelay != 0.0):
                full_df = addExtraDelay(full_df, differentiationDelay, errorRate)
            full_df = prune_data(full_df, projectColumn, steadyStart, steadyEnd)
            df_res = calc_RTT_per_path(full_df, df_res, checkColumn, linkDelays)
            # for the reconstructed signal:
            # reconstructedSignal_df = reconstructSignal(full_df, linksRates, file_path.replace(f'{df_name}_EndToEnd_packets.csv', ''))
            # df_res = calculate_reconstructedSignal_delays(reconstructedSignal_df, df_res, linksRates[1])
            # avgPacktSize = 1502
            # PacktsInQueue = (full_df["Delay"].mean() * linksRates[1]) / (avgPacktSize * 8)
            # samplingMethod = "Orig" if PacktsInQueue > 1 else "DA"
            # interarrival_99 = np.percentile(np.diff(full_df['SentTime'].values), 99)
            # if interarrival_99 < txDelay * 1.05 and samplingMethod == "DA":
            #     samplingMethod = "Orig"
            samplingMethod = "Orig"

            df_res = calculate_offline_E2E_lossRates(__ns3_path, full_df, df_res, checkColumn, txDelay, linksRates[1], swtichDstREDQueueDiscMaxSize, df_name, passiveProbe, samplingMethod)
            df_res = calculate_offline_E2E_delays(full_df, removeDrops, checkColumn, txDelay, df_res, df_name, passiveProbe, samplingMethod)
            df_res = calculate_offline_E2E_workload(full_df, df_res, steadyStart, steadyEnd)
            df_res = calculate_offline_E2E_markingProb(full_df, df_res, checkColumn, txDelay, swtichDstREDQueueDiscMaxSize, linksRates[1], __ns3_path, tsh, df_name, passiveProbe, samplingMethod)
            # for all values in df_res['bias'], multiply them by 1000 to convert to ms
            for metric in df_res['bias']:
                for path in df_res['bias'][metric]:
                    df_res['bias'][metric][path] = abs(df_res['bias'][metric][path] * ((load * (nHosts - 1)) - (nHosts * rate)))
                    if metric == 'delay':
                        df_res['bias'][metric][path] = (df_res['bias'][metric][path] * 8) / linksRates[1]
        if 'Poisson' in segment:
            packets_cfd = PacketCDF()
            packets_cfd.load_cdf_data('{}/scratch/ECNMC/DCWorkloads/packet_size_cdf_{}.csv'.format(__ns3_path, results_folder.split('/')[-1]))
            full_df = prune_data(full_df, projectColumn, steadyStart, steadyEnd)
            # apply a thinning function to the data. The thinning function is a bernoulli process with a probability of 0.8 to keep the data
            # full_df = full_df.sample(frac=0.01, random_state=1)
            # full_df = full_df.sort_values(by=[projectColumn], ignore_index=True)
            # full_df['MarkingProb'] = full_df.apply(lambda x: packets_cfd.calculate_probability_greater_than(swtichDstREDQueueDiscMaxSize * 0.15 - x['QueueSize']) if x['MarkingProb'] != 1.0 else 1.0, axis=1)
            # df_res = calculate_offline_switch_congestionEstimation(full_df, df_res)
            full_df['Delay'] = (full_df['TotalQueueSize'] * 8) / linksRates[0]
            df_res['DelayMean'] = full_df['Delay'].mean()
            # print("Delay Mean:", df_res['DelayMean'])
            df_res['DelayStd'] = full_df['Delay'].std()
            full_df['LastDelay'] = (full_df['LastTotalQueueSize'] * 8) / linksRates[0]
            df_res['LastDelayMean'] = full_df['LastDelay'].mean()
            df_res['LastDelayStd'] = full_df['LastDelay'].std()
            df_res['InterArrivals'] = full_df['Time'].diff().mean()
            # df_res['DelayMeanDisc'] = full_df['QueuingDelay'].mean()
            # df_res['DelayStdDisc'] = full_df['QueuingDelay'].std()
            df_res['first'] = full_df['Time'].iloc[0]
            df_res['last'] = full_df['Time'].iloc[-1]
            df_res['sampleSize'] = len(full_df)
            df_res['SuccessProbMean'] = 1 - full_df['DropProb'].mean()
            df_res['SuccessProbStd'] = full_df['DropProb'].std()
            df_res['LastSuccessProbMean'] = 1 - full_df['LastDropProb'].mean()
            df_res['LastSuccessProbStd'] = full_df['LastDropProb'].std()
            df_res['NonMarkingProbMean'] = 1 - full_df['MarkingProb'].mean()
            df_res['NonMarkingProbStd'] = full_df['MarkingProb'].std()
            df_res['LastNonMarkingProbMean'] = 1 - full_df['LastMarkingProb'].mean()
            df_res['LastNonMarkingProbStd'] = full_df['LastMarkingProb'].std()
            df_res["Occupancy"] = full_df['QueueSize'].mean() / swtichDstREDQueueDiscMaxSize * 100
            # compute the avergae packet size from the CDF
            avgPacktSize = 1500 if "Nagle" in results_folder.split('/')[0] else packets_cfd.compute_average_packet_size_from_cdf()
            df_res["PacktsInQueue"] = full_df['TotalQueueSize'].mean() / avgPacktSize
            df_res["BytesInQueue"] = full_df['TotalQueueSize'].mean()
            df_res["EmptyFrac"] = len(full_df[full_df['TotalQueueSize'] == 0]) / len(full_df) * 100
            df_res["GT1PktsFrac"] = len(full_df[full_df['TotalQueueSize'] > avgPacktSize]) / len(full_df) * 100
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

def read_data_flowIndicator(__ns3_path, rate, results_folder, differentiationDelay=None, errorRate=None, load=None):
    flows_name = []
    file_paths = []
    i = 0
    if differentiationDelay is not None and errorRate is not None:
        while len(file_paths) == 0:
            file_paths = glob.glob('{}/scratch/{}/{}/{}/D_{}/f_{}/{}/*_EndToEnd_packets.csv'.format(__ns3_path, results_folder, rate, load, differentiationDelay, errorRate, i))
            i += 1
    else:
        while len(file_paths) == 0:
            file_paths = glob.glob('{}/scratch/{}/{}/{}/{}/*_EndToEnd_packets.csv'.format(__ns3_path, results_folder, rate, load, i))
            i += 1
    for file_path in file_paths:
        flows_name.append(file_path.split('/')[-1].split('_')[0])
    return flows_name

def read_queues_indicators(__ns3_path, rate, results_folder, differentiationDelay=None, errorRate=None, load=None):
    flows_name = []
    file_paths = []
    i = 0
    if differentiationDelay is not None and errorRate is not None:
        while len(file_paths) == 0:
            file_paths = glob.glob('{}/scratch/{}/{}/{}/D_{}/f_{}/{}/*_PoissonSampler_events.csv'.format(__ns3_path, results_folder, rate, load, differentiationDelay, errorRate, i))
            i += 1
    else:
        while len(file_paths) == 0:
            file_paths = glob.glob('{}/scratch/{}/{}/{}/{}/*_PoissonSampler_events.csv'.format(__ns3_path, results_folder, rate, load, i))
            i += 1
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

def calc_epsilon(confidenceValue, segement_statistics, last=""):
    return (confidenceValue * segement_statistics[last + 'DelayStd']) / (np.sqrt(segement_statistics['sampleSize']) * segement_statistics[last + 'DelayMean'])

def calc_epsilon_loss_with_bias(confidenceValue, segement_statistics, bias):
    return (calc_epsilon_loss(confidenceValue, segement_statistics) + (bias / segement_statistics['successProbMean']))

def calc_epsilon_loss(confidenceValue, segement_statistics, last=""):
    return (confidenceValue * segement_statistics[last + 'SuccessProbStd']) / (np.sqrt(segement_statistics['sampleSize']) * segement_statistics[last + 'SuccessProbMean'])

def calc_epsilon_last_marking_with_bias(confidenceValue, segement_statistics, bias):
    return (calc_epsilon_last_marking(confidenceValue, segement_statistics) + (bias / segement_statistics['lastNonMarkingProbMean']))

def calc_epsilon_marking_with_bias(confidenceValue, segement_statistics, bias):
    return (calc_epsilon_marking(confidenceValue, segement_statistics) + (bias / segement_statistics['nonMarkingProbMean']))

def calc_epsilon_marking(confidenceValue, segement_statistics, last=""):
    return (confidenceValue * segement_statistics[last + 'NonMarkingProbStd']) / (np.sqrt(segement_statistics['sampleSize']) * segement_statistics[last + 'NonMarkingProbMean'])

def calc_epsilon_loss_2(confidenceValue, segement_statistics):
    return (confidenceValue * segement_statistics['successProbStd_2']) / (np.sqrt(segement_statistics['sampleSize']) * segement_statistics['successProbMean_2'])

def calc_error(confidenceValue, segement_statistics):
    return (confidenceValue * segement_statistics['DelayStd']) / np.sqrt(segement_statistics['sampleSize'])

def calc_min_e2e_samples(confidenceValue, maxError, samples_paths_aggregated_statistics):
    if samples_paths_aggregated_statistics['MaxEpsilonDelay'] >= maxError:
        print("Warning: MaxEpsilonDelay is greater than or equal to maxError. Cannot achieve the desired confidence level with the current data.")
        return None
    return int(((confidenceValue * samples_paths_aggregated_statistics['e2eDelayStd']) / ((maxError - samples_paths_aggregated_statistics['MaxEpsilonDelay']) * samples_paths_aggregated_statistics['DelayMean'])) ** 2)

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