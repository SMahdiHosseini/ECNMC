import psutil, os
import multiprocessing
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from enum import Enum
import seaborn as sns
import numpy as np
from scipy.stats import anderson
from scipy.stats import f_oneway, kruskal
from scipy.stats import bernoulli, ks_2samp
from scipy.stats import wasserstein_distance
from math import factorial, exp
import csv
from collections import defaultdict, OrderedDict
from colorama import Fore, Back, Style
import pprint
from functools import lru_cache
from pathlib import Path
import re
import contextlib
import io
import time

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

def calculate_offline_E2E_lossRates_DC(full_df, df_res, checkColumn, txDelay, df_name, passiveProbe, samplingMethod, steadyStart, steadyEnd, 
                                 samples_paths_aggregated_statistics=None, queue_names=None, linkDelays=None, linkRates=None, queue_size_trshs=None):
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
            minimum_samples = 0 if samples_paths_aggregated_statistics is None else samples_paths_aggregated_statistics.get(path, {}).get('MinimumE2ESampleSizeSuccessProb', 0)
            samples_times = find_samples_path_new(
                time,
                txDelay,
                df_res['RTT'][path],
                df_name,
                samplingMethod,
                steadyStart,
                steadyEnd,
                steps=1,
                MinimumNumberOfSamples=minimum_samples,
            )
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
            samples_times, _ = find_samples_path(time, 0)
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
    maxD = (
        (steadyEnd - steadyStart) / MinimumNumberOfSamples
        if MinimumNumberOfSamples > 0
        else steadyEnd - steadyStart
    )
    
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
    if MinimumNumberOfSamples > 0 and len(time) < MinimumNumberOfSamples:
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
    if MinimumNumberOfSamples > 0 and len(samples) < (MinimumNumberOfSamples * 0.95):
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
    # #     # t_sel, subSamplingError = find_samples_path(interval_times, MinimumNumberOfSamples=0)
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
    #     intervalSamples, subSamplingError = find_samples_path(interval_times, MinimumNumberOfSamples=0)
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
    #         intervalSamples, _ = find_samples_path(interval_times, MinimumNumberOfSamples=0)
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

def find_samples_path(time, MinimumNumberOfSamples=0, window=None):
    subSamplingError = SubSamplingError.NoError

    time = np.asarray(time)
    if len(time) <= 1:
        if MinimumNumberOfSamples > len(time):
            subSamplingError = SubSamplingError.NotEnoughPackets + "+" + subSamplingError.value
        return [], subSamplingError

    start_time = time[0]
    end_time = time[-1]
    duration = end_time - start_time

    try:
        minimum_number_of_samples = int(np.ceil(float(MinimumNumberOfSamples)))
    except (TypeError, ValueError):
        minimum_number_of_samples = 0
    minimum_number_of_samples = max(0, minimum_number_of_samples)

    if minimum_number_of_samples > 0 and minimum_number_of_samples > len(time):
        subSamplingError = SubSamplingError.NotEnoughPackets + "+" + subSamplingError.value
        return np.array([], dtype=time.dtype), subSamplingError
    if duration <= 0:
        return [], subSamplingError
    if window is None:
        try:
            window, _ = find_delta_for_empty_prob(time, p0_max=0.01)
        except ValueError:
            subSamplingError = SubSamplingError.NotEnoughSamples + "+" + subSamplingError.value
            return np.array([], dtype=time.dtype), subSamplingError

    number_of_windows = max(int(np.floor(duration / window)) + 1, 1)
    bin_ids = np.floor((time - start_time) / window).astype(int)
    bin_ids = np.clip(bin_ids, 0, number_of_windows - 1)
    counts = np.bincount(bin_ids, minlength=number_of_windows)

    window_sample_count = int(np.count_nonzero(counts))
    if window_sample_count <= 1:
        print("Failed to find enough non-empty windows for sampling.")
        subSamplingError = SubSamplingError.NotEnoughSamples + "+" + subSamplingError.value
        return np.array([], dtype=time.dtype), subSamplingError

    if minimum_number_of_samples > 0 and minimum_number_of_samples > window_sample_count:
        print("Warning: Not enough windows after one-per-window sampling! Got {}, expected {}".format(
            window_sample_count, minimum_number_of_samples
        ))
        subSamplingError = SubSamplingError.NotEnoughSamples + "+" + subSamplingError.value
        return np.array([], dtype=time.dtype), subSamplingError

    brnvals = []
    if minimum_number_of_samples > 0:
        brnvals.append(min(1.0, max(0.0, minimum_number_of_samples / window_sample_count)))
    else:
        brnvals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    sorted_indices = np.argsort(bin_ids, kind='stable')
    split_points = np.flatnonzero(np.diff(bin_ids[sorted_indices])) + 1
    window_starts = np.concatenate(([0], split_points))
    window_sizes = np.diff(np.concatenate((window_starts, [len(sorted_indices)])))

    max_sample_size = 0
    max_sample_size_times = np.array([], dtype=time.dtype)
    for brnval in brnvals:
        tries = 20
        while tries > 0:
            # Vectorized equivalent of "pick one uniformly random packet per non-empty
            # window": one np.random.randint call over all windows at once, instead of
            # looping np.random.choice per window (the dominant cost of this function,
            # called up to 20x per (k, run) pair).
            offsets = np.random.randint(0, window_sizes)
            selected_indices = sorted_indices[window_starts + offsets]
            selected_indices.sort()
            selected_times = time[selected_indices]

            if brnval < 1.0:
                keep_mask = bernoulli.rvs(brnval, size=len(selected_times)).astype(bool)
                final_times = selected_times[keep_mask]
            else:
                final_times = selected_times

            if minimum_number_of_samples > 0 and len(final_times) < minimum_number_of_samples:
                tries -= 1
                continue
            if len(final_times) <= 1:
                tries -= 1
                continue
            
            # anderson_statistic, anderson_critical_values, _ = anderson(np.diff(final_times), 'expon')
            # if anderson_statistic <= anderson_critical_values[4]:
            anderson_res = anderson(np.diff(final_times), 'expon', method='interpolate')
            if anderson_res.pvalue > 0.05:
                if len(final_times) > max_sample_size:
                    max_sample_size = len(final_times)
                    max_sample_size_times = final_times
                    break
            tries -= 1

    if max_sample_size:
        return max_sample_size_times, subSamplingError
    print("Failed to find exponentially distributed interarrival times after 20 tries. Window: {}, Bernoulli probability: {}".format(
        window, brnvals
    ))
    subSamplingError = SubSamplingError.NotPoisson + "+" + subSamplingError.value
    return np.array([], dtype=time.dtype), subSamplingError


def _winsorized_mean_1d(values, upper_trim_frac):
    """Mean of `values` after capping (not discarding) its largest
    `upper_trim_frac` fraction at the value just below the cut, so a single
    extreme point pulls the mean toward, rather than past, the bulk of the
    data. Only the upper tail is capped (not a symmetric trim): the only
    contamination this guards against is an occasional very long gap, never
    an implausibly short one, and one-sided Winsorizing keeps every ordinary
    (non-outlier) observation's full value in the average."""
    values = np.sort(values)
    m = len(values)
    k = max(0, int(round(m * upper_trim_frac)))
    if k > 0 and m > k:
        values = values.copy()
        values[m - k:] = values[m - k - 1]
    return values.mean()


def _causal_winsorized_mean(gaps, window, upper_trim_frac):
    """Winsorized mean of the trailing up-to-`window` values of `gaps` ending
    at (and including) each position -- position i uses
    gaps[max(0, i-window+1):i+1], the same "gap ending at this candidate"
    convention distanceAwareSampling uses for its single-gap estimate, just
    averaged (Winsorized) over the last `window` such gaps instead of only
    the last one. Vectorized via sliding_window_view once i >= window - 1;
    the O(window) ramp-up positions before that, where fewer than `window`
    gaps are available, are handled one at a time since there are at most
    `window` of them."""
    n = len(gaps)
    out = np.empty(n, dtype=float)
    if n == 0:
        return out
    ramp = min(window - 1, n)
    for i in range(ramp):
        out[i] = _winsorized_mean_1d(gaps[max(0, i - window + 1):i + 1], upper_trim_frac)
    if n >= window:
        windows = np.lib.stride_tricks.sliding_window_view(gaps, window)
        sorted_windows = np.sort(windows, axis=1)
        k = max(0, int(round(window * upper_trim_frac)))
        if k > 0:
            sorted_windows[:, window - k:] = sorted_windows[:, window - k - 1:window - k]
        out[window - 1:n] = sorted_windows.mean(axis=1)
    return out


def intensity_thinning(time, rate, window=16, upper_trim_frac=0.2):
    """Thin a captured packet-timestamp stream toward a target Poisson rate
    `rate`, using a causal, outlier-robust estimate of the local arrival
    intensity rather than the single immediately-preceding gap.

    For each candidate packet, this estimates the local mean gap as the
    one-sided Winsorized mean of the `window` most recent raw gaps ending at
    that packet's own gap (see _causal_winsorized_mean), and retains the
    packet independently with probability
    1 - exp(-rate * local_gap_estimate). This is the classical
    construction for thinning a point process with a (here, data-estimated)
    predictable/conditional intensity down to a homogeneous Poisson process
    of rate `rate` -- the same idea underlying the Lewis-Shedler thinning
    algorithm and the time-rescaling theorem used for point-process
    goodness-of-fit; distanceAwareSampling above is the special case
    `window=1` (no smoothing) with this same functional form.

    Two things this deliberately does NOT do, both changed after review:

    - It does not use the sample median rescaled by a constant to estimate
      the local mean gap. median/mean equals ln(2) only asymptotically for
      exponential gaps, is measurably off already at small window (the exact
      ratio for k iid Exp(1) is 1.202 at k=4, 1.047 at k=16, 1.011 at k=64 --
      not 1 as a fixed ln(2) correction assumes), and is a different, unknown
      ratio entirely once the local gaps are not exponential -- which is
      precisely the regime this function exists to handle: if the raw gaps
      already were exponential, no thinning would be needed. The Winsorized
      mean above needs no such correction: an (un-Winsorized) sample mean is
      unbiased for the population mean under any distribution, and
      Winsorizing only the upper tail bounds the damage a single long idle
      gap does to that estimate without asserting anything about the shape
      of the local gaps.
    - It does not clip retention_prob with min(1, ...). A ratio-form
      probability (rate * local_gap_estimate) exceeds 1 whenever the local
      gap estimate exceeds 1/rate -- i.e. in every locally-sparse stretch --
      and clipping then forces a deterministic keep there; at high overall
      retention that is most positions, and the method degenerates to
      "keep nearly everything, then test whether the raw stream already
      looks Poisson" rather than actually thinning. 1 - exp(-rate * g) is
      already in [0, 1) for every g >= 0 with no clip, and is never larger
      than the ratio form (1 - exp(-x) <= x for x >= 0), so it thins at
      least as aggressively everywhere while remaining a valid probability
      by construction.

    The very first raw packet is only used to anchor the first gap and is
    never itself a retention candidate (same convention as
    distanceAwareSampling/poissonLikeSampling).
    """
    time = np.asarray(time, dtype=float)
    if len(time) < 2 or rate <= 0:
        return np.array([], dtype=float)

    gaps = np.diff(time)
    local_gap_est = _causal_winsorized_mean(gaps, window, upper_trim_frac)

    retention_prob = -np.expm1(-rate * local_gap_est)
    selected_mask = bernoulli.rvs(retention_prob).astype(bool)
    return time[1:][selected_mask]


def find_samples_path_intensity(
    time,
    MinimumNumberOfSamples=0,
    windows=(4, 16, 64),
    num_candidates=10,
    tries_per_candidate=3,
    steadyStart=None,
    steadyEnd=None,
    confirm_lags=None,
):
    # TODO: Needs a modification and verification and be added to the list of sampling methods
    """Poisson-adaptive subsampling of a captured e2e packet stream: retain as
    many packets as possible while still passing a genuine Poisson-process
    validation, so that the resulting sample mean can be trusted as a PASTA
    (Poisson-Arrivals-See-Time-Averages) estimate of the path's time-average
    delay.

    This performs a grid search over (smoothing window, target rate) and
    keeps the single best-validated candidate seen anywhere in the grid --
    "best" meaning largest retained sample count among those that pass both
    of the following on intensity_thinning's output (see intensity_thinning
    for the thinning rule itself):

      1. Marginal exponentiality of the retained inter-arrival times
         (Anderson-Darling), the same test find_samples_path already uses.
      2. The multi-lag independence test in chi_squared_test. This is the
         part find_samples_path/distanceAwareSampling/poissonLikeSampling
         above do not check: Anderson-Darling alone can pass a stream whose
         gaps are individually exponential-looking but serially dependent
         (e.g. RTT-periodic ACK clocking, or a burst boundary re-appearing
         at a fixed lag). Undetected serial dependence silently shrinks the
         effective sample size below N, which is exactly the count the
         downstream confidence bound (eta * sigma / sqrt(N)) assumes is
         independent. Passing this test is what lets N be trusted, not just
         the marginal shape of the gaps.

    Both the target rate and the smoothing window are searched, not just the
    rate for a fixed window: empirically (see the accompanying discussion),
    the right smoothing window depends on the traffic's burst scale relative
    to the window -- e.g. a window close to the size of a typical burst can
    let a stale pre-burst gap contaminate the local-intensity estimate for
    much of the following burst -- and no single fixed window dominates
    across traffic patterns. This mirrors the paper's own windowed method,
    which searches jointly over its window and Bernoulli-keep probability
    rather than fixing one of them; distanceAwareSampling's single fixed-gap
    estimate is the special case `windows=(1,)`.

    The full scan is used (highest-rate-first within each window, tracking a
    running best) rather than an adaptive bisection on the rate: because
    validation is a noisy Bernoulli-driven outcome, a single unlucky draw at
    an otherwise-good rate can look like a failure, and a bisection that
    permanently narrows its bracket on one such draw gets trapped far below
    the true achievable rate and never revisits it. The expensive multi-lag
    independence test only runs on a candidate that already passed the cheap
    Anderson-Darling test and could improve on the current best count, so
    its cost scales with the number of genuine improvements the search
    makes, not with num_candidates * tries_per_candidate * len(windows).

    Returns (samples, subSamplingError) with the same contract as
    find_samples_path: samples is an empty array and subSamplingError names
    the failure reason when no (window, rate) candidate could be validated,
    or when the best validated candidate has fewer than
    MinimumNumberOfSamples packets.
    """
    subSamplingError = SubSamplingError.NoError
    time = np.asarray(time, dtype=float)
    time = np.sort(time)

    try:
        minimum_number_of_samples = max(0, int(np.ceil(float(MinimumNumberOfSamples))))
    except (TypeError, ValueError):
        minimum_number_of_samples = 0

    if len(time) <= 1:
        if minimum_number_of_samples > len(time):
            subSamplingError = SubSamplingError.NotEnoughPackets + "+" + subSamplingError.value
        return np.array([], dtype=float), subSamplingError

    if minimum_number_of_samples > 0 and minimum_number_of_samples > len(time):
        subSamplingError = SubSamplingError.NotEnoughPackets + "+" + subSamplingError.value
        return np.array([], dtype=float), subSamplingError

    duration = time[-1] - time[0]
    if duration <= 0:
        return np.array([], dtype=float), subSamplingError

    if steadyStart is None:
        steadyStart = time[0]
    if steadyEnd is None:
        steadyEnd = time[-1]

    raw_rate = (len(time) - 1) / duration
    floor_rate = (
        minimum_number_of_samples / duration if minimum_number_of_samples > 0 else raw_rate * 1e-3
    )
    # low = max(floor_rate, raw_rate * 1e-4)
    low = floor_rate
    high = raw_rate
    if low >= high:
        low = high * 1e-3

    rates = np.geomspace(high, low, num=num_candidates)
    if isinstance(windows, (int, np.integer)):
        windows = (windows,)

    best_count = 0
    best_samples = np.array([], dtype=float)

    for window in windows:
        for rate in rates:
            for _ in range(tries_per_candidate):
                trial = intensity_thinning(time, rate, window=window)
                if len(trial) <= max(1, best_count):
                    continue
                ad_res = anderson(np.diff(trial), 'expon', method='interpolate')
                if ad_res.pvalue <= 0.05:
                    continue
                lags, res, _ = chi_squared_test(trial, steadyStart, steadyEnd, lags=confirm_lags)
                if len(lags) == 0:
                    continue
                upper_band = 0.05 + 1.96 * np.sqrt(0.95 * 0.05) / np.sqrt(len(lags))
                if (sum(res) / len(lags)) < upper_band:
                    best_count, best_samples = len(trial), trial
                    break  # this (window, rate) already improved on the best; no need for another draw

    if best_count == 0:
        subSamplingError = SubSamplingError.NotPoisson + "+" + subSamplingError.value
        return np.array([], dtype=float), subSamplingError

    if minimum_number_of_samples > 0 and best_count < minimum_number_of_samples:
        subSamplingError = SubSamplingError.NotEnoughSamples + "+" + subSamplingError.value
        return np.array([], dtype=float), subSamplingError

    return np.sort(best_samples), subSamplingError


# Step by which the growing-window subsampling methods grow the monitoring window they draw
# their subsample from: they first try [steadyStart, steadyStart + 5ms], then
# [steadyStart, steadyStart + 10ms], and so on. 5 ms is a few RTTs at this DC's scale, so
# each step adds a meaningful amount of new traffic without making the first (cheapest,
# most interesting) candidate window so long that the early-stopping has nothing to win.
GROWING_WINDOW_STEP_NS = 5e6

# Floor on the switch-side Poisson observation count of a candidate window (see
# windowed_poisson_agg_stats): the observation count scales with the window's length to
# keep the probing RATE fixed, and a very short window would otherwise be handed a
# handful of observations whose mean/std are meaningless.
MIN_WINDOWED_POISSON_OBSERVATIONS = 100


def _trim_to_minimum_samples(samples, minimum_number_of_samples):
    """Cut an already-validated subsample down to exactly `minimum_number_of_samples`
    points by dropping its tail -- i.e. stop monitoring at the n-th retained sample rather
    than at the end of the candidate window, which is what "the minimum required number of
    samples, and not more than that" means once a window that can deliver them is found.

    Dropping the *tail* (rather than a random subset) is what keeps the result a valid
    Poisson sample: the first n points of a Poisson process are still a Poisson process
    observed up to its n-th arrival, and that stopping rule depends only on the arrival
    instants, never on the queue state those instants sample -- so PASTA still applies and
    the retained gaps are still i.i.d. exponential. A random subset, by contrast, would
    have non-exponential (Erlang-mixture) gaps.

    The trimmed set is nonetheless re-run through the same Anderson-Darling exponentiality
    test the base sampler validated the untrimmed set with, and the untrimmed set is kept
    if the trimmed one fails it: fewer points is a weaker test, but a set our own
    Poisson-ness criterion rejects should never be handed to the estimator just because it
    happens to have the requested size.
    """
    samples = np.asarray(samples)
    if minimum_number_of_samples <= 0 or len(samples) <= minimum_number_of_samples:
        return samples
    trimmed = np.sort(samples)[:minimum_number_of_samples]
    if len(trimmed) < 5:
        return trimmed
    if anderson(np.diff(trimmed), 'expon', method='interpolate').pvalue > 0.05:
        return trimmed
    return samples


def _growing_window_result(samples, error, window_end=None, agg_stats=None, min_samples=None,
                            windows_tried=0):
    """One growing-window search outcome (see _growing_window_search). `window_end` is the
    end of the window the samples were actually drawn from -- None when the search never
    got to try a window -- and `agg_stats`/`min_samples` are the switch-side statistics of
    that same window, so a caller can run the consistency check against statistics measured
    over exactly the interval the samples came from."""
    return {'samples': samples, 'error': error, 'window_end': window_end,
            'agg_stats': agg_stats, 'min_samples': min_samples, 'windows_tried': windows_tried}


def _growing_window_search(base_method, time, MinimumNumberOfSamples=0,
                            step_ns=GROWING_WINDOW_STEP_NS, steadyStart=None, steadyEnd=None,
                            trim_to_minimum=True, window_stats=None, base_wants_window=False,
                            delta_cache=None):
    """Run `base_method` (any POISSON_SUBSAMPLING_METHODS-style callable) not on the whole
    monitoring window at once, but on the *shortest prefix of it that suffices*: try
    [steadyStart, steadyStart + step_ns], then [steadyStart, steadyStart + 2*step_ns], and
    so on, stopping at the first candidate window from which a valid subsample of the
    minimum required number of samples can be drawn. If no window up to steadyEnd yields
    one, the result carries empty samples and the last failure reason -- the caller then
    simply does no consistency check at that flow count (the same contract the base methods
    already have).

    Why this exists, next to calling the base method on the full window: the minimum sample
    size the consistency check needs (calc_min_e2e_samples, driven by
    DelayConsistencyGaurantee) is a *budget*, and spending more than it on a longer
    observation window buys no extra guarantee -- it only makes the answer arrive later.
    Sampling the full steady window always collects however many samples that window
    happens to allow; this collects the minimum required number and stops, so the result
    answers "how long must we watch this flow before we can certify it?" rather than
    "what do we learn from watching all of it?".

    `window_stats`, when given, is called as window_stats(window_end) for each candidate
    window and must return that window's own (agg_stats, min_samples) -- the switch-side
    per-segment statistics measured over [steadyStart, window_end] and the minimum e2e
    sample size they demand. This is what keeps the whole procedure inside one window: the
    required sample count for a candidate window is derived from switch statistics measured
    over that same candidate window, not from the full steady period. A window whose
    statistics cannot support the guarantee at all (min_samples None, i.e.
    calc_min_e2e_samples found MaxEpsilonDelay already at or above the tolerated error --
    the expected case for a short window, whose switch-side epsilon is larger because the
    probe collected proportionally fewer observations) is simply skipped in favour of the
    next, longer one. Without `window_stats` the search falls back to the caller-supplied
    MinimumNumberOfSamples for every window, which is what a bare
    (time, MinimumNumberOfSamples=...) call of a growing-window method does.

    Two properties worth noting:

    - It is a strict refinement of the whole-window behaviour, not a different algorithm.
      The last candidate window is [steadyStart, steadyEnd] itself, so anything the base
      method could find over the full window this can still find; it just prefers an
      earlier, shorter window when one works. With no sample target at all (no
      `window_stats` and MinimumNumberOfSamples <= 0) there is nothing to stop early on and
      this degenerates to exactly one whole-window call.
    - The window grows from a *fixed* start rather than sliding, so no packet that has
      already arrived is ever discarded, and the candidate window boundaries are identical
      across flow counts and runs (they come from steadyStart and step_ns alone), which is
      what makes the retained sample sizes and monitoring durations comparable across them.

    The bin width the base sampler needs (find_delta_for_empty_prob) is recomputed inside
    each candidate window rather than taken once from the full window, since a window's own
    arrivals are all that a deployment stopping there would actually have seen. Set
    `base_wants_window` for a base method that takes steadyStart/steadyEnd itself (e.g.
    find_samples_path_intensity, whose independence test needs the observation interval) so
    it, too, sees the candidate window rather than the full steady period.

    `delta_cache` is {prefix length: bin width} from precompute_subsample_deltas: when a
    candidate window's prefix is in it, that bin width is handed to the base method instead
    of being re-derived (identical value, since find_delta_for_empty_prob is deterministic,
    but derived once per experiment rather than once per run per window).

    Candidate windows that hold fewer packets than their own target, or that added no new
    packet over the previously tried candidate at the same target, are skipped without
    calling the base method: neither can do better than what was already tried, and
    skipping them avoids that call's dominant cost (the per-window
    find_delta_for_empty_prob search).
    """
    subSamplingError = SubSamplingError.NoError
    time = np.sort(np.asarray(time, dtype=float))

    try:
        minimum_number_of_samples = max(0, int(np.ceil(float(MinimumNumberOfSamples))))
    except (TypeError, ValueError):
        minimum_number_of_samples = 0

    if len(time) == 0 or not step_ns or step_ns <= 0 or (
            window_stats is None and minimum_number_of_samples <= 0):
        samples, err = base_method(time, MinimumNumberOfSamples=MinimumNumberOfSamples)
        return _growing_window_result(samples, err, min_samples=minimum_number_of_samples or None)

    start = float(steadyStart) if steadyStart is not None else float(time[0])
    end = float(steadyEnd) if steadyEnd is not None else float(time[-1])
    if end <= start:
        samples, err = base_method(time, MinimumNumberOfSamples=MinimumNumberOfSamples)
        return _growing_window_result(samples, err, min_samples=minimum_number_of_samples or None)

    num_steps = max(1, int(np.ceil((end - start) / float(step_ns))))
    window_ends = np.minimum(start + float(step_ns) * np.arange(1, num_steps + 1), end)
    # How many packets each candidate window [start, window_end] holds. `time` is sorted,
    # so one searchsorted gives every candidate's packet count at once.
    counts = np.searchsorted(time, window_ends, side='right')

    last_error = SubSamplingError.NotEnoughPackets + "+" + subSamplingError.value
    previous_attempt = None
    windows_tried = 0
    for count, window_end in zip(counts, window_ends):
        if window_stats is not None:
            window_agg_stats, window_min_samples = window_stats(float(window_end))
            if window_min_samples is None:
                # This window's own switch statistics cannot support the guarantee, so no
                # number of e2e samples drawn inside it would certify anything.
                last_error = SubSamplingError.NotEnoughSamples + "+" + subSamplingError.value
                continue
            target = max(0, int(np.ceil(float(window_min_samples))))
        else:
            window_agg_stats, target = None, minimum_number_of_samples
        if target <= 0 or count < target or (count, target) == previous_attempt:
            continue
        previous_attempt = (count, target)
        windows_tried += 1
        base_kwargs = {'steadyStart': start, 'steadyEnd': float(window_end)} if base_wants_window else {}
        # The bin width for this exact prefix, precomputed once in the parent
        # (precompute_subsample_deltas). Deterministic, so this is the same value the base
        # method would derive for itself -- just not re-derived by every run.
        if delta_cache is not None:
            cached_delta = delta_cache.get(int(count))
            if cached_delta is not None:
                base_kwargs['window'] = cached_delta
        # A failing candidate window is the expected case here, not an anomaly, and each
        # one is loud (find_samples_path prints on every failure). Keep that chatter out of
        # the log -- the one line below reports the outcome of the whole search instead.
        with contextlib.redirect_stdout(io.StringIO()):
            samples, err = base_method(time[:count], MinimumNumberOfSamples=target, **base_kwargs)
        if err == SubSamplingError.NoError and len(samples) >= target:
            if trim_to_minimum:
                samples = _trim_to_minimum_samples(samples, target)
            return _growing_window_result(samples, err, window_end=float(window_end),
                                           agg_stats=window_agg_stats, min_samples=target,
                                           windows_tried=windows_tried)
        last_error = err

    print("Growing-window subsampling found no valid subsample in any window from {:g} ns up "
          "to {:g} ns ({} candidate window(s) actually tried, {:g} ns step); last error: "
          "{}".format(start, end, windows_tried, step_ns, last_error))
    return _growing_window_result(np.array([], dtype=float), last_error, windows_tried=windows_tried)


def find_samples_path_growing_window(time, MinimumNumberOfSamples=0,
                                      step_ns=GROWING_WINDOW_STEP_NS, steadyStart=None,
                                      steadyEnd=None, trim_to_minimum=True, window_stats=None,
                                      window=None):
    """find_samples_path, but drawing its subsample from the shortest prefix of the
    monitoring window that can supply the minimum required number of samples instead of
    from the whole window: try [steadyStart, steadyStart + step_ns], grow by step_ns
    (default GROWING_WINDOW_STEP_NS = 5 ms) whenever that fails, and stop at the first
    window that yields a valid Poisson subsample of the required size -- or, if no window
    up to steadyEnd does, return no samples so the caller skips the consistency check. See
    _growing_window_search for the rationale and the exact search; everything about how
    packets are picked *within* a candidate window (window division, one uniform pick per
    non-empty bin, Bernoulli thinning to the target rate, Anderson-Darling validation of
    the retained gaps) is find_samples_path unchanged.

    `steadyStart`/`steadyEnd` anchor the candidate windows; without them the first and last
    packet of `time` stand in. Callers inside the EMD-vs-flows pipeline go through
    find_samples_growing_window_with_stats instead of calling this directly, so that each
    candidate window's switch-side statistics -- and hence its required sample size -- are
    measured over that same window (see call_subsampling_method)."""
    # `window`, when given, is the bin width for the WHOLE set -- the only prefix a
    # whole-window caller can mean (call_subsampling_method); the per-candidate widths come
    # from delta_cache in find_samples_growing_window_with_stats instead.
    found = _growing_window_search(
        find_samples_path, time, MinimumNumberOfSamples=MinimumNumberOfSamples,
        step_ns=step_ns, steadyStart=steadyStart, steadyEnd=steadyEnd,
        trim_to_minimum=trim_to_minimum, window_stats=window_stats,
        delta_cache=({int(len(np.asarray(time))): window} if window is not None else None))
    return found['samples'], found['error']


def find_samples_path_intensity_growing_window(time, MinimumNumberOfSamples=0,
                                                step_ns=GROWING_WINDOW_STEP_NS, steadyStart=None,
                                                steadyEnd=None, trim_to_minimum=True,
                                                window_stats=None):
    """find_samples_path_intensity under the same shortest-sufficient-window search as
    find_samples_path_growing_window (see _growing_window_search). The candidate window is
    passed into the base method as its steadyStart/steadyEnd, so its multi-lag independence
    test (chi_squared_test) is evaluated over the interval the samples were actually drawn
    from rather than over the full steady period."""
    found = _growing_window_search(
        find_samples_path_intensity, time, MinimumNumberOfSamples=MinimumNumberOfSamples,
        step_ns=step_ns, steadyStart=steadyStart, steadyEnd=steadyEnd,
        trim_to_minimum=trim_to_minimum, window_stats=window_stats, base_wants_window=True)
    return found['samples'], found['error']


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

def calculate_offline_E2E_markingProb(full_df, df_res, checkColumn, txDelay, swtichDstREDQueueDiscMaxSize, linkRate, __ns3_path, tsh, df_name, passiveProbe, samplingMethod, steadyStart, steadyEnd, samples_paths_aggregated_statistics=None):
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
            minimum_samples = 0 if samples_paths_aggregated_statistics is None else samples_paths_aggregated_statistics.get(path, {}).get('MinimumE2ESampleSizeNonMarkingProb', 0)
            samples_times = find_samples_path_new(
                time,
                txDelay,
                df_res['RTT'][path],
                df_name,
                samplingMethod,
                steadyStart,
                steadyEnd,
                steps=1,
                MinimumNumberOfSamples=minimum_samples,
            )
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
                                 samples_paths_aggregated_statistics=None, queue_names=None, linkDelays=None, linkRates=None, queue_size_trshs=None,
                                 flow_name=None, delay_cdf_sample_interval_ns=10):
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
            minimum_samples = 0 if samples_paths_aggregated_statistics is None else samples_paths_aggregated_statistics.get(path, {}).get('MinimumE2ESampleSizeDelay', 0)
            samples_times, subSamplingError = find_samples_path(
                time,
                MinimumNumberOfSamples=minimum_samples,
            )
            # samples_times, subSamplingError = find_samples_path_new(
            #     time,
            #     txDelay,
            #     df_res['RTT'][path],
            #     df_name,
            #     samplingMethod,
            #     steadyStart,
            #     steadyEnd,
            #     steps=1,
            #     MinimumNumberOfSamples=minimum_samples,
            # )
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
        if (
            queue_names is not None and len(queue_names) > 0
            and linkDelays is not None and len(linkDelays) > 0
            and linkRates is not None and len(linkRates) > 0
            and path == 0 
        ):
            print("Constructing path delay distribution for path:", path)
            groundtruth_values = construct_path_delay_distribution(
                queue_names,
                df_name,
                steadyStart,
                steadyEnd,
                linkDelays,
                linkRates,
                sample_interval_ns=delay_cdf_sample_interval_ns,
            )
            plot_name = flow_name or "flow"
            # print(f"Plotting delay distribution CDF for path {path} with {len(groundtruth_values)} ground truth samples, total {len(values)} samples, and {len(samples_values)} Poisson-sampled values.")
            plot_delay_distribution_cdfs(
                groundtruth_values,
                values,
                samples_values,
                Path(df_name) / f"{plot_name}_path_{path}_delay_cdf.png",
                title=f"Delay CDF: {plot_name}, path {path}",
            )
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

        samples_times, _ = find_samples_path(time, 0)
        samples_values = full_df_M[full_df_M['SentTime'].isin(samples_times)]['Delay'].values
        # samples_values = full_df_M[full_df_M['Time'].isin(samples_times)]['Delay'].values
        d3 = np.asarray(samples_values, dtype=float)
        d3 = d3[np.isfinite(d3)]

        distanceAwareSampling_samples_times = distanceAwareSampling(time, 2e-6)
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
    samples_times, _ = find_samples_path(time, 0)
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
            samples_times, _ = find_samples_path(time, 0)
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
    """Order whichever datacenter path stages are present.

    The physical link arrays use indices 1, 2, and 3 for T->A, A->T, and
    T->H respectively. A same-rack path contains only the final T->H stage.
    """
    stage_patterns = (
        (re.compile(r'^T\d+A\d+$'), 1),
        (re.compile(r'^A\d+T\d+$'), 2),
        (re.compile(r'^T\d+H\d+$'), 3),
    )
    ordered = []
    for queue_name in queue_names:
        for stage, (pattern, physical_index) in enumerate(stage_patterns):
            if pattern.fullmatch(queue_name):
                ordered.append(
                    (stage, queue_name, linkDelays[physical_index], linkRates[physical_index])
                )
                break
        else:
            raise ValueError('Unrecognized datacenter queue name: {}'.format(queue_name))

    ordered.sort(key=lambda item: item[0])
    if len({stage for stage, _, _, _ in ordered}) != len(ordered):
        raise ValueError('A datacenter path contains duplicate queue stages: {}'.format(queue_names))
    return (
        [item[1] for item in ordered],
        [item[2] for item in ordered],
        [item[3] for item in ordered],
    )


def _product_covariance_corrections(samples):
    """Return pair and higher-order corrections for a path product."""
    number_of_queues = samples.shape[0]
    if number_of_queues == 1:
        # A one-variable product has no cross-variable covariance terms.
        return 0.0, 0.0
    if number_of_queues == 2:
        return float(np.cov(samples)[0, 1]), 0.0
    if number_of_queues != 3:
        raise ValueError('Covariance correction supports paths with 1, 2, or 3 queues')

    means = np.nanmean(samples, axis=1)
    covariances = np.cov(samples)
    pair_correction = np.sum([
        means[i] * covariances[(i + 1) % 3][(i + 2) % 3]
        for i in range(3)
    ], axis=0)
    centered_product = np.prod(
        np.array([samples[i] - means[i] for i in range(3)]), axis=0
    )
    return pair_correction, np.mean(centered_product)

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

def remove_nan_samples(times, queue_sizes, queue_ECN_samples, queue_delay_samples, queue_drop_prob_samples):
    valid_indices = ~np.isnan(queue_sizes)
    return times[valid_indices], queue_sizes[valid_indices], queue_ECN_samples[valid_indices], queue_delay_samples[valid_indices], queue_drop_prob_samples[valid_indices]

@lru_cache(maxsize=12)
def _load_queue_trace(file_path):
    """Load all queue events plus enqueue times partitioned by source rack.

    Also carries RED's own per-event DropProb/MarkingProb, in the same event order as the
    queue sizes, so the loss/marking side of the consistency check can be sampled at
    arbitrary instants (sample_queue_probs) the way the queue size already is. Those two
    columns are deliberately the reference used for the probability metrics rather than a
    reconstruction from the queue size (sample_ECN_marking / sample_drop_probability): they
    are the switch's own marking/drop probability at that instant, and they are what
    calculate_offline_computations_DC -- and hence the per-segment
    SuccessProbMean/NonMarkingProbMean the existing consistency check compares against --
    has always used."""
    full_df = pd.read_csv(
        file_path,
        usecols=['Time', 'TotalQueueSize', 'Label', 'Action', 'DropProb', 'MarkingProb'],
    )
    times = full_df['Time'].to_numpy(dtype=float)
    queue_sizes = full_df['TotalQueueSize'].to_numpy(dtype=float)
    drop_probs = full_df['DropProb'].to_numpy(dtype=float)
    marking_probs = full_df['MarkingProb'].to_numpy(dtype=float)
    order = np.lexsort((-queue_sizes, times))

    enqueue_df = full_df[full_df['Action'] == 'E']
    rack_octets = pd.to_numeric(
        enqueue_df['Label'].str.extract(r'^10\.(\d+)\.', expand=False),
        errors='coerce',
    )
    enqueue_times_by_rack = {
        int(rack_octet) - 1: enqueue_df.loc[rack_octets == rack_octet, 'Time'].to_numpy(dtype=float)
        for rack_octet in rack_octets.dropna().unique()
    }

    # Queue sampling uses every event. Only packets_of_interest is rack-filtered.
    return (times[order], queue_sizes[order], enqueue_times_by_rack,
            drop_probs[order], marking_probs[order])


def total_packets_of_interest(file_path, start_time, end_time, source_rack):
    _, _, enqueue_times_by_rack, _, _ = _load_queue_trace(file_path)
    interest_times = enqueue_times_by_rack.get(source_rack, np.array([], dtype=float))
    return int(np.count_nonzero(
        (interest_times >= start_time) & (interest_times <= end_time)
    ))

def sample_queue_size(times, file_path, link_rate):
    # print(f"Sampling total queue size from {file_path} with link rate {link_rate} bpns")
    df_times, df_queue_sizes, _, _, _ = _load_queue_trace(file_path)
    # queue_name = file_path.split('/')[-1].split('_')[0]
    # exp = file_path.split('/')[-2]
    # # if "T0A0" in queue_name or "A0T2" in queue_name:
    # # times_t = [[10000000, 20000000], [20000000, 30000000], [30000000, 40000000], [40000000, 50000000], [50000000, 60000000], [60000000, 70000000], 
    # #             [70000000, 80000000], [80000000, 90000000], [90000000, 100000000], [10000000, 100000000]]
    # times_t = [[10000000, 100000000]]
    # if "T0A0" in queue_name:
    #     for t in times_t:
    #         temp_10_1_1 = len(full_df[(full_df['Label'].str.contains('10.1.1.', na=False, regex=False)) & (full_df['Action'] == 'E') & (full_df['Time'] >= t[0]) & (full_df['Time'] <= t[-1])])
    #         temp_10_1_2 = len(full_df[(full_df['Label'].str.contains('10.1.2.', na=False, regex=False)) & (full_df['Action'] == 'E') & (full_df['Time'] >= t[0]) & (full_df['Time'] <= t[-1])])
    #         temp_10_1_3 = len(full_df[(full_df['Label'].str.contains('10.1.3.', na=False, regex=False)) & (full_df['Action'] == 'E') & (full_df['Time'] >= t[0]) & (full_df['Time'] <= t[-1])])
    #         temp_10_1_4 = len(full_df[(full_df['Label'].str.contains('10.1.4.', na=False, regex=False)) & (full_df['Action'] == 'E') & (full_df['Time'] >= t[0]) & (full_df['Time'] <= t[-1])])
    #         temp_10_1_5 = len(full_df[(full_df['Label'].str.contains('10.1.5.', na=False, regex=False)) & (full_df['Action'] == 'E') & (full_df['Time'] >= t[0]) & (full_df['Time'] <= t[-1])])
    #         temp_10_1_6 = len(full_df[(full_df['Label'].str.contains('10.1.6.', na=False, regex=False)) & (full_df['Action'] == 'E') & (full_df['Time'] >= t[0]) & (full_df['Time'] <= t[-1])])
    #         temp_total = len(full_df[(full_df['Action'] == 'E') & (full_df['Time'] >= t[0]) & (full_df['Time'] <= t[-1])])
    #         print(f"Number of packets for experiment {exp}, queue {Fore.GREEN} {queue_name} {Fore.RESET} from {t[0]} to {t[-1]}: {temp_total}")
    #         print(f"percentage of 10.1.1 : {Fore.RED} {temp_10_1_1/temp_total:.2%} {Fore.RESET} expected around {Fore.BLUE} 16.67% {Fore.RESET}. Number of packets: {temp_10_1_1}")
    #         print(f"percentage of 10.1.2 : {Fore.RED} {temp_10_1_2/temp_total:.2%} {Fore.RESET} expected around {Fore.BLUE} 16.67% {Fore.RESET}. Number of packets: {temp_10_1_2}")
    #         print(f"percentage of 10.1.3 : {Fore.RED} {temp_10_1_3/temp_total:.2%} {Fore.RESET} expected around {Fore.BLUE} 16.67% {Fore.RESET}. Number of packets: {temp_10_1_3}")
    #         print(f"percentage of 10.1.4 : {Fore.RED} {temp_10_1_4/temp_total:.2%} {Fore.RESET} expected around {Fore.BLUE} 16.67% {Fore.RESET}. Number of packets: {temp_10_1_4}")
    #         print(f"percentage of 10.1.5 : {Fore.RED} {temp_10_1_5/temp_total:.2%} {Fore.RESET} expected around {Fore.BLUE} 16.67% {Fore.RESET}. Number of packets: {temp_10_1_5}")
    #         print(f"percentage of 10.1.6 : {Fore.RED} {temp_10_1_6/temp_total:.2%} {Fore.RESET} expected around {Fore.BLUE} 16.67% {Fore.RESET}. Number of packets: {temp_10_1_6}")
    # if "A0T2" in queue_name:
    #     for t in times_t:
    #         temp_10_1 = len(full_df[(full_df['Label'].str.contains('10.1.', na=False, regex=False)) & (full_df['Action'] == 'E') & (full_df['Time'] >= t[0]) & (full_df['Time'] <= t[-1])])
    #         temp_10_2 = len(full_df[(full_df['Label'].str.contains('10.2.', na=False, regex=False)) & (full_df['Action'] == 'E') & (full_df['Time'] >= t[0]) & (full_df['Time'] <= t[-1])])
    #         temp_10_4 = len(full_df[(full_df['Label'].str.contains('10.4.', na=False, regex=False)) & (full_df['Action'] == 'E') & (full_df['Time'] >= t[0]) & (full_df['Time'] <= t[-1])])
    #         temp_total = len(full_df[(full_df['Action'] == 'E') & (full_df['Time'] >= t[0]) & (full_df['Time'] <= t[-1])])
    #         print(f"Number of packets for experiment {exp}, queue {Fore.GREEN} {queue_name} {Fore.RESET} from {t[0]} to {t[-1]}: {temp_total}")
    #         print(f"percentage of 10.1 : {Fore.RED} {temp_10_1/temp_total:.2%} {Fore.RESET} expected around {Fore.BLUE} 33.33% {Fore.RESET}. Number of packets: {temp_10_1}")
    #         print(f"percentage of 10.2 : {Fore.RED} {temp_10_2/temp_total:.2%} {Fore.RESET} expected around {Fore.BLUE} 33.33% {Fore.RESET}. Number of packets: {temp_10_2}")
    #         print(f"percentage of 10.4 : {Fore.RED} {temp_10_4/temp_total:.2%} {Fore.RESET} expected around {Fore.BLUE} 33.33% {Fore.RESET}. Number of packets: {temp_10_4}")
    # if "T2H3" in queue_name:
    #     for t in times_t:
    #         temp_10_1 = len(full_df[(full_df['Label'].str.contains('10.1.', na=False, regex=False)) & (full_df['Action'] == 'E') & (full_df['Time'] >= t[0]) & (full_df['Time'] <= t[-1])])
    #         temp_10_2 = len(full_df[(full_df['Label'].str.contains('10.2.', na=False, regex=False)) & (full_df['Action'] == 'E') & (full_df['Time'] >= t[0]) & (full_df['Time'] <= t[-1])])
    #         temp_10_3 = len(full_df[(full_df['Label'].str.contains('10.3.', na=False, regex=False)) & (full_df['Action'] == 'E') & (full_df['Time'] >= t[0]) & (full_df['Time'] <= t[-1])])
    #         temp_10_4 = len(full_df[(full_df['Label'].str.contains('10.4.', na=False, regex=False)) & (full_df['Action'] == 'E') & (full_df['Time'] >= t[0]) & (full_df['Time'] <= t[-1])])
    #         temp_total = len(full_df[(full_df['Action'] == 'E') & (full_df['Time'] >= t[0]) & (full_df['Time'] <= t[-1])])
    #         print(f"Number of packets for experiment {exp}, queue {Fore.GREEN} {queue_name} {Fore.RESET} from {t[0]} to {t[-1]}: {temp_total}")
    #         print(f"percentage of 10.1 : {Fore.RED} {temp_10_1/temp_total:.2%} {Fore.RESET} expected around {Fore.BLUE} 25% {Fore.RESET}. Number of packets: {temp_10_1}")
    #         print(f"percentage of 10.2 : {Fore.RED} {temp_10_2/temp_total:.2%} {Fore.RESET} expected around {Fore.BLUE} 25% {Fore.RESET}. Number of packets: {temp_10_2}")
    #         print(f"percentage of 10.3 : {Fore.RED} {temp_10_3/temp_total:.2%} {Fore.RESET} expected around {Fore.BLUE} 25% {Fore.RESET}. Number of packets: {temp_10_3}")
    #         print(f"percentage of 10.4 : {Fore.RED} {temp_10_4/temp_total:.2%} {Fore.RESET} expected around {Fore.BLUE} 25% {Fore.RESET}. Number of packets: {temp_10_4}")
    sample_times = np.asarray(times, dtype=float)
    return find_queue_size_at_time(df_times, df_queue_sizes, sample_times, link_rate)

def sample_queue_probs(times, file_path):
    """RED's own drop and marking probability in force at each instant of `times`, as
    (drop_probs, marking_probs) -- the loss/marking counterpart of sample_queue_size, read
    from the same cached queue trace (_load_queue_trace) and looked up with the same
    last-event-at-or-before-t step convention find_queue_size_at_time uses for the size.

    Instants outside the trace's span yield NaN, exactly as the queue-size sampler does, so
    a caller can drop them with the same isfinite mask."""
    df_times, _, _, df_drop, df_marking = _load_queue_trace(file_path)
    sample_times = np.asarray(times, dtype=float)
    if df_times.size == 0:
        nan = np.full(sample_times.shape, np.nan)
        return nan, nan.copy()
    positions = np.searchsorted(df_times, sample_times, side='right') - 1
    invalid = (positions < 0) | (positions >= df_times.size - 1)
    positions = np.clip(positions, 0, df_times.size - 1)
    drop = df_drop[positions].astype(float)
    marking = df_marking[positions].astype(float)
    drop[invalid] = np.nan
    marking[invalid] = np.nan
    return drop, marking


def sample_ECN_marking(queue_size_samples, queue_size_trsh):
    return (queue_size_samples >= queue_size_trsh).astype(int)

def sample_drop_probability(queue_size_samples, queue_capacity, packets_cfd):
    drop_prob = np.zeros(queue_size_samples.shape, dtype=float)
    available_space = queue_capacity - queue_size_samples
    for i in range(len(queue_size_samples)):
        drop_prob[i] = packets_cfd.calculate_probability_greater_than(available_space[i])
    return drop_prob

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

def sample_total_queue_size_non_combined(res, times, queue_names, dir_prefix, linkDelays, linkRates, queue_size_trshs, queue_capacity,
                                         path_observation=False, sampling_factor=None, source_rack=None):
    queue_names, linkDelays, linkRates = sort_queues_by_path(queue_names, linkDelays, linkRates)
    packets_cfd = PacketCDF()
    packets_cfd.load_cdf_data('/media/experiments/ns-allinone-3.41/ns-3.41/scratch/ECNMC/DCWorkloads/packet_size_cdf_{}.csv'.format(dir_prefix.split('/')[-5]))
    tag = 'poisson'
    if path_observation:
        tag = 'e2e'
    for queue_name in queue_names:
        res[queue_name+ tag + '_samples_queue_delay_mean'] = 0
        res[queue_name+ tag + '_samples_queue_success_prob_mean'] = 0
        res[queue_name+ tag + '_samples_queue_nonmarking_prob_mean'] = 0
        res[queue_name+ tag + '_samples_queue_delay_std'] = 0
        res[queue_name+ tag + '_samples_queue_success_prob_std'] = 0
        res[queue_name+ tag + '_samples_queue_nonmarking_prob_std'] = 0
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
        queue_ECN_samples = np.zeros((len(queue_names), len(sample_times_itr)))
        queue_success_prob_samples = np.zeros((len(queue_names), len(sample_times_itr)))
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
            queue_ECN_samples[idx][invalid_indices] = np.nan
            queue_success_prob_samples[idx][~invalid_indices] = 1 - sample_drop_probability(queue_size_samples[idx][~invalid_indices], queue_capacity[idx], packets_cfd)
            queue_success_prob_samples[idx][invalid_indices] = np.nan
            queue_delay_samples[idx][~invalid_indices] = sample_queueing_delay(queue_size_samples[idx][~invalid_indices], linkRates[idx])

            prob_non_empty = queue_size_samples[idx][~invalid_indices] > 0
            prob_non_empty = np.sum(prob_non_empty) / len(prob_non_empty)

            if path_observation:
                if source_rack is None:
                    raise ValueError('source_rack is required for path-observation packet counts')
                res[queue_name+ 'packets_of_interest'] = total_packets_of_interest(
                    file_path, sample_times[0], sample_times[-1], source_rack
                )
            res[queue_name+ tag + '_samples_queue_delay_mean'] = (res[queue_name+ tag + '_samples_queue_delay_mean'] * itr + np.nanmean(queue_delay_samples[idx])) / (itr + 1)
            res[queue_name+ tag + '_samples_queue_success_prob_mean'] = (res[queue_name+ tag + '_samples_queue_success_prob_mean'] * itr + np.nanmean(queue_success_prob_samples[idx])) / (itr + 1)
            res[queue_name+ tag + '_samples_queue_nonmarking_prob_mean'] = (res[queue_name+ tag + '_samples_queue_nonmarking_prob_mean'] * itr + (1 - np.nanmean(queue_ECN_samples[idx][~invalid_indices]))) / (itr + 1)
            res[queue_name+ tag + '_samples_queue_delay_std'] = (res[queue_name+ tag + '_samples_queue_delay_std'] * itr + np.nanstd(queue_delay_samples[idx])) / (itr + 1)
            res[queue_name+ tag + '_samples_queue_success_prob_std'] = (res[queue_name+ tag + '_samples_queue_success_prob_std'] * itr + np.nanstd(queue_success_prob_samples[idx][~invalid_indices])) / (itr + 1)
            res[queue_name+ tag + '_samples_queue_nonmarking_prob_std'] = (res[queue_name+ tag + '_samples_queue_nonmarking_prob_std'] * itr + np.nanstd(queue_ECN_samples[idx][~invalid_indices])) / (itr + 1)
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
    nonmarking_samples = 1 - queue_ECN_samples
    nonmarking_samples = nonmarking_samples[:, ~np.isnan(nonmarking_samples).any(axis=0)]
    diff, diff_extra = _product_covariance_corrections(nonmarking_samples)
    res['sum_poisson_samples_queue_nonmarking_prob_pair_covariance'] = diff
    res['sum_poisson_samples_queue_nonmarking_prob_triple_covariance'] = diff_extra

    queue_success_prob_samples_ = queue_success_prob_samples[:, ~np.isnan(queue_success_prob_samples).any(axis=0)]
    diff_success_prob, diff_success_prob_extra = _product_covariance_corrections(
        queue_success_prob_samples_
    )
    res['sum_poisson_samples_queue_success_prob_pair_covariance'] = diff_success_prob
    res['sum_poisson_samples_queue_success_prob_triple_covariance'] = diff_success_prob_extra
    return remove_nan_samples(sample_times_itr, np.sum(queue_size_samples, axis=0), np.any(queue_ECN_samples, axis=0).astype(int), np.sum(queue_delay_samples, axis=0), np.prod(queue_success_prob_samples, axis=0)), res

def combine_sampling_results(res, queue_names):
    queue_names, _, _ = sort_queues_by_path(
        queue_names, [None, None, None, None], [None, None, None, None]
    )
    for queue_name in queue_names:
        idx = queue_names.index(queue_name)
        res[queue_name+'error_bound'] = res[queue_name+'e2e_samples_queue_delay_std'] * 1.96 / np.sqrt(res[queue_name+'e2e_samples_queue_delay_count']) + res[queue_name+'poisson_samples_queue_delay_std'] * 1.96 / np.sqrt(res[queue_name+'poisson_samples_queue_delay_count'])
        res[queue_name+'e2e_vs_poisson_consistent'] = int(abs(res[queue_name+'e2e_samples_queue_delay_mean'] - res[queue_name+'poisson_samples_queue_delay_mean']) <= res[queue_name+'error_bound'])

        res[queue_name+'success_prob_error_bound'] = res[queue_name+'e2e_samples_queue_success_prob_std'] * 1.96 / np.sqrt(res[queue_name+'e2e_samples_queue_delay_count']) + res[queue_name+'poisson_samples_queue_success_prob_std'] * 1.96 / np.sqrt(res[queue_name+'poisson_samples_queue_delay_count'])
        res[queue_name+'e2e_vs_poisson_consistent_success_prob'] = int(abs(res[queue_name+'e2e_samples_queue_success_prob_mean'] - res[queue_name+'poisson_samples_queue_success_prob_mean']) <= res[queue_name+'success_prob_error_bound'])

        res[queue_name+'nonmarking_prob_error_bound'] = res[queue_name+'e2e_samples_queue_nonmarking_prob_std'] * 1.96 / np.sqrt(res[queue_name+'e2e_samples_queue_delay_count']) + res[queue_name+'poisson_samples_queue_nonmarking_prob_std'] * 1.96 / np.sqrt(res[queue_name+'poisson_samples_queue_delay_count'])
        res[queue_name+'e2e_vs_poisson_consistent_nonmarking_prob'] = int(abs(res[queue_name+'e2e_samples_queue_nonmarking_prob_mean'] - res[queue_name+'poisson_samples_queue_nonmarking_prob_mean']) <= res[queue_name+'nonmarking_prob_error_bound'])
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

    total_queue_sizes = np.sum(queue_size_samples, axis=0)
    total_queue_delays = np.sum(queue_delay_samples, axis=0)
    valid = np.isfinite(total_queue_sizes) & np.isfinite(total_queue_delays)
    sampled = (
        np.asarray(times)[valid],
        total_queue_sizes[valid],
        np.any(queue_ECN_samples, axis=0).astype(int)[valid],
        total_queue_delays[valid],
    )
    return sampled, res


def construct_path_delay_distribution(
    queue_names,
    dir_prefix,
    steady_start,
    steady_end,
    link_delays,
    link_rates,
    sample_interval_ns=10,
):
    """Construct simultaneous network observations of all path queues.

    Every queue is observed at exactly the same network times, drawn from one
    Poisson-process realization with mean inter-arrival interval
    `sample_interval_ns` (see generate_poisson_observation_times) -- not a
    uniform grid. The queueing delays at each time are summed to form the
    path-delay ground truth. Unlike an end-to-end observer, sample times are
    never shifted by propagation or by the delay encountered at an earlier
    queue.
    """
    if steady_end <= steady_start:
        raise ValueError("steady_end must be greater than steady_start")
    if sample_interval_ns <= 0:
        raise ValueError("sample_interval_ns must be positive")
    if not queue_names:
        return np.array([], dtype=float)

    num_observations = max(2, int((steady_end - steady_start) / sample_interval_ns))
    sample_times = generate_poisson_observation_times(steady_start, steady_end, num_observations)
    prefix = str(dir_prefix)
    if not prefix.endswith("/"):
        prefix += "/"
    ordered_queues, _, ordered_rates = sort_queues_by_path(
        queue_names, link_delays, link_rates
    )
    total_delay = np.zeros(len(sample_times), dtype=float)
    valid = np.ones(len(sample_times), dtype=bool)
    for queue_name, link_rate in zip(ordered_queues, ordered_rates):
        queue_sizes = sample_queue_size(
            sample_times,
            prefix + queue_name + '_PoissonSampler_queueSize.csv',
            link_rate,
        )
        queue_valid = np.isfinite(queue_sizes)
        valid &= queue_valid
        total_delay[queue_valid] += sample_queueing_delay(
            queue_sizes[queue_valid], link_rate
        )
    return total_delay[valid]


def construct_path_prob_ground_truth(queue_names, dir_prefix, steady_start, steady_end,
                                      link_delays, link_rates, sample_interval_ns=10):
    """The path's true success and non-marking probabilities over [steady_start,
    steady_end], as {metric: probability} -- the probability counterpart of
    construct_path_delay_distribution, and the reference every family's own estimate is
    scored against.

    Same construction as the delay ground truth: one Poisson realization dense enough to
    give `(steady_end - steady_start) / sample_interval_ns` observations, every path queue
    observed at those instants, per-segment probabilities formed from RED's own
    drop/marking probability there, and the path probability taken as their product (see
    sample_path_prob_stats). Being a product of means rather than a distribution, this is a
    single number per metric -- a Bernoulli has nothing else to describe -- which is why the
    delay pipeline's EMD becomes a plain |p_ground_truth - p_family| here.

    The confidence value only affects the per-segment epsilons, which a ground truth at this
    rate does not use, so it is fixed at 1.96 for the call."""
    if not queue_names:
        return {metric: np.nan for metric in PROB_METRIC_KEYS}
    num_observations = max(2, int((steady_end - steady_start) / sample_interval_ns))
    sample_times = generate_poisson_observation_times(steady_start, steady_end, num_observations)
    prefix = str(dir_prefix)
    if not prefix.endswith("/"):
        prefix += "/"
    stats = sample_path_prob_stats(sample_times, queue_names, prefix, link_delays, link_rates, 1.96)
    return {metric: stats[metric + 'PathProb'] for metric in PROB_METRIC_KEYS}


def construct_path_delay_distribution_path_observation(
    queue_names,
    dir_prefix,
    steady_start,
    steady_end,
    link_delays,
    link_rates,
    sample_interval_ns=10,
):
    """Construct the path delay seen by a *travelling* observer along the path.

    Same idea calculate_offline_delay_bias_DC already uses for its mean-delay
    study (sample_total_queue_size_non_combined with path_observation=True),
    applied here to build a whole ground-truth CDF: a virtual probe is
    released at each instant of one Poisson-process realization
    (generate_poisson_observation_times, mean inter-arrival
    `sample_interval_ns`), observes the first queue's instantaneous queueing
    delay, then *waits out* that queueing delay plus the link's propagation
    delay before observing the second queue, and so on down the path. The
    per-queue delays it observes are summed into one path-delay sample, so
    each queue after the first is observed at the time the probe would
    actually reach it rather than at the release instant.

    Contrast construct_path_delay_distribution, which observes every queue at
    exactly the same instant (a simultaneous snapshot of the whole path).
    Both give the same ground-truth *mean* (linearity of expectation), but
    their CDFs differ whenever the queues are correlated in time -- and it is
    the travelling observer's sum, not the snapshot's, that an actual packet
    experiences, hence what an end-to-end packet-delay measurement should be
    compared against. Selected via GROUNDTRUTH_METHODS / the
    `groundtruth_method` argument threaded through the EMD pipeline.

    A probe whose (shifted) observation time falls outside a queue's recorded
    trace window is dropped outright rather than partially counted, since its
    path total would be missing that queue's term.
    """
    if steady_end <= steady_start:
        raise ValueError("steady_end must be greater than steady_start")
    if sample_interval_ns <= 0:
        raise ValueError("sample_interval_ns must be positive")
    if not queue_names:
        return np.array([], dtype=float)

    num_observations = max(2, int((steady_end - steady_start) / sample_interval_ns))
    release_times = generate_poisson_observation_times(steady_start, steady_end, num_observations)
    prefix = str(dir_prefix)
    if not prefix.endswith("/"):
        prefix += "/"
    ordered_queues, ordered_delays, ordered_rates = sort_queues_by_path(
        queue_names, link_delays, link_rates
    )

    total_delay = np.zeros(len(release_times), dtype=float)
    # Indices (into total_delay) of the probes still travelling; a probe that
    # falls outside a queue's trace window drops out for good.
    alive = np.arange(len(release_times))
    observation_times = np.asarray(release_times, dtype=float)
    for queue_name, link_delay, link_rate in zip(ordered_queues, ordered_delays, ordered_rates):
        if alive.size == 0:
            break
        queue_sizes = sample_queue_size(
            observation_times,
            prefix + queue_name + '_PoissonSampler_queueSize.csv',
            link_rate,
        )
        observed = np.isfinite(queue_sizes)
        alive = alive[observed]
        queue_delays = sample_queueing_delay(queue_sizes[observed], link_rate)
        total_delay[alive] += queue_delays
        # The travelling observer's defining step: advance to when this probe
        # would actually arrive at the next queue -- after draining this
        # queue and crossing the link (the same shift
        # sample_total_queue_size_non_combined applies for path_observation).
        observation_times = (observation_times[observed] + queue_delays + link_delay).round()
    return total_delay[alive]


GROUNDTRUTH_METHODS = {
    # Every queue observed at the same instant: a simultaneous snapshot of the path.
    'simultaneous': construct_path_delay_distribution,
    # A probe travelling the path, waiting out each queue before observing the next.
    'path_observation': construct_path_delay_distribution_path_observation,
}

# Filename tag per ground-truth method. 'simultaneous' deliberately maps to the
# empty string so that everything already computed (and every filename already
# on disk) keeps its existing name -- only the newer ground-truth methods add a
# tag of their own on top of the subsampling-method tag.
_GROUNDTRUTH_METHOD_TAGS = {
    'simultaneous': '',
    'path_observation': 'pathobs',
}


def _resolve_groundtruth_method(groundtruth_method):
    """Look up a ground-truth path-delay CDF constructor by name (a key of
    GROUNDTRUTH_METHODS). All entries share the
    (queue_names, dir_prefix, steady_start, steady_end, link_delays, link_rates,
    sample_interval_ns=...) signature, so either can be dropped in wherever
    construct_path_delay_distribution was called directly."""
    try:
        return GROUNDTRUTH_METHODS[groundtruth_method]
    except KeyError:
        raise ValueError("Unknown groundtruth_method {!r}; choose one of {}".format(
            groundtruth_method, list(GROUNDTRUTH_METHODS)))


def groundtruth_method_tag(groundtruth_method):
    """The filename tag identifying which ground truth an output was computed
    against, as an already-prefixed '_<tag>' fragment (or '' for the original
    'simultaneous' ground truth, so pre-existing filenames are unchanged).
    Appended after the subsampling-method tag by every output path in the
    EMD-vs-flows pipeline, see subsampling_methods_tag."""
    if groundtruth_method not in _GROUNDTRUTH_METHOD_TAGS:
        _resolve_groundtruth_method(groundtruth_method)  # raises with the valid choices
    tag = _GROUNDTRUTH_METHOD_TAGS[groundtruth_method]
    return '_' + tag if tag else ''


def plot_delay_distribution_cdfs(
    groundtruth_delays,
    all_packet_delays,
    subsampled_packet_delays,
    output_path,
    title="Delay distribution comparison",
    extra_series=None,
    subsampled_label="Subsampled received packets",
    groundtruth_label="Network queues (ground truth)",
):
    """Plot reconstructed, all-packet, and subsampled delay CDFs together.
    `extra_series`, if given, is an iterable of (values, label, color) tuples
    appended after the three fixed series -- e.g. one per further
    Poisson-adaptive subsampling method and one per uniform "1-in-stride"
    method, see plot_one_run_delay_cdfs. `subsampled_label` /
    `groundtruth_label` name the two non-obvious fixed series, since which
    subsampling algorithm and which ground-truth construction
    (GROUNDTRUTH_METHODS) produced them is now the caller's choice."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    series = [
        (groundtruth_delays, groundtruth_label, "C0"),
        (all_packet_delays, "All received packets on path", "C1"),
        (subsampled_packet_delays, subsampled_label, "C2"),
    ]
    if extra_series:
        series.extend(extra_series)

    fig, axis = plt.subplots(figsize=(30, 15))
    for raw_values, label, color in series:
        values = np.asarray(raw_values, dtype=float).reshape(-1)
        values = np.sort(values[np.isfinite(values)])
        curve_label = f"{label} (n={len(values)})"
        if values.size:
            probabilities = np.arange(1, len(values) + 1) / len(values)
            axis.step(values, probabilities, where="post", label=curve_label, color=color, linewidth=5)
        else:
            # Keep unavailable samples visible in the legend without inventing
            # a fallback distribution.
            axis.plot([], [], label=curve_label, color=color)

    axis.set_title(title)
    axis.set_xlabel("Queuing delay (ns)")
    axis.set_ylabel("Cumulative probability")
    axis.set_ylim(0, 1.02)
    axis.grid(True, alpha=0.35)
    axis.legend(fontsize=30, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------------------
# Probability metrics (loss / ECN marking) alongside delay
# ---------------------------------------------------------------------------------------
# The consistency check exists for three e2e quantities, not one: the path's mean queuing
# delay, its success (non-drop) probability and its non-marking probability. The delay side
# compares a MEAN against a per-segment SUM; both probabilities compare a PRODUCT against a
# per-segment product, which the check does in log space (see PostProcessing's
# check_all_successProbConsistency / check_all_nonMarkingProbConsistency, whose
# 'event_poisson_eventAvg' branch is what prob_consistency_band below reproduces).
#
# Three facts about these two metrics shape everything here, all measured on
# Results_forward_DCW_DC24Servers_WOIncast/Google_AllRPC 0.5/0.7:
#
# 1. A per-packet outcome is BERNOULLI, so its "distribution" is one number. The
#    Wasserstein distance between two Bernoullis is exactly |p1 - p2|, so the EMD of the
#    delay pipeline collapses to an absolute probability difference here -- reported as
#    such rather than dressed up as a distribution distance -- and percentiles/CDFs of a
#    0/1 variable carry no information at all, so they are not produced.
# 2. Marking is well posed on both sides: RED's own per-instant MarkingProb gives a path
#    non-marking probability of 0.9226 (product over the three path queues) against 0.9121
#    observed end to end (every sent packet, dropped ones counted as marked). The
#    alternative "queue >= 15% of capacity" reconstruction gives 0.835, which does not
#    match, so RED's own column is the reference used. Note the residual asymmetry the
#    missing drop information leaves: the e2e side counts a dropped packet as not having
#    passed unmarked, while the switch side -- which sees no drops at all (point 3) -- has
#    no way to. That biases the e2e estimate low by the loss rate, 0.26% here.
# 3. Loss is NOT well posed on the switch side of these traces: RED's DropProb is
#    identically 0 at every event of every path queue, and the queue never comes within one
#    packet of capacity (P(queue > 90% of capacity) <= 1e-4), so the reconstruction
#    sample_drop_probability yields exactly 0 too. The 0.26% of packets that never arrive
#    are therefore not attributable to these three queues at all. The success-probability
#    machinery below is complete and will work on a trace that records drops, but on this
#    data its reference is p=1 with zero variance, which makes its check uninformative --
#    see the warning the text summary prints.
PROB_METRIC_KEYS = ('success_prob', 'non_marking_prob')

PROB_METRICS = {
    'success_prob': {
        'label': 'Success probability',
        'short': 'P(success)',
        # Per-packet outcome: did the packet arrive at all. Defined on every SENT packet --
        # restricting it to received packets would make the estimate 1 by construction.
        'packet_column': 'Success',
        'received_only': False,
        'queue_column': 'drop',
        'mean_key': 'SuccessProbMean',
        'epsilon_key': 'MaxEpsilonSuccessProb',
        'std_key': 'e2eSuccessProbStd',
    },
    'non_marking_prob': {
        'label': 'Non-marking probability',
        'short': 'P(not marked)',
        # Per-packet outcome: did the packet get through unmarked. Over every SENT packet,
        # with a dropped packet counting as MARKED -- a packet dropped by a full queue was
        # necessarily past the ECN marking threshold on its way in, so semantically its ECN
        # is 1. That matters for matching the switch side: the per-segment
        # 1 - mean(MarkingProb) product is the unconditional probability of passing a queue
        # unmarked, so the e2e estimate must be unconditional too. Excluding dropped
        # packets (an earlier version of this) biases the estimate upward instead.
        'packet_column': 'NonMarked',
        'received_only': False,
        'queue_column': 'marking',
        'mean_key': 'NonMarkingProbMean',
        'epsilon_key': 'MaxEpsilonNonMarkingProb',
        'std_key': 'e2eNonMarkingProbStd',
    },
}


def path_prob_product_std(segment_probs, segment_stds):
    """Standard deviation of the PATH probability P = P1*P2*...*Pm, given each segment's own
    mean and standard deviation and assuming the segments are independent:

        Var(prod Pi) = prod(sigma_i^2 + mu_i^2) - prod(mu_i^2)

    (each factor's second moment multiplied out, minus the square of the product of means --
    the exact variance of a product of independent variables).

    This is deliberately NOT the sum of the per-segment stds. That convention belongs to the
    delay side, where the path quantity is a SUM of per-segment delays and summing stds is a
    deliberately conservative bound (Var of a sum would be sqrt(sum of variances) under
    independence; summing stds is the comonotonic worst case). A probability is a PRODUCT,
    so summing its segments' stds has no interpretation at all and badly overstates the
    spread: on the marking data measured here it gives 0.428 where the product rule gives
    0.276, which widens the consistency band by ~55% and makes the check correspondingly
    too permissive.

    Returns NaN unless every segment contributes a finite mean and std."""
    probs = np.asarray(segment_probs, dtype=float)
    stds = np.asarray(segment_stds, dtype=float)
    if probs.size == 0 or probs.size != stds.size:
        return np.nan
    if not (np.all(np.isfinite(probs)) and np.all(np.isfinite(stds))):
        return np.nan
    variance = float(np.prod(stds ** 2 + probs ** 2) - np.prod(probs ** 2))
    # Mathematically non-negative; clamp the floating-point residue when every segment is
    # deterministic (all stds 0), where the two products are equal.
    return float(np.sqrt(max(variance, 0.0)))


def received_rows(packet_df):
    """The rows of a packet frame whose packets actually arrived -- the set on which a
    DELAY is observable at all. The frame itself holds every SENT packet (see
    prepare_emd_vs_flows_data), because neither probability metric may condition on
    receipt: the success probability would then be 1 by construction, and the non-marking
    probability would drop the dropped packets that must count as marked."""
    return packet_df[packet_df['IsReceived'] == 1]


def prob_metric_values(packet_df, metric):
    """One probability metric's per-packet 0/1 outcomes from a packet frame. Both metrics
    are defined over every SENT packet (see PROB_METRICS): a dropped packet is a failure for
    the success probability and counts as marked for the non-marking one. NaNs are dropped,
    so the length of the result is the sample size that metric's estimate rests on."""
    spec = PROB_METRICS[metric]
    rows = received_rows(packet_df) if spec['received_only'] else packet_df
    values = np.asarray(rows[spec['packet_column']].values, dtype=float)
    return values[np.isfinite(values)]


def prob_metric_label(metric):
    """Human-readable name of a probability metric (a PROB_METRICS key)."""
    return (PROB_METRICS.get(metric) or {}).get('label', str(metric))


def sample_path_prob_stats(times, queue_names, dir_prefix, linkDelays, linkRates,
                            confidenceValue):
    """Per-segment loss/marking statistics of the path, as seen by a probe observing every
    queue at the instants `times` -- the probability counterpart of the delay half of
    compute_poisson_agg_stats.

    For each queue this takes RED's own drop and marking probability in force at those
    instants (sample_queue_probs) and forms, exactly as calculate_offline_computations_DC
    always has, `1 - mean(prob)` and `std(prob)`. It then aggregates them the way
    analyze_single_experiment does for the consistency check:

      - '<metric>Mean'   = SUM of log(per-segment probability) -- a path probability is the
                           product of its segments', so the check works in log space;
      - 'MaxEpsilon<..>' = the LARGEST per-segment relative confidence interval
                           (calc_epsilon_loss / calc_epsilon_marking), the same
                           worst-segment convention MaxEpsilonDelay uses;
      - 'e2e<..>Std'     = the std of the PATH probability under segment independence,
                           sqrt(prod(sigma_i^2+mu_i^2) - prod(mu_i^2)) -- see
                           path_prob_product_std. NOT the sum of the per-segment stds: a
                           path probability is a product, so its variance follows the
                           product rule, unlike the delay side where the path quantity is a
                           sum and summing stds is the conservative convention.

    Returns a dict carrying those three keys per probability metric plus the per-segment
    probabilities themselves ('<metric>PerSegment') for reporting."""
    ordered_queues, _, _ = sort_queues_by_path(queue_names, linkDelays, linkRates)
    stats = {}
    per_segment = {metric: [] for metric in PROB_METRIC_KEYS}
    per_queue_stats = {metric: [] for metric in PROB_METRIC_KEYS}
    for queue_name in ordered_queues:
        drop, marking = sample_queue_probs(times, dir_prefix + queue_name + '_PoissonSampler_queueSize.csv')
        for metric in PROB_METRIC_KEYS:
            raw = drop if PROB_METRICS[metric]['queue_column'] == 'drop' else marking
            raw = raw[np.isfinite(raw)]
            prob = float(1 - np.mean(raw)) if raw.size else np.nan
            std = float(np.std(raw)) if raw.size else np.nan
            per_segment[metric].append(prob)
            per_queue_stats[metric].append({'prob': prob, 'std': std, 'sampleSize': int(raw.size)})

    for metric in PROB_METRIC_KEYS:
        spec = PROB_METRICS[metric]
        queues = per_queue_stats[metric]
        probs = np.array([q['prob'] for q in queues], dtype=float)
        stds = np.array([q['std'] for q in queues], dtype=float)
        sizes = np.array([max(q['sampleSize'], 1) for q in queues], dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            # log(0) would mean a segment that drops/marks everything; NaN is the honest
            # answer there rather than -inf propagating through the whole check.
            log_probs = np.where(probs > 0, np.log(np.clip(probs, 1e-300, None)), np.nan)
            epsilons = np.where(probs > 0, confidenceValue * stds / (np.sqrt(sizes) * probs), np.nan)
        stats[spec['mean_key']] = float(np.sum(log_probs)) if np.all(np.isfinite(log_probs)) else np.nan
        stats[spec['epsilon_key']] = float(np.max(epsilons)) if np.all(np.isfinite(epsilons)) else np.nan
        stats[spec['std_key']] = path_prob_product_std(probs, stds)
        stats[metric + 'PerSegment'] = [float(p) for p in probs]
        # The path probability itself, for reporting next to an e2e estimate of the same
        # thing (the log-space sum above is what the check consumes).
        stats[metric + 'PathProb'] = float(np.exp(stats[spec['mean_key']])) if np.isfinite(
            stats[spec['mean_key']]) else np.nan
    return stats


def prob_consistency_band(agg_stats, metric, sample_size, confidenceValue, e2e_prob,
                           number_of_segments=3):
    """The acceptance band the probability consistency check applies, as
    (log_diff, lower, upper): the check passes exactly when
    `lower <= log_diff <= upper`, where

        log_diff = log(e2e probability) - SUM of log(per-segment probability)
        upper    =  m * log(1 + MaxEpsilon) - log(1 - epsp)
        lower    =  m * log(1 - MaxEpsilon) - log(1 + epsp)
        epsp     =  eta * e2eStd / (e2e probability * sqrt(n))

    i.e. the switch side's own worst-segment relative error compounded over the m segments,
    widened by the e2e side's own relative error at n samples. `e2eStd` here is the path
    probability's std under segment independence (path_prob_product_std), since a path
    probability is a product of its segments' -- not the sum of their stds, which is the
    delay side's convention for a quantity that really is a sum. This is
    check_all_successProbConsistency / check_all_nonMarkingProbConsistency's
    'event_poisson_eventAvg' branch, lifted out so the band can be reported and plotted
    rather than only applied.

    Unlike the delay bound this band is **asymmetric** (a multiplicative band is symmetric
    in ratio, not in difference) and it is in log space, which is why the two are reported
    separately rather than squeezed into one "+/- bound" column.

    Returns (nan, nan, nan) when there is nothing to test -- no samples, a degenerate e2e
    probability, or switch statistics that carry no information about this metric."""
    spec = PROB_METRICS[metric]
    if agg_stats is None or not sample_size or sample_size <= 0:
        return np.nan, np.nan, np.nan
    switch_log = agg_stats.get(spec['mean_key'], np.nan)
    max_eps = agg_stats.get(spec['epsilon_key'], np.nan)
    switch_std = agg_stats.get(spec['std_key'], np.nan)
    if not (np.isfinite(switch_log) and np.isfinite(max_eps) and np.isfinite(switch_std)):
        return np.nan, np.nan, np.nan
    if not np.isfinite(e2e_prob) or e2e_prob <= 0:
        return np.nan, np.nan, np.nan
    epsp = confidenceValue * switch_std / (e2e_prob * np.sqrt(sample_size))
    if max_eps >= 1 or epsp >= 1:
        return np.nan, np.nan, np.nan
    log_diff = float(np.log(e2e_prob) - switch_log)
    upper = float(number_of_segments * np.log(1 + max_eps) - np.log(1 - epsp))
    lower = float(number_of_segments * np.log(1 - max_eps) - np.log(1 + epsp))
    if upper - lower <= 0:
        # A zero-width band means neither side of the comparison carries any variance: on
        # these traces that is exactly what the success probability looks like (RED reports
        # no drop probability anywhere, and the flow loses no packets on the path), and the
        # metric's "verdict" would then be decided by whether two numbers that are both
        # exactly 1 differ in the last floating-point bit. That is not a test, so it is
        # reported as untestable (NaN band) rather than as a pass.
        return log_diff, np.nan, np.nan
    return log_diff, lower, upper


def prob_consistency_check(agg_stats, metric, values, confidenceValue, min_sample_size,
                            number_of_segments=3):
    """Whether a family's per-packet 0/1 outcomes are consistent with the switch-side
    per-segment probabilities, by the band in prob_consistency_band. Returns
    (passed, e2e_prob, sample_size, log_diff, lower, upper); `passed` is None when no test
    could be made (too few samples, or no usable statistics) rather than False, so a
    "could not test" run is never counted as a failure -- the same convention the
    Poisson-adaptive families' delay checks use."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    sample_size = int(values.size)
    if sample_size < min_sample_size:
        return None, np.nan, sample_size, np.nan, np.nan, np.nan
    e2e_prob = float(np.mean(values))
    log_diff, lower, upper = prob_consistency_band(
        agg_stats, metric, sample_size, confidenceValue, e2e_prob, number_of_segments)
    if not (np.isfinite(log_diff) and np.isfinite(lower) and np.isfinite(upper)):
        return None, e2e_prob, sample_size, log_diff, lower, upper
    return bool(lower <= log_diff <= upper), e2e_prob, sample_size, log_diff, lower, upper


# The per-family record one probability metric produces, in the order the text summary
# and the plots read it.
PROB_FAMILY_FIELDS = ('prob', 'distance', 'distance_normalized', 'consistency_pass',
                       'log_diff', 'band_lower', 'band_upper', 'sample_size')


def _prob_pass_rates(pass_counts, verdicts_by_k):
    """Pass rate per flow count for one probability family: passes over the runs that could
    actually be tested (a verdict of None means the band was untestable -- see
    prob_consistency_band -- and such a run neither passed nor failed), the same
    denominator convention pass_rate_sampled uses for the delay families."""
    rates = []
    for count, verdicts in zip(pass_counts, verdicts_by_k):
        testable = sum(1 for v in verdicts if v is not None)
        rates.append((count / testable) if testable else 0.0)
    return rates


def evaluate_prob_estimate(e2e_prob, sample_size, groundtruth_prob, agg_stats, metric,
                            confidenceValue, min_sample_size, number_of_segments=3):
    """Score one already-formed probability estimate against the switch side and the ground
    truth -- the probability counterpart of _evaluate_delay_family, used both for families
    estimated from packets (evaluate_prob_family) and for the ideal Poisson probe, whose
    estimate comes from the switch trace rather than from packets.

    `distance` is |ground truth - estimate|, which for a two-point (Bernoulli) distribution
    IS the Wasserstein distance the delay side reports as EMD -- W1(Bern(p), Bern(q)) =
    |p - q| exactly -- so it is the same quantity under the same name, and there is nothing
    else to a 0/1 variable's distribution (no percentile or CDF counterpart).
    `distance_normalized` is its relative form, EMD / reference probability, the direct
    counterpart of the delay side's relEMD (EMD / mean ground-truth delay): both divide the
    distance by the ground truth's own mean, which for a Bernoulli is its probability."""
    if sample_size < min_sample_size or not np.isfinite(e2e_prob):
        return {'prob': e2e_prob, 'distance': np.nan, 'distance_normalized': np.nan,
                'consistency_pass': None, 'log_diff': np.nan, 'band_lower': np.nan,
                'band_upper': np.nan, 'sample_size': int(sample_size)}
    log_diff, lower, upper = prob_consistency_band(
        agg_stats, metric, sample_size, confidenceValue, e2e_prob, number_of_segments)
    passed = (bool(lower <= log_diff <= upper)
               if np.isfinite(log_diff) and np.isfinite(lower) and np.isfinite(upper) else None)
    distance = (abs(float(groundtruth_prob) - e2e_prob)
                 if np.isfinite(groundtruth_prob) else np.nan)
    return {'prob': e2e_prob, 'distance': distance,
            'distance_normalized': normalize_emd_values(distance, groundtruth_prob),
            'consistency_pass': passed, 'log_diff': log_diff, 'band_lower': lower,
            'band_upper': upper, 'sample_size': int(sample_size)}


def evaluate_prob_family(values, groundtruth_prob, agg_stats, metric, confidenceValue,
                          min_sample_size, number_of_segments=3):
    """Score one family's per-packet 0/1 outcomes for one probability metric: the estimate
    is their mean and the sample size is how many of them there were. See
    evaluate_prob_estimate for what comes back."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    e2e_prob = float(np.mean(values)) if values.size else np.nan
    return evaluate_prob_estimate(e2e_prob, int(values.size), groundtruth_prob, agg_stats,
                                   metric, confidenceValue, min_sample_size, number_of_segments)


def construct_oracle_path_probs(queue_names, dir_prefix, steady_start, steady_end,
                                 link_delays, link_rates, target_count):
    """The path probabilities an **ideal Poisson probe** with about `target_count`
    observations would report -- the probability counterpart of
    construct_oracle_poisson_delays, and the same ceiling argument: its instants are a
    genuine Poisson process independent of queue state, so its only error is finite-sample
    noise. Returns {metric: probability}."""
    target_count = int(target_count)
    if target_count <= 0 or steady_end <= steady_start:
        return {metric: np.nan for metric in PROB_METRIC_KEYS}
    return construct_path_prob_ground_truth(
        queue_names, dir_prefix, steady_start, steady_end, link_delays, link_rates,
        sample_interval_ns=(steady_end - steady_start) / target_count)

def _delay_consistency_check(values, agg_stats, confidenceValue, min_sample_size):
    """Maximum-Epsilon inequality delay consistency check (same formula used
    for the full flow's 'event_poisson_eventAvg' check), applied to any set
    of per-packet delay values: does the mean of `values` fall within the
    per-segment aggregated delay bound `agg_stats`?"""
    if len(values) < min_sample_size:
        return False
    sample_mean = np.mean(values)
    epsilon_bound = agg_stats['DelayMean'] * agg_stats['MaxEpsilonDelay']
    epsilon_bound += confidenceValue * agg_stats['e2eDelayStd'] / np.sqrt(len(values))
    return bool(abs(sample_mean - agg_stats['DelayMean']) <= epsilon_bound)


def delay_consistency_error_bound(agg_stats, sample_size, confidenceValue):
    """The consistency check's own error bound at a given e2e sample size, as
    (absolute_ns, relative_to_switch_mean).

    This is exactly the threshold _delay_consistency_check compares |switch mean - packet
    mean| against, pulled out so it can be reported rather than only applied:

        absolute = DelayMean * MaxEpsilonDelay + eta * e2eDelayStd / sqrt(n)
        relative = absolute / DelayMean
                 = MaxEpsilonDelay + eta * e2eDelayStd / (sqrt(n) * DelayMean)

    The first term is the switch side's own sampling error (MaxEpsilonDelay, the largest
    per-segment relative epsilon), the second the e2e side's at n samples.

    Worth reporting because of how the required sample size is chosen: calc_min_e2e_samples
    solves the second term down to exactly (maxError - MaxEpsilonDelay), so a family holding
    **exactly** the minimum required number of samples has a relative bound of exactly
    `maxError` -- the DelayConsistencyGaurantee the run was configured with. Plotting this
    against that guarantee is therefore the direct check that the guarantee the numbers
    claim is the one actually delivered. Two known, benign departures from landing exactly
    on it:

    - calc_min_e2e_samples truncates its n to an integer, and flooring n loosens the bound
      by the fraction of a sample thrown away: at most ~1/(2n) of the e2e term, so a hair
      *above* the guarantee. Observed on real runs at n~110: 0.4001-0.4012 against a 0.40
      guarantee, i.e. within 0.3%.
    - when the formula asks for fewer samples than MINIMUM_E2E_SAMPLE_SIZE, that floor
      forces more, and the bound comes out materially *tighter* than the guarantee (e.g.
      0.30 instead of 0.40 where the formula wanted 51 samples and the floor gave 100).

    Anything else below the line means that family simply retained more than the minimum;
    anything materially above it means the check ran at a looser bound than the run claims,
    which should not happen.

    Returns (nan, nan) when there is nothing to bound (no samples, or no statistics because
    the run certified no window)."""
    if agg_stats is None or not sample_size or sample_size <= 0:
        return np.nan, np.nan
    switch_mean = agg_stats.get('DelayMean', np.nan)
    absolute = (switch_mean * agg_stats.get('MaxEpsilonDelay', np.nan)
                 + confidenceValue * agg_stats.get('e2eDelayStd', np.nan) / np.sqrt(sample_size))
    relative = absolute / switch_mean if switch_mean else np.nan
    return float(absolute), float(relative)


def sample_uniform_count(subset_sorted_by_time, target_count):
    """Systematic uniform packet sampling of `subset_sorted_by_time` (which must
    already be sorted by SentTime) returning **exactly** `target_count` rows --
    the rate-matched counterpart of sFlow-style fixed-rate sampling.

    The sampling period is `N / target_count`, which is generally fractional, and
    the phase is drawn fresh each call from [0, period) so repeated calls (one per
    run) see a different subset. A fractional period is what makes the count exact:
    an integer "1-in-stride" rule can only hit counts of the form floor(N/stride)
    and would overshoot a requested count by up to ~stride/N (e.g. N=1000,
    target=300 gives stride=3 and 334 samples, 11% too many). Rows are still
    evenly spaced in index, so this keeps the "blind, non-adaptive, fixed-rate"
    character that makes uniform sampling the baseline of interest -- it just
    spends exactly the sample budget it was given.

    `target_count` >= N returns every row (the sampler cannot invent packets), and
    a non-positive count or empty input returns no rows.
    """
    n = len(subset_sorted_by_time)
    target_count = int(target_count)
    if n == 0 or target_count <= 0:
        return subset_sorted_by_time.iloc[0:0]
    if target_count >= n:
        return subset_sorted_by_time
    period = n / target_count
    phase = np.random.uniform(0.0, period)
    # period >= 1 here, so these indices are strictly increasing and all < n.
    positions = np.floor(phase + np.arange(target_count) * period).astype(int)
    return subset_sorted_by_time.iloc[positions]


def matched_uniform_target_count(sampled_size, min_samples):
    """How many packets the rate-matched uniform baseline should draw so that it
    is compared against a Poisson-adaptive method at the *same sample size*.

    Normally that is simply `sampled_size`, the count that method actually
    retained: EMD and the consistency-check bound both tighten with sample size,
    so a uniform baseline drawing a different number of packets would confound
    "which selection rule is better" with "which one kept more packets". Matching
    the count isolates the selection rule.

    When the Poisson-adaptive method found no valid subsample at all
    (`sampled_size == 0`) there is no count to match, so the budget falls back to
    `min_samples` -- the minimum sample size that method was required to reach
    (agg_stats['MinimumE2ESampleSizeDelay'], the same figure handed to it as
    MinimumNumberOfSamples). That answers the natural question at those flow
    counts: what would blind uniform sampling have produced with the sample
    budget the adaptive method was asked for and could not deliver?
    """
    if sampled_size and sampled_size > 0:
        return int(sampled_size)
    try:
        return max(0, int(np.ceil(float(min_samples))))
    except (TypeError, ValueError):
        return 0


POISSON_TEST_NAMES = ('ad', 'ad_chi')

_POISSON_TEST_LABELS = {
    'ad': 'Anderson-Darling (exponential gaps)',
    'ad_chi': 'Anderson-Darling + multi-lag chi-squared (independence)',
}


def poisson_test_label(test_name):
    """Human-readable name of a Poisson-ness test combination (POISSON_TEST_NAMES)."""
    return _POISSON_TEST_LABELS.get(test_name, str(test_name))


def poisson_process_tests(times, steady_start=None, steady_end=None, lags=None,
                           run_chi_squared=True):
    """Test whether a set of sampling instants looks like a Poisson process, using exactly
    the two criteria the Poisson-adaptive samplers validate themselves against:

      - **Anderson-Darling** on the inter-arrival gaps against an exponential: the marginal
        shape test find_samples_path uses (pass = p > 0.05).
      - **Multi-lag chi-squared** independence of arrivals across lags (chi_squared_test):
        the test find_samples_path_intensity adds, since AD alone passes streams whose gaps
        are individually exponential-looking but serially dependent. Pass = the fraction of
        lags rejecting independence stays under 0.05 plus its own 95% binomial band, the
        same rule find_samples_path_intensity applies.

    The point of running these on families that are *not* Poissonized -- all packets, and
    the fixed-rate uniform subsets -- is that those families never had to pass anything: it
    says whether their instants happen to look Poisson anyway, which is the premise PASTA
    needs before their sample mean can stand in for a time average.

    `steady_start`/`steady_end` default to the span of `times` itself, matching
    find_samples_path_intensity's convention. Set `run_chi_squared=False` to skip the
    chi-squared part, which is by far the more expensive of the two (~1s per call: it bins
    the whole window at 120ns and sweeps 511 lags); 'chi_pass' is then None.

    Returns a dict with 'ad_pass', 'ad_pvalue', 'chi_pass', 'chi_reject_fraction',
    'chi_upper_band', and 'both_pass' (AD and chi together; None when chi was skipped).
    Fewer than 3 instants cannot be tested at all and come back as failures.
    """
    result = {'ad_pass': False, 'ad_pvalue': np.nan, 'chi_pass': None,
               'chi_reject_fraction': np.nan, 'chi_upper_band': np.nan, 'both_pass': None}
    times = np.sort(np.asarray(times, dtype=float).reshape(-1))
    times = times[np.isfinite(times)]
    if times.size < 3:
        result['both_pass'] = False if run_chi_squared else None
        return result

    gaps = np.diff(times)
    if not np.any(gaps > 0):
        result['both_pass'] = False if run_chi_squared else None
        return result
    try:
        ad_res = anderson(gaps, 'expon', method='interpolate')
        result['ad_pvalue'] = float(ad_res.pvalue)
        result['ad_pass'] = bool(ad_res.pvalue > 0.05)
    except (ValueError, ZeroDivisionError):
        # A degenerate gap distribution is not exponential; treat it as a failed test
        # rather than letting one family abort a whole run.
        result['ad_pass'] = False

    if not run_chi_squared:
        return result

    if steady_start is None:
        steady_start = times[0]
    if steady_end is None:
        steady_end = times[-1]
    try:
        test_lags, rejects, _ = chi_squared_test(times, steady_start, steady_end, lags=lags)
    except (ValueError, ZeroDivisionError):
        test_lags, rejects = [], []
    if len(test_lags) == 0:
        result['chi_pass'] = False
    else:
        upper_band = 0.05 + 1.96 * np.sqrt(0.95 * 0.05) / np.sqrt(len(test_lags))
        reject_fraction = sum(rejects) / len(test_lags)
        result['chi_reject_fraction'] = float(reject_fraction)
        result['chi_upper_band'] = float(upper_band)
        result['chi_pass'] = bool(reject_fraction < upper_band)
    result['both_pass'] = bool(result['ad_pass'] and result['chi_pass'])
    return result


def _all_packets_test_flags(results, i, test_name):
    """The all-packet family's Poisson-test verdicts at flow-count index `i`, as a list
    aligned with that family's plotted values: one entry per experiment for an aggregated
    result, or a single entry for one experiment (where the all-packet instants are the
    same fixed set every run, so there is one verdict, not one per run). Empty when the
    family was not tested. Entries may be None where a test was not evaluated."""
    tests = results.get('poisson_tests_all_packets') or {}
    if not tests or 'ad_pass' not in tests:
        return []
    ad, chi = tests['ad_pass'][i], tests['chi_pass'][i]
    pairs = list(zip(ad, chi)) if isinstance(ad, (list, tuple)) else [(ad, chi)]
    return [poisson_test_outcome({'ad_pass': a, 'chi_pass': c}, test_name) for a, c in pairs]


def _all_packets_plot_values(results, i, value_key, by_experiment_key):
    """The all-packet family's plotted values at flow-count index `i`, as a list aligned
    with _all_packets_test_flags: the per-experiment values when aggregated, else the
    single value wrapped in a list."""
    by_experiment = results.get(by_experiment_key)
    if by_experiment:
        return list(by_experiment[i])
    return [results[value_key][i]]


def poisson_test_outcome(tests, test_name):
    """Whether one recorded poisson_process_tests result passes the named test
    combination ('ad' -> Anderson-Darling only; 'ad_chi' -> both). None when that
    combination was not evaluated (e.g. chi-squared was skipped)."""
    if test_name == 'ad':
        return tests.get('ad_pass')
    if test_name == 'ad_chi':
        if tests.get('chi_pass') is None:
            return None
        return bool(tests.get('ad_pass') and tests.get('chi_pass'))
    raise ValueError("Unknown Poisson test name {!r}; choose one of {}".format(
        test_name, list(POISSON_TEST_NAMES)))


def _uniform_series_label(series_key):
    """Legend/table name of a uniform-sampling family. New results key these by
    the Poisson-adaptive method whose sample count they match (a string); results
    pickles written before rate matching keyed them by a fixed integer stride, and
    still plot, so both shapes are named here."""
    if isinstance(series_key, str):
        return 'Rate-matched uniform ({} sample count)'.format(series_key)
    return 'Uniform 1-in-{} subsample'.format(series_key)


def _evaluate_delay_family(values, groundtruth_values, agg_stats, confidenceValue, min_sample_size):
    """Shared per-family evaluation used for every non-fixed subsampling
    method (Poisson-adaptive, uniform-stride, ...): the EMD of `values`
    against the ground-truth CDF, the delay-consistency check outcome, and
    the signed switch-vs-packet mean delay difference. Returns
    (emd, consistency_pass, mean_diff, sample_size)."""
    sample_size = len(values)
    if sample_size and len(groundtruth_values):
        emd = wasserstein_distance(groundtruth_values, values)
    else:
        emd = np.nan
    consistency_pass = _delay_consistency_check(values, agg_stats, confidenceValue, min_sample_size)
    mean_diff = agg_stats['DelayMean'] - np.mean(values) if sample_size else np.nan
    return emd, consistency_pass, mean_diff, sample_size


def generate_poisson_observation_times(steadyStart, steadyEnd, num_observations):
    """Draw the observation instants of one homogeneous Poisson-process
    realization over [steadyStart, steadyEnd] targeting `num_observations`
    points, generated the same way as elsewhere in this file (see
    e2e_poisson_sampling, calculate_offline_delay_bias_DC): i.i.d. exponential
    inter-arrival times -- mean interval = duration / num_observations --
    cumulatively summed from steadyStart. The realized point count fluctuates
    around num_observations rather than being fixed exactly, which is the
    correct behavior of an actual Poisson probe.
    """
    duration = steadyEnd - steadyStart
    mean_interval = duration / num_observations
    times = steadyStart + np.cumsum(np.random.exponential(mean_interval, size=int(num_observations)))
    return times[times <= steadyEnd]


def preload_queue_traces(dir_prefix, queue_names):
    """Eagerly parse and cache each queue's recorded size trace (see
    _load_queue_trace, which memoizes by file path) so that every later call
    to sample_queue_size for these queues -- once per run in
    compute_poisson_agg_stats, plus construct_path_delay_distribution's
    ground-truth reconstruction -- reuses the already-parsed trace instead of
    re-reading its CSV file. Call once before looping over runs; on Linux,
    fork()'d worker processes inherit the warmed cache too."""
    prefix = str(dir_prefix)
    if not prefix.endswith("/"):
        prefix += "/"
    for queue_name in queue_names:
        _load_queue_trace(prefix + queue_name + '_PoissonSampler_queueSize.csv')


def compute_poisson_agg_stats(dir_prefix, queue_names, linkDelays, linkRates, steadyStart, steadyEnd,
                               num_observations, confidenceValue, DelayConsistencyGaurantee):
    """Draw one Poisson-process realization of `num_observations` observation
    instants over [steadyStart, steadyEnd] and use it to sample every queue
    on the path *at those same simultaneous instants* (same convention as
    construct_path_delay_distribution), producing a fresh, finite-sample
    estimate of the per-segment aggregated delay statistics that feed the
    consistency check -- in place of reading the whole recorded
    PoissonSampler_events log, which used a much larger, effectively fixed
    sample. This mimics what a real deployment's Poisson probe would see
    with only `num_observations` samples per run.
    """
    times = generate_poisson_observation_times(steadyStart, steadyEnd, num_observations)
    ordered_queues, _, ordered_rates = sort_queues_by_path(queue_names, linkDelays, linkRates)

    queue_stats = {}
    for queue_name, link_rate in zip(ordered_queues, ordered_rates):
        queue_size_samples = sample_queue_size(times, dir_prefix + queue_name + '_PoissonSampler_queueSize.csv', link_rate)
        valid = np.isfinite(queue_size_samples)
        delay_samples = sample_queueing_delay(queue_size_samples[valid], link_rate)
        queue_stats[queue_name] = {
            'DelayMean': float(np.mean(delay_samples)) if delay_samples.size else 0.0,
            'DelayStd': float(np.std(delay_samples)) if delay_samples.size else 0.0,
            'sampleSize': int(delay_samples.size),
        }

    agg_stats = {}
    agg_stats['DelayMean'] = sum(queue_stats[q]['DelayMean'] for q in ordered_queues)
    agg_stats['MaxEpsilonDelay'] = max(calc_epsilon(confidenceValue, queue_stats[q]) for q in ordered_queues)
    agg_stats['e2eDelayStd'] = sum(queue_stats[q]['DelayStd'] for q in ordered_queues)
    agg_stats['MinimumE2ESampleSizeDelay'] = calc_min_e2e_samples(confidenceValue, DelayConsistencyGaurantee, agg_stats, metric='Delay')
    # The same probe instants also carry the loss/marking side of the check
    # (sample_path_prob_stats). Deliberately no separate minimum sample size for those: the
    # delay figure above is the one every family is sized by, since a 0/1 outcome varies far
    # less than a delay and the delay-derived count is the conservative choice.
    agg_stats.update(sample_path_prob_stats(
        times, queue_names, dir_prefix, linkDelays, linkRates, confidenceValue))
    return agg_stats


def _flow_count_values(total_flows, step, all_flows_only=False):
    """The list of flow counts (k) at which the EMD-vs-flows sweep is
    evaluated: 1, 1+step, 1+2*step, ..., always ending at `total_flows` (even
    if it doesn't fall on the step) so the full-flow-count point is never
    skipped. step<=1 evaluates every k, matching the original behavior.
    Coarsening this is the main lever on the per-run cost, since
    the subsampling search (the dominant cost) runs once per k per run per
    method.

    `all_flows_only` collapses the sweep to the single point k=`total_flows`:
    all flows on the path, i.e. every received e2e packet. That is the headline
    configuration (the same one the k='max' plots single out), and evaluating
    only it is far and away the cheapest way to run the pipeline -- the per-run
    cost drops by roughly the number of k values the sweep would otherwise
    have had."""
    if total_flows <= 0:
        return []
    if all_flows_only:
        return [total_flows]
    step = max(1, int(step))
    if step <= 1:
        return list(range(1, total_flows + 1))
    values = list(range(1, total_flows + 1, step))
    if values[-1] != total_flows:
        values.append(total_flows)
    return values


DEFAULT_DELAY_PERCENTILES = (90, 99)


def compute_delay_percentiles(values, percentiles):
    """The requested percentiles of a set of delay values, as a {q: value} dict
    (NaN per entry when there is nothing to take a percentile of). Non-finite
    entries are dropped first, matching how the EMD path treats them."""
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {q: np.nan for q in percentiles}
    computed = np.percentile(values, list(percentiles))
    return {q: float(v) for q, v in zip(percentiles, np.atleast_1d(computed))}


def percentile_diffs(values, groundtruth_percentiles):
    """Signed **absolute** tail-shape error of one comparison family, in ns:
    `groundtruth_percentile - family_percentile` at each requested percentile,
    as a {q: diff} dict.

    Sign convention matches the mean-delay difference this pipeline already
    reports (switch/ground-truth minus packet-side, see
    compute_emd_vs_num_tcp_flows_run), so a **positive** value means the family
    *under*-states that percentile -- it is missing tail delay the ground truth
    has -- and negative means it overstates it.

    Why this is worth having next to the EMD: EMD is a single number summarizing
    the whole distribution, so a family can score well on it while still getting
    the tail wrong, and the tail (p90/p99) is what actually matters for delay
    SLOs. NaN wherever either side has no value at that percentile."""
    family = compute_delay_percentiles(values, groundtruth_percentiles.keys())
    return {q: (groundtruth_percentiles[q] - family[q]) for q in groundtruth_percentiles}


def relative_percentile_diffs(absolute_diffs, groundtruth_percentiles):
    """The same tail-shape errors as a fraction of the ground truth's own
    percentile: `(gt_q - family_q) / gt_q`, so 0.1 reads as "this family
    understates the qth percentile by 10%". Like the normalized EMD, this is
    what stays comparable across offered loads, since the absolute ns gap grows
    with the delay level that load itself drives.

    `absolute_diffs` is a {q: <nested structure>} dict as produced by
    percentile_diffs (or a whole per-k/per-run structure of them); each
    percentile is divided by *its own* ground-truth percentile."""
    return {q: scale_nested_values(diffs, groundtruth_percentiles.get(q))
             for q, diffs in absolute_diffs.items()}


# percentile_avg_relative_error (below) reports mean absolute percentage error across a
# dense, fixed percentile grid, independent of whatever `delay_percentiles` a run happens
# to be configured with (that is a separate, sparse set used only for the tail-shape-error
# reporting a few percentiles at a time -- percentile_diffs/relative_percentile_diffs).
#
# Note: a percentile-grid *EMD* estimate (mean |Q_ref(q) - Q_family(q)| over this same
# grid, before the relative-error division below) was tried and deliberately removed --
# Wasserstein-1 equals exactly this integral in the limit of infinite grid resolution
# (Vallender 1974), but at any fixed finite resolution it is only an approximation of the
# exact value normalize_emd_values(wasserstein_distance(...), mean) already gives for
# free at the same O(n log n) cost. There is no real-data scenario in this pipeline where
# the approximation is cheaper or more accurate than just using the real (rel)EMD, so
# relEMD remains the metric of record; only the self-normalized relative-error form below
# is kept as a distinct, additional metric.
_EMD_PERCENTILE_GRID = tuple(range(1, 100))  # every integer percentile 1..99


def _finite_1d(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    return values[np.isfinite(values)]


def percentile_avg_relative_error(reference_values, family_values):
    """Mean absolute percentage error between `reference_values` and `family_values`,
    averaged over a dense, fixed percentile grid (_EMD_PERCENTILE_GRID, 1st through
    99th): the mean, over q = 1%..99%, of |Q_ref(q) - Q_family(q)| / Q_ref(q).
    Self-normalized (each percentile scaled by its own reference value). Percentiles
    where the reference value is <= 0 are excluded from the average (a relative error
    there is undefined) rather than forcing the whole thing to NaN; the result is NaN
    only if every percentile is excluded or either side is empty."""
    reference_values = _finite_1d(reference_values)
    family_values = _finite_1d(family_values)
    if reference_values.size == 0 or family_values.size == 0:
        return np.nan
    ref_q = np.percentile(reference_values, _EMD_PERCENTILE_GRID)
    fam_q = np.percentile(family_values, _EMD_PERCENTILE_GRID)
    valid = ref_q > 0
    if not np.any(valid):
        return np.nan
    return float(np.mean(np.abs(ref_q[valid] - fam_q[valid]) / ref_q[valid]))


ORACLE_MIN_REQUIRED_KEY = 'min_required'
# The ideal-Poisson-probe family run at the SAME rate as "all packets of the considered
# flows" -- i.e. what a perfectly Poisson process would look like at the sample count/rate
# the no-subsampling ceiling itself draws, as a reference point alongside the
# subsampling-method-matched probes (see oracle_target_counts).
ORACLE_ALL_PACKETS_RATE_KEY = 'all_packets_rate'


def _oracle_series_label(series_key):
    """Legend/table name of an ideal-Poisson-probe family. Keyed by
    ORACLE_MIN_REQUIRED_KEY (the probe run at the minimum sample rate the consistency check
    demands), ORACLE_ALL_PACKETS_RATE_KEY (matched to all packets' own count), or the
    Poisson-adaptive method whose retained sample count it matches."""
    if series_key == ORACLE_MIN_REQUIRED_KEY:
        return 'Ideal Poisson probe (minimum required samples)'
    if series_key == ORACLE_ALL_PACKETS_RATE_KEY:
        return 'Ideal Poisson probe (all-packets rate)'
    return 'Ideal Poisson probe ({} sample count)'.format(series_key)


def construct_oracle_poisson_delays(groundtruth_method, queue_names, dir_prefix, steady_start,
                                     steady_end, link_delays, link_rates, target_count):
    """Path delays seen by an **imaginary, perfectly Poisson** probe: exactly the
    ground-truth construction (GROUNDTRUTH_METHODS -- so with
    'path_observation' the probe arrives at the first queue at Poisson instants
    and waits out each queue's delay before observing the next), but run at a
    realistic measurement rate of about `target_count` observations instead of
    the ground truth's ~million.

    This is the *ceiling* for any Poissonization scheme, and the reason it is
    worth plotting next to the real subsampling families. Its sampling instants
    are a genuine Poisson process generated independently of queue state, so
    PASTA holds exactly and there is no selection bias by construction -- the
    only thing separating it from the ground truth is finite-sample noise at
    that sample size. So the gap between a real Poissonized subsample and this
    family is the part of the error that is *not* explained by having few
    samples: it is what selecting from the flow's own packets costs (see the
    selection-bias discussion in find_samples_path_intensity).

    The rate is set so the count is about `target_count`
    (interval = duration / target_count) rather than exactly it: a Poisson
    process observed over a fixed window has a random number of points, and
    pinning the count would make the instants uniform order statistics rather
    than a Poisson process -- the very property being demonstrated. In practice
    the realized count runs ~1-3% *under* the budget, because
    generate_poisson_observation_times draws exactly `target_count`
    inter-arrivals and discards any whose cumulative time overshoots the
    window; this is the same generator the ground truth itself uses, so the
    probe and the ground truth stay directly comparable. Realized counts are
    recorded per run ('sample_sizes_oracle_by_run') and reported in the text
    summary, so the achieved budget is never left implicit.
    """
    target_count = int(target_count)
    if target_count <= 0 or steady_end <= steady_start:
        return np.array([], dtype=float)
    construct = _resolve_groundtruth_method(groundtruth_method)
    return construct(
        queue_names, dir_prefix, steady_start, steady_end, link_delays, link_rates,
        sample_interval_ns=(steady_end - steady_start) / target_count,
    )


def oracle_target_counts(sampled_sizes_by_method, min_samples, all_packets_size=None):
    """The sample budget for each ideal-Poisson-probe family, as an ordered
    {key: target_count} dict:

      - ORACLE_MIN_REQUIRED_KEY -> the minimum sample size the consistency check
        demands (agg_stats['MinimumE2ESampleSizeDelay']): "what could a perfect
        Poisson probe do with the bare minimum budget?"
      - one entry per Poisson-adaptive method -> that method's own retained count
        at this flow count, so probe and method are compared at equal sample
        size, falling back to the required minimum where the method found no
        valid subsample (the same rule the rate-matched uniform baseline uses,
        matched_uniform_target_count).
      - ORACLE_ALL_PACKETS_RATE_KEY (only if `all_packets_size` is given) -> the
        all-packets family's own count at this flow count: "what would a perfectly
        Poisson process look like at the rate all packets arrive, with no
        subsampling at all?"
    """
    targets = {ORACLE_MIN_REQUIRED_KEY: matched_uniform_target_count(0, min_samples)}
    for method, size in sampled_sizes_by_method.items():
        targets[method] = matched_uniform_target_count(size, min_samples)
    if all_packets_size is not None:
        targets[ORACLE_ALL_PACKETS_RATE_KEY] = matched_uniform_target_count(all_packets_size, min_samples)
    return targets


def _empty_percentile_structure(percentiles, keys, num_k):
    """A {q: {key: [[] per k]}} skeleton -- the shape the per-run percentile-diff
    records take, with nothing recorded yet."""
    return {q: {key: [[] for _ in range(num_k)] for key in keys} for q in percentiles}


MSS_BYTES = 1500
# The reference round-trip time IDC(1RTT) is measured at, per the project's assumed DC RTT.
ONE_RTT_NS = 8000.0


def burst_gap_threshold_ns(host_link_rate_gbps):
    """Max inter-arrival gap (ns) for two arrivals to belong to the same burst: the
    transmission time of one full MSS (1500B) at the sender's own outgoing link rate
    (`host_link_rate_gbps`, in Gbit/s -- ECNMC's link-rate convention makes this numerically
    bit/ns, see linkRates in run_emd_vs_flows_experiment). Packets arriving no slower than
    back-to-back line rate from the source are considered part of the same burst."""
    return (MSS_BYTES * 8) / host_link_rate_gbps


def detect_bursts(times, gap_threshold):
    """Group arrival times (ns) into bursts: consecutive arrivals gapped by <= gap_threshold
    (ns) belong to the same burst (see burst_gap_threshold_ns for the threshold this project
    uses). Returns (starts, ends, sizes), one triple per burst, sorted by time -- a lone
    arrival not within gap_threshold of any neighbour is its own size-1, zero-duration burst
    (a degenerate case, not a "real" burst -- see burstiness_metrics, which excludes these
    from the duration average). Empty arrays if `times` is empty."""
    times = np.sort(np.asarray(times, dtype=float))
    n = len(times)
    if n == 0:
        return np.array([]), np.array([]), np.array([], dtype=int)
    is_new_burst = np.empty(n, dtype=bool)
    is_new_burst[0] = True
    is_new_burst[1:] = np.diff(times) > gap_threshold
    starts = times[is_new_burst]
    is_burst_end = np.empty(n, dtype=bool)
    is_burst_end[:-1] = is_new_burst[1:]
    is_burst_end[-1] = True
    ends = times[is_burst_end]
    burst_id = np.cumsum(is_new_burst) - 1
    sizes = np.bincount(burst_id)
    return starts, ends, sizes


def idc_at_delta(times, delta):
    """IDC(delta) = Var(N_delta)/E[N_delta] for one specific bin width `delta` (ns), reusing
    idc_curve's binning (non-overlapping bins spanning [min(times), max(times)]). NaN if
    `times` has fewer than 2 points, `delta` isn't positive, or idc_curve can't produce a
    value at this delta (e.g. mean bin count is 0)."""
    times = np.asarray(times, dtype=float)
    if len(times) < 2 or not np.isfinite(delta) or delta <= 0:
        return float('nan')
    try:
        _, idc_vals, _, _ = idc_curve(times, np.array([delta]))
    except (ValueError, RuntimeError):
        return float('nan')
    return float(idc_vals[0]) if len(idc_vals) else float('nan')


def burstiness_metrics(times, gap_threshold, rtt_ns=ONE_RTT_NS):
    """The three burstiness metrics for one arrival-time array `times` (ns): IDC at one RTT,
    mean burst duration, and mean inter-burst gap (see detect_bursts for the burst
    definition). NaN for any metric `times` has too little data for.

    `avg_burst_duration_ns` averages only over MULTI-packet bursts (size >= 2) -- a lone
    arrival is a degenerate, zero-duration "burst" by detect_bursts' bookkeeping, not a real
    one, and on most real traffic the large majority of bursts are lone arrivals (e.g.
    ~95-98% observed on Google_AllRPC), so including them would dilute the average toward 0
    and mostly measure how rare multi-packet bursts are rather than how long one lasts when
    it happens. NaN when there are no multi-packet bursts at all in `times`. Inter-burst gap
    does NOT have this issue (a gap is well-defined between any two bursts regardless of
    either one's size), so it still averages over all of them."""
    starts, ends, sizes = detect_bursts(times, gap_threshold)
    if len(starts) == 0:
        avg_duration, avg_interarrival = float('nan'), float('nan')
    else:
        multi = sizes > 1
        avg_duration = float(np.mean((ends - starts)[multi])) if multi.any() else float('nan')
        avg_interarrival = float(np.mean(starts[1:] - ends[:-1])) if len(starts) >= 2 else float('nan')
    return {
        'idc_1rtt': idc_at_delta(times, rtt_ns),
        'avg_burst_duration_ns': avg_duration,
        'avg_burst_interarrival_ns': avg_interarrival,
    }


BURSTINESS_METRIC_LABELS = {
    'idc_1rtt': 'IDC(1 RTT = {:g}ns)'.format(ONE_RTT_NS),
    'avg_burst_duration_ns': 'Avg burst duration (ns)',
    'avg_burst_interarrival_ns': 'Avg inter-burst gap (ns)',
}


def prepare_emd_vs_flows_data(
    ns3_path,
    results_folder,
    rate,
    load,
    experiment,
    flow_name,
    queue_names,
    linkDelays,
    linkRates,
    steadyStart,
    steadyEnd,
    path=0,
    delay_cdf_sample_interval_ns=10,
    max_num_flows=None,
    flow_count_step=1,
    all_flows_only=False,
    groundtruth_method='simultaneous',
    delay_percentiles=DEFAULT_DELAY_PERCENTILES,
    run_chi_squared_test=True,
    poisson_test_lags=None,
    differentiationDelay=None,
    errorRate=None,
):
    """Load and preprocess everything that stays fixed across repeated runs
    of the flow-count EMD sweep: the flow's received packets on `path`,
    grouped into TCP flows (SourceIp, SourcePort, DestinationIp,
    DestinationPort) ordered by first-seen SentTime; the ground-truth
    reconstructed network queuing delay CDF (see construct_path_delay_distribution);
    and the EMD between that ground truth and the all-packet CDF of the first
    k flows, for every k. None of these depend on the per-run Poisson
    probing, so all are computed once here and reused by every run of
    compute_emd_vs_num_tcp_flows_multi_run -- unlike the subsampled CDF,
    "all packets of the first k flows" is the same fixed set of packets on
    every run, so its EMD is a single number per k, not a distribution.

    Set `all_flows_only` to skip the flow-count sweep entirely and evaluate
    only k = all flows on the path (every received e2e packet) -- see
    _flow_count_values.

    Also runs the Poisson-ness tests (poisson_process_tests) on the all-packet
    arrival instants at every k. That family is never Poissonized, so whether
    its instants happen to look Poisson at all is exactly the premise PASTA
    needs -- and, like its EMD, it is the same fixed packet set every run, so
    the test outcome is computed once here rather than per run.

    `groundtruth_method` (one of GROUNDTRUTH_METHODS) selects how that
    ground-truth path-delay CDF is built: 'simultaneous' observes every queue
    at the same instant, 'path_observation' releases a probe that waits out
    each queue's delay before observing the next (see
    construct_path_delay_distribution_path_observation).

    Also returns `groundtruth_mean` -- the mean ground-truth path delay, which
    is what every EMD is normalized by (see normalize_emd_values): EMD is in ns
    and grows with the delay level, so the raw value is hard to compare across
    loads, whereas EMD / E[ground-truth delay] reads as a fraction of the true
    mean delay and is directly comparable.

    Also returns the ground truth's own `delay_percentiles` (default p90/p99)
    and, for every k, the all-packet family's signed absolute and relative
    error at each of them (percentile_diffs / relative_percentile_diffs) -- the
    tail-shape counterpart to the EMD, which a single distance number can hide.
    Like the all-packet EMD these are the same fixed packet set every run, so
    they are computed once here.

    Pass `differentiationDelay`/`errorRate` for the reverse (TBF-differentiation)
    experiments, whose raw per-experiment data sits one level deeper, under
    `D_<differentiationDelay>/f_<errorRate>/` (mirroring analyze_single_experiment /
    calculate_offline_computations_DC's own convention -- see Utils.py's
    calculate_offline_computations_DC). Note the `D_` folder there is actually named for
    whatever sweep parameter exp.py substitutes in (e.g. tbfFlowRedirectFraction for
    reverse_delay, not literally 'differentiationDelay'); the parameter is still called
    `differentiationDelay` here only to match that folder-naming convention already used
    throughout the rest of this file."""
    if differentiationDelay is not None and errorRate is not None:
        dir_prefix = '{}/scratch/{}/{}/{}/D_{}/f_{}/{}/'.format(
            ns3_path, results_folder, rate, load, differentiationDelay, errorRate, experiment)
    else:
        dir_prefix = '{}/scratch/{}/{}/{}/{}/'.format(ns3_path, results_folder, rate, load, experiment)
    file_path = dir_prefix + '{}_EndToEnd_packets.csv'.format(flow_name)

    preload_queue_traces(dir_prefix, queue_names)

    full_df = pd.read_csv(file_path)
    full_df = addRemoveTransmission_data(full_df, linkDelays, linkRates)
    full_df = prune_data(full_df, 'SentTime', steadyStart, steadyEnd)
    full_df = full_df[full_df['Path'] == path].copy()
    full_df = full_df.sort_values(by='SentTime').reset_index(drop=True)
    # Every SENT packet is kept, not only the received ones, because the success
    # probability cannot be estimated from received packets alone -- the estimate would be 1
    # by construction. The sampled stream is therefore the sender's, by SentTime, which is
    # also what the non-EMD path has always sampled (calculate_offline_computations_DC is
    # called with 'SentTime' and IsReceived as the outcome). Delay and marking are only
    # observable on packets that arrived, so those two families take the received subset of
    # whatever was selected (see received_rows); on this data that is 99.74% of it.
    full_df['Success'] = full_df['IsReceived'].astype(float)
    # A dropped packet counts as marked (NonMarked = 0), regardless of what its ECN column
    # says: every DC24Servers trace checked records ECN=0 on dropped packets (70/70, 23/23,
    # 8/8 across three traffic/load combinations), which if taken literally would count
    # them as having passed *unmarked* and bias the non-marking probability upward. A
    # packet dropped by a full queue was necessarily above the ECN threshold, so the
    # semantically correct value is 1 (marked) and it is forced here rather than read.
    full_df['NonMarked'] = np.where(full_df['IsReceived'] == 1, 1.0 - full_df['ECN'], 0.0)

    full_df['FlowKey'] = list(zip(full_df['SourceIp'], full_df['SourcePort'], full_df['DestinationIp'], full_df['DestinationPort']))
    flow_first_seen = full_df.groupby('FlowKey')['SentTime'].min().sort_values()
    flow_order = flow_first_seen.index.tolist()
    if max_num_flows is not None:
        flow_order = flow_order[:max_num_flows]
    flow_rank = {key: rank for rank, key in enumerate(flow_order, start=1)}
    full_df['FlowRank'] = full_df['FlowKey'].map(flow_rank)
    full_df = full_df[full_df['FlowRank'].notna()]

    construct_groundtruth = _resolve_groundtruth_method(groundtruth_method)
    groundtruth_values = construct_groundtruth(
        queue_names, dir_prefix, steadyStart, steadyEnd, linkDelays, linkRates,
        sample_interval_ns=delay_cdf_sample_interval_ns,
    )

    # Normalize 90.0 -> 90 so percentiles read as "p90" in filenames/tables and compare
    # equal as dict keys regardless of whether they arrived as int, float or CLI string.
    delay_percentiles = tuple(int(q) if float(q).is_integer() else float(q)
                               for q in (delay_percentiles or ()))
    groundtruth_percentiles = compute_delay_percentiles(groundtruth_values, delay_percentiles)

    num_flows = _flow_count_values(len(flow_order), flow_count_step, all_flows_only=all_flows_only)
    emd_all_packets, all_packet_sizes = [], []
    percentile_diff_all = {q: [] for q in delay_percentiles}
    percentile_avg_relerror_all = []
    poisson_tests_all = {'ad_pass': [], 'ad_pvalue': [], 'chi_pass': [], 'chi_reject_fraction': []}
    burst_gap = burst_gap_threshold_ns(linkRates[0])
    burstiness_all_packets = {field: [] for field in BURSTINESS_METRIC_LABELS}
    groundtruth_probs = construct_path_prob_ground_truth(
        queue_names, dir_prefix, steadyStart, steadyEnd, linkDelays, linkRates,
        sample_interval_ns=delay_cdf_sample_interval_ns)
    prob_all_packets = {metric: [] for metric in PROB_METRIC_KEYS}
    prob_all_packet_sizes = {metric: [] for metric in PROB_METRIC_KEYS}

    for k in num_flows:
        considered_sent = full_df[full_df['FlowRank'] <= k]
        considered = received_rows(considered_sent)
        for metric in PROB_METRIC_KEYS:
            metric_values = prob_metric_values(considered_sent, metric)
            prob_all_packets[metric].append(
                float(np.mean(metric_values)) if len(metric_values) else np.nan)
            prob_all_packet_sizes[metric].append(int(len(metric_values)))
        all_values = considered['Delay'].values
        all_packet_sizes.append(len(all_values))
        tests = poisson_process_tests(
            considered['SentTime'].values, steadyStart, steadyEnd,
            lags=poisson_test_lags, run_chi_squared=run_chi_squared_test)
        for field in poisson_tests_all:
            poisson_tests_all[field].append(tests[field])
        if len(all_values) and len(groundtruth_values):
            emd_all_packets.append(wasserstein_distance(groundtruth_values, all_values))
        else:
            emd_all_packets.append(np.nan)
        percentile_avg_relerror_all.append(percentile_avg_relative_error(groundtruth_values, all_values))
        diffs = percentile_diffs(all_values, groundtruth_percentiles)
        for q in delay_percentiles:
            percentile_diff_all[q].append(diffs[q])
        burst_metrics = burstiness_metrics(considered['SentTime'].values, burst_gap)
        for field in burstiness_all_packets:
            burstiness_all_packets[field].append(burst_metrics[field])

    return {
        'dir_prefix': dir_prefix,
        'full_df': full_df,
        # The path's true success / non-marking probability over the whole steady window,
        # and the all-packets estimate of each per flow count (see
        # construct_path_prob_ground_truth). A windowed run rebuilds both over its own
        # window instead.
        'groundtruth_probs': groundtruth_probs,
        'prob_all_packets': prob_all_packets,
        'prob_all_packet_sizes': prob_all_packet_sizes,
        'flow_order': flow_order,
        'num_flows': num_flows,
        'groundtruth_method': groundtruth_method,
        # Kept so a growing-window method can rebuild the same ground-truth construction
        # over its own shorter window at the same resolution (see windowed_groundtruth).
        'delay_cdf_sample_interval_ns': delay_cdf_sample_interval_ns,
        'groundtruth_values': groundtruth_values,
        'groundtruth_mean': float(np.mean(groundtruth_values)) if len(groundtruth_values) else np.nan,
        'groundtruth_std': float(np.std(groundtruth_values)) if len(groundtruth_values) else np.nan,
        'emd_all_packets': emd_all_packets,
        'all_packet_sizes': all_packet_sizes,
        'delay_percentiles': list(delay_percentiles),
        'groundtruth_percentiles': groundtruth_percentiles,
        'percentile_diff_all_packets': percentile_diff_all,
        'percentile_reldiff_all_packets': relative_percentile_diffs(
            percentile_diff_all, groundtruth_percentiles),
        # Mean absolute relative percentile error (see percentile_avg_relative_error):
        # self-normalized, evaluated via a dense percentile grid independent of
        # delay_percentiles above.
        'percentile_avg_relerror_all_packets': percentile_avg_relerror_all,
        'poisson_tests_all_packets': poisson_tests_all,
        'run_chi_squared_test': run_chi_squared_test,
        'poisson_test_lags': poisson_test_lags,
        # Burstiness of the all-packets arrival process itself (see burstiness_metrics):
        # IDC at one RTT, and mean burst duration/inter-burst gap using a burst defined as
        # consecutive SentTime arrivals no farther apart than one MSS's transmission time on
        # the sender's own outgoing link (burst_gap_threshold_ns). Same fixed packet set
        # every run, so computed once here like emd_all_packets.
        'burst_gap_threshold_ns': burst_gap,
        'burstiness_all_packets': burstiness_all_packets,
        # Kept so a run can rebuild the ground-truth construction at a lower rate for the
        # ideal-Poisson-probe family (construct_oracle_poisson_delays).
        'queue_names': list(queue_names),
        'link_delays': list(linkDelays),
        'link_rates': list(linkRates),
        'steady_start': steadyStart,
        'steady_end': steadyEnd,
        'flow_name': flow_name,
        'path': path,
    }


def scale_nested_values(values, divisor):
    """Divide every scalar in an arbitrarily nested list/dict structure by one
    scalar `divisor`, preserving the structure. Yields NaN wherever the divisor
    is missing, non-finite or non-positive rather than inventing a ratio, so a
    degenerate reference (an empty or all-zero ground truth) shows up as "no
    value" instead of an infinity. Used to turn absolute quantities into
    relative ones in one pass over a whole per-k (or per-k-per-run) structure --
    see normalize_emd_values and relative_percentile_diffs."""
    if isinstance(values, dict):
        return {key: scale_nested_values(value, divisor) for key, value in values.items()}
    if isinstance(values, (list, tuple)):
        return [scale_nested_values(value, divisor) for value in values]
    if divisor is None or not np.isfinite(divisor) or divisor <= 0:
        return np.nan
    if values is None or not np.isfinite(values):
        return np.nan
    return float(values) / float(divisor)


def normalize_emd_values(values, groundtruth_mean):
    """Scale raw EMD values (ns) into load-comparable normalized EMDs by
    dividing by the mean ground-truth path delay, so a value of 0.1 reads as
    "the mismatch is ~10% of the true mean network delay". Raw EMD grows with
    the delay level itself, which makes the absolute number hard to read across
    offered loads; this ratio does not. Nests through lists/dicts so a whole
    per-k (or per-k-per-run) structure can be normalized in one call, and
    yields NaN when the ground truth is empty or degenerate (mean <= 0) rather
    than inventing a ratio."""
    return scale_nested_values(values, groundtruth_mean)


POISSON_SUBSAMPLING_METHODS = {
    'find_samples_path': find_samples_path,
    'find_samples_path_intensity': find_samples_path_intensity,
    'find_samples_path_growing_window': find_samples_path_growing_window,
    'find_samples_path_intensity_growing_window': find_samples_path_intensity_growing_window,
}

# The methods that look for the shortest sufficient prefix of the steady window instead of
# sampling all of it (see _growing_window_search), mapped to the base sampler each one
# wraps. Membership here is what makes the EMD-vs-flows pipeline evaluate a method's whole
# comparison -- ground truth, switch-side statistics, rate-matched uniform baseline and
# ideal Poisson probe -- over the window that method actually stopped at, instead of over
# the full steady period (see compute_emd_vs_num_tcp_flows_run).
GROWING_WINDOW_SUBSAMPLING_METHODS = {
    'find_samples_path_growing_window': ('find_samples_path', False),
    'find_samples_path_intensity_growing_window': ('find_samples_path_intensity', True),
}

# The subset of POISSON_SUBSAMPLING_METHODS whose behaviour depends on the analysis window
# itself, not only on the packet timestamps handed to them, and which therefore get
# steadyStart/steadyEnd passed (see call_subsampling_method). Deliberately a whitelist
# rather than "pass it to anything that accepts the keyword": find_samples_path_intensity
# also takes steadyStart/steadyEnd and has always run with its own time[0]/time[-1]
# defaults, so signature-sniffing would silently change what already-published results for
# that method mean.
STEADY_WINDOW_AWARE_SUBSAMPLING_METHODS = frozenset(GROWING_WINDOW_SUBSAMPLING_METHODS)

# The methods built on find_samples_path, whose per-window bin width comes from
# find_delta_for_empty_prob and can therefore be precomputed once and reused (see
# precompute_subsample_deltas). find_samples_path_intensity needs no delta at all -- it
# thins by local intensity instead -- so nothing here applies to it.
FIND_DELTA_BASED_METHODS = frozenset(['find_samples_path', 'find_samples_path_growing_window'])


def growing_window_method_in(subsampling_methods):
    """The single growing-window method a run is analyzing, or None for an ordinary
    whole-steady-window run.

    A growing-window method must be the ONLY method in its run, and this is where that is
    enforced. The reason is the whole point of the mode: such a run reports nothing over the
    full steady window -- every family (all packets, the subsample, its rate-matched uniform
    baseline, every ideal Poisson probe) is evaluated over the window the search settled on,
    because comparing quantities measured over different windows is exactly the error this
    mode exists to avoid. Two growing-window methods stop at two different windows, so one
    run could not host both without reporting something over a window that some family in it
    never saw; and pairing a growing-window method with a whole-window one has the same
    problem in sharper form. Compare them by running the same experiment once per method --
    the packets, the traces and the ground-truth construction are identical inputs, and each
    method's output already lands under its own `<config_tag>` folder."""
    growing = [name for name in subsampling_methods if name in GROWING_WINDOW_SUBSAMPLING_METHODS]
    if not growing:
        return None
    if len(subsampling_methods) > 1:
        raise ValueError(
            "A growing-window subsampling method must be the only method in a run, got {}. "
            "Such a run reports every quantity over the window its search settled on, and "
            "two methods settle on two different windows -- so nothing in the run could be "
            "compared like for like. Run each method separately (same inputs, separate "
            "output folders) and compare the results.".format(list(subsampling_methods)))
    return growing[0]

def _resolve_subsampling_method(subsampling_method):
    """Look up a Poisson-adaptive subsampling callable by name (a key of
    POISSON_SUBSAMPLING_METHODS) -- all entries share the (time, MinimumNumberOfSamples=...)
    call signature, so any can be dropped in wherever find_samples_path was called
    directly before this was made selectable."""
    try:
        return POISSON_SUBSAMPLING_METHODS[subsampling_method]
    except KeyError:
        raise ValueError("Unknown subsampling_method {!r}; choose one of {}".format(
            subsampling_method, list(POISSON_SUBSAMPLING_METHODS)))


def call_subsampling_method(subsampling_method, times, min_samples, steady_start=None,
                             steady_end=None, delta_cache=None):
    """Draw one Poisson-adaptive subsample of `times` with the named method, targeting
    `min_samples` samples. Every method shares the (time, MinimumNumberOfSamples=...)
    contract and returns (samples, subSamplingError); the only extra thing passed here is
    the analysis window, and only to the methods listed in
    STEADY_WINDOW_AWARE_SUBSAMPLING_METHODS -- the growing-window methods need it because
    their candidate windows are anchored at steadyStart, and anchoring them at the flow's
    own first packet instead would move the window boundaries with the flow count.

    This is the whole-steady-window entry point, so a growing-window method reached through
    it derives its sample target from the `min_samples` it is handed. Inside the
    EMD-vs-flows pipeline use find_samples_growing_window_with_stats instead, which gives
    each candidate window its own switch-side statistics and its own target."""
    method = _resolve_subsampling_method(subsampling_method)
    kwargs = {}
    # A whole-window sampler derives one bin width per call, for the same packets every run;
    # hand it the precomputed one (see precompute_subsample_deltas) when there is one.
    if delta_cache is not None and subsampling_method in FIND_DELTA_BASED_METHODS:
        cached_delta = delta_cache.get(int(len(times)))
        if cached_delta is not None:
            kwargs['window'] = cached_delta
    if subsampling_method in STEADY_WINDOW_AWARE_SUBSAMPLING_METHODS:
        return method(times, MinimumNumberOfSamples=min_samples,
                      steadyStart=steady_start, steadyEnd=steady_end, **kwargs)
    return method(times, MinimumNumberOfSamples=min_samples, **kwargs)


def growing_window_context(prepared, confidenceValue, DelayConsistencyGaurantee,
                            num_poisson_observations, step_ns=GROWING_WINDOW_STEP_NS):
    """Everything a growing-window method needs to evaluate a candidate window end-to-end
    inside one window: where the raw traces are, which ground truth to rebuild, and at what
    rate the switch-side Poisson probe observes. Built once per run (see
    _run_one_poisson_run) and handed to find_samples_growing_window_with_stats /
    windowed_groundtruth.

    `observation_rate_per_ns` is the key quantity: the switch-side probe keeps its RATE
    fixed at num_poisson_observations / (steadyEnd - steadyStart), so a candidate window of
    length L is given about rate*L observations rather than the full run's count. A probe
    does not observe faster because we chose to watch a shorter window, and pretending
    otherwise would hand a 5 ms window the accuracy of a 90 ms one -- which is precisely
    the cost of stopping early that this whole mode exists to measure."""
    duration = float(prepared['steady_end']) - float(prepared['steady_start'])
    return {
        'dir_prefix': prepared['dir_prefix'],
        'queue_names': list(prepared['queue_names']),
        'link_delays': list(prepared['link_delays']),
        'link_rates': list(prepared['link_rates']),
        'steady_start': float(prepared['steady_start']),
        'steady_end': float(prepared['steady_end']),
        'groundtruth_method': prepared['groundtruth_method'],
        'delay_cdf_sample_interval_ns': prepared.get('delay_cdf_sample_interval_ns', 10),
        'delay_percentiles': tuple(prepared.get('delay_percentiles') or ()),
        'observation_rate_per_ns': (num_poisson_observations / duration) if duration > 0 else 0.0,
        'confidence_value': confidenceValue,
        'delay_consistency_guarantee': DelayConsistencyGaurantee,
        'step_ns': step_ns,
    }


def windowed_poisson_agg_stats(window_ctx, window_end):
    """The switch-side Poisson realization over [steadyStart, window_end] and the
    per-segment aggregated delay statistics it yields (compute_poisson_agg_stats), with the
    observation count scaled to the window's length so the probing rate matches the full
    run's (see growing_window_context). Returns (agg_stats, min_samples), where
    `min_samples` is None when this window's statistics cannot support the consistency
    guarantee at all -- calc_min_e2e_samples' own verdict, which for a short window is the
    normal outcome rather than an error.

    Memoized inside `window_ctx`, which lives for exactly one run (see
    _run_one_poisson_run), for both correctness and cost: a run has ONE switch-side probe,
    so observing a given window must yield the same statistics no matter which flow count's
    subsample is being evaluated -- exactly as the full-window agg_stats is drawn once per
    run and shared across every k. Redrawing per flow count would also make the search the
    dominant cost of a flow-count sweep."""
    cache = window_ctx.setdefault('agg_stats_cache', {})
    cached = cache.get(float(window_end))
    if cached is not None:
        return cached
    num_obs = int(round(window_ctx['observation_rate_per_ns'] * (window_end - window_ctx['steady_start'])))
    num_obs = max(num_obs, MIN_WINDOWED_POISSON_OBSERVATIONS)
    with contextlib.redirect_stdout(io.StringIO()):
        # calc_min_e2e_samples prints a warning whenever a window cannot support the
        # guarantee; here that is the expected outcome for the early candidates, reported
        # once by the search itself rather than once per window per run per flow count.
        agg_stats = compute_poisson_agg_stats(
            window_ctx['dir_prefix'], window_ctx['queue_names'], window_ctx['link_delays'],
            window_ctx['link_rates'], window_ctx['steady_start'], window_end, num_obs,
            window_ctx['confidence_value'], window_ctx['delay_consistency_guarantee'])
    agg_stats['num_poisson_observations'] = num_obs
    cache[float(window_end)] = (agg_stats, agg_stats.get('MinimumE2ESampleSizeDelay'))
    return cache[float(window_end)]


# Ground truths rebuilt over a sub-window are keyed by that window and reused: a window end
# can only take one of the (few) candidate values GROWING_WINDOW_STEP_NS allows, while the
# reconstruction itself is the single most expensive step in this pipeline, and it is the
# same for every run and flow count that stopped at the same window. Capped and evicted
# oldest-first so a long multi-traffic/load sweep in one process cannot accumulate every
# window of every experiment (each entry is up to ~1M delay samples).
_WINDOWED_GROUNDTRUTH_CACHE = OrderedDict()
_WINDOWED_GROUNDTRUTH_CACHE_MAX = 24


def windowed_groundtruth(window_ctx, window_end):
    """The ground-truth reconstructed path-delay CDF over [steadyStart, window_end] --
    the same construction (GROUNDTRUTH_METHODS) and the same sampling interval
    prepare_emd_vs_flows_data used for the full steady window, just restricted to the
    window a growing-window method actually stopped at. Returns
    (values, percentiles, mean, path_probabilities) -- the last being the window's own
    success / non-marking probability reference (construct_path_prob_ground_truth).

    This exists because a consistency check or an EMD must never mix windows: a subsample
    drawn from the first 5 ms of the steady period compared against a ground truth averaged
    over all 90 ms measures how much the network changed between them, not how well the
    subsample recovered the delay distribution it was drawn from. The nominal steady window
    is measurably non-stationary in these experiments (the mean queuing delay of its first
    5 ms runs several times that of its last 5 ms), so this is a first-order effect, not a
    refinement."""
    key = (window_ctx['groundtruth_method'], window_ctx['dir_prefix'],
           tuple(window_ctx['queue_names']), window_ctx['steady_start'], float(window_end),
           float(window_ctx['delay_cdf_sample_interval_ns']), window_ctx['delay_percentiles'])
    entry = _WINDOWED_GROUNDTRUTH_CACHE.get(key)
    if entry is None:
        construct = _resolve_groundtruth_method(window_ctx['groundtruth_method'])
        values = construct(
            window_ctx['queue_names'], window_ctx['dir_prefix'], window_ctx['steady_start'],
            float(window_end), window_ctx['link_delays'], window_ctx['link_rates'],
            sample_interval_ns=window_ctx['delay_cdf_sample_interval_ns'])
        # The loss/marking reference for the same window, at the same dense rate -- so the
        # probability metrics are scored against their own window too, not the full period.
        probs = construct_path_prob_ground_truth(
            window_ctx['queue_names'], window_ctx['dir_prefix'], window_ctx['steady_start'],
            float(window_end), window_ctx['link_delays'], window_ctx['link_rates'],
            sample_interval_ns=window_ctx['delay_cdf_sample_interval_ns'])
        entry = (values,
                 compute_delay_percentiles(values, window_ctx['delay_percentiles']),
                 float(np.mean(values)) if len(values) else np.nan,
                 probs)
        _WINDOWED_GROUNDTRUTH_CACHE[key] = entry
        while len(_WINDOWED_GROUNDTRUTH_CACHE) > _WINDOWED_GROUNDTRUTH_CACHE_MAX:
            _WINDOWED_GROUNDTRUTH_CACHE.popitem(last=False)
    else:
        _WINDOWED_GROUNDTRUTH_CACHE.move_to_end(key)
    return entry


def _delta_for_prefix(args):
    """One (key, timestamps) -> (key, delta) step of precompute_subsample_deltas, at module
    level so it can be mapped over a process pool."""
    key, times = args
    try:
        delta, _ = find_delta_for_empty_prob(times, p0_max=0.01)
    except ValueError:
        # Too few distinct timestamps for a delta to exist; the sampler takes its own
        # (identical) failure path when it gets there.
        return key, None
    return key, delta


def precompute_subsample_deltas(prepared, subsampling_methods, step_ns=GROWING_WINDOW_STEP_NS,
                                 num_workers=1):
    """Precompute find_delta_for_empty_prob for every packet prefix the run's samplers can
    be handed, as {k: {prefix length: delta}} -- done ONCE in the parent, before the runs
    fork, so every worker inherits the answers instead of recomputing them.

    This is pure memoization of a deterministic function: find_delta_for_empty_prob depends
    only on the timestamps it is given (it sorts and de-duplicates them, then scans a fixed
    grid of candidate bin widths) and draws no randomness, so a cached value is bit-identical
    to a recomputed one. What it saves is large: the growing-window search walks up to
    (steady window / step) candidate prefixes, and every one of the run's 50 repetitions
    would otherwise re-derive the same delta for the same packets. At 18 candidate windows
    that is 900 calls per flow count where 18 distinct answers exist, and the call is the
    dominant cost of the whole search (a ~3000-point grid over every timestamp, see
    find_delta_for_empty_prob).

    Returns an empty dict when no method in the run uses a delta at all
    (FIND_DELTA_BASED_METHODS -- the intensity sampler does not), so an intensity-only run
    pays nothing for this.

    `num_workers` > 1 spreads the widths over a process pool. They are independent and the
    function is pure, so the results do not depend on the order or the worker count -- and
    this is worth doing because the precompute is otherwise the largest SERIAL part of an
    experiment (~17s for 18 widths on one core), which on a 1350-experiment sweep is hours
    of one core while the rest of the machine waits."""
    methods = normalize_subsampling_methods(subsampling_methods)
    if not any(m in FIND_DELTA_BASED_METHODS for m in methods):
        return {}
    full_df = prepared['full_df']
    steady_start = float(prepared['steady_start'])
    steady_end = float(prepared['steady_end'])
    growing = any(m in GROWING_WINDOW_SUBSAMPLING_METHODS for m in methods)
    cache, tasks = {}, []
    for k in prepared['num_flows']:
        times = np.sort(np.asarray(
            full_df[full_df['FlowRank'] <= k]['SentTime'].values, dtype=float))
        if len(times) < 2:
            continue
        # The prefix lengths the search can reach: one per candidate window end, plus the
        # whole set (which is what a non-growing method always uses).
        counts = {len(times)}
        if growing and step_ns and step_ns > 0:
            num_steps = max(1, int(np.ceil((steady_end - steady_start) / float(step_ns))))
            window_ends = np.minimum(
                steady_start + float(step_ns) * np.arange(1, num_steps + 1), steady_end)
            counts.update(int(c) for c in np.searchsorted(times, window_ends, side='right'))
        cache[k] = {}
        for count in sorted(c for c in counts if c >= 2):
            tasks.append(((k, int(count)), times[:count]))

    if not tasks:
        return cache
    if num_workers and num_workers > 1 and len(tasks) > 1:
        with multiprocessing.Pool(min(int(num_workers), len(tasks))) as pool:
            results = pool.map(_delta_for_prefix, tasks)
    else:
        results = [_delta_for_prefix(task) for task in tasks]
    for (k, count), delta in results:
        if delta is not None:
            cache[k][count] = delta
    return cache


def find_samples_growing_window_with_stats(subsampling_method, times, window_ctx,
                                            trim_to_minimum=True, delta_cache=None):
    """Run one growing-window subsampling method over `times`, giving every candidate
    window its own switch-side statistics (windowed_poisson_agg_stats) and therefore its
    own required sample size, and return the full outcome dict (see
    _growing_window_result): the samples, the error, the window they came from, and that
    window's agg_stats/min_samples -- so the caller can do the consistency check and the
    EMD against statistics measured over exactly that window.

    This is the pipeline entry point for the growing-window methods; the plain
    (time, MinimumNumberOfSamples=...) callables in POISSON_SUBSAMPLING_METHODS remain
    available for a direct call with a caller-chosen target."""
    try:
        base_name, base_wants_window = GROWING_WINDOW_SUBSAMPLING_METHODS[subsampling_method]
    except KeyError:
        raise ValueError("{!r} is not a growing-window subsampling method; choose one of {}".format(
            subsampling_method, list(GROWING_WINDOW_SUBSAMPLING_METHODS)))
    return _growing_window_search(
        _resolve_subsampling_method(base_name), times,
        step_ns=window_ctx.get('step_ns', GROWING_WINDOW_STEP_NS),
        steadyStart=window_ctx['steady_start'], steadyEnd=window_ctx['steady_end'],
        trim_to_minimum=trim_to_minimum, base_wants_window=base_wants_window,
        window_stats=lambda window_end: windowed_poisson_agg_stats(window_ctx, window_end),
        delta_cache=delta_cache if base_name in FIND_DELTA_BASED_METHODS else None)


def normalize_subsampling_methods(subsampling_methods):
    """Accept either a single subsampling-method name or an iterable of them and
    return them as a de-duplicated list in the given order, so that one run can
    evaluate several Poisson-adaptive algorithms side by side (each becoming its
    own comparison family, exactly like each uniform stride does) while a bare
    string keeps behaving as it always did."""
    if isinstance(subsampling_methods, str):
        subsampling_methods = [subsampling_methods]
    ordered = []
    for name in subsampling_methods:
        _resolve_subsampling_method(name)  # fail fast on a typo, before any real work
        if name not in ordered:
            ordered.append(name)
    if not ordered:
        raise ValueError("At least one subsampling method is required")
    return ordered


def subsampling_methods_tag(subsampling_methods):
    """The filename tag identifying which Poisson-adaptive subsampling
    algorithm(s) an output was computed with. A single method yields just its
    name -- byte-identical to what this pipeline has always written -- and
    several are joined with '+' ('a+b'), so a multi-method run's outputs never
    collide with either single-method run's. Combined with
    groundtruth_method_tag to form the full per-output tag."""
    return '+'.join(normalize_subsampling_methods(subsampling_methods))


# The relative-error guarantee (DelayConsistencyGaurantee) every EMD-vs-flows result
# computed before 2026-09-15 was produced at, and the value the guarantee tag treats as
# "no tag" so those results keep the paths they already have. Also the default of
# PostProcessing's --delay-consistency-guarantee.
DEFAULT_DELAY_CONSISTENCY_GUARANTEE = 0.40


def delay_consistency_guarantee_tag(delay_consistency_guarantee):
    """Filename/folder tag identifying the relative-error guarantee a set of EMD-vs-flows
    outputs was computed at, as an already-prefixed '_g<percent>' fragment ('_g20' for 0.20,
    '_g7.5' for 0.075) -- or '' for DEFAULT_DELAY_CONSISTENCY_GUARANTEE, so every result
    computed at the historical 0.40 keeps the path it already has (the same
    backwards-compatibility convention groundtruth_method_tag uses for 'simultaneous').

    This belongs in the path because the guarantee changes what a run *is*, not just how
    strictly it is judged: it sets the minimum sample size the subsampler must reach
    (calc_min_e2e_samples), so the samples, the monitoring window a growing-window method
    settles on, the error bound and every EMD in the result are all specific to it. Two
    guarantees' outputs for the same traffic/rate/load/experiment would otherwise overwrite
    each other."""
    if delay_consistency_guarantee is None or delay_consistency_guarantee == DEFAULT_DELAY_CONSISTENCY_GUARANTEE:
        return ''
    return '_g{:g}'.format(float(delay_consistency_guarantee) * 100)


def emd_vs_flows_file_tag(subsampling_methods, groundtruth_method='simultaneous',
                           all_flows_only=False, delay_consistency_guarantee=None):
    """The full tag that identifies one EMD-vs-flows configuration in every
    output filename: which subsampling algorithm(s) were compared, which
    ground truth they were compared against, whether the run swept flow
    counts or evaluated only all-flows, and which relative-error guarantee it
    was computed at. Kept deliberately backwards
    compatible -- a single subsampling method against the original
    'simultaneous' ground truth, swept, at the historical
    DEFAULT_DELAY_CONSISTENCY_GUARANTEE, reproduces the pre-existing '<method>'
    tag exactly, so already-computed results stay discoverable.

    The all-flows-only tag matters because such a run's results cover a single
    k while a swept run covers many; without it the two would overwrite each
    other's pickles for the same traffic/rate/load/experiment. The guarantee
    tag matters for the same reason -- see delay_consistency_guarantee_tag."""
    return (subsampling_methods_tag(subsampling_methods)
            + groundtruth_method_tag(groundtruth_method)
            + ('_allflows' if all_flows_only else '')
            + delay_consistency_guarantee_tag(delay_consistency_guarantee))


def steady_window_tag(steadyStart, steadyEnd):
    """Folder-name fragment identifying the steady-state analysis window (both in ns) a set
    of EMD-vs-flows outputs was computed over, e.g. steadyStart=1e7, steadyEnd=1e8 ->
    'steady_10-100ms'. The same raw ns-3 run can be re-analyzed over a different window (e.g.
    to check stationarity, or because a longer/shorter steady period was configured) without
    colliding with or silently overwriting an earlier window's outputs -- every EMD-vs-flows
    output path (per-experiment, per-traffic/load aggregate, and cross-traffic/load aggregate)
    includes this tag as its own folder level."""
    return 'steady_{:g}-{:g}ms'.format(steadyStart / 1e6, steadyEnd / 1e6)


def resolve_emd_vs_flows_pickle_path(ns3_path, results_folder, rate, load, experiment,
                                      steady_tag, config_tag, flow_name, path):
    """The on-disk path of one run_emd_vs_flows_experiment pickle for one specific
    experiment, preferring the current nested `<steady_tag>/<config_tag>/` layout and
    falling back to the pre-2026-09-09 flat layout (config_tag as a filename infix, no
    steady-window folder) if that's where it actually lives -- see
    aggregate_emd_vs_flows_across_experiments for the equivalent multi-experiment scan this
    mirrors. Returns (pkl_path, file_prefix) for whichever layout exists on disk, so a caller
    can save companion outputs (replots, the .txt summary) alongside it with the same
    prefix; (None, None) if neither layout has it."""
    base = '{}/scratch/{}/{}/{}/{}/'.format(ns3_path, results_folder, rate, load, experiment)
    new_prefix = '{}{}/{}/{}_path_{}'.format(base, steady_tag, config_tag, flow_name, path)
    new_pkl = new_prefix + '_emd_vs_num_flows_results.pkl'
    if os.path.isfile(new_pkl):
        return new_pkl, new_prefix
    legacy_prefix = '{}{}_path_{}_{}'.format(base, flow_name, path, config_tag)
    legacy_pkl = legacy_prefix + '_emd_vs_num_flows_results.pkl'
    if os.path.isfile(legacy_pkl):
        return legacy_pkl, legacy_prefix
    return None, None


def compute_emd_vs_num_tcp_flows_run(prepared, agg_stats, confidenceValue, min_sample_size=30,
                                      subsampling_methods='find_samples_path', window_ctx=None):
    """Run one realization of the flow-count EMD sweep against a given
    per-run `agg_stats` (see compute_poisson_agg_stats): grow the set of
    considered TCP flows one at a time and, for each size, compare the
    ground-truth CDF in `prepared` against every way of subsampling the
    considered flows' packets:

      - one fresh Poisson-adaptive subsample per entry in `subsampling_methods`
        (any number of POISSON_SUBSAMPLING_METHODS keys, e.g. both
        find_samples_path and find_samples_path_intensity in the same run, so
        the algorithms are compared on identical packets, flows, ground truth
        and per-run switch statistics), and
      - for each of those methods, one **rate-matched** systematic uniform
        subsample drawing exactly as many packets as that method just retained
        (sample_uniform_count / matched_uniform_target_count). Matching the
        count is what isolates the question of interest: EMD and the
        consistency bound both tighten with sample size, so a uniform baseline
        at a different size would confound "better selection rule" with "more
        packets". Where a method found no valid subsample, its uniform
        counterpart falls back to the minimum sample size that method was
        required to reach.
      - an **ideal Poisson probe** family per entry of oracle_target_counts:
        one at the minimum required sample size and one matched to each
        method's retained count. These do not select from the flow's packets at
        all -- they re-run the ground-truth construction at that low rate
        (construct_oracle_poisson_delays), so their sampling instants are a
        genuine Poisson process independent of queue state and their only error
        is finite-sample noise. They are the ceiling any real Poissonization
        scheme is trying to reach.

    **Windowed runs.** If `subsampling_methods` is a growing-window method
    (necessarily the run's only method, see growing_window_method_in) and
    `window_ctx` is given (growing_window_context), the method first searches
    for the shortest prefix [steadyStart, W] of the steady window whose own
    switch-side statistics and packets can supply the samples the consistency
    check needs (_growing_window_search). That window then becomes the
    analysis window for **everything this run reports at that flow count**:

      - the switch-side statistics are the ones drawn over [steadyStart, W] at
        the run's probing rate (windowed_poisson_agg_stats), so the
        consistency bound, the required sample size and the
        switch-vs-packet mean difference all come from the same interval;
      - the ground-truth CDF and its percentiles are rebuilt over
        [steadyStart, W] (windowed_groundtruth), so every EMD, relative EMD
        and percentile error is measured against the delay distribution that
        actually held while the samples were collected;
      - the all-packets family is the packets in [steadyStart, W] -- not the
        whole steady window -- and so are the rate-matched uniform baseline and
        every ideal Poisson probe, including the minimum-required-budget probe
        and the all-packets-rate probe;
      - nothing is reported over [steadyStart, steadyEnd] unless the search
        itself reached steadyEnd. A run whose search found no usable window at
        all reports nothing at that flow count (every family NaN/None): it
        never certified anything, so there is no window to report in.

    Mixing windows is what this avoids: the nominal steady period is
    measurably non-stationary here, so comparing an early-window subsample
    against a full-window reference would report how much the network changed
    rather than how well the subsample recovered its own window's
    distribution.

    In a windowed run the all-packets family therefore varies from run to run
    (each run's window differs) and is computed here; in an ordinary run it is
    the same fixed packet set every run and its EMD, percentile errors,
    Poisson-ness tests and burstiness come precomputed from
    prepare_emd_vs_flows_data, with only the consistency check redone per run.
    'all_packets_windowed' in the returned dict says which of the two it was.

    Every family's monitoring duration is recorded per run
    ('sampled_window_duration') -- for a growing-window method it is the
    answer to "how long did we have to watch?", and for every other method it
    is the full steady window.

    Returns a dict with, for each k=1..N considered flows:
      - 'consistency_pass_all_packets' / 'mean_diff_all_packets': the
        consistency check and signed mean difference applied to the mean of
        the all-packets family (in-window in a windowed run); None/NaN when
        there are no packets.
      - 'emd_all_packets' / 'emd_all_packets_normalized' / 'all_packet_size' /
        'percentile_diff_all_packets' / 'percentile_reldiff_all_packets' /
        'percentile_avg_relerror_all_packets' / 'poisson_tests_all_packets' /
        'burstiness_all_packets': the all-packets family's own per-run values,
        filled only in a windowed run (NaN/None otherwise, where
        prepare_emd_vs_flows_data's fixed per-k values stand instead).
      - 'sampled_emd' / 'sampled_emd_normalized' / 'sampled_consistency' /
        'sampled_mean_diff' / 'sampled_sample_sizes' /
        'sampled_window_duration': each a dict keyed by subsampling-method
        name, holding that method's per-k list; NaN/None/0 at a k where that
        method found no valid subsample (e.g. too few packets/windows).
      - 'uniform_emd' / 'uniform_emd_normalized' / 'uniform_consistency' /
        'uniform_mean_diff' / 'uniform_sample_sizes': the same, keyed by the
        method name whose sample count each uniform family matches.
      - 'sampled_percentile_diff' / 'uniform_percentile_diff': signed absolute
        `ground_truth_percentile - family_percentile` (ns) at every percentile
        in prepared['delay_percentiles'], as {q: {name: per-k list}} -- the
        tail-shape error the EMD can hide. NaN where that family had no values.
        '..._percentile_reldiff' are the same as a fraction of that family's
        own ground-truth percentile.
    """
    full_df = prepared['full_df']
    groundtruth_values = prepared['groundtruth_values']
    groundtruth_mean = prepared.get('groundtruth_mean')
    if groundtruth_mean is None:
        groundtruth_mean = float(np.mean(groundtruth_values)) if len(groundtruth_values) else np.nan
    min_samples = agg_stats.get('MinimumE2ESampleSizeDelay', 0)
    subsampling_methods = normalize_subsampling_methods(subsampling_methods)
    windowed_method = growing_window_method_in(subsampling_methods) if window_ctx is not None else None
    delay_percentiles = tuple(prepared.get('delay_percentiles') or ())
    groundtruth_percentiles = prepared.get('groundtruth_percentiles', {})
    run_chi = prepared.get('run_chi_squared_test', True)
    test_lags = prepared.get('poisson_test_lags')
    steady_start = prepared.get('steady_start')
    steady_end = prepared.get('steady_end')
    burst_gap = prepared.get('burst_gap_threshold_ns')
    groundtruth_probs = prepared.get('groundtruth_probs') or {
        metric: np.nan for metric in PROB_METRIC_KEYS}

    num_flows_list = []
    consistency_all_list, mean_diff_all_list = [], []
    # All-packets quantities that only a windowed run computes per run (an ordinary run's
    # are fixed per flow count and already in `prepared`).
    emd_all_list, emd_all_norm_list, size_all_list = [], [], []
    # The mean queuing delay each family actually estimates, next to the ground truth's own
    # -- the delay counterpart of the probability metrics' estimate-and-reference pair. The
    # EMD says how far a family's whole distribution is from the truth; this says what its
    # headline number is, which is what the consistency check thresholds.
    delay_mean_all_list, groundtruth_delay_mean_list = [], []
    # The consistency check's own threshold at each family's sample size, per run: relative
    # to the switch-side mean (directly comparable to DelayConsistencyGaurantee) and in ns
    # (directly comparable to mean_diff). See delay_consistency_error_bound.
    bound_all_list, bound_ns_all_list = [], []
    pdiff_all = {q: [] for q in delay_percentiles}
    preldiff_all = {q: [] for q in delay_percentiles}
    pctrelerr_all_list = []
    tests_all = {'ad_pass': [], 'ad_pvalue': [], 'chi_pass': [], 'chi_reject_fraction': []}
    burstiness_all = {field: [] for field in BURSTINESS_METRIC_LABELS}
    sampled_emd = {name: [] for name in subsampling_methods}
    sampled_emd_norm = {name: [] for name in subsampling_methods}
    sampled_consistency = {name: [] for name in subsampling_methods}
    sampled_mean_diff = {name: [] for name in subsampling_methods}
    sampled_sample_sizes = {name: [] for name in subsampling_methods}
    sampled_error_bound = {name: [] for name in subsampling_methods}
    sampled_error_bound_ns = {name: [] for name in subsampling_methods}
    sampled_delay_mean = {name: [] for name in subsampling_methods}
    # Length of the monitoring window each method's samples actually came from: the whole
    # point of the growing-window methods, and the full steady window for everything else.
    sampled_window_duration = {name: [] for name in subsampling_methods}
    uniform_emd = {name: [] for name in subsampling_methods}
    uniform_emd_norm = {name: [] for name in subsampling_methods}
    uniform_consistency = {name: [] for name in subsampling_methods}
    uniform_mean_diff = {name: [] for name in subsampling_methods}
    uniform_sample_sizes = {name: [] for name in subsampling_methods}
    uniform_error_bound = {name: [] for name in subsampling_methods}
    uniform_error_bound_ns = {name: [] for name in subsampling_methods}
    uniform_delay_mean = {name: [] for name in subsampling_methods}
    sampled_percentile_diff = {q: {name: [] for name in subsampling_methods} for q in delay_percentiles}
    sampled_percentile_reldiff = {q: {name: [] for name in subsampling_methods} for q in delay_percentiles}
    uniform_percentile_diff = {q: {name: [] for name in subsampling_methods} for q in delay_percentiles}
    uniform_percentile_reldiff = {q: {name: [] for name in subsampling_methods} for q in delay_percentiles}
    # Mean absolute relative percentile error (see percentile_avg_relative_error):
    # self-normalized, evaluated via a dense, fixed percentile grid, independent of
    # delay_percentiles/sampled_percentile_diff above.
    sampled_percentile_avg_relerror = {name: [] for name in subsampling_methods}
    uniform_percentile_avg_relerror = {name: [] for name in subsampling_methods}
    oracle_series = [ORACLE_MIN_REQUIRED_KEY] + list(subsampling_methods) + [ORACLE_ALL_PACKETS_RATE_KEY]
    oracle_emd = {key: [] for key in oracle_series}
    oracle_emd_norm = {key: [] for key in oracle_series}
    oracle_consistency = {key: [] for key in oracle_series}
    oracle_mean_diff = {key: [] for key in oracle_series}
    oracle_sample_sizes = {key: [] for key in oracle_series}
    oracle_error_bound = {key: [] for key in oracle_series}
    oracle_error_bound_ns = {key: [] for key in oracle_series}
    oracle_delay_mean = {key: [] for key in oracle_series}
    oracle_percentile_diff = {q: {key: [] for key in oracle_series} for q in delay_percentiles}
    oracle_percentile_reldiff = {q: {key: [] for key in oracle_series} for q in delay_percentiles}
    oracle_percentile_avg_relerror = {key: [] for key in oracle_series}
    # Per-run Poisson-ness of each uniform subset's own instants, kept strictly in lockstep
    # with that family's EMD/mean-diff so the plots can split runs by test outcome without
    # any risk of pairing a value with another run's verdict.
    uniform_test_split = {name: {'emd': [], 'emd_normalized': [], 'mean_diff': [],
                                  'ad_pass': [], 'chi_pass': []}
                           for name in subsampling_methods}
    # The loss/marking side of the check, for exactly the same families over exactly the
    # same packets and window as the delay side above (see PROB_METRICS).
    prob_results = {
        metric: {
            'groundtruth_prob': [],
            'all_packets': {field: [] for field in PROB_FAMILY_FIELDS},
            'sampled': {name: {field: [] for field in PROB_FAMILY_FIELDS}
                         for name in subsampling_methods},
            'uniform': {name: {field: [] for field in PROB_FAMILY_FIELDS}
                         for name in subsampling_methods},
            'oracle': {key: {field: [] for field in PROB_FAMILY_FIELDS}
                        for key in oracle_series},
        } for metric in PROB_METRIC_KEYS}

    def _record_prob(store, metric, rows, gt_probs, stats):
        """Evaluate one probability metric for one family's packets and append the record."""
        result = evaluate_prob_family(
            prob_metric_values(rows, metric), gt_probs.get(metric, np.nan), stats, metric,
            confidenceValue, min_sample_size)
        for field in PROB_FAMILY_FIELDS:
            store[field].append(result[field])

    def _reldiff(absolute_diff, reference_percentiles, q):
        """One percentile's error as a fraction of the reference percentile it was measured
        against -- the per-family counterpart of relative_percentile_diffs, computed here
        because a windowed family's reference is its own window's ground truth, not the
        full window's."""
        return scale_nested_values(absolute_diff, reference_percentiles.get(q))

    for k in prepared['num_flows']:
        subset = full_df[full_df['FlowRank'] <= k]
        num_flows_list.append(k)
        times = subset['SentTime'].values
        # The bin widths this flow count's prefixes imply, precomputed once in the parent
        # (precompute_subsample_deltas) and inherited by every worker.
        k_delta_cache = (prepared.get('delta_cache') or {}).get(k)

        # ------------------------------------------------------------------ analysis window
        # In a windowed run the search runs FIRST: the window it settles on is what every
        # family below -- all packets included -- is then evaluated over.
        found = None
        if windowed_method is not None:
            found = find_samples_growing_window_with_stats(
                windowed_method, times, window_ctx, delta_cache=k_delta_cache)
            certified = found['window_end'] is not None
            window_end = found['window_end'] if certified else None
            run_agg_stats = found['agg_stats'] if certified else None
            run_min_samples = found['min_samples'] if certified else None
            if certified:
                run_subset = subset[subset['SentTime'] <= window_end]
                run_gt_values, run_gt_percentiles, run_gt_mean, run_gt_probs = windowed_groundtruth(
                    window_ctx, float(window_end))
            else:
                # No window could supply the samples the check needs, so this run certified
                # nothing at this flow count and has no window to report anything in. The
                # percentile reference is still keyed by every tracked percentile (all NaN)
                # rather than left empty: percentile_diffs returns a dict keyed by whatever
                # its reference holds, and every caller below indexes it by q.
                run_subset = subset.iloc[0:0]
                run_gt_values, run_gt_mean = np.array([]), np.nan
                run_gt_percentiles = {q: np.nan for q in delay_percentiles}
                run_gt_probs = {metric: np.nan for metric in PROB_METRIC_KEYS}
        else:
            certified = True
            window_end = steady_end
            run_agg_stats, run_min_samples, run_subset = agg_stats, min_samples, subset
            run_gt_values, run_gt_percentiles, run_gt_mean = (
                groundtruth_values, groundtruth_percentiles, groundtruth_mean)
            run_gt_probs = groundtruth_probs

        # Delay is only observable on packets that arrived; the frame holds every sent
        # packet so the success probability can be estimated at all (see received_rows).
        all_values = received_rows(run_subset)['Delay'].values

        # ------------------------------------------------------------------- all packets
        if len(all_values) and run_agg_stats is not None:
            consistency_all_list.append(_delay_consistency_check(
                all_values, run_agg_stats, confidenceValue, min_sample_size))
            mean_diff_all_list.append(run_agg_stats['DelayMean'] - np.mean(all_values))
        else:
            consistency_all_list.append(None)
            mean_diff_all_list.append(np.nan)
        bound_ns, bound_rel = delay_consistency_error_bound(
            run_agg_stats, len(all_values), confidenceValue)
        bound_all_list.append(bound_rel)
        bound_ns_all_list.append(bound_ns)
        delay_mean_all_list.append(float(np.mean(all_values)) if len(all_values) else np.nan)
        groundtruth_delay_mean_list.append(run_gt_mean)
        for metric in PROB_METRIC_KEYS:
            prob_results[metric]['groundtruth_prob'].append(run_gt_probs.get(metric, np.nan))
            _record_prob(prob_results[metric]['all_packets'], metric, run_subset,
                          run_gt_probs, run_agg_stats)
        if windowed_method is not None:
            # Only a windowed run computes these here: the all-packets family is a different
            # packet set every run because every run's window differs. An ordinary run's
            # fixed per-flow-count values come from prepare_emd_vs_flows_data instead.
            emd_all = (wasserstein_distance(run_gt_values, all_values)
                        if len(all_values) and len(run_gt_values) else np.nan)
            emd_all_list.append(emd_all)
            emd_all_norm_list.append(normalize_emd_values(emd_all, run_gt_mean))
            size_all_list.append(len(all_values))
            all_diffs = percentile_diffs(all_values, run_gt_percentiles)
            for q in delay_percentiles:
                pdiff_all[q].append(all_diffs[q])
                preldiff_all[q].append(_reldiff(all_diffs[q], run_gt_percentiles, q))
            pctrelerr_all_list.append(percentile_avg_relative_error(run_gt_values, all_values))
            all_tests = poisson_process_tests(
                run_subset['SentTime'].values, steady_start, window_end,
                lags=test_lags, run_chi_squared=run_chi)
            for field in tests_all:
                tests_all[field].append(all_tests[field])
            all_burst = burstiness_metrics(run_subset['SentTime'].values, burst_gap)
            for field in burstiness_all:
                burstiness_all[field].append(all_burst[field])

        # --------------------------------------------------------- Poisson-adaptive families
        # Per-method evaluation context, reused by the ideal-Poisson-probe loop below so a
        # method's probe is built over the same window and compared against the same ground
        # truth as the method itself.
        method_ctx = {}
        for name in subsampling_methods:
            if name == windowed_method:
                samples_times, sub_err = found['samples'], found['error']
            else:
                samples_times, sub_err = call_subsampling_method(
                    name, times, min_samples, steady_start, steady_end,
                    delta_cache=k_delta_cache)
            # A run that produced no subsample has no monitoring duration to report: the
            # column means "the window these samples came from" and there are none.
            search_failed = sub_err != SubSamplingError.NoError or len(samples_times) == 0
            if search_failed or window_end is None or steady_start is None:
                sampled_window_duration[name].append(np.nan)
            else:
                sampled_window_duration[name].append(float(window_end) - float(steady_start))
            method_ctx[name] = {'window_end': window_end, 'agg_stats': run_agg_stats,
                                 'gt_values': run_gt_values, 'gt_percentiles': run_gt_percentiles,
                                 'gt_mean': run_gt_mean}

            if search_failed or run_agg_stats is None:
                sampled_emd[name].append(np.nan)
                sampled_emd_norm[name].append(np.nan)
                sampled_consistency[name].append(None)
                sampled_sample_sizes[name].append(0)
                sampled_mean_diff[name].append(np.nan)
                sampled_size = 0
                sample_values = np.array([])
                selected_rows = run_subset.iloc[0:0]
            else:
                selected_rows = run_subset[run_subset['SentTime'].isin(samples_times)]
                sample_values = received_rows(selected_rows)['Delay'].values
                emd, consistency_pass, mean_diff, sampled_size = _evaluate_delay_family(
                    sample_values, run_gt_values, run_agg_stats, confidenceValue, min_sample_size)
                sampled_emd[name].append(emd)
                sampled_emd_norm[name].append(normalize_emd_values(emd, run_gt_mean))
                sampled_consistency[name].append(consistency_pass)
                sampled_sample_sizes[name].append(sampled_size)
                sampled_mean_diff[name].append(mean_diff)
            bound_ns, bound_rel = delay_consistency_error_bound(
                run_agg_stats, sampled_size, confidenceValue)
            sampled_error_bound[name].append(bound_rel)
            sampled_error_bound_ns[name].append(bound_ns)
            sampled_delay_mean[name].append(
                float(np.mean(sample_values)) if len(sample_values) else np.nan)
            sampled_diffs = percentile_diffs(sample_values, run_gt_percentiles)
            for q in delay_percentiles:
                sampled_percentile_diff[q][name].append(sampled_diffs[q])
                sampled_percentile_reldiff[q][name].append(
                    _reldiff(sampled_diffs[q], run_gt_percentiles, q))
            sampled_percentile_avg_relerror[name].append(
                percentile_avg_relative_error(run_gt_values, sample_values))
            for metric in PROB_METRIC_KEYS:
                _record_prob(prob_results[metric]['sampled'][name], metric, selected_rows,
                              run_gt_probs, run_agg_stats)

            # Spend exactly this method's sample budget on a blind uniform subsample,
            # so the two differ only in *which* packets they pick, not how many. With no
            # certified window there is nothing to draw from and no reference to score
            # against, so the baseline is skipped along with the method itself.
            if not certified:
                uniform_emd[name].append(np.nan)
                uniform_emd_norm[name].append(np.nan)
                uniform_consistency[name].append(None)
                uniform_mean_diff[name].append(np.nan)
                uniform_sample_sizes[name].append(0)
                uniform_values = np.array([])
                uniform_rows = run_subset
                emd = np.nan
                mean_diff = np.nan
            else:
                target_count = matched_uniform_target_count(sampled_size, run_min_samples)
                uniform_rows = sample_uniform_count(run_subset, target_count)
                uniform_values = received_rows(uniform_rows)['Delay'].values
                emd, consistency_pass, mean_diff, uniform_size = _evaluate_delay_family(
                    uniform_values, run_gt_values, run_agg_stats, confidenceValue, min_sample_size)
                uniform_emd[name].append(emd)
                uniform_emd_norm[name].append(normalize_emd_values(emd, run_gt_mean))
                uniform_consistency[name].append(consistency_pass if uniform_size else None)
                uniform_mean_diff[name].append(mean_diff)
                uniform_sample_sizes[name].append(uniform_size)
            bound_ns, bound_rel = delay_consistency_error_bound(
                run_agg_stats, len(uniform_values), confidenceValue)
            uniform_error_bound[name].append(bound_rel)
            uniform_error_bound_ns[name].append(bound_ns)
            uniform_delay_mean[name].append(
                float(np.mean(uniform_values)) if len(uniform_values) else np.nan)
            uniform_diffs = percentile_diffs(uniform_values, run_gt_percentiles)
            for q in delay_percentiles:
                uniform_percentile_diff[q][name].append(uniform_diffs[q])
                uniform_percentile_reldiff[q][name].append(
                    _reldiff(uniform_diffs[q], run_gt_percentiles, q))
            uniform_percentile_avg_relerror[name].append(
                percentile_avg_relative_error(run_gt_values, uniform_values))
            for metric in PROB_METRIC_KEYS:
                _record_prob(prob_results[metric]['uniform'][name], metric,
                              uniform_rows if certified else run_subset.iloc[0:0],
                              run_gt_probs, run_agg_stats)

            uniform_tests = poisson_process_tests(
                uniform_rows['SentTime'].values, steady_start, window_end,
                lags=test_lags, run_chi_squared=run_chi)
            uniform_test_split[name]['emd'].append(emd)
            uniform_test_split[name]['emd_normalized'].append(normalize_emd_values(emd, run_gt_mean))
            uniform_test_split[name]['mean_diff'].append(mean_diff)
            uniform_test_split[name]['ad_pass'].append(uniform_tests['ad_pass'])
            uniform_test_split[name]['chi_pass'].append(uniform_tests['chi_pass'])

        # ------------------------------------------------------------- ideal Poisson probes
        # The ceiling: same construction as the ground truth, at the sample budget each real
        # method actually achieved (plus the bare minimum budget, plus all packets' own
        # count/rate with no subsampling at all) -- all of it inside the run's window, and
        # scored against that window's ground truth and statistics.
        targets = oracle_target_counts(
            {name: sampled_sample_sizes[name][-1] for name in subsampling_methods},
            run_min_samples, all_packets_size=len(all_values))
        for key in oracle_series:
            if not certified:
                oracle_emd[key].append(np.nan)
                oracle_emd_norm[key].append(np.nan)
                oracle_consistency[key].append(None)
                oracle_mean_diff[key].append(np.nan)
                oracle_sample_sizes[key].append(0)
                oracle_values = np.array([])
            else:
                oracle_values = construct_oracle_poisson_delays(
                    prepared['groundtruth_method'], prepared['queue_names'], prepared['dir_prefix'],
                    steady_start, window_end, prepared['link_delays'],
                    prepared['link_rates'], targets[key])
                emd, consistency_pass, mean_diff, oracle_size = _evaluate_delay_family(
                    oracle_values, run_gt_values, run_agg_stats, confidenceValue, min_sample_size)
                oracle_emd[key].append(emd)
                oracle_emd_norm[key].append(normalize_emd_values(emd, run_gt_mean))
                oracle_consistency[key].append(consistency_pass if oracle_size else None)
                oracle_mean_diff[key].append(mean_diff)
                oracle_sample_sizes[key].append(oracle_size)
            bound_ns, bound_rel = delay_consistency_error_bound(
                run_agg_stats, len(oracle_values), confidenceValue)
            oracle_error_bound[key].append(bound_rel)
            oracle_error_bound_ns[key].append(bound_ns)
            oracle_delay_mean[key].append(
                float(np.mean(oracle_values)) if len(oracle_values) else np.nan)
            oracle_diffs = percentile_diffs(oracle_values, run_gt_percentiles)
            for q in delay_percentiles:
                oracle_percentile_diff[q][key].append(oracle_diffs[q])
                oracle_percentile_reldiff[q][key].append(
                    _reldiff(oracle_diffs[q], run_gt_percentiles, q))
            oracle_percentile_avg_relerror[key].append(
                percentile_avg_relative_error(run_gt_values, oracle_values))
            # The probe's loss/marking estimate is not a packet subsample at all: it is the
            # same switch-side construction as the reference, run at this family's budget --
            # so it carries only finite-sample noise, the same ceiling argument the delay
            # probe rests on.
            probe_probs = (construct_oracle_path_probs(
                prepared['queue_names'], prepared['dir_prefix'], steady_start, window_end,
                prepared['link_delays'], prepared['link_rates'], targets[key])
                if certified else {metric: np.nan for metric in PROB_METRIC_KEYS})
            for metric in PROB_METRIC_KEYS:
                result = evaluate_prob_estimate(
                    probe_probs.get(metric, np.nan), targets[key],
                    run_gt_probs.get(metric, np.nan), run_agg_stats, metric,
                    confidenceValue, min_sample_size)
                for field in PROB_FAMILY_FIELDS:
                    prob_results[metric]['oracle'][key][field].append(result[field])

    return {
        'num_flows': num_flows_list,
        'subsampling_methods': subsampling_methods,
        'all_packets_windowed': windowed_method is not None,
        'consistency_pass_all_packets': consistency_all_list,
        'mean_diff_all_packets': mean_diff_all_list,
        'emd_all_packets': emd_all_list,
        'emd_all_packets_normalized': emd_all_norm_list,
        'all_packet_size': size_all_list,
        'error_bound_all_packets': bound_all_list,
        'delay_mean_all_packets': delay_mean_all_list,
        'groundtruth_delay_mean': groundtruth_delay_mean_list,
        'error_bound_ns_all_packets': bound_ns_all_list,
        'percentile_diff_all_packets': pdiff_all,
        'percentile_reldiff_all_packets': preldiff_all,
        'percentile_avg_relerror_all_packets': pctrelerr_all_list,
        'poisson_tests_all_packets': tests_all,
        'burstiness_all_packets': burstiness_all,
        'sampled_emd': sampled_emd,
        'sampled_emd_normalized': sampled_emd_norm,
        'sampled_consistency': sampled_consistency,
        'sampled_mean_diff': sampled_mean_diff,
        'sampled_sample_sizes': sampled_sample_sizes,
        'sampled_error_bound': sampled_error_bound,
        'sampled_delay_mean': sampled_delay_mean,
        'sampled_error_bound_ns': sampled_error_bound_ns,
        'sampled_window_duration': sampled_window_duration,
        'uniform_emd': uniform_emd,
        'uniform_emd_normalized': uniform_emd_norm,
        'uniform_consistency': uniform_consistency,
        'uniform_mean_diff': uniform_mean_diff,
        'uniform_sample_sizes': uniform_sample_sizes,
        'uniform_error_bound': uniform_error_bound,
        'uniform_delay_mean': uniform_delay_mean,
        'uniform_error_bound_ns': uniform_error_bound_ns,
        'sampled_percentile_diff': sampled_percentile_diff,
        'sampled_percentile_reldiff': sampled_percentile_reldiff,
        'uniform_percentile_diff': uniform_percentile_diff,
        'uniform_percentile_reldiff': uniform_percentile_reldiff,
        'sampled_percentile_avg_relerror': sampled_percentile_avg_relerror,
        'uniform_percentile_avg_relerror': uniform_percentile_avg_relerror,
        'oracle_series': oracle_series,
        'oracle_emd': oracle_emd,
        'oracle_emd_normalized': oracle_emd_norm,
        'oracle_consistency': oracle_consistency,
        'oracle_mean_diff': oracle_mean_diff,
        'oracle_sample_sizes': oracle_sample_sizes,
        'oracle_error_bound': oracle_error_bound,
        'oracle_delay_mean': oracle_delay_mean,
        'oracle_error_bound_ns': oracle_error_bound_ns,
        'oracle_percentile_diff': oracle_percentile_diff,
        'oracle_percentile_reldiff': oracle_percentile_reldiff,
        'oracle_percentile_avg_relerror': oracle_percentile_avg_relerror,
        'uniform_test_split': uniform_test_split,
        'prob_metrics': prob_results,
    }


def _run_one_poisson_run(prepared, dir_prefix, queue_names, linkDelays, linkRates, steadyStart, steadyEnd,
                          num_poisson_observations, confidenceValue, DelayConsistencyGaurantee, min_sample_size,
                          subsampling_methods, step_ns=GROWING_WINDOW_STEP_NS):
    agg_stats = compute_poisson_agg_stats(
        dir_prefix, queue_names, linkDelays, linkRates, steadyStart, steadyEnd,
        num_poisson_observations, confidenceValue, DelayConsistencyGaurantee,
    )
    # Built per run, not once: a growing-window method redraws its candidate windows' own
    # switch-side statistics every run, exactly as agg_stats above is redrawn.
    window_ctx = growing_window_context(prepared, confidenceValue, DelayConsistencyGaurantee,
                                         num_poisson_observations, step_ns=step_ns)
    return compute_emd_vs_num_tcp_flows_run(prepared, agg_stats, confidenceValue, min_sample_size,
                                             subsampling_methods, window_ctx=window_ctx)


def _poisson_run_worker(return_dict, run_indices, prepared, dir_prefix, queue_names, linkDelays, linkRates,
                         steadyStart, steadyEnd, num_poisson_observations, confidenceValue,
                         DelayConsistencyGaurantee, min_sample_size, subsampling_methods,
                         step_ns=GROWING_WINDOW_STEP_NS):
    # A forked worker inherits the parent's numpy random state verbatim, so without
    # reseeding here every worker would draw the exact same "independent" runs.
    np.random.seed()
    for idx in run_indices:
        return_dict[idx] = _run_one_poisson_run(
            prepared, dir_prefix, queue_names, linkDelays, linkRates, steadyStart, steadyEnd,
            num_poisson_observations, confidenceValue, DelayConsistencyGaurantee, min_sample_size,
            subsampling_methods, step_ns=step_ns,
        )


def _run_poisson_runs(prepared, dir_prefix, queue_names, linkDelays, linkRates, steadyStart, steadyEnd,
                       num_poisson_observations, confidenceValue, DelayConsistencyGaurantee, min_sample_size,
                       num_runs, num_workers, subsampling_methods, step_ns=GROWING_WINDOW_STEP_NS):
    if num_workers is None or num_workers <= 1:
        return [
            _run_one_poisson_run(prepared, dir_prefix, queue_names, linkDelays, linkRates, steadyStart, steadyEnd,
                                  num_poisson_observations, confidenceValue, DelayConsistencyGaurantee, min_sample_size,
                                  subsampling_methods, step_ns=step_ns)
            for _ in range(num_runs)
        ]

    num_workers = max(1, min(num_workers, num_runs))
    return_dict = multiprocessing.Manager().dict()
    processes = []
    for worker in range(num_workers):
        run_indices = list(range(worker, num_runs, num_workers))
        if not run_indices:
            continue
        p = multiprocessing.Process(
            target=_poisson_run_worker,
            args=(return_dict, run_indices, prepared, dir_prefix, queue_names, linkDelays, linkRates,
                  steadyStart, steadyEnd, num_poisson_observations, confidenceValue,
                  DelayConsistencyGaurantee, min_sample_size, subsampling_methods, step_ns),
        )
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    missing = [i for i in range(num_runs) if i not in return_dict]
    if missing:
        # A worker that raised leaves its runs absent from the shared dict; indexing it
        # blindly used to surface as a bare KeyError from the manager, which says nothing
        # about the real failure (the child's traceback is printed above, on its own stderr).
        raise RuntimeError(
            "{} of {} Poisson runs produced no result -- a worker process died (its "
            "traceback is printed above, before this error). Missing run indices: {}{}".format(
                len(missing), num_runs, missing[:10], '...' if len(missing) > 10 else ''))
    return [return_dict[i] for i in range(num_runs)]


def _collect_one_run_delay_cdfs(prepared, agg_stats, min_sample_size,
                                 subsampling_methods='find_samples_path', window_ctx=None):
    """For a single concrete Poisson-process realization (`agg_stats`, as
    produced by one call to compute_poisson_agg_stats), collect the raw
    per-packet delay values -- not just their EMD summary -- for every
    subsampling method being compared against the ground-truth CDF: all
    packets of every currently-considered flow, one Poisson-adaptive
    subsample per entry in `subsampling_methods` (keys of
    POISSON_SUBSAMPLING_METHODS), and, for each of those, its rate-matched
    uniform counterpart drawing the same number of packets
    (sample_uniform_count), and the ideal-Poisson-probe families at those same
    budgets (construct_oracle_poisson_delays -- the ceiling, no selection bias
    by construction). Uses the full flow_order (all considered flows)
    since this is meant to illustrate what each method's delay distribution
    actually looks like, not to sweep over flow count. See
    plot_one_run_delay_cdfs for the corresponding plot.
    """
    full_df = prepared['full_df']
    subset = full_df[full_df['FlowRank'] <= len(prepared['flow_order'])]
    times = subset['SentTime'].values
    min_samples = agg_stats.get('MinimumE2ESampleSizeDelay', 0)
    subsampling_methods = normalize_subsampling_methods(subsampling_methods)

    # All flows here, so one cache slice: the full flow count's.
    k_delta_cache = (prepared.get('delta_cache') or {}).get(len(prepared['flow_order']))
    poisson_values, uniform_values = {}, {}
    # The window each method's values came from, and (for a growing-window method, whose
    # window is a prefix of the steady period) that window's own ground truth -- so the CDF
    # plot can show a windowed family next to the reference it was actually scored against
    # rather than only next to the full-window one.
    window_ends, groundtruth_by_method = {}, {}
    for name in subsampling_methods:
        windowed = window_ctx is not None and name in GROWING_WINDOW_SUBSAMPLING_METHODS
        if windowed:
            found = find_samples_growing_window_with_stats(
                name, times, window_ctx, delta_cache=k_delta_cache)
            samples_times, sub_err = found['samples'], found['error']
            window_end = found['window_end'] if found['window_end'] is not None else prepared['steady_end']
            fam_subset = subset[subset['SentTime'] <= window_end]
            fam_min_samples = found['min_samples'] if found['min_samples'] is not None else min_samples
            groundtruth_by_method[name] = windowed_groundtruth(window_ctx, float(window_end))[0]
        else:
            samples_times, sub_err = call_subsampling_method(
                name, times, min_samples, prepared['steady_start'], prepared['steady_end'],
                delta_cache=k_delta_cache)
            window_end, fam_subset, fam_min_samples = prepared['steady_end'], subset, min_samples
        window_ends[name] = window_end
        if sub_err != SubSamplingError.NoError or len(samples_times) == 0:
            poisson_values[name] = np.array([])
        else:
            poisson_values[name] = fam_subset[fam_subset['SentTime'].isin(samples_times)]['Delay'].values
        target_count = matched_uniform_target_count(len(poisson_values[name]), fam_min_samples)
        uniform_values[name] = sample_uniform_count(fam_subset, target_count)['Delay'].values

    # In a windowed run the all-packets curve is the packets inside the run's window, the
    # same set every other family here was drawn from and scored against.
    windowed_method = growing_window_method_in(subsampling_methods) if window_ctx is not None else None
    all_packets_end = window_ends.get(windowed_method, prepared['steady_end'])
    all_values = subset[subset['SentTime'] <= all_packets_end]['Delay'].values

    targets = oracle_target_counts({name: len(poisson_values[name]) for name in subsampling_methods},
                                    min_samples, all_packets_size=len(all_values))
    oracle_values = {
        key: construct_oracle_poisson_delays(
            prepared['groundtruth_method'], prepared['queue_names'], prepared['dir_prefix'],
            prepared['steady_start'], window_ends.get(key, prepared['steady_end']),
            prepared['link_delays'], prepared['link_rates'], target)
        for key, target in targets.items()}

    return {
        'all_packets': all_values,
        'poisson_subsample_by_method': poisson_values,
        'uniform': uniform_values,
        'oracle': oracle_values,
        'window_end_by_method': window_ends,
        # The ground truth this realization's families were scored against: the run's own
        # window in a windowed run (empty otherwise, where the result's full-window
        # 'groundtruth_values' is already the right reference).
        'groundtruth_by_method': groundtruth_by_method,
    }


def _one_run_poisson_subsamples(one_run):
    """The per-method Poisson-adaptive subsample values of a one-run CDF record
    (see _collect_one_run_delay_cdfs), as an ordered {method: values} dict.
    Understands the older single-method record shape ('poisson_subsample', a
    bare array) so one-run CDFs inside results pickles written before multiple
    methods per run existed still plot."""
    by_method = one_run.get('poisson_subsample_by_method')
    if by_method is not None:
        return dict(by_method)
    if 'poisson_subsample' in one_run:
        return {'find_samples_path': one_run['poisson_subsample']}
    return {}


_GROUNDTRUTH_LABELS = {
    'simultaneous': 'Simultaneous network queues (ground truth)',
    'path_observation': 'Path-observing probe through network queues (ground truth)',
}


def groundtruth_method_label(groundtruth_method):
    """Human-readable name of a ground-truth construction (GROUNDTRUTH_METHODS),
    for plot legends/titles and the results text file."""
    return _GROUNDTRUTH_LABELS.get(groundtruth_method, str(groundtruth_method))


def plot_one_run_delay_cdfs(results, output_path, title="Delay CDF comparison (one run)"):
    """Plot the ground-truth reconstructed delay CDF against every
    subsampling method's delay CDF from a single concrete Poisson-process
    realization (see _collect_one_run_delay_cdfs / results['one_run_delay_cdfs']):
    all packets of the considered flows, one Poisson-adaptive subsample per
    entry in results['subsampling_methods'], one uniform subsample per
    entry in results['uniform_series'] (for current results, one per
    Poisson-adaptive method, drawing that method's own sample count), and one
    ideal Poisson probe per entry in results['oracle_series']. Unlike
    the EMD/mean-diff boxplots (which summarize across all `num_runs` runs),
    this shows one concrete instance so it's visually obvious what each
    method's delay distribution actually looks like next to the ground
    truth."""
    results = upgrade_emd_vs_flows_results_schema(results)
    one_run = results['one_run_delay_cdfs']
    poisson_by_method = _one_run_poisson_subsamples(one_run)
    methods = [m for m in results['subsampling_methods'] if m in poisson_by_method] or list(poisson_by_method)

    # Long enough that a run comparing several methods (each bringing a uniform baseline, an
    # ideal probe and, for a growing-window method, its own window's ground truth) does not
    # start reusing colours: three methods already need 14 series.
    palette = ['C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'navy', 'darkorange',
                'darkgreen', 'crimson', 'purple', 'saddlebrown', 'teal', 'magenta',
                'olive', 'dimgray']
    extra_series = []
    # The first Poisson-adaptive method fills plot_delay_distribution_cdfs' fixed
    # "subsampled" slot; any further method is just another extra series, exactly
    # like the uniform strides.
    for i, method in enumerate(methods[1:], start=1):
        extra_series.append((poisson_by_method[method],
                              'Poisson-adaptive subsample ({})'.format(method),
                              palette[i % len(palette)]))
    for j, key in enumerate(results.get('uniform_series', []), start=len(methods)):
        extra_series.append((one_run['uniform'].get(key, []),
                              _uniform_series_label(key),
                              palette[j % len(palette)]))
    start = len(methods) + len(results.get('uniform_series', []))
    for j, key in enumerate(results.get('oracle_series', []), start=start):
        extra_series.append(((one_run.get('oracle') or {}).get(key, []),
                              _oracle_series_label(key),
                              palette[j % len(palette)]))

    first_method = methods[0] if methods else None
    # In a windowed run every curve here -- the ground truth included -- covers the window
    # that realization's search settled on, so say which window that was rather than letting
    # the reader assume the full steady period.
    gt_label = groundtruth_method_label(results.get('groundtruth_method', 'simultaneous'))
    window_method = results.get('analysis_window_method')
    window_end = (one_run.get('window_end_by_method') or {}).get(window_method)
    steady_start = results.get('steady_start')
    if window_method and window_end is not None and steady_start is not None:
        gt_label += ', {:.3g} ms window'.format((float(window_end) - float(steady_start)) / 1e6)
        title += '\nall series over this run\'s {:.3g} ms window'.format(
            (float(window_end) - float(steady_start)) / 1e6)
    return plot_delay_distribution_cdfs(
        results['groundtruth_values'],
        one_run['all_packets'],
        poisson_by_method.get(first_method, np.array([])),
        output_path,
        title=title,
        extra_series=extra_series,
        subsampled_label='Poisson-adaptive subsample ({})'.format(first_method),
        groundtruth_label=gt_label,
    )


def upgrade_emd_vs_flows_results_schema(results):
    """Bring a results dict up to the current schema, so results pickles written
    before this pipeline supported several Poisson-adaptive subsampling methods
    per run (and before EMD normalization) still plot and aggregate.

    Older dicts carried exactly one Poisson-adaptive family in the singular keys
    'emd_sampled_packets_by_run' / 'pass_rate_sampled' / 'mean_diff_sampled_by_run'
    (plain per-k lists). Here those become {method: per-k list} dicts keyed by
    that run's single 'subsampling_method', which is what every consumer now
    expects. Returns the dict unchanged (not a copy) when it is already current,
    and never mutates its input otherwise."""
    if ('subsampling_methods' in results and 'uniform_series' in results
            and 'delay_percentiles' in results and 'oracle_series' in results
            and 'poisson_test_series' in results
            and 'percentile_avg_relerror_all_packets' in results
            and 'window_duration_sampled_by_run' in results and 'all_packet_sizes' in results
            and 'error_bound_sampled_by_run' in results and 'prob_metrics' in results
            and 'delay_mean_sampled_by_run' in results
            and isinstance(results.get('emd_sampled_packets_by_run'), dict)):
        return results

    upgraded = dict(results)
    method = upgraded.get('subsampling_method', 'find_samples_path')
    upgraded['subsampling_methods'] = list(upgraded.get('subsampling_methods') or [method])
    for key in ('emd_sampled_packets_by_run', 'emd_sampled_packets_by_run_normalized',
                 'pass_rate_sampled', 'mean_diff_sampled_by_run', 'sample_sizes_sampled_by_run'):
        if key in upgraded and not isinstance(upgraded[key], dict):
            upgraded[key] = {method: upgraded[key]}
    upgraded.setdefault('groundtruth_method', 'simultaneous')
    upgraded.setdefault('all_flows_only', False)
    # Poisson-ness testing of the non-Poissonized families postdates these pickles and
    # needs the raw per-run instants, which were never stored -- so an old result carries
    # no verdicts and the split plots/tables are skipped for it rather than faked.
    if 'poisson_test_series' not in upgraded:
        upgraded['poisson_test_series'] = []
        upgraded['run_chi_squared_test'] = False
        upgraded['poisson_tests_all_packets'] = {}
        upgraded['uniform_test_split_by_run'] = {}
    # Uniform families used to be a fixed set of integer "1-in-stride" rates; they are
    # now one rate-matched family per Poisson-adaptive method, keyed by that method's
    # name. Either way they are enumerated by 'uniform_series', so both shapes plot.
    if 'uniform_series' not in upgraded:
        upgraded['uniform_series'] = list(upgraded.get('uniform_sample_strides') or [])
    # Per-run retained sample counts were not recorded before rate-matched uniform
    # sampling made them worth reporting; an empty record per k simply shows as "n/a".
    n_k = len(upgraded.get('num_flows', []))
    upgraded.setdefault('sample_sizes_sampled_by_run',
                         {m: [[] for _ in range(n_k)] for m in upgraded['subsampling_methods']})
    upgraded.setdefault('sample_sizes_uniform_by_run',
                         {s: [[] for _ in range(n_k)] for s in upgraded['uniform_series']})
    # Percentile errors postdate these pickles entirely. Recovering them would need the
    # raw per-run family values, which were never stored, so an old result simply carries
    # no percentiles and every percentile table/plot is skipped for it rather than faked.
    # The ideal-Poisson-probe family postdates these pickles; like the percentiles it
    # cannot be recovered without redoing the run, so an old result simply carries none
    # and every oracle table/plot is skipped for it rather than faked.
    if 'oracle_series' not in upgraded:
        upgraded['oracle_series'] = []
        upgraded['emd_oracle_by_run'] = {}
        upgraded['emd_oracle_by_run_normalized'] = {}
        upgraded['pass_rate_oracle'] = {}
        upgraded['mean_diff_oracle_by_run'] = {}
        upgraded['sample_sizes_oracle_by_run'] = {}
        upgraded['percentile_diff_oracle_by_run'] = {}
        upgraded['percentile_reldiff_oracle_by_run'] = {}
        upgraded['percentile_avg_relerror_oracle_by_run'] = {}
    if 'delay_percentiles' not in upgraded:
        upgraded['delay_percentiles'] = []
        upgraded['groundtruth_percentiles'] = {}
        upgraded['percentile_diff_all_packets'] = {}
        upgraded['percentile_reldiff_all_packets'] = {}
        upgraded['percentile_diff_sampled_by_run'] = {}
        upgraded['percentile_reldiff_sampled_by_run'] = {}
        upgraded['percentile_diff_uniform_by_run'] = {}
        upgraded['percentile_reldiff_uniform_by_run'] = {}
        upgraded['percentile_diff_oracle_by_run'] = {}
        upgraded['percentile_reldiff_oracle_by_run'] = {}
    # percentile_avg_relerror_* (percentile_avg_relative_error) postdates these pickles
    # entirely and, unlike the normalized-EMD backfill below, cannot be recovered from
    # anything already stored: it is computed from each run's raw retained sample
    # *values*, which are discarded after each run rather than persisted (only the final
    # EMD/percentile-diff-at-tracked-q survive). So an old result simply carries NaN/empty
    # placeholders here and every such table/plot is skipped for it rather than faked --
    # getting real values means rerunning run_emd_vs_flows_experiment for that
    # combination. Shaped so they are safe to index into directly (see the oracle_series/
    # delay_percentiles branches above for why bare {} is fine there but not here): a
    # plain per-k NaN list for all_packets, and a per-method/key empty-per-run-list-per-k
    # dict for sampled/uniform/oracle, so aggregating this experiment alongside others
    # that do have real values never index/key-errors into it.
    if 'percentile_avg_relerror_all_packets' not in upgraded:
        upgraded['percentile_avg_relerror_all_packets'] = [float('nan')] * n_k
        upgraded['percentile_avg_relerror_sampled_by_run'] = {
            m: [[] for _ in range(n_k)] for m in upgraded['subsampling_methods']}
        upgraded['percentile_avg_relerror_uniform_by_run'] = {
            s: [[] for _ in range(n_k)] for s in upgraded['uniform_series']}
        upgraded['percentile_avg_relerror_oracle_by_run'] = {
            key: [[] for _ in range(n_k)] for key in upgraded['oracle_series']}
    # Burstiness metrics postdate these pickles. Unlike the percentiles/oracle probe above,
    # they ARE cheaply recoverable without redoing the run (they only need the same raw
    # packet CSV, not a fresh multi-run Poisson sweep) -- see backfill_burstiness_metrics --
    # but this function only has the pickle itself to work with, so it fills in NaN
    # placeholders here and the real values come from running that backfill separately.
    if 'burstiness_all_packets' not in upgraded:
        upgraded.setdefault('burst_gap_threshold_ns', float('nan'))
        upgraded['burstiness_all_packets'] = {field: [float('nan')] * n_k for field in BURSTINESS_METRIC_LABELS}
    upgraded.setdefault('burstiness_all_packets_by_experiment',
                         {field: [[] for _ in range(n_k)] for field in BURSTINESS_METRIC_LABELS})
    # The all-packets packet count per k and the per-run monitoring-window durations both
    # postdate these pickles. Neither can be recovered from what was stored, so an old
    # result carries NaN/empty placeholders and the sample-size plot's all-packets reference
    # (and the monitoring-window plot entirely) is skipped for it rather than faked.
    upgraded.setdefault('all_packet_sizes', [float('nan')] * n_k)
    upgraded.setdefault('window_duration_sampled_by_run',
                         {m: [[] for _ in range(n_k)] for m in upgraded['subsampling_methods']})
    # Consistency-check error bounds (delay_consistency_error_bound) postdate these pickles
    # too. They are derivable in principle (bound = f(agg_stats, n)) but the per-run
    # agg_stats were never stored, so an old result carries empty placeholders and the
    # error-bound plot/columns are skipped for it rather than faked.
    # The loss/marking metrics postdate these pickles and cannot be recovered from what was
    # stored (they need the per-run packet outcomes and switch probabilities), so an old
    # result simply carries none and every probability table/plot is skipped for it.
    upgraded.setdefault('prob_metrics', {})
    # Per-family mean delays postdate these pickles; they are not recoverable from the
    # stored summaries, so an old result carries empty placeholders and the delay-value
    # plot/columns are skipped for it.
    upgraded.setdefault('delay_mean_all_packets_by_run', [[] for _ in range(n_k)])
    upgraded.setdefault('groundtruth_delay_mean_by_run', [[] for _ in range(n_k)])
    upgraded.setdefault('delay_mean_sampled_by_run',
                         {m: [[] for _ in range(n_k)] for m in upgraded['subsampling_methods']})
    upgraded.setdefault('delay_mean_uniform_by_run',
                         {s: [[] for _ in range(n_k)] for s in upgraded['uniform_series']})
    upgraded.setdefault('delay_mean_oracle_by_run',
                         {key: [[] for _ in range(n_k)] for key in upgraded['oracle_series']})
    for prefix in ('error_bound_', 'error_bound_ns_'):
        upgraded.setdefault(prefix + 'all_packets_by_run', [[] for _ in range(n_k)])
        upgraded.setdefault(prefix + 'sampled_by_run',
                             {m: [[] for _ in range(n_k)] for m in upgraded['subsampling_methods']})
        upgraded.setdefault(prefix + 'uniform_by_run',
                             {s: [[] for _ in range(n_k)] for s in upgraded['uniform_series']})
        upgraded.setdefault(prefix + 'oracle_by_run',
                             {key: [[] for _ in range(n_k)] for key in upgraded['oracle_series']})
    if 'groundtruth_mean' not in upgraded:
        gt = np.asarray(upgraded.get('groundtruth_values', []), dtype=float)
        upgraded['groundtruth_mean'] = float(np.mean(gt)) if gt.size else np.nan
    # Normalized EMDs are derived from the raw ones, so an older pickle can be
    # brought fully up to date without recomputing anything.
    gt_mean = upgraded['groundtruth_mean']
    if 'emd_all_packets_normalized' not in upgraded and 'emd_all_packets' in upgraded:
        upgraded['emd_all_packets_normalized'] = normalize_emd_values(upgraded['emd_all_packets'], gt_mean)
    if 'emd_all_packets_by_experiment' in upgraded and 'emd_all_packets_by_experiment_normalized' not in upgraded:
        upgraded['emd_all_packets_by_experiment_normalized'] = normalize_emd_values(
            upgraded['emd_all_packets_by_experiment'], gt_mean)
    if 'emd_sampled_packets_by_run_normalized' not in upgraded and 'emd_sampled_packets_by_run' in upgraded:
        upgraded['emd_sampled_packets_by_run_normalized'] = normalize_emd_values(
            upgraded['emd_sampled_packets_by_run'], gt_mean)
    if 'emd_uniform_packets_by_run_normalized' not in upgraded and 'emd_uniform_packets_by_run' in upgraded:
        upgraded['emd_uniform_packets_by_run_normalized'] = normalize_emd_values(
            upgraded['emd_uniform_packets_by_run'], gt_mean)
    return upgraded


# The one and only 'num_flows' entry an all_flows_only result is pooled onto by
# aggregate_emd_vs_flows_results (see there) -- never a real flow count (always >=1), so
# plotting code can check for it unambiguously and label that tick 'all packets' instead of
# showing this sentinel as if it meant something.
ALL_FLOWS_ONLY_K = -1


def aggregate_emd_vs_flows_results(results_list):
    """Combine per-experiment results (each a dict returned by
    compute_emd_vs_num_tcp_flows_multi_run / loaded from a
    '..._emd_vs_num_flows_results.pkl' file, for the same traffic/rate/load but a
    different `experiment` index) into one aggregated results dict with the same
    keys/shape as a single-experiment result, so it can be fed directly into
    plot_emd_vs_num_flows_boxplot, plot_mean_diff_vs_num_flows, plot_one_run_delay_cdfs,
    and save_emd_vs_flows_results_text unchanged.

    Per-run lists (EMD/mean-diff by run, for every subsampling method) are concatenated
    across experiments, so a box now shows variability across both runs *and* experiments.
    Pass rates are re-derived from pooled pass/found counts (not averaged rates) so
    experiments with different `n_samp` are weighted correctly -- the same denominator
    convention used for pass_rate_sampled within a single experiment (see
    compute_emd_vs_num_tcp_flows_multi_run).

    Normalized EMDs (EMD relative to mean queuing delay, see normalize_emd_values) are
    concatenated from each experiment's own already-normalized values rather than
    re-derived from the pooled raw ones: every experiment reconstructs its own ground
    truth and therefore has its own normalizer, so normalizing must happen before
    pooling, not after. Relative percentile errors are pooled the same way and for the
    same reason (each experiment's own p90/p99 is its own reference). Percentiles are
    intersected rather than unioned across experiments -- a percentile only some
    experiments measured would otherwise produce boxes backed by an inconsistent subset;
    aggregating results computed with different `delay_percentiles` therefore keeps only
    the ones common to all (and legacy results, which carry none, contribute none).

    `num_flows` is the *union* of every experiment's flow-count list, since different
    experiment realizations (different random seeds) can end up with slightly different
    numbers of received TCP flows on the path -- using the intersection would silently
    drop the true maximum k whenever even one experiment fell short of it. An experiment
    missing a particular k simply doesn't contribute to that k (neither its values nor its
    run count), so 'num_flows' entries near the union's upper end are typically backed by
    fewer experiments than ones every experiment reached. Poisson-adaptive subsampling
    methods are unioned the same way, so aggregating a single-method run together with a
    multi-method one keeps whatever each actually measured; each method's rate-matched
    uniform family (see compute_emd_vs_num_tcp_flows_run) follows its method.

    The all-packets EMD is no longer a single fixed value once aggregated (each
    experiment reconstructs its own ground truth), so it becomes 'emd_all_packets_by_experiment'
    (one list per k, one entry per experiment) alongside the usual 'emd_all_packets' (now the
    per-k mean across experiments, kept for any caller that still expects a scalar).

    The one-run CDF (plot_one_run_delay_cdfs) and the ground-truth summary stats use the
    first experiment's data -- pooling raw per-packet delay samples across differently-seeded
    experiments would mix reconstructions that aren't really the same underlying distribution.
    Each input dict may carry an 'experiment' key (its experiment index/label); if absent,
    its position in `results_list` is used instead.
    """
    if not results_list:
        raise ValueError("aggregate_emd_vs_flows_results requires at least one results dict")

    results_list = [upgrade_emd_vs_flows_results_schema(r) for r in results_list]
    experiments = [r.get('experiment', i) for i, r in enumerate(results_list)]

    groundtruth_methods_seen = {r.get('groundtruth_method', 'simultaneous') for r in results_list}
    if len(groundtruth_methods_seen) > 1:
        raise ValueError(
            "Refusing to aggregate experiments computed against different ground truths: {}. "
            "EMDs measured against different ground-truth constructions are not comparable.".format(
                sorted(groundtruth_methods_seen)))

    # An all_flows_only experiment's own num_flows is just whatever total TCP flow count
    # that particular (randomly-seeded) experiment happened to receive on the path --
    # incidental per-experiment noise, not a swept independent variable. Pooling by the
    # literal value the way a real flow-count sweep is pooled would fragment what should be
    # one point into several near-identical ones (e.g. 22, 23, 24 flows) purely because
    # different experiments landed on slightly different counts. Remap every experiment onto
    # the one shared ALL_FLOWS_ONLY_K sentinel first so they all pool into a single bucket
    # below; plotting code recognizes the sentinel and labels that tick 'all packets'
    # instead.
    if all(r.get('all_flows_only', False) for r in results_list):
        results_list = [dict(r, num_flows=[ALL_FLOWS_ONLY_K]) for r in results_list]

    if len(results_list) == 1:
        result = dict(results_list[0])
        result['num_experiments'] = 1
        result['experiments'] = experiments
        # A single experiment's all-packets value at each k becomes a one-entry spread, so
        # every consumer can read the '..._by_experiment' shape uniformly. A WINDOWED result
        # already carries a real per-run spread in exactly these keys (each run measured the
        # family inside its own window), so those are left untouched -- collapsing them to
        # the per-k mean here would throw away the very variation they exist to show.
        if not result.get('all_packets_windowed'):
            result['percentile_diff_all_packets_by_experiment'] = {
                q: [[v] for v in per_k] for q, per_k in result['percentile_diff_all_packets'].items()}
            result['percentile_reldiff_all_packets_by_experiment'] = {
                q: [[v] for v in per_k] for q, per_k in result['percentile_reldiff_all_packets'].items()}
            result['percentile_avg_relerror_all_packets_by_experiment'] = [
                [v] for v in result['percentile_avg_relerror_all_packets']]
            result['emd_all_packets_by_experiment'] = [[v] for v in result['emd_all_packets']]
            result['emd_all_packets_by_experiment_normalized'] = [
                [v] for v in result['emd_all_packets_normalized']]
            result['all_packet_sizes_by_experiment'] = [[v] for v in result.get('all_packet_sizes') or []]
            result['burstiness_all_packets_by_experiment'] = {
                field: [([v] if v == v else []) for v in per_k]
                for field, per_k in (result.get('burstiness_all_packets') or {}).items()}
        return result

    all_k = sorted(set().union(*(set(r['num_flows']) for r in results_list)))
    if not all_k:
        raise ValueError("No flow-count (k) value found in any experiment to aggregate")

    methods = []
    for r in results_list:
        for name in r['subsampling_methods']:
            if name not in methods:
                methods.append(name)
    uniform_series = []
    for r in results_list:
        for key in r['uniform_series']:
            if key not in uniform_series:
                uniform_series.append(key)
    # Intersection, not union: a percentile only some experiments measured would give
    # boxes backed by a different set of experiments than their neighbours.
    percentiles = [q for q in results_list[0]['delay_percentiles']
                    if all(q in r['delay_percentiles'] for r in results_list[1:])]
    oracle_series = []
    for r in results_list:
        for key in r['oracle_series']:
            if key not in oracle_series:
                oracle_series.append(key)
    # Only families every experiment tested: a split backed by a subset of experiments at
    # some k and all of them at another would not be comparable across k.
    poisson_test_series = [key for key in results_list[0]['poisson_test_series']
                            if all(key in r['poisson_test_series'] for r in results_list[1:])]
    run_chi = all(r.get('run_chi_squared_test', False) for r in results_list)

    emd_all_by_experiment, emd_all_by_experiment_norm, mean_diff_all = [], [], []
    all_packet_sizes_by_experiment = []
    pctrelerr_all_by_experiment = []
    pass_all_count, pass_all_total = [], []
    emd_sampled_by_run = {m: [] for m in methods}
    emd_sampled_by_run_norm = {m: [] for m in methods}
    pctrelerr_sampled_by_run = {m: [] for m in methods}
    mean_diff_sampled = {m: [] for m in methods}
    pass_sampled_count = {m: [] for m in methods}
    pass_sampled_total = {m: [] for m in methods}
    sample_sizes_sampled = {m: [] for m in methods}
    window_durations_sampled = {m: [] for m in methods}
    bound_all, bound_ns_all = [], []
    delay_mean_all, groundtruth_delay_mean = [], []
    delay_mean_sampled = {m: [] for m in methods}
    delay_mean_uniform = {s: [] for s in uniform_series}
    delay_mean_oracle = {key: [] for key in oracle_series}
    bounds_sampled = {m: [] for m in methods}
    bounds_ns_sampled = {m: [] for m in methods}
    bounds_uniform = {s: [] for s in uniform_series}
    bounds_ns_uniform = {s: [] for s in uniform_series}
    bounds_oracle = {key: [] for key in oracle_series}
    bounds_ns_oracle = {key: [] for key in oracle_series}
    sample_sizes_uniform = {s: [] for s in uniform_series}
    pdiff_all = {q: [] for q in percentiles}
    preldiff_all = {q: [] for q in percentiles}
    pdiff_sampled = {q: {m: [] for m in methods} for q in percentiles}
    preldiff_sampled = {q: {m: [] for m in methods} for q in percentiles}
    pdiff_uniform = {q: {s: [] for s in uniform_series} for q in percentiles}
    preldiff_uniform = {q: {s: [] for s in uniform_series} for q in percentiles}
    emd_oracle_by_run = {key: [] for key in oracle_series}
    emd_oracle_by_run_norm = {key: [] for key in oracle_series}
    pctrelerr_oracle_by_run = {key: [] for key in oracle_series}
    mean_diff_oracle = {key: [] for key in oracle_series}
    pass_oracle_count = {key: [] for key in oracle_series}
    pass_oracle_total = {key: [] for key in oracle_series}
    sample_sizes_oracle = {key: [] for key in oracle_series}
    pdiff_oracle = {q: {key: [] for key in oracle_series} for q in percentiles}
    split_fields = ('emd', 'emd_normalized', 'mean_diff', 'ad_pass', 'chi_pass')
    uniform_split = {m: {field: [] for field in split_fields} for m in methods}
    tests_all_fields = ('ad_pass', 'ad_pvalue', 'chi_pass', 'chi_reject_fraction')
    tests_all_by_experiment = {field: [] for field in tests_all_fields}
    preldiff_oracle = {q: {key: [] for key in oracle_series} for q in percentiles}
    emd_uniform_by_run = {s: [] for s in uniform_series}
    emd_uniform_by_run_norm = {s: [] for s in uniform_series}
    pctrelerr_uniform_by_run = {s: [] for s in uniform_series}
    mean_diff_uniform = {s: [] for s in uniform_series}
    pass_uniform_count = {s: [] for s in uniform_series}
    pass_uniform_total = {s: [] for s in uniform_series}

    for k in all_k:
        emd_all_vals, emd_all_vals_norm, mean_diff_all_vals = [], [], []
        all_packet_size_vals = []
        pctrelerr_all_vals = []
        pass_all_c = pass_all_t = 0
        samp_emd_vals = {m: [] for m in methods}
        samp_emd_vals_norm = {m: [] for m in methods}
        samp_pctrelerr_vals = {m: [] for m in methods}
        samp_diff_vals = {m: [] for m in methods}
        samp_pass_c = {m: 0 for m in methods}
        samp_pass_t = {m: 0 for m in methods}
        samp_size_vals = {m: [] for m in methods}
        samp_window_vals = {m: [] for m in methods}
        bound_all_vals, bound_ns_all_vals = [], []
        delay_mean_all_vals, gt_delay_mean_vals = [], []
        samp_delay_mean_vals = {m: [] for m in methods}
        uniform_delay_mean_vals = {s: [] for s in uniform_series}
        oracle_delay_mean_vals = {key: [] for key in oracle_series}
        samp_bound_vals = {m: [] for m in methods}
        samp_bound_ns_vals = {m: [] for m in methods}
        uniform_bound_vals = {s: [] for s in uniform_series}
        uniform_bound_ns_vals = {s: [] for s in uniform_series}
        oracle_bound_vals = {key: [] for key in oracle_series}
        oracle_bound_ns_vals = {key: [] for key in oracle_series}
        uniform_size_vals = {s: [] for s in uniform_series}
        uniform_emd_vals = {s: [] for s in uniform_series}
        uniform_emd_vals_norm = {s: [] for s in uniform_series}
        uniform_pctrelerr_vals = {s: [] for s in uniform_series}
        uniform_diff_vals = {s: [] for s in uniform_series}
        uniform_pass_c = {s: 0 for s in uniform_series}
        uniform_pass_t = {s: 0 for s in uniform_series}
        pdiff_all_vals = {q: [] for q in percentiles}
        preldiff_all_vals = {q: [] for q in percentiles}
        pdiff_samp_vals = {q: {m: [] for m in methods} for q in percentiles}
        preldiff_samp_vals = {q: {m: [] for m in methods} for q in percentiles}
        pdiff_uni_vals = {q: {s: [] for s in uniform_series} for q in percentiles}
        preldiff_uni_vals = {q: {s: [] for s in uniform_series} for q in percentiles}
        oracle_emd_vals = {key: [] for key in oracle_series}
        oracle_emd_vals_norm = {key: [] for key in oracle_series}
        oracle_pctrelerr_vals = {key: [] for key in oracle_series}
        oracle_diff_vals = {key: [] for key in oracle_series}
        oracle_pass_c = {key: 0 for key in oracle_series}
        oracle_pass_t = {key: 0 for key in oracle_series}
        oracle_size_vals = {key: [] for key in oracle_series}
        uniform_split_vals = {m: {field: [] for field in split_fields} for m in methods}
        tests_all_vals = {field: [] for field in tests_all_fields}
        pdiff_ora_vals = {q: {key: [] for key in oracle_series} for q in percentiles}
        preldiff_ora_vals = {q: {key: [] for key in oracle_series} for q in percentiles}

        for r in results_list:
            if k not in r['num_flows']:
                continue
            i = r['num_flows'].index(k)
            num_runs = r['num_runs']

            # An ordinary experiment contributes ONE all-packets value per k (there the
            # family is a fixed packet set); a windowed one contributes its per-run spread,
            # since every run measured the family inside its own window. This pools
            # whichever of the two the experiment has, so a mixed-mode or mixed-vintage
            # aggregation still puts every value that exists into the same box.
            windowed_all = bool(r.get('all_packets_windowed'))

            def _all_packets(scalar_key, spread_key, index=i, result=r, windowed=windowed_all):
                spread = result.get(spread_key)
                if windowed and spread is not None:
                    return list(spread[index]) if index < len(spread) else []
                per_k = result.get(scalar_key) or []
                return [per_k[index]] if index < len(per_k) else []

            emd_all_contribution = _all_packets('emd_all_packets', 'emd_all_packets_by_experiment')
            emd_all_vals.extend(emd_all_contribution)
            emd_all_vals_norm.extend(_all_packets(
                'emd_all_packets_normalized', 'emd_all_packets_by_experiment_normalized'))
            all_packet_size_vals.extend(_all_packets(
                'all_packet_sizes', 'all_packet_sizes_by_experiment'))
            pctrelerr_all_vals.extend(_all_packets(
                'percentile_avg_relerror_all_packets',
                'percentile_avg_relerror_all_packets_by_experiment'))
            mean_diff_all_vals.extend(r['mean_diff_all_packets_by_run'][i])
            n_k_all = len(r['num_flows'])
            delay_mean_all_vals.extend((r.get('delay_mean_all_packets_by_run') or [[]] * n_k_all)[i])
            gt_delay_mean_vals.extend((r.get('groundtruth_delay_mean_by_run') or [[]] * n_k_all)[i])
            bound_all_vals.extend((r.get('error_bound_all_packets_by_run') or [[]] * len(r['num_flows']))[i])
            bound_ns_all_vals.extend((r.get('error_bound_ns_all_packets_by_run') or [[]] * len(r['num_flows']))[i])
            # A windowed experiment's all-packets pass rate is out of the runs that
            # certified a window (the ones that produced a value), not all num_runs -- the
            # same denominator convention pass_rate_sampled has always used.
            all_packets_total = len(emd_all_contribution) if windowed_all else num_runs
            pass_all_c += round(r['pass_rate_all_packets'][i] * all_packets_total)
            pass_all_t += all_packets_total

            for q in percentiles:
                if windowed_all:
                    pdiff_spread = (r.get('percentile_diff_all_packets_by_experiment') or {}).get(q)
                    preldiff_spread = (r.get('percentile_reldiff_all_packets_by_experiment') or {}).get(q)
                    pdiff_all_vals[q].extend(pdiff_spread[i] if pdiff_spread else [])
                    preldiff_all_vals[q].extend(preldiff_spread[i] if preldiff_spread else [])
                else:
                    pdiff_all_vals[q].append(r['percentile_diff_all_packets'][q][i])
                    preldiff_all_vals[q].append(r['percentile_reldiff_all_packets'][q][i])

            if 'all_packets' in poisson_test_series:
                for field in tests_all_fields:
                    verdicts = r['poisson_tests_all_packets'][field][i]
                    # A windowed experiment stores one verdict per run here; an ordinary one
                    # a single verdict for the whole experiment.
                    if isinstance(verdicts, (list, tuple, np.ndarray)):
                        tests_all_vals[field].extend(verdicts)
                    else:
                        tests_all_vals[field].append(verdicts)

            for m in r['subsampling_methods']:
                sampled_vals = r['emd_sampled_packets_by_run'][m][i]
                samp_emd_vals[m].extend(sampled_vals)
                samp_emd_vals_norm[m].extend(r['emd_sampled_packets_by_run_normalized'][m][i])
                samp_pctrelerr_vals[m].extend(r['percentile_avg_relerror_sampled_by_run'][m][i])
                samp_diff_vals[m].extend(r['mean_diff_sampled_by_run'][m][i])
                samp_size_vals[m].extend(r['sample_sizes_sampled_by_run'][m][i])
                # Absent from pickles written before monitoring windows were recorded; such
                # an experiment simply contributes no duration rather than a made-up one.
                samp_window_vals[m].extend(
                    (r.get('window_duration_sampled_by_run') or {}).get(m, [[]] * len(r['num_flows']))[i])
                samp_delay_mean_vals[m].extend(
                    (r.get('delay_mean_sampled_by_run') or {}).get(m, [[]] * len(r['num_flows']))[i])
                samp_bound_vals[m].extend(
                    (r.get('error_bound_sampled_by_run') or {}).get(m, [[]] * len(r['num_flows']))[i])
                samp_bound_ns_vals[m].extend(
                    (r.get('error_bound_ns_sampled_by_run') or {}).get(m, [[]] * len(r['num_flows']))[i])
                n_samp = len(sampled_vals)
                samp_pass_c[m] += round(r['pass_rate_sampled'][m][i] * n_samp)
                samp_pass_t[m] += n_samp
                for q in percentiles:
                    pdiff_samp_vals[q][m].extend(r['percentile_diff_sampled_by_run'][q][m][i])
                    preldiff_samp_vals[q][m].extend(r['percentile_reldiff_sampled_by_run'][q][m][i])

            for s in r['uniform_series']:
                uniform_emd_vals[s].extend(r['emd_uniform_packets_by_run'][s][i])
                uniform_emd_vals_norm[s].extend(r['emd_uniform_packets_by_run_normalized'][s][i])
                uniform_pctrelerr_vals[s].extend(r['percentile_avg_relerror_uniform_by_run'][s][i])
                uniform_diff_vals[s].extend(r['mean_diff_uniform_packets_by_run'][s][i])
                uniform_pass_c[s] += round(r['pass_rate_uniform'][s][i] * num_runs)
                uniform_pass_t[s] += num_runs
                uniform_size_vals[s].extend(r['sample_sizes_uniform_by_run'][s][i])
                uniform_delay_mean_vals[s].extend(
                    (r.get('delay_mean_uniform_by_run') or {}).get(s, [[]] * len(r['num_flows']))[i])
                uniform_bound_vals[s].extend(
                    (r.get('error_bound_uniform_by_run') or {}).get(s, [[]] * len(r['num_flows']))[i])
                uniform_bound_ns_vals[s].extend(
                    (r.get('error_bound_ns_uniform_by_run') or {}).get(s, [[]] * len(r['num_flows']))[i])
                for q in percentiles:
                    pdiff_uni_vals[q][s].extend(r['percentile_diff_uniform_by_run'][q][s][i])
                    preldiff_uni_vals[q][s].extend(r['percentile_reldiff_uniform_by_run'][q][s][i])

            for m in methods:
                if ('uniform', m) not in poisson_test_series or m not in r['uniform_test_split_by_run']:
                    continue
                for field in split_fields:
                    uniform_split_vals[m][field].extend(r['uniform_test_split_by_run'][m][field][i])

            for key in r['oracle_series']:
                oracle_emd_vals[key].extend(r['emd_oracle_by_run'][key][i])
                oracle_emd_vals_norm[key].extend(r['emd_oracle_by_run_normalized'][key][i])
                oracle_pctrelerr_vals[key].extend(r['percentile_avg_relerror_oracle_by_run'][key][i])
                oracle_diff_vals[key].extend(r['mean_diff_oracle_by_run'][key][i])
                oracle_size_vals[key].extend(r['sample_sizes_oracle_by_run'][key][i])
                oracle_delay_mean_vals[key].extend(
                    (r.get('delay_mean_oracle_by_run') or {}).get(key, [[]] * len(r['num_flows']))[i])
                oracle_bound_vals[key].extend(
                    (r.get('error_bound_oracle_by_run') or {}).get(key, [[]] * len(r['num_flows']))[i])
                oracle_bound_ns_vals[key].extend(
                    (r.get('error_bound_ns_oracle_by_run') or {}).get(key, [[]] * len(r['num_flows']))[i])
                oracle_pass_c[key] += round(r['pass_rate_oracle'][key][i] * num_runs)
                oracle_pass_t[key] += num_runs
                for q in percentiles:
                    pdiff_ora_vals[q][key].extend(r['percentile_diff_oracle_by_run'][q][key][i])
                    preldiff_ora_vals[q][key].extend(r['percentile_reldiff_oracle_by_run'][q][key][i])

        emd_all_by_experiment.append(emd_all_vals)
        emd_all_by_experiment_norm.append(emd_all_vals_norm)
        all_packet_sizes_by_experiment.append(all_packet_size_vals)
        pctrelerr_all_by_experiment.append(pctrelerr_all_vals)
        mean_diff_all.append(mean_diff_all_vals)
        bound_all.append(bound_all_vals)
        bound_ns_all.append(bound_ns_all_vals)
        delay_mean_all.append(delay_mean_all_vals)
        groundtruth_delay_mean.append(gt_delay_mean_vals)
        pass_all_count.append(pass_all_c)
        pass_all_total.append(pass_all_t)

        for q in percentiles:
            pdiff_all[q].append(pdiff_all_vals[q])
            preldiff_all[q].append(preldiff_all_vals[q])
            for m in methods:
                pdiff_sampled[q][m].append(pdiff_samp_vals[q][m])
                preldiff_sampled[q][m].append(preldiff_samp_vals[q][m])
            for s in uniform_series:
                pdiff_uniform[q][s].append(pdiff_uni_vals[q][s])
                preldiff_uniform[q][s].append(preldiff_uni_vals[q][s])
            for key in oracle_series:
                pdiff_oracle[q][key].append(pdiff_ora_vals[q][key])
                preldiff_oracle[q][key].append(preldiff_ora_vals[q][key])

        for m in methods:
            emd_sampled_by_run[m].append(samp_emd_vals[m])
            emd_sampled_by_run_norm[m].append(samp_emd_vals_norm[m])
            pctrelerr_sampled_by_run[m].append(samp_pctrelerr_vals[m])
            mean_diff_sampled[m].append(samp_diff_vals[m])
            pass_sampled_count[m].append(samp_pass_c[m])
            pass_sampled_total[m].append(samp_pass_t[m])
            sample_sizes_sampled[m].append(samp_size_vals[m])
            window_durations_sampled[m].append(samp_window_vals[m])
            bounds_sampled[m].append(samp_bound_vals[m])
            delay_mean_sampled[m].append(samp_delay_mean_vals[m])
            bounds_ns_sampled[m].append(samp_bound_ns_vals[m])

        for s in uniform_series:
            emd_uniform_by_run[s].append(uniform_emd_vals[s])
            emd_uniform_by_run_norm[s].append(uniform_emd_vals_norm[s])
            pctrelerr_uniform_by_run[s].append(uniform_pctrelerr_vals[s])
            mean_diff_uniform[s].append(uniform_diff_vals[s])
            pass_uniform_count[s].append(uniform_pass_c[s])
            pass_uniform_total[s].append(uniform_pass_t[s])
            sample_sizes_uniform[s].append(uniform_size_vals[s])
            bounds_uniform[s].append(uniform_bound_vals[s])
            delay_mean_uniform[s].append(uniform_delay_mean_vals[s])
            bounds_ns_uniform[s].append(uniform_bound_ns_vals[s])

        for m in methods:
            for field in split_fields:
                uniform_split[m][field].append(uniform_split_vals[m][field])
        for field in tests_all_fields:
            tests_all_by_experiment[field].append(tests_all_vals[field])

        for key in oracle_series:
            emd_oracle_by_run[key].append(oracle_emd_vals[key])
            emd_oracle_by_run_norm[key].append(oracle_emd_vals_norm[key])
            pctrelerr_oracle_by_run[key].append(oracle_pctrelerr_vals[key])
            mean_diff_oracle[key].append(oracle_diff_vals[key])
            pass_oracle_count[key].append(oracle_pass_c[key])
            pass_oracle_total[key].append(oracle_pass_t[key])
            sample_sizes_oracle[key].append(oracle_size_vals[key])
            bounds_oracle[key].append(oracle_bound_vals[key])
            delay_mean_oracle[key].append(oracle_delay_mean_vals[key])
            bounds_ns_oracle[key].append(oracle_bound_ns_vals[key])

    # Probability metrics pool exactly like the delay per-run lists: concatenate every
    # experiment's per-run values at each k, and re-derive the pass rates from pooled
    # pass/testable counts rather than averaging rates (an experiment with fewer testable
    # runs must not weigh the same as one with more).
    agg_prob = {}
    for metric in PROB_METRIC_KEYS:
        blocks = [(r.get('prob_metrics') or {}).get(metric) for r in results_list]
        if not any(blocks):
            continue
        metric_out = {'groundtruth_prob_by_run': []}
        family_specs = ([('all_packets_by_run', 'pass_rate_all_packets', [None])]
                         + [('sampled_by_run', 'pass_rate_sampled', methods)]
                         + [('uniform_by_run', 'pass_rate_uniform', uniform_series)]
                         + [('oracle_by_run', 'pass_rate_oracle', oracle_series)])
        for family_key, rate_key, keys in family_specs:
            if keys == [None]:
                metric_out[family_key] = {field: [] for field in PROB_FAMILY_FIELDS}
                metric_out[rate_key] = []
            else:
                metric_out[family_key] = {key: {field: [] for field in PROB_FAMILY_FIELDS}
                                           for key in keys}
                metric_out[rate_key] = {key: [] for key in keys}
        for k in all_k:
            gt_vals = []
            for r, block in zip(results_list, blocks):
                if block is None or k not in r['num_flows']:
                    continue
                i = r['num_flows'].index(k)
                gt_vals.extend((block.get('groundtruth_prob_by_run') or [[]] * len(r['num_flows']))[i])
            metric_out['groundtruth_prob_by_run'].append(gt_vals)
            for family_key, rate_key, keys in family_specs:
                for key in keys:
                    pooled = {field: [] for field in PROB_FAMILY_FIELDS}
                    passes = testable = 0
                    for r, block in zip(results_list, blocks):
                        if block is None or k not in r['num_flows']:
                            continue
                        i = r['num_flows'].index(k)
                        family = block.get(family_key) or {}
                        family = family if key is None else (family.get(key) or {})
                        if not family:
                            continue
                        n_k_r = len(r['num_flows'])
                        for field in PROB_FAMILY_FIELDS:
                            pooled[field].extend((family.get(field) or [[]] * n_k_r)[i])
                        verdicts = (family.get('consistency_pass') or [[]] * n_k_r)[i]
                        passes += sum(1 for v in verdicts if v is True)
                        testable += sum(1 for v in verdicts if v is not None)
                    target = (metric_out[family_key] if key is None
                               else metric_out[family_key][key])
                    for field in PROB_FAMILY_FIELDS:
                        target[field].append(pooled[field])
                    rate = (passes / testable) if testable else 0.0
                    if key is None:
                        metric_out[rate_key].append(rate)
                    else:
                        metric_out[rate_key][key].append(rate)
        agg_prob[metric] = metric_out

    groundtruth_values = np.concatenate(
        [np.asarray(r['groundtruth_values'], dtype=float) for r in results_list])

    def _rate(c, t):
        return c / t if t else 0.0

    # Burstiness of all-packets (see burstiness_metrics) only depends on the traffic's own
    # sending pattern, not on the switch-side Poisson probing, so unlike EMD it doesn't need
    # per-run values -- one value per (k, experiment) is all there is (self-contained pass,
    # independent of the per-run loop above). Kept both as a per-k mean (agg_burstiness,
    # for a quick single-number read) and as the full per-k-per-experiment list
    # (agg_burstiness_by_experiment, so it can be drawn as a boxplot across experiments --
    # the same treatment emd_all_packets_by_experiment gets).
    agg_burst_gap = next((r['burst_gap_threshold_ns'] for r in results_list
                           if np.isfinite(r.get('burst_gap_threshold_ns', np.nan))), np.nan)
    agg_burstiness = {}
    agg_burstiness_by_experiment = {}
    for field in BURSTINESS_METRIC_LABELS:
        per_k, per_k_values = [], []
        for k in all_k:
            vals = []
            for r in results_list:
                if k in r['num_flows']:
                    idx = r['num_flows'].index(k)
                    spread = (r.get('burstiness_all_packets_by_experiment') or {}).get(field)
                    if r.get('all_packets_windowed') and spread is not None:
                        # Windowed run: one value per run, each measured over that run's
                        # own window (see compute_emd_vs_num_tcp_flows_multi_run).
                        vals.extend([v for v in (spread[idx] if idx < len(spread) else []) if v == v])
                        continue
                    v = (r.get('burstiness_all_packets') or {}).get(field, [])
                    if idx < len(v) and v[idx] == v[idx]:
                        vals.append(v[idx])
            per_k_values.append(vals)
            per_k.append(float(np.mean(vals)) if vals else float('nan'))
        agg_burstiness[field] = per_k
        agg_burstiness_by_experiment[field] = per_k_values

    return {
        'flow_name': results_list[0]['flow_name'],
        'path': results_list[0]['path'],
        'subsampling_methods': methods,
        'subsampling_method': methods[0],
        'groundtruth_method': results_list[0].get('groundtruth_method', 'simultaneous'),
        'all_flows_only': all(r.get('all_flows_only', False) for r in results_list),
        'num_runs': sum(r['num_runs'] for r in results_list),
        'num_experiments': len(results_list),
        'experiments': experiments,
        'num_poisson_observations': results_list[0]['num_poisson_observations'],
        'uniform_series': list(uniform_series),
        'oracle_series': list(oracle_series),
        # Each experiment's own stored total_flows (not max(all_k), which is the
        # ALL_FLOWS_ONLY_K sentinel once results_list has been pooled onto it above) -- the
        # two agree for a real flow-count sweep anyway, since every experiment's own num_flows
        # tops out at its own total_flows.
        'total_flows': max(r.get('total_flows', 0) for r in results_list),
        'num_flows': all_k,
        'groundtruth_values': groundtruth_values,
        'groundtruth_mean': float(np.mean(groundtruth_values)) if groundtruth_values.size else np.nan,
        'groundtruth_std': float(np.std(groundtruth_values)) if groundtruth_values.size else np.nan,
        'delay_percentiles': list(percentiles),
        'groundtruth_percentiles': compute_delay_percentiles(groundtruth_values, percentiles),
        'burst_gap_threshold_ns': agg_burst_gap,
        'burstiness_all_packets': agg_burstiness,
        'burstiness_all_packets_by_experiment': agg_burstiness_by_experiment,
        # All-packets percentile error is one value per experiment once aggregated (each
        # experiment has its own ground truth), so it becomes a distribution too -- the same
        # treatment emd_all_packets_by_experiment gets.
        'percentile_diff_all_packets': {q: [float(np.mean(v)) if len(v) else np.nan for v in pdiff_all[q]]
                                         for q in percentiles},
        'percentile_reldiff_all_packets': {q: [float(np.mean(v)) if len(v) else np.nan for v in preldiff_all[q]]
                                            for q in percentiles},
        'percentile_diff_all_packets_by_experiment': pdiff_all,
        'percentile_reldiff_all_packets_by_experiment': preldiff_all,
        'percentile_diff_sampled_by_run': pdiff_sampled,
        'percentile_reldiff_sampled_by_run': preldiff_sampled,
        'percentile_diff_uniform_by_run': pdiff_uniform,
        'percentile_reldiff_uniform_by_run': preldiff_uniform,
        'percentile_diff_oracle_by_run': pdiff_oracle,
        'percentile_reldiff_oracle_by_run': preldiff_oracle,
        # Mean absolute relative percentile error (see percentile_avg_relative_error).
        # Pooled the same way as the normalized EMD/relative percentile errors above: each
        # experiment's own already-computed values are concatenated (all_packets: one
        # value per experiment; sampled/uniform/oracle: concatenated across runs and
        # experiments), never re-derived from pooled raw values, since each experiment has
        # its own ground truth and normalizer.
        'percentile_avg_relerror_all_packets': [
            float(np.mean(v)) if len(v) else np.nan for v in pctrelerr_all_by_experiment],
        'percentile_avg_relerror_all_packets_by_experiment': pctrelerr_all_by_experiment,
        'percentile_avg_relerror_sampled_by_run': pctrelerr_sampled_by_run,
        'percentile_avg_relerror_uniform_by_run': pctrelerr_uniform_by_run,
        'percentile_avg_relerror_oracle_by_run': pctrelerr_oracle_by_run,
        'poisson_test_series': poisson_test_series,
        'run_chi_squared_test': run_chi,
        # All-packets is one verdict per k per experiment. Aggregated, each k therefore
        # holds a *list* of verdicts (one per experiment) aligned with
        # emd_all_packets_by_experiment -- exactly the shape the uniform families' per-run
        # verdicts have, so the same split logic covers both. See _all_packets_test_flags,
        # which also understands the single-experiment (one bare verdict per k) shape.
        'poisson_tests_all_packets': tests_all_by_experiment,
        'uniform_test_split_by_run': uniform_split,
        'emd_all_packets': [float(np.mean(v)) for v in emd_all_by_experiment],
        'emd_all_packets_normalized': [float(np.mean(v)) for v in emd_all_by_experiment_norm],
        'emd_all_packets_by_experiment': emd_all_by_experiment,
        'emd_all_packets_by_experiment_normalized': emd_all_by_experiment_norm,
        # Carried through so a reader (and any further aggregation) knows these values were
        # measured over each run's own growing window, not the full steady period.
        'all_packets_windowed': any(r.get('all_packets_windowed') for r in results_list),
        'analysis_window_method': next((r.get('analysis_window_method') for r in results_list
                                         if r.get('analysis_window_method')), None),
        'all_packet_sizes': [float(np.mean(v)) if len(v) else np.nan
                              for v in all_packet_sizes_by_experiment],
        'all_packet_sizes_by_experiment': all_packet_sizes_by_experiment,
        'emd_sampled_packets_by_run': emd_sampled_by_run,
        'emd_sampled_packets_by_run_normalized': emd_sampled_by_run_norm,
        'pass_rate_all_packets': [_rate(c, t) for c, t in zip(pass_all_count, pass_all_total)],
        'pass_rate_sampled': {m: [_rate(c, t) for c, t in zip(pass_sampled_count[m], pass_sampled_total[m])]
                               for m in methods},
        'mean_diff_all_packets_by_run': mean_diff_all,
        'mean_diff_sampled_by_run': mean_diff_sampled,
        'sample_sizes_sampled_by_run': sample_sizes_sampled,
        'window_duration_sampled_by_run': window_durations_sampled,
        'prob_metrics': agg_prob,
        'delay_mean_all_packets_by_run': delay_mean_all,
        'delay_mean_sampled_by_run': delay_mean_sampled,
        'delay_mean_uniform_by_run': delay_mean_uniform,
        'delay_mean_oracle_by_run': delay_mean_oracle,
        'groundtruth_delay_mean_by_run': groundtruth_delay_mean,
        'error_bound_all_packets_by_run': bound_all,
        'error_bound_ns_all_packets_by_run': bound_ns_all,
        'error_bound_sampled_by_run': bounds_sampled,
        'error_bound_ns_sampled_by_run': bounds_ns_sampled,
        'error_bound_uniform_by_run': bounds_uniform,
        'error_bound_ns_uniform_by_run': bounds_ns_uniform,
        'error_bound_oracle_by_run': bounds_oracle,
        'error_bound_ns_oracle_by_run': bounds_ns_oracle,
        # Every experiment in an aggregation was run against the same configured guarantee
        # (it is a constant of the pipeline, not a per-experiment choice); the first one
        # that recorded it stands for all.
        'delay_consistency_guarantee': next((r.get('delay_consistency_guarantee') for r in results_list
                                              if r.get('delay_consistency_guarantee')), None),
        'steady_start': results_list[0].get('steady_start'),
        'steady_end': results_list[0].get('steady_end'),
        'growing_window_step_ns': results_list[0].get('growing_window_step_ns'),
        'sample_sizes_uniform_by_run': sample_sizes_uniform,
        'emd_uniform_packets_by_run': emd_uniform_by_run,
        'emd_uniform_packets_by_run_normalized': emd_uniform_by_run_norm,
        'pass_rate_uniform': {s: [_rate(c, t) for c, t in zip(pass_uniform_count[s], pass_uniform_total[s])]
                               for s in uniform_series},
        'mean_diff_uniform_packets_by_run': mean_diff_uniform,
        'emd_oracle_by_run': emd_oracle_by_run,
        'emd_oracle_by_run_normalized': emd_oracle_by_run_norm,
        'pass_rate_oracle': {key: [_rate(c, t) for c, t in zip(pass_oracle_count[key], pass_oracle_total[key])]
                              for key in oracle_series},
        'mean_diff_oracle_by_run': mean_diff_oracle,
        'sample_sizes_oracle_by_run': sample_sizes_oracle,
        'one_run_delay_cdfs': results_list[0]['one_run_delay_cdfs'],
    }


def compute_emd_vs_num_tcp_flows_multi_run(
    ns3_path,
    results_folder,
    rate,
    load,
    experiment,
    flow_name,
    queue_names,
    linkDelays,
    linkRates,
    steadyStart,
    steadyEnd,
    confidenceValue,
    DelayConsistencyGaurantee,
    num_runs=100,
    num_poisson_observations=9000,
    min_sample_size=100,
    delay_cdf_sample_interval_ns=10,
    path=0,
    max_num_flows=None,
    num_workers=1,
    flow_count_step=1,
    all_flows_only=False,
    subsampling_methods='find_samples_path',
    groundtruth_method='simultaneous',
    delay_percentiles=DEFAULT_DELAY_PERCENTILES,
    run_chi_squared_test=True,
    poisson_test_lags=None,
    growing_window_step_ns=GROWING_WINDOW_STEP_NS,
    differentiationDelay=None,
    errorRate=None,
):
    """Repeat the flow-count EMD sweep `num_runs` times. Each run draws its
    own Poisson-process realization of `num_poisson_observations` switch
    observation instants (generate_poisson_observation_times) to derive a
    fresh per-segment aggregated delay statistic for the consistency check
    (compute_poisson_agg_stats), then re-derives, for every flow count
    independently: one fresh Poisson-adaptive subsample per entry in
    `subsampling_methods` (any number of POISSON_SUBSAMPLING_METHODS keys --
    pass several, e.g. ['find_samples_path', 'find_samples_path_intensity'],
    to compare the algorithms within one run, on identical packets, flows,
    ground truth and per-run switch statistics) and, for each of those
    methods, a fresh **rate-matched** systematic uniform subsample drawing
    exactly as many packets as that method retained -- a blind, non-adaptive
    baseline at the same sample size, so the comparison isolates the selection
    rule rather than the sample count (see compute_emd_vs_num_tcp_flows_run
    and matched_uniform_target_count) -- plus an **ideal Poisson probe** at the
    minimum required sample size and at each method's own sample size
    (construct_oracle_poisson_delays), which is the best any Poissonization
    scheme could do at that budget, since its instants are Poisson by
    construction and so carry no selection bias at all. The
    ground-truth reconstructed delay CDF, the underlying packet/flow data,
    and the all-packet EMD curve (prepare_emd_vs_flows_data) do not depend on
    the switch-side Poisson probing, so they are computed once and shared
    across all runs -- only their delay-consistency check varies per run.
    Adding a subsampling method therefore costs only that method's own
    subsampling searches, not another ground-truth reconstruction.

    Set `all_flows_only` to evaluate only k = all flows on the path (every
    received e2e packet) instead of sweeping flow counts -- the headline
    configuration, and much the cheapest to run.

    `groundtruth_method` (one of GROUNDTRUTH_METHODS) selects what every EMD
    is measured against: 'simultaneous' (all queues observed at one instant)
    or 'path_observation' (a probe that waits out each queue's delay before
    observing the next -- what a real packet experiences).

    Set `num_workers` > 1 to fan the `num_runs` runs out across worker
    processes (each handling a subset of runs) -- the dominant per-run cost
    is the repeated Poisson-subsampling search, which is CPU-bound and
    embarrassingly parallel across runs.

    Returns a dict with, for each k=1..N considered flows: the single
    all-packet EMD value; the list of Poisson-subsampled-CDF EMD values
    observed across the `num_runs` runs (a run that found no valid subsample
    at a given k simply contributes no value there), keyed by subsampling
    method, and the same for each method's rate-matched uniform family (keyed
    by that same method name);
    the fraction of runs for which the delay consistency check passed at that
    k, for the all-packet mean and for every subsampling method -- for the
    Poisson-adaptive subsamples this is a fraction of the runs that actually
    found a valid subsample at that k, not of all `num_runs` runs, since a run
    that found none neither passed nor failed the check; and the list, across
    runs, of the signed difference between the switch samples' mean delay and
    the packet-side mean delay -- the quantity the consistency check itself
    thresholds -- again for the all-packet mean and for every subsampling
    method.

    Every EMD series also comes with a '..._normalized' counterpart holding
    the same values divided by the mean ground-truth path delay
    (normalize_emd_values), which is what makes EMDs comparable across
    offered loads; the raw ns values are kept alongside them.

    The two families that are *not* Poissonized -- all packets, and each
    rate-matched uniform subset -- additionally get their sampling instants put
    through the Poisson-ness tests the adaptive samplers validate themselves
    against (poisson_process_tests: Anderson-Darling on the gaps, and the
    multi-lag chi-squared independence test). For the uniform families the
    outcome is recorded per run *in lockstep with that run's EMD and
    mean-difference*, so the plots can split runs by whether their instants
    actually looked Poisson; for all-packets it is one verdict per flow count,
    since that family is the same fixed packet set every run. Set
    `run_chi_squared_test=False` to skip the chi-squared half, which dominates
    the cost of this (~1s per call, per family, per flow count, per run).

    Alongside the EMD, every family also reports its **percentile** error at
    each percentile in `delay_percentiles` (default p90/p99): signed
    `ground_truth_percentile - family_percentile`, both absolute (ns,
    '..._percentile_diff_...') and relative to the ground truth's own
    percentile ('..._percentile_reldiff_...'). The EMD is one number for the
    whole distribution, so a family can look good on it and still misplace the
    tail -- which is the part delay SLOs are written against.
    """
    subsampling_methods = normalize_subsampling_methods(subsampling_methods)
    # Fail before any data is loaded if the method combination cannot be analyzed in one
    # run (a growing-window method must be alone -- see growing_window_method_in).
    growing_window_method_in(subsampling_methods)
    prepared = prepare_emd_vs_flows_data(
        ns3_path, results_folder, rate, load, experiment, flow_name, queue_names,
        linkDelays, linkRates, steadyStart, steadyEnd, path=path,
        delay_cdf_sample_interval_ns=delay_cdf_sample_interval_ns, max_num_flows=max_num_flows,
        flow_count_step=flow_count_step, all_flows_only=all_flows_only,
        groundtruth_method=groundtruth_method, delay_percentiles=delay_percentiles,
        run_chi_squared_test=run_chi_squared_test, poisson_test_lags=poisson_test_lags,
        differentiationDelay=differentiationDelay, errorRate=errorRate,
    )
    dir_prefix = prepared['dir_prefix']
    num_flows = prepared['num_flows']
    groundtruth_mean = prepared['groundtruth_mean']
    delay_percentiles = tuple(prepared['delay_percentiles'])
    groundtruth_percentiles = prepared['groundtruth_percentiles']

    # One extra, cheap concrete Poisson realization (compute_poisson_agg_stats itself is
    # not the expensive part -- reconstructing the ground truth is, and that's already
    # done above) purely to have real per-packet delay values for plot_one_run_delay_cdfs,
    # since the num_runs loop below only keeps EMD/pass/mean-diff summaries, not raw values.
    one_run_agg_stats = compute_poisson_agg_stats(
        dir_prefix, queue_names, linkDelays, linkRates, steadyStart, steadyEnd,
        num_poisson_observations, confidenceValue, DelayConsistencyGaurantee,
    )
    # Precomputed once here, in the parent, so every worker inherits it through fork (see
    # precompute_subsample_deltas): the bin width each candidate window's packet prefix
    # implies. Deterministic, so no result changes -- only how often
    # find_delta_for_empty_prob runs (18 times per flow count instead of 18 per run).
    delta_t0 = time.time()
    prepared['delta_cache'] = precompute_subsample_deltas(
        prepared, subsampling_methods, step_ns=growing_window_step_ns, num_workers=num_workers)
    if prepared['delta_cache']:
        print("Precomputed {} subsampling bin width(s) in {:.1f}s, shared by all {} runs".format(
            sum(len(v) for v in prepared['delta_cache'].values()), time.time() - delta_t0, num_runs))

    one_run_window_ctx = growing_window_context(
        prepared, confidenceValue, DelayConsistencyGaurantee, num_poisson_observations,
        step_ns=growing_window_step_ns)
    one_run_delay_cdfs = _collect_one_run_delay_cdfs(prepared, one_run_agg_stats, min_sample_size,
                                                      subsampling_methods,
                                                      window_ctx=one_run_window_ctx)

    run_results = _run_poisson_runs(
        prepared, dir_prefix, queue_names, linkDelays, linkRates, steadyStart, steadyEnd,
        num_poisson_observations, confidenceValue, DelayConsistencyGaurantee, min_sample_size,
        num_runs, num_workers, subsampling_methods, step_ns=growing_window_step_ns,
    )

    windowed_method = growing_window_method_in(subsampling_methods)
    per_k_pass_all = [0] * len(num_flows)
    per_k_mean_diff_all = [[] for _ in num_flows]
    # In a windowed run the all-packets family is a different packet set every run (each
    # run's window differs), so like every other family it becomes a per-run distribution
    # rather than one fixed value per flow count.
    per_k_emd_all = [[] for _ in num_flows]
    per_k_emd_all_norm = [[] for _ in num_flows]
    per_k_size_all = [[] for _ in num_flows]
    per_k_pdiff_all = {q: [[] for _ in num_flows] for q in delay_percentiles}
    per_k_preldiff_all = {q: [[] for _ in num_flows] for q in delay_percentiles}
    per_k_pctrelerr_all = [[] for _ in num_flows]
    # The consistency check's own threshold at each family's sample size, per run (see
    # delay_consistency_error_bound): relative to the switch mean, and in ns.
    per_k_bound_all = [[] for _ in num_flows]
    per_k_bound_ns_all = [[] for _ in num_flows]
    # Each family's own mean queuing delay, and the ground truth's, per run.
    per_k_delay_mean_all = [[] for _ in num_flows]
    per_k_groundtruth_delay_mean = [[] for _ in num_flows]
    per_k_tests_all = {field: [[] for _ in num_flows]
                        for field in ('ad_pass', 'ad_pvalue', 'chi_pass', 'chi_reject_fraction')}
    per_k_burstiness_all = {field: [[] for _ in num_flows] for field in BURSTINESS_METRIC_LABELS}
    per_k_emd_sampled = {m: [[] for _ in num_flows] for m in subsampling_methods}
    # Normalized EMDs are pooled from each run rather than derived here by dividing the
    # pooled raw values: a growing-window family's normalizer is the mean ground-truth
    # delay of the window *that run* stopped at, which only the run itself knows. For every
    # other family the run divides by the full-window mean, exactly as before.
    per_k_emd_sampled_norm = {m: [[] for _ in num_flows] for m in subsampling_methods}
    per_k_pass_sampled = {m: [0] * len(num_flows) for m in subsampling_methods}
    per_k_mean_diff_sampled = {m: [[] for _ in num_flows] for m in subsampling_methods}
    per_k_sample_sizes_sampled = {m: [[] for _ in num_flows] for m in subsampling_methods}
    # How long a monitoring window each method's samples actually came from, per run (ns).
    per_k_window_duration_sampled = {m: [[] for _ in num_flows] for m in subsampling_methods}
    per_k_bound_sampled = {m: [[] for _ in num_flows] for m in subsampling_methods}
    per_k_bound_ns_sampled = {m: [[] for _ in num_flows] for m in subsampling_methods}
    per_k_bound_uniform = {m: [[] for _ in num_flows] for m in subsampling_methods}
    per_k_bound_ns_uniform = {m: [[] for _ in num_flows] for m in subsampling_methods}
    per_k_delay_mean_sampled = {m: [[] for _ in num_flows] for m in subsampling_methods}
    per_k_delay_mean_uniform = {m: [[] for _ in num_flows] for m in subsampling_methods}
    # One rate-matched uniform family per Poisson-adaptive method, keyed by that method.
    per_k_emd_uniform = {m: [[] for _ in num_flows] for m in subsampling_methods}
    per_k_emd_uniform_norm = {m: [[] for _ in num_flows] for m in subsampling_methods}
    per_k_pass_uniform = {m: [0] * len(num_flows) for m in subsampling_methods}
    per_k_mean_diff_uniform = {m: [[] for _ in num_flows] for m in subsampling_methods}
    per_k_sample_sizes_uniform = {m: [[] for _ in num_flows] for m in subsampling_methods}
    per_k_pdiff_sampled = _empty_percentile_structure(delay_percentiles, subsampling_methods, len(num_flows))
    per_k_preldiff_sampled = _empty_percentile_structure(delay_percentiles, subsampling_methods, len(num_flows))
    per_k_pdiff_uniform = _empty_percentile_structure(delay_percentiles, subsampling_methods, len(num_flows))
    per_k_preldiff_uniform = _empty_percentile_structure(delay_percentiles, subsampling_methods, len(num_flows))
    # Mean absolute relative percentile error (see percentile_avg_relative_error), pooled
    # across runs the same way as the raw EMD.
    per_k_pctrelerr_sampled = {m: [[] for _ in num_flows] for m in subsampling_methods}
    per_k_pctrelerr_uniform = {m: [[] for _ in num_flows] for m in subsampling_methods}
    oracle_series = [ORACLE_MIN_REQUIRED_KEY] + list(subsampling_methods) + [ORACLE_ALL_PACKETS_RATE_KEY]
    per_k_emd_oracle = {key: [[] for _ in num_flows] for key in oracle_series}
    per_k_emd_oracle_norm = {key: [[] for _ in num_flows] for key in oracle_series}
    per_k_pass_oracle = {key: [0] * len(num_flows) for key in oracle_series}
    per_k_mean_diff_oracle = {key: [[] for _ in num_flows] for key in oracle_series}
    per_k_sample_sizes_oracle = {key: [[] for _ in num_flows] for key in oracle_series}
    per_k_bound_oracle = {key: [[] for _ in num_flows] for key in oracle_series}
    per_k_bound_ns_oracle = {key: [[] for _ in num_flows] for key in oracle_series}
    per_k_delay_mean_oracle = {key: [[] for _ in num_flows] for key in oracle_series}
    per_k_pdiff_oracle = _empty_percentile_structure(delay_percentiles, oracle_series, len(num_flows))
    per_k_preldiff_oracle = _empty_percentile_structure(delay_percentiles, oracle_series, len(num_flows))
    per_k_pctrelerr_oracle = {key: [[] for _ in num_flows] for key in oracle_series}
    # The loss/marking metrics, pooled across runs exactly like every delay quantity: one
    # list per flow count per family, plus the per-run reference probability (which in a
    # windowed run is that run's own window's).
    per_k_prob = {
        metric: {
            'groundtruth_prob': [[] for _ in num_flows],
            'all_packets': {field: [[] for _ in num_flows] for field in PROB_FAMILY_FIELDS},
            'sampled': {m: {field: [[] for _ in num_flows] for field in PROB_FAMILY_FIELDS}
                         for m in subsampling_methods},
            'uniform': {m: {field: [[] for _ in num_flows] for field in PROB_FAMILY_FIELDS}
                         for m in subsampling_methods},
            'oracle': {key: {field: [[] for _ in num_flows] for field in PROB_FAMILY_FIELDS}
                        for key in oracle_series},
            'pass_count': {'all_packets': [0] * len(num_flows),
                            'sampled': {m: [0] * len(num_flows) for m in subsampling_methods},
                            'uniform': {m: [0] * len(num_flows) for m in subsampling_methods},
                            'oracle': {key: [0] * len(num_flows) for key in oracle_series}},
        } for metric in PROB_METRIC_KEYS}

    # Aligned per-run records for the Poisson-ness split: value and verdict appended
    # together, so index j of every list below belongs to the same run.
    split_fields = ('emd', 'emd_normalized', 'mean_diff', 'ad_pass', 'chi_pass')
    per_k_uniform_split = {name: {field: [[] for _ in num_flows] for field in split_fields}
                            for name in subsampling_methods}

    for run_result in run_results:
        for i in range(len(num_flows)):
            if run_result['consistency_pass_all_packets'][i] is True:
                per_k_pass_all[i] += 1
            if np.isfinite(run_result['mean_diff_all_packets'][i]):
                per_k_mean_diff_all[i].append(run_result['mean_diff_all_packets'][i])
            bound_all_value = run_result['error_bound_all_packets'][i]
            if np.isfinite(bound_all_value):
                per_k_bound_all[i].append(bound_all_value)
                per_k_bound_ns_all[i].append(run_result['error_bound_ns_all_packets'][i])
            delay_mean_all_value = run_result['delay_mean_all_packets'][i]
            if np.isfinite(delay_mean_all_value):
                per_k_delay_mean_all[i].append(delay_mean_all_value)
            gt_delay_mean_value = run_result['groundtruth_delay_mean'][i]
            if np.isfinite(gt_delay_mean_value):
                per_k_groundtruth_delay_mean[i].append(gt_delay_mean_value)
            if windowed_method is not None:
                emd_all_value = run_result['emd_all_packets'][i]
                if np.isfinite(emd_all_value):
                    per_k_emd_all[i].append(emd_all_value)
                    per_k_emd_all_norm[i].append(run_result['emd_all_packets_normalized'][i])
                if run_result['all_packet_size'][i]:
                    per_k_size_all[i].append(run_result['all_packet_size'][i])
                for q in delay_percentiles:
                    all_pdiff = run_result['percentile_diff_all_packets'][q][i]
                    if np.isfinite(all_pdiff):
                        per_k_pdiff_all[q][i].append(all_pdiff)
                        per_k_preldiff_all[q][i].append(
                            run_result['percentile_reldiff_all_packets'][q][i])
                all_relerr = run_result['percentile_avg_relerror_all_packets'][i]
                if np.isfinite(all_relerr):
                    per_k_pctrelerr_all[i].append(all_relerr)
                for field in per_k_tests_all:
                    per_k_tests_all[field][i].append(run_result['poisson_tests_all_packets'][field][i])
                for field in per_k_burstiness_all:
                    burst_value = run_result['burstiness_all_packets'][field][i]
                    if burst_value == burst_value:
                        per_k_burstiness_all[field][i].append(burst_value)

            for m in subsampling_methods:
                if np.isfinite(run_result['sampled_emd'][m][i]):
                    per_k_emd_sampled[m][i].append(run_result['sampled_emd'][m][i])
                    per_k_emd_sampled_norm[m][i].append(run_result['sampled_emd_normalized'][m][i])
                if run_result['sampled_consistency'][m][i] is True:
                    per_k_pass_sampled[m][i] += 1
                if np.isfinite(run_result['sampled_mean_diff'][m][i]):
                    per_k_mean_diff_sampled[m][i].append(run_result['sampled_mean_diff'][m][i])
                if run_result['sampled_sample_sizes'][m][i]:
                    per_k_sample_sizes_sampled[m][i].append(run_result['sampled_sample_sizes'][m][i])
                window_duration = run_result['sampled_window_duration'][m][i]
                if window_duration is not None and np.isfinite(window_duration):
                    per_k_window_duration_sampled[m][i].append(window_duration)
                sampled_bound = run_result['sampled_error_bound'][m][i]
                if np.isfinite(sampled_bound):
                    per_k_bound_sampled[m][i].append(sampled_bound)
                    per_k_bound_ns_sampled[m][i].append(run_result['sampled_error_bound_ns'][m][i])
                sampled_delay_mean_value = run_result['sampled_delay_mean'][m][i]
                if np.isfinite(sampled_delay_mean_value):
                    per_k_delay_mean_sampled[m][i].append(sampled_delay_mean_value)
                uniform_delay_mean_value = run_result['uniform_delay_mean'][m][i]
                if np.isfinite(uniform_delay_mean_value):
                    per_k_delay_mean_uniform[m][i].append(uniform_delay_mean_value)
                uniform_bound = run_result['uniform_error_bound'][m][i]
                if np.isfinite(uniform_bound):
                    per_k_bound_uniform[m][i].append(uniform_bound)
                    per_k_bound_ns_uniform[m][i].append(run_result['uniform_error_bound_ns'][m][i])

                if np.isfinite(run_result['uniform_emd'][m][i]):
                    per_k_emd_uniform[m][i].append(run_result['uniform_emd'][m][i])
                    per_k_emd_uniform_norm[m][i].append(run_result['uniform_emd_normalized'][m][i])
                if run_result['uniform_consistency'][m][i] is True:
                    per_k_pass_uniform[m][i] += 1
                if np.isfinite(run_result['uniform_mean_diff'][m][i]):
                    per_k_mean_diff_uniform[m][i].append(run_result['uniform_mean_diff'][m][i])
                if run_result['uniform_sample_sizes'][m][i]:
                    per_k_sample_sizes_uniform[m][i].append(run_result['uniform_sample_sizes'][m][i])

                split = run_result['uniform_test_split'][m]
                emd_value = split['emd'][i]
                if np.isfinite(emd_value):
                    per_k_uniform_split[m]['emd'][i].append(emd_value)
                    per_k_uniform_split[m]['emd_normalized'][i].append(split['emd_normalized'][i])
                    per_k_uniform_split[m]['mean_diff'][i].append(split['mean_diff'][i])
                    per_k_uniform_split[m]['ad_pass'][i].append(split['ad_pass'][i])
                    per_k_uniform_split[m]['chi_pass'][i].append(split['chi_pass'][i])

                for q in delay_percentiles:
                    sampled_pdiff = run_result['sampled_percentile_diff'][q][m][i]
                    if np.isfinite(sampled_pdiff):
                        per_k_pdiff_sampled[q][m][i].append(sampled_pdiff)
                        per_k_preldiff_sampled[q][m][i].append(
                            run_result['sampled_percentile_reldiff'][q][m][i])
                    uniform_pdiff = run_result['uniform_percentile_diff'][q][m][i]
                    if np.isfinite(uniform_pdiff):
                        per_k_pdiff_uniform[q][m][i].append(uniform_pdiff)
                        per_k_preldiff_uniform[q][m][i].append(
                            run_result['uniform_percentile_reldiff'][q][m][i])

                sampled_relerr = run_result['sampled_percentile_avg_relerror'][m][i]
                if np.isfinite(sampled_relerr):
                    per_k_pctrelerr_sampled[m][i].append(sampled_relerr)
                uniform_relerr = run_result['uniform_percentile_avg_relerror'][m][i]
                if np.isfinite(uniform_relerr):
                    per_k_pctrelerr_uniform[m][i].append(uniform_relerr)

            for metric in PROB_METRIC_KEYS:
                run_prob = run_result['prob_metrics'][metric]
                store = per_k_prob[metric]
                store['groundtruth_prob'][i].append(run_prob['groundtruth_prob'][i])
                for family, keys in (('all_packets', [None]),
                                      ('sampled', subsampling_methods),
                                      ('uniform', subsampling_methods),
                                      ('oracle', oracle_series)):
                    for key in keys:
                        run_family = run_prob[family] if key is None else run_prob[family][key]
                        target = store[family] if key is None else store[family][key]
                        # A family with no usable estimate at this k contributes nothing,
                        # so every list stays in lockstep with the runs that did produce
                        # one -- the denominator convention the delay families use.
                        if not np.isfinite(run_family['prob'][i]):
                            continue
                        for field in PROB_FAMILY_FIELDS:
                            value = run_family[field][i]
                            if field == 'consistency_pass':
                                if value is True:
                                    if key is None:
                                        store['pass_count'][family][i] += 1
                                    else:
                                        store['pass_count'][family][key][i] += 1
                                continue
                            target[field][i].append(value)
                        target['consistency_pass'][i].append(run_family['consistency_pass'][i])

            for key in oracle_series:
                if np.isfinite(run_result['oracle_emd'][key][i]):
                    per_k_emd_oracle[key][i].append(run_result['oracle_emd'][key][i])
                    per_k_emd_oracle_norm[key][i].append(run_result['oracle_emd_normalized'][key][i])
                if run_result['oracle_consistency'][key][i] is True:
                    per_k_pass_oracle[key][i] += 1
                if np.isfinite(run_result['oracle_mean_diff'][key][i]):
                    per_k_mean_diff_oracle[key][i].append(run_result['oracle_mean_diff'][key][i])
                if run_result['oracle_sample_sizes'][key][i]:
                    per_k_sample_sizes_oracle[key][i].append(run_result['oracle_sample_sizes'][key][i])
                oracle_bound = run_result['oracle_error_bound'][key][i]
                if np.isfinite(oracle_bound):
                    per_k_bound_oracle[key][i].append(oracle_bound)
                    per_k_bound_ns_oracle[key][i].append(run_result['oracle_error_bound_ns'][key][i])
                oracle_delay_mean_value = run_result['oracle_delay_mean'][key][i]
                if np.isfinite(oracle_delay_mean_value):
                    per_k_delay_mean_oracle[key][i].append(oracle_delay_mean_value)
                for q in delay_percentiles:
                    oracle_pdiff = run_result['oracle_percentile_diff'][q][key][i]
                    if np.isfinite(oracle_pdiff):
                        per_k_pdiff_oracle[q][key][i].append(oracle_pdiff)
                        per_k_preldiff_oracle[q][key][i].append(
                            run_result['oracle_percentile_reldiff'][q][key][i])
                oracle_relerr = run_result['oracle_percentile_avg_relerror'][key][i]
                if np.isfinite(oracle_relerr):
                    per_k_pctrelerr_oracle[key][i].append(oracle_relerr)

    def _mean_or_nan(values):
        return float(np.mean(values)) if len(values) else np.nan

    if windowed_method is None:
        # Ordinary run: the all-packets family is the same fixed packet set every run, so
        # prepare_emd_vs_flows_data's per-flow-count values stand as they always have.
        all_packets_results = {
            'emd_all_packets': prepared['emd_all_packets'],
            'emd_all_packets_normalized': normalize_emd_values(prepared['emd_all_packets'], groundtruth_mean),
            'all_packet_sizes': prepared['all_packet_sizes'],
            'percentile_diff_all_packets': prepared['percentile_diff_all_packets'],
            'percentile_reldiff_all_packets': prepared['percentile_reldiff_all_packets'],
            'percentile_avg_relerror_all_packets': prepared['percentile_avg_relerror_all_packets'],
            'poisson_tests_all_packets': prepared['poisson_tests_all_packets'],
            'burstiness_all_packets': prepared['burstiness_all_packets'],
            'pass_rate_all_packets': [c / num_runs for c in per_k_pass_all],
        }
    else:
        # Windowed run: every quantity here was measured over the window that run's search
        # settled on -- never over the full steady period (unless the search reached it).
        # The scalar keys keep their shape (now the mean across runs) so every existing
        # reader still works, and the '..._by_experiment' keys carry the per-run values
        # behind them, which is the same "set of values pooled behind this point" role they
        # play in an aggregated result -- so the plots box them and the cross-experiment
        # aggregation concatenates them, both with no special casing.
        all_packets_results = {
            'emd_all_packets': [_mean_or_nan(v) for v in per_k_emd_all],
            'emd_all_packets_by_experiment': per_k_emd_all,
            'emd_all_packets_normalized': [_mean_or_nan(v) for v in per_k_emd_all_norm],
            'emd_all_packets_by_experiment_normalized': per_k_emd_all_norm,
            'all_packet_sizes': [_mean_or_nan(v) for v in per_k_size_all],
            'all_packet_sizes_by_experiment': per_k_size_all,
            'percentile_diff_all_packets': {q: [_mean_or_nan(v) for v in per_k_pdiff_all[q]]
                                             for q in delay_percentiles},
            'percentile_diff_all_packets_by_experiment': per_k_pdiff_all,
            'percentile_reldiff_all_packets': {q: [_mean_or_nan(v) for v in per_k_preldiff_all[q]]
                                                for q in delay_percentiles},
            'percentile_reldiff_all_packets_by_experiment': per_k_preldiff_all,
            'percentile_avg_relerror_all_packets': [_mean_or_nan(v) for v in per_k_pctrelerr_all],
            'percentile_avg_relerror_all_packets_by_experiment': per_k_pctrelerr_all,
            # Per-run verdict lists -- the same shape an aggregated result's all-packets
            # tests take, which plot_poisson_test_split_vs_num_flows already understands.
            'poisson_tests_all_packets': per_k_tests_all,
            'burstiness_all_packets': {field: [_mean_or_nan(v) for v in per_k_burstiness_all[field]]
                                        for field in per_k_burstiness_all},
            'burstiness_all_packets_by_experiment': per_k_burstiness_all,
            # Denominator is the runs that actually certified a window (and so produced an
            # all-packets value at all), matching the convention pass_rate_sampled uses: a
            # run that found no usable window neither passed nor failed the check.
            'pass_rate_all_packets': [(c / len(v)) if len(v) else 0.0
                                       for c, v in zip(per_k_pass_all, per_k_emd_all)],
        }

    # With no certified window a run reports nothing at all, so the uniform and probe
    # families' denominators follow the same "runs that produced a value" rule as the
    # Poisson-adaptive families. In an ordinary run every run always produces one, which is
    # why num_runs has always been the denominator there.
    def _family_pass_rates(counts_by_key, values_by_key):
        if windowed_method is None:
            return {key: [c / num_runs for c in counts] for key, counts in counts_by_key.items()}
        return {key: [(c / len(v)) if len(v) else 0.0
                       for c, v in zip(counts_by_key[key], values_by_key[key])]
                 for key in counts_by_key}

    # A windowed run has no single ground truth: each run reconstructed one over its own
    # window. What is reported here is the window ground truth of the one concrete
    # realization plot_one_run_delay_cdfs draws, so the printed summary stats and the
    # plotted reference curve are the same object -- never the full steady window's, which
    # no family in a windowed run was ever compared against.
    reported_gt_values = prepared['groundtruth_values']
    if windowed_method is not None:
        one_run_gt = (one_run_delay_cdfs.get('groundtruth_by_method') or {}).get(windowed_method)
        if one_run_gt is not None and len(one_run_gt):
            reported_gt_values = one_run_gt
    reported_gt_values = np.asarray(reported_gt_values)
    reported_gt_mean = float(np.mean(reported_gt_values)) if len(reported_gt_values) else np.nan
    reported_gt_std = float(np.std(reported_gt_values)) if len(reported_gt_values) else np.nan

    return {
        'flow_name': flow_name,
        'path': path,
        'subsampling_methods': subsampling_methods,
        # Which method's growing window defined the analysis window every quantity in this
        # result was measured over, or None for an ordinary whole-steady-window run.
        'analysis_window_method': windowed_method,
        'all_packets_windowed': windowed_method is not None,
        # Kept for anything that only ever knew about one method (older readers,
        # text-summary headers); the full list lives in 'subsampling_methods'.
        'subsampling_method': subsampling_methods[0],
        'groundtruth_method': groundtruth_method,
        'all_flows_only': all_flows_only,
        'num_runs': num_runs,
        'num_poisson_observations': num_poisson_observations,
        'uniform_series': list(subsampling_methods),
        'oracle_series': oracle_series,
        'total_flows': len(prepared['flow_order']),
        'num_flows': num_flows,
        'groundtruth_values': reported_gt_values,
        'groundtruth_mean': reported_gt_mean,
        'groundtruth_std': reported_gt_std,
        'delay_percentiles': list(delay_percentiles),
        'groundtruth_percentiles': (compute_delay_percentiles(reported_gt_values, delay_percentiles)
                                     if windowed_method is not None else groundtruth_percentiles),
        'burst_gap_threshold_ns': prepared['burst_gap_threshold_ns'],
        'percentile_diff_sampled_by_run': per_k_pdiff_sampled,
        # Relative percentile errors come from the runs themselves, each divided by the
        # ground-truth percentile of the window that family was scored in (identical to
        # dividing by the full-window percentile for every non-growing-window family).
        'percentile_reldiff_sampled_by_run': per_k_preldiff_sampled,
        'percentile_diff_uniform_by_run': per_k_pdiff_uniform,
        'percentile_reldiff_uniform_by_run': per_k_preldiff_uniform,
        'percentile_diff_oracle_by_run': per_k_pdiff_oracle,
        'percentile_reldiff_oracle_by_run': per_k_preldiff_oracle,
        # Mean absolute relative percentile error (see percentile_avg_relative_error):
        # self-normalized already, evaluated via a dense percentile grid independent of
        # delay_percentiles.
        'percentile_avg_relerror_sampled_by_run': per_k_pctrelerr_sampled,
        'percentile_avg_relerror_uniform_by_run': per_k_pctrelerr_uniform,
        'percentile_avg_relerror_oracle_by_run': per_k_pctrelerr_oracle,
        'poisson_test_series': ['all_packets'] + [('uniform', m) for m in subsampling_methods],
        'run_chi_squared_test': run_chi_squared_test,
        'uniform_test_split_by_run': per_k_uniform_split,
        'emd_sampled_packets_by_run': per_k_emd_sampled,
        'emd_sampled_packets_by_run_normalized': per_k_emd_sampled_norm,
        # Denominator is the number of runs that actually found a valid Poisson-adaptive
        # subsample at this k (len(per_k_emd_sampled[m][i])), not num_runs -- a run that
        # found no subsample at all didn't pass or fail the check, so it shouldn't count
        # against the pass rate. 0.0 when no run ever found a subsample (nothing to divide by).
        'pass_rate_sampled': {
            m: [(c / len(v)) if len(v) else 0.0 for c, v in zip(per_k_pass_sampled[m], per_k_emd_sampled[m])]
            for m in subsampling_methods},
        'mean_diff_all_packets_by_run': per_k_mean_diff_all,
        'mean_diff_sampled_by_run': per_k_mean_diff_sampled,
        'sample_sizes_sampled_by_run': per_k_sample_sizes_sampled,
        'sample_sizes_uniform_by_run': per_k_sample_sizes_uniform,
        # Per-run length (ns) of the monitoring window each method's samples came from: the
        # growing-window methods' headline result ("how long did we have to watch?"), and
        # the full steady window for every method that samples all of it.
        'window_duration_sampled_by_run': per_k_window_duration_sampled,
        # The consistency check's own threshold at each family's realized sample size, per
        # run (delay_consistency_error_bound): relative to the switch-side mean -- which a
        # family holding exactly the minimum required samples puts at exactly
        # DelayConsistencyGaurantee -- and the same bound in ns, the figure |mean_diff| is
        # actually tested against.
        # Each family's own mean queuing delay per run, and the ground truth's, in ns: the
        # delay counterpart of the probability metrics' estimate-and-reference pair (the EMD
        # measures the whole distribution, this the headline number the check thresholds).
        'delay_mean_all_packets_by_run': per_k_delay_mean_all,
        'delay_mean_sampled_by_run': per_k_delay_mean_sampled,
        'delay_mean_uniform_by_run': per_k_delay_mean_uniform,
        'delay_mean_oracle_by_run': per_k_delay_mean_oracle,
        'groundtruth_delay_mean_by_run': per_k_groundtruth_delay_mean,
        'error_bound_all_packets_by_run': per_k_bound_all,
        'error_bound_ns_all_packets_by_run': per_k_bound_ns_all,
        'error_bound_sampled_by_run': per_k_bound_sampled,
        'error_bound_ns_sampled_by_run': per_k_bound_ns_sampled,
        'error_bound_uniform_by_run': per_k_bound_uniform,
        'error_bound_ns_uniform_by_run': per_k_bound_ns_uniform,
        'error_bound_oracle_by_run': per_k_bound_oracle,
        'error_bound_ns_oracle_by_run': per_k_bound_ns_oracle,
        # The relative error the run was configured to guarantee -- what the relative
        # bounds above are to be read against.
        'delay_consistency_guarantee': DelayConsistencyGaurantee,
        'steady_start': steadyStart,
        'steady_end': steadyEnd,
        'growing_window_step_ns': growing_window_step_ns,
        'emd_uniform_packets_by_run': per_k_emd_uniform,
        'emd_uniform_packets_by_run_normalized': per_k_emd_uniform_norm,
        'pass_rate_uniform': _family_pass_rates(per_k_pass_uniform, per_k_emd_uniform),
        'mean_diff_uniform_packets_by_run': per_k_mean_diff_uniform,
        'emd_oracle_by_run': per_k_emd_oracle,
        'emd_oracle_by_run_normalized': per_k_emd_oracle_norm,
        'pass_rate_oracle': _family_pass_rates(per_k_pass_oracle, per_k_emd_oracle),
        'mean_diff_oracle_by_run': per_k_mean_diff_oracle,
        'sample_sizes_oracle_by_run': per_k_sample_sizes_oracle,
        'one_run_delay_cdfs': one_run_delay_cdfs,
        # Loss and ECN-marking, over the same packets, families and window as the delay
        # results above. Per metric: the reference probability, and per family the per-run
        # estimate, its distance to that reference (|dp|, which for a 0/1 outcome is the
        # Wasserstein distance the delay side calls EMD), the log-space difference and
        # acceptance band the check applied, the sample size, and the pass rate out of the
        # runs that could be tested at all. See PROB_METRICS for why there are no
        # percentile or CDF counterparts, and why the success probability is degenerate on
        # traces that record no drops.
        'prob_metrics': {
            metric: {
                'groundtruth_prob_by_run': per_k_prob[metric]['groundtruth_prob'],
                'all_packets_by_run': per_k_prob[metric]['all_packets'],
                'sampled_by_run': per_k_prob[metric]['sampled'],
                'uniform_by_run': per_k_prob[metric]['uniform'],
                'oracle_by_run': per_k_prob[metric]['oracle'],
                'pass_rate_all_packets': _prob_pass_rates(
                    per_k_prob[metric]['pass_count']['all_packets'],
                    per_k_prob[metric]['all_packets']['consistency_pass']),
                'pass_rate_sampled': {m: _prob_pass_rates(
                    per_k_prob[metric]['pass_count']['sampled'][m],
                    per_k_prob[metric]['sampled'][m]['consistency_pass'])
                    for m in subsampling_methods},
                'pass_rate_uniform': {m: _prob_pass_rates(
                    per_k_prob[metric]['pass_count']['uniform'][m],
                    per_k_prob[metric]['uniform'][m]['consistency_pass'])
                    for m in subsampling_methods},
                'pass_rate_oracle': {key: _prob_pass_rates(
                    per_k_prob[metric]['pass_count']['oracle'][key],
                    per_k_prob[metric]['oracle'][key]['consistency_pass'])
                    for key in oracle_series},
            } for metric in PROB_METRIC_KEYS},
        # The all-packets family: full-window and fixed per flow count in an ordinary run,
        # per-run and inside each run's own window in a windowed one (see above).
        **all_packets_results,
    }


# One entry per subsampling family drawn on a flow-count plot -- every
# Poisson-adaptive method first, then every uniform stride -- so a run
# comparing several algorithms at once still tells them apart by border alone.
# Border *style* encodes what KIND of family a box is, so the plot answers "is this the
# no-subsampling ceiling, a Poisson-instant estimator, its ideal-probe ceiling, or a blind
# fixed-rate baseline?" before you read any legend: all packets is solid, Poisson-adaptive
# subsamples are dashed, ideal Poisson probes are dash-dot, and uniform fixed-rate baselines
# are dotted -- four kinds, four styles, none shared, so no two kinds are ever separable by
# border width alone. Individual families within a kind are then told apart by border
# *colour* on flow-count plots. Load plots don't use this table at all -- colour there is
# spent on the traffic, so a series can only be told apart by *style*, and every load plot
# assigns its own series one of the four canonical styles fresh, in the order they're added
# (see _load_series_spec / _STYLE_ORDER), rather than by `kind`.
_FAMILY_EDGE_STYLE_BY_KIND = {
    'all_packets': 'solid',
    'sampled': 'dashed',
    'oracle': 'dashdot',
    'uniform': 'dotted',
}

# One distinct colour per comparison family, assigned in draw order across all kinds so no
# two families on a plot ever share one.
# Long enough for a run comparing three Poisson-adaptive methods: every method brings its
# own sampled box, its rate-matched uniform box and its ideal-probe box, so three methods
# plus all-packets and the two method-independent probes already need 12 colours.
_FAMILY_COLORS = ['navy', 'darkorange', 'purple', 'teal', 'crimson',
                   'olive', 'saddlebrown', 'magenta', 'dimgray', 'darkgreen',
                   'deepskyblue', 'gold', 'indigo', 'mediumseagreen', 'tomato',
                   'slateblue', 'darkkhaki', 'hotpink']

# Geometry of one x-tick's cluster of boxes. `_FAMILY_GROUP_SPAN` is how much of the gap to
# the neighbouring tick the whole cluster may occupy; `_FAMILY_BOX_FILL` is how much of each
# family's slot within that cluster the box itself fills -- the rest is the gap that keeps
# adjacent boxes visually separate, which matters more the more families there are.
_FAMILY_GROUP_SPAN = 0.80
_FAMILY_BOX_FILL = 0.62


def family_border_style(kind, color_index):
    """The border (colour, dash) a comparison family is drawn with: dash from its `kind`
    ('all_packets' / 'sampled' / 'oracle' / 'uniform', see _FAMILY_EDGE_STYLE_BY_KIND) and
    colour from its position in the plot's family order. Warns rather than silently
    reusing a colour if a plot ever carries more families than the palette holds, since two
    families sharing both colour and dash would be indistinguishable."""
    if color_index >= len(_FAMILY_COLORS):
        print("Warning: {} comparison families exceed the {} distinct border colours "
              "available; colours now repeat and some families are indistinguishable".format(
                  color_index + 1, len(_FAMILY_COLORS)))
    return dict(edge_color=_FAMILY_COLORS[color_index % len(_FAMILY_COLORS)],
                 edge_style=_FAMILY_EDGE_STYLE_BY_KIND[kind])


def _draw_boxplot_family(axis, num_flows, values_by_k, pass_rate_by_k, position_offset, box_width,
                          pass_threshold, pass_color, fail_color, style, edge_width=4.5,
                          fill_color=None):
    """Draw one boxplot family (one box per k with data) at x = k + position_offset.
    The fill is *only* the pass/fail color (green/red) -- no hatch -- so it stays a clean,
    unambiguous read of the consistency check; families are told apart purely by the box
    border (edge_color/edge_style from `style`, see family_border_style) drawn thick
    enough to read at a glance.

    Pass `fill_color` to fill every box with that one colour instead, for quantities the
    consistency check says nothing about -- it tests the *mean*, so colouring e.g. a
    percentile-error box by it would imply a verdict the check never made.

    Returns {k: whisker-top y-value} for every k that got a box drawn here -- the actual
    rendered top of that box's whisker, which can sit well below the raw data's true max
    since showfliers=False hides anything beyond it as an outlier. A caller anchoring
    something above the box (e.g. _annotate_all_packets_burstiness) should use this rather
    than the raw max, which can land outside the axes' own autoscaled view and simply never
    render."""
    positions, data, colors = [], [], []
    plotted_k = []
    for k, values, pass_rate in zip(num_flows, values_by_k, pass_rate_by_k):
        if len(values) == 0:
            continue
        positions.append(k + position_offset)
        data.append(values)
        plotted_k.append(k)
        colors.append(fill_color if fill_color is not None
                       else (pass_color if pass_rate >= pass_threshold else fail_color))
    if not data:
        return {}
    bp = axis.boxplot(data, positions=positions, widths=box_width, patch_artist=True,
                       showfliers=False, manage_ticks=False, zorder=2)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor(style['edge_color'])
        patch.set_linewidth(edge_width)
        patch.set_linestyle(style['edge_style'])
    for part in ('whiskers', 'caps'):
        for line in bp[part]:
            line.set_color(style['edge_color'])
            line.set_linewidth(edge_width)
            line.set_linestyle(style['edge_style'])
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2.5)
    # bp['caps'] alternates (bottom, top) per box, in the same order as `plotted_k`.
    return {k: bp['caps'][2 * i + 1].get_ydata()[0] for i, k in enumerate(plotted_k)}


def _annotate_all_packets_burstiness(axis, num_flows, offset, y_by_k, burstiness_by_k):
    """Small horizontal text label right above each all-packets position on a flow-count
    plot, showing its burstiness (see burstiness_metrics): IDC at one RTT and mean burst
    duration/inter-burst gap. `y_by_k` is the y-value to anchor each label above (the
    all-packets point itself, or the top of its box once aggregated across experiments) --
    a k with no finite anchor or no burstiness data is simply skipped.

    Horizontal, not rotated: a rotated multi-line block's *width* (the longest line, easily
    100+ points at this font size) becomes its on-screen *height*, needlessly inflating
    whatever margin fig.tight_layout() reserves above the axes for it.

    Deliberately does NOT pass `annotation_clip=False`: that flag makes fig.tight_layout()
    treat the annotation as able to render anywhere, unbounded, and it responds by reserving
    a huge margin (observed: the axes shrinking to under half the figure height) just in
    case -- even though `xy` here is always the plotted data's own max, so it is always
    inside the axes' own view already and never needs clipping protection in the first
    place. Dropping the flag (the default already keeps it visible) fixes that outsized
    margin with no change to which labels actually get drawn."""
    if not burstiness_by_k:
        return
    idc = burstiness_by_k.get('idc_1rtt', [])
    dur = burstiness_by_k.get('avg_burst_duration_ns', [])
    gap = burstiness_by_k.get('avg_burst_interarrival_ns', [])
    for i, k in enumerate(num_flows):
        if i >= len(y_by_k) or not np.isfinite(y_by_k[i]):
            continue
        parts = []
        if i < len(idc) and idc[i] == idc[i]:
            parts.append('IDC(1RTT)={:.2g}'.format(idc[i]))
        if i < len(dur) and dur[i] == dur[i]:
            parts.append('burst_dur={:.3g}ns'.format(dur[i]))
        if i < len(gap) and gap[i] == gap[i]:
            parts.append('burst_gap={:.3g}ns'.format(gap[i]))
        if not parts:
            continue
        axis.annotate('\n'.join(parts), xy=(k + offset, y_by_k[i]), xytext=(0, 4),
                       textcoords='offset points', ha='center', va='bottom',
                       fontsize=7, color='0.25', zorder=4)


def _all_packets_spread_label(results):
    """What an all-packets boxplot's spread is taken over: 'runs' in a windowed result (the
    family was measured inside each run's own growing window, so it varies run to run) and
    'experiments' otherwise (there it is a fixed packet set within one experiment, and only
    differs between them). Purely for the legend, so a reader is never told a box spans
    experiments when it actually spans runs."""
    if results.get('all_packets_windowed'):
        return 'runs' + (' and experiments' if results.get('num_experiments', 1) > 1 else '')
    return 'experiments'


def _draw_all_packets_series(axis, num_flows, scalar_by_k, by_experiment, pass_rate_by_k, offset,
                              box_width, pass_threshold, pass_color, fail_color, num_runs,
                              num_experiments, quantity_name, fill_color=None, burstiness_by_k=None,
                              spread_label='experiments'):
    """Render the all-packets comparison family and return its legend handles.

    All packets of the first k flows is the same fixed set of packets on every run, so
    within one experiment its distance/error to the ground truth is a single number per k
    -- drawn as dots on a connecting line rather than a degenerate one-value boxplot. Once
    aggregated over several experiments it varies again (each experiment reconstructs its
    own ground truth), so it becomes a boxplot family like everything else. Both shapes are
    handled here so every per-flow-count plot renders all-packets identically.

    Pass `burstiness_by_k` (results['burstiness_all_packets']) to additionally annotate
    each all-packets position with its IDC(1RTT)/burst-duration/inter-burst-gap (see
    _annotate_all_packets_burstiness) -- omitted (None) wherever that would just clutter a
    plot that isn't about the all-packets family specifically."""
    # A spread with more than one value per k means the quantity is not fixed within this
    # result, whether that is because several experiments contributed (num_experiments > 1)
    # or because a windowed run measured the all-packets family in a different window every
    # run (see compute_emd_vs_num_tcp_flows_run) -- either way it is a distribution and
    # belongs in a boxplot rather than as a single dot.
    is_boxplot = bool(by_experiment) and (num_experiments > 1
                                           or any(len(values) > 1 for values in by_experiment))
    if fill_color is not None:
        pass_color = fail_color = fill_color
    if is_boxplot:
        whisker_top_by_k = _draw_boxplot_family(
            axis, num_flows, by_experiment, pass_rate_by_k, offset, box_width,
            pass_threshold, pass_color, fail_color, _ALL_PACKETS_STYLE, fill_color=fill_color)
        missing = [k for k, values in zip(num_flows, by_experiment) if len(values) == 0]
        handles = [Patch(facecolor='white', edgecolor=_ALL_PACKETS_STYLE['edge_color'], linewidth=4.5,
                          label='All packets of considered flows (boxplot across {})'.format(spread_label))]
        # The rendered whisker top, not the raw max -- showfliers=False hides anything
        # beyond it as an outlier, and anchoring to a value the box itself doesn't reach can
        # land outside the axes' own autoscaled view, silently dropping the annotation (see
        # _draw_boxplot_family).
        y_by_k = [whisker_top_by_k.get(k, np.nan) for k in num_flows]
    else:
        values = np.asarray(scalar_by_k, dtype=float)
        x = np.asarray(num_flows, dtype=float) + offset
        valid = np.isfinite(values)
        axis.plot(x[valid], values[valid], color='0.4', linewidth=2, zorder=1)
        rates = np.asarray(pass_rate_by_k, dtype=float)
        for mask, color in ((valid & (rates >= pass_threshold), pass_color),
                             (valid & ~(rates >= pass_threshold), fail_color)):
            if mask.any():
                axis.scatter(x[mask], values[mask], marker='o', color=color,
                             edgecolor='black', s=220, zorder=3)
        missing = [k for k, v in zip(num_flows, values) if not np.isfinite(v)]
        handles = [Line2D([0], [0], marker='o', color='0.4', markerfacecolor='white',
                           markeredgecolor='black', markersize=16, linewidth=2,
                           label='All packets of considered flows (dots + line)')]
        y_by_k = list(values)
    if missing:
        print("No all-packet {} value for {} flow-count(s), skipped: {}".format(
            quantity_name, len(missing), missing))
    _annotate_all_packets_burstiness(axis, num_flows, offset, y_by_k, burstiness_by_k)
    if fill_color is None:
        handles = [
            Patch(facecolor=pass_color, edgecolor='black', alpha=0.85,
                  label='Consistency check passed (>={:.0f}% of {} runs)'.format(pass_threshold * 100, num_runs)),
            Patch(facecolor=fail_color, edgecolor='black', alpha=0.85,
                  label='Consistency check failed (<{:.0f}% of {} runs)'.format(pass_threshold * 100, num_runs)),
        ] + handles
    return handles


def _subsample_family_layout(subsampling_methods, uniform_series, oracle_series=()):
    """Evenly space the all-packets series plus one boxplot family per
    comparison family -- one per Poisson-adaptive method, then one per uniform
    family in `uniform_series`, then one per ideal-Poisson-probe family in
    `oracle_series` -- around each flow-count tick. Returns
    (offset_all_packets, offsets_poisson (dict keyed by method name),
    offsets_uniform (dict keyed by uniform-series key), offsets_oracle (dict
    keyed by oracle-series key), box_width)."""
    subsampling_methods = list(subsampling_methods)
    uniform_series = list(uniform_series)
    oracle_series = list(oracle_series)
    # all-packets + one per Poisson-adaptive method + one per uniform + one per ideal probe
    n_slots = 1 + len(subsampling_methods) + len(uniform_series) + len(oracle_series)
    span = _FAMILY_GROUP_SPAN
    box_width = (span / n_slots) * _FAMILY_BOX_FILL
    offsets = np.linspace(-span / 2, span / 2, n_slots)
    offset_all_packets = offsets[0]
    offsets_poisson = {method: offsets[1 + i] for i, method in enumerate(subsampling_methods)}
    base = 1 + len(subsampling_methods)
    offsets_uniform = {key: offsets[base + i] for i, key in enumerate(uniform_series)}
    base += len(uniform_series)
    offsets_oracle = {key: offsets[base + i] for i, key in enumerate(oracle_series)}
    return offset_all_packets, offsets_poisson, offsets_uniform, offsets_oracle, box_width


_TRAFFIC_COLORS = ['navy', 'darkorange', 'purple', 'teal', 'crimson', 'olive']

# On the load plots border *colour* is taken by the traffic, so a series can only be
# identified by its border *style* -- and only that, deliberately: earlier revisions also
# varied border *width* and invented custom per-kind dash patterns to fit more series on one
# axis, but a thick dash-dot border on a narrow boxplot box reads almost like a thick dashed
# one -- variable thickness and synthetic dash patterns make things *harder* to tell apart, not
# easier. So a load plot uses only the four canonical, maximally-distinct matplotlib
# linestyles, always at the same width, one per series, in the order series are added to the
# plot (_STYLE_ORDER) -- not tied to family `kind` at all (a kind can land on a different style
# on a different plot, e.g. 'oracle' is 'dashed' on one comparison and 'dotted' on another,
# depending only on what order it was added on that particular plot). 'dotted' is deliberately
# placed before 'dashdot' in _STYLE_ORDER since dotted is the one most visually distinct from
# both solid and dashed, while dashdot is the one most easily confused with dashed at these
# widths -- so dashdot is the last style reached for, used only when a plot truly has four
# series. A plot with more than four series can't be told apart this way -- that warns loudly
# rather than silently repeating a style.
_STYLE_ORDER = ['solid', 'dashed', 'dotted', 'dashdot']
_LOAD_SERIES_EDGE_WIDTH = 4.5


def _load_series_spec(specs, kind, key, label):
    """Append one series spec to `specs`, taking the next unused style from _STYLE_ORDER (by
    position in `specs`, not by `kind` -- see the comment above _STYLE_ORDER) at a fixed
    border width. `kind` is kept on the spec only for callers/labels that want it, not used
    for styling."""
    index = len(specs)
    if index >= len(_STYLE_ORDER):
        print("Warning: {} series on one load plot exceeds the {} distinct border styles "
              "available; '{}' repeats an earlier series' style and the two cannot be told "
              "apart".format(index + 1, len(_STYLE_ORDER), label))
    specs.append(dict(key=key, kind=kind, edge_style=_STYLE_ORDER[index % len(_STYLE_ORDER)],
                       label=label, edge_width=_LOAD_SERIES_EDGE_WIDTH))
    return specs


def all_packets_vs_sampled_load_plot_series(subsampling_methods):
    """Series specs for plot_emd_vs_load_by_traffic comparing all packets of the
    considered flows (solid) against every Poisson-adaptive subsampling method that was
    run (dashed, dotted, ... one style per method, see _load_series_spec). With one method
    this is the pair this plot has always drawn; with several, each method gets its own
    style, so the algorithms are compared against each other and against the all-packets
    ceiling in one picture without relying on colour, which is spent on the traffic here.
    Keep this to at most 3 methods -- a 4th plus all_packets exceeds the 4 styles available."""
    specs = []
    _load_series_spec(specs, 'all_packets', 'all_packets', 'all packets of considered flows')
    for method in normalize_subsampling_methods(subsampling_methods):
        _load_series_spec(specs, 'sampled', ('sampled', method),
                           'Poisson-adaptive subsample ({})'.format(method))
    return specs


def sampled_vs_oracle_load_plot_series(subsampling_method):
    """Series specs pairing ONE Poisson-adaptive subsampling method (solid) against the ideal
    Poisson probe at that method's own sample count (dashed), plus the probe at the minimum
    required sample size (dotted). The gap between the method and its own ideal probe is the
    part of its error that is *not* finite-sample noise -- i.e. what selecting from the
    flow's own packets costs.

    One method per plot on purpose: three series is the most one load plot can show while
    keeping every border a distinct style (see _load_series_spec) with colour already spent
    on the traffic. Call once per method."""
    method = normalize_subsampling_methods(subsampling_method)[0]
    specs = []
    _load_series_spec(specs, 'sampled', ('sampled', method),
                       'Poisson-adaptive subsample ({})'.format(method))
    _load_series_spec(specs, 'oracle', ('oracle', method), _oracle_series_label(method))
    _load_series_spec(specs, 'oracle', ('oracle', ORACLE_MIN_REQUIRED_KEY),
                       _oracle_series_label(ORACLE_MIN_REQUIRED_KEY))
    return specs


def poisson_vs_uniform_vs_ideal_load_plot_series(subsampling_methods):
    """Series specs for plot_emd_vs_load_by_traffic putting THREE families on one axis per
    Poisson-adaptive method: the method itself (solid), its own rate-matched uniform baseline
    (dashed), and the ideal Poisson probe at that same sample count (dotted) -- all three draw
    ~the same number of packets, so the three-way reads as a verdict on the selection rule
    (method vs. uniform) *and* on how much of the method's own remaining error is
    finite-sample noise vs. selection bias (method vs. its ideal probe) in one picture. Pass
    one method for a clean three-series plot (several would exceed the 4 distinct styles a
    load plot can show, see _load_series_spec)."""
    specs = []
    for method in normalize_subsampling_methods(subsampling_methods):
        _load_series_spec(specs, 'sampled', ('sampled', method),
                           'Poisson-adaptive subsample ({})'.format(method))
        _load_series_spec(specs, 'uniform', ('uniform', method), _uniform_series_label(method))
        _load_series_spec(specs, 'oracle', ('oracle', method), _oracle_series_label(method))
    return specs


def all_packets_vs_oracle_load_plot_series(subsampling_methods):
    """Series specs for plot_emd_vs_load_by_traffic comparing all packets of the considered
    flows (solid) against the ideal Poisson probe (one style per method, see
    _load_series_spec) at each Poisson-adaptive method's own sample count -- the theoretical
    ceiling a real sampler could reach, independent of any particular sampler's own
    selection-rule imperfections. Reads alongside all_packets_vs_sampled_load_plot_series: the
    gap there that ISN'T explained by this plot is what the real sampler itself is costing,
    versus what subsampling costs in principle. Keep this to at most 3 methods -- a 4th plus
    all_packets exceeds the 4 styles available."""
    specs = []
    _load_series_spec(specs, 'all_packets', 'all_packets', 'all packets of considered flows')
    for method in normalize_subsampling_methods(subsampling_methods):
        _load_series_spec(specs, 'oracle', ('oracle', method), _oracle_series_label(method))
    return specs


def all_packets_vs_own_rate_oracle_load_plot_series():
    """Series specs pairing all packets of the considered flows (solid) against the ideal
    Poisson probe run at that SAME rate/count (dashed, ORACLE_ALL_PACKETS_RATE_KEY) -- with
    no method-specific comparison mixed in, this isolates exactly what "arriving as a real
    application's traffic" costs relative to an idealized Poisson process at an identical
    rate, no subsampling involved at all. See all_packets_vs_oracle_load_plot_series for the
    per-subsampling-method version (each method's own, smaller, sample count)."""
    specs = []
    _load_series_spec(specs, 'all_packets', 'all_packets', 'all packets of considered flows')
    _load_series_spec(specs, 'oracle', ('oracle', ORACLE_ALL_PACKETS_RATE_KEY),
                       _oracle_series_label(ORACLE_ALL_PACKETS_RATE_KEY))
    return specs


def _series_key_label(series_key):
    """Human-readable name of a series key ('all_packets', ('sampled', method),
    ('uniform', key), or a bare 'sampled'), for legends and titles."""
    if series_key == 'all_packets':
        return 'all packets'
    if series_key == 'sampled':
        return 'Poisson-adaptive subsample'
    if isinstance(series_key, tuple) and series_key[0] == 'sampled':
        return 'Poisson-adaptive subsample ({})'.format(series_key[1])
    if isinstance(series_key, tuple) and series_key[0] == 'uniform':
        return _uniform_series_label(series_key[1])
    if isinstance(series_key, tuple) and series_key[0] == 'oracle':
        return _oracle_series_label(series_key[1])
    return str(series_key)


def _metric_result_keys(metric, normalized):
    """The (all-packets scalar key, all-packets per-experiment key, sampled key, uniform key,
    axis label) quadruple+label naming where one plotted quantity lives in a results dict.

    `metric` is 'emd' (the default), or ('percentile_diff', q) / ('percentile_reldiff', q)
    for the signed p-q error. `normalized` only applies to the EMD, whose normalized twin is
    a separate stored series; the percentile error's "relative" form is its own metric."""
    if metric in (None, 'emd'):
        suffix = '_normalized' if normalized else ''
        return ('emd_all_packets' + suffix,
                'emd_all_packets_by_experiment' + suffix,
                'emd_sampled_packets_by_run' + suffix,
                'emd_uniform_packets_by_run' + suffix,
                'emd_oracle_by_run' + suffix,
                "EMD relative to mean queuing delay" if normalized
                else "EMD to reconstructed network delay CDF (ns)")
    if isinstance(metric, tuple) and metric[0] in ('percentile_diff', 'percentile_reldiff'):
        name, q = metric
        label = ("Ground-truth p{0} - family p{0}, relative to ground-truth p{0}" if name.endswith('reldiff')
                  else "Ground-truth p{0} - family p{0} (ns)").format(q)
        return ('{}_all_packets'.format(name),
                '{}_all_packets_by_experiment'.format(name),
                '{}_sampled_by_run'.format(name),
                '{}_uniform_by_run'.format(name),
                '{}_oracle_by_run'.format(name),
                label)
    if _metric_is_delay_mean(metric):
        return (None, None, None, None, None, "Mean queuing delay (ns)")
    runstat = _metric_runstat_name(metric)
    if runstat is not None:
        return (None, None, None, None, None, RUNSTAT_METRICS[runstat]['label'])
    prob_spec = _metric_prob_spec(metric)
    if prob_spec is not None:
        metric_name, quantity = prob_spec
        label = ('{} ({})'.format(PROB_PLOT_QUANTITIES[quantity], prob_metric_label(metric_name).lower())
                  if quantity != 'prob' else prob_metric_label(metric_name))
        return (None, None, None, None, None, label)
    if metric == 'percentile_avg_relerror':
        # Mean absolute percentage error over a dense, fixed percentile grid (see
        # percentile_avg_relative_error) -- self-normalized already, so `normalized` does
        # not apply (there is only one stored series). Carries no separate q dimension
        # (_metric_percentile returns None for it, so _load_plot_series_values's `_at`
        # skips the q-indexing step) -- unlike ('percentile_diff'|'percentile_reldiff', q)
        # above, which are about one specific, sparsely-tracked percentile.
        return ('percentile_avg_relerror_all_packets',
                'percentile_avg_relerror_all_packets_by_experiment',
                'percentile_avg_relerror_sampled_by_run',
                'percentile_avg_relerror_uniform_by_run',
                'percentile_avg_relerror_oracle_by_run',
                "Mean absolute relative percentile error (avg over dense percentile grid)")
    raise ValueError("Unknown metric: {!r}".format(metric))


def _metric_percentile(metric):
    """The percentile a metric is about, or None for a whole-distribution metric."""
    if isinstance(metric, tuple) and metric[0] in ('percentile_diff', 'percentile_reldiff'):
        return metric[1]
    return None


# The quantities a probability metric can be plotted as, with the axis label each gets.
PROB_PLOT_QUANTITIES = {
    'prob': 'estimate',
    # |estimate - reference| IS the Wasserstein distance between the two Bernoullis, so it
    # is named EMD like the delay side's, with the same relative twin.
    'distance': 'EMD to the reference probability',
    'distance_normalized': 'EMD relative to the reference probability',
    'log_diff': 'log(estimate) - SUM log(segment probability)',
}


def prob_plot_metric(metric_name, quantity='distance'):
    """The `metric` identifier that selects one probability metric's quantity in the
    cross-traffic load/burstiness plots: ('prob', <PROB_METRICS key>, <PROB_PLOT_QUANTITIES
    key>). Delay metrics stay exactly as they were ('emd', ('percentile_diff', q), ...), so
    nothing about the existing plots changes."""
    if metric_name not in PROB_METRICS:
        raise ValueError("Unknown probability metric {!r}; choose one of {}".format(
            metric_name, list(PROB_METRICS)))
    if quantity not in PROB_PLOT_QUANTITIES:
        raise ValueError("Unknown probability quantity {!r}; choose one of {}".format(
            quantity, list(PROB_PLOT_QUANTITIES)))
    return ('prob', metric_name, quantity)


DELAY_MEAN_METRIC = ('delay', 'mean')


# Per-run quantities that are properties of the MEASUREMENT rather than distances to the
# ground truth: how many packets a family ended up holding, how long the growing-window
# search had to watch to get them, and how tight the consistency check's own bound was at
# that sample size. All three are already recorded per run and plotted per experiment
# (plot_sample_sizes_vs_num_flows, plot_monitor_window_vs_num_flows,
# plot_error_bound_vs_num_flows); routing them through the same ('runstat', <name>) metric
# plumbing the probability metrics use makes them available to the cross-traffic
# load/burstiness/fraction plots too, with no new plotting function.
#
# `families` lists which comparison series carry the quantity at all -- only the
# Poisson-adaptive families settle on a monitoring window, so an all-packets or uniform box
# would be empty on that plot rather than zero, and is left out instead of drawn empty.
RUNSTAT_METRICS = OrderedDict((
    ('sample_size', {
        'label': 'Retained sample size (packets)',
        'prefix': 'sample_sizes_',
        'families': ('all_packets', 'sampled', 'uniform', 'oracle'),
        'scale': 1.0,
    }),
    ('monitor_window', {
        'label': 'Monitoring window needed (ms)',
        'prefix': 'window_duration_',
        'families': ('sampled',),
        'scale': 1e-6,   # stored in ns
    }),
    ('error_bound', {
        'label': 'Consistency-check error bound (fraction of mean delay)',
        'prefix': 'error_bound_',
        'families': ('all_packets', 'sampled', 'uniform', 'oracle'),
        'scale': 1.0,
    }),
))


def runstat_plot_metric(name):
    """The `metric` identifier selecting one per-run measurement property (sample size,
    monitoring window, relative error bound) in the cross-traffic load/burstiness/fraction
    plots: ('runstat', <RUNSTAT_METRICS key>)."""
    if name not in RUNSTAT_METRICS:
        raise ValueError("Unknown run-statistic metric {!r}; choose one of {}".format(
            name, list(RUNSTAT_METRICS)))
    return ('runstat', name)


def _metric_runstat_name(metric):
    """The RUNSTAT_METRICS key when `metric` selects a per-run measurement property, else
    None."""
    if (isinstance(metric, tuple) and len(metric) == 2 and metric[0] == 'runstat'
            and metric[1] in RUNSTAT_METRICS):
        return metric[1]
    return None


def _runstat_plot_series_values(r, i, series_key, name):
    """(values, pass_rate) for one comparison series' per-run measurement property at
    flow-count index `i` -- the RUNSTAT_METRICS counterpart of _prob_plot_series_values.
    Empty for a series that does not carry the quantity (see RUNSTAT_METRICS['families'])
    and for results predating its recording, so an older aggregation simply plots nothing
    rather than failing."""
    spec = RUNSTAT_METRICS[name]
    n_k = len(r.get('num_flows') or [])
    if series_key == 'sampled':
        series_key = ('sampled', r['subsampling_methods'][0])
    kind = series_key if isinstance(series_key, str) else series_key[0]
    if kind not in spec['families']:
        return [], 0.0
    if series_key == 'all_packets':
        if name == 'sample_size':
            # All packets is not sampled per run: its count is one scalar per flow count
            # (per experiment, once aggregated), not a per-run distribution.
            by_experiment = r.get('all_packet_sizes_by_experiment')
            if by_experiment:
                values = by_experiment
            else:
                scalars = r.get('all_packet_sizes') or []
                values = [[v] for v in scalars]
        else:
            values = r.get(spec['prefix'] + 'all_packets_by_run') or [[]] * n_k
        rates = r.get('pass_rate_all_packets') or []
    else:
        kind, key = series_key
        values = (r.get(spec['prefix'] + kind + '_by_run') or {}).get(key) or [[]] * n_k
        rates = (r.get('pass_rate_' + kind) or {}).get(key) or []
    per_k = list(values[i]) if i < len(values) else []
    scale = spec['scale']
    return [float(v) * scale for v in per_k if v == v], (rates[i] if i < len(rates) else 0.0)


def _metric_is_delay_mean(metric):
    """Whether `metric` selects the per-family MEAN QUEUING DELAY (DELAY_MEAN_METRIC) rather
    than a distance to the ground truth -- the delay counterpart of a probability metric's
    'prob' quantity, see plot_delay_value_vs_num_flows."""
    return metric == DELAY_MEAN_METRIC


def _delay_mean_plot_series_values(r, i, series_key):
    """(values, pass_rate) for one comparison series' mean queuing delay at flow-count index
    `i` -- the DELAY_MEAN_METRIC counterpart of _load_plot_series_values' EMD branches,
    reading the per-run mean each family recorded."""
    n_k = len(r.get('num_flows') or [])
    if series_key == 'sampled':
        series_key = ('sampled', r['subsampling_methods'][0])
    if series_key == 'all_packets':
        values = r.get('delay_mean_all_packets_by_run') or [[]] * n_k
        rates = r.get('pass_rate_all_packets') or []
    elif isinstance(series_key, tuple) and series_key[0] in ('sampled', 'uniform', 'oracle'):
        kind, key = series_key
        values = (r.get('delay_mean_{}_by_run'.format(kind)) or {}).get(key) or [[]] * n_k
        rates = (r.get('pass_rate_' + kind) or {}).get(key) or []
    else:
        raise ValueError("Unknown series_key: {!r}".format(series_key))
    per_k = list(values[i]) if i < len(values) else []
    return [v for v in per_k if v == v], (rates[i] if i < len(rates) else 0.0)


def _metric_autoscales(metric):
    """Whether a plotted metric's y-axis should autoscale instead of taking the default
    +/-100% / 500ns view cap: true for every probability quantity (all inherently bounded
    and small, so the cap would spend the axis on empty space) and for the mean queuing
    delay (an absolute level, routinely past a 500ns cap built for distances)."""
    return (_metric_prob_spec(metric) is not None or _metric_is_delay_mean(metric)
            or _metric_runstat_name(metric) is not None)


def _metric_prob_spec(metric):
    """(metric_name, quantity) when `metric` selects a probability metric, else None."""
    if (isinstance(metric, tuple) and len(metric) == 3 and metric[0] == 'prob'
            and metric[1] in PROB_METRICS):
        return metric[1], metric[2]
    return None


def _prob_plot_series_values(r, i, series_key, metric_name, quantity):
    """(values, pass_rate) for one comparison series' probability metric at flow-count index
    `i` -- the probability counterpart of _load_plot_series_values' delay branches, reading
    the nested per-metric block compute_emd_vs_num_tcp_flows_multi_run stores (see
    'prob_metrics'). Empty when this result carries no such metric (it predates them) or
    the series is not one of its families."""
    block = (r.get('prob_metrics') or {}).get(metric_name)
    if not block:
        return [], 0.0
    n_k = len(r.get('num_flows') or [])
    if series_key == 'sampled':
        series_key = ('sampled', r['subsampling_methods'][0])
    if series_key == 'all_packets':
        family, rates = block.get('all_packets_by_run') or {}, block.get('pass_rate_all_packets') or []
    elif isinstance(series_key, tuple) and series_key[0] in ('sampled', 'uniform', 'oracle'):
        kind, key = series_key
        family = (block.get(kind + '_by_run') or {}).get(key) or {}
        rates = (block.get('pass_rate_' + kind) or {}).get(key) or []
    else:
        raise ValueError("Unknown series_key: {!r}".format(series_key))
    if not family:
        return [], 0.0
    values = (family.get(quantity) or [[]] * n_k)
    values = list(values[i]) if i < len(values) else []
    values = [v for v in values if v == v]  # drop NaN
    return values, (rates[i] if i < len(rates) else 0.0)


def _metric_is_relative(metric, normalized):
    """Whether a plotted (metric, normalized) combination is a fraction/ratio (view-capped at
    +/-100% by default) rather than an absolute ns quantity (capped at 500ns): normalized EMD,
    a relative percentile error, or percentile_avg_relative_error's mean relative percentile
    error -- itself already self-normalized regardless of `normalized`, see
    _metric_result_keys. Shared by plot_emd_vs_load_by_traffic, plot_emd_vs_burstiness_by_traffic
    and plot_emd_vs_num_flows_boxplot_by_flow so a metric added to one is classified the same
    way everywhere, rather than each copy risking its own default-cap misclassification."""
    return (normalized or metric == 'percentile_avg_relerror'
            or _metric_prob_spec(metric) is not None
            or (isinstance(metric, tuple) and metric[0] == 'percentile_reldiff'))


def _set_flow_count_xaxis(axis, num_flows):
    """x-axis ticks/limits for a flow-count plot (plot_emd_vs_num_flows_boxplot and its
    percentile/mean-diff/Poisson-split siblings). A pooled all_flows_only result collapses
    every experiment onto the single ALL_FLOWS_ONLY_K sentinel (see
    aggregate_emd_vs_flows_results) rather than each experiment's own incidental total flow
    count, so render that one tick as text ('all packets') instead of the sentinel number,
    with fixed padding around it since there is no neighboring tick to space against."""
    axis.set_xticks(num_flows)
    if list(num_flows) == [ALL_FLOWS_ONLY_K]:
        axis.set_xticklabels(['all packets'])
        axis.set_xlim(ALL_FLOWS_ONLY_K - 0.6, ALL_FLOWS_ONLY_K + 0.6)
    else:
        axis.set_xlim(min(num_flows) - 0.6, max(num_flows) + 0.6)


def _per_run_series(series_key):
    """Whether a series spec's key names a family with one value PER RUN (the
    Poisson-adaptive subsample, its rate-matched uniform baseline, the ideal Poisson
    probes) rather than one per experiment ('all_packets', outside a growing-window run).
    Only the former can be compared against MIN_POISSONIZED_RUNS directly."""
    if series_key == 'sampled':
        return True
    return isinstance(series_key, tuple) and series_key[0] in ('sampled', 'uniform', 'oracle')


def _load_plot_series_values(r, i, series_key, normalized=False, metric='emd'):
    """Return (values, pass_rate) for one series spec's `key` at flow-count index `i` of an
    aggregated/single results dict `r` (see plot_emd_vs_load_by_traffic). `series_key` is
    'all_packets', ('sampled', method), or ('uniform', key) -- a bare 'sampled' still
    works and resolves to the results' first subsampling method.

    `metric` selects *which* quantity ('emd', or ('percentile_diff'|'percentile_reldiff', q)
    -- see _metric_result_keys); `normalized` additionally switches the EMD to its
    normalized twin. Pass rates are unaffected by either, since the consistency check
    itself is always the same mean-delay test."""
    all_key, all_by_exp_key, sampled_key, uniform_key, oracle_key, _ = _metric_result_keys(
        metric, normalized)
    prob_spec = _metric_prob_spec(metric)
    if prob_spec is not None:
        return _prob_plot_series_values(r, i, series_key, *prob_spec)
    if _metric_is_delay_mean(metric):
        return _delay_mean_plot_series_values(r, i, series_key)
    runstat = _metric_runstat_name(metric)
    if runstat is not None:
        return _runstat_plot_series_values(r, i, series_key, runstat)
    q = _metric_percentile(metric)
    if q is not None and q not in (r.get('delay_percentiles') or []):
        return [], 0.0

    def _at(container):
        # Percentile series are nested one level deeper: {q: {name: per-k}}.
        return container[q] if q is not None else container

    if series_key == 'all_packets':
        by_experiment = r.get(all_by_exp_key)
        if by_experiment:
            values = _at(by_experiment)[i]
        else:
            values = [_at(r[all_key])[i]]
        values = [v for v in values if v == v]  # drop NaN
        return values, r['pass_rate_all_packets'][i]
    if series_key == 'sampled':
        series_key = ('sampled', r['subsampling_methods'][0])
    if isinstance(series_key, tuple) and series_key[0] == 'sampled':
        method = series_key[1]
        if method not in r['emd_sampled_packets_by_run']:
            return [], 0.0
        return _at(r[sampled_key])[method][i], r['pass_rate_sampled'][method][i]
    if isinstance(series_key, tuple) and series_key[0] == 'uniform':
        key = series_key[1]
        if key not in r['emd_uniform_packets_by_run']:
            return [], 0.0
        return _at(r[uniform_key])[key][i], r['pass_rate_uniform'][key][i]
    if isinstance(series_key, tuple) and series_key[0] == 'oracle':
        key = series_key[1]
        if key not in (r.get('emd_oracle_by_run') or {}):
            return [], 0.0
        return _at(r[oracle_key])[key][i], r['pass_rate_oracle'][key][i]
    raise ValueError("Unknown series_key: {!r}".format(series_key))


def _load_plot_layout(n_traffics, loads, n_series=2):
    """Offsets/box-width for plot_emd_vs_load_by_traffic: `n_series` boxes per traffic,
    grouped as an adjacent cluster, evenly spread around each load tick. Unlike the
    flow-count plots (whose x-ticks are integers spaced >=1 apart), loads are typically
    spaced ~0.1 apart, so the spread must scale with the *actual* gap between loads rather
    than a fixed constant -- otherwise neighboring load groups collide."""
    n_slots = max(n_series * n_traffics, 1)
    min_gap = float(np.min(np.diff(sorted(loads)))) if len(loads) > 1 else 1.0
    span = min_gap * _FAMILY_GROUP_SPAN
    box_width = (span / n_slots) * _FAMILY_BOX_FILL
    offsets = np.linspace(-span / 2, span / 2, n_slots) if n_slots > 1 else np.array([0.0])
    return offsets, box_width, span


# Minimum comfortable rendered box width (inches, at the fig.savefig dpi=150 these plots use)
# below which adjacent boxes' fixed-point-width borders start to visually merge -- the
# "collide" complaint. _load_plot_figsize back-solves the figure width needed to keep every
# box at least this wide given how many (traffic x series) slots share each load tick.
_LOAD_PLOT_MIN_BOX_WIDTH_IN = 0.30


def _load_plot_figsize(n_traffics, n_series, n_loads, height=15.0):
    """Figure size for plot_emd_vs_load_by_traffic/plot_burstiness_vs_load_by_traffic: wide
    enough that every box in the busiest tick's cluster (n_series * n_traffics boxes) stays
    at least _LOAD_PLOT_MIN_BOX_WIDTH_IN wide, given _load_plot_layout's span/fill geometry
    -- rather than a fixed width that looks fine with 2-3 traffics but packs boxes into
    illegibility once there are 5+ traffics x 2-3 series per tick. Never smaller than the
    original fixed 30in default."""
    n_slots = max(n_series * n_traffics, 1)
    n_loads = max(n_loads, 1)
    tick_spacing_needed = (n_slots * _LOAD_PLOT_MIN_BOX_WIDTH_IN / _FAMILY_BOX_FILL) / _FAMILY_GROUP_SPAN
    width = max(30.0, tick_spacing_needed * n_loads * 1.15)
    return (width, height)


# Below this fraction of a plot's data actually visible at the default cap, the cap is
# widened rather than left to quietly hide most of the load sweep -- see _adaptive_view_cap.
_MIN_VISIBLE_FRACTION = 0.5
# ... but even then, only widened enough to show this fraction, not the full range (a single
# extreme run could otherwise blow the axis back out to where everything else gets crushed).
_TARGET_VISIBLE_FRACTION = 0.8


def _adaptive_view_cap(values_by_series, default_cap, signed):
    """The y-axis view limits for plot_emd_vs_load_by_traffic / plot_emd_vs_num_flows_boxplot_by_flow:
    `default_cap` (the usual +/-100% or 500ns) unless fewer than _MIN_VISIBLE_FRACTION of
    *the whole plot's* pooled values would actually fall within it, in which case the cap is
    raised to whatever value brings _TARGET_VISIBLE_FRACTION of that pooled data into view
    (its 80th percentile) -- e.g. a heavy-tailed traffic in the mix whose errors mostly
    exceed the default cap, rather than silently rendering a plot where most of that
    traffic's boxes are invisibly clipped.

    Pooled across every box on the plot deliberately (not decided per box): letting a single
    outlier-heavy box's own 80th percentile set the cap would stretch the shared axis out to
    cover it and crush every other, better-behaved box into a sliver at the bottom -- worse
    for the plot as a whole than that one box occasionally reading as clipped at the cap.
    `values_by_series` is still a list of per-box value arrays (see the call sites), pooled
    here by concatenation -- so a box backed by many raw points (e.g. per-run values pooled
    across every run of every experiment) does carry proportionally more weight in the
    pooled visible_fraction than one backed by few (e.g. one experiment-level value per
    experiment). A box that is both low-n and genuinely extreme can therefore still end up
    clipped at the cap with nothing on the plot flagging it -- accepted here as the smaller
    problem next to blowing out the shared axis for every other box on its behalf.

    Never lowers the cap below `default_cap`. Returns (bottom, top, cap_was_widened)."""
    finite = (np.concatenate([np.abs(np.asarray(values, dtype=float)) for values in values_by_series])
              if values_by_series else np.array([]))
    finite = finite[np.isfinite(finite)]
    cap = default_cap
    widened = False
    if finite.size:
        visible_fraction = float(np.mean(finite <= default_cap))
        if visible_fraction < _MIN_VISIBLE_FRACTION:
            widened_cap = float(np.percentile(finite, _TARGET_VISIBLE_FRACTION * 100))
            if widened_cap > cap:
                cap = widened_cap
                widened = True
    bottom = -cap if signed else 0.0
    return bottom, cap, widened


def plot_emd_vs_load_by_traffic(results_by_traffic_load, k, output_path, pass_threshold=0.9, title=None,
                                 series_specs=None, normalized=False, metric='emd',
                                 x_label='Load'):
    """Cross-traffic, cross-load comparison at one fixed flow count `k`: x-axis is load,
    y-axis is EMD to the reconstructed ground-truth delay CDF. Both comparison series are
    drawn together -- by default all packets of the k considered flows, and the
    Poisson-adaptive subsample -- as one boxplot cluster per traffic per load
    (len(series_specs) x len(traffics) boxes at each load tick). Color identifies the
    traffic (_TRAFFIC_COLORS); the border identifies the series -- each series in
    `series_specs` gets one of the four canonical linestyles (solid, dashed, dotted,
    dash-dot), assigned fresh per plot in the order the series were added (see
    _load_series_spec / _STYLE_ORDER), all drawn at the same width, since colour is already
    spent on the traffic here; fill is only ever the pass/fail color. At most 4 series per
    plot can be told apart this way.

    The y-axis is capped (view only, not the underlying data) to keep one extreme run from
    washing out the rest of the load sweep: relative quantities (`normalized`, or
    `metric=('percentile_reldiff', q)`) to +/-100%, absolute ones (raw EMD in ns, or
    `('percentile_diff', q)`) to 500ns (+/-500ns if signed) -- only ever narrowing the
    autoscaled range, never expanding a tighter one, with a note drawn on the plot when it
    actually clips something.

    `series_specs` is a list of {'key', 'edge_style', 'label'} dicts (see
    _load_plot_series_values for valid `key`s); defaults to
    all_packets_vs_sampled_load_plot_series over whichever Poisson-adaptive methods the
    `x_label` renames the x-axis only. The second key of `results_by_traffic_load` is used
    purely as a number to place each cluster on that axis, so passing a dict keyed by
    (traffic, tbfFlowRedirectFraction) together with x_label='Differentiation fraction'
    yields the same plot over the reverse experiments' shaping fraction instead of the load
    (see aggregate_emd_vs_flows_across_traffics_and_fractions) -- same boxes, same
    consistency-pass colouring, same series.

    results actually contain. Pass poisson_vs_uniform_load_plot_series(stride, methods)
    instead to compare those methods against a uniform "1-in-stride" subsample the same
    way. Any number of series is supported, not just 2.

    With `normalized` set, the y-axis is the EMD divided by the mean ground-truth path
    delay (normalize_emd_values) instead of raw nanoseconds -- the comparable-across-loads
    view, since raw EMD grows with the delay level that load itself drives.

    `metric` switches the plotted quantity away from the EMD entirely: pass
    ('percentile_diff', q) for the signed absolute p-q error in ns, or
    ('percentile_reldiff', q) for it as a fraction of the ground truth's own p-q (see
    _metric_result_keys). A combination whose results carry no such percentile is left
    without boxes rather than failing.

    Whenever `series_specs` includes the 'all_packets' key, each (traffic, load)'s
    all-packets box is also annotated with its burstiness (IDC(1RTT), mean burst
    duration/inter-burst gap -- see burstiness_metrics), via the same
    _annotate_all_packets_burstiness helper plot_emd_vs_num_flows_boxplot uses.

    `k` is normally an int looked up exactly in each combination's num_flows. Pass the
    string 'max' instead to use each (traffic, load) combination's own maximum flow count
    (its last num_flows entry) regardless of what that count actually is -- e.g. one
    combination's max might be 19 considered flows and another's 21; this compares "all the
    flows we have" for each, since the exact count isn't the point of that comparison.

    `results_by_traffic_load` is a dict {(traffic, load): results} where each `results` is
    what aggregate_emd_vs_flows_results (or a single run_emd_vs_flows_experiment call)
    produces. A (traffic, load) combination missing entirely, or with no data at this k for
    a given series, is simply left without a box there."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    use_max_k = (k == 'max')
    results_by_traffic_load = {key: upgrade_emd_vs_flows_results_schema(r)
                                for key, r in results_by_traffic_load.items()}
    if not series_specs:
        methods = []
        for r in results_by_traffic_load.values():
            for name in r['subsampling_methods']:
                if name not in methods:
                    methods.append(name)
        series_specs = all_packets_vs_sampled_load_plot_series(methods or ['find_samples_path'])

    traffics = sorted({t for (t, _l) in results_by_traffic_load})
    loads = sorted({l for (_t, l) in results_by_traffic_load})
    pass_color, fail_color = 'tab:green', 'tab:red'

    n_traffics = max(len(traffics), 1)
    n_series = max(len(series_specs), 1)
    offsets, box_width, span = _load_plot_layout(n_traffics, loads, n_series)

    fig, axis = plt.subplots(figsize=_load_plot_figsize(n_traffics, n_series, len(loads)))
    legend_handles = [
        Patch(facecolor=pass_color, edgecolor='black', alpha=0.85,
              label='Consistency check passed (≥{:.0f}%)'.format(pass_threshold * 100)),
        Patch(facecolor=fail_color, edgecolor='black', alpha=0.85,
              label='Consistency check failed (<{:.0f}%)'.format(pass_threshold * 100)),
    ]

    any_data = False
    all_plotted_values = []
    # (x, y-anchor, burstiness dict) for each (traffic, load) where the 'all_packets' series
    # is on this plot -- annotated after the main loop (see _annotate_all_packets_burstiness).
    burstiness_annotations = []
    # Cells dropped for resting on too few Poissonized runs, reported once after the loop
    # rather than per series (every series of a dropped cell hits the same condition), and
    # separately the individual thin families dropped inside cells that were otherwise fine.
    low_poissonization_cells = {}
    low_poissonization_series = {}
    for ti, traffic in enumerate(traffics):
        color = _TRAFFIC_COLORS[ti % len(_TRAFFIC_COLORS)]
        legend_handles.append(Patch(facecolor='white', edgecolor=color, linewidth=4.5, label=traffic))
        for si, series_spec in enumerate(series_specs):
            style = dict(edge_color=color, edge_style=series_spec['edge_style'])
            values_by_load, pass_rate_by_load = [], []
            burstiness_by_load = {}
            for load in loads:
                r = results_by_traffic_load.get((traffic, load))
                if r is None or not r['num_flows'] or (not use_max_k and k not in r['num_flows']):
                    values_by_load.append([])
                    pass_rate_by_load.append(0.0)
                    continue
                i = -1 if use_max_k else r['num_flows'].index(k)
                # Too few runs ever produced a subsample here for any box in this cell to
                # mean anything (see MIN_POISSONIZED_RUNS) -- drop the whole cell.
                n_poisson = poissonized_run_count(r, i)
                if n_poisson < MIN_POISSONIZED_RUNS:
                    low_poissonization_cells[(traffic, load)] = n_poisson
                    values_by_load.append([])
                    pass_rate_by_load.append(0.0)
                    continue
                values, pass_rate = _load_plot_series_values(r, i, series_spec['key'],
                                                             normalized=normalized, metric=metric)
                # ... and a single per-run family that is itself thin, even in a cell whose
                # other methods are well populated (e.g. one sampler succeeding in 1500 runs
                # while another managed 50).
                if (_per_run_series(series_spec['key']) and 0 < len(values) < MIN_POISSONIZED_RUNS):
                    low_poissonization_series[(traffic, load, series_spec['label'])] = len(values)
                    values = []
                values_by_load.append(values)
                pass_rate_by_load.append(pass_rate)
                all_plotted_values.append(values)
                if series_spec['key'] == 'all_packets' and len(values):
                    burstiness = (r.get('burstiness_all_packets') or {})
                    burstiness_by_load[load] = {field: [vals[i] if i < len(vals) else float('nan')]
                                                 for field, vals in burstiness.items()}

            if any(len(v) for v in values_by_load):
                any_data = True
            # Anchor each burstiness label at the box's actual rendered whisker top (returned
            # here), not the raw max computed above -- showfliers=False can hide the true max
            # as an outlier beyond that whisker, and anchoring there instead lands outside the
            # axes' own autoscaled view, silently dropping the annotation (see
            # _draw_boxplot_family / _draw_all_packets_series).
            whisker_top_by_load = _draw_boxplot_family(
                axis, loads, values_by_load, pass_rate_by_load, offsets[n_series * ti + si],
                box_width, pass_threshold, pass_color, fail_color, style,
                edge_width=series_spec.get('edge_width', 4.5))
            for load, burstiness in burstiness_by_load.items():
                if load in whisker_top_by_load:
                    burstiness_annotations.append((
                        load + offsets[n_series * ti + si], whisker_top_by_load[load], burstiness))

    for series_spec in series_specs:
        legend_handles.append(Line2D([0], [0], color='black', linestyle=series_spec['edge_style'],
                                      linewidth=series_spec.get('edge_width', 3),
                                      label='{} (border)'.format(series_spec['label'])))

    if low_poissonization_cells:
        print("plot_emd_vs_load_by_traffic: dropped {} traffic/load cell(s) with fewer than {} "
              "Poissonized runs: {}".format(
                  len(low_poissonization_cells), MIN_POISSONIZED_RUNS,
                  ', '.join('{} @ {} ({} run(s))'.format(t, l, n)
                             for (t, l), n in sorted(low_poissonization_cells.items()))))
        legend_handles.append(Line2D([0], [0], color='none',
                                      label='{} traffic/load cell(s) omitted: fewer than {} runs '
                                            'found a Poissonized subsample'.format(
                                                len(low_poissonization_cells), MIN_POISSONIZED_RUNS)))
    if low_poissonization_series:
        print("plot_emd_vs_load_by_traffic: dropped {} individual series with fewer than {} "
              "Poissonized runs (their cell had others above it): {}".format(
                  len(low_poissonization_series), MIN_POISSONIZED_RUNS,
                  ', '.join('{} @ {} / {} ({} run(s))'.format(t, l, label, n)
                             for (t, l, label), n in sorted(low_poissonization_series.items()))))
        legend_handles.append(Line2D([0], [0], color='none',
                                      label='{} individual box(es) omitted: that family found a '
                                            'Poissonized subsample in fewer than {} runs'.format(
                                                len(low_poissonization_series), MIN_POISSONIZED_RUNS)))

    if not any_data:
        print("plot_emd_vs_load_by_traffic: no data at k={}, writing empty plot".format(k))

    series_names = ' vs. '.join(s['label'] for s in series_specs)
    y_label = _metric_result_keys(metric, normalized)[-1]
    quantity_percentile = _metric_percentile(metric)
    if quantity_percentile is not None:
        quantity_name = 'p{} error{}'.format(quantity_percentile,
                                              ' (relative)' if metric[0].endswith('reldiff') else ' (ns)')
    else:
        quantity_name = 'EMD relative to mean queuing delay' if normalized else 'EMD'
    default_title = '{} vs load by traffic ({}), all considered flows (each combination\'s own max)'.format(quantity_name, series_names) if use_max_k \
        else '{} vs load by traffic ({}), k={}'.format(quantity_name, series_names, k)
    axis.set_title(title or default_title, fontsize=34)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    if quantity_percentile is not None:
        # Zero is "the family's tail matches the ground truth's" -- the reference the whole
        # plot is read against, unlike EMD where zero is just the axis floor.
        axis.axhline(0, color='black', linewidth=2, linestyle=':', zorder=1)

    # Cap the y-axis view to a known scale so a rare extreme run doesn't wash out the rest of
    # the load sweep -- same starting point as plot_emd_vs_num_flows_boxplot's y_max: relative
    # quantities (normalized EMD, or a percentile error taken as a fraction of the ground
    # truth's own p_q) default to +/-100%; absolute ones (raw EMD ns, or a percentile error in
    # ns) default to 500ns. Percentile errors are signed (ground truth - family) so their cap
    # is symmetric; EMD is never negative, so only its top is ever clipped. Unlike a fixed
    # cap, this is data-driven per box (see _adaptive_view_cap): if fewer than half a given
    # box's own values would actually be visible at the default, the cap widens to bring 80%
    # of THAT box into view -- e.g. a heavy-tailed workload mixed in with others that
    # comfortably fit the default.
    is_relative = _metric_is_relative(metric, normalized)
    signed = isinstance(metric, tuple)
    default_cap = 1.0 if is_relative else 500.0
    bottom, cap, cap_widened = _adaptive_view_cap(all_plotted_values, default_cap, signed)
    # A probability metric's quantities are inherently bounded and small (an estimate and a
    # distance live in [0,1], a log-difference within a fraction of it), so the +/-100%
    # default cap would spend the whole axis on empty space rather than resolving the range
    # the boxes actually occupy -- those axes autoscale. The computed cap is still kept,
    # since the burstiness-annotation clipping below reads it.
    if _metric_autoscales(metric):
        axis.autoscale(axis='y')
    else:
        axis.set_ylim(bottom=bottom, top=cap)

    # Drawn only now that the view is capped: a burstiness label anchored above its box's own
    # whisker top can still land above `cap` (a box can legitimately be taller than the capped
    # view), and annotate() clips to the axes' CURRENT limits at draw/save time regardless of
    # when it was called -- so anchoring before
    # this point can silently drop a label the same way an uncapped raw max did before.
    for x, y, burstiness in burstiness_annotations:
        _annotate_all_packets_burstiness(axis, [x], 0.0, [min(max(y, bottom), cap)], burstiness)

    axis.set_xticks(loads)
    if loads:
        pad = max(np.min(np.diff(loads)) * 0.6, span / 2 + box_width) if len(loads) > 1 else max(span / 2, 0.05)
        axis.set_xlim(min(loads) - pad, max(loads) + pad)
    axis.grid(True, alpha=0.35, axis='y')
    axis.legend(handles=legend_handles, fontsize=16, loc='best', ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_emd_vs_num_flows_boxplot_by_flow(results_by_flow, output_path, pass_threshold=0.9, title=None,
                                           series_specs=None, normalized=False, metric='emd'):
    """Multi-flow comparison at every flow count k: x-axis is the number of TCP flows
    considered, y-axis is the EMD (or other `metric`) to the reconstructed ground-truth
    delay CDF -- the same quantity plot_emd_vs_num_flows_boxplot draws for one flow, but
    with several named e2e flows drawn side by side instead of one (e.g. the reverse
    experiment's TBF-differentiated flow R0H0R2H3 against its undifferentiated control
    R0H1R2H3), one box-group per flow per k (len(series_specs) x len(results_by_flow)
    boxes at each k tick).

    Structurally this is plot_emd_vs_load_by_traffic with the x-axis swapped from load to
    flow count and the group axis swapped from traffic to (e2e) flow name -- same
    conventions apply: colour identifies the flow (_TRAFFIC_COLORS, reused here for
    "which named flow" rather than "which traffic"), the border identifies the series
    (one of the 4 canonical linestyles per _load_series_spec/_STYLE_ORDER, since colour is
    already spent on the flow), and fill is only ever the pass/fail colour.

    `series_specs` is a list of {'key', 'edge_style', 'label'} dicts (see
    _load_plot_series_values for valid `key`s); defaults to
    all_packets_vs_sampled_load_plot_series over whichever Poisson-adaptive methods the
    results actually contain. `metric`/`normalized` select the plotted quantity exactly as
    in plot_emd_vs_load_by_traffic: EMD raw/normalized, 'percentile_avg_relerror', or
    ('percentile_diff'|'percentile_reldiff', q) for one percentile's signed/relative error.

    `results_by_flow` is a dict {flow_name: results} where each `results` is what
    aggregate_emd_vs_flows_results (or a single run_emd_vs_flows_experiment call) produced
    for that flow. The k axis is the union of every flow's own num_flows; a flow missing a
    given k, or with no data there for a given series, is simply left without a box there."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_by_flow = {flow: upgrade_emd_vs_flows_results_schema(r) for flow, r in results_by_flow.items()}
    if not series_specs:
        methods = []
        for r in results_by_flow.values():
            for name in r['subsampling_methods']:
                if name not in methods:
                    methods.append(name)
        series_specs = all_packets_vs_sampled_load_plot_series(methods or ['find_samples_path'])

    flows = sorted(results_by_flow)
    all_k = sorted(set().union(*(set(r['num_flows']) for r in results_by_flow.values())))
    pass_color, fail_color = 'tab:green', 'tab:red'

    n_flows = max(len(flows), 1)
    n_series = max(len(series_specs), 1)
    offsets, box_width, span = _load_plot_layout(n_flows, all_k, n_series)

    fig, axis = plt.subplots(figsize=_load_plot_figsize(n_flows, n_series, len(all_k)))
    legend_handles = [
        Patch(facecolor=pass_color, edgecolor='black', alpha=0.85,
              label='Consistency check passed (>={:.0f}%)'.format(pass_threshold * 100)),
        Patch(facecolor=fail_color, edgecolor='black', alpha=0.85,
              label='Consistency check failed (<{:.0f}%)'.format(pass_threshold * 100)),
    ]

    any_data = False
    all_plotted_values = []
    burstiness_annotations = []
    for fi, flow in enumerate(flows):
        color = _TRAFFIC_COLORS[fi % len(_TRAFFIC_COLORS)]
        legend_handles.append(Patch(facecolor='white', edgecolor=color, linewidth=4.5, label=flow))
        r = results_by_flow[flow]
        for si, series_spec in enumerate(series_specs):
            style = dict(edge_color=color, edge_style=series_spec['edge_style'])
            values_by_k, pass_rate_by_k = [], []
            burstiness_by_tick = {}
            for k in all_k:
                if k not in r['num_flows']:
                    values_by_k.append([])
                    pass_rate_by_k.append(0.0)
                    continue
                i = r['num_flows'].index(k)
                values, pass_rate = _load_plot_series_values(r, i, series_spec['key'],
                                                             normalized=normalized, metric=metric)
                values_by_k.append(values)
                pass_rate_by_k.append(pass_rate)
                all_plotted_values.append(values)
                if series_spec['key'] == 'all_packets' and len(values):
                    burstiness = (r.get('burstiness_all_packets') or {})
                    burstiness_by_tick[k] = {field: [vals[i] if i < len(vals) else float('nan')]
                                              for field, vals in burstiness.items()}

            if any(len(v) for v in values_by_k):
                any_data = True
            # Anchor each burstiness label at the box's actual rendered whisker top (returned
            # here), not the raw max computed above -- see the identical fix in
            # plot_emd_vs_load_by_traffic / _draw_all_packets_series.
            whisker_top_by_tick = _draw_boxplot_family(
                axis, all_k, values_by_k, pass_rate_by_k, offsets[n_series * fi + si],
                box_width, pass_threshold, pass_color, fail_color, style,
                edge_width=series_spec.get('edge_width', 4.5))
            for k, burstiness in burstiness_by_tick.items():
                if k in whisker_top_by_tick:
                    burstiness_annotations.append((
                        k + offsets[n_series * fi + si], whisker_top_by_tick[k], burstiness))

    for series_spec in series_specs:
        legend_handles.append(Line2D([0], [0], color='black', linestyle=series_spec['edge_style'],
                                      linewidth=series_spec.get('edge_width', 3),
                                      label='{} (border)'.format(series_spec['label'])))

    if not any_data:
        print("plot_emd_vs_num_flows_boxplot_by_flow: no data, writing empty plot")

    series_names = ' vs. '.join(s['label'] for s in series_specs)
    flow_names_desc = ' vs. '.join(flows)
    y_label = _metric_result_keys(metric, normalized)[-1]
    quantity_percentile = _metric_percentile(metric)
    if quantity_percentile is not None:
        quantity_name = 'p{} error{}'.format(quantity_percentile,
                                              ' (relative)' if metric[0].endswith('reldiff') else ' (ns)')
    else:
        quantity_name = 'EMD relative to mean queuing delay' if normalized else 'EMD'
    default_title = '{} vs number of TCP flows, {} ({})'.format(quantity_name, flow_names_desc, series_names)
    axis.set_title(title or default_title, fontsize=34)
    axis.set_xlabel('Number of TCP flows considered')
    axis.set_ylabel(y_label)
    if quantity_percentile is not None:
        # Zero is "the family's tail matches the ground truth's" -- the reference the whole
        # plot is read against, unlike EMD where zero is just the axis floor.
        axis.axhline(0, color='black', linewidth=2, linestyle=':', zorder=1)

    # Same adaptive view cap as plot_emd_vs_load_by_traffic -- see _adaptive_view_cap.
    is_relative = _metric_is_relative(metric, normalized)
    signed = isinstance(metric, tuple)
    default_cap = 1.0 if is_relative else 500.0
    bottom, cap, cap_widened = _adaptive_view_cap(all_plotted_values, default_cap, signed)
    # A probability metric's quantities are inherently bounded and small (an estimate and a
    # distance live in [0,1], a log-difference within a fraction of it), so the +/-100%
    # default cap would spend the whole axis on empty space rather than resolving the range
    # the boxes actually occupy -- those axes autoscale. The computed cap is still kept,
    # since the burstiness-annotation clipping below reads it.
    if _metric_autoscales(metric):
        axis.autoscale(axis='y')
    else:
        axis.set_ylim(bottom=bottom, top=cap)

    # Drawn only now that the view is capped -- see the identical comment in
    # plot_emd_vs_load_by_traffic for why (a whisker top can still exceed a deliberately
    # capped view, and annotate() clips to the axes' CURRENT limits at draw/save time).
    for x, y, burstiness in burstiness_annotations:
        _annotate_all_packets_burstiness(axis, [x], 0.0, [min(max(y, bottom), cap)], burstiness)

    axis.set_xticks(all_k)
    if all_k == [ALL_FLOWS_ONLY_K]:
        axis.set_xticklabels(['all packets'])
        axis.set_xlim(ALL_FLOWS_ONLY_K - max(span / 2, 0.5), ALL_FLOWS_ONLY_K + max(span / 2, 0.5))
    elif all_k:
        pad = max(np.min(np.diff(all_k)) * 0.6, span / 2 + box_width) if len(all_k) > 1 else max(span / 2, 0.5)
        axis.set_xlim(min(all_k) - pad, max(all_k) + pad)
    axis.grid(True, alpha=0.35, axis='y')
    axis.legend(handles=legend_handles, fontsize=16, loc='best', ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_burstiness_vs_load_by_traffic(results_by_traffic_load, k, burstiness_field, output_path,
                                        title=None):
    """One burstiness metric of the all-packets arrival process vs load, one boxplot (across
    that combination's experiments) per traffic per load -- the burstiness counterpart of
    plot_emd_vs_load_by_traffic, but with the metric itself as the plotted quantity rather
    than EMD, so there is exactly one series per traffic (no series_specs / families to
    compare) and no pass/fail colouring (burstiness isn't part of the delay consistency
    check) -- each traffic's boxes share that traffic's colour (_TRAFFIC_COLORS) with a
    plain solid border. `k` behaves as in plot_emd_vs_load_by_traffic ('max' = each
    combination's own maximum flow count)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    use_max_k = (k == 'max')
    results_by_traffic_load = {key: upgrade_emd_vs_flows_results_schema(r)
                                for key, r in results_by_traffic_load.items()}
    traffics = sorted({t for (t, _l) in results_by_traffic_load})
    loads = sorted({l for (_t, l) in results_by_traffic_load})
    n_traffics = max(len(traffics), 1)
    offsets, box_width, span = _load_plot_layout(n_traffics, loads, n_series=1)

    fig, axis = plt.subplots(figsize=_load_plot_figsize(n_traffics, 1, len(loads)))
    legend_handles = []
    any_data = False
    for ti, traffic in enumerate(traffics):
        color = _TRAFFIC_COLORS[ti % len(_TRAFFIC_COLORS)]
        legend_handles.append(Patch(facecolor='white', edgecolor=color, linewidth=4.5, label=traffic))
        positions, data = [], []
        for load in loads:
            r = results_by_traffic_load.get((traffic, load))
            if r is None or not r['num_flows'] or (not use_max_k and k not in r['num_flows']):
                continue
            i = -1 if use_max_k else r['num_flows'].index(k)
            by_experiment = (r.get('burstiness_all_packets_by_experiment') or {}).get(burstiness_field)
            if by_experiment and i < len(by_experiment):
                values = [v for v in by_experiment[i] if v == v]
            else:
                scalar = (r.get('burstiness_all_packets') or {}).get(burstiness_field, [])
                values = [scalar[i]] if i < len(scalar) and scalar[i] == scalar[i] else []
            if not values:
                continue
            positions.append(load + offsets[ti])
            data.append(values)
        if data:
            any_data = True
            bp = axis.boxplot(data, positions=positions, widths=box_width, patch_artist=True,
                               showfliers=False, manage_ticks=False)
            for patch in bp['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.85)
                patch.set_edgecolor(color)
                patch.set_linewidth(4.5)
            for part in ('whiskers', 'caps'):
                for line in bp[part]:
                    line.set_color(color)
                    line.set_linewidth(4.5)
            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(2.5)

    if not any_data:
        print("plot_burstiness_vs_load_by_traffic: no {} data at k={}, writing empty plot".format(
            burstiness_field, k))

    y_label = BURSTINESS_METRIC_LABELS.get(burstiness_field, burstiness_field)
    default_title = '{} vs load by traffic, all considered flows (each combination\'s own max)'.format(y_label) \
        if use_max_k else '{} vs load by traffic, k={}'.format(y_label, k)
    axis.set_title(title or default_title, fontsize=34)
    axis.set_xlabel('Load')
    axis.set_ylabel(y_label)
    axis.set_xticks(loads)
    if loads:
        pad = max(np.min(np.diff(loads)) * 0.6, span / 2 + box_width) if len(loads) > 1 else max(span / 2, 0.05)
        axis.set_xlim(min(loads) - pad, max(loads) + pad)
    axis.grid(True, alpha=0.35, axis='y')
    axis.legend(handles=legend_handles, fontsize=16, loc='best')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_emd_vs_burstiness_by_traffic(results_by_traffic_load, k, burstiness_field, output_path,
                                       pass_threshold=0.9, title=None, series_specs=None,
                                       normalized=False, metric='emd'):
    """Like plot_emd_vs_load_by_traffic, but the x-axis is a burstiness metric of the
    all-packets arrival process at this k (results['burstiness_all_packets'][burstiness_field],
    see burstiness_metrics) instead of load. One point/box per (traffic, load) combination,
    positioned at ITS OWN measured burstiness rather than a shared nominal x-tick -- unlike
    load, burstiness isn't a controlled experimental parameter with a handful of common
    values, it's whatever that combination's traffic actually did, so two combinations can
    legitimately land at (near-)identical x if their sending pattern was equally bursty.

    `burstiness_field` is one of 'idc_1rtt' / 'avg_burst_duration_ns' /
    'avg_burst_interarrival_ns' (see BURSTINESS_METRIC_LABELS, which also supplies the axis
    label). `k`, `series_specs`, `normalized`, `metric` and the y-axis capping behave exactly
    as in plot_emd_vs_load_by_traffic; series are still colour-per-traffic,
    style-per-series (see _load_series_spec)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    use_max_k = (k == 'max')
    results_by_traffic_load = {key: upgrade_emd_vs_flows_results_schema(r)
                                for key, r in results_by_traffic_load.items()}
    if not series_specs:
        methods = []
        for r in results_by_traffic_load.values():
            for name in r['subsampling_methods']:
                if name not in methods:
                    methods.append(name)
        series_specs = all_packets_vs_sampled_load_plot_series(methods or ['find_samples_path'])

    traffics = sorted({t for (t, _l) in results_by_traffic_load})
    pass_color, fail_color = 'tab:green', 'tab:red'
    n_series = max(len(series_specs), 1)

    # Each (traffic, load) combination's x position is its OWN measured burstiness at this k
    # -- not a shared discrete value like load -- so the usual "gap between distinct loads"
    # spacing doesn't apply. Derive a data-driven slot width from the actual spread of
    # positions instead, so series at one combination stay visually distinct from each other
    # without assuming anything about the scale of this particular metric (IDC is O(1),
    # burst gaps are O(1e3) ns, etc).
    combo_x = {}
    low_poissonization_cells = {}
    for (traffic, load), r in results_by_traffic_load.items():
        if not r['num_flows'] or (not use_max_k and k not in r['num_flows']):
            continue
        i = -1 if use_max_k else r['num_flows'].index(k)
        # Same gate as the load-axis plots: a cell whose Poissonization succeeded in too
        # few runs is not given a position on this axis either (see MIN_POISSONIZED_RUNS).
        n_poisson = poissonized_run_count(r, i)
        if n_poisson < MIN_POISSONIZED_RUNS:
            low_poissonization_cells[(traffic, load)] = n_poisson
            continue
        values = (r.get('burstiness_all_packets') or {}).get(burstiness_field, [])
        if i < len(values) and np.isfinite(values[i]):
            combo_x[(traffic, load)] = values[i]
    if low_poissonization_cells:
        print("plot_emd_vs_burstiness_by_traffic: dropped {} traffic/load cell(s) with fewer than "
              "{} Poissonized runs: {}".format(
                  len(low_poissonization_cells), MIN_POISSONIZED_RUNS,
                  ', '.join('{} @ {} ({} run(s))'.format(t, l, n)
                             for (t, l), n in sorted(low_poissonization_cells.items()))))

    x_label = BURSTINESS_METRIC_LABELS.get(burstiness_field, burstiness_field)
    fig, axis = plt.subplots(figsize=(30, 15))
    if not combo_x:
        print("plot_emd_vs_burstiness_by_traffic: no {} data at k={}, writing empty plot".format(
            burstiness_field, k))
        axis.set_title(title or 'No {} data available'.format(x_label), fontsize=34)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    xs = sorted(combo_x.values())
    x_span = (xs[-1] - xs[0]) if len(xs) > 1 else max(abs(xs[0]), 1.0)
    slot = max(x_span * 0.03, abs(x_span) * 1e-6 + 1e-9)
    span = slot * n_series
    box_width = (span / n_series) * _FAMILY_BOX_FILL
    offsets = np.linspace(-span / 2, span / 2, n_series) if n_series > 1 else np.array([0.0])

    legend_handles = [
        Patch(facecolor=pass_color, edgecolor='black', alpha=0.85,
              label='Consistency check passed (≥{:.0f}%)'.format(pass_threshold * 100)),
        Patch(facecolor=fail_color, edgecolor='black', alpha=0.85,
              label='Consistency check failed (<{:.0f}%)'.format(pass_threshold * 100)),
    ]

    any_data = False
    all_plotted_values = []
    for ti, traffic in enumerate(traffics):
        color = _TRAFFIC_COLORS[ti % len(_TRAFFIC_COLORS)]
        combo_loads = sorted(load for (t, load) in combo_x if t == traffic)
        if not combo_loads:
            continue
        legend_handles.append(Patch(facecolor='white', edgecolor=color, linewidth=4.5, label=traffic))
        for load in combo_loads:
            r = results_by_traffic_load[(traffic, load)]
            i = -1 if use_max_k else r['num_flows'].index(k)
            x0 = combo_x[(traffic, load)]
            for si, series_spec in enumerate(series_specs):
                style = dict(edge_color=color, edge_style=series_spec['edge_style'])
                values, pass_rate = _load_plot_series_values(r, i, series_spec['key'],
                                                             normalized=normalized, metric=metric)
                if (_per_run_series(series_spec['key']) and 0 < len(values) < MIN_POISSONIZED_RUNS):
                    values = []
                if len(values):
                    any_data = True
                    all_plotted_values.extend(values)
                _draw_boxplot_family(axis, [x0], [values], [pass_rate], offsets[si], box_width,
                                     pass_threshold, pass_color, fail_color, style,
                                     edge_width=series_spec.get('edge_width', 4.5))

    for series_spec in series_specs:
        legend_handles.append(Line2D([0], [0], color='black', linestyle=series_spec['edge_style'],
                                      linewidth=series_spec.get('edge_width', 3),
                                      label='{} (border)'.format(series_spec['label'])))

    if not any_data:
        print("plot_emd_vs_burstiness_by_traffic: no data at k={}, writing empty plot".format(k))

    series_names = ' vs. '.join(s['label'] for s in series_specs)
    y_label = _metric_result_keys(metric, normalized)[-1]
    quantity_percentile = _metric_percentile(metric)
    if quantity_percentile is not None:
        quantity_name = 'p{} error{}'.format(quantity_percentile,
                                              ' (relative)' if metric[0].endswith('reldiff') else ' (ns)')
    else:
        quantity_name = 'EMD relative to mean queuing delay' if normalized else 'EMD'
    default_title = '{} vs {} by traffic ({}), all considered flows (each combination\'s own max)'.format(
        quantity_name, x_label, series_names) if use_max_k \
        else '{} vs {} by traffic ({}), k={}'.format(quantity_name, x_label, series_names, k)
    axis.set_title(title or default_title, fontsize=34)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    if quantity_percentile is not None:
        axis.axhline(0, color='black', linewidth=2, linestyle=':', zorder=1)

    # Same fixed-scale y-axis capping as plot_emd_vs_load_by_traffic (see there for the
    # rationale) -- kept silent (no "capped at..." annotation) to match that plot's current
    # convention.
    is_relative = _metric_is_relative(metric, normalized)
    signed = isinstance(metric, tuple)
    cap = 1.0 if is_relative else 500.0
    bottom = -cap if signed else 0.0
    if _metric_autoscales(metric):
        axis.autoscale(axis='y')   # see _metric_autoscales
    else:
        axis.set_ylim(bottom=bottom, top=cap)

    axis.grid(True, alpha=0.35, axis='y')
    axis.legend(handles=legend_handles, fontsize=16, loc='best', ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# Below this many runs in which the sampler actually produced a Poissonized subsample
# (pooled across every experiment of a traffic/load combination -- see
# aggregate_emd_vs_flows_results), that combination is left out of the cross-traffic
# comparison plots entirely rather than drawn from a handful of runs.
#
# This gates EVERY family's box in the cell, not just the Poisson-adaptive one, because in
# a growing-window run nothing in the cell is independent of it: the window each run
# settles on is the analysis window for that run's ground truth, its all-packets family,
# its uniform baseline and its ideal probes alike, so a run that certified no window
# contributes to none of them. Measured case that prompted this: WOIncast,
# find_samples_path_intensity_growing_window, Facebook_HadoopDist_All at load 0.95 -- 4
# certified runs out of 30 experiments x 25 runs, with all-packets, sampled, uniform and
# oracle boxes all resting on those same 4.
MIN_POISSONIZED_RUNS = 100
# The pass-rate plots' own threshold is the same quantity under a different name (a
# consistency-check attempt only happens in a run that found a subsample), so they share
# one number rather than drifting apart.
_MIN_PASS_RATE_CHECKS = MIN_POISSONIZED_RUNS


def poissonized_run_count(r, i):
    """How many runs of this results dict produced a Poissonized subsample at flow-count
    index `i` -- the number of runs every comparison drawn from it ultimately rests on.

    Taken over the result's own subsampling method(s) (their per-run EMD series, one entry
    per run that found a subsample). A growing-window result carries exactly one method, so
    this is that method's certified-run count; a result comparing several methods within
    one run reports the best-supported one, since they share the runs."""
    counts = [0]
    by_method = r.get('emd_sampled_packets_by_run') or {}
    for method in (r.get('subsampling_methods') or []):
        series = by_method.get(method) or []
        if not series:
            continue
        try:
            counts.append(len(series[i]))
        except IndexError:
            pass
    return max(counts)


def plot_pass_rate_vs_load_by_traffic(results_by_traffic_load, k, output_path, series_key='sampled',
                                       pass_threshold=0.9, title=None, metric='emd'):
    """Cross-traffic comparison at one fixed flow count `k` of the delay consistency check's
    success rate itself (not the EMD distribution): x-axis is load, y-axis is the pass rate
    (0-100%), one line + markers per traffic, colored per _TRAFFIC_COLORS. Markers are
    colored green/red by whether they clear `pass_threshold` (dashed reference line), the
    same convention used elsewhere; the connecting line carries the traffic's color so
    multiple traffics stay distinguishable on one axis.

    `series_key` defaults to 'sampled' -- the pass rate of the results' first
    Poisson-adaptive subsampling method, which is what this was built for -- but accepts
    any key plot_emd_vs_load_by_traffic's series specs do: ('sampled', method) to pick a
    specific method when several were run in one go, 'all_packets', or
    ('uniform', stride) if a similar success-rate view is ever wanted for those.

    `metric` selects WHICH check's pass rate is drawn: the delay one by default, or a
    probability metric's via prob_plot_metric(name) -- the loss/marking checks have their
    own verdicts over the same families (see PROB_METRICS), and their pass rate is out of
    the runs that were testable at all.

    `k` is normally an int looked up exactly in each combination's num_flows; pass 'max' to
    use each combination's own maximum flow count instead (see plot_emd_vs_load_by_traffic).
    A (traffic, load) combination missing entirely, or with no data at this k, is simply
    left without a point there (the line breaks across the gap) -- and so is one where the
    consistency check was actually performed fewer than _MIN_PASS_RATE_CHECKS times (see
    there): the pass rate there is a ratio of a few actual attempts (e.g. 2 of 50 runs
    finding a valid subsample at all), not a meaningful success rate, and plotting it as one
    would be misleading."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    use_max_k = (k == 'max')
    results_by_traffic_load = {key: upgrade_emd_vs_flows_results_schema(r)
                                for key, r in results_by_traffic_load.items()}

    traffics = sorted({t for (t, _l) in results_by_traffic_load})
    loads = sorted({l for (_t, l) in results_by_traffic_load})
    pass_color, fail_color = 'tab:green', 'tab:red'

    fig, axis = plt.subplots(figsize=(30, 15))
    axis.axhline(pass_threshold, color='black', linewidth=2, linestyle=':', zorder=1)

    legend_handles = [
        Line2D([0], [0], marker='o', color='0.4', markerfacecolor=pass_color, markeredgecolor='black',
               markersize=16, linewidth=0, label='≥{:.0f}% pass threshold'.format(pass_threshold * 100)),
        Line2D([0], [0], marker='o', color='0.4', markerfacecolor=fail_color, markeredgecolor='black',
               markersize=16, linewidth=0, label='<{:.0f}% pass threshold'.format(pass_threshold * 100)),
    ]

    any_data = False
    for ti, traffic in enumerate(traffics):
        color = _TRAFFIC_COLORS[ti % len(_TRAFFIC_COLORS)]
        x_vals, y_vals = [], []
        for load in loads:
            r = results_by_traffic_load.get((traffic, load))
            if r is None or not r['num_flows'] or (not use_max_k and k not in r['num_flows']):
                continue
            i = -1 if use_max_k else r['num_flows'].index(k)
            values, pass_rate = _load_plot_series_values(r, i, series_key, metric=metric)
            if len(values) < _MIN_PASS_RATE_CHECKS:
                continue
            x_vals.append(load)
            y_vals.append(pass_rate)

        if not x_vals:
            continue
        any_data = True
        axis.plot(x_vals, y_vals, color=color, linewidth=2.5, zorder=2)
        y_arr = np.asarray(y_vals)
        pass_mask = y_arr >= pass_threshold
        if pass_mask.any():
            axis.scatter(np.asarray(x_vals)[pass_mask], y_arr[pass_mask], marker='o', color=pass_color,
                         edgecolor=color, linewidth=2, s=260, zorder=3)
        if (~pass_mask).any():
            axis.scatter(np.asarray(x_vals)[~pass_mask], y_arr[~pass_mask], marker='o', color=fail_color,
                         edgecolor=color, linewidth=2, s=260, zorder=3)
        legend_handles.append(Line2D([0], [0], color=color, linewidth=4, marker='o', markersize=0, label=traffic))

    if not any_data:
        print("plot_pass_rate_vs_load_by_traffic: no data at k={}, writing empty plot".format(k))

    series_label = _series_key_label(series_key)
    default_title = 'Consistency check pass rate vs load by traffic ({}), all considered flows (each combination\'s own max)'.format(series_label) if use_max_k \
        else 'Consistency check pass rate vs load by traffic ({}), k={}'.format(series_label, k)
    axis.set_title(title or default_title, fontsize=34)
    axis.set_xlabel('Load')
    axis.set_ylabel('Consistency check pass rate')
    axis.set_xticks(loads)
    axis.set_ylim(-0.02, 1.02)
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    if loads:
        span_x = (max(loads) - min(loads)) if len(loads) > 1 else 1.0
        pad = max(span_x * 0.05, 0.03)
        axis.set_xlim(min(loads) - pad, max(loads) + pad)
    axis.grid(True, alpha=0.35)
    axis.legend(handles=legend_handles, fontsize=18, loc='best')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_emd_vs_num_flows_boxplot(results, output_path, title="EMD vs number of TCP flows", pass_threshold=0.9,
                                   y_max=None, normalized=False, metric='emd'):
    """Plot results from compute_emd_vs_num_tcp_flows_multi_run: at each
    number of considered TCP flows, the all-packet CDF is the same fixed set
    of packets on every run, so its EMD-to-ground-truth is a single value --
    plotted as a connected line of dots. Every subsampling method (one per
    Poisson-adaptive method in results['subsampling_methods'], plus one
    uniform family per entry in results['uniform_series'] -- for current
    results, each method's rate-matched uniform counterpart) differs every
    run, so its EMD is
    plotted as a boxplot of the distribution across runs, each with a
    distinct, thick outline (colour per family, dash per family *kind* -- solid for all
    packets, dashed for Poisson-adaptive subsamples, dash-dot for ideal Poisson probes,
    dotted for uniform baselines, see family_border_style) so
    the methods stay visually distinguishable -- the fill itself is only ever
    the plain pass/fail color, never a pattern. All
    are colored green when at least `pass_threshold` (e.g. 90%) of the runs'
    delay consistency check passed at that flow count, and red otherwise; a
    flow count for which no run produced a valid subsample of a given method
    is left without a box for that method. When `results` was produced by
    aggregate_emd_vs_flows_results over more than one experiment (i.e.
    results['num_experiments'] > 1 and results['emd_all_packets_by_experiment']
    is present), the all-packets series also varies now (one value per
    experiment) and is drawn as a boxplot family too instead of dots + line.

    With `normalized` set, the plotted quantity is the EMD divided by the mean
    ground-truth path delay (normalize_emd_values) rather than raw nanoseconds
    -- the view that stays comparable across offered loads.

    Each all-packets position is also annotated with its burstiness (IDC at one RTT, mean
    burst duration/inter-burst gap -- see burstiness_metrics), just above the
    point/box (_annotate_all_packets_burstiness), when results['burstiness_all_packets']
    has it (absent for pre-burstiness-metrics results -- see backfill_burstiness_metrics).

    `metric` selects which quantity to plot: 'emd' (the default, real Wasserstein-1,
    supports `normalized`) or 'percentile_avg_relerror' (percentile_avg_relative_error,
    mean absolute relative percentile error over a dense percentile grid -- already
    self-normalized, so `normalized` does not apply -- see _metric_result_keys). Unlike
    the raw ('percentile_diff', q)/('percentile_reldiff', q) plots
    (plot_percentile_diff_vs_num_flows), both metrics here keep the green/red
    consistency-check coloring, since both measure the same kind of whole-distribution
    agreement the real EMD does."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = upgrade_emd_vs_flows_results_schema(results)
    all_key, all_by_exp_key, sampled_key, uniform_key, oracle_key, metric_label = _metric_result_keys(
        metric, normalized)
    num_flows = results['num_flows']
    methods = results['subsampling_methods']
    emd_all = np.asarray(results[all_key], dtype=float)
    pass_rate_all = np.asarray(results['pass_rate_all_packets'], dtype=float)
    uniform_series = results.get('uniform_series', [])
    oracle_series = results.get('oracle_series', [])
    emd_all_by_experiment = results.get(all_by_exp_key)
    all_packets_is_boxplot = bool(emd_all_by_experiment) and results.get('num_experiments', 1) > 1

    pass_color, fail_color = 'tab:green', 'tab:red'
    offset_all, offsets_poisson, offsets_uniform, offsets_oracle, box_width = _subsample_family_layout(
        methods, uniform_series, oracle_series)

    fig, axis = plt.subplots(figsize=(30, 15))

    legend_handles = _draw_all_packets_series(
        axis, num_flows, emd_all, emd_all_by_experiment, pass_rate_all, offset_all, box_width,
        pass_threshold, pass_color, fail_color, results['num_runs'],
        results.get('num_experiments', 1), metric_label,
        burstiness_by_k=results.get('burstiness_all_packets'),
        spread_label=_all_packets_spread_label(results))

    # Poisson-adaptive subsamples: each differs every run -- one boxplot family per
    # method, so several algorithms run together are compared on the same axis.
    emd_sampled_by_run = results[sampled_key]
    for i, method in enumerate(methods):
        style = family_border_style('sampled', i)
        values_by_k = emd_sampled_by_run[method]
        _draw_boxplot_family(axis, num_flows, values_by_k, results['pass_rate_sampled'][method],
                             offsets_poisson[method], box_width, pass_threshold, pass_color, fail_color, style)
        missing_sampled_k = [k for k, values in zip(num_flows, values_by_k) if len(values) == 0]
        if missing_sampled_k:
            print("No {} {} values for {} flow-count(s), skipped: {}".format(
                method, metric_label, len(missing_sampled_k), missing_sampled_k))
        legend_handles.append(Patch(facecolor='white', edgecolor=style['edge_color'], linewidth=4.5,
                                     linestyle=style['edge_style'],
                                     label='Poisson-adaptive subsample, {} (boxplot)'.format(method)))

    # Uniform families: same idea, one boxplot family each -- for current results one
    # per Poisson-adaptive method, drawing that method's own sample count.
    emd_uniform_by_run = results.get(uniform_key, {})
    pass_rate_uniform = results.get('pass_rate_uniform', {})
    for i, key in enumerate(uniform_series):
        style = family_border_style('uniform', len(methods) + i)
        values_by_k = emd_uniform_by_run[key]
        _draw_boxplot_family(axis, num_flows, values_by_k, pass_rate_uniform[key],
                             offsets_uniform[key], box_width, pass_threshold, pass_color, fail_color, style)
        missing_k = [k for k, values in zip(num_flows, values_by_k) if len(values) == 0]
        if missing_k:
            print("No {} {} values for {} flow-count(s), skipped: {}".format(
                _uniform_series_label(key), metric_label, len(missing_k), missing_k))
        legend_handles.append(Patch(facecolor='white', edgecolor=style['edge_color'], linewidth=4.5,
                                     linestyle=style['edge_style'],
                                     label='{} (boxplot)'.format(_uniform_series_label(key))))

    # Ideal Poisson probes: the ceiling, drawn last so it reads as the reference the real
    # families are being judged against.
    emd_oracle_by_run = results.get(oracle_key, {})
    pass_rate_oracle = results.get('pass_rate_oracle', {})
    for i, key in enumerate(oracle_series):
        style = family_border_style('oracle', len(methods) + len(uniform_series) + i)
        values_by_k = emd_oracle_by_run[key]
        _draw_boxplot_family(axis, num_flows, values_by_k, pass_rate_oracle[key],
                             offsets_oracle[key], box_width, pass_threshold, pass_color, fail_color, style)
        missing_k = [k for k, values in zip(num_flows, values_by_k) if len(values) == 0]
        if missing_k:
            print("No {} {} values for {} flow-count(s), skipped: {}".format(
                _oracle_series_label(key), metric_label, len(missing_k), missing_k))
        legend_handles.append(Patch(facecolor='white', edgecolor=style['edge_color'], linewidth=4.5,
                                     linestyle=style['edge_style'],
                                     label='{} (boxplot)'.format(_oracle_series_label(key))))

    axis.set_title(title, fontsize=34)
    axis.set_xlabel('Number of TCP flows considered')
    axis.set_ylabel(metric_label)
    _set_flow_count_xaxis(axis, num_flows)
    axis.grid(True, alpha=0.35, axis='y')
    if y_max is not None:
        all_packets_values = ([np.asarray(v, dtype=float) for v in emd_all_by_experiment] if all_packets_is_boxplot
                               else [emd_all[np.isfinite(emd_all)]])
        all_values = np.concatenate(all_packets_values
                                     + [np.asarray(v, dtype=float) for m in methods for v in emd_sampled_by_run[m]]
                                     + [np.asarray(v, dtype=float) for s in uniform_series for v in emd_uniform_by_run[s]]
                                     + [np.asarray(v, dtype=float) for s in oracle_series for v in emd_oracle_by_run[s]])
        # if all_values.size and np.nanmax(all_values) > y_max:
        #     axis.text(0.995, 0.01, 'y-axis capped at {:g}; some boxes/whiskers extend beyond\n'
        #                             '(see results text file for full range)'.format(y_max),
        #                transform=axis.transAxes, ha='right', va='top', fontsize=14, style='italic',
        #                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axis.set_ylim(bottom=0, top=y_max)
    axis.legend(handles=legend_handles, fontsize=18, loc='best')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


_ALL_PACKETS_STYLE = dict(edge_color='black', edge_style='solid')


def plot_burstiness_vs_num_flows(results, burstiness_field, output_path, title=None):
    """One burstiness metric of the all-packets arrival process (IDC(1RTT),
    avg_burst_duration_ns, or avg_burst_interarrival_ns -- see burstiness_metrics /
    BURSTINESS_METRIC_LABELS) against the number of considered TCP flows, for a single
    (traffic, load) combination -- the burstiness counterpart of
    plot_emd_vs_num_flows_boxplot's all-packets series (dots + line for a single experiment,
    a boxplot across experiments once aggregated over more than one via
    aggregate_emd_vs_flows_results). Not part of the delay consistency check, so drawn in a
    single plain colour rather than pass/fail green/red."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = upgrade_emd_vs_flows_results_schema(results)
    num_flows = results['num_flows']
    by_experiment = (results.get('burstiness_all_packets_by_experiment') or {}).get(burstiness_field)
    scalar = (results.get('burstiness_all_packets') or {}).get(burstiness_field, [])
    is_boxplot = bool(by_experiment) and results.get('num_experiments', 1) > 1
    color = 'teal'

    fig, axis = plt.subplots(figsize=(30, 15))
    missing = []
    if is_boxplot:
        positions, data = [], []
        for k, values in zip(num_flows, by_experiment):
            finite = [v for v in values if v == v]
            if not finite:
                missing.append(k)
                continue
            positions.append(k)
            data.append(finite)
        if data:
            bp = axis.boxplot(data, positions=positions, widths=0.5, patch_artist=True,
                               showfliers=False, manage_ticks=False)
            for patch in bp['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.85)
                patch.set_edgecolor('black')
                patch.set_linewidth(2.5)
            for part in ('whiskers', 'caps'):
                for line in bp[part]:
                    line.set_color('black')
                    line.set_linewidth(2.5)
            for median in bp['medians']:
                median.set_color(color)
                median.set_linewidth(2.5)
    else:
        values = np.asarray(scalar, dtype=float)
        x = np.asarray(num_flows, dtype=float)
        valid = np.isfinite(values)
        missing = [k for k, v in zip(num_flows, values) if not np.isfinite(v)]
        axis.plot(x[valid], values[valid], color='0.4', linewidth=2, zorder=1)
        axis.scatter(x[valid], values[valid], marker='o', color=color,
                     edgecolor='black', s=220, zorder=3)
    if missing:
        print("No {} value for {} flow-count(s), skipped: {}".format(
            burstiness_field, len(missing), missing))

    y_label = BURSTINESS_METRIC_LABELS.get(burstiness_field, burstiness_field)
    axis.set_title(title or '{} vs number of TCP flows'.format(y_label), fontsize=34)
    axis.set_xlabel('Number of TCP flows considered')
    axis.set_ylabel(y_label)
    _set_flow_count_xaxis(axis, num_flows)
    axis.grid(True, alpha=0.35, axis='y')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_mean_diff_vs_num_flows(results, output_path, title="Switch vs. packet mean delay difference", pass_threshold=0.9, y_limit=None):
    """Plot results from compute_emd_vs_num_tcp_flows_multi_run: at each
    number of considered TCP flows, boxplots (across runs) of the signed
    difference between the switch samples' mean delay and the packet-side
    mean delay -- the quantity abs()-thresholded by the delay consistency
    check -- for the all-packet CDF (thick solid black outline), each
    Poisson-adaptive subsampling method in results['subsampling_methods'],
    and one uniform family per entry in results['uniform_series'] -- for
    current results each method's rate-matched uniform counterpart -- each
    with its own thick outline style
    (colour per family, dash per family kind -- see family_border_style). All
    quantities vary run to run here (the
    switch-side mean is re-drawn every run, and every subsampling method is
    redrawn every run too), so all are boxplots; the fill is only ever the
    plain pass/fail color (green/red), never a pattern, so the box outline
    style (color, dash pattern) is what tells families apart. Colored green when at
    least `pass_threshold` of the runs' consistency check passed at that
    flow count, red otherwise; a horizontal line at zero marks perfect
    agreement. A flow count for which no run produced a valid subsample of a
    given method is left without a box for that method."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = upgrade_emd_vs_flows_results_schema(results)
    num_flows = results['num_flows']
    methods = results['subsampling_methods']
    uniform_series = results.get('uniform_series', [])
    oracle_series = results.get('oracle_series', [])
    pass_color, fail_color = 'tab:green', 'tab:red'
    offset_all, offsets_poisson, offsets_uniform, offsets_oracle, box_width = _subsample_family_layout(
        methods, uniform_series, oracle_series)

    fig, axis = plt.subplots(figsize=(30, 15))
    axis.axhline(0, color='black', linewidth=2, linestyle=':', zorder=1)

    diff_all_by_run = results['mean_diff_all_packets_by_run']
    missing_all_k = [k for k, values in zip(num_flows, diff_all_by_run) if len(values) == 0]
    if missing_all_k:
        print("No all-packet mean-diff values for {} flow-count(s), skipped: {}".format(len(missing_all_k), missing_all_k))
    _draw_boxplot_family(axis, num_flows, diff_all_by_run, results['pass_rate_all_packets'],
                         offset_all, box_width, pass_threshold, pass_color, fail_color,
                         _ALL_PACKETS_STYLE)

    legend_handles = [
        Patch(facecolor=pass_color, edgecolor='black', alpha=0.85,
              label='Consistency check passed (>={:.0f}% of {} runs)'.format(pass_threshold * 100, results['num_runs'])),
        Patch(facecolor=fail_color, edgecolor='black', alpha=0.85,
              label='Consistency check failed (<{:.0f}% of {} runs)'.format(pass_threshold * 100, results['num_runs'])),
        Patch(facecolor='white', edgecolor=_ALL_PACKETS_STYLE['edge_color'], linewidth=4.5,
              label='All packets of considered flows (solid black edge)'),
    ]

    diff_sampled_by_run = results['mean_diff_sampled_by_run']
    for i, method in enumerate(methods):
        style = family_border_style('sampled', i)
        values_by_k = diff_sampled_by_run[method]
        missing_sampled_k = [k for k, values in zip(num_flows, values_by_k) if len(values) == 0]
        if missing_sampled_k:
            print("No {} mean-diff values for {} flow-count(s), skipped: {}".format(
                method, len(missing_sampled_k), missing_sampled_k))
        _draw_boxplot_family(axis, num_flows, values_by_k, results['pass_rate_sampled'][method],
                             offsets_poisson[method], box_width, pass_threshold, pass_color, fail_color, style)
        legend_handles.append(Patch(facecolor='white', edgecolor=style['edge_color'], linewidth=4.5,
                                     linestyle=style['edge_style'],
                                     label='Poisson-adaptive subsample, {}'.format(method)))

    diff_uniform_by_run = results.get('mean_diff_uniform_packets_by_run', {})
    pass_rate_uniform = results.get('pass_rate_uniform', {})
    for i, key in enumerate(uniform_series):
        style = family_border_style('uniform', len(methods) + i)
        values_by_k = diff_uniform_by_run[key]
        missing_k = [k for k, values in zip(num_flows, values_by_k) if len(values) == 0]
        if missing_k:
            print("No {} mean-diff values for {} flow-count(s), skipped: {}".format(
                _uniform_series_label(key), len(missing_k), missing_k))
        _draw_boxplot_family(axis, num_flows, values_by_k, pass_rate_uniform[key],
                             offsets_uniform[key], box_width, pass_threshold, pass_color, fail_color, style)
        legend_handles.append(Patch(facecolor='white', edgecolor=style['edge_color'], linewidth=4.5,
                                     linestyle=style['edge_style'],
                                     label=_uniform_series_label(key)))

    diff_oracle_by_run = results.get('mean_diff_oracle_by_run', {})
    pass_rate_oracle = results.get('pass_rate_oracle', {})
    for i, key in enumerate(oracle_series):
        style = family_border_style('oracle', len(methods) + len(uniform_series) + i)
        values_by_k = diff_oracle_by_run[key]
        missing_k = [k for k, values in zip(num_flows, values_by_k) if len(values) == 0]
        if missing_k:
            print("No {} mean-diff values for {} flow-count(s), skipped: {}".format(
                _oracle_series_label(key), len(missing_k), missing_k))
        _draw_boxplot_family(axis, num_flows, values_by_k, pass_rate_oracle[key],
                             offsets_oracle[key], box_width, pass_threshold, pass_color, fail_color, style)
        legend_handles.append(Patch(facecolor='white', edgecolor=style['edge_color'], linewidth=4.5,
                                     linestyle=style['edge_style'],
                                     label=_oracle_series_label(key)))

    axis.set_title(title, fontsize=34)
    axis.set_xlabel('Number of TCP flows considered')
    axis.set_ylabel("Switch samples mean delay - packet mean delay (ns)")
    _set_flow_count_xaxis(axis, num_flows)
    axis.grid(True, alpha=0.35, axis='y')
    if y_limit is not None:
        all_values = np.concatenate([np.asarray(v, dtype=float) for v in diff_all_by_run]
                                     + [np.asarray(v, dtype=float) for m in methods for v in diff_sampled_by_run[m]]
                                     + [np.asarray(v, dtype=float) for s in uniform_series for v in diff_uniform_by_run[s]]
                                     + [np.asarray(v, dtype=float) for s in oracle_series for v in diff_oracle_by_run[s]])
        # if all_values.size and np.nanmax(np.abs(all_values)) > y_limit:
        #     axis.text(0.995, 0.01, 'y-axis capped at +/-{:.0f} ns; some boxes/whiskers extend beyond\n'
        #                             '(see results text file for full range)'.format(y_limit),
        #                transform=axis.transAxes, ha='right', va='bottom', fontsize=14, style='italic',
        #                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axis.set_ylim(-y_limit, y_limit)
    axis.legend(handles=legend_handles, fontsize=18, loc='best')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path



def plot_sample_sizes_vs_num_flows(results, output_path, title="Sample size vs number of TCP flows",
                                    log_y=True):
    """Plot how many packets each comparison family actually had at every flow count:
    boxplots (across runs) of the retained sample size for each Poisson-adaptive
    subsampling method, each rate-matched uniform family and each ideal Poisson probe,
    next to the un-subsampled all-packets count (dots on a line for one experiment, a
    boxplot once aggregated across experiments -- each experiment receives its own number
    of packets).

    This is the visual form of the text summary's 'n_pkts' columns, and it is worth a plot
    of its own for two reasons. First, it is the direct check that the comparison is
    rate-matched: a method's box and its paired uniform family's box must coincide exactly
    (sample_uniform_count spends precisely the count the method retained), so any visible
    gap between the two means the pairing broke. Second, against the all-packets reference
    it shows the subsampling ratio the whole approach buys -- three orders of magnitude on
    these runs -- which is why `log_y` is on by default.

    Unlike the EMD and mean-difference plots, the boxes here are filled one neutral colour
    rather than pass/fail green/red: a sample size is an input the consistency check
    consumes, not something the check renders a verdict on, and colouring it by the verdict
    would read as "this sample size passed" (see _draw_boxplot_family's fill_color). The
    families are told apart by border colour and dash exactly as everywhere else
    (family_border_style)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = upgrade_emd_vs_flows_results_schema(results)
    num_flows = results['num_flows']
    methods = results['subsampling_methods']
    uniform_series = results.get('uniform_series', [])
    oracle_series = results.get('oracle_series', [])
    num_experiments = results.get('num_experiments', 1)
    fill_color = '0.85'
    offset_all, offsets_poisson, offsets_uniform, offsets_oracle, box_width = _subsample_family_layout(
        methods, uniform_series, oracle_series)

    fig, axis = plt.subplots(figsize=(30, 15))

    legend_handles = _draw_all_packets_series(
        axis, num_flows, results.get('all_packet_sizes') or [np.nan] * len(num_flows),
        results.get('all_packet_sizes_by_experiment'), results['pass_rate_all_packets'],
        offset_all, box_width, 1.0, fill_color, fill_color, results['num_runs'],
        num_experiments, 'sample size', fill_color=fill_color,
        spread_label=_all_packets_spread_label(results))

    for kind, series, offsets, values_by_key, labeller in (
            ('sampled', methods, offsets_poisson, results.get('sample_sizes_sampled_by_run', {}),
             lambda key: 'Poisson-adaptive subsample, {}'.format(key)),
            ('uniform', uniform_series, offsets_uniform, results.get('sample_sizes_uniform_by_run', {}),
             _uniform_series_label),
            ('oracle', oracle_series, offsets_oracle, results.get('sample_sizes_oracle_by_run', {}),
             _oracle_series_label)):
        for i, key in enumerate(series):
            color_index = {'sampled': 0, 'uniform': len(methods),
                            'oracle': len(methods) + len(uniform_series)}[kind] + i
            style = family_border_style(kind, color_index)
            values_by_k = values_by_key.get(key, [[]] * len(num_flows))
            missing_k = [k for k, values in zip(num_flows, values_by_k) if len(values) == 0]
            if missing_k:
                print("No {} sample-size values for {} flow-count(s), skipped: {}".format(
                    labeller(key), len(missing_k), missing_k))
            _draw_boxplot_family(axis, num_flows, values_by_k, [1.0] * len(num_flows),
                                 offsets[key], box_width, 1.0, fill_color, fill_color, style,
                                 fill_color=fill_color)
            legend_handles.append(Patch(facecolor=fill_color, edgecolor=style['edge_color'],
                                         linewidth=4.5, linestyle=style['edge_style'],
                                         label=labeller(key)))

    axis.set_title(title, fontsize=34)
    axis.set_xlabel('Number of TCP flows considered')
    axis.set_ylabel('Packets in the family (sample size)')
    if log_y:
        axis.set_yscale('log')
    _set_flow_count_xaxis(axis, num_flows)
    axis.grid(True, alpha=0.35, axis='y', which='both')
    axis.legend(handles=legend_handles, fontsize=18, loc='best')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _prob_family_layout(results):
    """(families, box_width) for a probability plot: the same comparison families the delay
    plots draw, each with its offset around the flow-count tick and its border style, as
    (label, family_key, rate_key, key, offset, style) tuples."""
    methods = results['subsampling_methods']
    uniform_series = results.get('uniform_series', [])
    oracle_series = results.get('oracle_series', [])
    offset_all, offsets_poisson, offsets_uniform, offsets_oracle, box_width = _subsample_family_layout(
        methods, uniform_series, oracle_series)
    families = [('All packets of considered flows', 'all_packets_by_run',
                  'pass_rate_all_packets', None, offset_all, _ALL_PACKETS_STYLE)]
    for i, m in enumerate(methods):
        families.append(('Poisson-adaptive subsample, {}'.format(m), 'sampled_by_run',
                          'pass_rate_sampled', m, offsets_poisson[m],
                          family_border_style('sampled', i)))
    for i, s in enumerate(uniform_series):
        families.append((_uniform_series_label(s), 'uniform_by_run', 'pass_rate_uniform', s,
                          offsets_uniform[s], family_border_style('uniform', len(methods) + i)))
    for i, o in enumerate(oracle_series):
        families.append((_oracle_series_label(o), 'oracle_by_run', 'pass_rate_oracle', o,
                          offsets_oracle[o],
                          family_border_style('oracle', len(methods) + len(uniform_series) + i)))
    return families, box_width


def plot_delay_value_vs_num_flows(results, output_path,
                                   title="Mean queuing delay vs number of TCP flows",
                                   pass_threshold=0.9):
    """Plot the mean queuing delay each comparison family actually estimates, per flow
    count, with a dashed reference line at the ground truth's own mean -- the delay
    counterpart of plot_prob_metric_vs_num_flows' `quantity='prob'` view.

    Why this sits next to the EMD plots rather than replacing them: the EMD measures how far
    a family's whole delay distribution is from the truth, while this is the single number
    the consistency check actually thresholds (`mean_diff` is this minus the switch side's
    own mean). Seeing the absolute values makes it obvious when two families with similar
    EMDs disagree about the delay level itself, and when a family's error is a shift rather
    than a shape difference.

    Boxes are coloured by the consistency check, like the EMD and mean-difference plots,
    since the check is a test on exactly this quantity. Returns None without writing
    anything for results that carry no per-family means (they predate the recording)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = upgrade_emd_vs_flows_results_schema(results)
    num_flows = results['num_flows']
    methods = results['subsampling_methods']
    uniform_series = results.get('uniform_series', [])
    oracle_series = results.get('oracle_series', [])
    n_k = len(num_flows)
    all_by_run = results.get('delay_mean_all_packets_by_run') or [[] for _ in range(n_k)]
    families = [('sampled', methods, results.get('delay_mean_sampled_by_run') or {},
                  lambda key: 'Poisson-adaptive subsample, {}'.format(key)),
                 ('uniform', uniform_series, results.get('delay_mean_uniform_by_run') or {},
                  _uniform_series_label),
                 ('oracle', oracle_series, results.get('delay_mean_oracle_by_run') or {},
                  _oracle_series_label)]
    if not any(len(v) for v in all_by_run) and not any(
            any(len(v) for v in values.get(key, [])) for _, series, values, _ in families
            for key in series):
        print("plot_delay_value_vs_num_flows: no per-family mean delays recorded in these "
              "results (they predate it), skipping {}".format(output_path))
        return None

    pass_color, fail_color = 'tab:green', 'tab:red'
    offset_all, offsets_poisson, offsets_uniform, offsets_oracle, box_width = _subsample_family_layout(
        methods, uniform_series, oracle_series)
    offsets_by_kind = {'sampled': offsets_poisson, 'uniform': offsets_uniform,
                        'oracle': offsets_oracle}
    fig, axis = plt.subplots(figsize=(30, 15))

    legend_handles = []
    reference = [float(np.mean(v)) if len(v) else np.nan
                  for v in (results.get('groundtruth_delay_mean_by_run') or [[]] * n_k)]
    x = np.asarray(num_flows, dtype=float)
    valid = np.isfinite(reference)
    if np.any(valid):
        # Markers as well as the line, so an all-flows-only result's single point shows.
        axis.plot(x[valid], np.asarray(reference)[valid], color='black', linewidth=3,
                   linestyle='--', marker='D', markersize=14, zorder=1)
        legend_handles.append(Line2D([0], [0], color='black', linewidth=3, linestyle='--',
                                      marker='D', markersize=14,
                                      label='Mean of the reconstructed ground-truth delay (reference)'))
    legend_handles += [
        Patch(facecolor=pass_color, edgecolor='black', alpha=0.85,
              label='Consistency check passed (>={:.0f}% of the runs)'.format(pass_threshold * 100)),
        Patch(facecolor=fail_color, edgecolor='black', alpha=0.85,
              label='Consistency check failed (<{:.0f}% of the runs)'.format(pass_threshold * 100)),
    ]

    _draw_boxplot_family(axis, num_flows, all_by_run, results['pass_rate_all_packets'],
                         offset_all, box_width, pass_threshold, pass_color, fail_color,
                         _ALL_PACKETS_STYLE)
    legend_handles.append(Patch(facecolor='white', edgecolor=_ALL_PACKETS_STYLE['edge_color'],
                                 linewidth=4.5, label='All packets of considered flows'))
    for kind, series, values_by_key, labeller in families:
        rates_by_key = results.get('pass_rate_' + kind) or {}
        for i, key in enumerate(series):
            color_index = {'sampled': 0, 'uniform': len(methods),
                            'oracle': len(methods) + len(uniform_series)}[kind] + i
            style = family_border_style(kind, color_index)
            values_by_k = values_by_key.get(key, [[]] * n_k)
            missing_k = [k for k, values in zip(num_flows, values_by_k) if len(values) == 0]
            if missing_k:
                print("No {} mean delay for {} flow-count(s), skipped: {}".format(
                    labeller(key), len(missing_k), missing_k))
            _draw_boxplot_family(axis, num_flows, values_by_k,
                                 rates_by_key.get(key, [0.0] * n_k), offsets_by_kind[kind][key],
                                 box_width, pass_threshold, pass_color, fail_color, style)
            legend_handles.append(Patch(facecolor='white', edgecolor=style['edge_color'],
                                         linewidth=4.5, linestyle=style['edge_style'],
                                         label=labeller(key)))

    axis.set_title(title, fontsize=34)
    axis.set_xlabel('Number of TCP flows considered')
    axis.set_ylabel('Mean queuing delay (ns)')
    axis.set_ylim(bottom=0)
    _set_flow_count_xaxis(axis, num_flows)
    axis.grid(True, alpha=0.35, axis='y')
    axis.legend(handles=legend_handles, fontsize=18, loc='best')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path

def plot_prob_metric_vs_num_flows(results, metric, output_path, quantity='prob', title=None,
                                   pass_threshold=0.9):
    """Plot one probability metric (a PROB_METRICS key -- loss or ECN marking) per flow
    count, for every comparison family, in one of three views:

      - `quantity='prob'`: each family's own estimate of the path probability, with a
        dashed reference line at the value the switch traces give at the ground-truth rate.
        Boxes are coloured by the consistency check, like the delay EMD plot.
      - `quantity='distance'`: |estimate - reference|, which for a 0/1 outcome IS the
        Wasserstein distance the delay side reports as EMD (a Bernoulli has nothing else to
        its distribution). Filled neutrally, since a distance is not what the check judges.
      - `quantity='log_diff'`: the log-space difference the check actually thresholds,
        drawn against that family's own acceptance band (the shaded region between the mean
        lower and upper edge) -- the direct picture of the test passing or failing.

    Returns None without writing anything when the metric carries no testable data at all
    (see PROB_METRICS: the success probability is degenerate on traces that record no drop
    probability), rather than an empty axes that would read as a result."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = upgrade_emd_vs_flows_results_schema(results)
    block = (results.get('prob_metrics') or {}).get(metric)
    num_flows = results['num_flows']
    if not block:
        print("plot_prob_metric_vs_num_flows: results carry no '{}' metric (they predate it), "
              "skipping {}".format(metric, output_path))
        return None
    if quantity not in PROB_PLOT_QUANTITIES:
        raise ValueError("Unknown probability quantity {!r}; choose one of {}".format(
            quantity, list(PROB_PLOT_QUANTITIES)))
    families, box_width = _prob_family_layout(results)
    n_k = len(num_flows)

    def _values(family_key, key, field):
        family = block.get(family_key) or {}
        family = family if key is None else (family.get(key) or {})
        return (family.get(field) or [[] for _ in range(n_k)])

    if not any(len(v) for _, fk, _, key, _, _ in families for v in _values(fk, key, quantity)):
        print("plot_prob_metric_vs_num_flows: no {} values for '{}' -- nothing to plot, "
              "skipping {}".format(quantity, metric, output_path))
        return None

    pass_color, fail_color = 'tab:green', 'tab:red'
    neutral = '0.85'
    fill = None if quantity in ('prob', 'log_diff') else neutral
    fig, axis = plt.subplots(figsize=(30, 15))
    legend_handles = []

    if quantity == 'prob':
        reference = [float(np.mean(v)) if len(v) else np.nan
                      for v in (block.get('groundtruth_prob_by_run') or [[]] * n_k)]
        x = np.asarray(num_flows, dtype=float)
        valid = np.isfinite(reference)
        if np.any(valid):
            # Markers as well as the line: an all-flows-only result has a single flow count,
            # and a one-point line draws nothing at all.
            axis.plot(x[valid], np.asarray(reference)[valid], color='black', linewidth=3,
                       linestyle='--', marker='D', markersize=14, zorder=1)
            legend_handles.append(Line2D([0], [0], color='black', linewidth=3, linestyle='--',
                                          marker='D', markersize=14,
                                          label='Path {} from the switch traces (reference)'.format(
                                              prob_metric_label(metric).lower())))
    elif quantity == 'log_diff':
        axis.axhline(0, color='black', linewidth=2, linestyle=':', zorder=1)

    for label, family_key, rate_key, key, offset, style in families:
        values_by_k = _values(family_key, key, quantity)
        rates = block.get(rate_key) or {}
        rates = rates if key is None else (rates.get(key) or [0.0] * n_k)
        if quantity == 'log_diff':
            # The band is per run; its mean edges are drawn as a shaded strip behind this
            # family's boxes so "inside the band" is readable at a glance.
            lowers = _values(family_key, key, 'band_lower')
            uppers = _values(family_key, key, 'band_upper')
            for k, lo, hi in zip(num_flows, lowers, uppers):
                if not len(lo) or not len(hi):
                    continue
                lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
                lo, hi = lo[np.isfinite(lo)], hi[np.isfinite(hi)]
                if not lo.size or not hi.size:
                    continue
                lo_mean, hi_mean = float(lo.mean()), float(hi.mean())
                axis.add_patch(Rectangle((k + offset - box_width / 2, lo_mean), box_width,
                                          hi_mean - lo_mean, facecolor=style['edge_color'],
                                          alpha=0.12, edgecolor='none', zorder=0))
        _draw_boxplot_family(axis, num_flows, values_by_k, rates, offset, box_width,
                             pass_threshold, pass_color if fill is None else fill,
                             fail_color if fill is None else fill, style, fill_color=fill)
        legend_handles.append(Patch(facecolor=fill if fill is not None else 'white',
                                     edgecolor=style['edge_color'], linewidth=4.5,
                                     linestyle=style['edge_style'], label=label))

    if fill is None:
        legend_handles = [
            Patch(facecolor=pass_color, edgecolor='black', alpha=0.85,
                  label='Consistency check passed (>={:.0f}% of the testable runs)'.format(
                      pass_threshold * 100)),
            Patch(facecolor=fail_color, edgecolor='black', alpha=0.85,
                  label='Consistency check failed (<{:.0f}% of those runs)'.format(
                      pass_threshold * 100)),
        ] + legend_handles
    if quantity == 'log_diff':
        legend_handles.append(Patch(facecolor='0.6', alpha=0.25, edgecolor='none',
                                     label='Acceptance band of the check (mean edges, per family)'))

    # One source of truth for the axis wording, so a quantity added to PROB_PLOT_QUANTITIES
    # never needs a second edit here.
    y_label = (prob_metric_label(metric) if quantity == 'prob'
                else '{} ({})'.format(PROB_PLOT_QUANTITIES[quantity],
                                       prob_metric_label(metric).lower()))
    axis.set_title(title or '{} vs number of TCP flows'.format(prob_metric_label(metric)),
                    fontsize=34)
    axis.set_xlabel('Number of TCP flows considered')
    axis.set_ylabel(y_label)
    _set_flow_count_xaxis(axis, num_flows)
    axis.grid(True, alpha=0.35, axis='y')
    axis.legend(handles=legend_handles, fontsize=18, loc='best')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path

def plot_error_bound_vs_num_flows(results, output_path,
                                   title="Consistency-check error bound vs number of TCP flows",
                                   relative=True):
    """Plot the consistency check's own error bound -- the threshold it compares
    |switch mean - packet mean| against -- at every family's realized sample size, per run
    (results['error_bound_*_by_run'], see delay_consistency_error_bound). `relative=True`
    plots it as a fraction of the switch-side mean delay, with a dashed reference line at
    the run's DelayConsistencyGaurantee; `relative=False` plots the same bound in ns, the
    figure the mean difference is literally tested against.

    What to read off the relative plot: the required sample size is *defined* as the n that
    brings the bound down to exactly the guarantee (calc_min_e2e_samples), so a family
    holding exactly that many samples lands on the reference line -- to within the fraction
    of a sample lost when n is floored to an integer, which nudges it a few tenths of a
    percent above (observed 0.4001-0.4012 against a 0.40 guarantee). A box
    below the line means that family ended up with more samples than the minimum, either
    because the sampler retained more or because MINIMUM_E2E_SAMPLE_SIZE's floor forced
    more than the formula asked for (which on these runs is the common case, and puts the
    bound materially below the guarantee); it is then claiming a tighter guarantee than
    configured, which is safe. Nothing should sit *materially* above the line: that would
    mean a check ran at a looser bound than the run claims to guarantee.

    Boxes are filled one neutral colour rather than pass/fail green/red: the bound is the
    check's threshold, not its outcome, and colouring a threshold by the verdict it
    produced would conflate the two (see _draw_boxplot_family's fill_color). Families are
    told apart by border colour and dash as everywhere else (family_border_style)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = upgrade_emd_vs_flows_results_schema(results)
    num_flows = results['num_flows']
    methods = results['subsampling_methods']
    uniform_series = results.get('uniform_series', [])
    oracle_series = results.get('oracle_series', [])
    num_experiments = results.get('num_experiments', 1)
    prefix = 'error_bound_' if relative else 'error_bound_ns_'
    fill_color = '0.85'
    offset_all, offsets_poisson, offsets_uniform, offsets_oracle, box_width = _subsample_family_layout(
        methods, uniform_series, oracle_series)

    all_by_run = results.get(prefix + 'all_packets_by_run') or []
    families = [(kind, series, offsets, results.get(prefix + key + '_by_run') or {}, labeller)
                 for kind, series, offsets, key, labeller in (
                     ('sampled', methods, offsets_poisson, 'sampled',
                      lambda key: 'Poisson-adaptive subsample, {}'.format(key)),
                     ('uniform', uniform_series, offsets_uniform, 'uniform', _uniform_series_label),
                     ('oracle', oracle_series, offsets_oracle, 'oracle', _oracle_series_label))]
    if not any(len(v) for v in all_by_run) and not any(
            any(len(v) for v in values.get(key, [])) for _, series, _, values, _ in families
            for key in series):
        if results.get('analysis_window_method'):
            reason = ("no run certified a window at any flow count, so no check ever ran -- "
                      "the guarantee ({}) is out of reach for this experiment, see the "
                      "results text file".format(
                          "{:.0%}".format(results['delay_consistency_guarantee'])
                          if results.get('delay_consistency_guarantee') else 'configured'))
        else:
            reason = "these results predate error-bound recording"
        print("plot_error_bound_vs_num_flows: nothing to plot ({}), skipping {}".format(
            reason, output_path))
        return None

    fig, axis = plt.subplots(figsize=(30, 15))

    guarantee = results.get('delay_consistency_guarantee')
    legend_handles = []
    if relative and guarantee:
        axis.axhline(guarantee, color='black', linewidth=3, linestyle='--', zorder=1)
        legend_handles.append(Line2D([0], [0], color='black', linewidth=3, linestyle='--',
                                      label='Guaranteed relative error (DelayConsistencyGaurantee '
                                            '= {:.0%}) -- exactly where the minimum required '
                                            'sample size lands'.format(guarantee)))

    legend_handles += _draw_all_packets_series(
        axis, num_flows, [float(np.mean(v)) if len(v) else np.nan for v in all_by_run],
        all_by_run, results['pass_rate_all_packets'], offset_all, box_width, 1.0,
        fill_color, fill_color, results['num_runs'], num_experiments, 'error bound',
        fill_color=fill_color, spread_label=_all_packets_spread_label(results))

    for kind, series, offsets, values_by_key, labeller in families:
        for i, key in enumerate(series):
            color_index = {'sampled': 0, 'uniform': len(methods),
                            'oracle': len(methods) + len(uniform_series)}[kind] + i
            style = family_border_style(kind, color_index)
            values_by_k = values_by_key.get(key, [[]] * len(num_flows))
            missing_k = [k for k, values in zip(num_flows, values_by_k) if len(values) == 0]
            if missing_k:
                print("No {} error bound for {} flow-count(s), skipped: {}".format(
                    labeller(key), len(missing_k), missing_k))
            _draw_boxplot_family(axis, num_flows, values_by_k, [1.0] * len(num_flows),
                                 offsets[key], box_width, 1.0, fill_color, fill_color, style,
                                 fill_color=fill_color)
            legend_handles.append(Patch(facecolor=fill_color, edgecolor=style['edge_color'],
                                         linewidth=4.5, linestyle=style['edge_style'],
                                         label=labeller(key)))

    axis.set_title(title, fontsize=34)
    axis.set_xlabel('Number of TCP flows considered')
    # Kept short: a longer label overflows the left edge of this figure size rather than
    # getting room reserved for it, and the title already says what the quantity is.
    axis.set_ylabel('Error bound' + (' (of switch mean delay)' if relative else ' (ns)'))
    axis.set_ylim(bottom=0)
    if relative:
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    _set_flow_count_xaxis(axis, num_flows)
    axis.grid(True, alpha=0.35, axis='y')
    axis.legend(handles=legend_handles, fontsize=18, loc='best')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path

def plot_monitor_window_vs_num_flows(results, output_path,
                                      title="Monitoring window needed vs number of TCP flows",
                                      pass_threshold=0.9):
    """Plot how long each Poisson-adaptive subsampling method had to watch the flow before
    it could draw the samples the consistency check needs: boxplots (across runs) of the
    monitoring-window length (results['window_duration_sampled_by_run'], in ms) at every
    flow count, with a dashed reference line at the full steady window every run had
    available.

    This is the growing-window methods' headline result (see _growing_window_search): their
    window is the shortest prefix of the steady period whose own switch-side statistics and
    packets could supply the required samples, so a box well below the reference line is
    the actual answer to "how long must we watch this flow to certify it?". A method that
    samples the whole steady window instead sits exactly on the line by construction, which
    is what makes the comparison legible -- the same plot shows both kinds without
    special-casing either.

    Only the Poisson-adaptive families appear: a rate-matched uniform family and an ideal
    probe inherit their method's window rather than choosing one, and the all-packets
    family is the whole steady window by definition, so neither would contribute anything
    the reference line does not already show.

    Boxes are coloured by the consistency-check pass rate (green/red, like the EMD and
    mean-difference plots) rather than filled neutrally: here the verdict belongs with the
    quantity -- the claim being made is "this much monitoring was enough (or not)" -- and a
    window whose samples then failed the check is exactly the case a reader must not
    mistake for a cheap success. Runs that found no valid subsample at all contribute no
    value (they never settled on a window), so a box's n matches that method's 'n_samp' in
    the text summary."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = upgrade_emd_vs_flows_results_schema(results)
    num_flows = results['num_flows']
    methods = results['subsampling_methods']
    windows_by_method = results.get('window_duration_sampled_by_run', {})
    pass_color, fail_color = 'tab:green', 'tab:red'
    # Only the Poisson-adaptive families are drawn, so the layout is asked for exactly those
    # (no uniform/oracle offsets to reserve space for).
    offset_all, offsets_poisson, _, _, box_width = _subsample_family_layout(methods, [], [])

    fig, axis = plt.subplots(figsize=(30, 15))

    steady_start, steady_end = results.get('steady_start'), results.get('steady_end')
    legend_handles = []
    if steady_start is not None and steady_end is not None:
        full_window_ms = (float(steady_end) - float(steady_start)) / 1e6
        axis.axhline(full_window_ms, color='black', linewidth=3, linestyle='--', zorder=1)
        legend_handles.append(Line2D([0], [0], color='black', linewidth=3, linestyle='--',
                                      label='Full steady window available ({:.3g} ms)'.format(full_window_ms)))
    step_ns = results.get('growing_window_step_ns')
    if step_ns:
        legend_handles.append(Line2D([0], [0], color='none',
                                      label='Growing-window step: {:.3g} ms'.format(step_ns / 1e6)))
    legend_handles += [
        Patch(facecolor=pass_color, edgecolor='black', alpha=0.85,
              label='Consistency check passed (>={:.0f}% of the runs that found a subsample)'.format(
                  pass_threshold * 100)),
        Patch(facecolor=fail_color, edgecolor='black', alpha=0.85,
              label='Consistency check failed (<{:.0f}% of those runs)'.format(pass_threshold * 100)),
    ]

    drew_any = False
    for i, method in enumerate(methods):
        style = family_border_style('sampled', i)
        values_by_k = [np.asarray(values, dtype=float) / 1e6
                        for values in windows_by_method.get(method, [[]] * len(num_flows))]
        missing_k = [k for k, values in zip(num_flows, values_by_k) if len(values) == 0]
        if missing_k:
            print("No {} monitoring-window values for {} flow-count(s), skipped: {}".format(
                method, len(missing_k), missing_k))
        if any(len(values) for values in values_by_k):
            drew_any = True
        whisker_tops = _draw_boxplot_family(
            axis, num_flows, values_by_k, results['pass_rate_sampled'][method],
            offsets_poisson[method], box_width, pass_threshold, pass_color, fail_color, style)
        # The y axis starts at 0 and is scaled by the full window on purpose (a box's height
        # off the floor IS the fraction of the available monitoring time it needed), which
        # leaves a method that stops after a few ms as a sliver. Label each box with its
        # median so the cheap cases stay readable without rescaling away the comparison.
        for k, values in zip(num_flows, values_by_k):
            if not len(values):
                continue
            axis.annotate('{:.3g} ms'.format(float(np.median(values))),
                           xy=(k + offsets_poisson[method], whisker_tops.get(k, np.median(values))),
                           xytext=(0, 6), textcoords='offset points', ha='center', va='bottom',
                           fontsize=13, color=style['edge_color'], zorder=4)
        legend_handles.append(Patch(facecolor='white', edgecolor=style['edge_color'], linewidth=4.5,
                                     linestyle=style['edge_style'],
                                     label='Poisson-adaptive subsample, {}'.format(method)))

    if not drew_any:
        # Two quite different reasons land here, and saying the wrong one sends a reader
        # hunting for a schema problem when the run simply never certified anything: a
        # result predating window recording carries no durations at all
        # (upgrade_emd_vs_flows_results_schema fills empty placeholders), while a current
        # result with no durations means no run ever found a usable window. Either way an
        # empty axes would read as "no monitoring was needed", so nothing is written.
        if any(len(v) for v in (results.get('window_duration_sampled_by_run') or {}).get(
                methods[0] if methods else None, []) or []):
            reason = "unexpected: durations exist but none were plotted"
        elif results.get('analysis_window_method'):
            reason = ("no run certified a window at any flow count -- the guarantee "
                      "({}) is out of reach for this experiment, see the results text file"
                      .format("{:.0%}".format(results['delay_consistency_guarantee'])
                               if results.get('delay_consistency_guarantee') else 'configured'))
        else:
            reason = "these results predate monitoring-window recording"
        print("plot_monitor_window_vs_num_flows: nothing to plot ({}), skipping {}".format(
            reason, output_path))
        plt.close(fig)
        return None

    axis.set_title(title, fontsize=34)
    axis.set_xlabel('Number of TCP flows considered')
    axis.set_ylabel('Monitoring window needed (ms)')
    axis.set_ylim(bottom=0)
    _set_flow_count_xaxis(axis, num_flows)
    axis.grid(True, alpha=0.35, axis='y')
    axis.legend(handles=legend_handles, fontsize=18, loc='best')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path

_PERCENTILE_FAMILY_FILLS = ['0.85', 'lightsteelblue', 'navajowhite', 'thistle',
                             'lightseagreen', 'lightcoral', 'khaki']


def plot_percentile_diff_vs_num_flows(results, percentile, output_path, relative=False,
                                       title=None, pass_threshold=0.9, y_limit=None):
    """Plot the signed percentile (tail-shape) error against the number of considered TCP
    flows: at each flow count, `ground_truth_p<percentile> - family_p<percentile>` for the
    all-packet CDF, every Poisson-adaptive subsampling method, and every rate-matched
    uniform baseline. With `relative` set, each error is divided by the ground truth's own
    percentile, which is the form comparable across offered loads.

    A **positive** value means the family understates that percentile -- it is missing tail
    delay the ground truth has; negative means it overstates it. The dotted line at zero is
    exact agreement.

    Unlike the EMD and mean-difference plots, boxes here are *not* coloured green/red: the
    consistency check tests the mean, so it makes no claim about a percentile, and colouring
    by it would imply one. Families are identified by border colour/dash (as elsewhere) plus a
    per-family fill shade. All packets of the first k flows is a fixed packet set, so within
    one experiment its error is one value per k (dots + line) and only becomes a boxplot once
    aggregated across experiments -- see _draw_all_packets_series.

    Returns the output path, or None when `results` carries no such percentile (e.g. a
    results pickle predating percentile errors, see upgrade_emd_vs_flows_results_schema)."""
    results = upgrade_emd_vs_flows_results_schema(results)
    if percentile not in (results.get('delay_percentiles') or []):
        print("plot_percentile_diff_vs_num_flows: results carry no p{} error, skipped".format(percentile))
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kind = 'percentile_reldiff' if relative else 'percentile_diff'
    all_key, all_by_exp_key, sampled_key, uniform_key, oracle_key, y_label = _metric_result_keys(
        (kind, percentile), False)

    num_flows = results['num_flows']
    methods = results['subsampling_methods']
    uniform_series = results.get('uniform_series', [])
    oracle_series = results.get('oracle_series', [])
    offset_all, offsets_poisson, offsets_uniform, offsets_oracle, box_width = _subsample_family_layout(
        methods, uniform_series, oracle_series)
    gt_percentile = (results.get('groundtruth_percentiles') or {}).get(percentile, np.nan)

    fig, axis = plt.subplots(figsize=(30, 15))
    axis.axhline(0, color='black', linewidth=2, linestyle=':', zorder=1)

    fills = iter(_PERCENTILE_FAMILY_FILLS)
    legend_handles = _draw_all_packets_series(
        axis, num_flows, results[all_key][percentile],
        (results.get(all_by_exp_key) or {}).get(percentile),
        results['pass_rate_all_packets'], offset_all, box_width, pass_threshold,
        None, None, results['num_runs'], results.get('num_experiments', 1),
        'p{} error'.format(percentile), fill_color=next(fills),
        spread_label=_all_packets_spread_label(results))

    for i, method in enumerate(methods):
        style = family_border_style('sampled', i)
        fill = next(fills, _PERCENTILE_FAMILY_FILLS[-1])
        values_by_k = results[sampled_key][percentile][method]
        _draw_boxplot_family(axis, num_flows, values_by_k, results['pass_rate_sampled'][method],
                             offsets_poisson[method], box_width, pass_threshold, fill, fill, style,
                             fill_color=fill)
        legend_handles.append(Patch(facecolor=fill, edgecolor=style['edge_color'], linewidth=4.5,
                                     linestyle=style['edge_style'],
                                     label='Poisson-adaptive subsample, {}'.format(method)))

    for i, key in enumerate(uniform_series):
        style = family_border_style('uniform', len(methods) + i)
        fill = next(fills, _PERCENTILE_FAMILY_FILLS[-1])
        values_by_k = results[uniform_key][percentile][key]
        _draw_boxplot_family(axis, num_flows, values_by_k, results['pass_rate_uniform'][key],
                             offsets_uniform[key], box_width, pass_threshold, fill, fill, style,
                             fill_color=fill)
        legend_handles.append(Patch(facecolor=fill, edgecolor=style['edge_color'], linewidth=4.5,
                                     linestyle=style['edge_style'],
                                     label=_uniform_series_label(key)))

    for i, key in enumerate(oracle_series):
        style = family_border_style('oracle', len(methods) + len(uniform_series) + i)
        fill = next(fills, _PERCENTILE_FAMILY_FILLS[-1])
        values_by_k = results[oracle_key][percentile][key]
        _draw_boxplot_family(axis, num_flows, values_by_k, results['pass_rate_oracle'][key],
                             offsets_oracle[key], box_width, pass_threshold, fill, fill, style,
                             fill_color=fill)
        legend_handles.append(Patch(facecolor=fill, edgecolor=style['edge_color'], linewidth=4.5,
                                     linestyle=style['edge_style'],
                                     label=_oracle_series_label(key)))

    legend_handles.append(Line2D([0], [0], color='black', linewidth=2, linestyle=':',
                                  label='zero = family percentile matches ground truth'))

    gt_note = '' if not np.isfinite(gt_percentile) else ', ground-truth p{} = {:.1f} ns'.format(
        percentile, gt_percentile)
    axis.set_title(title or '{} vs number of TCP flows{}'.format(
        y_label, gt_note), fontsize=34)
    axis.set_xlabel('Number of TCP flows considered')
    axis.set_ylabel(y_label)
    _set_flow_count_xaxis(axis, num_flows)
    axis.grid(True, alpha=0.35, axis='y')
    if relative:
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    if y_limit is not None:
        axis.set_ylim(-y_limit, y_limit)
    axis.legend(handles=legend_handles, fontsize=18, loc='best')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# Fills for the Poisson-test split. Deliberately NOT green/red: on every other plot that
# pair means the delay consistency check, and this split is a different question entirely.
_POISSON_TEST_FILLS = {True: 'steelblue', False: '0.78'}


def _poisson_split_layout(n_families):
    """Offsets/box-width for a Poisson-test split plot: two boxes (test passed, test
    failed) per tested family, grouped so each family's pair sits together. Shares the
    spacing constants with the other flow-count plots."""
    n_slots = max(2 * n_families, 1)
    span = _FAMILY_GROUP_SPAN
    box_width = (span / n_slots) * _FAMILY_BOX_FILL
    offsets = np.linspace(-span / 2, span / 2, n_slots) if n_slots > 1 else np.array([0.0])
    return offsets, box_width


def plot_poisson_test_split_vs_num_flows(results, output_path, test_name='ad', quantity='emd',
                                          title=None, y_max=None, y_limit=None):
    """Plot one metric against flow count for the families that are *not* Poissonized --
    all packets, and each rate-matched uniform subset -- with each family's runs **split by
    whether that run's own sampling instants passed the Poisson-ness test**
    (poisson_process_tests): one box for the runs that passed, one for those that failed.

    `test_name` selects the criterion: 'ad' = Anderson-Darling on the inter-arrival gaps
    alone; 'ad_chi' = Anderson-Darling *and* the multi-lag chi-squared independence test
    (see poisson_test_label). `quantity` is 'emd', 'emd_normalized' or 'mean_diff'.

    The question this answers: those two families never had to pass anything, so does it
    matter whether their instants happen to look Poisson? If the passing runs' boxes sit
    closer to the ground truth than the failing ones, Poisson-ness of the instants is
    doing real work; if the two boxes coincide, it is not the thing driving the error here.

    Fill encodes the *test* outcome (blue = passed, grey = failed) and deliberately avoids
    the green/red used everywhere else for the delay consistency check, which is a
    different question -- the check tests the mean against the switch bound, not whether
    the instants form a Poisson process. Border colour/dash identifies the family as usual.

    An empty box simply means no run landed on that side of the split, which is itself the
    result (these families are expected to fail AD most of the time -- that is why
    Poissonization exists). Returns None when `results` carries no test verdicts, e.g. a
    pickle predating this, or a run with `run_chi_squared_test=False` asked for 'ad_chi'.
    """
    results = upgrade_emd_vs_flows_results_schema(results)
    series = results.get('poisson_test_series') or []
    if not series:
        print("plot_poisson_test_split_vs_num_flows: results carry no Poisson-test verdicts, skipped")
        return None
    if test_name == 'ad_chi' and not results.get('run_chi_squared_test'):
        print("plot_poisson_test_split_vs_num_flows: chi-squared test was not run, "
              "skipping the '{}' split".format(test_name))
        return None

    value_keys = {
        'emd': ('emd_all_packets', 'emd_all_packets_by_experiment', 'emd',
                 'EMD to reconstructed network delay CDF (ns)'),
        'emd_normalized': ('emd_all_packets_normalized', 'emd_all_packets_by_experiment_normalized',
                            'emd_normalized', 'EMD relative to mean queuing delay'),
        'mean_diff': (None, 'mean_diff_all_packets_by_run', 'mean_diff',
                       'Switch samples mean delay - packet mean delay (ns)'),
    }
    if quantity not in value_keys:
        raise ValueError("Unknown quantity {!r}; choose one of {}".format(quantity, list(value_keys)))
    all_key, all_by_exp_key, split_field, y_label = value_keys[quantity]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_flows = results['num_flows']
    offsets, box_width = _poisson_split_layout(len(series))

    fig, axis = plt.subplots(figsize=(30, 15))
    if quantity == 'mean_diff':
        axis.axhline(0, color='black', linewidth=2, linestyle=':', zorder=1)

    legend_handles = [
        Patch(facecolor=_POISSON_TEST_FILLS[True], edgecolor='black', alpha=0.85,
              label='Runs whose sampling instants PASSED {}'.format(poisson_test_label(test_name))),
        Patch(facecolor=_POISSON_TEST_FILLS[False], edgecolor='black', alpha=0.85,
              label='Runs whose sampling instants FAILED {}'.format(poisson_test_label(test_name))),
    ]

    any_data = False
    for fi, key in enumerate(series):
        if key == 'all_packets':
            style = family_border_style('all_packets', fi)
            label = 'All packets of considered flows'
            values_by_k, flags_by_k = [], []
            for i in range(len(num_flows)):
                if quantity == 'mean_diff':
                    # mean_diff varies per run (the switch mean is redrawn), while the
                    # all-packet verdict is per experiment -- so every run of an experiment
                    # inherits that experiment's verdict. With one experiment that puts the
                    # whole box on one side of the split, which is the honest rendering.
                    flags = _all_packets_test_flags(results, i, test_name)
                    values = list(results['mean_diff_all_packets_by_run'][i])
                    flag = flags[0] if len(set(flags)) == 1 and flags else None
                    values_by_k.append(values)
                    flags_by_k.append([flag] * len(values))
                    continue
                values = _all_packets_plot_values(results, i, all_key, all_by_exp_key)
                flags = _all_packets_test_flags(results, i, test_name)
                if len(flags) != len(values):
                    values, flags = [], []
                values_by_k.append(values)
                flags_by_k.append(flags)
        else:
            method = key[1]
            style = family_border_style('uniform', fi)
            label = _uniform_series_label(method)
            record = (results.get('uniform_test_split_by_run') or {}).get(method)
            if record is None:
                continue
            values_by_k = [list(v) for v in record[split_field]]
            flags_by_k = [[poisson_test_outcome({'ad_pass': a, 'chi_pass': c}, test_name)
                            for a, c in zip(record['ad_pass'][i], record['chi_pass'][i])]
                           for i in range(len(num_flows))]

        for oi, passed in enumerate((True, False)):
            selected = []
            for values, flags in zip(values_by_k, flags_by_k):
                selected.append([v for v, f in zip(values, flags)
                                  if f is passed and v == v])
            if any(len(v) for v in selected):
                any_data = True
            fill = _POISSON_TEST_FILLS[passed]
            _draw_boxplot_family(axis, num_flows, selected, [1.0] * len(num_flows),
                                 offsets[2 * fi + oi], box_width, 0.0, fill, fill, style,
                                 fill_color=fill)
            counts = [len(v) for v in selected]
            print("{} -- {} {}: runs per flow count {}".format(
                label, poisson_test_label(test_name), 'PASSED' if passed else 'FAILED', counts))
        legend_handles.append(Patch(facecolor='white', edgecolor=style['edge_color'], linewidth=4.5,
                                     linestyle=style['edge_style'], label=label))

    if not any_data:
        print("plot_poisson_test_split_vs_num_flows: no run fell on either side of the "
              "'{}' split, writing empty plot".format(test_name))

    axis.set_title(title or '{} split by {}'.format(y_label, poisson_test_label(test_name)),
                    fontsize=34)
    axis.set_xlabel('Number of TCP flows considered')
    axis.set_ylabel(y_label)
    _set_flow_count_xaxis(axis, num_flows)
    axis.grid(True, alpha=0.35, axis='y')
    if quantity == 'emd_normalized' and y_max is not None:
        axis.set_ylim(bottom=0, top=y_max)
    elif quantity == 'emd' and y_max is not None:
        axis.set_ylim(bottom=0, top=y_max)
    elif quantity == 'mean_diff' and y_limit is not None:
        axis.set_ylim(-y_limit, y_limit)
    axis.legend(handles=legend_handles, fontsize=18, loc='best')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _prob_metric_text_section(results, metric, num_flows_display, stat_fn):
    """The text-summary block for one probability metric: the reference probability, then one
    table per comparison family with its estimate, its distance to that reference, the
    log-space difference and band the check applied, the sample size, and the pass rate.

    `stat_fn` is save_emd_vs_flows_results_text's own mean +/- std formatter, so these
    tables read exactly like the delay ones."""
    block = (results.get('prob_metrics') or {}).get(metric)
    if not block:
        return []
    n_k = len(results['num_flows'])
    methods = results['subsampling_methods']
    uniform_series = results.get('uniform_series', [])
    oracle_series = results.get('oracle_series', [])
    lines = ['', '=' * 70, '{} ({})'.format(prob_metric_label(metric), metric), '=' * 70]

    gt_by_k = block.get('groundtruth_prob_by_run') or [[] for _ in range(n_k)]
    lines.append("Reference (path probability from the switch traces at the ground-truth rate):")
    for i, k in enumerate(num_flows_display):
        lines.append("  k={:<4} {}".format(k, stat_fn(gt_by_k[i], fmt="{:.6f}")))

    # A metric whose switch side carries no variance at all cannot be tested; say so once,
    # loudly, instead of leaving a reader to wonder why every pass rate is 0%.
    testable = sum(1 for per_k in (block.get('all_packets_by_run') or {}).get('consistency_pass', [])
                    for v in per_k if v is not None)
    if testable == 0:
        lines.append("")
        lines.append("!! NOT TESTABLE on these traces: the acceptance band has zero width, which means")
        lines.append("   neither side of this comparison carries any variance -- for the success")
        lines.append("   probability that is exactly what happens when the queue traces record no drop")
        lines.append("   probability anywhere and the flow loses no packets on the path, so both sides")
        lines.append("   are exactly 1. The estimates below are still reported, but every 'pass' column")
        lines.append("   is out of zero testable runs and means nothing. See Utils.PROB_METRICS.")

    families = ([('All packets of the considered flows', 'all_packets_by_run', 'pass_rate_all_packets', None)]
                 + [('Poisson-adaptive subsample -- {}'.format(m), 'sampled_by_run', 'pass_rate_sampled', m)
                    for m in methods]
                 + [(_uniform_series_label(s), 'uniform_by_run', 'pass_rate_uniform', s)
                    for s in uniform_series]
                 + [(_oracle_series_label(o), 'oracle_by_run', 'pass_rate_oracle', o)
                    for o in oracle_series])
    for title, family_key, rate_key, key in families:
        family = block.get(family_key) or {}
        family = family if key is None else (family.get(key) or {})
        rates = block.get(rate_key) or {}
        rates = rates if key is None else (rates.get(key) or [0.0] * n_k)
        if not family:
            continue
        lines.append("")
        lines.append("{}:".format(title))
        header = "{:>3} | {:>6} | {:>22} | {:>22} | {:>22} | {:>22} | {:>22} | {:>9} | {:>14}".format(
            "k", "n_test", prob_metric_label(metric), "EMD to reference", "relEMD",
            "log_diff", "band (lower/upper)", "pass", "n_pkts")
        lines.append(header)
        lines.append("-" * len(header))
        for i, k in enumerate(num_flows_display):
            verdicts = (family.get('consistency_pass') or [[]] * n_k)[i]
            n_test = sum(1 for v in verdicts if v is not None)
            lower = (family.get('band_lower') or [[]] * n_k)[i]
            upper = (family.get('band_upper') or [[]] * n_k)[i]
            lower = np.asarray(lower, dtype=float)
            upper = np.asarray(upper, dtype=float)
            lower, upper = lower[np.isfinite(lower)], upper[np.isfinite(upper)]
            band = "n/a"
            if lower.size and upper.size:
                band = "{:+.4f} / {:+.4f}".format(float(lower.mean()), float(upper.mean()))
            lines.append("{:>3} | {:>6} | {:>22} | {:>22} | {:>22} | {:>22} | {:>22} | {:>8.0%} | {:>14}".format(
                k, n_test,
                stat_fn((family.get('prob') or [[]] * n_k)[i], fmt="{:.6f}"),
                stat_fn((family.get('distance') or [[]] * n_k)[i], fmt="{:.6f}"),
                stat_fn((family.get('distance_normalized') or [[]] * n_k)[i], fmt="{:.4f}"),
                stat_fn((family.get('log_diff') or [[]] * n_k)[i], fmt="{:+.5f}"),
                band, rates[i],
                stat_fn((family.get('sample_size') or [[]] * n_k)[i], fmt="{:.0f}"),
            ))
    lines.append("")
    lines.append("Notes for this metric:")
    lines.append("  * The per-packet outcome is 0/1, so the whole distribution IS its mean: the 'EMD to")
    lines.append("    reference' column is |estimate - reference|, which is exactly the Wasserstein")
    lines.append("    distance the delay tables call EMD (W1 between two Bernoullis is |p-q|), and")
    lines.append("    'relEMD' is it divided by the reference probability -- the same normalization the")
    lines.append("    delay side's relEMD uses (EMD / mean ground-truth delay). There is no percentile or")
    lines.append("    CDF counterpart to report for a 0/1 variable.")
    lines.append("  * The check is multiplicative (a path probability is the product of its segments'),")
    lines.append("    so it is applied in log space: it passes when log_diff = log(estimate) -")
    lines.append("    SUM log(segment probability) falls inside the band, which is the switch side's")
    lines.append("    worst-segment relative error compounded over the 3 segments, widened by the e2e")
    lines.append("    side's own error at n_pkts samples. The band is asymmetric, which is why it is")
    lines.append("    shown as lower/upper rather than as a single +/- bound.")
    lines.append("  * n_pkts is the same subsample the delay families used -- no separate minimum")
    lines.append("    sample size is computed for a 0/1 outcome (it varies far less than a delay, so")
    lines.append("    the delay-derived count is the conservative choice). n_test counts the runs whose")
    lines.append("    band was testable at all.")
    return lines

def save_emd_vs_flows_results_text(results, output_path):
    """Write a plain-text, human-readable summary of the per-flow-count
    results from compute_emd_vs_num_tcp_flows_multi_run: run parameters, a
    summary of the (very large) ground-truth sample array, a main table
    (one row per number of considered TCP flows) covering all-packets, and
    one further table per subsampling method -- each Poisson-adaptive method
    that was run, then each uniform family. All-packet EMD is a
    single value (the same fixed packet set every run); everything else that
    varies run to run is reported as mean +/- std across the runs that had a
    valid value at that flow count, alongside consistency-check pass rates.

    Each per-method table carries an 'n_pkts' column: the number of packets that
    method actually retained. For a rate-matched uniform family this is the count it
    was told to match, so the two paired tables' n_pkts columns agreeing is the
    direct confirmation that the comparison is at equal sample size.

    Every EMD is reported twice: raw (ns) and normalized by the mean
    ground-truth path delay (see normalize_emd_values), the latter being the
    figure comparable across offered loads.

    A final section reports the signed percentile (tail-shape) error of every family at
    each of results['delay_percentiles'], absolute (ns) and relative to the ground truth's
    own percentile -- one row per flow count, one column per family, so the families are
    directly comparable at a glance. Omitted entirely for results that carry no
    percentiles (see upgrade_emd_vs_flows_results_schema)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = upgrade_emd_vs_flows_results_schema(results)

    def _stat(values, fmt="{:.2f}"):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return "n/a"
        if values.size == 1:
            return (fmt + " (n=1)").format(values[0])
        return (fmt + " +/- " + fmt + " (n={})").format(np.mean(values), np.std(values), values.size)

    methods = results['subsampling_methods']
    uniform_series = results.get('uniform_series', [])
    oracle_series = results.get('oracle_series', [])
    num_experiments = results.get('num_experiments', 1)
    # Display-only: a pooled all_flows_only result's 'k' is the ALL_FLOWS_ONLY_K sentinel
    # (see aggregate_emd_vs_flows_results), not a real flow count -- show 'all' instead of
    # that sentinel number in every per-k table below.
    num_flows_display = ['all' if k == ALL_FLOWS_ONLY_K else k for k in results['num_flows']]
    emd_all_by_experiment = results.get('emd_all_packets_by_experiment')
    emd_all_by_experiment_norm = results.get('emd_all_packets_by_experiment_normalized')
    all_packets_is_aggregated = bool(emd_all_by_experiment) and (
        num_experiments > 1 or any(len(v) > 1 for v in emd_all_by_experiment))
    gt = np.asarray(results.get('groundtruth_values', []), dtype=float)
    gt_mean = results.get('groundtruth_mean', np.nan)
    lines = []
    lines.append("EMD vs number of TCP flows -- results summary")
    lines.append("=" * 70)
    lines.append("Flow: {}".format(results['flow_name']))
    lines.append("Path: {}".format(results['path']))
    lines.append("Poisson-adaptive subsampling method(s): {}".format(", ".join(methods)))
    lines.append("Ground truth: {} ({})".format(
        results.get('groundtruth_method', 'simultaneous'),
        groundtruth_method_label(results.get('groundtruth_method', 'simultaneous'))))
    if results.get('all_flows_only'):
        lines.append("Flow-count sweep: skipped -- all flows on the path only (every received e2e packet)")
    if num_experiments > 1:
        lines.append("Aggregated over {} experiments: {}".format(num_experiments, results.get('experiments')))
    lines.append("Number of runs (N): {}".format(results['num_runs']))
    lines.append("Poisson observations per run (M): {}".format(results['num_poisson_observations']))
    lines.append("Uniform baselines: {}".format(
        ", ".join(_uniform_series_label(key) for key in uniform_series) or "none"))
    lines.append("Ideal Poisson probes: {}".format(
        ", ".join(_oracle_series_label(key) for key in oracle_series) or "none"))
    lines.append("Total TCP flows considered (max k): {}".format(results['total_flows']))
    window_method = results.get('analysis_window_method')
    if window_method:
        certified_runs = sum(len(v) for v in (results.get('window_duration_sampled_by_run') or {}).get(
            window_method, []) or [])
        if certified_runs == 0:
            guarantee = results.get('delay_consistency_guarantee')
            lines.append("!! NOTHING CERTIFIED: no run found a window, at any flow count, whose own")
            lines.append("   switch-side statistics and packets could supply the samples the consistency")
            lines.append("   check needs at the configured guarantee{}. Every family below is therefore".format(
                " of {:.0%}".format(guarantee) if guarantee else ""))
            lines.append("   empty (n/a) and no plot was written. This is a feasibility result, not a")
            lines.append("   failure of the sampler: a tighter guarantee needs a smaller switch-side")
            lines.append("   epsilon (more Poisson observations per run) or more e2e samples than the")
            lines.append("   window can yield. Raise the guarantee, raise --num-poisson-observations, or")
            lines.append("   check the per-window arithmetic with Utils.windowed_poisson_agg_stats.")
            lines.append("")
        lines.append("Analysis window: PER RUN, the growing window {} settled on (see the".format(window_method))
        lines.append("  monitor_window column below). Every quantity in this file -- including the")
        lines.append("  all-packets family, the uniform baselines and the ideal probes -- was measured")
        lines.append("  over [steadyStart, window_end] of its own run. Nothing here is computed over the")
        lines.append("  full steady window unless a run's search reached steadyEnd, and a run that found")
        lines.append("  no usable window reports nothing at all (it certified nothing), which is why every")
        lines.append("  family's pass rate is out of the runs that produced a value.")
    else:
        lines.append("Analysis window: the full steady window, identical for every run and family.")
    lines.append("")
    if window_method:
        lines.append("Ground-truth reconstructed delay samples (of ONE run's window -- every run rebuilt")
        lines.append("  its own over the window it settled on; this is the realization the delay-CDF plot")
        lines.append("  draws): {}".format(gt.size))
    else:
        lines.append("Ground-truth reconstructed delay samples: {}".format(gt.size))
    if gt.size:
        lines.append("  mean={:.2f} ns, std={:.2f} ns, min={:.2f} ns, max={:.2f} ns".format(
            np.mean(gt), np.std(gt), np.min(gt), np.max(gt)))
        p5, p25, p50, p75, p95 = np.percentile(gt, [5, 25, 50, 75, 95])
        lines.append("  percentiles (5/25/50/75/95) ns: {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(
            p5, p25, p50, p75, p95))
    if window_method:
        lines.append("EMD normalizer: each value was divided by the mean ground-truth delay of ITS OWN run's"
                      " window{}".format(" (this realization's: {:.2f} ns)".format(gt_mean)
                                          if np.isfinite(gt_mean) else ""))
    else:
        lines.append("EMD normalizer (mean ground-truth path delay): {:.2f} ns".format(gt_mean)
                      if np.isfinite(gt_mean) else "EMD normalizer (mean ground-truth path delay): n/a")
    burst_gap = results.get('burst_gap_threshold_ns', np.nan)
    lines.append("Burst gap threshold (1 MSS={}B transmission time at the sender's own link rate): {:.2f} ns".format(
        MSS_BYTES, burst_gap) if np.isfinite(burst_gap)
        else "Burst gap threshold: n/a (results predate burstiness metrics -- see backfill_burstiness_metrics)")
    gt_percentiles = results.get('groundtruth_percentiles') or {}
    percentiles = list(results.get('delay_percentiles') or [])
    if percentiles:
        lines.append("Ground-truth percentiles{}: {}".format(
            " (of that one run's window; each run's percentile errors use its own)" if window_method else "",
            ", ".join("p{}={:.2f} ns".format(q, gt_percentiles.get(q, np.nan)) for q in percentiles)))
        if num_experiments > 1:
            # Same caveat the normalizer above carries: these are of the pooled ground-truth
            # samples, whereas each experiment's relative errors were divided by its *own*
            # percentile (they must be, since each reconstructs its own ground truth). So
            # dividing an absolute figure below by the number on this line will not exactly
            # reproduce the relative one.
            lines.append("  (of the pooled samples; each experiment's relative errors below use its own"
                          " percentiles as the reference)")
    lines.append("")
    lines.append("Notes:")
    if window_method:
        lines.append("  * EMD(all) and mean_diff(all): all packets of the first k flows *inside that run's")
        lines.append("    window*, so unlike an ordinary run this is a different packet set every run, with")
        lines.append("    its own ground truth -- reported as mean +/- std across runs{}.".format(
            " and experiments" if num_experiments > 1 else ""))
    elif all_packets_is_aggregated:
        lines.append("  * EMD(all): all packets of the first k flows, one value per experiment (each experiment")
        lines.append("    reconstructs its own ground truth), reported as mean +/- std across experiments.")
        lines.append("    mean_diff(all) still varies run *and* experiment to experiment.")
    else:
        lines.append("  * EMD(all) and mean_diff(all): all packets of the first k flows (same fixed set every run;")
        lines.append("    mean_diff still varies run to run because the switch-side mean is redrawn each run).")
    lines.append("  * One table per Poisson-adaptive subsampling method below, each a fresh subsample drawn")
    lines.append("    every run; 'n_samp' is how many of the N runs found a valid subsample at that flow count.")
    if oracle_series:
        lines.append("  * Ideal-Poisson-probe tables: NOT a subsample of the flow's packets at all -- the")
        lines.append("    ground-truth construction re-run at a realistic rate (about the stated sample")
        lines.append("    budget). Its instants are a genuine Poisson process independent of queue state, so")
        lines.append("    PASTA holds exactly and there is no selection bias: its only error is finite-sample")
        lines.append("    noise. It is therefore the CEILING for any Poissonization scheme, and the gap")
        lines.append("    between a real method and the probe at that method's own sample count is the part")
        lines.append("    of its error that having few samples does NOT explain.")
        lines.append("  * The probe's n_pkts fluctuates around its budget rather than matching it exactly:")
        lines.append("    a Poisson process over a fixed window has a random number of points, and pinning")
        lines.append("    the count would make the instants uniform order statistics, not a Poisson process.")
    lines.append("  * Uniform tables further below: a fresh systematic uniform subsample drawn each run, drawing")
    lines.append("    exactly as many packets as its paired Poisson-adaptive method retained that run (or, where")
    lines.append("    that method found no valid subsample, the minimum sample size it was required to reach).")
    lines.append("    Equal sample size is the point: it makes the pair a verdict on *which* packets each rule")
    lines.append("    picks, not on how many. Matching is exact run by run; the two n_pkts columns therefore")
    lines.append("    agree at any k where the method found a subsample on every run. Where it failed on some")
    lines.append("    runs (n_samp < N) the uniform column averages the matched runs *together with* the")
    lines.append("    fallback runs, so its aggregate n_pkts sits below the method's own -- not a mismatch.")
    lines.append("  * relEMD = EMD relative to mean queuing delay (EMD / mean ground-truth path delay) --")
    lines.append("    dimensionless, and unlike the raw ns figure it stays comparable across offered loads")
    lines.append("    (raw EMD grows with the delay level).")
    lines.append("  * monitor_window: length of the window the method's samples were actually drawn from,")
    lines.append("    per run. For a growing-window method (see Utils._growing_window_search) this is the")
    lines.append("    result: the shortest prefix of the steady period, in {:g} ms steps, that could supply".format(
        (results.get('growing_window_step_ns') or GROWING_WINDOW_STEP_NS) / 1e6))
    lines.append("    the minimum required samples. For every other method it is the full steady window.")
    if any(m in GROWING_WINDOW_SUBSAMPLING_METHODS for m in methods):
        lines.append("  * A growing-window method's whole comparison is computed INSIDE that window: the")
        lines.append("    ground truth is rebuilt over it (so its EMD/relEMD/percentile errors are measured")
        lines.append("    against the delay distribution that held while the samples were collected, and")
        lines.append("    relEMD/relative percentile errors are divided by that window's own reference), the")
        lines.append("    switch-side Poisson probe is redrawn over it at the same probing RATE (so a shorter")
        lines.append("    window gets proportionally fewer observations, a larger epsilon and hence a larger")
        lines.append("    required sample size -- the real cost of stopping early), and its rate-matched")
        lines.append("    uniform baseline and ideal Poisson probe are restricted to it too. Nothing in such")
        lines.append("    a row mixes statistics from two different windows.")
    lines.append("  * IDC(1RTT)/burst_dur/burst_gap (all-packets table only): burstiness of the all-packets")
    lines.append("    SentTime arrival process -- IDC(1RTT) = Var/Mean of arrival counts in non-overlapping")
    lines.append("    windows one RTT ({:g}ns) wide (1.0 = Poisson-like, >1 = bursty); a burst is consecutive".format(ONE_RTT_NS))
    lines.append("    arrivals no farther apart than the burst gap threshold above (see detect_bursts).")
    lines.append("    burst_dur is the mean duration of MULTI-packet bursts only (a lone arrival is a")
    lines.append("    degenerate size-1/zero-duration \"burst\", not a real one, and excluding it matters --")
    lines.append("    on real traffic most bursts are lone arrivals, e.g. ~95-98% observed on Google_AllRPC,")
    lines.append("    so including them would mostly measure how rare real bursts are rather than how long")
    lines.append("    one lasts; n/a when there are no multi-packet bursts at all). burst_gap is the mean")
    lines.append("    gap between bursts of any size, all-packets, same fixed packet set every run like EMD(all).")
    lines.append("  * pass(...): fraction of runs where the delay consistency check passed. For pass(all)")
    lines.append("    and the uniform tables, this is out of all N runs. For a Poisson-adaptive method, this")
    lines.append("    is out of only the 'n_samp' runs that actually found a valid subsample at that flow")
    lines.append("    count (a run that found none neither passed nor failed).")
    lines.append("  * mean_diff = switch samples mean delay - packet-side mean delay (ns); this is the signed")
    lines.append("    quantity the consistency check thresholds (abs(mean_diff) <= epsilon bound).")
    lines.append("  * mean_delay = the family's own mean queuing delay (ns), next to 'reference mean' =")
    lines.append("    the mean of the reconstructed ground truth. The EMD columns say how far a family's")
    lines.append("    whole distribution is from the truth; these say what its headline number is, which")
    lines.append("    is the quantity the consistency check thresholds (mean_diff is this minus the")
    lines.append("    switch side's own mean).")
    lines.append("  * err_bound = that epsilon bound itself, relative to the switch-side mean delay (see")
    lines.append("    Utils.delay_consistency_error_bound): MaxEpsilonDelay + eta*e2eStd/(sqrt(n)*mean), at")
    lines.append("    each family's own realized n. The minimum required sample size is *defined* as the n")
    lines.append("    that brings this to exactly the configured guarantee{}, so a family holding".format(
        " ({:.0%})".format(results['delay_consistency_guarantee'])
        if results.get('delay_consistency_guarantee') else ""))
    lines.append("    exactly the minimum lands exactly there; anything below it holds more samples than")
    lines.append("    the minimum (or hit the MINIMUM_E2E_SAMPLE_SIZE floor) and so is claiming a tighter")
    lines.append("    guarantee than configured. Nothing should be above it.")
    if percentiles:
        lines.append("  * Percentile-error section at the end: signed ground-truth p_q minus family p_q, so")
        lines.append("    POSITIVE means the family understates that percentile (missing tail delay the ground")
        lines.append("    truth has) and negative means it overstates it. Reported absolute (ns) and relative")
        lines.append("    to the ground truth's own p_q. The EMD is one number for the whole distribution, so")
        lines.append("    a family can score well on it while still misplacing the tail -- which is the part")
        lines.append("    delay SLOs are written against. Not covered by the consistency check, which is a")
        lines.append("    test on the mean only.")
    lines.append("")

    lines.append("All packets of the considered flows:")
    burstiness_all = results.get('burstiness_all_packets') or {}
    bound_all_by_run = results.get('error_bound_all_packets_by_run') or []
    delay_mean_all_by_run = results.get('delay_mean_all_packets_by_run') or []
    gt_delay_mean_by_run = results.get('groundtruth_delay_mean_by_run') or []
    header = "{:>3} | {:>24} | {:>24} | {:>9} | {:>24} | {:>22} | {:>22} | {:>22} | {:>10} | {:>14} | {:>14}".format(
        "k", "EMD(all) [ns]", "relEMD(all)", "pass(all)", "mean_diff(all) [ns]",
        "err_bound(all)", "mean_delay(all) [ns]", "reference mean [ns]",
        "IDC(1RTT)", "burst_dur[ns]", "burst_gap[ns]")
    lines.append(header)
    lines.append("-" * len(header))
    for i, k in enumerate(num_flows_display):
        if all_packets_is_aggregated:
            emd_all_str = _stat(emd_all_by_experiment[i])
            emd_all_norm_str = _stat(emd_all_by_experiment_norm[i], fmt="{:.4f}")
        else:
            emd_all = results['emd_all_packets'][i]
            emd_all_str = "{:.2f}".format(emd_all) if emd_all == emd_all else "n/a"
            emd_all_norm = results['emd_all_packets_normalized'][i]
            emd_all_norm_str = "{:.4f}".format(emd_all_norm) if emd_all_norm == emd_all_norm else "n/a"

        def _burst_field(field):
            values = burstiness_all.get(field, [])
            v = values[i] if i < len(values) else float('nan')
            return "{:.3f}".format(v) if v == v else "n/a"

        lines.append("{:>3} | {:>24} | {:>24} | {:>8.0%} | {:>24} | {:>22} | {:>22} | {:>22} | {:>10} | {:>14} | {:>14}".format(
            k, emd_all_str, emd_all_norm_str, results['pass_rate_all_packets'][i],
            _stat(results['mean_diff_all_packets_by_run'][i]),
            _stat(bound_all_by_run[i] if i < len(bound_all_by_run) else [], fmt="{:.4f}"),
            _stat(delay_mean_all_by_run[i] if i < len(delay_mean_all_by_run) else []),
            _stat(gt_delay_mean_by_run[i] if i < len(gt_delay_mean_by_run) else []),
            _burst_field('idc_1rtt'), _burst_field('avg_burst_duration_ns'),
            _burst_field('avg_burst_interarrival_ns'),
        ))

    emd_sampled_by_run = results['emd_sampled_packets_by_run']
    emd_sampled_by_run_norm = results['emd_sampled_packets_by_run_normalized']
    sizes_sampled = results.get('sample_sizes_sampled_by_run', {})
    windows_sampled = results.get('window_duration_sampled_by_run', {})
    bounds_sampled = results.get('error_bound_sampled_by_run', {})
    bounds_uniform = results.get('error_bound_uniform_by_run', {})
    bounds_oracle = results.get('error_bound_oracle_by_run', {})
    means_sampled = results.get('delay_mean_sampled_by_run', {})
    means_uniform = results.get('delay_mean_uniform_by_run', {})
    means_oracle = results.get('delay_mean_oracle_by_run', {})
    n_k = len(results['num_flows'])
    for method in methods:
        lines.append("")
        lines.append("Poisson-adaptive subsample -- {}:".format(method))
        header = "{:>3} | {:>6} | {:>24} | {:>24} | {:>9} | {:>24} | {:>20} | {:>22} | {:>22} | {:>22}".format(
            "k", "n_samp", "EMD [ns]", "relEMD", "pass", "mean_diff [ns]", "n_pkts",
            "err_bound", "mean_delay [ns]", "monitor_window [ms]")
        lines.append(header)
        lines.append("-" * len(header))
        for i, k in enumerate(num_flows_display):
            window_values = windows_sampled.get(method, [[]] * n_k)[i]
            lines.append("{:>3} | {:>6} | {:>24} | {:>24} | {:>8.0%} | {:>24} | {:>20} | {:>22} | {:>22} | {:>22}".format(
                k, len(emd_sampled_by_run[method][i]),
                _stat(emd_sampled_by_run[method][i]),
                _stat(emd_sampled_by_run_norm[method][i], fmt="{:.4f}"),
                results['pass_rate_sampled'][method][i],
                _stat(results['mean_diff_sampled_by_run'][method][i]),
                _stat(sizes_sampled.get(method, [[]] * n_k)[i], fmt="{:.0f}"),
                _stat(bounds_sampled.get(method, [[]] * n_k)[i], fmt="{:.4f}"),
                _stat(means_sampled.get(method, [[]] * n_k)[i]),
                _stat(np.asarray(window_values, dtype=float) / 1e6, fmt="{:.2f}"),
            ))

    emd_uniform_by_run = results.get('emd_uniform_packets_by_run', {})
    emd_uniform_by_run_norm = results.get('emd_uniform_packets_by_run_normalized', {})
    pass_rate_uniform = results.get('pass_rate_uniform', {})
    diff_uniform_by_run = results.get('mean_diff_uniform_packets_by_run', {})
    sizes_uniform = results.get('sample_sizes_uniform_by_run', {})
    emd_oracle_by_run = results.get('emd_oracle_by_run', {})
    emd_oracle_by_run_norm = results.get('emd_oracle_by_run_normalized', {})
    pass_rate_oracle = results.get('pass_rate_oracle', {})
    diff_oracle_by_run = results.get('mean_diff_oracle_by_run', {})
    sizes_oracle = results.get('sample_sizes_oracle_by_run', {})
    for key in oracle_series:
        lines.append("")
        lines.append("{}:".format(_oracle_series_label(key)))
        header = "{:>3} | {:>6} | {:>24} | {:>24} | {:>9} | {:>24} | {:>20} | {:>22} | {:>22}".format(
            "k", "n_runs", "EMD [ns]", "relEMD", "pass", "mean_diff [ns]", "n_pkts", "err_bound",
            "mean_delay [ns]")
        lines.append(header)
        lines.append("-" * len(header))
        for i, k in enumerate(num_flows_display):
            lines.append("{:>3} | {:>6} | {:>24} | {:>24} | {:>8.0%} | {:>24} | {:>20} | {:>22} | {:>22}".format(
                k, len(emd_oracle_by_run[key][i]),
                _stat(emd_oracle_by_run[key][i]),
                _stat(emd_oracle_by_run_norm[key][i], fmt="{:.4f}"),
                pass_rate_oracle[key][i],
                _stat(diff_oracle_by_run[key][i]),
                _stat(sizes_oracle.get(key, [[]] * len(results['num_flows']))[i], fmt="{:.0f}"),
                _stat(bounds_oracle.get(key, [[]] * len(results['num_flows']))[i], fmt="{:.4f}"),
                _stat(means_oracle.get(key, [[]] * len(results['num_flows']))[i]),
            ))
    for key in uniform_series:
        lines.append("")
        lines.append("{}:".format(_uniform_series_label(key)))
        header = "{:>3} | {:>6} | {:>24} | {:>24} | {:>9} | {:>24} | {:>20} | {:>22} | {:>22}".format(
            "k", "n_samp", "EMD [ns]", "relEMD", "pass", "mean_diff [ns]", "n_pkts", "err_bound",
            "mean_delay [ns]")
        lines.append(header)
        lines.append("-" * len(header))
        for i, k in enumerate(num_flows_display):
            lines.append("{:>3} | {:>6} | {:>24} | {:>24} | {:>8.0%} | {:>24} | {:>20} | {:>22} | {:>22}".format(
                k, len(emd_uniform_by_run[key][i]),
                _stat(emd_uniform_by_run[key][i]),
                _stat(emd_uniform_by_run_norm[key][i], fmt="{:.4f}"),
                pass_rate_uniform[key][i],
                _stat(diff_uniform_by_run[key][i]),
                _stat(sizes_uniform.get(key, [[]] * len(results['num_flows']))[i], fmt="{:.0f}"),
                _stat(bounds_uniform.get(key, [[]] * len(results['num_flows']))[i], fmt="{:.4f}"),
                _stat(means_uniform.get(key, [[]] * len(results['num_flows']))[i]),
            ))

    poisson_series = results.get('poisson_test_series') or []
    if poisson_series:
        lines.append("")
        lines.append("=" * 70)
        lines.append("Poisson-ness of the NON-Poissonized families' own sampling instants")
        lines.append("=" * 70)
        lines.append("  Anderson-Darling (AD) on the inter-arrival gaps, and the multi-lag chi-squared")
        lines.append("  independence test -- the same two criteria the Poisson-adaptive samplers must")
        lines.append("  satisfy. All packets and the fixed-rate uniform subsets never had to pass")
        lines.append("  anything, so this says whether their instants happen to look Poisson anyway,")
        lines.append("  which is the premise PASTA needs before their sample mean can stand in for a")
        lines.append("  time average. 'pass' columns are the fraction of runs (for all-packets: of")
        lines.append("  experiments, its instants being the same fixed set every run) that passed.")
        if not results.get('run_chi_squared_test'):
            lines.append("  NOTE: the chi-squared test was not run for this result; only AD is reported.")
        for key in poisson_series:
            if key == 'all_packets':
                label = 'All packets of considered flows'
                get = lambda i: (_all_packets_test_flags(results, i, 'ad'),
                                  _all_packets_test_flags(results, i, 'ad_chi'))
            else:
                record = (results.get('uniform_test_split_by_run') or {}).get(key[1])
                if record is None:
                    continue
                label = _uniform_series_label(key[1])
                get = lambda i, record=record: (
                    [poisson_test_outcome({'ad_pass': a, 'chi_pass': c}, 'ad')
                      for a, c in zip(record['ad_pass'][i], record['chi_pass'][i])],
                    [poisson_test_outcome({'ad_pass': a, 'chi_pass': c}, 'ad_chi')
                      for a, c in zip(record['ad_pass'][i], record['chi_pass'][i])])
            lines.append("")
            lines.append("{}:".format(label))
            header = "{:>3} | {:>6} | {:>14} | {:>18}".format("k", "n", "pass(AD)", "pass(AD + chi2)")
            lines.append(header)
            lines.append("-" * len(header))
            for i, k in enumerate(num_flows_display):
                ad_flags, both_flags = get(i)
                ad_known = [f for f in ad_flags if f is not None]
                both_known = [f for f in both_flags if f is not None]
                lines.append("{:>3} | {:>6} | {:>14} | {:>18}".format(
                    k, len(ad_flags),
                    "{:.0%}".format(sum(ad_known) / len(ad_known)) if ad_known else "n/a",
                    "{:.0%}".format(sum(both_known) / len(both_known)) if both_known else "n/a",
                ))

    # Short column tags keep one row per flow count readable with every family side by
    # side; used by both the sparse-percentile-error table below (when delay_percentiles
    # is non-empty) and the dense-percentile-grid EMD stand-ins after it (which need no
    # such guard, since they're independent of delay_percentiles). The legend maps them
    # back to full family names.
    families = [('all', 'all_packets', 'all packets of considered flows')]
    for j, method in enumerate(methods, start=1):
        families.append(('P{}'.format(j), ('sampled', method),
                          'Poisson-adaptive subsample, {}'.format(method)))
    for j, key in enumerate(uniform_series, start=1):
        families.append(('U{}'.format(j), ('uniform', key), _uniform_series_label(key)))
    for j, key in enumerate(oracle_series, start=1):
        families.append(('I{}'.format(j), ('oracle', key), _oracle_series_label(key)))

    if percentiles:
        lines.append("")
        lines.append("=" * 70)
        lines.append("Percentile (tail-shape) error: ground-truth p_q  -  family p_q")
        lines.append("=" * 70)
        for tag, _key, label in families:
            lines.append("  {:>4} = {}".format(tag, label))

        for q in percentiles:
            for kind, unit, fmt in (('percentile_diff', 'ns', "{:.1f}"),
                                     ('percentile_reldiff', 'relative to ground-truth p{}'.format(q), "{:.1%}")):
                lines.append("")
                lines.append("p{} error [{}]  (positive = family understates the percentile):".format(q, unit))
                header = "{:>3}".format("k") + "".join(" | {:>18}".format(tag) for tag, _, _ in families)
                lines.append(header)
                lines.append("-" * len(header))
                for i, k in enumerate(num_flows_display):
                    row = "{:>3}".format(k)
                    for _tag, key, _label in families:
                        values, _ = _load_plot_series_values(results, i, key, metric=(kind, q))
                        values = np.asarray(values, dtype=float).reshape(-1)
                        values = values[np.isfinite(values)]
                        if values.size == 0:
                            cell = "n/a"
                        elif values.size == 1:
                            cell = fmt.format(values[0])
                        else:
                            cell = (fmt + " +/- " + fmt).format(np.mean(values), np.std(values))
                        row += " | {:>18}".format(cell)
                    lines.append(row)

    if results.get('percentile_avg_relerror_all_packets'):
        lines.append("")
        lines.append("=" * 70)
        lines.append("Mean absolute relative percentile error (dense percentile grid)")
        lines.append("=" * 70)
        lines.append("  * For each of a dense, fixed set of percentiles (p1..p99, independent of")
        lines.append("    delay_percentiles above), |ground-truth p_q - family p_q| / ground-truth")
        lines.append("    p_q, averaged over that whole grid -- see percentile_avg_relative_error.")
        lines.append("    A percentile-based EMD analog (average |diff| over the same grid,")
        lines.append("    without the relative-to-its-own-percentile scaling) was considered and")
        lines.append("    deliberately dropped: it only approximates the real EMD/relEMD above at")
        lines.append("    finite grid resolution, at the same O(n log n) cost as computing the")
        lines.append("    exact value directly, so relEMD remains the metric of record.")
        for kind, normalized, label, fmt in (
                ('percentile_avg_relerror', False, 'mean absolute relative percentile error', "{:.1%}"),):
            lines.append("")
            lines.append("{}:".format(label))
            header = "{:>3}".format("k") + "".join(" | {:>18}".format(tag) for tag, _, _ in families)
            lines.append(header)
            lines.append("-" * len(header))
            for i, k in enumerate(num_flows_display):
                row = "{:>3}".format(k)
                for _tag, key, _label in families:
                    values, _ = _load_plot_series_values(results, i, key, normalized=normalized, metric=kind)
                    values = np.asarray(values, dtype=float).reshape(-1)
                    values = values[np.isfinite(values)]
                    if values.size == 0:
                        cell = "n/a"
                    elif values.size == 1:
                        cell = fmt.format(values[0])
                    else:
                        cell = (fmt + " +/- " + fmt).format(np.mean(values), np.std(values))
                    row += " | {:>18}".format(cell)
                lines.append(row)

    # Loss and ECN marking get their own sections, with the same family tables as delay.
    for metric in PROB_METRIC_KEYS:
        lines.extend(_prob_metric_text_section(results, metric, num_flows_display, _stat))

    with open(output_path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    return output_path


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
        # Equivalent to histogramming tt into nbins bins of width Delta and computing
        # mean(counts==0) / mean(counts), but without ever materializing an nbins-sized
        # array: for small Delta, nbins can reach into the millions (this loop tries up
        # to ~3000 candidate Deltas), so np.linspace/np.histogram over nbins dominated the
        # cost. Both quantities are exactly recoverable from just the number of *distinct*
        # occupied bins, which is at most len(tt) regardless of nbins: mean(counts) is
        # always len(tt)/nbins (every point falls in exactly one bin), and mean(counts==0)
        # is 1 - (occupied bins)/nbins.
        bin_ids = np.floor(tt / Delta).astype(np.int64)
        bin_ids = np.clip(bin_ids, 0, nbins - 1)
        occupied_bins = np.unique(bin_ids).size

        p0_hat = 1.0 - occupied_bins / nbins
        mu_hat = tt.size / nbins

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
    # count_path = 0
    for file in glob.glob(file_path + '*EndToEnd_packets.csv'):
        df = pd.read_csv(file)
        if 'PayloadSize' in df.columns:
            sum_size += df['PayloadSize'].sum()
            count += df['PayloadSize'].count()
            # count_path += len(df[df['Path'] == 0])
    average_packet_size = sum_size / count if count > 0 else 0
    return average_packet_size

def infer_alternative_routes(file_path):
    """Infer [other racks, hosts per rack] from recorded queue filenames."""
    racks = set()
    hosts_by_rack = defaultdict(set)
    for path in glob.glob(file_path + 'T*_PoissonSampler_queueSize.csv'):
        queue_name = os.path.basename(path).split('_', 1)[0]
        tor_agg = re.fullmatch(r'T(\d+)A(\d+)', queue_name)
        tor_host = re.fullmatch(r'T(\d+)H(\d+)', queue_name)
        if tor_agg:
            racks.add(int(tor_agg.group(1)))
        elif tor_host:
            rack = int(tor_host.group(1))
            racks.add(rack)
            hosts_by_rack[rack].add(int(tor_host.group(2)))

    host_counts = {len(hosts) for hosts in hosts_by_rack.values()}
    if len(racks) < 2 or len(host_counts) != 1:
        raise ValueError(
            'Could not infer a uniform rack/host topology from {}'.format(file_path)
        )
    return [len(racks) - 1, host_counts.pop()]


def compute_bias_based_on_average_packet_size(sampling_results, average_packet_size, queue_names, linkRates, alternative_routes):
    if len(alternative_routes) != 2 or any(value <= 0 for value in alternative_routes):
        raise ValueError('alternative_routes must be [number_of_racks - 1, hosts_per_rack]')
    queue_names, _, linkRates = sort_queues_by_path(queue_names, [None, None, None, None], linkRates)
    
    for queue_name in queue_names:
        idx = queue_names.index(queue_name)
        sampling_results[queue_name+'NPkts'] = sampling_results[queue_name+'e2e_samples_queue_delay_mean'] * linkRates[idx] / (average_packet_size * 8)
        sampling_results[queue_name+'NBytes'] = sampling_results[queue_name+'NPkts'] * average_packet_size
        if idx == 0:
            sampling_results[queue_name+'split_ratio'] = 1.0
            continue
        # sampling_results[queue_name+'bias'] = sampling_results[queue_names[idx - 1]+'poisson_prob_non_empty'] * average_packet_size * 8 / linkRates[idx] * (1 / alternative_routes[idx - 1])
        if idx == 1:
        #     sampling_results[queue_name+'bias'] = sampling_results[queue_names[idx - 1]+'poisson_prob_non_empty'] * average_packet_size * 8 / linkRates[idx] * (1 / alternative_routes[idx - 1])
            sampling_results[queue_name+'bias'] = sampling_results[queue_names[idx - 1]+'poisson_prob_non_empty'] * average_packet_size * 8 / linkRates[idx] * (sampling_results[queue_name+ 'packets_of_interest'] / sampling_results[queue_names[idx - 1]+ 'packets_of_interest'])
            sampling_results[queue_name+'split_ratio'] = sampling_results[queue_name+ 'packets_of_interest'] / sampling_results[queue_names[idx - 1]+ 'packets_of_interest']
        if idx == 2:
            sampling_results[queue_name+'bias'] = sampling_results[queue_names[idx - 2]+'poisson_prob_non_empty'] * average_packet_size * 8 / linkRates[idx] * (1 / alternative_routes[idx - 1]) * (1 / alternative_routes[idx - 2])
            sampling_results[queue_name+'bias'] += sampling_results[queue_names[idx - 2]+'poisson_prob_non_empty'] * sampling_results[queue_names[idx - 1]+'poisson_prob_non_empty'] * average_packet_size * 8 / linkRates[idx] * (1 / alternative_routes[idx - 1]) * (1 - 1 / alternative_routes[idx - 2])
            sampling_results[queue_name+'bias'] += sampling_results[queue_names[idx - 1]+'poisson_prob_non_empty'] * (1 - sampling_results[queue_names[idx - 2]+'poisson_prob_non_empty']) * average_packet_size * 8 / linkRates[idx] * (1 / alternative_routes[idx - 1])
            sampling_results[queue_name+'split_ratio'] = 1 / alternative_routes[idx - 1]
        sampling_results[queue_name+'e2e_vs_poisson_consistent_with_bias'] = int(abs(sampling_results[queue_name+'e2e_samples_queue_delay_mean'] - (sampling_results[queue_name+'poisson_samples_queue_delay_mean'] + sampling_results[queue_name+'bias'])) <= sampling_results[queue_name+'error_bound'])
    
    return sampling_results

def calculate_offline_delay_bias_DC(__ns3_path, rate, experiment, results_folder, steadyStart, steadyEnd, linkRates=[], linkDelays=[], 
                                    swtichDstREDQueueDiscMaxSize=[0], tsh=0.15, differentiationDelay=None, errorRate=None, load=None, 
                                    queue_names=[], flow_names=[], e2e_intervals=10000, sampling_factor=None,
                                    average_packet_size=None, alternative_routes=None, source_rack=None):
    if differentiationDelay is not None and errorRate is not None:
        file_path = '{}/scratch/{}/{}/{}/D_{}/f_{}/{}/'.format(__ns3_path, results_folder, rate, load, differentiationDelay, errorRate, experiment)
    else:
        file_path = '{}/scratch/{}/{}/{}/{}/'.format(__ns3_path, results_folder, rate, load, experiment)

    times = np.array(np.cumsum(np.random.exponential(e2e_intervals, size=int((steadyEnd - steadyStart) // e2e_intervals))) + steadyStart, dtype=np.int64)
    if source_rack is None and flow_names:
        source_match = re.match(r'^R(\d+)H\d+', flow_names[0])
        if source_match:
            source_rack = int(source_match.group(1))
    # (_, queue_size_samples, _, queue_delay_samples_poisson_e2e), res = sample_total_queue_size(times, queue_names, file_path, linkDelays, linkRates, np.array(swtichDstREDQueueDiscMaxSize[1:], dtype=float) * tsh)
    res = {}
    (_, _, queue_ECN_samples_poisson_e2e, queue_delay_samples_poisson_e2e, queue_success_prob_samples_poisson_e2e), res = sample_total_queue_size_non_combined(res, times, queue_names, file_path, linkDelays, linkRates, np.array(swtichDstREDQueueDiscMaxSize[1:], dtype=float) * tsh, swtichDstREDQueueDiscMaxSize[1:], path_observation=True, sampling_factor=sampling_factor, source_rack=source_rack)
    (_, _, _, _, _), res = sample_total_queue_size_non_combined(res, times, queue_names, file_path, linkDelays, linkRates, np.array(swtichDstREDQueueDiscMaxSize[1:], dtype=float) * tsh, swtichDstREDQueueDiscMaxSize[1:], path_observation=False)
    res = combine_sampling_results(res, queue_names)
    if average_packet_size is None:
        average_packet_size = compute_average_packet_size(file_path)
    if alternative_routes is None:
        alternative_routes = infer_alternative_routes(file_path)
    res = compute_bias_based_on_average_packet_size(
        res, average_packet_size, queue_names, linkRates, alternative_routes
    )

    res['sum_poisson_samples_queue_delay_mean'] = sum([res[queue_name+'poisson_samples_queue_delay_mean'] for queue_name in queue_names])
    res['sum_poisson_samples_queue_success_prob_mean'] = np.prod(np.array([res[queue_name+'poisson_samples_queue_success_prob_mean'] for queue_name in queue_names]), axis=0)
    res['sum_poisson_samples_queue_nonmarking_prob_mean'] = np.prod(np.array([res[queue_name+'poisson_samples_queue_nonmarking_prob_mean'] for queue_name in queue_names]), axis=0)
    # sum_poisson_samples_queue_delay_std = sum([res[queue_name+'poisson_samples_queue_delay_std'] for queue_name in queue_names])
    res['e2e_poisson_samples_queue_delay_mean'] = np.mean(queue_delay_samples_poisson_e2e)
    res['e2e_poisson_samples_queue_success_prob_mean'] = np.mean(queue_success_prob_samples_poisson_e2e)
    res['e2e_poisson_samples_queue_nonmarking_prob_mean'] = np.mean(1 - queue_ECN_samples_poisson_e2e)
    res['e2e_poisson_samples_queue_delay_std'] = np.std(queue_delay_samples_poisson_e2e)
    res['e2e_poisson_samples_queue_success_prob_std'] = np.std(queue_success_prob_samples_poisson_e2e)
    res['e2e_poisson_samples_queue_nonmarking_prob_std'] = np.std(1 - queue_ECN_samples_poisson_e2e)
    res['e2e_vs_sum_error_bound'] = 1.96 * res['sum_poisson_samples_queue_delay_mean'] * np.max([res[queue_name+'poisson_samples_queue_delay_std'] / (np.sqrt(res[queue_name+'poisson_samples_queue_delay_count']) * res[queue_name+'poisson_samples_queue_delay_mean']) for queue_name in queue_names])
    res['e2e_vs_sum_error_bound'] += 1.96 * res['e2e_poisson_samples_queue_delay_std'] / np.sqrt(len(queue_delay_samples_poisson_e2e))
    # res['e2e_vs_sum_error_bound'] += 1.96 * sum_poisson_samples_queue_delay_std / np.sqrt(len(queue_delay_samples_poisson_e2e))
    res['e2e_vs_sum_consistent'] = int(abs(res['e2e_poisson_samples_queue_delay_mean'] - res['sum_poisson_samples_queue_delay_mean']) <= res['e2e_vs_sum_error_bound'])
    bias = sum([res[queue_name+'bias'] for queue_name in queue_names])
    res['total_estimated_bias'] = bias
    res['e2e_vs_sum_consistent_with_bias'] = int(abs(res['e2e_poisson_samples_queue_delay_mean'] - (res['sum_poisson_samples_queue_delay_mean'] + bias)) <= res['e2e_vs_sum_error_bound'])

    X = 1.96 * np.max([res[queue_name+'poisson_samples_queue_success_prob_std'] / (np.sqrt(res[queue_name+'poisson_samples_queue_delay_count']) * res[queue_name+'poisson_samples_queue_success_prob_mean']) for queue_name in queue_names])
    Y = 1.96 * res['e2e_poisson_samples_queue_success_prob_std'] / np.sqrt(len(queue_success_prob_samples_poisson_e2e))
    res['e2e_vs_sum_error_success_prob_bound'] = [res['sum_poisson_samples_queue_success_prob_mean'] * (((1 + X) ** len(queue_names)) / (1 - Y) - 1), res['sum_poisson_samples_queue_success_prob_mean'] * (((1 - X) ** len(queue_names)) / (1 + Y) - 1)]
    res['e2e_vs_sum_consistent_success_prob'] = int((res['e2e_poisson_samples_queue_success_prob_mean'] - res['sum_poisson_samples_queue_success_prob_mean'] <= res['e2e_vs_sum_error_success_prob_bound'][0]) and (res['e2e_poisson_samples_queue_success_prob_mean'] - res['sum_poisson_samples_queue_success_prob_mean'] >= res['e2e_vs_sum_error_success_prob_bound'][1]))

    X = 1.96 * np.max([res[queue_name+'poisson_samples_queue_nonmarking_prob_std'] / (np.sqrt(res[queue_name+'poisson_samples_queue_delay_count']) * res[queue_name+'poisson_samples_queue_nonmarking_prob_mean']) for queue_name in queue_names])
    Y = 1.96 * res['e2e_poisson_samples_queue_nonmarking_prob_std'] / np.sqrt(len(queue_ECN_samples_poisson_e2e))
    res['e2e_vs_sum_error_nonmarking_prob_bound'] = [res['sum_poisson_samples_queue_nonmarking_prob_mean'] * (((1 + X) ** len(queue_names)) / (1 - Y) - 1), res['sum_poisson_samples_queue_nonmarking_prob_mean'] * (((1 - X) ** len(queue_names)) / (1 + Y) - 1)]
    res['e2e_vs_sum_consistent_nonmarking_prob'] = int((res['e2e_poisson_samples_queue_nonmarking_prob_mean'] - res['sum_poisson_samples_queue_nonmarking_prob_mean'] <= res['e2e_vs_sum_error_nonmarking_prob_bound'][0]) and (res['e2e_poisson_samples_queue_nonmarking_prob_mean'] - res['sum_poisson_samples_queue_nonmarking_prob_mean'] >= res['e2e_vs_sum_error_nonmarking_prob_bound'][1]))

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
            # df_res = calculate_offline_E2E_lossRates_DC(full_df, df_res, checkColumn, txDelay_to_firstQ,
            #                                       '{}/scratch/{}/{}/{}/{}/'.format(__ns3_path, results_folder, rate, load, experiment), 
            #                                       passiveProbe, samplingMethod, steadyStart, steadyEnd, samples_paths_aggregated_statistics[df_name], queue_names, linkDelays, linkRates, 
            #                                       np.array(swtichDstREDQueueDiscMaxSize, dtype=float) * tsh)
        
            df_res = calculate_offline_E2E_delays(full_df, removeDrops, checkColumn, txDelay_to_firstQ, df_res, 
                                                  '{}/scratch/{}/{}/{}/{}/'.format(__ns3_path, results_folder, rate, load, experiment), 
                                                  passiveProbe, samplingMethod, steadyStart, steadyEnd, samples_paths_aggregated_statistics[df_name], queue_names, linkDelays, linkRates, 
                                                  np.array(swtichDstREDQueueDiscMaxSize, dtype=float) * tsh,
                                                  flow_name=df_name)
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
            # avgPacktSize = 1500 if "Nagle" in results_folder.split('/')[0] else packets_cfd.compute_average_packet_size_from_cdf()
            # Keep up to last / in the file path to get the directory
            avgPacktSize = compute_average_packet_size(file_path.rsplit('/', 1)[0] + '/')
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

# Hard floor on the minimum required e2e sample size: below this, the consistency check's
# bound is so loose it passes almost by construction rather than by evidence, so we never
# target fewer samples than this regardless of what the CV-based formula below computes.
MINIMUM_E2E_SAMPLE_SIZE = 100

def calc_min_e2e_samples(confidenceValue, maxError, samples_paths_aggregated_statistics, metric='Delay'):
    if samples_paths_aggregated_statistics['MaxEpsilon' + metric] >= maxError:
        print(f"Warning: MaxEpsilon{metric} is greater than or equal to maxError. Cannot achieve the desired confidence level with the current data.")
        return None
    if samples_paths_aggregated_statistics['' + metric + 'Mean'] == 0:
        print(f"Warning: Mean {metric} is zero. Cannot calculate the required sample size. Picking the default of {MINIMUM_E2E_SAMPLE_SIZE} samples.")
        return MINIMUM_E2E_SAMPLE_SIZE
    computed = int(((confidenceValue * samples_paths_aggregated_statistics['e2e' + metric + 'Std']) / ((maxError - samples_paths_aggregated_statistics['MaxEpsilon' + metric]) * samples_paths_aggregated_statistics['' + metric + 'Mean'])) ** 2)
    if computed < MINIMUM_E2E_SAMPLE_SIZE:
        print(f"Warning: computed minimum required {metric} sample size ({computed}) is below the floor of "
              f"{MINIMUM_E2E_SAMPLE_SIZE}; using {MINIMUM_E2E_SAMPLE_SIZE} instead.")
        return MINIMUM_E2E_SAMPLE_SIZE
    return computed

def calc_min_e2e_samples_prob(confidenceValue, maxError, samples_paths_aggregated_statistics, number_of_segments, metric='SuccessProb'):
    mean_key = metric + 'Mean'
    epsilon_key = 'MaxEpsilon' + metric
    std_key = 'e2e' + metric + 'Std'

    if maxError <= 0 or maxError >= 1:
        print(f"Warning: maxError must be a relative error in (0, 1) for {metric}.")
        return None

    if samples_paths_aggregated_statistics[epsilon_key] >= 1:
        print(f"Warning: MaxEpsilon{metric} is invalid for probability sampling.")
        return None

    mean_value = float(np.exp(samples_paths_aggregated_statistics[mean_key]))
    if mean_value <= 0:
        print(f"Warning: Mean {metric} is zero. Cannot calculate the required sample size.")
        return None

    epsp_from_upper = 1 - (((1 + samples_paths_aggregated_statistics[epsilon_key]) ** number_of_segments) / (1 + maxError))
    epsp_from_lower = (((1 - samples_paths_aggregated_statistics[epsilon_key]) ** number_of_segments) / (1 - maxError)) - 1
    print(f"Calculated epsp_from_upper: {epsp_from_upper}, epsp_from_lower: {epsp_from_lower} for metric: {metric}")
    epsp = min(epsp_from_upper, epsp_from_lower)
    if epsp <= 0:
        print(f"Warning: Total error budget leaves no room for {metric} sampling error.")
        return None

    return int(np.ceil(((confidenceValue * samples_paths_aggregated_statistics[std_key]) / (mean_value * epsp)) ** 2))

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
