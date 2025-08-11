import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import anderson
from scipy.stats import f_oneway, kruskal
from scipy.stats import expon, bernoulli, ks_2samp

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

def calculate_offline_E2E_lossRates(__ns3_path, full_df, df_res, checkColumn, txDelay, linksRate, swtichDstREDQueueDiscMaxSize, df_name):
    df_res['successProb'] = {}
    for var in ['event', 'probability']:
        for method in ['rightCont_timeAvg', 'leftCont_timeAvg', 'linearInterp_timeAvg', 'poisson_eventAvg', 'eventAvg']:
            df_res['successProb'][var + '_' + method] = {}

    packets_cfd = PacketCDF()
    packets_cfd.load_cdf_data('{}/scratch/ECNMC/Helpers/packet_size_cdf_singleQueue.csv'.format(__ns3_path))
    df_res['sampleSize']['successProb'] = {}
    full_df_ = full_df[full_df['SentTime'] != -1].copy()
    for path in full_df_['Path'].unique():
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
   
        samples_times = find_samples_path(time, txDelay, df_res['RTT'][path], df_name)
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


def find_samples_path_new(time, txDelay, maxLength):
    # Step 1: Compute interarrival times
    interarrival = np.diff(time)

    # Step 2: Check if the entire sequence is exponential
    anderson_statistic, anderson_critical_values, _ = anderson(interarrival, 'expon')
    if anderson_statistic <= anderson_critical_values[2]:
        print("Interarrival times are exponentially distributed.")
        return time

    # Step 3: Custom binning by txDelay and maxLength
    bins = []
    i = 0
    n = len(time)

    while i < n:
        bin_start = i
        bin_times = [time[i]]
        i += 1
        while i < n:
            gap = time[i] - time[i - 1]
            span = time[i] - bin_times[0]
            if gap > txDelay:
                break
            bin_times.append(time[i])
            i += 1
        bins.append(np.array(bin_times))

    # Step 4: Try random sampling from bins + exponential test
    max_sample_size = 0
    best_sample = None

    for brnval in [0.1, 0.15, 0.2]:
        tries = 20
        while tries > 0:
            selected_indices = []
            for bin_times in bins:
                if len(bin_times) > 0:
                    chosen = np.random.choice(bin_times)
                    selected_indices.append(chosen)
            selected_times = np.array(sorted(selected_indices))

            if len(selected_times) <= 1:
                tries -= 1
                continue

            # First check: directly test selected times
            selected_interarrival = np.diff(selected_times)
            anderson_statistic, anderson_critical_values, _ = anderson(selected_interarrival, 'expon')
            if anderson_statistic <= anderson_critical_values[2]:
                if len(selected_times) > max_sample_size:
                    max_sample_size = len(selected_times)
                    best_sample = selected_times
                    break

            # Second check: apply Bernoulli sampling
            keep_mask = bernoulli.rvs(brnval, size=len(selected_times))
            final_times = selected_times[keep_mask == 1]

            if len(final_times) <= 1:
                tries -= 1
                continue

            anderson_statistic, anderson_critical_values, _ = anderson(np.diff(final_times), 'expon')
            if anderson_statistic <= anderson_critical_values[2]:
                if len(final_times) > max_sample_size:
                    max_sample_size = len(final_times)
                    best_sample = final_times
                    break
            tries -= 1

    if best_sample is not None:
        return best_sample

    print("Failed to find exponentially distributed interarrival times after 20 tries.")
    return []


def find_samples_path(time, txDelay, avg_interarrival_=None, df_name=None):
    if "P0D0" in df_name:
        return time
    # return find_samples_path_new(time, txDelay, 480*1e3)  
    # Step 1: Compute interarrival times
    interarrival = np.diff(time)

    # Step 2: Check if interarrivals follow an exponential distribution
    # selection_mask = bernoulli.rvs(0.2, size=len(time))
    # temp_times = time[selection_mask == 1]
    # return temp_times
    # interarrival = np.diff(temp_times)
    # if len(interarrival) > 1:
    #     anderson_statistic, anderson_critical_values, _ = anderson(interarrival, 'expon')
    #     if anderson_statistic <= anderson_critical_values[4]:
    #         # print("Interarrival times are exponentially distributed.")
    #         return time
            # return temp_times
    
    # print("Interarrival times are not exponentially distributed. Proceeding with sampling...")

    # Step 3: Divide into chunks of average interarrival time
    # avg_interarrival = np.mean(interarrival) * 10
    avg_interarrival  = avg_interarrival_ * 5
    # avg_interarrival = np.mean(interarrival[interarrival > txDelay])
    # print("Average interarrival time:", avg_interarrival)
    start_time = time[0]
    end_time = time[-1]
    bins = np.arange(start_time, end_time, avg_interarrival)
    max_sample_size = 0
    max_sample_size_times = []
    # for brnval in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    for brnval in [0.1, 0.2, 0.3, 0.5]:
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
                    break
            tries -= 1
            # print("Tries left:", tries, "with brnval:", brnval)
    if max_sample_size:
        # print("Max sample size found:", max_sample_size, len(max_sample_size_times))
        return max_sample_size_times
    print("Failed to find exponentially distributed interarrival times after 20 tries.")
    return []

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

def calculate_offline_E2E_markingProb(full_df, df_res, checkColumn, txDelay, swtichDstREDQueueDiscMaxSize, linkRate, __ns3_path, tsh, df_name):
    # timeAvg_methods = ['rightCont_timeAvg', 'leftCont_timeAvg', 'linearInterp_timeAvg']
    # nonMarkingProb_timeAvg_vars = ['event_currentProb', 'event_lastProb']
    df_res['nonMarkingProb'] = {}
    for var in ['event']:
        for method in ['rightCont_timeAvg', 'leftCont_timeAvg', 'linearInterp_timeAvg', 'poisson_eventAvg', 'eventAvg']:
            df_res['nonMarkingProb'][var + '_' + method] = {}
    
    full_df_ = full_df[full_df['SentTime'] != -1].copy()
    df_res['sampleSize']['nonMarkingProb'] = {}
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

        samples_times = find_samples_path(time, txDelay, df_res['RTT'][path], df_name)
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

def calculate_offline_E2E_delays(full_df, removeDrops, checkColumn, txDelay, df_res, df_name):
    df_res['delay'] = {}
    for var in ['event']:
        for method in ['rightCont_timeAvg', 'leftCont_timeAvg', 'linearInterp_timeAvg', 'poisson_eventAvg', 'eventAvg']:
            df_res['delay'][var + '_' + method] = {}
    
    full_df_ = full_df.copy()
    if removeDrops:
        full_df_ = full_df_[full_df_[checkColumn] == 1]
    df_res['sampleSize']['delay'] = {}
    for path in full_df_['Path'].unique():
        df = full_df_[full_df_['Path'] == path]
        df = df.sort_values(by='SentTime').reset_index(drop=True)
        df_res['totalPckts'][path] = len(df)
        time = df['SentTime'].values
        values = df['Delay'].values

        rightCont_time_average = np.sum(values[:-1] * np.diff(time)) / (time[-1] - time[0])
        df_res['delay']['event_rightCont_timeAvg'][path] = rightCont_time_average

        leftCont_time_average = np.sum(values[1:] * np.diff(time)) / (time[-1] - time[0])
        df_res['delay']['event_leftCont_timeAvg'][path] = leftCont_time_average

        linearInterp_time_average = np.sum(((values[:-1] + values[1:]) / 2) * np.diff(time)) / (time[-1] - time[0])
        df_res['delay']['event_linearInterp_timeAvg'][path] = linearInterp_time_average
        # print("Calculating delay for path:", path, "with", len(time), "packets.")
        samples_times = find_samples_path(time, txDelay, df_res['RTT'][path], df_name)
        # print("Calculating delay for path:", path, "with", len(time), "packets. is done! ")
        df_res['sampleSize']['delay'][path] = len(samples_times)
        samples_values = df[df['SentTime'].isin(samples_times)]['Delay'].values
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

def addRemoveTransmission_data(full_df, linkDelays, linksRates):
    full_df['Delay'] = abs(full_df['ReceiveTime'] - full_df['SentTime'] - full_df['transmissionDelay'])
    # full_df['Delay'] = abs(full_df['ReceiveTime'] - full_df['TxEnqueueTime'] - full_df['transmissionDelay'])
    full_df['SentTime'] = full_df['SentTime'] + linkDelays[0] + (full_df['PayloadSize'] * 8) / linksRates[0]
    # full_df['SentTime'] = full_df['TxEnqueueTime']
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

def plot_queuingDelay_distribution(__ns3_path, results_folder, rate, experiment, segment, steadyStart, steadyEnd, paths, linkRate, onlyMeasurement):
    file_paths = glob.glob('{}/scratch/{}/{}/{}/*_{}.csv'.format(__ns3_path, results_folder, rate, experiment, segment))
    dfs = {}
    for file_path in file_paths:
        full_df = pd.read_csv(file_path)
        full_df = prune_data(full_df, 'Time', steadyStart, steadyEnd)
        if onlyMeasurement:
            full_df = full_df[full_df['Label'].str.contains('10.1.1.1', na=False)]
        full_df = full_df.sort_values(by=['Time', 'TotalQueueSize'], ascending=[True, True]).reset_index(drop=True)
        full_df['Delay'] = ((full_df['TotalQueueSize'] * 8) / linkRate).astype(int)
        # plot the distribution of the queuing delay
        plt.figure(figsize=(10, 6))
        plt.hist(full_df['Delay'], bins=200, density=True, color='g')
        # plot the mean as a vertical line with its value
        mean = full_df['Delay'].mean()
        plt.axvline(mean, color='r', linestyle='dashed', linewidth=1)
        plt.text(mean, 0, 'Mean: {:.2f}'.format(mean), color='r', fontsize=12)
        plt.title('Queuing Delay Distribution', fontsize=16)
        plt.xlabel('Delay (ns)', fontsize=16)
        plt.ylabel('Density', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid()
        if onlyMeasurement:
            plt.savefig('{}/scratch/{}/{}/{}/queuingDelayOfMeasurmentTraffic_distribution.png'.format(__ns3_path, results_folder, rate, experiment, segment))
        else:
            plt.savefig('{}/scratch/{}/{}/{}/queuingDelay_distribution.png'.format(__ns3_path, results_folder, rate, experiment, segment))

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

def plot_queuingDelay_time(__ns3_path, results_folder, rate, experiment, segment, steadyStart, steadyEnd, paths, linkRate):
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
        plt.ylim(0, 19000)
        # add the mean and variance of all the delays
        mean_full = full_df['TotalQueueSize'].mean()
        std_full = full_df['TotalQueueSize'].std()
        # plt.axhline(mean_full, color='g', linestyle='dashed', linewidth=1, label='Mean: {:.2f} B'.format(mean_full))
        # plt.axhline(mean_full + std_full, color='g', linestyle='dotted', linewidth=1, label='Mean + Std: {:.2f} B'.format(mean_full + std_full))
        # plt.axhline(mean_full - std_full, color='g', linestyle='dotted', linewidth=1, label='Mean - Std: {:.2f} B'.format(mean_full - std_full))
        plt.legend()
        plt.title('Queue Size per time', fontsize=16)
        plt.grid()
        plt.xlabel('Time (ns)', fontsize=16)
        # plt.ylabel('Delay (ns)', fontsize=16)
        plt.ylabel('Size (B)', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig('{}/scratch/{}/{}/{}/queuingDelay_time_{}_{}.png'.format(__ns3_path, results_folder, rate, experiment, segment, steadyStart, steadyEnd))
        # lags, corr = cross_correlation_delay_time_series(full_df_M['Time'].values, full_df_M['TotalQueueSize'].values, full_df_CT['Time'].values, full_df_CT['TotalQueueSize'].values, bin_width=1000000, max_lag=100000000, normalize=True, plot=False)
        # print(f"Cross-correlation lags: {lags}")
        # print(f"Cross-correlation values: {corr}")
        # max_corr = np.max(corr) 
        # lag_at_max = lags[np.argmax(corr)]
        # symmetry = np.corrcoef(corr[:len(corr)//2], corr[:len(corr)//2:-1])[0, 1]
        # print(f"Max correlation: {max_corr} at lag {lag_at_max} with symmetry {symmetry}")

def calculate_offline_computations_on_switch(__ns3_path, results_folder, rate, experiment, segment, steadyStart, steadyEnd, paths, linkRate, load):
    file_paths = glob.glob('{}/scratch/{}/{}/{}/{}/*_{}.csv'.format(__ns3_path, results_folder, rate, load, experiment, segment))
    dfs = {}
    for file_path in file_paths:
        df_res = {}
        df_name = 'A0D0'
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
        # full_df, delay_linearInterp_time_average = manipulate_for_delay_Q_m(full_df, linkRate)
        full_df, delay_linearInterp_time_average = manipulate_for_delay_Q(full_df, linkRate)
        full_df
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


def cross_correlation_delay_time_series(t_A, d_A, t_B, d_B, bin_width, max_lag, normalize=False, plot=False):
    """
    Compute cross-correlation between two delay time processes using bin-based averaging.

    Parameters:
    - t_A, d_A: timestamps and delays from source A
    - t_B, d_B: timestamps and delays from source B
    - bin_width: bin size in seconds (e.g., 0.01 for 10ms)
    - max_lag: maximum lag (in seconds) for cross-correlation
    - normalize: whether to normalize the correlation
    - plot: whether to plot the result

    Returns:
    - lags: array of time lags (in seconds)
    - corr: cross-correlation values
    """
    t_min = min(min(t_A), min(t_B))
    t_max = max(max(t_A), max(t_B))
    n_bins = int(np.ceil((t_max - t_min) / bin_width))

    # Bin the delays
    bins_A = [[] for _ in range(n_bins)]
    bins_B = [[] for _ in range(n_bins)]

    for t, d in zip(t_A, d_A):
        idx = int((t - t_min) // bin_width)
        if 0 <= idx < n_bins:
            bins_A[idx].append(d)

    for t, d in zip(t_B, d_B):
        idx = int((t - t_min) // bin_width)
        if 0 <= idx < n_bins:
            bins_B[idx].append(d)

    # Compute mean delay per bin (use NaN for empty bins)
    mean_A = np.array([np.mean(b) if b else np.nan for b in bins_A])
    mean_B = np.array([np.mean(b) if b else np.nan for b in bins_B])

    # Keep only bins where both A and B have data
    valid_mask = ~np.isnan(mean_A) & ~np.isnan(mean_B)
    series_A = mean_A[valid_mask]
    series_B = mean_B[valid_mask]
    for i in range(len(series_A)):
        print(series_A[i], series_B[i])
    
    if len(series_A) < 2:
        raise ValueError("Not enough overlapping bins with data to compute correlation.")

    # Normalize (zero mean)
    if normalize:
        series_A -= np.mean(series_A)
        series_B -= np.mean(series_B)

    # Compute full cross-correlation
    corr = np.correlate(series_A, series_B, mode='full')
    lags = np.arange(-len(series_A) + 1, len(series_A)) * bin_width

    # Restrict lag range
    mask = np.abs(lags) <= max_lag
    lags = lags[mask]
    corr = corr[mask]

    if normalize and np.max(np.abs(corr)) > 0:
        corr /= np.max(np.abs(corr))

    # Plot
    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(lags, corr, label='Cross-correlation')
        plt.axvline(0, color='gray', linestyle='--', alpha=0.6)
        plt.xlabel('Lag (s)')
        plt.ylabel('Correlation')
        plt.title('Cross-Correlation of Delay Time Series (Bin-Based)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return lags, 
    

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


def calculate_offline_computations(__ns3_path, rate, segment, experiment, results_folder, steadyStart, steadyEnd, projectColumn, removeDrops=True, checkColumn="", linksRates=[], linkDelays=[], swtichDstREDQueueDiscMaxSize=0, stats=None, tsh=0.15, differentiationDelay=None, errorRate=None, load=None):
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
            df_res['first'] = {}
            df_res['last'] = {}
            df_res['workload'] = {}
            df_res['sampleSize'] = {}
            df_res['successProbMean'] = {}
            df_res['sampleSize'] = {}
            df_res['totalPckts'] = {}
            df_res['RTT'] = {}
            df_res['InterArrivals'] = {}
            txDelay = (1502 * 8 / linksRates[0])
            full_df = addRemoveTransmission_data(full_df, linkDelays, linksRates)
            if (differentiationDelay is not None) and (differentiationDelay != 0.0):
                full_df = addExtraDelay(full_df, differentiationDelay, errorRate)
            full_df = prune_data(full_df, projectColumn, steadyStart, steadyEnd)
            df_res = calc_RTT_per_path(full_df, df_res, checkColumn, linkDelays)
            # for the reconstructed signal:
            # reconstructedSignal_df = reconstructSignal(full_df, linksRates, file_path.replace(f'{df_name}_EndToEnd_packets.csv', ''))
            # df_res = calculate_reconstructedSignal_delays(reconstructedSignal_df, df_res, linksRates[1])

            df_res = calculate_offline_E2E_lossRates(__ns3_path, full_df, df_res, checkColumn, txDelay, linksRates[1], swtichDstREDQueueDiscMaxSize, df_name)
            df_res = calculate_offline_E2E_delays(full_df, removeDrops, checkColumn, txDelay, df_res, df_name)
            df_res = calculate_offline_E2E_workload(full_df, df_res, steadyStart, steadyEnd)
            df_res = calculate_offline_E2E_markingProb(full_df, df_res, checkColumn, txDelay, swtichDstREDQueueDiscMaxSize, linksRates[1], __ns3_path, tsh, df_name)
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
            file_paths = glob.glob('{}/scratch/{}/{}/D_{}/f_{}/{}/*_EndToEnd_packets.csv'.format(__ns3_path, results_folder, rate, differentiationDelay, errorRate, i))
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
            file_paths = glob.glob('{}/scratch/{}/{}/D_{}/f_{}/{}/*_PoissonSampler.csv'.format(__ns3_path, results_folder, rate, differentiationDelay, errorRate, i))
            i += 1
    else:
        while len(file_paths) == 0:
            file_paths = glob.glob('{}/scratch/{}/{}/{}/{}/*_PoissonSampler.csv'.format(__ns3_path, results_folder, rate, load, i))
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