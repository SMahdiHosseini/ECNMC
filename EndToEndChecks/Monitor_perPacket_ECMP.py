import pandas as pd
import glob
import configparser
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import anderson
from scipy.stats import f_oneway, kruskal
import json as js

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

__ns3_path = os.popen('locate "ns-3.41" | grep /ns-3.41$').read().splitlines()[0]
sample_rate = 0.05
confidenceValue = 1.96 # 95% confidence interval

def calculate_drop_rate(__ns3_path, steadyStart, steadyEnd, rate, segment, checkColumn, projectColumn, experiment):
    file_paths = glob.glob('{}/scratch/Results_1/{}/{}/*_{}.csv'.format(__ns3_path, rate, experiment, segment))
    swtiches_dropRates = {}
    for file_path in file_paths:
        df_name = file_path.split('/')[-1].split('_')[0]
        if 'C' in df_name:
            continue
        df = pd.read_csv(file_path)
        df = df[df[projectColumn] > steadyStart * 1000000000]
        df = df[df[projectColumn] < steadyEnd * 1000000000]
        # calculate the drop rate by dividing the some of the payload of dropped packets by the total payload of the sent packets
        total_payload = df['PayloadSize'].sum()
        dropped_payload = df[df[checkColumn] == 0]['PayloadSize'].sum()
        swtiches_dropRates[df_name] = dropped_payload / total_payload
    if len([value for value in swtiches_dropRates.values() if value != 0]) == 0:
        return 0
    return sum([value for value in swtiches_dropRates.values() if value != 0]) / len([value for value in swtiches_dropRates.values() if value != 0])

def read_data(__ns3_path, steadyStart, steadyEnd, rate, segment, checkColumn, projectColumn, experiment, remove_duplicates):
    file_paths = glob.glob('{}/scratch/Results_1/{}/{}/*_{}.csv'.format(__ns3_path, rate, experiment, segment))
    dfs = {}
    for file_path in file_paths:
        df_name = file_path.split('/')[-1].split('_')[0]
        df = pd.read_csv(file_path)
        df = df[df[checkColumn] == 1]
        df = df.reset_index(drop=True)
        df = df[df[projectColumn] > steadyStart * 1000000000]
        df = df[df[projectColumn] < steadyEnd * 1000000000]
        df = df.sort_values(by=[projectColumn], ignore_index=True)
        df = df.drop(columns=[checkColumn])
        if segment == 'EndToEnd' or segment == 'EndToEnd_crossTraffic':
            df['Delay'] = abs(df['ReceiveTime'] - df['SentTime'])
        if remove_duplicates:
            df = df.drop_duplicates(subset=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb', 'Id'], keep='first', ignore_index=True)
        dfs[df_name] = df
    return dfs

def read_data_flowIndicator(__ns3_path, rate):
    flows_name = []
    file_paths = glob.glob('{}/scratch/Results_1/{}/0/*_EndToEnd.csv'.format(__ns3_path, rate))
    for file_path in file_paths:
        flows_name.append(file_path.split('/')[-1].split('_')[0])
    return flows_name

def read_queues_indicators(__ns3_path, rate):
    flows_name = []
    file_paths = glob.glob('{}/scratch/Results_1/{}/0/*_PoissonSampler.csv'.format(__ns3_path, rate))
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
    else:
        return float(x)
    
def calc_epsilon(confidenceValue, segement_statistics):
    return (confidenceValue * segement_statistics['DelayStd']) / (np.sqrt(segement_statistics['sampleSize']) * segement_statistics['DelayMean'])

def calc_error(confidenceValue, segement_statistics):
    return (confidenceValue * segement_statistics['DelayStd']) / np.sqrt(segement_statistics['sampleSize'])

def sample_data(data, sample_column):
    exit = False
    while not exit:
        # option 1: sample data with a fixed rate
        data_copy = data.sample(frac=sample_rate).sort_values(by=[sample_column])
        
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

def addDelay(data):
    if 'Delay' not in data.columns:
        data['Delay'] = abs(data['ReceiveTime'] - data['SentTime'])
    return data

def get_timeAvg(data):
    # sort the data based on the receive time
    data = data.sort_values(by=['ReceiveTime'])
    # calculate the interarrival time
    data['InterArrivalTime'] = data['ReceiveTime'].diff().fillna(0)
    # calculate the time average: sum(InterArrivalTime * Delay) / sum(InterArrivalTime)
    timeAvg = (data['InterArrivalTime'] * data['Delay']).sum() / data['InterArrivalTime'].sum()
    return timeAvg


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

def check_dominant_bottleneck_consistency(endToEnd_statistics, samples_paths_aggregated_statistics, confidenceValue, pathsNum):
    # print(abs(endToEnd_statistics['timeAvg'] - samples_paths_aggregated_statistics['DelayMean']), confidenceValue * (endToEnd_statistics['DelayStd'] * np.sqrt(1 / samples_paths_aggregated_statistics['MinSampleSize'])))
    if abs(endToEnd_statistics['timeAvg'] - samples_paths_aggregated_statistics['DelayMean']) <= confidenceValue * (endToEnd_statistics['DelayStd'] * np.sqrt(1 / samples_paths_aggregated_statistics['MinSampleSize'])):
        return True
    else:
        return False
        
def check_MaxEpsilon_ineq(endToEnd_statistics, samples_paths_aggregated_statistics, confidenceValue):
    # print(abs(endToEnd_statistics['timeAvg'] - samples_paths_aggregated_statistics['DelayMean']) / samples_paths_aggregated_statistics['DelayMean'], samples_paths_aggregated_statistics['MaxEpsilon'])
    if abs(endToEnd_statistics['timeAvg'] - samples_paths_aggregated_statistics['DelayMean']) / samples_paths_aggregated_statistics['DelayMean'] <= samples_paths_aggregated_statistics['MaxEpsilon']:
        return True
    else:
        return False

def check_basic_delayConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, confidenceValue):
    # print(abs(endToEnd_statistics['timeAvg'] - samples_paths_aggregated_statistics['DelayMean']), samples_paths_aggregated_statistics['SumOfErrors'])
    if abs(endToEnd_statistics['timeAvg'] - samples_paths_aggregated_statistics['DelayMean']) <= samples_paths_aggregated_statistics['SumOfErrors']:
        return True
    else:
        return False
    
def check_all_delayConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, confidenceValue, pathsNum):
    res = {}
    res['DominantAssumption'] = {}
    res['MaxEpsilonIneq'] = {}
    res['Basic'] = {}
    for flow in endToEnd_statistics.keys():
        res['DominantAssumption'][flow] = check_dominant_bottleneck_consistency(endToEnd_statistics[flow], samples_paths_aggregated_statistics[flow], confidenceValue, pathsNum)
        res['MaxEpsilonIneq'][flow] = check_MaxEpsilon_ineq(endToEnd_statistics[flow], samples_paths_aggregated_statistics[flow], confidenceValue)
        res['Basic'][flow] = check_basic_delayConsistency(endToEnd_statistics[flow], samples_paths_aggregated_statistics[flow], confidenceValue)
    return res

def prepare_results(flows, queues):
    rounds_results = {}
    rounds_results['DominantAssumption'] = {}
    rounds_results['MaxEpsilonIneq'] = {}
    rounds_results['Basic'] = {}
    rounds_results['EndToEndMean'] = {}
    rounds_results['EndToEndStd'] = {}
    rounds_results['ANOVA'] = {}
    rounds_results['Kruskal'] = {}
    rounds_results['DropRate'] = []

    for q in queues:
        if not("H" in q or "C" in q):
            if not((q[0] == 'T' and (q[1] == '2' or q[1] == '3')) or (q[0] == 'A' and (q[3] == '0' or q[3] == '1'))):
                rounds_results['ANOVA'][q] = 0
                rounds_results['Kruskal'][q] = 0

        if q[0] == 'T' and q[2] == 'H' and (q[1] == '2' or q[1] == '3'):
            rounds_results[q+'std'] = []
        if q[0] == 'T' and q[2] == 'A' and (q[1] == '0' or q[1] == '1'):
            rounds_results[q+'std'] = []
        if q[0] == 'A' and q[2] == 'T' and (q[3] == '2' or q[3] == '3'):
            rounds_results[q+'std'] = []

    for flow in flows:
        rounds_results['DominantAssumption'][flow] = 0
        rounds_results['MaxEpsilonIneq'][flow] = 0
        rounds_results['Basic'][flow] = 0
        rounds_results['EndToEndMean'][flow] = []
        rounds_results['EndToEndStd'][flow] = []

    rounds_results['experiments'] = 0
    return rounds_results

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

def plot_overall_delay_distribution_noncommonSwitch(rate, switch_df, nonCommon_switch_sample_df, flow):
    # plot the delay distribution of Sample T1 and Switch T1
    fig, ax = plt.subplots(1, 2)
    sns.histplot(nonCommon_switch_sample_df['SentTime'] - nonCommon_switch_sample_df['ReceiveTime'], bins=100, ax=ax[0])
    sns.histplot(switch_df['SentTime'] - switch_df['ReceiveTime'], bins=100, ax=ax[1])
    ax[0].set_title('Sample T1')
    ax[1].set_title('Switch T1')
    ax[0].set_xlabel('Delay (ns)')
    ax[1].set_xlabel('Delay (ns)')
    plt.savefig('results/{}/{}_T1_{}_overall_delayDist.png'.format(rate, rate, flow))
    plt.close()

def plot_overall_delay_distribution(rate, common_switch_sample_df, queue):
    # plot the delay distribution of SWitch T0 and Sample T0
    fig, ax = plt.subplots(1, 1)
    sns.histplot(common_switch_sample_df['SentTime'] - common_switch_sample_df['ReceiveTime'], bins=100)
    ax.set_title('Sample T0')
    ax.set_xlabel('Delay (ns)')
    plt.savefig('../results_perPacket_ECMP/{}/{}_{}_overall_delayDist.png'.format(rate, rate, queue))
    plt.close()

def plot_seperate_delay_distribution(rate, flows):
    fig, ax = plt.subplots(1, 1)
    for i in range(len(flows)):
        sns.histplot(np.array(flows[i]['Delay']), ax=ax, label='Flow {}'.format(i), bins=100)
    plt.legend()
    ax.set_title('Delay Distribution of Flows')
    ax.set_xlabel('Delay (ns)')
    plt.savefig('results/{}/{}_T0_seperate_delayDist.png'.format(rate, rate))
    plt.close()

def plot_endToEnd_delay_distribution(rate, endToEnd_dfs):
    fig, ax = plt.subplots(1, 1)
    for flow in endToEnd_dfs.keys():
        endToEnd_dfs[flow]['Delay'] = abs(endToEnd_dfs[flow]['ReceiveTime'] - endToEnd_dfs[flow]['SentTime'])
        sns.histplot(np.array(endToEnd_dfs[flow]['Delay']), ax=ax, label='Flow {}'.format(flow), bins=100)
    plt.legend()
    ax.set_title('End To End Delay Distribution of Flows')
    ax.set_xlabel('Delay (ns)')
    plt.savefig('results/{}/{}_EndToEnd_delayDist.png'.format(rate, rate))
    plt.close()

def compatibility_check(confidenceValue, rounds_results, samples_paths_aggregated_statistics, endToEnd_statistics, flows_name, pathsNum):
    # End to End and Persegment Compatibility Check
    results = check_all_delayConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, confidenceValue, pathsNum)

    for flow in flows_name:
        if results['DominantAssumption'][flow]:
            rounds_results['DominantAssumption'][flow] += 1
        if results['MaxEpsilonIneq'][flow]:
            rounds_results['MaxEpsilonIneq'][flow] += 1
        if results['Basic'][flow]:
            rounds_results['Basic'][flow] += 1


def remove_interlinks_trasmission_delay(endToEnd_dfs, switches_dfs, aggSwitchesNum):
    for flow in endToEnd_dfs.keys():
        src_Tor_swich = 'T' + flow[1]
        Agg_switch = ['A' + str(i) for i in range(aggSwitchesNum)]
        Tor_dest_switch = 'T' + flow[5]

        src_Tor = intermediateLink_transmission(endToEnd_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay']), endToEnd_dfs[flow].drop(columns=['Delay']), switches_dfs[src_Tor_swich], 0)
        Tor_Agg = []
        Agg_Tor = []
        for i in range(aggSwitchesNum):
            Tor_Agg.append(intermediateLink_transmission(endToEnd_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay']), switches_dfs[src_Tor_swich], switches_dfs[Agg_switch[i]], 1))
            Agg_Tor.append(intermediateLink_transmission(endToEnd_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay']), switches_dfs[Agg_switch[i]], switches_dfs[Tor_dest_switch], 2))
        Tor_Agg = pd.concat(Tor_Agg)
        Agg_Tor = pd.concat(Agg_Tor)
        Tor_dst = intermediateLink_transmission(endToEnd_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay']), switches_dfs[Tor_dest_switch], endToEnd_dfs[flow].drop(columns=['Delay']), 3)
        endToEnd_dfs[flow] = pd.merge(endToEnd_dfs[flow], src_Tor, on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb', 'Id'], how='inner')        
        endToEnd_dfs[flow] = pd.merge(endToEnd_dfs[flow], Tor_Agg, on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb', 'Id'], how='inner')
        endToEnd_dfs[flow] = pd.merge(endToEnd_dfs[flow], Agg_Tor, on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb', 'Id'], how='inner')
        endToEnd_dfs[flow] = pd.merge(endToEnd_dfs[flow], Tor_dst, on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb', 'Id'], how='inner')
        endToEnd_dfs[flow]['Delay'] = endToEnd_dfs[flow]['Delay'] - sum([endToEnd_dfs[flow]['Delay_' + str(i)] for i in range(4)])
        endToEnd_dfs[flow] = endToEnd_dfs[flow].drop(columns=['Delay_0', 'Delay_1', 'Delay_2', 'Delay_3'])
        endToEnd_dfs[flow] = endToEnd_dfs[flow][endToEnd_dfs[flow]['Delay'] > 0]

def clear_data_from_outliers_in_time(endToEnd_dfs, switches_dfs):
    for switch in switches_dfs.keys():
        per_traffic_data = []
        for flow in endToEnd_dfs.keys():
            per_traffic_data.append(pd.merge(endToEnd_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay']), switches_dfs[switch], on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb', 'Id'], how='inner'))
        switches_dfs[switch] = pd.concat(per_traffic_data)
        

def check_grountruth_delayConsistency(endToEnd_dfs, switches_dfs):
    # merge the endToEnd data with the switch data and check the delay consistency
    switches_dfs_copy = switches_dfs.copy()
    for switch in switches_dfs_copy.keys():
        switches_dfs_copy[switch]['Delay_' + switch] = abs(switches_dfs_copy[switch]['ReceiveTime'] - switches_dfs_copy[switch]['SentTime'])

    for flow in endToEnd_dfs.keys():
        l_df = endToEnd_dfs[flow].copy().drop(columns=['SentTime', 'ReceiveTime'])
        l_df = pd.merge(l_df, switches_dfs_copy['T0'].drop(columns=['SentTime', 'ReceiveTime']), on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb'], how='inner')
        l_df = pd.merge(l_df, switches_dfs_copy['T1'].drop(columns=['SentTime', 'ReceiveTime']), on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb'], how='inner')
        l_df['Deviation'] = l_df['Delay'] - l_df['Delay_T0'] - l_df['Delay_T1']
        l_df = l_df[l_df['Deviation'] != 0]
        print(l_df)
        # # remove the rows from the endToEnd data that have the same combination of ('SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb') with the rows in l_df
        # endToEnd_dfs[flow] = endToEnd_dfs[flow][~endToEnd_dfs[flow].apply(lambda x: (x['SourceIp'], x['SourcePort'], x['DestinationIp'], x['DestinationPort'], x['PayloadSize'], x['SequenceNb']) in l_df[['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb']].values, axis=1)]
        # for switch in switches_dfs.keys():
        #     switches_dfs[switch] = switches_dfs[switch][~switches_dfs[switch].apply(lambda x: (x['SourceIp'], x['SourcePort'], x['DestinationIp'], x['DestinationPort'], x['PayloadSize'], x['SequenceNb']) in l_df[['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb']].values, axis=1)]

def print_traffic_rate(endToEnd_dfs):
    endToEnd_dataRates = {}
    for flow in endToEnd_dfs.keys():
        endToEnd_dataRates[flow] = endToEnd_dfs[flow]['PayloadSize'].sum() * 8 / (10) / 1000000
    print([(key, value / sum(endToEnd_dataRates.values()) * 100) for key, value in endToEnd_dataRates.items()])

def analyze_single_experiment(rate, steadyStart, steadyEnd, confidenceValue, rounds_results, queues_names, experiment=0, ns3_path=__ns3_path):
    num_of_agg_switches = 2
    endToEnd_dfs = read_data(__ns3_path, steadyStart, steadyEnd, rate, 'EndToEnd', 'IsReceived', 'SentTime', str(experiment), True)
    switches_dfs = read_data(__ns3_path, steadyStart, steadyEnd, rate, 'Switch', 'IsSent', 'ReceiveTime', str(experiment), True)
    samples_dfs = read_data(__ns3_path, steadyStart, steadyEnd, rate, 'PoissonSampler', 'IsDeparted', 'SampleTime', str(experiment), False)

    # print_traffic_rate(endToEnd_dfs)
    rounds_results['DropRate'].append(calculate_drop_rate(__ns3_path, steadyStart, steadyEnd, rate, 'Switch', 'IsSent', 'ReceiveTime', str(experiment)))

    # Intermediate links groundtruth statistics
    remove_interlinks_trasmission_delay(endToEnd_dfs, switches_dfs, num_of_agg_switches)
    # integrate the switch data with the endToEnd data
    clear_data_from_outliers_in_time(endToEnd_dfs, switches_dfs)

    # check_grountruth_delayConsistency(endToEnd_dfs, switches_dfs)
    # switch_different_traffics_delaymean(switches_dfs['A0'])

    # samples switches statistics
    samples_switches_statistics = {}
    samples_queues_dfs = {}
    for sample_df in samples_dfs.keys():
        # print(sample_df, sample_df[0:2])
        samples_queues_dfs[sample_df] = get_switch_samples_delays(switches_dfs[sample_df[0:2]], samples_dfs[sample_df])
        samples_switches_statistics[sample_df] = get_statistics(samples_queues_dfs[sample_df])
        # print(samples_switches_statistics[sample_df])

    # samples_paths_statistics
    samples_paths_aggregated_statistics = {}
    for flow in endToEnd_dfs.keys():
        samples_paths_aggregated_statistics[flow] = {}
        samples_paths_aggregated_statistics[flow]['DelayMean'] = np.average([sum([samples_switches_statistics['T' + flow[1] + 'A' + str(i)]['DelayMean'], 
                                                                                  samples_switches_statistics['A' + str(i) + 'T' + flow[5]]['DelayMean'],
                                                                                  samples_switches_statistics['T' + flow[5] + 'H' + flow[7]]['DelayMean']]) for i in range(num_of_agg_switches)])
        samples_paths_aggregated_statistics[flow]['MinSampleSize'] = min([min([samples_switches_statistics['T' + flow[1] + 'A' + str(i)]['sampleSize'],
                                                                               samples_switches_statistics['A' + str(i) + 'T' + flow[5]]['sampleSize'],
                                                                               samples_switches_statistics['T' + flow[5] + 'H' + flow[7]]['sampleSize']]) for i in range(num_of_agg_switches)])
        samples_paths_aggregated_statistics[flow]['MaxEpsilon'] = max([max([calc_epsilon(confidenceValue, samples_switches_statistics['T' + flow[1] + 'A' + str(i)]),
                                                                            calc_epsilon(confidenceValue, samples_switches_statistics['A' + str(i) + 'T' + flow[5]]),
                                                                            calc_epsilon(confidenceValue, samples_switches_statistics['T' + flow[5] + 'H' + flow[7]])]) for i in range(num_of_agg_switches)])
        samples_paths_aggregated_statistics[flow]['SumOfErrors'] = np.average([sum([calc_error(confidenceValue, samples_switches_statistics['T' + flow[1] + 'A' + str(i)]),
                                                                                    calc_error(confidenceValue, samples_switches_statistics['A' + str(i) + 'T' + flow[5]]),
                                                                                    calc_error(confidenceValue, samples_switches_statistics['T' + flow[5] + 'H' + flow[7]])]) for i in range(num_of_agg_switches)])

    # endToEnd_statistics
    endToEnd_statistics = {}
    for flow in endToEnd_dfs.keys():
        endToEnd_statistics[flow] = get_statistics(endToEnd_dfs[flow], timeAvg=True)
        rounds_results['EndToEndMean'][flow].append(endToEnd_statistics[flow]['timeAvg'])
        rounds_results['EndToEndStd'][flow].append(endToEnd_statistics[flow]['DelayStd'])

    rounds_results['experiments'] += 1

    # sampling for ANOVA and Kruskal-Wallis test
    flows_sampled = {}
    for q in queues_names:
        if 'H' in q or 'C' in q:
            continue
        flows_sampled[q] = []
        for flow in endToEnd_dfs.keys():
            if q[0] == 'A' and flow[5] == q[3]:
                flows_sampled[q].append(switch_data(endToEnd_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay']), switches_dfs[q[0:2]], True))
            elif q[0] == 'T' and flow[1] == q[1]:
                flows_sampled[q].append(switch_data(endToEnd_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay']), switches_dfs[q[0:2]], True))
        if len(flows_sampled[q]) == 0:
            del flows_sampled[q]

    for q in queues_names:
        if q[0] == 'T' and q[2] == 'H' and (q[1] == '2' or q[1] == '3'):
            rounds_results[q+'std'].append(samples_switches_statistics[q]['DelayStd'])
        if q[0] == 'T' and q[2] == 'A' and (q[1] == '0' or q[1] == '1'):
            rounds_results[q+'std'].append(samples_switches_statistics[q]['DelayStd'])
        if q[0] == 'A' and q[2] == 'T' and (q[3] == '2' or q[3] == '3'):
            rounds_results[q+'std'].append(samples_switches_statistics[q]['DelayStd'])

    delayProcess_consistency_check(flows_sampled, rounds_results)
    
    if experiment == 0:
        for sample_df in samples_queues_dfs.keys():
            if sample_df[0] == 'T' and sample_df[2] == 'H' and (sample_df[1] == '2' or sample_df[1] == '3'):
                plot_overall_delay_distribution(rate, samples_queues_dfs[sample_df], sample_df)
            if sample_df[0] == 'T' and sample_df[2] == 'A' and (sample_df[1] == '0' or sample_df[1] == '1'):
                plot_overall_delay_distribution(rate, samples_queues_dfs[sample_df], sample_df)
            if sample_df[0] == 'A' and sample_df[2] == 'T' and (sample_df[3] == '2' or sample_df[3] == '3'):
                plot_overall_delay_distribution(rate, samples_queues_dfs[sample_df], sample_df)

    #     plot_seperate_delay_distribution(rate, flows)
    #     plot_endToEnd_delay_distribution(rate, endToEnd_dfs)
    #     for flow in endToEnd_dfs.keys():
    #         plot_overall_delay_distribution_noncommonSwitch(rate, 
    #                                                         pd.merge(switches_dfs['T1'], endToEnd_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay']), on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb'], how='inner'), 
    #                                                         nonCommon_switch_sample_df[flow], flow)
    
    compatibility_check(confidenceValue, rounds_results, samples_paths_aggregated_statistics, endToEnd_statistics, endToEnd_dfs.keys(), num_of_agg_switches)

def analyze_all_experiments(rate, steadyStart, steadyEnd, confidenceValue, experiments_start=0, experiments_end=3, ns3_path=__ns3_path):
    flows_name = read_data_flowIndicator(ns3_path, rate)
    flows_name.sort()

    queues_names = read_queues_indicators(ns3_path, rate)
    queues_names.sort()

    rounds_results = prepare_results(flows_name, queues_names)

    for experiment in range(experiments_start, experiments_end):
        if len(os.listdir('{}/scratch/Results_1/{}/{}'.format(__ns3_path, rate, experiment))) == 0:
            print(experiment)
            continue
        print("Analyzing experiment: ", experiment)
        analyze_single_experiment(rate, steadyStart, steadyEnd, confidenceValue, rounds_results, queues_names, experiment, ns3_path, )

    with open('../results_perPacket_ECMP/{}/{}_{}_results.json'.format(rate,rate, experiments_end), 'w') as f:
        # save the results in a well formatted json file
        js.dump(rounds_results, f, indent=4)

# main function
def __main__():
    config = configparser.ConfigParser()
    config.read('../Parameters.config')
    hostToTorLinkRate = convert_to_float(config.get('Settings', 'hostToTorLinkRate'))
    torToAggLinkRate = config.get('Settings', 'torToAggLinkRate')
    aggToCoreLinkRate = convert_to_float(config.get('Settings', 'aggToCoreLinkRate'))
    hostToTorLinkDelay = convert_to_float(config.get('Settings', 'hostToTorLinkDelay'))
    torToAggLinkDelay = convert_to_float(config.get('Settings', 'torToAggLinkDelay'))
    aggToCoreLinkDelay = convert_to_float(config.get('Settings', 'aggToCoreLinkDelay'))
    pctPacedBack = convert_to_float(config.get('Settings', 'pctPacedBack'))
    appDataRate = convert_to_float(config.get('Settings', 'appDataRate'))
    duration = convert_to_float(config.get('Settings', 'duration'))
    steadyStart = convert_to_float(config.get('Settings', 'steadyStart'))
    steadyEnd = convert_to_float(config.get('Settings', 'steadyEnd'))
    sampleRate = convert_to_float(config.get('Settings', 'sampleRate'))
    experiments = int(config.get('Settings', 'experiments'))
    serviceRateScales = [float(x) for x in config.get('Settings', 'serviceRateScales').split(',')]

    print("hostToTorLinkRate: ", hostToTorLinkRate, " Mbps")
    print("torToAggLinkRate: ", torToAggLinkRate)
    print("aggToCoreLinkRate: ", aggToCoreLinkRate, " Mbps")
    print("hostToTorLinkDelay: ", hostToTorLinkDelay, " ms")
    print("torToAggLinkDelay: ", torToAggLinkDelay, " ms")
    print("aggToCoreLinkDelay: ", aggToCoreLinkDelay, " ms")
    print("pctPacedBack: ", pctPacedBack, " %")
    print("appDataRate: ", appDataRate, " Mbps")
    print("duration: ", duration, " s")
    print("steadyStart: ", steadyStart, " s")
    print("steadyEnd: ", steadyEnd, " s")
    print("sampleRate", sampleRate)
    print("experiments: ", experiments)
    print("serviceRateScales: ", serviceRateScales)
    # serviceRateScales = [0.85]
    experiments = 1
    # steadyStart = 4
    # steadyEnd = 9

    for rate in serviceRateScales:
        print("\nAnalyzing experiments for rate: ", rate)
        analyze_all_experiments(rate, steadyStart, steadyEnd, confidenceValue, experiments_start=0, experiments_end=experiments, ns3_path=__ns3_path)
        print("Rate {} {} done".format(rate, experiments))

__main__()