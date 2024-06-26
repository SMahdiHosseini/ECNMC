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
    else:
        return float(x)

def calc_epsilon(confidenceValue, segement_statistics):
    return (confidenceValue * segement_statistics['DelayStd']) / (np.sqrt(segement_statistics['sampleSize']) * segement_statistics['DelayMean'])

def calc_epsilon_loss(confidenceValue, segement_statistics):
    return (confidenceValue * segement_statistics['successProbStd']) / (np.sqrt(segement_statistics['sampleSize']) * segement_statistics['successProbMean'])

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
            per_traffic_data.append(pd.merge(endToEnd_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay', 'Path', 'ECN']), switches_dfs[switch], on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb', 'Id'], how='inner'))
        switches_dfs[switch] = pd.concat(per_traffic_data)

    for queue in start_dfs.keys():
        per_traffic_data = []
        for flow in endToEnd_dfs.keys():
            per_traffic_data.append(pd.merge(endToEnd_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay', 'Path', 'ECN']), start_dfs[queue], on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb', 'Id'], how='inner'))
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
            plt.savefig('../results_postProcessing_reverse_delay_1/{}/{}_delayOverTime_{}.png'.format(rate, flow, results_folder))
            # plt.savefig('../results_postProcessing/{}/{}_{}_delayOverTime_{}.png'.format(1.0, rate, flow, results_folder))
            plt.close()
            plt.clf()