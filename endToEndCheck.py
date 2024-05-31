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
sample_rate = 0.01
confidenceValue = 1.96 # 95% confidence interval

def calculate_drop_rate(__ns3_path, steadyStart, steadyEnd, rate, segment, checkColumn, projectColumn, experiment):
    file_paths = glob.glob('{}/scratch/Results/{}/{}/*_{}.csv'.format(__ns3_path, rate, experiment, segment))
    swtiches_dropRates = {}
    for file_path in file_paths:
        df_name = file_path.split('/')[-1].split('_')[0]
        df = pd.read_csv(file_path)
        df = df[df[projectColumn] > steadyStart * 1000000000]
        df = df[df[projectColumn] < steadyEnd * 1000000000]
        # calculate the drop rate by dividing the some of the payload of dropped packets by the total payload of the sent packets
        total_payload = df['PayloadSize'].sum()
        dropped_payload = df[df[checkColumn] == 0]['PayloadSize'].sum()
        swtiches_dropRates[df_name] = dropped_payload / total_payload
    return swtiches_dropRates

def read_data(__ns3_path, steadyStart, steadyEnd, rate, segment, checkColumn, projectColumn, experiment, remove_duplicates):
    file_paths = glob.glob('{}/scratch/Results/{}/{}/*_{}.csv'.format(__ns3_path, rate, experiment, segment))
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
            df = df.drop_duplicates(subset=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb'], keep='first', ignore_index=True)
        dfs[df_name] = df
    return dfs

def read_data_flowIndicator(__ns3_path, rate):
    flows_name = []
    file_paths = glob.glob('{}/scratch/Results/{}/0/*_EndToEnd.csv'.format(__ns3_path, rate))
    for file_path in file_paths:
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
    l_df = pd.merge(l_df, switchDf, on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb'], how='right')
    l_df['SentTime'] = l_df['SentTime'].mask(l_df['SourceIp'] == '0.0.0.0', l_df['SampleTime'])
    l_df['ReceiveTime'] = l_df['ReceiveTime'].mask(l_df['SourceIp'] == '0.0.0.0' , l_df['SampleTime'])
    l_df = l_df.dropna(subset=['SentTime', 'ReceiveTime'])
    return l_df

def switch_data(flowIndicatorDf, switchDf, sampling):
    l_df = flowIndicatorDf.copy()
    l_df = pd.merge(l_df, switchDf, on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb'], how='inner')
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
    l_df = pd.merge(l_df, source.drop(columns=['ReceiveTime']), on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb'], how='inner')
    l_df = pd.merge(l_df, dest.drop(columns=['SentTime']), on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb'], how='inner')
    l_df['Delay_' + str(linkNum)] = abs(l_df['ReceiveTime'] - l_df['SentTime'])
    return l_df.drop(columns=['ReceiveTime', 'SentTime'])

def addDelay(data):
    if 'Delay' not in data.columns:
        data['Delay'] = abs(data['ReceiveTime'] - data['SentTime'])
    return data

def get_statistics(data, removeZeroes=False):
    statistics = {}
    data_copy = addDelay(data.copy())
    if removeZeroes:
        data_copy = data_copy[data_copy['Delay'] > 0]
    statistics['DelayMean'] = data_copy['Delay'].mean()
    statistics['DelayStd'] = data_copy['Delay'].std()
    statistics['sampleSize'] = len(data_copy)
    statistics['DelaySkew'] = data_copy['Delay'].skew()
    statistics['sizeStd'] = data_copy['PayloadSize'].std()
    return statistics

def ECNMC(endToEnd_delayMean, sumOfSegments_DelayMeans, endToEnd_delayStd, MinSampleSize, confidenceValue):
    if abs(endToEnd_delayMean - sumOfSegments_DelayMeans) <= confidenceValue * (endToEnd_delayStd / np.sqrt(MinSampleSize)):
        return True
    else:  
        return False

def ECNMC_V2(endToEnd_delayMean, sumOfSegments_DelayMeans, maxEpsilon):
    if abs(endToEnd_delayMean - sumOfSegments_DelayMeans) / sumOfSegments_DelayMeans <= maxEpsilon:
        return True
    else:  
        return False
    
def ECNMC_V3(endToEnd_delayMean, sumOfSegments_DelayMeans, approximate_endToEnd_delayStd, MinSampleSize, confidenceValue):
    if abs(endToEnd_delayMean - sumOfSegments_DelayMeans) <= confidenceValue * (approximate_endToEnd_delayStd / np.sqrt(MinSampleSize)):
        return True
    else:  
        return False

def check_single_delayConsistency(endToEnd_statistics, switches_statistics, confidenceValue):
    switches_delayMeans = [value['DelayMean'] for value in switches_statistics.values()]
    switches_sampleSizes = [value['sampleSize'] for value in switches_statistics.values()]
    MinSampleSize = min(switches_sampleSizes)
    sumOfSegmentsDelayMeans = sum(switches_delayMeans)
    # print(endToEnd_statistics['DelayMean'], sumOfSegmentsDelayMeans, endToEnd_statistics['DelayStd'], MinSampleSize, confidenceValue)
    return ECNMC(endToEnd_statistics['DelayMean'], sumOfSegmentsDelayMeans, endToEnd_statistics['DelayStd'], MinSampleSize, confidenceValue)
        
def check_single_delayConsistency_V2(endToEnd_statistics, switches_statistics, confidenceValue):
    # calculate the epsilon = confidenceValue * (switches_delayStd / (sqrt(switches_sampleSize) * switches_delayMean)) for each switch
    switches_delayMeans = [value['DelayMean'] for value in switches_statistics.values()]
    switches_delayStds = [value['DelayStd'] for value in switches_statistics.values()]
    switches_sampleSizes = [value['sampleSize'] for value in switches_statistics.values()]

    segments_delayMeans = switches_delayMeans
    segments_delayStds = switches_delayStds
    segments_sampleSizes = switches_sampleSizes

    epsilons = [confidenceValue * (segments_delayStds[i] / (np.sqrt(segments_sampleSizes[i]) * segments_delayMeans[i])) for i in range(len(segments_delayMeans))]
    maxEpsilon = max(epsilons)
    sumOfSegmentsDelayMeans = sum(segments_delayMeans)

    return ECNMC_V2(endToEnd_statistics['DelayMean'], sumOfSegmentsDelayMeans, maxEpsilon)

def check_single_delayConsistency_V3(endToEnd_statistics, switches_statistics, confidenceValue):
    switches_delayMeans = [value['DelayMean'] for value in switches_statistics.values()]
    if switches_statistics['T0']['DelayStd'] > switches_statistics['T1']['DelayStd']:
        approximate_endToEnd_delayStd = switches_statistics['T0']['DelayStd']
    else:
        approximate_endToEnd_delayStd = switches_statistics['T1']['DelayStd']

    switches_sampleSizes = [value['sampleSize'] for value in switches_statistics.values()]
    MinSampleSize = min(switches_sampleSizes)
    sumOfSegmentsDelayMeans = sum(switches_delayMeans)
    return ECNMC_V3(endToEnd_statistics['DelayMean'], sumOfSegmentsDelayMeans, approximate_endToEnd_delayStd, MinSampleSize, confidenceValue)

def check_all_delayConsistency(endToEnd_statistics, switches_statistics, confidenceValue):
    res = {}
    res['DominantAssumption'] = {}
    res['General'] = {}
    res['RelaxedDominantAssumption'] = {}
    for flow in endToEnd_statistics.keys():
        res['DominantAssumption'][flow] = check_single_delayConsistency(endToEnd_statistics[flow], switches_statistics[flow], confidenceValue)
        res['General'][flow] = check_single_delayConsistency_V2(endToEnd_statistics[flow], switches_statistics[flow], confidenceValue)
        res['RelaxedDominantAssumption'][flow] = check_single_delayConsistency_V3(endToEnd_statistics[flow], switches_statistics[flow], confidenceValue)
    return res

def prepare_results(flows):
    rounds_results = {}
    rounds_results['Overall'] = {}
    rounds_results['PerTrafficStream'] = {}
    rounds_results['Overall']['samples'] = {}
    rounds_results['Overall']['samples']['DominantAssumption'] = {}
    rounds_results['Overall']['samples']['General'] = {}
    rounds_results['Overall']['samples']['RelaxedDominantAssumption'] = {}
    rounds_results['ANOVA'] = {}
    rounds_results['Kruskal'] = {}
    rounds_results['ANOVA']['samples'] = 0
    rounds_results['Kruskal']['samples'] = 0
    rounds_results['ANOVA']['groundtruth'] = 0
    rounds_results['Kruskal']['groundtruth'] = 0
    rounds_results['EndToEndMean'] = {}
    rounds_results['EndToEndStd'] = {}
    rounds_results['EndToEndSkew'] = {}
    rounds_results['DropRate'] = []
    rounds_results['T0std'] = []
    rounds_results['T1std'] = {}
    rounds_results['T0Ineq'] = {}
    rounds_results['T0IneqMaxEpsilon'] = {}
    # rounds_results['T0IeqHighRate'] = {}
    # rounds_results['T0IneqRegular'] = {}
    rounds_results['T0IneqRemovedZeroes'] = {}
    rounds_results['T1Ineq'] = {}
    rounds_results['T1IneqMaxEpsilon'] = {}
    # rounds_results['T1IeqHighRate'] = {}
    # rounds_results['T1IneqRegular'] = {}
    rounds_results['T1IneqRemovedZeroes'] = {}
    for flow in flows:
        rounds_results['Overall']['samples']['DominantAssumption'][flow] = 0
        rounds_results['Overall']['samples']['General'][flow] = 0
        rounds_results['Overall']['samples']['RelaxedDominantAssumption'][flow] = 0
        rounds_results['EndToEndMean'][flow] = []
        rounds_results['EndToEndStd'][flow] = []
        rounds_results['EndToEndSkew'][flow] = []
        rounds_results['T1std'][flow] = []
        rounds_results['T0Ineq'][flow] = 0
        # rounds_results['T0IeqHighRate'][flow] = 0
        # rounds_results['T0IneqRegular'][flow] = 0
        rounds_results['T0IneqRemovedZeroes'][flow] = 0
        rounds_results['T0IneqMaxEpsilon'][flow] = 0
        rounds_results['T1Ineq'][flow] = 0
        # rounds_results['T1IeqHighRate'][flow] = 0
        # rounds_results['T1IneqRegular'][flow] = 0
        rounds_results['T1IneqRemovedZeroes'][flow] = 0
        rounds_results['T1IneqMaxEpsilon'][flow] = 0


    rounds_results['experiments'] = 0
    return rounds_results

def delayProcess_consistency_check(flows, flows_sampled, rounds_results):
    # Observing the same delay process test on the common swtiches(T0)
    for flow_on_switch in flows:
        flow_on_switch['Delay'] = abs(flow_on_switch['ReceiveTime'] - flow_on_switch['SentTime'])

    anova_res  = f_oneway(*[flows[i]['Delay'] for i in range(len(flows))])
    kruskal_res = kruskal(*[flows[i]['Delay'] for i in range(len(flows))])
    if anova_res.pvalue > 0.05:
        rounds_results['ANOVA']['groundtruth'] += 1
    if kruskal_res.pvalue > 0.05:
        rounds_results['Kruskal']['groundtruth'] += 1

    for flow_on_switch in flows_sampled:
        flow_on_switch['Delay'] = abs(flow_on_switch['ReceiveTime'] - flow_on_switch['SentTime'])

    anova_res  = f_oneway(*[flows_sampled[i]['Delay'] for i in range(len(flows_sampled))])
    kruskal_res = kruskal(*[flows_sampled[i]['Delay'] for i in range(len(flows_sampled))])
    if anova_res.pvalue > 0.05:
        rounds_results['ANOVA']['samples'] += 1
    if kruskal_res.pvalue > 0.05:
        rounds_results['Kruskal']['samples'] += 1 

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

def plot_overall_delay_distribution(rate, switches_dfs, common_switch_sample_df):
    # plot the delay distribution of SWitch T0 and Sample T0
    fig, ax = plt.subplots(1, 2)
    sns.histplot(common_switch_sample_df['SentTime'] - common_switch_sample_df['ReceiveTime'], ax=ax[0], bins=100)
    sns.histplot(switches_dfs['T0']['SentTime'] - switches_dfs['T0']['ReceiveTime'], ax=ax[1], bins=100)
    ax[0].set_title('Sample T0')
    ax[1].set_title('Switch T0')
    ax[0].set_xlabel('Delay (ns)')
    ax[1].set_xlabel('Delay (ns)')
    plt.savefig('results/{}/{}_T0_overall_delayDist.png'.format(rate, rate))
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

def compatibility_check(confidenceValue, rounds_results, samples_statistics, endToEnd_statistics, groundtruth_statistics, flows_name):
    # End to End and Persegment Compatibility Check
    results = {}
    results['Overall'] = {}
    results['Overall']['samples'] = check_all_delayConsistency(endToEnd_statistics, samples_statistics['Overall'], confidenceValue)

    for flow in flows_name:
        if results['Overall']['samples']['DominantAssumption'][flow]:
                rounds_results['Overall']['samples']['DominantAssumption'][flow] += 1
        if results['Overall']['samples']['General'][flow]:
            rounds_results['Overall']['samples']['General'][flow] += 1
        if results['Overall']['samples']['RelaxedDominantAssumption'][flow]:
            rounds_results['Overall']['samples']['RelaxedDominantAssumption'][flow] += 1

def remove_interlinks_trasmission_delay(endToEnd_dfs, crossTraffic_dfs, switches_dfs):
    for flow in endToEnd_dfs.keys():
        src_T0 = intermediateLink_transmission(endToEnd_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay']), endToEnd_dfs[flow].drop(columns=['Delay']), switches_dfs['T0'], 0)
        T0_T1  = intermediateLink_transmission(endToEnd_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay']), switches_dfs['T0'], switches_dfs['T1'], 1)
        T1_dst = intermediateLink_transmission(endToEnd_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay']), switches_dfs['T1'], endToEnd_dfs[flow].drop(columns=['Delay']), 2)
        endToEnd_dfs[flow] = pd.merge(endToEnd_dfs[flow], src_T0, on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb'], how='inner')
        endToEnd_dfs[flow] = pd.merge(endToEnd_dfs[flow], T0_T1,  on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb'], how='inner')
        endToEnd_dfs[flow] = pd.merge(endToEnd_dfs[flow], T1_dst, on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb'], how='inner')
        endToEnd_dfs[flow]['Delay'] = endToEnd_dfs[flow]['Delay'] - endToEnd_dfs[flow]['Delay_0'] - endToEnd_dfs[flow]['Delay_1'] - endToEnd_dfs[flow]['Delay_2']
        endToEnd_dfs[flow] = endToEnd_dfs[flow].drop(columns=['Delay_0', 'Delay_1', 'Delay_2'])
        endToEnd_dfs[flow] = endToEnd_dfs[flow][endToEnd_dfs[flow]['Delay'] > 0]

    for flow in crossTraffic_dfs.keys():
        src_T0 = intermediateLink_transmission(crossTraffic_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay']), crossTraffic_dfs[flow].drop(columns=['Delay']), switches_dfs['T0'], 0)
        T0_T1  = intermediateLink_transmission(crossTraffic_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay']), switches_dfs['T0'], switches_dfs['T1'], 1)
        T1_dst = intermediateLink_transmission(crossTraffic_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay']), switches_dfs['T1'], crossTraffic_dfs[flow].drop(columns=['Delay']), 2)
        crossTraffic_dfs[flow] = pd.merge(crossTraffic_dfs[flow], src_T0, on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb'], how='inner')
        crossTraffic_dfs[flow] = pd.merge(crossTraffic_dfs[flow], T0_T1,  on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb'], how='inner')
        crossTraffic_dfs[flow] = pd.merge(crossTraffic_dfs[flow], T1_dst, on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb'], how='inner')
        crossTraffic_dfs[flow]['Delay'] = crossTraffic_dfs[flow]['Delay'] - crossTraffic_dfs[flow]['Delay_0'] - crossTraffic_dfs[flow]['Delay_1'] - crossTraffic_dfs[flow]['Delay_2']
        crossTraffic_dfs[flow] = crossTraffic_dfs[flow].drop(columns=['Delay_0', 'Delay_1', 'Delay_2'])
        crossTraffic_dfs[flow] = crossTraffic_dfs[flow][crossTraffic_dfs[flow]['Delay'] > 0]

def clear_data_from_outliers_in_time(endToEnd_dfs, crossTraffic_dfs, switches_dfs):
    for switch in switches_dfs.keys():
        per_traffic_data = []
        for flow in endToEnd_dfs.keys():
            per_traffic_data.append(pd.merge(endToEnd_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay']), switches_dfs[switch], on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb'], how='inner'))
        for flow in crossTraffic_dfs.keys():
            per_traffic_data.append(pd.merge(crossTraffic_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay']), switches_dfs[switch], on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb'], how='inner'))
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


def analyze_single_experiment(rate, steadyStart, steadyEnd, confidenceValue, rounds_results, experiment=0, ns3_path=__ns3_path):
    endToEnd_dfs = read_data(__ns3_path, steadyStart, steadyEnd, rate, 'EndToEnd', 'IsReceived', 'SentTime', str(experiment), True)
    crossTraffic_dfs = read_data(__ns3_path, steadyStart, steadyEnd, rate, 'EndToEnd_crossTraffic', 'IsReceived', 'SentTime', str(experiment), True)
    switches_dfs = read_data(__ns3_path, steadyStart, steadyEnd, rate, 'Switch', 'IsSent', 'ReceiveTime', str(experiment), True)
    samples_dfs = read_data(__ns3_path, steadyStart, steadyEnd, rate, 'PoissonSampler', 'IsDeparted', 'SampleTime', str(experiment), False)

    endToEnd_dataRates = {}
    for flow in endToEnd_dfs.keys():
        endToEnd_dataRates[flow] = endToEnd_dfs[flow]['PayloadSize'].sum() * 8 / (10) / 1000000
    print(endToEnd_dataRates)

    endToEnd_dataRates = {}
    for flow in crossTraffic_dfs.keys():
        endToEnd_dataRates[flow] = crossTraffic_dfs[flow]['PayloadSize'].sum() * 8 / (10) / 1000000
    print(endToEnd_dataRates)

    # samples_highRate_dfs = read_data(__ns3_path, steadyStart, steadyEnd, rate, 'PoissonSampler_highRate', 'IsDeparted', 'SampleTime', str(experiment), False)
    # samples_regular_dfs = read_data(__ns3_path, steadyStart, steadyEnd, rate, 'RegularSampler', 'IsDeparted', 'SampleTime', str(experiment), False)
    # # read and append the drop rate from the last line of the sample file T0
    # with open('{}/scratch/Results/{}/{}/T0T1_PoissonSampler.csv'.format(__ns3_path, rate, str(experiment))) as f:
    #     lines = f.readlines()
    #     rounds_results['DropRate'].append(float(lines[-1].split(':')[1]))

    rounds_results['DropRate'].append(calculate_drop_rate(__ns3_path, steadyStart, steadyEnd, rate, 'Switch', 'IsSent', 'ReceiveTime', str(experiment))['T0'])
    # Intermediate links groundtruth statistics
    remove_interlinks_trasmission_delay(endToEnd_dfs, crossTraffic_dfs, switches_dfs)

    # integrate the switch data with the endToEnd data
    # switch_different_traffics_delaymean(switches_dfs['T0'])
    clear_data_from_outliers_in_time(endToEnd_dfs, crossTraffic_dfs, switches_dfs)
    # switch_different_traffics_delaymean(switches_dfs['T0'])

    # check_grountruth_delayConsistency(endToEnd_dfs, switches_dfs)
    # switch_different_traffics_delaymean(switches_dfs['T0'])

    # samples switches statistics
    # highRate_samples_statistics = {}
    # highRate_samples_statistics['Overall'] = {}
    # regular_samples_statistics = {}
    # regular_samples_statistics['Overall'] = {}
    samples_statistics = {}
    samples_statistics['Overall'] = {}
    samples_statistics['PerTrafficStream'] = {}
    samples_statistics['RemovedZeroes'] = {}
    common_switch_sample_df = get_switch_samples_delays(switches_dfs['T0'], samples_dfs['T0T1'])
    # common_switch_highRate_sample_df = get_switch_samples_delays(switches_dfs['T0'], samples_highRate_dfs['T0T1'])
    # common_switch_regular_sample_df = get_switch_samples_delays(switches_dfs['T0'], samples_regular_dfs['T0T1'])
    nonCommon_switch_sample_df = {}
    # nonCommin_switch_highRate_sample_df = {}
    # nonCommon_switch_regular_sample_df = {}
    for flow in endToEnd_dfs.keys():
        samples_statistics['Overall'][flow] = {}
        samples_statistics['Overall'][flow]['T0'] = get_statistics(common_switch_sample_df)
        samples_statistics['RemovedZeroes'][flow] = {}
        samples_statistics['RemovedZeroes'][flow]['T0'] = get_statistics(common_switch_sample_df, True)
        # highRate_samples_statistics['Overall'][flow] = {}
        # highRate_samples_statistics['Overall'][flow]['T0'] = get_statistics(common_switch_highRate_sample_df)
        # regular_samples_statistics['Overall'][flow] = {}
        # regular_samples_statistics['Overall'][flow]['T0'] = get_statistics(common_switch_regular_sample_df)
        nonCommon_switch_sample_df[flow] = get_switch_samples_delays(switches_dfs['T1'], samples_dfs['T1.R' + flow.split('R')[-1]])
        # nonCommin_switch_highRate_sample_df[flow] = get_switch_samples_delays(switches_dfs['T1'], samples_highRate_dfs['T1.R' + flow.split('R')[-1]])
        # nonCommon_switch_regular_sample_df[flow] = get_switch_samples_delays(switches_dfs['T1'], samples_regular_dfs['T1.R' + flow.split('R')[-1]])
        samples_statistics['Overall'][flow]['T1'] = get_statistics(nonCommon_switch_sample_df[flow])
        samples_statistics['RemovedZeroes'][flow]['T1'] = get_statistics(nonCommon_switch_sample_df[flow], True)
        # highRate_samples_statistics['Overall'][flow]['T1'] = get_statistics(nonCommin_switch_highRate_sample_df[flow])
        # regular_samples_statistics['Overall'][flow]['T1'] = get_statistics(nonCommon_switch_regular_sample_df[flow])

        # samples_statistics['PerTrafficStream'][flow] = {}
        # samples_statistics['PerTrafficStream'][flow]['T0'] = get_statistics(switch_data(endToEnd_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay']), switches_dfs['T0'], True))
        # samples_statistics['PerTrafficStream'][flow]['T1'] = get_statistics(switch_data(endToEnd_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay']), switches_dfs['T1'], True))

    # groundtruth switches statistics
    groundtruth_statistics = {}
    groundtruth_statistics['Overall'] = {}
    groundtruth_statistics['PerTrafficStream'] = {}
    groundtruth_statistics['RemovedZeroes'] = {}

    flows = []
    flows_sampled = []
    # switch_different_traffics_delaymean(switches_dfs['T0'])
    for flow in endToEnd_dfs.keys():
        groundtruth_statistics['Overall'][flow] = {}
        groundtruth_statistics['Overall'][flow]['T0'] = get_statistics(switches_dfs['T0'])
        groundtruth_statistics['Overall'][flow]['T1'] = get_statistics(switches_dfs['T1'])
        # groundtruth_statistics['Overall'][flow]['T0']['sampleSize'] = samples_statistics['Overall'][flow]['T0']['sampleSize']
        # groundtruth_statistics['Overall'][flow]['T1']['sampleSize'] = samples_statistics['Overall'][flow]['T1']['sampleSize']


        groundtruth_statistics['PerTrafficStream'][flow] = {}
        flow_on_switch = switch_data(endToEnd_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay']), switches_dfs['T0'], False)
        flows_sampled.append(switch_data(endToEnd_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay']), switches_dfs['T0'], True))
        flows.append(flow_on_switch)
        
        groundtruth_statistics['PerTrafficStream'][flow]['T0'] = get_statistics(flow_on_switch)
        groundtruth_statistics['PerTrafficStream'][flow]['T1'] = get_statistics(switch_data(endToEnd_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay']), switches_dfs['T1'], False))
        
        groundtruth_statistics['RemovedZeroes'][flow] = {}
        groundtruth_statistics['RemovedZeroes'][flow]['T0'] = get_statistics(switches_dfs['T0'], True)
        groundtruth_statistics['RemovedZeroes'][flow]['T1'] = get_statistics(switch_data(endToEnd_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay']), switches_dfs['T1'], False), True)

        # print(groundtruth_statistics['PerTrafficStream'][flow]['T0']['DelayMean'], samples_statistics['Overall'][flow]['T0']['DelayMean'], samples_statistics['PerTrafficStream'][flow]['T0']['DelayMean'])
        # print("T0 groundtruth_statistics['PerTrafficStream']: ", groundtruth_statistics['PerTrafficStream'][flow]['T0']['DelayMean'])
        # print("T0 regular_samples_statistics['Overall']: ", regular_samples_statistics['Overall'][flow]['T0']['DelayMean'])
        # print("T0 RemovedZeroes: ", groundtruth_statistics['RemovedZeroes'][flow]['T0']['DelayMean'])

        # print("T0 samples_statistics['Overall']: ", samples_statistics['Overall'][flow]['T0']['DelayMean'])
        # print("T0 highRate_samples_statistics['Overall']: ", highRate_samples_statistics['Overall'][flow]['T0']['DelayMean'])
        # print("T0 RemovedZeroes: ", samples_statistics['RemovedZeroes'][flow]['T0']['DelayMean'])
        # print()
        
        if abs(groundtruth_statistics['Overall'][flow]['T0']['DelayMean'] - samples_statistics['Overall'][flow]['T0']['DelayMean']) <= confidenceValue * samples_statistics['Overall'][flow]['T0']['DelayStd'] / np.sqrt(samples_statistics['Overall'][flow]['T0']['sampleSize']):
            rounds_results['T0Ineq'][flow] += 1
        # if abs(groundtruth_statistics['Overall'][flow]['T0']['DelayMean'] - highRate_samples_statistics['Overall'][flow]['T0']['DelayMean']) <= confidenceValue * highRate_samples_statistics['Overall'][flow]['T0']['DelayStd'] / np.sqrt(highRate_samples_statistics['Overall'][flow]['T0']['sampleSize']):
        #     rounds_results['T0IeqHighRate'][flow] += 1
        # if abs(regular_samples_statistics['Overall'][flow]['T0']['DelayMean'] - samples_statistics['Overall'][flow]['T0']['DelayMean']) <= confidenceValue * samples_statistics['Overall'][flow]['T0']['DelayStd'] / np.sqrt(samples_statistics['Overall'][flow]['T0']['sampleSize']):
        #     rounds_results['T0IneqRegular'][flow] += 1
        if abs(groundtruth_statistics['RemovedZeroes'][flow]['T0']['DelayMean'] - samples_statistics['RemovedZeroes'][flow]['T0']['DelayMean']) <= confidenceValue * samples_statistics['RemovedZeroes'][flow]['T0']['DelayStd'] / np.sqrt(samples_statistics['RemovedZeroes'][flow]['T0']['sampleSize']):  
            rounds_results['T0IneqRemovedZeroes'][flow] += 1
        
        # print("T1 groundtruth_statistics['PerTrafficStream']: ", groundtruth_statistics['PerTrafficStream'][flow]['T1']['DelayMean'])
        # print("T1 regular_samples_statistics['Overall']: ", regular_samples_statistics['Overall'][flow]['T1']['DelayMean'])
        # print("T1 RemovedZeroes: ", groundtruth_statistics['RemovedZeroes'][flow]['T1']['DelayMean'])

        # print("T1 samples_statistics['Overall']: ", samples_statistics['Overall'][flow]['T1']['DelayMean'])
        # print("T1 highRate_samples_statistics['Overall']: ", highRate_samples_statistics['Overall'][flow]['T1']['DelayMean'])
        # print("T1 RemovedZeroes: ", samples_statistics['RemovedZeroes'][flow]['T1']['DelayMean'])
        # print("***********************")

        if abs(groundtruth_statistics['PerTrafficStream'][flow]['T1']['DelayMean'] - samples_statistics['Overall'][flow]['T1']['DelayMean']) <= confidenceValue * samples_statistics['Overall'][flow]['T1']['DelayStd'] / np.sqrt(samples_statistics['Overall'][flow]['T1']['sampleSize']):
            # print("T1 Overall: ", True)
            rounds_results['T1Ineq'][flow] += 1
        # if abs(groundtruth_statistics['PerTrafficStream'][flow]['T1']['DelayMean'] - highRate_samples_statistics['Overall'][flow]['T1']['DelayMean']) <= confidenceValue * highRate_samples_statistics['Overall'][flow]['T1']['DelayStd'] / np.sqrt(highRate_samples_statistics['Overall'][flow]['T1']['sampleSize']):
        #     rounds_results['T1IeqHighRate'][flow] += 1
        # if abs(regular_samples_statistics['Overall'][flow]['T1']['DelayMean'] - samples_statistics['Overall'][flow]['T1']['DelayMean']) <= confidenceValue * samples_statistics['Overall'][flow]['T1']['DelayStd'] / np.sqrt(samples_statistics['Overall'][flow]['T1']['sampleSize']):
        #     rounds_results['T1IneqRegular'][flow] += 1
        if abs(groundtruth_statistics['RemovedZeroes'][flow]['T1']['DelayMean'] - samples_statistics['RemovedZeroes'][flow]['T1']['DelayMean']) <= confidenceValue * samples_statistics['RemovedZeroes'][flow]['T1']['DelayStd'] / np.sqrt(samples_statistics['RemovedZeroes'][flow]['T1']['sampleSize']):
            rounds_results['T1IneqRemovedZeroes'][flow] += 1

        segments_delayMeans = [value['DelayMean'] for value in samples_statistics['Overall'][flow].values()]
        segments_delayStds = [value['DelayStd'] for value in samples_statistics['Overall'][flow].values()]
        segments_sampleSizes = [value['sampleSize'] for value in samples_statistics['Overall'][flow].values()]
        epsilons = [confidenceValue * (segments_delayStds[i] / (np.sqrt(segments_sampleSizes[i]) * segments_delayMeans[i])) for i in range(len(segments_delayMeans))]
        maxEpsilon = max(epsilons)
        if abs(groundtruth_statistics['Overall'][flow]['T0']['DelayMean'] - samples_statistics['Overall'][flow]['T0']['DelayMean']) / samples_statistics['Overall'][flow]['T0']['DelayMean'] <= maxEpsilon:
            rounds_results['T0IneqMaxEpsilon'][flow] += 1 
        if abs(groundtruth_statistics['PerTrafficStream'][flow]['T1']['DelayMean']- samples_statistics['Overall'][flow]['T1']['DelayMean']) / samples_statistics['Overall'][flow]['T1']['DelayMean'] <= maxEpsilon:
            rounds_results['T1IneqMaxEpsilon'][flow] += 1

    # endToEnd_statistics
    endToEnd_statistics = {}
    rounds_results['T0std'].append(samples_statistics['Overall']['R0h0R1h0']['T0']['DelayStd'])
    for flow in endToEnd_dfs.keys():
        endToEnd_statistics[flow] = get_statistics(endToEnd_dfs[flow])
        rounds_results['EndToEndMean'][flow].append(endToEnd_statistics[flow]['DelayMean'])
        rounds_results['EndToEndSkew'][flow].append(endToEnd_statistics[flow]['DelaySkew'])
        rounds_results['EndToEndStd'][flow].append(endToEnd_statistics[flow]['DelayStd'])
        rounds_results['T1std'][flow].append(samples_statistics['Overall'][flow]['T1']['DelayStd'])

    rounds_results['experiments'] += 1
    delayProcess_consistency_check(flows, flows_sampled, rounds_results)
    
    if experiment == 0:
        plot_overall_delay_distribution(rate, switches_dfs, common_switch_sample_df)
        plot_seperate_delay_distribution(rate, flows)
        plot_endToEnd_delay_distribution(rate, endToEnd_dfs)
        for flow in endToEnd_dfs.keys():
            plot_overall_delay_distribution_noncommonSwitch(rate, 
                                                            pd.merge(switches_dfs['T1'], endToEnd_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay']), on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb'], how='inner'), 
                                                            nonCommon_switch_sample_df[flow], flow)

    compatibility_check(confidenceValue, rounds_results, samples_statistics, endToEnd_statistics, groundtruth_statistics, endToEnd_dfs.keys())

def analyze_all_experiments(rate, steadyStart, steadyEnd, confidenceValue, experiments=3, ns3_path=__ns3_path):
    flows_name = read_data_flowIndicator(ns3_path, rate)

    rounds_results = prepare_results(flows_name)
    for experiment in range(experiments):
        if len(os.listdir('{}/scratch/Results/{}/{}'.format(__ns3_path, rate, experiment))) == 0:
            print(experiment)
            continue
        analyze_single_experiment(rate, steadyStart, steadyEnd, confidenceValue, rounds_results, experiment, ns3_path)

    with open('results/{}/{}_results.json'.format(rate,rate), 'w') as f:
        # save the results in a well formatted json file
        js.dump(rounds_results, f, indent=4)

# main function
def __main__():
    config = configparser.ConfigParser()
    config.read('Parameters.config')
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
    serviceRateScales = [0.8, 0.85, 0.9, 0.95, 1.0]
    # experiments = 10
    # steadyStart = 4
    # steadyEnd = 9

    for rate in serviceRateScales:
        print("\nAnalyzing experiments for rate: ", rate)
        analyze_all_experiments(rate, steadyStart, steadyEnd, confidenceValue, experiments, __ns3_path)
        print("Rate {} done".format(rate))

__main__()