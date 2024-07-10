from Utils import *
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
import multiprocessing
import argparse

# __ns3_path = os.popen('locate "ns-3.41" | grep /ns-3.41$').read().splitlines()[0]
__ns3_path = "/home/shossein/ns-allinone-3.41/ns-3.41"
errorRate = 0.05
difference = 1.30
sample_rate = 0.05
confidenceValue = 1.96 # 95% confidence interval

def check_dominant_bottleneck_consistency(endToEnd_statistics, samples_paths_aggregated_statistics, confidenceValue):
    # print(abs(endToEnd_statistics['timeAvg'] - samples_paths_aggregated_statistics['DelayMean']), confidenceValue * (endToEnd_statistics['DelayStd'] * np.sqrt(1 / samples_paths_aggregated_statistics['MinSampleSize'])))
    sampling_error = confidenceValue * (endToEnd_statistics['DelayStd'] * np.sqrt(1 / endToEnd_statistics['sampleSize']))
    if abs(endToEnd_statistics['DelayMean'] - samples_paths_aggregated_statistics['DelayMean']) <= sampling_error + (confidenceValue * (endToEnd_statistics['DelayStd'] * np.sqrt(1 / samples_paths_aggregated_statistics['MinSampleSize']))):
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
    sampling_error = confidenceValue * (endToEnd_statistics['DelayStd'] * np.sqrt(1 / endToEnd_statistics['sampleSize']))
    if abs(endToEnd_statistics['DelayMean'] - samples_paths_aggregated_statistics['DelayMean']) <= sampling_error + samples_paths_aggregated_statistics['SumOfErrors']:
        return True
    else:
        return False
    
def check_all_delayConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, confidenceValue, paths):
    res = {}
    res['DominantAssumption'] = {}
    res['MaxEpsilonIneq'] = {}
    res['Basic'] = {}
    for flow in endToEnd_statistics.keys():
        res['DominantAssumption'][flow] = {}
        res['MaxEpsilonIneq'][flow] = {}
        res['Basic'][flow] = {}
        for path in paths:
            # print(flow, path)
            res['DominantAssumption'][flow][path] = check_dominant_bottleneck_consistency(endToEnd_statistics[flow][path], samples_paths_aggregated_statistics[flow][path], confidenceValue)
            res['MaxEpsilonIneq'][flow][path] = check_MaxEpsilon_ineq(endToEnd_statistics[flow][path], samples_paths_aggregated_statistics[flow][path], confidenceValue)
            res['Basic'][flow][path] = check_basic_delayConsistency(endToEnd_statistics[flow][path], samples_paths_aggregated_statistics[flow][path], confidenceValue)
    return res

def prepare_results(flows, queues, num_of_agg_switches):
    rounds_results = {}
    rounds_results['DominantAssumption'] = {}
    rounds_results['MaxEpsilonIneq'] = {}
    rounds_results['Basic'] = {}
    rounds_results['EndToEndMean'] = {}
    rounds_results['EndToEndStd'] = {}
    rounds_results['DropRate'] = []
    rounds_results['maxEpsilon'] = {}

    for q in queues:
        if q[0] == 'T' and q[2] == 'H' and (q[1] == '2' or q[1] == '3'):
            rounds_results[q+'std'] = []
        if q[0] == 'T' and q[2] == 'A' and (q[1] == '0' or q[1] == '1'):
            rounds_results[q+'std'] = []
        if q[0] == 'A' and q[2] == 'T' and (q[3] == '2' or q[3] == '3'):
            rounds_results[q+'std'] = []

    for flow in flows:
        rounds_results['DominantAssumption'][flow] = {}
        rounds_results['MaxEpsilonIneq'][flow] = {}
        rounds_results['Basic'][flow] = {}
        rounds_results['EndToEndMean'][flow] = {}
        rounds_results['EndToEndStd'][flow] = {}
        rounds_results['maxEpsilon'][flow] = {}

        for i in range(num_of_agg_switches):
            rounds_results['DominantAssumption'][flow]['A' + str(i)] = 0
            rounds_results['MaxEpsilonIneq'][flow]['A' + str(i)] = 0
            rounds_results['Basic'][flow]['A' + str(i)] = 0
            rounds_results['EndToEndMean'][flow]['A' + str(i)] = []
            rounds_results['EndToEndStd'][flow]['A' + str(i)] = []
            rounds_results['maxEpsilon'][flow]['A' + str(i)] = []

    rounds_results['experiments'] = 0
    return rounds_results

def compatibility_check(confidenceValue, rounds_results, samples_paths_aggregated_statistics, endToEnd_statistics, flows_name, paths):
    # End to End and Persegment Compatibility Check
    results = check_all_delayConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, confidenceValue, paths)

    for flow in flows_name:
        for path in paths:
            if results['DominantAssumption'][flow][path]:
                rounds_results['DominantAssumption'][flow][path] += 1
            if results['MaxEpsilonIneq'][flow][path]:
                rounds_results['MaxEpsilonIneq'][flow][path] += 1
            if results['Basic'][flow][path]:
                rounds_results['Basic'][flow][path] += 1

def remove_interlinks_trasmission_delay(endToEnd_dfs, switches_dfs, start_dfs, aggSwitchesNum):
    for flow in endToEnd_dfs.keys():
        host_switch = 'R' + flow[1] + 'H' + flow[3]
        src_Tor_swich = 'T' + flow[1]
        Agg_switch = ['A' + str(i) for i in range(aggSwitchesNum)]
        Tor_dest_switch = 'T' + flow[5]
        endToEnd_dfs[flow] = pd.merge(endToEnd_dfs[flow].drop(columns=['SentTime']), start_dfs[host_switch].drop(columns=['SentTime', 'ECN']).rename(columns={'ReceiveTime': 'SentTime'}), on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb', 'Id'], how='inner')
        endToEnd_dfs[flow]['Delay'] = endToEnd_dfs[flow]['ReceiveTime'] - endToEnd_dfs[flow]['SentTime']
        src_Tor = intermediateLink_transmission(endToEnd_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay', 'Path', 'ECN', 'PacketSize'], ), start_dfs[host_switch], switches_dfs[src_Tor_swich], 0)
        Tor_Agg = []
        Agg_Tor = []
        for i in range(aggSwitchesNum):
            Tor_Agg.append(intermediateLink_transmission(endToEnd_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay', 'Path', 'ECN', 'PacketSize']), switches_dfs[src_Tor_swich], switches_dfs[Agg_switch[i]], 1))
            Agg_Tor.append(intermediateLink_transmission(endToEnd_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay', 'Path', 'ECN', 'PacketSize']), switches_dfs[Agg_switch[i]], switches_dfs[Tor_dest_switch], 2))
        Tor_Agg = pd.concat(Tor_Agg)
        Agg_Tor = pd.concat(Agg_Tor)
        Tor_dst = intermediateLink_transmission(endToEnd_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay', 'Path', 'ECN', 'PacketSize']), switches_dfs[Tor_dest_switch], endToEnd_dfs[flow].drop(columns=['Delay', 'Path', 'ECN', 'PacketSize']), 3)
        endToEnd_dfs[flow] = pd.merge(endToEnd_dfs[flow], src_Tor, on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb', 'Id'], how='inner')
        endToEnd_dfs[flow] = pd.merge(endToEnd_dfs[flow], Tor_Agg, on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb', 'Id'], how='inner')
        endToEnd_dfs[flow] = pd.merge(endToEnd_dfs[flow], Agg_Tor, on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb', 'Id'], how='inner')
        endToEnd_dfs[flow] = pd.merge(endToEnd_dfs[flow], Tor_dst, on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb', 'Id'], how='inner')
        print(endToEnd_dfs[flow])
        endToEnd_dfs[flow]['Delay'] = endToEnd_dfs[flow]['Delay'] - sum([endToEnd_dfs[flow]['Delay_' + str(i)] for i in range(4)])
        endToEnd_dfs[flow] = endToEnd_dfs[flow].drop(columns=['Delay_0', 'Delay_1', 'Delay_2', 'Delay_3'])
        endToEnd_dfs[flow] = endToEnd_dfs[flow][endToEnd_dfs[flow]['Delay'] > 0]

def analyze_single_experiment(return_dict, rate, steadyStart, steadyEnd, confidenceValue, rounds_results, queues_names, results_folder, experiment=0, ns3_path=__ns3_path):
    np.random.seed(seed=experiment)
    num_of_agg_switches = 2
    paths = ['A' + str(i) for i in range(num_of_agg_switches)]
    endToEnd_dfs = read_data(__ns3_path, steadyStart, steadyEnd, rate, 'EndToEnd', 'IsReceived', 'SentTime', str(experiment), True, results_folder)
    switches_dfs = read_data(__ns3_path, steadyStart, steadyEnd + 0.5, rate, 'Switch', 'IsSent', 'ReceiveTime', str(experiment), True, results_folder)
    samples_dfs = read_data(__ns3_path, steadyStart, steadyEnd, rate, 'PoissonSampler', 'IsDeparted', 'SampleTime', str(experiment), False, results_folder)
    start_dfs = read_data(__ns3_path, steadyStart, steadyEnd + 0.5, rate, 'start', 'IsSent', 'ReceiveTime', str(experiment), True, results_folder)

    # print_traffic_rate(endToEnd_dfs)
    # rounds_results['DropRate'].append(calculate_drop_rate(__ns3_path, steadyStart, steadyEnd, rate, ['Switch', 'start'], 'IsSent', 'ReceiveTime', str(experiment), results_folder))
    rounds_results['DropRate'].append(calculate_drop_rate(__ns3_path, steadyStart, steadyEnd, rate, ['EndToEnd'], 'IsReceived', 'SentTime', str(experiment), results_folder))

    # integrate the switch data with the endToEnd data
    clear_data_from_outliers_in_time(endToEnd_dfs, switches_dfs, start_dfs)

    # Intermediate links groundtruth statistics
    remove_interlinks_trasmission_delay(endToEnd_dfs, switches_dfs, start_dfs, num_of_agg_switches)

    # # samples switches statistics
    # samples_switches_statistics = {}
    # samples_queues_dfs = {}
    # for sample_df in samples_dfs.keys():
    #     # print(sample_df, sample_df[0:2])
    #     if 'R' in sample_df:
    #         samples_queues_dfs[sample_df] = get_switch_samples_delays(start_dfs[sample_df], samples_dfs[sample_df])
    #     else:    
    #         samples_queues_dfs[sample_df] = get_switch_samples_delays(switches_dfs[sample_df[0:2]], samples_dfs[sample_df])

    #     samples_switches_statistics[sample_df] = get_statistics(samples_queues_dfs[sample_df])
    #     # print(sample_df, samples_switches_statistics[sample_df]['DelayMean'])

    # # samples_paths_statistics
    # samples_paths_aggregated_statistics = {}
    # for flow in endToEnd_dfs.keys():
    #     samples_paths_aggregated_statistics[flow] = {}
    #     for path in paths:
    #         samples_paths_aggregated_statistics[flow][path] = {}
    #         samples_paths_aggregated_statistics[flow][path]['DelayMean'] = sum([samples_switches_statistics['R' + flow[1] + 'H' + flow[3]]['DelayMean'],
    #                                                                             samples_switches_statistics['T' + flow[1] + path]['DelayMean'], 
    #                                                                             samples_switches_statistics[path + 'T' + flow[5]]['DelayMean'],
    #                                                                             samples_switches_statistics['T' + flow[5] + 'H' + flow[7]]['DelayMean']])
        
    #         samples_paths_aggregated_statistics[flow][path]['MinSampleSize'] = min([samples_switches_statistics['R' + flow[1] + 'H' + flow[3]]['sampleSize'],
    #                                                                                 samples_switches_statistics['T' + flow[1] + path]['sampleSize'],
    #                                                                                 samples_switches_statistics[path + 'T' + flow[5]]['sampleSize'],
    #                                                                                 samples_switches_statistics['T' + flow[5] + 'H' + flow[7]]['sampleSize']])
            
    #         samples_paths_aggregated_statistics[flow][path]['MaxEpsilon'] = max([calc_epsilon(confidenceValue, samples_switches_statistics['R' + flow[1] + 'H' + flow[3]]),
    #                                                                              calc_epsilon(confidenceValue, samples_switches_statistics['T' + flow[1] + path]),
    #                                                                              calc_epsilon(confidenceValue, samples_switches_statistics[path + 'T' + flow[5]]),
    #                                                                              calc_epsilon(confidenceValue, samples_switches_statistics['T' + flow[5] + 'H' + flow[7]])])
            
    #         samples_paths_aggregated_statistics[flow][path]['SumOfErrors'] = sum([calc_error(confidenceValue, samples_switches_statistics['R' + flow[1] + 'H' + flow[3]]),
    #                                                                               calc_error(confidenceValue, samples_switches_statistics['T' + flow[1] + path]),
    #                                                                               calc_error(confidenceValue, samples_switches_statistics[path + 'T' + flow[5]]),
    #                                                                               calc_error(confidenceValue, samples_switches_statistics['T' + flow[5] + 'H' + flow[7]])])
    # # endToEnd_statistics
    # endToEnd_statistics = {}
    # for flow in endToEnd_dfs.keys():
    #     endToEnd_statistics[flow] = {}
    #     new_endToEnd = []
    #     for path in paths:
    #         # remove A from the path
    #         temp = endToEnd_dfs[flow][endToEnd_dfs[flow]['Path'] == int(path[1])]
    #         if flow == 'R0H0R2H0' or flow == 'R0H1R2H1':
    #         # add the error packets
    #             rows_to_change = np.random.choice(temp.index, int(len(temp) * 6 * errorRate), replace=False)
    #             temp.loc[rows_to_change, 'Delay'] = (temp.loc[rows_to_change, 'Delay'] * difference).astype(int)

    #         endToEnd_statistics[flow][path] = get_statistics(temp, timeAvg=True)
    #         rounds_results['EndToEndMean'][flow][path].append(endToEnd_statistics[flow][path]['timeAvg'])
    #         rounds_results['EndToEndStd'][flow][path].append(endToEnd_statistics[flow][path]['DelayStd'])
    #         rounds_results['maxEpsilon'][flow][path].append(samples_paths_aggregated_statistics[flow][path]['MaxEpsilon'])
    #         new_endToEnd.append(temp)
    #     endToEnd_dfs[flow] = pd.concat(new_endToEnd)

    # rounds_results['experiments'] += 1

    # for q in queues_names:
    #     if q[0] == 'T' and q[2] == 'H' and (q[1] == '2' or q[1] == '3'):
    #         rounds_results[q+'std'].append(samples_switches_statistics[q]['DelayStd'])
    #     if q[0] == 'T' and q[2] == 'A' and (q[1] == '0' or q[1] == '1'):
    #         rounds_results[q+'std'].append(samples_switches_statistics[q]['DelayStd'])
    #     if q[0] == 'A' and q[2] == 'T' and (q[3] == '2' or q[3] == '3'):
    #         rounds_results[q+'std'].append(samples_switches_statistics[q]['DelayStd'])
    
    # compatibility_check(confidenceValue, rounds_results, samples_paths_aggregated_statistics, endToEnd_statistics, endToEnd_dfs.keys(), ['A' + str(i) for i in range(num_of_agg_switches)])

    # if experiment == 0:
    #     plot_delay_over_time(endToEnd_dfs, paths, rate, results_folder)

    # return_dict[experiment] = rounds_results

def merge_results(return_dict, merged_results, flows, queues):
    num_of_agg_switches = 2
    for exp in return_dict.keys():
        for q in queues:
            if q[0] == 'T' and q[2] == 'H' and (q[1] == '2' or q[1] == '3'):
                merged_results[q+'std'] += return_dict[exp][q+'std']
            if q[0] == 'T' and q[2] == 'A' and (q[1] == '0' or q[1] == '1'):
                merged_results[q+'std'] += return_dict[exp][q+'std']
            if q[0] == 'A' and q[2] == 'T' and (q[3] == '2' or q[3] == '3'):
                merged_results[q+'std'] += return_dict[exp][q+'std']

    for flow in flows:
        for i in range(num_of_agg_switches):
            for exp in return_dict.keys():
                merged_results['DominantAssumption'][flow]['A' + str(i)] += return_dict[exp]['DominantAssumption'][flow]['A' + str(i)]
                merged_results['MaxEpsilonIneq'][flow]['A' + str(i)] += return_dict[exp]['MaxEpsilonIneq'][flow]['A' + str(i)]
                merged_results['Basic'][flow]['A' + str(i)] += return_dict[exp]['Basic'][flow]['A' + str(i)]
                merged_results['EndToEndMean'][flow]['A' + str(i)] += return_dict[exp]['EndToEndMean'][flow]['A' + str(i)]
                merged_results['EndToEndStd'][flow]['A' + str(i)] += return_dict[exp]['EndToEndStd'][flow]['A' + str(i)]
                merged_results['maxEpsilon'][flow]['A' + str(i)] += return_dict[exp]['maxEpsilon'][flow]['A' + str(i)]
    for exp in return_dict.keys():
        merged_results['experiments'] += return_dict[exp]['experiments']
        merged_results['DropRate'] += return_dict[exp]['DropRate']
    
def analyze_all_experiments(rate, steadyStart, steadyEnd, confidenceValue, dir, experiments_end=3, ns3_path=__ns3_path):
    results_folder = 'Results_' + dir
    num_of_agg_switches = 2
    flows_name = read_data_flowIndicator(ns3_path, rate, results_folder)
    flows_name.sort()

    queues_names = read_queues_indicators(ns3_path, rate, results_folder)
    queues_names.sort()

    rounds_results = prepare_results(flows_name, queues_names, num_of_agg_switches)
    merged_results = prepare_results(flows_name, queues_names, num_of_agg_switches)
    for i in range(int(experiments_end / 10) + 1):
        ths = []
        return_dict = multiprocessing.Manager().dict()
        for experiment in range(10 * i, min(experiments_end, 10 * (i + 1))):
            if len(os.listdir('{}/scratch/{}/{}/{}'.format(__ns3_path, results_folder, rate, experiment))) == 0:
                print(experiment)
                continue
            print("Analyzing experiment: ", experiment)
            ths.append(multiprocessing.Process(target=analyze_single_experiment, args=(return_dict, rate, steadyStart, steadyEnd, confidenceValue, rounds_results, queues_names, results_folder, experiment, ns3_path)))
        
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        merge_results(return_dict, merged_results, flows_name, queues_names)
        print("{} joind".format(i))
    
    with open('../results_{}/{}/delay_{}_{}_{}_to_{}.json'.format(dir, rate, results_folder, experiments_end, steadyStart, steadyEnd), 'w') as f:
        js.dump(merged_results, f, indent=4)

# main function
def __main__():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dir",
                    required=True,
                    dest="dir",
                    help="The directory of the results",
                    default="")

    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read('../Parameters.config')
    steadyStart = convert_to_float(config.get('Settings', 'steadyStart'))
    steadyEnd = convert_to_float(config.get('Settings', 'steadyEnd'))
    experiments = int(config.get('Settings', 'experiments'))
    serviceRateScales = [float(x) for x in config.get('Settings', 'serviceRateScales').split(',')]

    serviceRateScales = [1.0]
    experiments = 1

    for rate in serviceRateScales:
        print("\nAnalyzing experiments for rate: ", rate)
        analyze_all_experiments(rate, steadyStart, steadyEnd, confidenceValue, args.dir, experiments_end=experiments, ns3_path=__ns3_path)
        print("Rate {} {} done".format(rate, experiments))

__main__()