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

__ns3_path = os.popen('locate "ns-3.41" | grep /ns-3.41$').read().splitlines()[0]
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
        rounds_results['DominantAssumption'][flow] = {}
        rounds_results['MaxEpsilonIneq'][flow] = {}
        rounds_results['Basic'][flow] = {}
        rounds_results['EndToEndMean'][flow] = {}
        rounds_results['EndToEndStd'][flow] = {}

        for i in range(num_of_agg_switches):
            rounds_results['DominantAssumption'][flow]['A' + str(i)] = 0
            rounds_results['MaxEpsilonIneq'][flow]['A' + str(i)] = 0
            rounds_results['Basic'][flow]['A' + str(i)] = 0
            rounds_results['EndToEndMean'][flow]['A' + str(i)] = []
            rounds_results['EndToEndStd'][flow]['A' + str(i)] = []

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
        endToEnd_dfs[flow] = pd.merge(endToEnd_dfs[flow].drop(columns=['SentTime']), start_dfs[host_switch].drop(columns=['SentTime']).rename(columns={'ReceiveTime': 'SentTime'}), on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb', 'Id'], how='inner')
        endToEnd_dfs[flow]['Delay'] = endToEnd_dfs[flow]['ReceiveTime'] - endToEnd_dfs[flow]['SentTime']
        src_Tor = intermediateLink_transmission(endToEnd_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay']), start_dfs[host_switch], switches_dfs[src_Tor_swich], 0)
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

def check_manual_delay_consistency(endToEnd_dfs, switches_dfs, start_dfs, aggSwitchesNum):
    for flow in endToEnd_dfs.keys():
        host_switch = 'R' + flow[1] + 'H' + flow[3]
        src_Tor_swich = 'T' + flow[1]
        Agg_switch = ['A' + str(i) for i in range(aggSwitchesNum)]
        Tor_dest_switch = 'T' + flow[5]
        host_switch = interSwitch_queuing(endToEnd_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay']), start_dfs[host_switch], 0)
        src_Tor_swich = interSwitch_queuing(endToEnd_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay']), switches_dfs[src_Tor_swich], 1)
        Agg = []

        for i in range(aggSwitchesNum):
            Agg.append(interSwitch_queuing(endToEnd_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay']), switches_dfs[Agg_switch[i]], 2))
        # Agg = pd.concat(Agg)
        Tor_dest = interSwitch_queuing(endToEnd_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay']), switches_dfs[Tor_dest_switch], 3)
        
        path_0 = pd.merge(endToEnd_dfs[flow], host_switch, on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb', 'Id'], how='inner')
        path_0 = pd.merge(path_0, src_Tor_swich, on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb', 'Id'], how='inner')
        path_0 = pd.merge(path_0, Agg[0], on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb', 'Id'], how='inner')
        path_0 = pd.merge(path_0, Tor_dest, on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb', 'Id'], how='inner')

        path_1 = pd.merge(endToEnd_dfs[flow], host_switch, on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb', 'Id'], how='inner')
        path_1 = pd.merge(path_1, src_Tor_swich, on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb', 'Id'], how='inner')
        path_1 = pd.merge(path_1, Agg[1], on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb', 'Id'], how='inner')
        path_1 = pd.merge(path_1, Tor_dest, on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb', 'Id'], how='inner')
        if flow == 'R1H0R3H0':
            path_0 = path_0.sort_values(by=['SentTime'])
            path_1 = path_1.sort_values(by=['SentTime'])
            # plot the dealy per sent time for each path on the same figure
            plt.plot(path_0['SentTime'], path_0['Delay'], label='Path 0')
            plt.plot(path_1['SentTime'], path_1['Delay'], label='Path 1')
            # # horizontal line for the mean of the delay and timeAvg
            # plt.axhline(y=path_0['Delay'].mean(), color='r', linestyle='--', label='Path 0 Mean')
            # plt.axhline(y=path_1['Delay'].mean(), color='g', linestyle='--', label='Path 1 Mean')
            # plt.axhline(y=get_timeAvg(path_0), color='b', linestyle='--', label='Path 0 timeAvg')
            # plt.axhline(y=get_timeAvg(path_1), color='y', linestyle='--', label='Path 1 timeAvg')
            # # set the color of the plot be boler
        
            plt.legend()
            plt.xlabel('Sent Time')
            plt.ylabel('Delay')
            plt.legend(prop={'size': 25})
            plt.savefig('../results/Path0_vs_Path1.png')
            print(path_0)
            # print the timweAvg for each path and for eac delay
            # print(get_timeAvg(path_0.drop(columns=['Delay']).rename(columns={'Delay_0': 'Delay'})),
            #         get_timeAvg(path_0.drop(columns=['Delay']).rename(columns={'Delay_1': 'Delay'})),
            #         get_timeAvg(path_0.drop(columns=['Delay']).rename(columns={'Delay_2': 'Delay'})),
            #         get_timeAvg(path_0.drop(columns=['Delay']).rename(columns={'Delay_3': 'Delay'}))
            #         )
            # print("************************************************************************")
            print(path_1)
            # print(get_timeAvg(path_1.drop(columns=['Delay']).rename(columns={'Delay_0': 'Delay'})),
            #         get_timeAvg(path_1.drop(columns=['Delay']).rename(columns={'Delay_1': 'Delay'})),
            #         get_timeAvg(path_1.drop(columns=['Delay']).rename(columns={'Delay_2': 'Delay'})),
            #         get_timeAvg(path_1.drop(columns=['Delay']).rename(columns={'Delay_3': 'Delay'}))
            #         )
            
        


def analyze_single_experiment(rate, steadyStart, steadyEnd, confidenceValue, rounds_results, queues_names, results_folder, experiment=0, ns3_path=__ns3_path):
    num_of_agg_switches = 2
    endToEnd_dfs = read_data(__ns3_path, steadyStart, steadyEnd, rate, 'EndToEnd', 'IsReceived', 'SentTime', str(experiment), True, results_folder)
    switches_dfs = read_data(__ns3_path, steadyStart, steadyEnd, rate, 'Switch', 'IsSent', 'ReceiveTime', str(experiment), True, results_folder)
    samples_dfs = read_data(__ns3_path, steadyStart, steadyEnd, rate, 'PoissonSampler', 'IsDeparted', 'SampleTime', str(experiment), False, results_folder)
    start_dfs = read_data(__ns3_path, steadyStart, steadyEnd, rate, 'start', 'IsSent', 'ReceiveTime', str(experiment), True, results_folder)
    uncorruped_switches_dfs = read_data(__ns3_path, steadyStart, steadyEnd, rate, 'Switch', 'IsSent', 'ReceiveTime', str(experiment), True, 'Results')
    uncorruped_endToEnd_dfs = read_data(__ns3_path, steadyStart, steadyEnd, rate, 'EndToEnd', 'IsReceived', 'SentTime', str(experiment), True, 'Results')

    # print_traffic_rate(endToEnd_dfs)
    rounds_results['DropRate'].append(calculate_drop_rate(__ns3_path, steadyStart, steadyEnd, rate, ['Switch', 'start'], 'IsSent', 'ReceiveTime', str(experiment), results_folder))

    # integrate the switch data with the endToEnd data
    clear_data_from_outliers_in_time(endToEnd_dfs, switches_dfs, start_dfs)

    # Intermediate links groundtruth statistics
    remove_interlinks_trasmission_delay(endToEnd_dfs, switches_dfs, start_dfs, num_of_agg_switches)

    # switch_different_traffics_delaymean(switches_dfs['A0'])
    # switch_different_traffics_delaymean(switches_dfs['A1'])

    paths_flows = {}
    for flow in endToEnd_dfs.keys():
        paths_flows[flow] = read_paths_flows({'A' + str(i): switch_data(uncorruped_endToEnd_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay']), uncorruped_switches_dfs['A' + str(i)], False) for i in range(num_of_agg_switches)}, False)

    test_paths_flows = {}
    for flow in endToEnd_dfs.keys():
        test_paths_flows[flow] = read_paths_flows({'A' + str(i): switch_data(endToEnd_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay']), switches_dfs['A' + str(i)], False) for i in range(num_of_agg_switches)}, True)

    # samples switches statistics
    samples_switches_statistics = {}
    samples_queues_dfs = {}
    for sample_df in samples_dfs.keys():
        # print(sample_df, sample_df[0:2])
        if 'R' in sample_df:
            samples_queues_dfs[sample_df] = get_switch_samples_delays(start_dfs[sample_df], samples_dfs[sample_df])
        else:    
            samples_queues_dfs[sample_df] = get_switch_samples_delays(switches_dfs[sample_df[0:2]], samples_dfs[sample_df])

        samples_switches_statistics[sample_df] = get_statistics(samples_queues_dfs[sample_df])
        # print(sample_df, samples_switches_statistics[sample_df]['DelayMean'])

    # samples_paths_statistics
    samples_paths_aggregated_statistics = {}
    for flow in endToEnd_dfs.keys():
        samples_paths_aggregated_statistics[flow] = {}
        for path in paths_flows[flow].keys():
            samples_paths_aggregated_statistics[flow][path] = {}
            samples_paths_aggregated_statistics[flow][path]['DelayMean'] = sum([samples_switches_statistics['R' + flow[1] + 'H' + flow[3]]['DelayMean'],
                                                                                samples_switches_statistics['T' + flow[1] + path]['DelayMean'], 
                                                                                samples_switches_statistics[path + 'T' + flow[5]]['DelayMean'],
                                                                                samples_switches_statistics['T' + flow[5] + 'H' + flow[7]]['DelayMean']])
        
            samples_paths_aggregated_statistics[flow][path]['MinSampleSize'] = min([samples_switches_statistics['R' + flow[1] + 'H' + flow[3]]['sampleSize'],
                                                                                    samples_switches_statistics['T' + flow[1] + path]['sampleSize'],
                                                                                    samples_switches_statistics[path + 'T' + flow[5]]['sampleSize'],
                                                                                    samples_switches_statistics['T' + flow[5] + 'H' + flow[7]]['sampleSize']])
            
            samples_paths_aggregated_statistics[flow][path]['MaxEpsilon'] = max([calc_epsilon(confidenceValue, samples_switches_statistics['R' + flow[1] + 'H' + flow[3]]),
                                                                                 calc_epsilon(confidenceValue, samples_switches_statistics['T' + flow[1] + path]),
                                                                                 calc_epsilon(confidenceValue, samples_switches_statistics[path + 'T' + flow[5]]),
                                                                                 calc_epsilon(confidenceValue, samples_switches_statistics['T' + flow[5] + 'H' + flow[7]])])
            
            samples_paths_aggregated_statistics[flow][path]['SumOfErrors'] = sum([calc_error(confidenceValue, samples_switches_statistics['R' + flow[1] + 'H' + flow[3]]),
                                                                                  calc_error(confidenceValue, samples_switches_statistics['T' + flow[1] + path]),
                                                                                  calc_error(confidenceValue, samples_switches_statistics[path + 'T' + flow[5]]),
                                                                                  calc_error(confidenceValue, samples_switches_statistics['T' + flow[5] + 'H' + flow[7]])])
    
    # endToEnd_statistics
    endToEnd_statistics = {}
    for flow in endToEnd_dfs.keys():
        endToEnd_statistics[flow] = {}
        for path in paths_flows[flow].keys():
            temp = pd.merge(endToEnd_dfs[flow], paths_flows[flow][path], on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort'], how='inner')
            test_temp = pd.merge(endToEnd_dfs[flow], test_paths_flows[flow][path], on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'SequenceNb', 'Id'], how='inner')
            endToEnd_statistics[flow][path] = get_statistics(temp, timeAvg=True)
            rounds_results['EndToEndMean'][flow][path].append(endToEnd_statistics[flow][path]['timeAvg'])
            rounds_results['EndToEndStd'][flow][path].append(endToEnd_statistics[flow][path]['DelayStd'])
            if flow == 'R0H1R2H1':
                # temp = temp.sort_values(by=['SentTime'])
                # test_temp = test_temp.sort_values(by=['SentTime'])
            #     print(path, len(temp), len(test_temp))
                # plt.plot(temp['SentTime'], temp['Delay'], label='Path {}'.format(path))
                # plt.axhline(y=temp['Delay'].mean(), color='r', linestyle='--', label='Path {} Mean'.format(path))
                # plt.axhline(y=test_temp['Delay'].mean(), color='g', linestyle='--', label='Path {} Test Mean'.format(path))
                # print(path, temp['Delay'].mean(), test_temp['Delay'].mean())
                print(endToEnd_statistics[flow][path])
                print(samples_paths_aggregated_statistics[flow][path])
    # plt.legend()
    # plt.xlabel('Sent Time')
    # plt.ylabel('Delay')
    # plt.legend(prop={'size': 25})
    # plt.savefig('../results/{}.png'.format(results_folder))
    # plt.clf()

    # check_manual_delay_consistency(endToEnd_dfs, switches_dfs, start_dfs, num_of_agg_switches)
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
                flows_sampled[q].append(switch_data(pd.merge(endToEnd_dfs[flow].drop(columns=['SentTime', 'ReceiveTime', 'Delay']), 
                                                             switches_dfs['A' + q[3]].drop(columns=['SentTime', 'ReceiveTime']),
                                                             on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb', 'Id'], how='inner'), 
                                                             switches_dfs[q[0:2]], True))
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
    
    # if experiment == 0:
    #     for sample_df in samples_queues_dfs.keys():
    #         if sample_df[0] == 'T' and sample_df[2] == 'H' and (sample_df[1] == '2' or sample_df[1] == '3'):
    #             plot_overall_delay_distribution(rate, samples_queues_dfs[sample_df], sample_df)
    #         if sample_df[0] == 'T' and sample_df[2] == 'A' and (sample_df[1] == '0' or sample_df[1] == '1'):
    #             plot_overall_delay_distribution(rate, samples_queues_dfs[sample_df], sample_df)
    #         if sample_df[0] == 'A' and sample_df[2] == 'T' and (sample_df[3] == '2' or sample_df[3] == '3'):
    #             plot_overall_delay_distribution(rate, samples_queues_dfs[sample_df], sample_df)
    
    compatibility_check(confidenceValue, rounds_results, samples_paths_aggregated_statistics, endToEnd_statistics, endToEnd_dfs.keys(), ['A' + str(i) for i in range(num_of_agg_switches)])

def analyze_all_experiments(rate, steadyStart, steadyEnd, confidenceValue, experiments_start=0, experiments_end=3, ns3_path=__ns3_path):
    results_folder = 'Results'
    # results_folder = 'Results_delay_normal'
    # results_folder = 'Results_delay_reverse'
    num_of_agg_switches = 2
    flows_name = read_data_flowIndicator(ns3_path, rate, results_folder)
    flows_name.sort()

    queues_names = read_queues_indicators(ns3_path, rate, results_folder)
    queues_names.sort()

    rounds_results = prepare_results(flows_name, queues_names, num_of_agg_switches)

    for experiment in range(experiments_start, experiments_end):
        if len(os.listdir('{}/scratch/{}/{}/{}'.format(__ns3_path, results_folder, rate, experiment))) == 0:
            print(experiment)
            continue
        print("Analyzing experiment: ", experiment)
        analyze_single_experiment(rate, steadyStart, steadyEnd, confidenceValue, rounds_results, queues_names, results_folder, experiment, ns3_path)

    with open('../results/{}/{}_{}_{}_{}_to_{}_results.json'.format(rate, results_folder, rate, experiments_end, steadyStart, steadyEnd), 'w') as f:
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

    # print("hostToTorLinkRate: ", hostToTorLinkRate, " Mbps")
    # print("torToAggLinkRate: ", torToAggLinkRate)
    # print("aggToCoreLinkRate: ", aggToCoreLinkRate, " Mbps")
    # print("hostToTorLinkDelay: ", hostToTorLinkDelay, " ms")
    # print("torToAggLinkDelay: ", torToAggLinkDelay, " ms")
    # print("aggToCoreLinkDelay: ", aggToCoreLinkDelay, " ms")
    # print("pctPacedBack: ", pctPacedBack, " %")
    # print("appDataRate: ", appDataRate, " Mbps")
    # print("duration: ", duration, " s")
    # print("steadyStart: ", steadyStart, " s")
    # print("steadyEnd: ", steadyEnd, " s")
    # print("sampleRate", sampleRate)
    # print("experiments: ", experiments)
    # print("serviceRateScales: ", serviceRateScales)
    # serviceRateScales = [0.85]
    experiments = 1
    # steadyStart = 4
    # steadyEnd = 9

    for rate in serviceRateScales:
        print("\nAnalyzing experiments for rate: ", rate)
        analyze_all_experiments(rate, steadyStart, steadyEnd, confidenceValue, experiments_start=0, experiments_end=experiments, ns3_path=__ns3_path)
        print("Rate {} {} done".format(rate, experiments))

__main__()