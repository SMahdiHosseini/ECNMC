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
__ns3_path = "/media/experiments/ns-allinone-3.41/ns-3.41"
errorRate = 0.05
difference = 1.30
# sample_rate = 0.30
sample_rates = [1.0]
confidenceValue = 1.96 # 95% confidence interval
propagationDelay = 50000
        
def check_MaxEpsilon_ineq_delay(endToEnd_statistics, samples_paths_aggregated_statistics):
    if abs(endToEnd_statistics['DelayMean'] - samples_paths_aggregated_statistics['DelayMean']) / samples_paths_aggregated_statistics['DelayMean'] <= samples_paths_aggregated_statistics['MaxEpsilonDelay']:
        return True
    else:
        return False

def check_MaxEpsilon_ineq_successProb(endToEnd_statistics, samples_paths_aggregated_statistics, number_of_segments):
    if (endToEnd_statistics - samples_paths_aggregated_statistics['successProbMean'] <= (number_of_segments * np.log(1 + samples_paths_aggregated_statistics['MaxEpsilonSuccessProb']))) and (endToEnd_statistics - samples_paths_aggregated_statistics['successProbMean'] >= (number_of_segments * np.log(1 - samples_paths_aggregated_statistics['MaxEpsilonSuccessProb']))):
        return True
    else:
        return False
      
def check_all_delayConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths):
    res = {}
    res['MaxEpsilonIneq'] = {}
    for flow in endToEnd_statistics.keys():
        res['MaxEpsilonIneq'][flow] = {}
        for path in paths:
            res['MaxEpsilonIneq'][flow][path] = check_MaxEpsilon_ineq_delay(endToEnd_statistics[flow][path], samples_paths_aggregated_statistics[flow][path])
    return res

def check_all_successProbConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths, number_of_segments):
    res = {}
    res['MaxEpsilonIneq'] = {}
    for flow in endToEnd_statistics.keys():
        res['MaxEpsilonIneq'][flow] = {}
        for path in paths:
            res['MaxEpsilonIneq'][flow][path] = {}
            res['MaxEpsilonIneq'][flow][path]['E2E_eventAvg'] = check_MaxEpsilon_ineq_successProb(np.log(endToEnd_statistics[flow][path]['successProbMean']['E2E_eventAvg']), samples_paths_aggregated_statistics[flow][path], number_of_segments)
            res['MaxEpsilonIneq'][flow][path]['sentTime_est'] = check_MaxEpsilon_ineq_successProb(np.log(endToEnd_statistics[flow][path]['successProbMean']['sentTime_est']), samples_paths_aggregated_statistics[flow][path], number_of_segments)
            res['MaxEpsilonIneq'][flow][path]['poisson_sentTime_est'] = {}
            for sample_rate in sample_rates:
                res['MaxEpsilonIneq'][flow][path]['poisson_sentTime_est'][sample_rate] = check_MaxEpsilon_ineq_successProb(np.log(endToEnd_statistics[flow][path]['successProbMean']['poisson_sentTime_est'][sample_rate]), samples_paths_aggregated_statistics[flow][path], number_of_segments)
    return res

def prepare_results(flows, queues, num_of_agg_switches):
    rounds_results = {}
    rounds_results['MaxEpsilonIneqDelay'] = {}
    rounds_results['MaxEpsilonIneqSuccessProb'] = {}
    rounds_results['MaxEpsilonIneqSuccessProb']['E2E_eventAvg'] = {}
    rounds_results['MaxEpsilonIneqSuccessProb']['sentTime_est'] = {}
    rounds_results['MaxEpsilonIneqSuccessProb']['poisson_sentTime_est'] = {}
    rounds_results['EndToEndDelayMean'] = {}
    rounds_results['EndToEndDelayStd'] = {}
    rounds_results['EndToEndSuccessProb'] = {}
    rounds_results['EndToEndSuccessProb']['E2E_eventAvg'] = {}
    rounds_results['EndToEndSuccessProb']['sentTime_est'] = {}
    rounds_results['EndToEndSuccessProb']['poisson_sentTime_est'] = {}
    for sample_rate in sample_rates:
        rounds_results['MaxEpsilonIneqSuccessProb']['poisson_sentTime_est'][sample_rate] = {}
        rounds_results['EndToEndSuccessProb']['poisson_sentTime_est'][sample_rate] = {}
    rounds_results['DropRate'] = []
    rounds_results['maxEpsilonDelay'] = {}
    rounds_results['maxEpsilonSuccessProb'] = {}
    rounds_results['errors'] = {}
    rounds_results['workLoad'] = {}
    rounds_results['AverageWorkLoad'] = []

    for q in queues:
        if q[0] == 'T' and q[2] == 'H' and (q[1] == '2' or q[1] == '3'):
            rounds_results[q+'Delaystd'] = []
        if q[0] == 'T' and q[2] == 'A' and (q[1] == '0' or q[1] == '1'):
            rounds_results[q+'Delaystd'] = []
        if q[0] == 'A' and q[2] == 'T' and (q[3] == '2' or q[3] == '3'):
            rounds_results[q+'Delaystd'] = []

    for flow in flows:
        rounds_results['MaxEpsilonIneqDelay'][flow] = {}
        rounds_results['MaxEpsilonIneqSuccessProb']['E2E_eventAvg'][flow] = {}
        rounds_results['MaxEpsilonIneqSuccessProb']['sentTime_est'][flow] = {}
        rounds_results['EndToEndDelayMean'][flow] = {}
        rounds_results['EndToEndDelayStd'][flow] = {}
        rounds_results['EndToEndSuccessProb']['E2E_eventAvg'][flow] = {}
        rounds_results['EndToEndSuccessProb']['sentTime_est'][flow] = {}
        for sample_rate in sample_rates:
            rounds_results['MaxEpsilonIneqSuccessProb']['poisson_sentTime_est'][sample_rate][flow] = {}
            rounds_results['EndToEndSuccessProb']['poisson_sentTime_est'][sample_rate][flow] = {}
        rounds_results['maxEpsilonDelay'][flow] = {}
        rounds_results['maxEpsilonSuccessProb'][flow] = {}
        rounds_results['errors'][flow] = {}
        rounds_results['workLoad'][flow] = {}

        for i in range(num_of_agg_switches):
            rounds_results['MaxEpsilonIneqDelay'][flow]['A' + str(i)] = 0
            rounds_results['MaxEpsilonIneqSuccessProb']['E2E_eventAvg'][flow]['A' + str(i)] = 0
            rounds_results['MaxEpsilonIneqSuccessProb']['sentTime_est'][flow]['A' + str(i)] = 0
            rounds_results['EndToEndDelayMean'][flow]['A' + str(i)] = []
            rounds_results['EndToEndDelayStd'][flow]['A' + str(i)] = []
            rounds_results['EndToEndSuccessProb']['E2E_eventAvg'][flow]['A' + str(i)] = []
            rounds_results['EndToEndSuccessProb']['sentTime_est'][flow]['A' + str(i)] = []
            for sample_rate in sample_rates:
                rounds_results['MaxEpsilonIneqSuccessProb']['poisson_sentTime_est'][sample_rate][flow]['A' + str(i)] = 0
                rounds_results['EndToEndSuccessProb']['poisson_sentTime_est'][sample_rate][flow]['A' + str(i)] = []
            rounds_results['maxEpsilonDelay'][flow]['A' + str(i)] = []
            rounds_results['maxEpsilonSuccessProb'][flow]['A' + str(i)] = []
            rounds_results['errors'][flow]['A' + str(i)] = []
            rounds_results['workLoad'][flow]['A' + str(i)] = []

    rounds_results['experiments'] = 0
    return rounds_results

def compatibility_check(rounds_results, samples_paths_aggregated_statistics, endToEnd_statistics, flows_name, paths, number_of_segments):
    # End to End and Persegment Compatibility Check
    delay_results = check_all_delayConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths)
    successProb_results = check_all_successProbConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths, number_of_segments)

    for flow in flows_name:
        for path in paths:
            if delay_results['MaxEpsilonIneq'][flow][path]:
                rounds_results['MaxEpsilonIneqDelay'][flow][path] += 1
            if successProb_results['MaxEpsilonIneq'][flow][path]['E2E_eventAvg']:
                rounds_results['MaxEpsilonIneqSuccessProb']['E2E_eventAvg'][flow][path] += 1
            if successProb_results['MaxEpsilonIneq'][flow][path]['sentTime_est']:
                rounds_results['MaxEpsilonIneqSuccessProb']['sentTime_est'][flow][path] += 1
            for sample_rate in sample_rates:
                if successProb_results['MaxEpsilonIneq'][flow][path]['poisson_sentTime_est'][sample_rate]:
                    rounds_results['MaxEpsilonIneqSuccessProb']['poisson_sentTime_est'][sample_rate][flow][path] += 1


def sample_endToEnd_packets(ns3_path, rate, segment, experiment, results_folder, _sample_rate, e2e_delays):
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
            lossProbs = []
            df = full_df[full_df['path'] == path]
            df = df.sort_values(by='sentTime').reset_index(drop=True)
            df['sentTime'] = df['sentTime'] - df['sentTime'].min()
            rtt = e2e_delays[df_name]['timeAverage'][path] + 4 * propagationDelay
            # generate sample times that are from a poisson distribution, the rate of samples is 4500 samples per second and the actual times are in nanoseconds
            sample_times = np.cumsum(np.random.exponential((_sample_rate * rtt), int(df['sentTime'].max() / (_sample_rate * rtt))))
            # print(_sample_rate, len(sample_times), sample_times.max(), rtt)
            # now for each sample time, pick the closest packet that was sent before or after the sample time. Then check if the packet was received or not. Then the lossProb is 0 or 1
            for sample_time in sample_times:
                if sample_time > df['sentTime'].max():
                    break
                # pick the closest packet that was sent after sample time
                closest_packet_after = df[df['sentTime'] > sample_time].iloc[0]
                # pick the closest packet that was sent before sample time
                closest_packet_before = df[df['sentTime'] < sample_time].iloc[-1]
                # now check if the difference between the closest packet and the sample time is less than the average delay of the path
                if abs(closest_packet_after['sentTime'] - sample_time) > (rtt / 2) or abs(closest_packet_before['sentTime'] - sample_time) > (rtt / 2):
                    continue
                if closest_packet_after['receivedTime'] != -1 and closest_packet_before['receivedTime'] != -1:
                    lossProbs.append(0)
                elif closest_packet_after['receivedTime'] == -1 and closest_packet_before['receivedTime'] == -1:
                    lossProbs.append(1)
                elif closest_packet_after['receivedTime'] != -1 and closest_packet_before['receivedTime'] == -1:
                    lossProbs.append(abs(closest_packet_before['sentTime'] - sample_time) / abs(closest_packet_after['sentTime'] - closest_packet_before['sentTime']))
                else:
                    lossProbs.append(abs(closest_packet_after['sentTime'] - sample_time) / abs(closest_packet_after['sentTime'] - closest_packet_before['sentTime']))

                # closest_packet = df.iloc[(df['sentTime'] - sample_time).abs().argsort()[:1]]
                # # now check if the difference between the closest packet and the sample time is less than the average delay of the path
                # if abs(closest_packet['sentTime'].values[0] - sample_time) > (rtt / 2):
                #     # print(_sample_rate, df_name, path, closest_packet['sentTime'].values[0], sample_time, e2e_delays[df_name]['timeAverage'][path])
                #     continue

                # if closest_packet['receivedTime'].values[0] != -1:
                #     lossProbs.append(0)
                # else:
                #     lossProbs.append(1)
            # now compute the time average of the lossProbs
            dfs[df_name]['timeAvgSuccessProb']['A' + str(path)] = 1 - np.mean(lossProbs)
    return dfs

def plot_packetsNumPerFlow_cdf(__ns3_path, rate, segment, experiment, results_folder):
    file_paths = glob.glob('{}/scratch/{}/{}/{}/*_{}.csv'.format(__ns3_path, results_folder, rate, experiment, segment))
    dfs = {}
    for file_path in file_paths:
        df_name = file_path.split('/')[-1].split('_')[0]
        full_df = pd.read_csv(file_path)
        dfs[df_name] = []
        full_df = full_df.rename(columns={'Path': 'path', 'SentTime': 'sentTime', 'ReceiveTime': 'receivedTime'})
        # plot the cdf of number of packets sent per flow. Each flow is indicated by 'SourcePort'
        for flow in full_df['SourcePort'].unique():
            df = full_df[full_df['SourcePort'] == flow]
            dfs[df_name].append(len(df))
        dfs[df_name].sort()
        y = np.arange(len(dfs[df_name])) / float(len(dfs[df_name]) - 1)
        plt.plot(dfs[df_name], y, label=df_name)
        # print 50th, 75th, 90th, 95th, 99th percentiles
        # print(df_name, "Number of flows: ", len(dfs[df_name]), "50th: ", np.percentile(dfs[df_name], 50), "75th: ", np.percentile(dfs[df_name], 75), "90th: ", np.percentile(dfs[df_name], 90), "95th: ", np.percentile(dfs[df_name], 95), "99th: ", np.percentile(dfs[df_name], 99), "average: ", np.mean(dfs[df_name]))
        # print number of flows with more than 30 packets
        # print(df_name, "Number of flows with only 30 packets: ", len([x for x in dfs[df_name] if x > 30]))
    plt.legend()
    plt.xlabel('Number of packets per flow')
    plt.ylabel('CDF')
    plt.savefig('../Results/results_forward/{}/CDF_packetsNumPerFlow.png'.format(rate))
    plt.close()

def calculate_delay_timeAvg(data):
    data = data.sort_values(by=['sentTime'])
    data = data[data['receivedTime'] != -1]
    data['Delay'] = data['receivedTime'] - data['sentTime'] - 4 * propagationDelay
    data['InterArrivalTime'] = data['receivedTime'].diff().fillna(0)
    timeAvg = (data['InterArrivalTime'] * data['Delay']).sum() / data['InterArrivalTime'].sum()
    return timeAvg

def calculate_successProb_timeAvg(data):
    data = data.sort_values(by=['sentTime'])
    data['lossProb'] = 0
    data.loc[data['receivedTime'] == -1, 'lossProb'] = 1
    data['InterArrivalTime'] = data['sentTime'].diff().fillna(0)
    lossProb = (data['InterArrivalTime'] * data['lossProb']).sum() / data['InterArrivalTime'].sum()
    return 1 - lossProb

def calculate_E2E_statistics(__ns3_path, rate, segment, experiment, results_folder, duration):
    file_paths = glob.glob('{}/scratch/{}/{}/{}/*_{}.csv'.format(__ns3_path, results_folder, rate, experiment, segment))
    dfs = {}
    for file_path in file_paths:
        df_name = file_path.split('/')[-1].split('_')[0]
        full_df = pd.read_csv(file_path)
        dfs[df_name] = []
        full_df = full_df.rename(columns={'Path': 'path', 'SentTime': 'sentTime', 'ReceiveTime': 'receivedTime'})
        # keep only flows with more than 30 packets
        for flow in full_df['SourcePort'].unique():
            df = full_df[full_df['SourcePort'] == flow]
            if len(df) < 30:
                continue
            res = {}
            res['flow'] = flow
            res['packetsNum'] = len(df)
            res['coverage'] = (df['sentTime'].max() - df['sentTime'].min()) / duration
            res['DelayMean'] = calculate_delay_timeAvg(df.copy())
            res['successProbMean'] = calculate_successProb_timeAvg(df.copy())
            res['path'] = df['path'].unique()[0]
            dfs[df_name].append(res)
        # plot the cdf of the coverage of the flows
        dfs[df_name] = pd.DataFrame(dfs[df_name])
        dfs[df_name] = dfs[df_name].sort_values(by=['coverage'])
        y = np.arange(len(dfs[df_name])) / float(len(dfs[df_name]) - 1)
        plt.plot(dfs[df_name]['coverage'], y, label=df_name)
        # # print 50th, 75th, 90th, 95th, 99th percentiles
        # print(df_name, "number of flows: ", len(dfs[df_name]), "50th: ", np.percentile(dfs[df_name]['coverage'], 50), "75th: ", np.percentile(dfs[df_name]['coverage'], 75), "90th: ", np.percentile(dfs[df_name]['coverage'], 90), "95th: ", np.percentile(dfs[df_name]['coverage'], 95), "99th: ", np.percentile(dfs[df_name]['coverage'], 99), "average: ", np.mean(dfs[df_name]['coverage']))
    plt.legend()
    plt.xlabel('Coverage of the flows')
    plt.ylabel('CDF')
    plt.savefig('../Results/results_forward/{}/CDF_coverage.png'.format(rate))
    plt.close()

    return dfs

def check_all_single_flows(endToEnd_stats, endToEnd_dfs, samples_paths_aggregated_statistics, rate, number_of_segments=3):
    coverage_checks_pairs = []
    for flow in endToEnd_dfs.keys():
        # endToEnd_stats[flow] is a dataframe with columns flow, packetsNum, coverage, delayMean, successProbMean, path, and I want to to iterate over each row
        for single_flow in endToEnd_stats[flow].to_dict(orient='records'):
            delay_check = check_MaxEpsilon_ineq_delay(single_flow, samples_paths_aggregated_statistics[flow]['A' + str(single_flow['path'])])
            successProb_check = check_MaxEpsilon_ineq_successProb(np.log(single_flow['successProbMean']), samples_paths_aggregated_statistics[flow]['A' + str(single_flow['path'])], number_of_segments)
            coverage_checks_pairs.append((single_flow['coverage'], delay_check, successProb_check))
    # plot the histogram of the checks per coverage. The x-axis is the coverage, the y-axis is the number of flows that passed the check over the total number of flows of each coverage range
    bins = 50
    coverage_checks_pairs = pd.DataFrame(coverage_checks_pairs, columns=['coverage', 'delay_check', 'successProb_check'])
    coverage_checks_pairs['delay_check'] = coverage_checks_pairs['delay_check'].astype(int)
    coverage_checks_pairs['successProb_check'] = coverage_checks_pairs['successProb_check'].astype(int)
    coverage_checks_pairs = coverage_checks_pairs.sort_values(by=['coverage'])
    coverage_checks_pairs['coverage'] = pd.cut(coverage_checks_pairs['coverage'], bins=bins, labels=False)
    coverage_checks_pairs['coverage'] = coverage_checks_pairs['coverage'] * (1 / bins)
    coverage_checks_pairs = coverage_checks_pairs.groupby('coverage').agg({'delay_check': ['sum', 'count'], 'successProb_check': ['sum', 'count']})
    coverage_checks_pairs['delay_check'] = coverage_checks_pairs['delay_check']['sum'] / coverage_checks_pairs['delay_check']['count']
    coverage_checks_pairs['successProb_check'] = coverage_checks_pairs['successProb_check']['sum'] / coverage_checks_pairs['successProb_check']['count']
    coverage_checks_pairs = coverage_checks_pairs.reset_index()
    plt.plot(coverage_checks_pairs['coverage'], coverage_checks_pairs['delay_check'], label='delay_check')
    plt.xlabel('Coverage')
    plt.ylabel('Percentage of flows that passed the delay flow/path consistency check')
    plt.savefig('../Results/results_forward/{}/delay_check.png'.format(rate))
    plt.close()
    plt.plot(coverage_checks_pairs['coverage'], coverage_checks_pairs['successProb_check'], label='successProb_check')
    plt.xlabel('Coverage')
    plt.ylabel('Percentage of flows that passed the delay flow/path consistency check')
    plt.savefig('../Results/results_forward/{}/successProb_check.png'.format(rate))
    plt.close()






def analyze_single_experiment(return_dict, rate, queues_names, confidenceValue, rounds_results, results_folder, duration, experiment=0, ns3_path=__ns3_path):
    num_of_agg_switches = 2
    paths = ['A' + str(i) for i in range(num_of_agg_switches)]
    endToEnd_dfs = read_online_computations(__ns3_path, rate, 'EndToEnd', str(experiment), results_folder)
    samples_dfs = read_online_computations(__ns3_path, rate, 'PoissonSampler', str(experiment), results_folder)
    successProbs = read_lossProb(__ns3_path, rate, 'EndToEnd_packets', str(experiment), results_folder)
    plot_packetsNumPerFlow_cdf(__ns3_path, rate, 'EndToEnd_packets', str(experiment), results_folder)
    endToEnd_stats = calculate_E2E_statistics(__ns3_path, rate, 'EndToEnd_packets', str(experiment), results_folder, duration)
    # successProbs_poisson = {}
    # for sample_rate in sample_rates:
    #     successProbs_poisson[sample_rate] = sample_endToEnd_packets(__ns3_path, rate, 'EndToEnd_packets', experiment, results_folder, sample_rate, endToEnd_dfs)
    # # switches_dfs = read_data(__ns3_path, 0.2, 1.2, rate, 'Switch', 'IsSent', 'ReceiveTime', str(experiment), True, results_folder)
    # # # # add delay columns which is ReceiveTime - SentTime and print the delay mean 9of the rows that path is 0 only for switch "T0"
    # # for switch in switches_dfs.keys():
    # #     switches_dfs[switch]['Delay'] = switches_dfs[switch]['SentTime'] - switches_dfs[switch]['ReceiveTime']
    # #     if switch == 'T0':
    # #         print(get_statistics(switches_dfs[switch][switches_dfs[switch]['path'] == 0], timeAvg=True)['timeAvg'])
    # #         print(switches_dfs[switch][switches_dfs[switch]['path'] == 0]['Delay'].mean())
    # #         # print(switches_dfs[switch][switches_dfs[switch]['SourceIp'] == '10.1.1.1']['Delay'].mean())
    # rounds_results['DropRate'].append(calculate_drop_rate_online(endToEnd_dfs, paths))

    # samples_paths_statistics
    samples_paths_aggregated_statistics = {}
    for flow in endToEnd_dfs.keys():
        samples_paths_aggregated_statistics[flow] = {}
        for path in paths:
            samples_paths_aggregated_statistics[flow][path] = {}
            samples_paths_aggregated_statistics[flow][path]['DelayMean'] = sum([samples_dfs['R' + flow[1] + 'H' + flow[3]]['DelayMean'],
                                                                                samples_dfs['T' + flow[1] + path]['DelayMean'], 
                                                                                samples_dfs[path + 'T' + flow[5]]['DelayMean'],
                                                                                samples_dfs['T' + flow[5] + 'H' + flow[7]]['DelayMean']])
            
            samples_paths_aggregated_statistics[flow][path]['MaxEpsilonDelay'] = max([calc_epsilon(confidenceValue, samples_dfs['R' + flow[1] + 'H' + flow[3]]),
                                                                                      calc_epsilon(confidenceValue, samples_dfs['T' + flow[1] + path]),
                                                                                      calc_epsilon(confidenceValue, samples_dfs[path + 'T' + flow[5]]),
                                                                                      calc_epsilon(confidenceValue, samples_dfs['T' + flow[5] + 'H' + flow[7]])])
            
            samples_paths_aggregated_statistics[flow][path]['successProbMean'] = sum([np.log(samples_dfs['T' + flow[1] + path]['successProbMean']),
                                                                                      np.log(samples_dfs[path + 'T' + flow[5]]['successProbMean']),
                                                                                      np.log(samples_dfs['T' + flow[5] + 'H' + flow[7]]['successProbMean'])])
            
            samples_paths_aggregated_statistics[flow][path]['MaxEpsilonSuccessProb'] = max([calc_epsilon_loss(confidenceValue, samples_dfs['T' + flow[1] + path]),
                                                                                     calc_epsilon_loss(confidenceValue, samples_dfs[path + 'T' + flow[5]]),
                                                                                     calc_epsilon_loss(confidenceValue, samples_dfs['T' + flow[5] + 'H' + flow[7]])])    
    # endToEnd_statistics
    endToEnd_statistics = {}
    AverageWorkLoad = 0
    for flow in endToEnd_dfs.keys():
        endToEnd_statistics[flow] = {}
        for path in paths:
            endToEnd_statistics[flow][path] = {}
            endToEnd_statistics[flow][path]['DelayMean'] = endToEnd_dfs[flow]['timeAverage'][int(path[1])]
            endToEnd_statistics[flow][path]['successProbMean'] = {}
            endToEnd_statistics[flow][path]['successProbMean']['E2E_eventAvg'] = endToEnd_dfs[flow]['successProbMean'][int(path[1])]   
            endToEnd_statistics[flow][path]['successProbMean']['sentTime_est'] = successProbs[flow]['timeAvgSuccessProb'][path]       
    #         endToEnd_statistics[flow][path]['successProbMean']['poisson_sentTime_est'] = {}
    #         for sample_rate in sample_rates:
    #             endToEnd_statistics[flow][path]['successProbMean']['poisson_sentTime_est'][sample_rate] = successProbs_poisson[sample_rate][flow]['timeAvgSuccessProb'][path]
            # if (flow == 'R0H0R2H0' and path == 'A0'):
            #     print(flow, path, endToEnd_statistics[flow][path]['successProbMean']['E2E_eventAvg'], successProbs[flow]['timeAvgSuccessProb'][path], samples_paths_aggregated_statistics[flow][path]['successProbMean'], samples_paths_aggregated_statistics[flow][path]['MaxEpsilonSuccessProb'])
            # rounds_results['EndToEndSuccessProb'][flow][path].append(endToEnd_dfs[flow]['successProbMean'][int(path[1])])
            # rounds_results['EndToEndSuccessProb'][flow][path].append(successProbs[flow]['timeAvgSuccessProb'][path])
            rounds_results['EndToEndSuccessProb']['E2E_eventAvg'][flow][path].append(endToEnd_dfs[flow]['successProbMean'][int(path[1])])
            rounds_results['EndToEndSuccessProb']['sentTime_est'][flow][path].append(successProbs[flow]['timeAvgSuccessProb'][path])
    #         for sample_rate in sample_rates:
    #             rounds_results['EndToEndSuccessProb']['poisson_sentTime_est'][sample_rate][flow][path].append(successProbs_poisson[sample_rate][flow]['timeAvgSuccessProb'][path])
            rounds_results['EndToEndDelayMean'][flow][path].append(endToEnd_dfs[flow]['timeAverage'][int(path[1])])
            rounds_results['EndToEndDelayStd'][flow][path].append(endToEnd_dfs[flow]['DelayStd'][int(path[1])])
            rounds_results['maxEpsilonDelay'][flow][path].append(samples_paths_aggregated_statistics[flow][path]['MaxEpsilonDelay'])
            rounds_results['maxEpsilonSuccessProb'][flow][path].append(samples_paths_aggregated_statistics[flow][path]['MaxEpsilonSuccessProb'])
            rounds_results['errors'][flow][path].append(abs((samples_paths_aggregated_statistics[flow][path]['DelayMean'] - endToEnd_statistics[flow][path]['DelayMean']) / samples_paths_aggregated_statistics[flow][path]['DelayMean']))
            AverageWorkLoad += (endToEnd_dfs[flow]['receivedPackets'][int(path[1])] * endToEnd_dfs[flow]['averagePacketSize'][int(path[1])] * 8)
    
        rounds_results['workLoad'][flow][path].append(((endToEnd_dfs[flow]['receivedPackets'][0] * endToEnd_dfs[flow]['averagePacketSize'][0]) + (endToEnd_dfs[flow]['receivedPackets'][1] * endToEnd_dfs[flow]['averagePacketSize'][1])) * 8 / 0.5)
    rounds_results['AverageWorkLoad'].append((AverageWorkLoad / 0.5) / 12)
    rounds_results['experiments'] += 1
    number_of_segments = 3

    # compatibility_check(rounds_results, samples_paths_aggregated_statistics, endToEnd_statistics, endToEnd_dfs.keys(), ['A' + str(i) for i in range(num_of_agg_switches)], number_of_segments)
    check_all_single_flows(endToEnd_stats, endToEnd_dfs, samples_paths_aggregated_statistics, rate, number_of_segments)

    for q in queues_names:
        if q[0] == 'T' and q[2] == 'H' and (q[1] == '2' or q[1] == '3'):
            rounds_results[q+'Delaystd'].append(samples_dfs[q]['DelayStd'])
        if q[0] == 'T' and q[2] == 'A' and (q[1] == '0' or q[1] == '1'):
            rounds_results[q+'Delaystd'].append(samples_dfs[q]['DelayStd'])
        if q[0] == 'A' and q[2] == 'T' and (q[3] == '2' or q[3] == '3'):
            rounds_results[q+'Delaystd'].append(samples_dfs[q]['DelayStd'])
    return_dict[experiment] = rounds_results

def merge_results(return_dict, merged_results, flows, queues):
    num_of_agg_switches = 2
    for exp in return_dict.keys():
        for q in queues:
            if q[0] == 'T' and q[2] == 'H' and (q[1] == '2' or q[1] == '3'):
                merged_results[q+'Delaystd'] += return_dict[exp][q+'Delaystd']
            if q[0] == 'T' and q[2] == 'A' and (q[1] == '0' or q[1] == '1'):
                merged_results[q+'Delaystd'] += return_dict[exp][q+'Delaystd']
            if q[0] == 'A' and q[2] == 'T' and (q[3] == '2' or q[3] == '3'):
                merged_results[q+'Delaystd'] += return_dict[exp][q+'Delaystd']

    for flow in flows:
        for i in range(num_of_agg_switches):
            for exp in return_dict.keys():
                merged_results['MaxEpsilonIneqDelay'][flow]['A' + str(i)] += return_dict[exp]['MaxEpsilonIneqDelay'][flow]['A' + str(i)]
                merged_results['MaxEpsilonIneqSuccessProb']['E2E_eventAvg'][flow]['A' + str(i)] += return_dict[exp]['MaxEpsilonIneqSuccessProb']['E2E_eventAvg'][flow]['A' + str(i)]
                merged_results['MaxEpsilonIneqSuccessProb']['sentTime_est'][flow]['A' + str(i)] += return_dict[exp]['MaxEpsilonIneqSuccessProb']['sentTime_est'][flow]['A' + str(i)]
                merged_results['EndToEndDelayMean'][flow]['A' + str(i)] += return_dict[exp]['EndToEndDelayMean'][flow]['A' + str(i)]
                merged_results['EndToEndDelayStd'][flow]['A' + str(i)] += return_dict[exp]['EndToEndDelayStd'][flow]['A' + str(i)]
                merged_results['EndToEndSuccessProb']['E2E_eventAvg'][flow]['A' + str(i)] += return_dict[exp]['EndToEndSuccessProb']['E2E_eventAvg'][flow]['A' + str(i)]
                merged_results['EndToEndSuccessProb']['sentTime_est'][flow]['A' + str(i)] += return_dict[exp]['EndToEndSuccessProb']['sentTime_est'][flow]['A' + str(i)]
                for sample_rate in sample_rates:
                    merged_results['MaxEpsilonIneqSuccessProb']['poisson_sentTime_est'][sample_rate][flow]['A' + str(i)] += return_dict[exp]['MaxEpsilonIneqSuccessProb']['poisson_sentTime_est'][sample_rate][flow]['A' + str(i)]
                    merged_results['EndToEndSuccessProb']['poisson_sentTime_est'][sample_rate][flow]['A' + str(i)] += return_dict[exp]['EndToEndSuccessProb']['poisson_sentTime_est'][sample_rate][flow]['A' + str(i)]
                merged_results['maxEpsilonDelay'][flow]['A' + str(i)] += return_dict[exp]['maxEpsilonDelay'][flow]['A' + str(i)]
                merged_results['maxEpsilonSuccessProb'][flow]['A' + str(i)] += return_dict[exp]['maxEpsilonSuccessProb'][flow]['A' + str(i)]
                merged_results['errors'][flow]['A' + str(i)] += return_dict[exp]['errors'][flow]['A' + str(i)]
                merged_results['workLoad'][flow]['A' + str(i)] += return_dict[exp]['workLoad'][flow]['A' + str(i)]
    for exp in return_dict.keys():
        merged_results['experiments'] += return_dict[exp]['experiments']
        merged_results['DropRate'] += return_dict[exp]['DropRate']
        merged_results['AverageWorkLoad'] += return_dict[exp]['AverageWorkLoad']
    
def analyze_all_experiments(rate, steadyStart, steadyEnd, confidenceValue, dir, experiments_end=3, ns3_path=__ns3_path):
    results_folder = 'Results_' + dir
    num_of_agg_switches = 2
    flows_name = read_data_flowIndicator(ns3_path, rate, results_folder)
    flows_name.sort()

    queues_names = read_queues_indicators(ns3_path, rate, results_folder)
    queues_names.sort()

    rounds_results = prepare_results(flows_name, queues_names, num_of_agg_switches)
    merged_results = prepare_results(flows_name, queues_names, num_of_agg_switches)
    batch_size = 30
    for i in range(int(experiments_end / batch_size) + 1):
        ths = []
        return_dict = multiprocessing.Manager().dict()
        for experiment in range(batch_size * i, min(experiments_end, batch_size * (i + 1))):
            if len(os.listdir('{}/scratch/{}/{}/{}'.format(__ns3_path, results_folder, rate, experiment))) == 0:
                print(experiment)
                continue
            print("Analyzing experiment: ", experiment)
            ths.append(multiprocessing.Process(target=analyze_single_experiment, args=(return_dict, rate, queues_names, confidenceValue, rounds_results, results_folder, (steadyEnd - steadyStart) * (1e9) , experiment, ns3_path)))
        
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        merge_results(return_dict, merged_results, flows_name, queues_names)
        print("{} joind".format(i))
    merged_results['AverageWorkLoad'] = sum(merged_results['AverageWorkLoad']) / merged_results['experiments']
    with open('../Results/results_{}/{}/delay_{}_{}_{}_to_{}.json'.format(dir, rate, results_folder, experiments_end, steadyStart, steadyEnd), 'w') as f:
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
    if "forward" in args.dir:
        serviceRateScales = [float(x) for x in config.get('Settings', 'serviceRateScales').split(',')]
    else:
        serviceRateScales = [float(x) for x in config.get('Settings', 'errorRateScale').split(',')]
    serviceRateScales = [0.79]
    # serviceRateScales = [1.0, 1.01, 1.03, 1.05]
    # serviceRateScales = [0.91, 0.93, 0.95, 0.97, 0.99, 1.01, 1.03, 1.05]
    # serviceRateScales = [float(x) for x in config.get('Settings', 'serviceRateScales').split(',')]
    experiments = 1

    for rate in serviceRateScales:
        print("\nAnalyzing experiments for rate: ", rate)
        analyze_all_experiments(rate, steadyStart, steadyEnd, confidenceValue, args.dir, experiments_end=experiments, ns3_path=__ns3_path)
        print("Rate {} {} done".format(rate, experiments))

__main__()