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
sample_rates = [0.5]
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

def check_MaxEpsilon_ineq_nonMarkingProb(endToEnd_statistics, samples_paths_aggregated_statistics, number_of_segments):
    if (endToEnd_statistics - samples_paths_aggregated_statistics['nonMarkingProbMean'] <= (number_of_segments * np.log(1 + samples_paths_aggregated_statistics['MaxEpsilonNonMarkingProb']))) and (endToEnd_statistics - samples_paths_aggregated_statistics['nonMarkingProbMean'] >= (number_of_segments * np.log(1 - samples_paths_aggregated_statistics['MaxEpsilonNonMarkingProb']))):
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
            # print(flow, path, endToEnd_statistics[flow][path]['DelayMean'], samples_paths_aggregated_statistics[flow][path]['DelayMean'], abs(endToEnd_statistics[flow][path]['DelayMean'] - samples_paths_aggregated_statistics[flow][path]['DelayMean']) / samples_paths_aggregated_statistics[flow][path]['DelayMean'], samples_paths_aggregated_statistics[flow][path]['MaxEpsilonDelay'])
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
            res['MaxEpsilonIneq'][flow][path]['enqueueTime_est'] = check_MaxEpsilon_ineq_successProb(np.log(endToEnd_statistics[flow][path]['successProbMean']['enqueueTime_est']), samples_paths_aggregated_statistics[flow][path], number_of_segments)
            res['MaxEpsilonIneq'][flow][path]['poisson_sentTime_est'] = {}
            for sample_rate in sample_rates:
                res['MaxEpsilonIneq'][flow][path]['poisson_sentTime_est'][sample_rate] = check_MaxEpsilon_ineq_successProb(np.log(endToEnd_statistics[flow][path]['successProbMean']['poisson_sentTime_est'][sample_rate]), samples_paths_aggregated_statistics[flow][path], number_of_segments)
            # print(flow, path, endToEnd_statistics[flow][path]['successProbMean']['sentTime_est'], np.exp(samples_paths_aggregated_statistics[flow][path]['successProbMean']))
    return res

def check_all_nonMarkingProbConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths, number_of_segments):
    res = {}
    res['MaxEpsilonIneq'] = {}
    for flow in endToEnd_statistics.keys():
        res['MaxEpsilonIneq'][flow] = {}
        for path in paths:
            res['MaxEpsilonIneq'][flow][path] = {}
            res['MaxEpsilonIneq'][flow][path]['E2E_eventAvg'] = check_MaxEpsilon_ineq_nonMarkingProb(np.log(endToEnd_statistics[flow][path]['nonMarkingProbMean']['E2E_eventAvg']), samples_paths_aggregated_statistics[flow][path], number_of_segments)
            res['MaxEpsilonIneq'][flow][path]['sentTime_est_events'] = check_MaxEpsilon_ineq_nonMarkingProb(np.log(endToEnd_statistics[flow][path]['nonMarkingProbMean']['sentTime_est_events']), samples_paths_aggregated_statistics[flow][path], number_of_segments)
            res['MaxEpsilonIneq'][flow][path]['sentTime_est_probs'] = check_MaxEpsilon_ineq_nonMarkingProb(np.log(endToEnd_statistics[flow][path]['nonMarkingProbMean']['sentTime_est_probs']), samples_paths_aggregated_statistics[flow][path], number_of_segments)
            # print(flow, path, endToEnd_statistics[flow][path]['successProbMean']['sentTime_est'], np.exp(samples_paths_aggregated_statistics[flow][path]['successProbMean']))
    return res

def prepare_results(flows, queues, num_of_paths):
    rounds_results = {}
    rounds_results['MaxEpsilonIneqDelay'] = {}
    rounds_results['MaxEpsilonIneqSuccessProb'] = {}
    rounds_results['MaxEpsilonIneqSuccessProb']['E2E_eventAvg'] = {}
    rounds_results['MaxEpsilonIneqSuccessProb']['sentTime_est'] = {}
    rounds_results['MaxEpsilonIneqSuccessProb']['enqueueTime_est'] = {}
    rounds_results['MaxEpsilonIneqSuccessProb']['poisson_sentTime_est'] = {}
    rounds_results['MaxEpsilonIneqNonMarkingProb'] = {}
    rounds_results['MaxEpsilonIneqNonMarkingProb']['E2E_eventAvg'] = {}
    rounds_results['MaxEpsilonIneqNonMarkingProb']['sentTime_est_events'] = {}
    rounds_results['MaxEpsilonIneqNonMarkingProb']['sentTime_est_probs'] = {}
    rounds_results['EndToEndDelayMean'] = {}
    rounds_results['EndToEndDelayStd'] = {}
    rounds_results['EndToEndSuccessProb'] = {}
    rounds_results['EndToEndSuccessProb']['E2E_eventAvg'] = {}
    rounds_results['EndToEndSuccessProb']['sentTime_est'] = {}
    rounds_results['EndToEndSuccessProb']['enqueueTime_est'] = {}
    rounds_results['EndToEndSuccessProb']['poisson_sentTime_est'] = {}
    for sample_rate in sample_rates:
        rounds_results['MaxEpsilonIneqSuccessProb']['poisson_sentTime_est'][sample_rate] = {}
        rounds_results['EndToEndSuccessProb']['poisson_sentTime_est'][sample_rate] = {}
    rounds_results['EndToEndNonMarkingProb'] = {}
    rounds_results['EndToEndNonMarkingProb']['E2E_eventAvg'] = {}
    rounds_results['EndToEndNonMarkingProb']['sentTime_est_events'] = {}
    rounds_results['EndToEndNonMarkingProb']['sentTime_est_probs'] = {}
    rounds_results['DropRate'] = []
    rounds_results['maxEpsilonDelay'] = {}
    rounds_results['maxEpsilonSuccessProb'] = {}
    rounds_results['maxEpsilonNonMarkingProb'] = {}
    rounds_results['errors'] = {}
    rounds_results['workLoad'] = {}
    rounds_results['AverageWorkLoad'] = []

    for q in queues:
        if q[0] == 'S' and q[1] == 'D':
            rounds_results[q+'Delaystd'] = []
            rounds_results[q+'DelayMean'] = []
            rounds_results[q+'successProbMean'] = []
            rounds_results[q+'nonMarkingProbMean'] = []

    for flow in flows:
        rounds_results['MaxEpsilonIneqDelay'][flow] = {}
        rounds_results['MaxEpsilonIneqSuccessProb']['E2E_eventAvg'][flow] = {}
        rounds_results['MaxEpsilonIneqSuccessProb']['sentTime_est'][flow] = {}
        rounds_results['MaxEpsilonIneqSuccessProb']['enqueueTime_est'][flow] = {}
        rounds_results['MaxEpsilonIneqNonMarkingProb']['E2E_eventAvg'][flow] = {}
        rounds_results['MaxEpsilonIneqNonMarkingProb']['sentTime_est_events'][flow] = {}
        rounds_results['MaxEpsilonIneqNonMarkingProb']['sentTime_est_probs'][flow] = {}
        rounds_results['EndToEndNonMarkingProb']['E2E_eventAvg'][flow] = {}
        rounds_results['EndToEndNonMarkingProb']['sentTime_est_events'][flow] = {}
        rounds_results['EndToEndNonMarkingProb']['sentTime_est_probs'][flow] = {}
        rounds_results['EndToEndDelayMean'][flow] = {}
        rounds_results['EndToEndDelayStd'][flow] = {}
        rounds_results['EndToEndSuccessProb']['E2E_eventAvg'][flow] = {}
        rounds_results['EndToEndSuccessProb']['sentTime_est'][flow] = {}
        rounds_results['EndToEndSuccessProb']['enqueueTime_est'][flow] = {}
        for sample_rate in sample_rates:
            rounds_results['MaxEpsilonIneqSuccessProb']['poisson_sentTime_est'][sample_rate][flow] = {}
            rounds_results['EndToEndSuccessProb']['poisson_sentTime_est'][sample_rate][flow] = {}
        rounds_results['maxEpsilonDelay'][flow] = {}
        rounds_results['maxEpsilonSuccessProb'][flow] = {}
        rounds_results['maxEpsilonNonMarkingProb'][flow] = {}
        rounds_results['errors'][flow] = {}
        rounds_results['workLoad'][flow] = {}

        for i in range(num_of_paths):
            rounds_results['MaxEpsilonIneqDelay'][flow]['A' + str(i)] = 0
            rounds_results['MaxEpsilonIneqSuccessProb']['E2E_eventAvg'][flow]['A' + str(i)] = 0
            rounds_results['MaxEpsilonIneqSuccessProb']['sentTime_est'][flow]['A' + str(i)] = 0
            rounds_results['MaxEpsilonIneqSuccessProb']['enqueueTime_est'][flow]['A' + str(i)] = 0
            rounds_results['MaxEpsilonIneqNonMarkingProb']['E2E_eventAvg'][flow]['A' + str(i)] = 0
            rounds_results['MaxEpsilonIneqNonMarkingProb']['sentTime_est_events'][flow]['A' + str(i)] = 0
            rounds_results['MaxEpsilonIneqNonMarkingProb']['sentTime_est_probs'][flow]['A' + str(i)] = 0
            rounds_results['EndToEndNonMarkingProb']['E2E_eventAvg'][flow]['A' + str(i)] = []
            rounds_results['EndToEndNonMarkingProb']['sentTime_est_events'][flow]['A' + str(i)] = []
            rounds_results['EndToEndNonMarkingProb']['sentTime_est_probs'][flow]['A' + str(i)] = []
            rounds_results['EndToEndDelayMean'][flow]['A' + str(i)] = []
            rounds_results['EndToEndDelayStd'][flow]['A' + str(i)] = []
            rounds_results['EndToEndSuccessProb']['E2E_eventAvg'][flow]['A' + str(i)] = []
            rounds_results['EndToEndSuccessProb']['sentTime_est'][flow]['A' + str(i)] = []
            rounds_results['EndToEndSuccessProb']['enqueueTime_est'][flow]['A' + str(i)] = []
            for sample_rate in sample_rates:
                rounds_results['MaxEpsilonIneqSuccessProb']['poisson_sentTime_est'][sample_rate][flow]['A' + str(i)] = 0
                rounds_results['EndToEndSuccessProb']['poisson_sentTime_est'][sample_rate][flow]['A' + str(i)] = []
            rounds_results['maxEpsilonDelay'][flow]['A' + str(i)] = []
            rounds_results['maxEpsilonSuccessProb'][flow]['A' + str(i)] = []
            rounds_results['maxEpsilonNonMarkingProb'][flow]['A' + str(i)] = []
            rounds_results['errors'][flow]['A' + str(i)] = []
            rounds_results['workLoad'][flow]['A' + str(i)] = []

    rounds_results['experiments'] = 0
    return rounds_results

def compatibility_check(rounds_results, samples_paths_aggregated_statistics, endToEnd_statistics, flows_name, paths, number_of_segments):
    # End to End and Persegment Compatibility Check
    delay_results = check_all_delayConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths)
    successProb_results = check_all_successProbConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths, number_of_segments)
    nonMarkingProb_results = check_all_nonMarkingProbConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths, number_of_segments)

    for flow in flows_name:
        for path in paths:
            if delay_results['MaxEpsilonIneq'][flow][path]:
                rounds_results['MaxEpsilonIneqDelay'][flow][path] += 1
            if successProb_results['MaxEpsilonIneq'][flow][path]['E2E_eventAvg']:
                rounds_results['MaxEpsilonIneqSuccessProb']['E2E_eventAvg'][flow][path] += 1
            if successProb_results['MaxEpsilonIneq'][flow][path]['sentTime_est']:
                rounds_results['MaxEpsilonIneqSuccessProb']['sentTime_est'][flow][path] += 1
            if successProb_results['MaxEpsilonIneq'][flow][path]['enqueueTime_est']:
                rounds_results['MaxEpsilonIneqSuccessProb']['enqueueTime_est'][flow][path] += 1
            for sample_rate in sample_rates:
                if successProb_results['MaxEpsilonIneq'][flow][path]['poisson_sentTime_est'][sample_rate]:
                    rounds_results['MaxEpsilonIneqSuccessProb']['poisson_sentTime_est'][sample_rate][flow][path] += 1
            if nonMarkingProb_results['MaxEpsilonIneq'][flow][path]['E2E_eventAvg']:
                rounds_results['MaxEpsilonIneqNonMarkingProb']['E2E_eventAvg'][flow][path] += 1
            if nonMarkingProb_results['MaxEpsilonIneq'][flow][path]['sentTime_est_events']:
                rounds_results['MaxEpsilonIneqNonMarkingProb']['sentTime_est_events'][flow][path] += 1
            if nonMarkingProb_results['MaxEpsilonIneq'][flow][path]['sentTime_est_probs']:
                rounds_results['MaxEpsilonIneqNonMarkingProb']['sentTime_est_probs'][flow][path] += 1


def sample_endToEnd_packets(ns3_path, rate, segment, experiment, results_folder, _sample_rate, e2e_delays):
    file_paths = glob.glob('{}/scratch/{}/{}/{}/*_{}.csv'.format(__ns3_path, results_folder, rate, experiment, segment))
    dfs = {}
    for file_path in file_paths:
        df_name = file_path.split('/')[-1].split('_')[0]
        full_df = pd.read_csv(file_path)
        # remove all columns other than path, sentTime, receivedTime
        # first rename the columns Path to path, SentTime to sentTime, ReceiveTime to receivedTime
        full_df = full_df.rename(columns={'Path': 'path', 'SentTime': 'sentTime', 'ReceiveTime': 'receivedTime'})
        full_df = full_df[['path', 'sentTime', 'receivedTime']]
        dfs[df_name] = {}
        dfs[df_name]['timeAvgSuccessProb'] = {}
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

            
def analyze_single_experiment(return_dict, rate, queues_names, confidenceValue, rounds_results, results_folder, config, experiment=0, ns3_path=__ns3_path):
    srcHostToSwitchLinkRate = convert_to_float(config.get('SingleQueue', 'srcHostToSwitchLinkRate')) * 1e-3
    bottleneckLinkRate = convert_to_float(config.get('SingleQueue', 'bottleneckLinkRate')) * rate * 1e-3
    linkDelay = convert_to_float(config.get('Settings', 'hostToTorLinkDelay')) * 1e3
    steadyStart = convert_to_float(config.get('Settings', 'steadyStart')) * 1e9
    steadyEnd = convert_to_float(config.get('Settings', 'steadyEnd')) * 1e9
    swtichDstREDQueueDiscMaxSize = convert_to_float(config.get('Settings', 'swtichDstREDQueueDiscMaxSize'))
    num_of_paths = 1
    paths = ['A' + str(i) for i in range(num_of_paths)]
    # endToEnd_dfs = read_online_computations(__ns3_path, rate, 'EndToEnd', str(experiment), results_folder)
    # samples_dfs = read_online_computations(__ns3_path, rate, 'PoissonSampler', str(experiment), results_folder)

    endToEndStats = calculate_offline_computations(__ns3_path, rate, 'EndToEnd_packets', str(experiment), results_folder, steadyStart, steadyEnd, "SentTime", True, "IsReceived", [srcHostToSwitchLinkRate, bottleneckLinkRate], [linkDelay, linkDelay], swtichDstREDQueueDiscMaxSize)
    # print(endToEndStats)
    calculate_offline_computations(__ns3_path, rate, 'EndToEnd_markings', str(experiment), results_folder, endToEndStats['A0D0']['first'][0], endToEndStats['A0D0']['last'][0], "Time", linksRates=[srcHostToSwitchLinkRate, bottleneckLinkRate], linkDelays=[linkDelay, linkDelay], stats=endToEndStats)
    # print(endToEndStats)
    samplesSats = calculate_offline_computations(__ns3_path, rate, 'PoissonSampler_events', str(experiment), results_folder, endToEndStats['A0D0']['first'][0], endToEndStats['A0D0']['last'][0], "Time")
    # print(samplesSats)

    successProbs = read_lossProb(__ns3_path, rate, 'EndToEnd_packets', str(experiment), results_folder)
    successProbs_poisson = {}
    for sample_rate in sample_rates:
        successProbs_poisson[sample_rate] = sample_endToEnd_packets(__ns3_path, rate, 'EndToEnd_packets', experiment, results_folder, sample_rate, endToEndStats)

    rounds_results['DropRate'].append(calculate_avgDrop_rate_offline(endToEndStats, paths))
    # samples_paths_statistics
    samples_paths_aggregated_statistics = {}
    for flow in endToEndStats.keys():
        samples_paths_aggregated_statistics[flow] = {}
        for path in paths:
            samples_paths_aggregated_statistics[flow][path] = {}
            samples_paths_aggregated_statistics[flow][path]['DelayMean'] = samplesSats['SD0']['DelayMean']
            
            samples_paths_aggregated_statistics[flow][path]['MaxEpsilonDelay'] = calc_epsilon(confidenceValue, samplesSats['SD0'])
            # print([(key, calc_epsilon(confidenceValue, samples_df)) for key, samples_df in samples_dfs.items()])
            
            samples_paths_aggregated_statistics[flow][path]['successProbMean'] = np.log(samplesSats['SD0']['successProbMean'])
            samples_paths_aggregated_statistics[flow][path]['MaxEpsilonSuccessProb'] = calc_epsilon_loss(confidenceValue, samplesSats['SD0'])

            samples_paths_aggregated_statistics[flow][path]['nonMarkingProbMean'] = np.log(samplesSats['SD0']['nonMarkingProbMean'])
            samples_paths_aggregated_statistics[flow][path]['MaxEpsilonNonMarkingProb'] = calc_epsilon_marking(confidenceValue, samplesSats['SD0'])

    # endToEnd_statistics
    endToEnd_statistics = {}
    AverageWorkLoad = 0
    for flow in endToEndStats.keys():
        endToEnd_statistics[flow] = {}
        for path in paths:
            endToEnd_statistics[flow][path] = {}
            endToEnd_statistics[flow][path]['DelayMean'] = endToEndStats[flow]['timeAverage'][int(path[1])]
            endToEnd_statistics[flow][path]['successProbMean'] = {}
            endToEnd_statistics[flow][path]['successProbMean']['E2E_eventAvg'] = endToEndStats[flow]['successProbMean'][int(path[1])]   
            endToEnd_statistics[flow][path]['successProbMean']['sentTime_est'] = successProbs[flow]['timeAvgSuccessProb'][path]
            endToEnd_statistics[flow][path]['successProbMean']['enqueueTime_est'] = endToEndStats[flow]['enqueueTimeAvgSuccessProb'][int(path[1])]       
            endToEnd_statistics[flow][path]['successProbMean']['poisson_sentTime_est'] = {}
            for sample_rate in sample_rates:
                endToEnd_statistics[flow][path]['successProbMean']['poisson_sentTime_est'][sample_rate] = successProbs_poisson[sample_rate][flow]['timeAvgSuccessProb'][path]
            # print(flow, path, (samples_paths_aggregated_statistics[flow][path]['DelayMean'] * rate * 0.6 / 8) / 100, "%")
            endToEnd_statistics[flow][path]['nonMarkingProbMean'] = {}
            endToEnd_statistics[flow][path]['nonMarkingProbMean']['E2E_eventAvg'] = endToEndStats[flow]['nonMarkingProbMean'][int(path[1])]
            endToEnd_statistics[flow][path]['nonMarkingProbMean']['sentTime_est_events'] = endToEndStats[flow]['enqueueTimeAvgNonMarkingProb'][int(path[1])]
            endToEnd_statistics[flow][path]['nonMarkingProbMean']['sentTime_est_probs'] = endToEndStats[flow]['enqueueTimeAvgNonMarkingFractionProb'][int(path[1])]
            rounds_results['EndToEndNonMarkingProb']['E2E_eventAvg'][flow][path].append(endToEndStats[flow]['nonMarkingProbMean'][int(path[1])])
            rounds_results['EndToEndNonMarkingProb']['sentTime_est_events'][flow][path].append(endToEndStats[flow]['enqueueTimeAvgNonMarkingProb'][int(path[1])])
            rounds_results['EndToEndNonMarkingProb']['sentTime_est_probs'][flow][path].append(endToEndStats[flow]['enqueueTimeAvgNonMarkingFractionProb'][int(path[1])])
            rounds_results['EndToEndSuccessProb']['E2E_eventAvg'][flow][path].append(endToEndStats[flow]['successProbMean'][int(path[1])])
            rounds_results['EndToEndSuccessProb']['sentTime_est'][flow][path].append(successProbs[flow]['timeAvgSuccessProb'][path])
            rounds_results['EndToEndSuccessProb']['enqueueTime_est'][flow][path].append(endToEndStats[flow]['enqueueTimeAvgSuccessProb'][int(path[1])])
            for sample_rate in sample_rates:
                rounds_results['EndToEndSuccessProb']['poisson_sentTime_est'][sample_rate][flow][path].append(successProbs_poisson[sample_rate][flow]['timeAvgSuccessProb'][path])
            rounds_results['EndToEndDelayMean'][flow][path].append(endToEndStats[flow]['timeAverage'][int(path[1])])
            rounds_results['EndToEndDelayStd'][flow][path].append(endToEndStats[flow]['DelayStd'][int(path[1])])
            rounds_results['maxEpsilonDelay'][flow][path].append(samples_paths_aggregated_statistics[flow][path]['MaxEpsilonDelay'])
            rounds_results['maxEpsilonSuccessProb'][flow][path].append(samples_paths_aggregated_statistics[flow][path]['MaxEpsilonSuccessProb'])
            rounds_results['maxEpsilonNonMarkingProb'][flow][path].append(samples_paths_aggregated_statistics[flow][path]['MaxEpsilonNonMarkingProb'])
            rounds_results['errors'][flow][path].append(abs((samples_paths_aggregated_statistics[flow][path]['DelayMean'] - endToEnd_statistics[flow][path]['DelayMean']) / samples_paths_aggregated_statistics[flow][path]['DelayMean']))
            AverageWorkLoad += (endToEndStats[flow]['worklaod'][int(path[1])])
    
        rounds_results['workLoad'][flow][path].append(endToEndStats[flow]['worklaod'][int(path[1])])
    rounds_results['AverageWorkLoad'].append(AverageWorkLoad / len(endToEndStats.keys()))
    rounds_results['experiments'] += 1
    number_of_segments = 1
    compatibility_check(rounds_results, samples_paths_aggregated_statistics, endToEnd_statistics, endToEndStats.keys(), ['A' + str(i) for i in range(num_of_paths)], number_of_segments)
              
    for q in queues_names:
        if q[0] == 'S' and q[1] == 'D':
            rounds_results[q+'Delaystd'].append(samplesSats[q]['DelayStd'])
            rounds_results[q+'DelayMean'].append(samplesSats[q]['DelayMean'])
            rounds_results[q+'successProbMean'].append(samplesSats[q]['successProbMean'])
            rounds_results[q+'nonMarkingProbMean'].append(samplesSats[q]['nonMarkingProbMean'])
    return_dict[experiment] = rounds_results

def merge_results(return_dict, merged_results, flows, queues, num_of_paths):
    for exp in return_dict.keys():
        for q in queues:
            if q[0] == 'S' and q[1] == 'D':
                merged_results[q+'Delaystd'] += return_dict[exp][q+'Delaystd']
                merged_results[q+'DelayMean'] += return_dict[exp][q+'DelayMean']
                merged_results[q+'successProbMean'] += return_dict[exp][q+'successProbMean']
                merged_results[q+'nonMarkingProbMean'] += return_dict[exp][q+'nonMarkingProbMean']

    for flow in flows:
        for i in range(num_of_paths):
            for exp in return_dict.keys():
                merged_results['MaxEpsilonIneqDelay'][flow]['A' + str(i)] += return_dict[exp]['MaxEpsilonIneqDelay'][flow]['A' + str(i)]
                merged_results['MaxEpsilonIneqSuccessProb']['E2E_eventAvg'][flow]['A' + str(i)] += return_dict[exp]['MaxEpsilonIneqSuccessProb']['E2E_eventAvg'][flow]['A' + str(i)]
                merged_results['MaxEpsilonIneqSuccessProb']['sentTime_est'][flow]['A' + str(i)] += return_dict[exp]['MaxEpsilonIneqSuccessProb']['sentTime_est'][flow]['A' + str(i)]
                merged_results['MaxEpsilonIneqSuccessProb']['enqueueTime_est'][flow]['A' + str(i)] += return_dict[exp]['MaxEpsilonIneqSuccessProb']['enqueueTime_est'][flow]['A' + str(i)]
                merged_results['MaxEpsilonIneqNonMarkingProb']['E2E_eventAvg'][flow]['A' + str(i)] += return_dict[exp]['MaxEpsilonIneqNonMarkingProb']['E2E_eventAvg'][flow]['A' + str(i)]
                merged_results['MaxEpsilonIneqNonMarkingProb']['sentTime_est_events'][flow]['A' + str(i)] += return_dict[exp]['MaxEpsilonIneqNonMarkingProb']['sentTime_est_events'][flow]['A' + str(i)]
                merged_results['MaxEpsilonIneqNonMarkingProb']['sentTime_est_probs'][flow]['A' + str(i)] += return_dict[exp]['MaxEpsilonIneqNonMarkingProb']['sentTime_est_probs'][flow]['A' + str(i)]
                merged_results['EndToEndDelayMean'][flow]['A' + str(i)] += return_dict[exp]['EndToEndDelayMean'][flow]['A' + str(i)]
                merged_results['EndToEndDelayStd'][flow]['A' + str(i)] += return_dict[exp]['EndToEndDelayStd'][flow]['A' + str(i)]
                merged_results['EndToEndNonMarkingProb']['E2E_eventAvg'][flow]['A' + str(i)] += return_dict[exp]['EndToEndNonMarkingProb']['E2E_eventAvg'][flow]['A' + str(i)]
                merged_results['EndToEndNonMarkingProb']['sentTime_est_events'][flow]['A' + str(i)] += return_dict[exp]['EndToEndNonMarkingProb']['sentTime_est_events'][flow]['A' + str(i)]
                merged_results['EndToEndNonMarkingProb']['sentTime_est_probs'][flow]['A' + str(i)] += return_dict[exp]['EndToEndNonMarkingProb']['sentTime_est_probs'][flow]['A' + str(i)]
                merged_results['EndToEndSuccessProb']['E2E_eventAvg'][flow]['A' + str(i)] += return_dict[exp]['EndToEndSuccessProb']['E2E_eventAvg'][flow]['A' + str(i)]
                merged_results['EndToEndSuccessProb']['sentTime_est'][flow]['A' + str(i)] += return_dict[exp]['EndToEndSuccessProb']['sentTime_est'][flow]['A' + str(i)]
                merged_results['EndToEndSuccessProb']['enqueueTime_est'][flow]['A' + str(i)] += return_dict[exp]['EndToEndSuccessProb']['enqueueTime_est'][flow]['A' + str(i)]
                for sample_rate in sample_rates:
                    merged_results['MaxEpsilonIneqSuccessProb']['poisson_sentTime_est'][sample_rate][flow]['A' + str(i)] += return_dict[exp]['MaxEpsilonIneqSuccessProb']['poisson_sentTime_est'][sample_rate][flow]['A' + str(i)]
                    merged_results['EndToEndSuccessProb']['poisson_sentTime_est'][sample_rate][flow]['A' + str(i)] += return_dict[exp]['EndToEndSuccessProb']['poisson_sentTime_est'][sample_rate][flow]['A' + str(i)]
                merged_results['maxEpsilonDelay'][flow]['A' + str(i)] += return_dict[exp]['maxEpsilonDelay'][flow]['A' + str(i)]
                merged_results['maxEpsilonSuccessProb'][flow]['A' + str(i)] += return_dict[exp]['maxEpsilonSuccessProb'][flow]['A' + str(i)]
                merged_results['maxEpsilonNonMarkingProb'][flow]['A' + str(i)] += return_dict[exp]['maxEpsilonNonMarkingProb'][flow]['A' + str(i)]
                merged_results['errors'][flow]['A' + str(i)] += return_dict[exp]['errors'][flow]['A' + str(i)]
                merged_results['workLoad'][flow]['A' + str(i)] += return_dict[exp]['workLoad'][flow]['A' + str(i)]
    for exp in return_dict.keys():
        merged_results['experiments'] += return_dict[exp]['experiments']
        merged_results['DropRate'] += return_dict[exp]['DropRate']
        merged_results['AverageWorkLoad'] += return_dict[exp]['AverageWorkLoad']
    
def analyze_all_experiments(rate, steadyStart, steadyEnd, confidenceValue, dir, config, experiments_end=3, ns3_path=__ns3_path):
    results_folder = 'Results_' + dir
    num_of_paths = 1
    flows_name = read_data_flowIndicator(ns3_path, rate, results_folder)
    flows_name.sort()

    queues_names = read_queues_indicators(ns3_path, rate, results_folder)
    queues_names.sort()

    rounds_results = prepare_results(flows_name, queues_names, num_of_paths)
    merged_results = prepare_results(flows_name, queues_names, num_of_paths)
    batch_size = 30
    for i in range(int(experiments_end / batch_size) + 1):
        ths = []
        return_dict = multiprocessing.Manager().dict()
        for experiment in range(batch_size * i, min(experiments_end, batch_size * (i + 1))):
            if len(os.listdir('{}/scratch/{}/{}/{}'.format(__ns3_path, results_folder, rate, experiment))) == 0:
                print(experiment)
                continue
            print("Analyzing experiment: ", experiment)
            ths.append(multiprocessing.Process(target=analyze_single_experiment, args=(return_dict, rate, queues_names, confidenceValue, rounds_results, results_folder, config, experiment, ns3_path)))
        
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        merge_results(return_dict, merged_results, flows_name, queues_names, num_of_paths)
        print("{} joind".format(i))
    merged_results['AverageWorkLoad'] = sum(merged_results['AverageWorkLoad']) / merged_results['experiments']
    print("average drop rate =", np.average(merged_results['DropRate']) * 100, "%", "average queue capacity usage = ", np.average(merged_results['SD0DelayMean']) * 0.6 * rate / 800, "%", "Delay consistency check " , merged_results['MaxEpsilonIneqDelay']['A0D0']["A0"] / merged_results['experiments'] * 100)
    with open('../Results/results_{}/{}/{}_{}_{}_{}_to_{}.json'.format(dir, rate, dir, results_folder, experiments_end, steadyStart, steadyEnd), 'w') as f:
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
    elif "param" in args.dir:
        serviceRateScales = [float(x) for x in config.get('Settings', 'sampleRateScales').split(',')]
    else:
        serviceRateScales = [float(x) for x in config.get('Settings', 'errorRateScale').split(',')]
    # serviceRateScales = [0.99]
    # serviceRateScales = [1.0, 1.01, 1.03, 1.05]
    # serviceRateScales = [0.91, 0.93, 0.95, 0.97, 0.99, 1.01, 1.03, 1.05]
    # serviceRateScales = [float(x) for x in config.get('Settings', 'serviceRateScales').split(',')]
    # experiments = 1

    for rate in serviceRateScales:
        print("\nAnalyzing experiments for rate: ", rate)
        analyze_all_experiments(rate, steadyStart, steadyEnd, confidenceValue, args.dir, config, experiments_end=experiments, ns3_path=__ns3_path)
        print("Rate {} {} done".format(rate, experiments))

__main__()