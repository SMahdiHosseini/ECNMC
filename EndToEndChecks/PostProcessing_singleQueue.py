from Utils import *
from BiasCalculation import *
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

timeAvg_methods = ['rightCont_timeAvg', 'leftCont_timeAvg', 'linearInterp_timeAvg', 'poisson_eventAvg']
delay_timeAvg_vars = ['event']
successProb_timeAvg_vars = ['event', 'probability']
# successProb_timeAvg_vars = ['event']
nonMarkingProb_timeAvg_vars = ['event']

def check_MaxEpsilon_ineq_delay(endToEnd_statistics, samples_paths_aggregated_statistics, last=""):
    if abs(endToEnd_statistics - samples_paths_aggregated_statistics[last + 'DelayMean']) / samples_paths_aggregated_statistics[last + 'DelayMean'] <= samples_paths_aggregated_statistics['MaxEpsilon' + last + 'Delay']:
        return True
    else:
        return False

def check_MaxEpsilon_ineq_successProb(endToEnd_statistics, samples_paths_aggregated_statistics, number_of_segments, last=""):
    if (endToEnd_statistics - samples_paths_aggregated_statistics[last + 'SuccessProbMean'] <= (number_of_segments * np.log(1 + samples_paths_aggregated_statistics['MaxEpsilon' + last + 'SuccessProb']))) and (endToEnd_statistics - samples_paths_aggregated_statistics[last + 'SuccessProbMean'] >= (number_of_segments * np.log(1 - samples_paths_aggregated_statistics['MaxEpsilon' + last + 'SuccessProb']))):
        return True
    else:
        return False

def check_MaxEpsilon_ineq_nonMarkingProb(endToEnd_statistics, samples_paths_aggregated_statistics, number_of_segments):
    if (endToEnd_statistics - samples_paths_aggregated_statistics['NonMarkingProbMean'] <= (number_of_segments * np.log(1 + samples_paths_aggregated_statistics['MaxEpsilonNonMarkingProb']))) and (endToEnd_statistics - samples_paths_aggregated_statistics['NonMarkingProbMean'] >= (number_of_segments * np.log(1 - samples_paths_aggregated_statistics['MaxEpsilonNonMarkingProb']))):
        return True
    else:
        return False

def check_MaxEpsilon_ineq_lastNonMarkingProb(endToEnd_statistics, samples_paths_aggregated_statistics, number_of_segments):
    if (endToEnd_statistics - samples_paths_aggregated_statistics['LastNonMarkingProbMean'] <= (number_of_segments * np.log(1 + samples_paths_aggregated_statistics['MaxEpsilonLastNonMarkingProb']))) and (endToEnd_statistics - samples_paths_aggregated_statistics['LastNonMarkingProbMean'] >= (number_of_segments * np.log(1 - samples_paths_aggregated_statistics['MaxEpsilonLastNonMarkingProb']))):
        return True
    else:
        return False
    
def check_all_delayConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths, last=""):
    res = {}
    res['MaxEpsilonIneq'] = {}
    for flow in endToEnd_statistics.keys():
        res['MaxEpsilonIneq'][flow] = {}
        for path in paths:
            res['MaxEpsilonIneq'][flow][path] = {}
            for var_method in endToEnd_statistics[flow]['delay'].keys():
                if var_method != 'event_poisson_eventAvg':
                    res['MaxEpsilonIneq'][flow][path][var_method] = check_MaxEpsilon_ineq_delay(endToEnd_statistics[flow]['delay'][var_method][path], samples_paths_aggregated_statistics[flow][path], last)
                else:
                    e = (samples_paths_aggregated_statistics[flow][path][last + 'DelayMean'] * samples_paths_aggregated_statistics[flow][path]['MaxEpsilon' + last + 'Delay']) + endToEnd_statistics[flow]['delay'][var_method][path][1] * confidenceValue
                    res['MaxEpsilonIneq'][flow][path][var_method] = (abs(endToEnd_statistics[flow]['delay'][var_method][path][0] - samples_paths_aggregated_statistics[flow][path][last + 'DelayMean']) <= e)
    return res

def check_all_successProbConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths, number_of_segments, last=""):
    res = {}
    res['MaxEpsilonIneq'] = {}
    for flow in endToEnd_statistics.keys():
        res['MaxEpsilonIneq'][flow] = {}
        for path in paths:
            res['MaxEpsilonIneq'][flow][path] = {}
            for var_method in endToEnd_statistics[flow]['successProb'].keys():
                if var_method != 'event_poisson_eventAvg' and var_method != 'probability_poisson_eventAvg':
                    res['MaxEpsilonIneq'][flow][path][var_method] = check_MaxEpsilon_ineq_successProb(np.log(endToEnd_statistics[flow]['successProb'][var_method][path]), samples_paths_aggregated_statistics[flow][path], number_of_segments, last)
                else:
                    epsp = (endToEnd_statistics[flow]['successProb'][var_method][path][1] * confidenceValue) / endToEnd_statistics[flow]['successProb'][var_method][path][0]
                    eps = samples_paths_aggregated_statistics[flow][path]['MaxEpsilon' + last + 'SuccessProb']
                    e = samples_paths_aggregated_statistics[flow][path][last + 'SuccessProbMean'] - np.log(endToEnd_statistics[flow]['successProb'][var_method][path][0])
                    res['MaxEpsilonIneq'][flow][path][var_method] = ((e <= (np.log(1+epsp) - np.log(1-eps))) and (e >= (np.log(1-epsp) - np.log(1+eps))))
    return res

def check_all_nonMarkingProbConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths, number_of_segments):
    res = {}
    res['MaxEpsilonIneq'] = {}
    for flow in endToEnd_statistics.keys():
        res['MaxEpsilonIneq'][flow] = {}
        for path in paths:
            res['MaxEpsilonIneq'][flow][path] = {}
            for var_method in endToEnd_statistics[flow]['nonMarkingProb'].keys():
                if var_method != 'event_poisson_eventAvg':
                    res['MaxEpsilonIneq'][flow][path][var_method] = check_MaxEpsilon_ineq_nonMarkingProb(np.log(endToEnd_statistics[flow]['nonMarkingProb'][var_method][path]), samples_paths_aggregated_statistics[flow][path], number_of_segments)
                else:
                    epsp = (endToEnd_statistics[flow]['nonMarkingProb'][var_method][path][1] * confidenceValue) / endToEnd_statistics[flow]['nonMarkingProb'][var_method][path][0]
                    eps = samples_paths_aggregated_statistics[flow][path]['MaxEpsilonNonMarkingProb']
                    e = samples_paths_aggregated_statistics[flow][path]['NonMarkingProbMean'] - np.log(endToEnd_statistics[flow]['nonMarkingProb'][var_method][path][0])
                    res['MaxEpsilonIneq'][flow][path][var_method] = ((e <= (np.log(1+epsp) - np.log(1-eps))) and (e >= (np.log(1-epsp) - np.log(1+eps))))
    return res

def check_all_lastNonMarkingProbConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths, number_of_segments):
    res = {}
    res['MaxEpsilonIneq'] = {}
    for flow in endToEnd_statistics.keys():
        res['MaxEpsilonIneq'][flow] = {}
        for path in paths:
            res['MaxEpsilonIneq'][flow][path] = {}
            for var_method in endToEnd_statistics[flow]['nonMarkingProb'].keys():
                if var_method != 'event_poisson_eventAvg':
                    res['MaxEpsilonIneq'][flow][path][var_method] = check_MaxEpsilon_ineq_lastNonMarkingProb(np.log(endToEnd_statistics[flow]['nonMarkingProb'][var_method][path]), samples_paths_aggregated_statistics[flow][path], number_of_segments)
                else:
                    epsp = (endToEnd_statistics[flow]['nonMarkingProb'][var_method][path][1] * confidenceValue) / endToEnd_statistics[flow]['nonMarkingProb'][var_method][path][0]
                    eps = samples_paths_aggregated_statistics[flow][path]['MaxEpsilonLastNonMarkingProb']
                    e = samples_paths_aggregated_statistics[flow][path]['LastNonMarkingProbMean'] - np.log(endToEnd_statistics[flow]['nonMarkingProb'][var_method][path][0])
                    res['MaxEpsilonIneq'][flow][path][var_method] = ((e <= (np.log(1+epsp) - np.log(1-eps))) and (e >= (np.log(1-epsp) - np.log(1+eps))))
    return res

def prepare_results(flows, queues, num_of_paths):
    rounds_results = {}
    rounds_results['MaxEpsilonIneqDelay'] = {}
    rounds_results['MaxEpsilonIneqLastDelay'] = {}
    rounds_results['MaxEpsilonIneqSuccessProb'] = {}
    rounds_results['MaxEpsilonIneqLastSuccessProb'] = {}
    rounds_results['MaxEpsilonIneqNonMarkingProb'] = {}
    rounds_results['MaxEpsilonIneqLastNonMarkingProb'] = {}
    rounds_results['EndToEndDelayMean'] = {}
    rounds_results['EndToEndSuccessProb'] = {}
    rounds_results['EndToEndNonMarkingProb'] = {}
    rounds_results['DropRate'] = []
    rounds_results['maxEpsilonDelay'] = {}
    rounds_results['maxEpsilonLastDelay'] = {}
    rounds_results['maxEpsilonSuccessProb'] = {}
    rounds_results['maxEpsilonLastSuccessProb'] = {}
    rounds_results['maxEpsilonNonMarkingProb'] = {}
    rounds_results['maxEpsilonLastNonMarkingProb'] = {}
    rounds_results['workLoad'] = {}
    rounds_results['AverageWorkLoad'] = []
    rounds_results['experiments'] = 0

    for var in delay_timeAvg_vars:
        for method in timeAvg_methods:
            rounds_results['MaxEpsilonIneqDelay'][var + '_' + method] = {}
            rounds_results['MaxEpsilonIneqLastDelay'][var + '_' + method] = {}
            rounds_results['EndToEndDelayMean'][var + '_' + method] = {}

    for var in successProb_timeAvg_vars:
        for method in timeAvg_methods:
            rounds_results['MaxEpsilonIneqSuccessProb'][var + '_' + method] = {}
            rounds_results['MaxEpsilonIneqLastSuccessProb'][var + '_' + method] = {}
            rounds_results['EndToEndSuccessProb'][var + '_' + method] = {}

    for var in nonMarkingProb_timeAvg_vars:
        for method in timeAvg_methods:
            rounds_results['MaxEpsilonIneqNonMarkingProb'][var + '_' + method] = {}
            rounds_results['EndToEndNonMarkingProb'][var + '_' + method] = {}

    for var in nonMarkingProb_timeAvg_vars:
        for method in timeAvg_methods:
            rounds_results['MaxEpsilonIneqLastNonMarkingProb'][var + '_' + method] = {}

    for q in queues:
        # if q[0] == 'S' and q[1] == 'D':
        rounds_results[q+'Delaystd'] = []
        rounds_results[q+'DelayMean'] = []
        rounds_results[q+'LastDelaystd'] = []
        rounds_results[q+'LastDelayMean'] = []
        rounds_results[q+'SuccessProbStd'] = []
        rounds_results[q+'SuccessProbMean'] = []
        rounds_results[q+'LastSuccessProbStd'] = []
        rounds_results[q+'LastSuccessProbMean'] = []
        rounds_results[q+'NonMarkingProbStd'] = []
        rounds_results[q+'NonMarkingProbMean'] = []
        rounds_results[q+'LastNonMarkingProbStd'] = []
        rounds_results[q+'LastNonMarkingProbMean'] = []

    for flow in flows:
        for var_method in rounds_results['MaxEpsilonIneqDelay'].keys():
            rounds_results['MaxEpsilonIneqDelay'][var_method][flow] = {}
            rounds_results['MaxEpsilonIneqLastDelay'][var_method][flow] = {}
            rounds_results['EndToEndDelayMean'][var_method][flow] = {}

        for var_method in rounds_results['MaxEpsilonIneqSuccessProb'].keys():
            rounds_results['MaxEpsilonIneqSuccessProb'][var_method][flow] = {}
            rounds_results['MaxEpsilonIneqLastSuccessProb'][var_method][flow] = {}
            rounds_results['EndToEndSuccessProb'][var_method][flow] = {}

        for var_method in rounds_results['MaxEpsilonIneqNonMarkingProb'].keys():
            rounds_results['MaxEpsilonIneqNonMarkingProb'][var_method][flow] = {}
            rounds_results['MaxEpsilonIneqLastNonMarkingProb'][var_method][flow] = {}
            rounds_results['EndToEndNonMarkingProb'][var_method][flow] = {}

        rounds_results['workLoad'][flow] = {}
        rounds_results['maxEpsilonDelay'][flow] = {}
        rounds_results['maxEpsilonLastDelay'][flow] = {}
        rounds_results['maxEpsilonSuccessProb'][flow] = {}
        rounds_results['maxEpsilonLastSuccessProb'][flow] = {}
        rounds_results['maxEpsilonNonMarkingProb'][flow] = {}
        rounds_results['maxEpsilonLastNonMarkingProb'][flow] = {}

        for i in range(num_of_paths):
            for var_method in rounds_results['MaxEpsilonIneqDelay'].keys():
                rounds_results['MaxEpsilonIneqDelay'][var_method][flow][i] = 0
                rounds_results['MaxEpsilonIneqLastDelay'][var_method][flow][i] = 0
                rounds_results['EndToEndDelayMean'][var_method][flow][i] = []

            for var_method in rounds_results['MaxEpsilonIneqSuccessProb'].keys():
                rounds_results['MaxEpsilonIneqSuccessProb'][var_method][flow][i] = 0
                rounds_results['MaxEpsilonIneqLastSuccessProb'][var_method][flow][i] = 0
                rounds_results['EndToEndSuccessProb'][var_method][flow][i] = []
            
            for var_method in rounds_results['MaxEpsilonIneqNonMarkingProb'].keys():
                rounds_results['MaxEpsilonIneqNonMarkingProb'][var_method][flow][i] = 0
                rounds_results['MaxEpsilonIneqLastNonMarkingProb'][var_method][flow][i] = 0
                rounds_results['EndToEndNonMarkingProb'][var_method][flow][i] = []
            
            rounds_results['workLoad'][flow][i] = []
            rounds_results['maxEpsilonDelay'][flow][i] = []
            rounds_results['maxEpsilonLastDelay'][flow][i] = []
            rounds_results['maxEpsilonSuccessProb'][flow][i] = []
            rounds_results['maxEpsilonLastSuccessProb'][flow][i] = []
            rounds_results['maxEpsilonNonMarkingProb'][flow][i] = []
            rounds_results['maxEpsilonLastNonMarkingProb'][flow][i] = []

    return rounds_results

def compatibility_check(rounds_results, samples_paths_aggregated_statistics, endToEnd_statistics, flows_name, paths, number_of_segments):
    # End to End and Persegment Compatibility Check
    delay_results = check_all_delayConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths)
    lastDelay_results = check_all_delayConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths, 'Last')
    successProb_results = check_all_successProbConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths, number_of_segments)
    lastSuccessProb_results = check_all_successProbConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths, number_of_segments, 'Last')
    nonMarkingProb_results = check_all_nonMarkingProbConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths, number_of_segments)
    lastNonMarkingProb_results = check_all_lastNonMarkingProbConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths, number_of_segments)

    for flow in flows_name:
        for path in paths:
            for var_method in rounds_results['MaxEpsilonIneqDelay'].keys():
                if delay_results['MaxEpsilonIneq'][flow][path][var_method]:
                    rounds_results['MaxEpsilonIneqDelay'][var_method][flow][path] += 1
                if lastDelay_results['MaxEpsilonIneq'][flow][path][var_method]:
                    rounds_results['MaxEpsilonIneqLastDelay'][var_method][flow][path] += 1

            for var_method in rounds_results['MaxEpsilonIneqSuccessProb'].keys():
                if successProb_results['MaxEpsilonIneq'][flow][path][var_method]:
                    rounds_results['MaxEpsilonIneqSuccessProb'][var_method][flow][path] += 1
                if lastSuccessProb_results['MaxEpsilonIneq'][flow][path][var_method]:
                    rounds_results['MaxEpsilonIneqLastSuccessProb'][var_method][flow][path] += 1

            for var_method in rounds_results['MaxEpsilonIneqNonMarkingProb'].keys():
                if nonMarkingProb_results['MaxEpsilonIneq'][flow][path][var_method]:
                    rounds_results['MaxEpsilonIneqNonMarkingProb'][var_method][flow][path] += 1
                if lastNonMarkingProb_results['MaxEpsilonIneq'][flow][path][var_method]:
                    rounds_results['MaxEpsilonIneqLastNonMarkingProb'][var_method][flow][path] += 1


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
    paths = range(num_of_paths)
    # endToEnd_dfs = read_online_computations(__ns3_path, rate, 'EndToEnd', str(experiment), results_folder)
    # samples_dfs = read_online_computations(__ns3_path, rate, 'PoissonSampler', str(experiment), results_folder)
    # biasCalculator = BiasCalculator(results_folder, rate, [experiment], steadyStart, steadyEnd, rounds_results, bottleneckLinkRate)
    # biasCalculator.calculateBias(['MarkingProb', 'DropProb', 'QueuingDelay', 'LastMarkingProb'])
    endToEndStats = calculate_offline_computations(__ns3_path, rate, 'EndToEnd_packets', str(experiment), results_folder, steadyStart, steadyEnd, "SentTime", True, "IsReceived", [srcHostToSwitchLinkRate, bottleneckLinkRate], [linkDelay, linkDelay], swtichDstREDQueueDiscMaxSize)
    # endToEndStats = calculate_offline_computations_on_switch(__ns3_path, results_folder, rate, experiment, 'PoissonSampler_queueSize', steadyStart, steadyEnd, paths, bottleneckLinkRate)
    # print(endToEndStats)
    # calculate_offline_computations(__ns3_path, rate, 'EndToEnd_markings', str(experiment), results_folder, endToEndStats['A0D0']['first'][0], endToEndStats['A0D0']['last'][0], "Time", linksRates=[srcHostToSwitchLinkRate, bottleneckLinkRate], linkDelays=[linkDelay, linkDelay], stats=endToEndStats)
    # print(endToEndStats)
    samplesSats = calculate_offline_computations(__ns3_path, rate, 'PoissonSampler_events', str(experiment), results_folder, endToEndStats['A0D0']['first'][0], endToEndStats['A0D0']['last'][0], "Time", linksRates=[bottleneckLinkRate], swtichDstREDQueueDiscMaxSize=swtichDstREDQueueDiscMaxSize)
    # print(samplesSats)

    rounds_results['DropRate'].append(calculate_avgDrop_rate_offline(endToEndStats, paths))
    # samples_paths_statistics
    samples_paths_aggregated_statistics = {}
    for flow in endToEndStats.keys():
        samples_paths_aggregated_statistics[flow] = {}
        for path in paths:
            samples_paths_aggregated_statistics[flow][path] = {}
            samples_paths_aggregated_statistics[flow][path]['DelayMean'] = samplesSats['SD0']['DelayMean']
            samples_paths_aggregated_statistics[flow][path]['MaxEpsilonDelay'] = calc_epsilon(confidenceValue, samplesSats['SD0'])
            # samples_paths_aggregated_statistics[flow][path]['DelayMean'] = samplesSats['SD0']['DelayMean'] + samplesSats['H']['DelayMean']
            # samples_paths_aggregated_statistics[flow][path]['MaxEpsilonDelay'] = max(calc_epsilon(confidenceValue, samplesSats['SD0']), calc_epsilon(confidenceValue, samplesSats['H']))
            # samples_paths_aggregated_statistics[flow][path]['MaxEpsilonDelay'] = calc_epsilon_with_bias(confidenceValue, samplesSats['SD0'], biasCalculator.GTBias['QueuingDelay'][1.0][0])
            
            samples_paths_aggregated_statistics[flow][path]['LastDelayMean'] = samplesSats['SD0']['LastDelayMean']
            samples_paths_aggregated_statistics[flow][path]['MaxEpsilonLastDelay'] = calc_epsilon(confidenceValue, samplesSats['SD0'], "Last")
            # print([(key, calc_epsilon(confidenceValue, samples_df)) for key, samples_df in samples_dfs.items()])
            
            samples_paths_aggregated_statistics[flow][path]['SuccessProbMean'] = np.log(samplesSats['SD0']['SuccessProbMean'])
            samples_paths_aggregated_statistics[flow][path]['MaxEpsilonSuccessProb'] = calc_epsilon_loss(confidenceValue, samplesSats['SD0'])
            # samples_paths_aggregated_statistics[flow][path]['MaxEpsilonSuccessProb'] = calc_epsilon_loss_with_bias(confidenceValue, samplesSats['SD0'], biasCalculator.GTBias['DropProb'][1.0][0])
            
            samples_paths_aggregated_statistics[flow][path]['LastSuccessProbMean'] = np.log(samplesSats['SD0']['LastSuccessProbMean'])
            samples_paths_aggregated_statistics[flow][path]['MaxEpsilonLastSuccessProb'] = calc_epsilon_loss(confidenceValue, samplesSats['SD0'], "Last")

            samples_paths_aggregated_statistics[flow][path]['NonMarkingProbMean'] = np.log(samplesSats['SD0']['NonMarkingProbMean'])
            samples_paths_aggregated_statistics[flow][path]['MaxEpsilonNonMarkingProb'] = calc_epsilon_marking(confidenceValue, samplesSats['SD0'])
            # samples_paths_aggregated_statistics[flow][path]['MaxEpsilonNonMarkingProb'] = calc_epsilon_marking_with_bias(confidenceValue, samplesSats['SD0'], biasCalculator.GTBias['MarkingProb'][1.0][0])

            samples_paths_aggregated_statistics[flow][path]['LastNonMarkingProbMean'] = np.log(samplesSats['SD0']['LastNonMarkingProbMean'])
            samples_paths_aggregated_statistics[flow][path]['MaxEpsilonLastNonMarkingProb'] = calc_epsilon_marking(confidenceValue, samplesSats['SD0'], last="Last")
            # samples_paths_aggregated_statistics[flow][path]['MaxEpsilonLastNonMarkingProb'] = calc_epsilon_last_marking_with_bias(confidenceValue, samplesSats['SD0'], biasCalculator.GTBias['LastMarkingProb'][1.0][0])

    # endToEnd_statistics
    AverageWorkLoad = 0
    for flow in endToEndStats.keys():
        for path in paths:
            for var_method in rounds_results['EndToEndDelayMean'].keys():
                rounds_results['EndToEndDelayMean'][var_method][flow][path].append(endToEndStats[flow]['delay'][var_method][path])
            for var_method in rounds_results['EndToEndSuccessProb'].keys():
                rounds_results['EndToEndSuccessProb'][var_method][flow][path].append(endToEndStats[flow]['successProb'][var_method][path])
            for var_method in rounds_results['EndToEndNonMarkingProb'].keys():
                rounds_results['EndToEndNonMarkingProb'][var_method][flow][path].append(endToEndStats[flow]['nonMarkingProb'][var_method][path])

            rounds_results['maxEpsilonDelay'][flow][path].append(samples_paths_aggregated_statistics[flow][path]['MaxEpsilonDelay'])
            rounds_results['maxEpsilonLastDelay'][flow][path].append(samples_paths_aggregated_statistics[flow][path]['MaxEpsilonLastDelay'])
            rounds_results['maxEpsilonSuccessProb'][flow][path].append(samples_paths_aggregated_statistics[flow][path]['MaxEpsilonSuccessProb'])
            rounds_results['maxEpsilonLastSuccessProb'][flow][path].append(samples_paths_aggregated_statistics[flow][path]['MaxEpsilonLastSuccessProb'])
            rounds_results['maxEpsilonNonMarkingProb'][flow][path].append(samples_paths_aggregated_statistics[flow][path]['MaxEpsilonNonMarkingProb'])
            rounds_results['maxEpsilonLastNonMarkingProb'][flow][path].append(samples_paths_aggregated_statistics[flow][path]['MaxEpsilonLastNonMarkingProb'])

            AverageWorkLoad += (endToEndStats[flow]['workload'][path])
    
        rounds_results['workLoad'][flow][path].append(endToEndStats[flow]['workload'][path])
    rounds_results['AverageWorkLoad'].append(AverageWorkLoad / len(endToEndStats.keys()))
    rounds_results['experiments'] += 1
    number_of_segments = 1
    compatibility_check(rounds_results, samples_paths_aggregated_statistics, endToEndStats, endToEndStats.keys(), range(num_of_paths), number_of_segments)
              
    for q in queues_names:
        # if q[0] == 'S' and q[1] == 'D':
        rounds_results[q+'Delaystd'].append(samplesSats[q]['DelayStd'])
        rounds_results[q+'DelayMean'].append(samplesSats[q]['DelayMean'])
        rounds_results[q+'LastDelaystd'].append(samplesSats[q]['LastDelayStd'])
        rounds_results[q+'LastDelayMean'].append(samplesSats[q]['LastDelayMean'])
        rounds_results[q+'SuccessProbStd'].append(samplesSats[q]['SuccessProbStd'])
        rounds_results[q+'SuccessProbMean'].append(samplesSats[q]['SuccessProbMean'])
        rounds_results[q+'LastSuccessProbStd'].append(samplesSats[q]['LastSuccessProbStd'])
        rounds_results[q+'LastSuccessProbMean'].append(samplesSats[q]['LastSuccessProbMean'])
        rounds_results[q+'NonMarkingProbStd'].append(samplesSats[q]['NonMarkingProbStd'])
        rounds_results[q+'NonMarkingProbMean'].append(samplesSats[q]['NonMarkingProbMean'])
        rounds_results[q+'LastNonMarkingProbStd'].append(samplesSats[q]['LastNonMarkingProbStd'])
        rounds_results[q+'LastNonMarkingProbMean'].append(samplesSats[q]['LastNonMarkingProbMean'])
    return_dict[experiment] = rounds_results

def merge_results(return_dict, merged_results, flows, queues, num_of_paths):
    for exp in return_dict.keys():
        for q in queues:
            # if q[0] == 'S' and q[1] == 'D':
            merged_results[q+'Delaystd'] += return_dict[exp][q+'Delaystd']
            merged_results[q+'DelayMean'] += return_dict[exp][q+'DelayMean']
            merged_results[q+'LastDelaystd'] += return_dict[exp][q+'LastDelaystd']
            merged_results[q+'LastDelayMean'] += return_dict[exp][q+'LastDelayMean']
            merged_results[q+'SuccessProbStd'] += return_dict[exp][q+'SuccessProbStd']
            merged_results[q+'SuccessProbMean'] += return_dict[exp][q+'SuccessProbMean']
            merged_results[q+'LastSuccessProbStd'] += return_dict[exp][q+'LastSuccessProbStd']
            merged_results[q+'LastSuccessProbMean'] += return_dict[exp][q+'LastSuccessProbMean']
            merged_results[q+'NonMarkingProbStd'] += return_dict[exp][q+'NonMarkingProbStd']
            merged_results[q+'NonMarkingProbMean'] += return_dict[exp][q+'NonMarkingProbMean']
            merged_results[q+'LastNonMarkingProbStd'] += return_dict[exp][q+'LastNonMarkingProbStd']
            merged_results[q+'LastNonMarkingProbMean'] += return_dict[exp][q+'LastNonMarkingProbMean']

    for flow in flows:
        for i in range(num_of_paths):
            for exp in return_dict.keys():
                for var_method in merged_results['MaxEpsilonIneqDelay'].keys():
                    merged_results['MaxEpsilonIneqDelay'][var_method][flow][i] += return_dict[exp]['MaxEpsilonIneqDelay'][var_method][flow][i]
                    merged_results['MaxEpsilonIneqLastDelay'][var_method][flow][i] += return_dict[exp]['MaxEpsilonIneqLastDelay'][var_method][flow][i]
                    merged_results['EndToEndDelayMean'][var_method][flow][i] += return_dict[exp]['EndToEndDelayMean'][var_method][flow][i]

                for var_method in merged_results['MaxEpsilonIneqSuccessProb'].keys():
                    merged_results['MaxEpsilonIneqSuccessProb'][var_method][flow][i] += return_dict[exp]['MaxEpsilonIneqSuccessProb'][var_method][flow][i]
                    merged_results['MaxEpsilonIneqLastSuccessProb'][var_method][flow][i] += return_dict[exp]['MaxEpsilonIneqLastSuccessProb'][var_method][flow][i]
                    merged_results['EndToEndSuccessProb'][var_method][flow][i] += return_dict[exp]['EndToEndSuccessProb'][var_method][flow][i]

                for var_method in merged_results['MaxEpsilonIneqNonMarkingProb'].keys():
                    merged_results['MaxEpsilonIneqNonMarkingProb'][var_method][flow][i] += return_dict[exp]['MaxEpsilonIneqNonMarkingProb'][var_method][flow][i]
                    merged_results['MaxEpsilonIneqLastNonMarkingProb'][var_method][flow][i] += return_dict[exp]['MaxEpsilonIneqLastNonMarkingProb'][var_method][flow][i]
                    merged_results['EndToEndNonMarkingProb'][var_method][flow][i] += return_dict[exp]['EndToEndNonMarkingProb'][var_method][flow][i]


                merged_results['maxEpsilonDelay'][flow][i] += return_dict[exp]['maxEpsilonDelay'][flow][i]
                merged_results['maxEpsilonLastDelay'][flow][i] += return_dict[exp]['maxEpsilonLastDelay'][flow][i]
                merged_results['maxEpsilonSuccessProb'][flow][i] += return_dict[exp]['maxEpsilonSuccessProb'][flow][i]
                merged_results['maxEpsilonLastSuccessProb'][flow][i] += return_dict[exp]['maxEpsilonLastSuccessProb'][flow][i]
                merged_results['maxEpsilonNonMarkingProb'][flow][i] += return_dict[exp]['maxEpsilonNonMarkingProb'][flow][i]
                merged_results['maxEpsilonLastNonMarkingProb'][flow][i] += return_dict[exp]['maxEpsilonLastNonMarkingProb'][flow][i]
                merged_results['workLoad'][flow][i] += return_dict[exp]['workLoad'][flow][i]
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
    with open('../Results/results_{}/{}/Q_e_m_WOTx_forward_Results_forward_{}_{}_to_{}.json'.format(dir, rate, experiments_end, steadyStart, steadyEnd), 'w') as f:
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
    config.read('../Results/results_{}/Parameters.config'.format(args.dir))
    steadyStart = convert_to_float(config.get('Settings', 'steadyStart'))
    steadyEnd = convert_to_float(config.get('Settings', 'steadyEnd'))
    experiments = int(config.get('Settings', 'experiments'))
    if "forward" in args.dir:
        serviceRateScales = [float(x) for x in config.get('Settings', 'serviceRateScales').split(',')]
    elif "param" in args.dir:
        serviceRateScales = [float(x) for x in config.get('Settings', 'sampleRateScales').split(',')]
    else:
        serviceRateScales = [float(x) for x in config.get('Settings', 'errorRateScale').split(',')]
    # serviceRateScales = [0.79]
    # serviceRateScales = [1.0, 1.01, 1.03, 1.05]
    # serviceRateScales = [0.91, 0.93, 0.95, 0.97, 0.99, 1.01, 1.03, 1.05]
    # serviceRateScales = [float(x) for x in config.get('Settings', 'serviceRateScales').split(',')]
    # experiments = 1

    for rate in serviceRateScales:
        print("\nAnalyzing experiments for rate: ", rate)
        analyze_all_experiments(rate, steadyStart, steadyEnd, confidenceValue, args.dir, config, experiments_end=experiments, ns3_path=__ns3_path)
        print("Rate {} {} done".format(rate, experiments))

__main__()