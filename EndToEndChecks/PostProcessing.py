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
# sample_rate = 0.30
sample_rates = [0.5]
confidenceValue = 1.96 # 95% confidence interval
propagationDelay = 50000
# timeAvg_methods = ['rightCont_timeAvg', 'leftCont_timeAvg', 'linearInterp_timeAvg', 'poisson_eventAvg', 'eventAvg']
# timeAvg_methods = ['rightCont_timeAvg', 'leftCont_timeAvg', 'linearInterp_timeAvg']
timeAvg_methods = ['poisson_eventAvg', 'eventAvg']
delay_timeAvg_vars = ['event']
successProb_timeAvg_vars = ['event']
# successProb_timeAvg_vars = ['probability']
nonMarkingProb_timeAvg_vars = ['event']
min_sample_size = 15

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
    rounds_results['MaxEpsilonIneqLastDelay'] = {}
    rounds_results['MaxEpsilonIneqSuccessProb'] = {}
    rounds_results['MaxEpsilonIneqLastSuccessProb'] = {}
    rounds_results['MaxEpsilonIneqNonMarkingProb'] = {}
    rounds_results['MaxEpsilonIneqLastNonMarkingProb'] = {}
    rounds_results['EndToEndSampleSizeDelay'] = {}
    rounds_results['EndToEndSampleSizeSuccess'] = {}
    rounds_results['EndToEndSampleSizeMarking'] = {}
    rounds_results['totalPckts'] = {}
    rounds_results['InterArrivals'] = {}
    rounds_results['EndToEndDelayMean'] = {}
    rounds_results['EndToEndSuccessProb'] = {}
    rounds_results['EndToEndNonMarkingProb'] = {}
    rounds_results['DelayBias'] = {}
    rounds_results['SuccessProbBias'] = {}
    rounds_results['NonMarkingProbBias'] = {}
    rounds_results['DropRate'] = []
    rounds_results['maxEpsilonDelay'] = {}
    rounds_results['maxEpsilonLastDelay'] = {}
    rounds_results['maxEpsilonSuccessProb'] = {}
    rounds_results['maxEpsilonLastSuccessProb'] = {}
    rounds_results['maxEpsilonNonMarkingProb'] = {}
    rounds_results['maxEpsilonLastNonMarkingProb'] = {}
    rounds_results['workLoad'] = {}
    rounds_results['RTT'] = {}
    rounds_results['AverageWorkLoad'] = []
    rounds_results['experiments'] = 0
    rounds_results['TrafficsComptDelay'] = {}
    rounds_results['TrafficsComptDelay']['event_poisson_eventAvg'] = {}
    rounds_results['expSuccessDelay'] = []
    rounds_results['ActiveFractionOfAll'] = {}
    rounds_results['ActiveFractionOfAll']['Packets'] = {}
    rounds_results['ActiveFractionOfAll']['Bytes'] = {}
    rounds_results['ActiveFractionOfTagged'] = {}
    rounds_results['ActiveFractionOfTagged']['Packets'] = {}
    rounds_results['ActiveFractionOfTagged']['Bytes'] = {}
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
        rounds_results[q+'SampleSize'] = []
        rounds_results[q+'InterArrivals'] = []
        rounds_results[q+'Occupancy'] = []
        rounds_results[q+'PacktsInQueue'] = []
        rounds_results[q+'EmptyFrac'] = []
        rounds_results[q+'GT1PktsFrac'] = []
        rounds_results[q+'mixingRate'] = []
        rounds_results[q+'mixingSignalAvg'] = []
        rounds_results[q+'mixingRateMonly'] = []
        rounds_results[q+'mixingRatePoisson'] = []
        rounds_results[q+'mixingRateE2EPoisson'] = []
        rounds_results[q+'mixingRatePoissonEventAvg'] = []
        rounds_results[q+'mixingDelayDiff'] = []

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
        rounds_results['RTT'][flow] = {}
        rounds_results['maxEpsilonDelay'][flow] = {}
        rounds_results['maxEpsilonLastDelay'][flow] = {}
        rounds_results['maxEpsilonSuccessProb'][flow] = {}
        rounds_results['maxEpsilonLastSuccessProb'][flow] = {}
        rounds_results['maxEpsilonNonMarkingProb'][flow] = {}
        rounds_results['maxEpsilonLastNonMarkingProb'][flow] = {}
        rounds_results['EndToEndSampleSizeDelay'][flow] = {}
        rounds_results['EndToEndSampleSizeSuccess'][flow] = {}
        rounds_results['EndToEndSampleSizeMarking'][flow] = {}
        rounds_results['totalPckts'][flow] = {}
        rounds_results['InterArrivals'][flow] = {}
        rounds_results['DelayBias'][flow] = {}
        rounds_results['SuccessProbBias'][flow] = {}
        rounds_results['NonMarkingProbBias'][flow] = {}
        rounds_results['ActiveFractionOfAll']['Packets'][flow] = {}
        rounds_results['ActiveFractionOfAll']['Bytes'][flow] = {}
        rounds_results['ActiveFractionOfTagged']['Packets'][flow] = {}
        rounds_results['ActiveFractionOfTagged']['Bytes'][flow] = {}
        for i in range(num_of_agg_switches):
            rounds_results['TrafficsComptDelay']['event_poisson_eventAvg'][i] = [0, 0]
            for var_method in rounds_results['MaxEpsilonIneqDelay'].keys():
                rounds_results['MaxEpsilonIneqDelay'][var_method][flow][i] = [{'WBias': 0, 'WOBias': 0}, 0]
                rounds_results['MaxEpsilonIneqLastDelay'][var_method][flow][i] = [{'WBias': 0, 'WOBias': 0}, 0]
                rounds_results['EndToEndDelayMean'][var_method][flow][i] = [[], 0]

            for var_method in rounds_results['MaxEpsilonIneqSuccessProb'].keys():
                rounds_results['MaxEpsilonIneqSuccessProb'][var_method][flow][i] = [{'WBias': 0, 'WOBias': 0}, 0]
                rounds_results['MaxEpsilonIneqLastSuccessProb'][var_method][flow][i] = [{'WBias': 0, 'WOBias': 0}, 0]
                rounds_results['EndToEndSuccessProb'][var_method][flow][i] = [[], 0]
            
            for var_method in rounds_results['MaxEpsilonIneqNonMarkingProb'].keys():
                rounds_results['MaxEpsilonIneqNonMarkingProb'][var_method][flow][i] = [{'WBias': 0, 'WOBias': 0}, 0]
                rounds_results['MaxEpsilonIneqLastNonMarkingProb'][var_method][flow][i] = [{'WBias': 0, 'WOBias': 0}, 0]
                rounds_results['EndToEndNonMarkingProb'][var_method][flow][i] = [[], 0]
            
            rounds_results['workLoad'][flow][i] = []
            rounds_results['RTT'][flow][i] = []
            rounds_results['maxEpsilonDelay'][flow][i] = []
            rounds_results['maxEpsilonLastDelay'][flow][i] = []
            rounds_results['maxEpsilonSuccessProb'][flow][i] = []
            rounds_results['maxEpsilonLastSuccessProb'][flow][i] = []
            rounds_results['maxEpsilonNonMarkingProb'][flow][i] = []
            rounds_results['maxEpsilonLastNonMarkingProb'][flow][i] = []
            rounds_results['EndToEndSampleSizeDelay'][flow][i] = []
            rounds_results['EndToEndSampleSizeSuccess'][flow][i] = []
            rounds_results['EndToEndSampleSizeMarking'][flow][i] = []
            rounds_results['totalPckts'][flow][i] = []
            rounds_results['InterArrivals'][flow][i] = []
            rounds_results['DelayBias'][flow][i] = []
            rounds_results['SuccessProbBias'][flow][i] = []
            rounds_results['NonMarkingProbBias'][flow][i] = []
            rounds_results['ActiveFractionOfAll']['Packets'][flow][i] = []
            rounds_results['ActiveFractionOfAll']['Bytes'][flow][i] = []
            rounds_results['ActiveFractionOfTagged']['Packets'][flow][i] = []
            rounds_results['ActiveFractionOfTagged']['Bytes'][flow][i] = []

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

            
def analyze_single_experiment(return_dict, rate, queues_names, confidenceValue, steadyStart, steadyEnd, rounds_results, results_folder, config, experiment=0, ns3_path=__ns3_path, differentiationDelay=None, errorRate=None, load=None):
    hostToTorLinkRate = convert_to_float(config.get('Settings', 'hostToTorLinkRate')) * 1e-3
    torToAggLinkRate = convert_to_float(config.get('Settings', 'torToAggLinkRate')) * rate * 1e-3
    switchSrcREDQueueDiscMaxSize = convert_to_float(config.get('Settings', 'switchSrcREDQueueDiscMaxSize'))
    switchREDQueueDiscMaxSize = convert_to_float(config.get('DCSim', 'switchREDQueueDiscMaxSize'))
    linkDelay = convert_to_float(config.get('Settings', 'hostToTorLinkDelay')) * 1e6
    passiveProbe = False if config.get('Settings', 'PassiveProbe') == "0" else True
    num_of_paths = 1 # this is the numnber of paths we want to consider for each flow, not the actual number of paths in the network
    nHosts = 24
    paths = ['A' + str(i) for i in range(num_of_paths)]
    endToEndStats = calculate_offline_computations_DC(__ns3_path, rate, 'EndToEnd_packets', str(experiment), results_folder, steadyStart, steadyEnd, "SentTime", nHosts, True, "IsReceived", [hostToTorLinkRate], [linkDelay, linkDelay, linkDelay, linkDelay], [0], differentiationDelay=differentiationDelay, errorRate=errorRate, load=load, passiveProbe=passiveProbe)
    print(endToEndStats)
    samples_dfs = calculate_offline_computations_DC(__ns3_path, rate, 'PoissonSampler_events', str(experiment), results_folder, endToEndStats['R0H0R2H0']['first'][0], endToEndStats['R0H0R2H0']['last'][0], "Time", nHosts, linkRates=[hostToTorLinkRate, torToAggLinkRate, torToAggLinkRate, hostToTorLinkRate], swtichDstREDQueueDiscMaxSize=[switchSrcREDQueueDiscMaxSize, switchREDQueueDiscMaxSize], differentiationDelay=differentiationDelay, errorRate=errorRate, load=load)
    print(samples_dfs)
    # successProbs = read_lossProb(__ns3_path, rate, 'EndToEnd_packets', str(experiment), results_folder)
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

    # # samples_paths_statistics
    # samples_paths_aggregated_statistics = {}
    # for flow in endToEnd_dfs.keys():
    #     samples_paths_aggregated_statistics[flow] = {}
    #     for path in paths:
    #         samples_paths_aggregated_statistics[flow][path] = {}
    #         samples_paths_aggregated_statistics[flow][path]['DelayMean'] = sum([samples_dfs['R' + flow[1] + 'H' + flow[3]]['DelayMean'],
    #                                                                             samples_dfs['T' + flow[1] + path]['DelayMean'], 
    #                                                                             samples_dfs[path + 'T' + flow[5]]['DelayMean'],
    #                                                                             samples_dfs['T' + flow[5] + 'H' + flow[7]]['DelayMean']])
            
    #         samples_paths_aggregated_statistics[flow][path]['MaxEpsilonDelay'] = max([calc_epsilon(confidenceValue, samples_dfs['R' + flow[1] + 'H' + flow[3]]),
    #                                                                                   calc_epsilon(confidenceValue, samples_dfs['T' + flow[1] + path]),
    #                                                                                   calc_epsilon(confidenceValue, samples_dfs[path + 'T' + flow[5]]),
    #                                                                                   calc_epsilon(confidenceValue, samples_dfs['T' + flow[5] + 'H' + flow[7]])])
            
    #         samples_paths_aggregated_statistics[flow][path]['successProbMean'] = sum([np.log(samples_dfs['T' + flow[1] + path]['successProbMean']),
    #                                                                                   np.log(samples_dfs[path + 'T' + flow[5]]['successProbMean']),
    #                                                                                   np.log(samples_dfs['T' + flow[5] + 'H' + flow[7]]['successProbMean'])])
            
    #         samples_paths_aggregated_statistics[flow][path]['MaxEpsilonSuccessProb'] = max([calc_epsilon_loss(confidenceValue, samples_dfs['T' + flow[1] + path]),
    #                                                                                  calc_epsilon_loss(confidenceValue, samples_dfs[path + 'T' + flow[5]]),
    #                                                                                  calc_epsilon_loss(confidenceValue, samples_dfs['T' + flow[5] + 'H' + flow[7]])])    
    # # endToEnd_statistics
    # endToEnd_statistics = {}
    # AverageWorkLoad = 0
    # for flow in endToEnd_dfs.keys():
    #     endToEnd_statistics[flow] = {}
    #     for path in paths:
    #         endToEnd_statistics[flow][path] = {}
    #         endToEnd_statistics[flow][path]['DelayMean'] = endToEnd_dfs[flow]['timeAverage'][int(path[1])]
    #         endToEnd_statistics[flow][path]['successProbMean'] = {}
    #         endToEnd_statistics[flow][path]['successProbMean']['E2E_eventAvg'] = endToEnd_dfs[flow]['successProbMean'][int(path[1])]   
    #         endToEnd_statistics[flow][path]['successProbMean']['sentTime_est'] = successProbs[flow]['timeAvgSuccessProb'][path]       
    #         endToEnd_statistics[flow][path]['successProbMean']['poisson_sentTime_est'] = {}
    #         for sample_rate in sample_rates:
    #             endToEnd_statistics[flow][path]['successProbMean']['poisson_sentTime_est'][sample_rate] = successProbs_poisson[sample_rate][flow]['timeAvgSuccessProb'][path]
    #         # if (flow == 'R0H0R2H0' and path == 'A0'):
    #         #     print(flow, path, endToEnd_statistics[flow][path]['successProbMean']['E2E_eventAvg'], successProbs[flow]['timeAvgSuccessProb'][path], samples_paths_aggregated_statistics[flow][path]['successProbMean'], samples_paths_aggregated_statistics[flow][path]['MaxEpsilonSuccessProb'])
    #         # rounds_results['EndToEndSuccessProb'][flow][path].append(endToEnd_dfs[flow]['successProbMean'][int(path[1])])
    #         # rounds_results['EndToEndSuccessProb'][flow][path].append(successProbs[flow]['timeAvgSuccessProb'][path])
    #         rounds_results['EndToEndSuccessProb']['E2E_eventAvg'][flow][path].append(endToEnd_dfs[flow]['successProbMean'][int(path[1])])
    #         rounds_results['EndToEndSuccessProb']['sentTime_est'][flow][path].append(successProbs[flow]['timeAvgSuccessProb'][path])
    #         for sample_rate in sample_rates:
    #             rounds_results['EndToEndSuccessProb']['poisson_sentTime_est'][sample_rate][flow][path].append(successProbs_poisson[sample_rate][flow]['timeAvgSuccessProb'][path])
    #         rounds_results['EndToEndDelayMean'][flow][path].append(endToEnd_dfs[flow]['timeAverage'][int(path[1])])
    #         rounds_results['EndToEndDelayStd'][flow][path].append(endToEnd_dfs[flow]['DelayStd'][int(path[1])])
    #         rounds_results['maxEpsilonDelay'][flow][path].append(samples_paths_aggregated_statistics[flow][path]['MaxEpsilonDelay'])
    #         rounds_results['maxEpsilonSuccessProb'][flow][path].append(samples_paths_aggregated_statistics[flow][path]['MaxEpsilonSuccessProb'])
    #         rounds_results['errors'][flow][path].append(abs((samples_paths_aggregated_statistics[flow][path]['DelayMean'] - endToEnd_statistics[flow][path]['DelayMean']) / samples_paths_aggregated_statistics[flow][path]['DelayMean']))
    #         AverageWorkLoad += (endToEnd_dfs[flow]['receivedPackets'][int(path[1])] * endToEnd_dfs[flow]['averagePacketSize'][int(path[1])] * 8)
    
    #     rounds_results['workLoad'][flow][path].append(((endToEnd_dfs[flow]['receivedPackets'][0] * endToEnd_dfs[flow]['averagePacketSize'][0]) + (endToEnd_dfs[flow]['receivedPackets'][1] * endToEnd_dfs[flow]['averagePacketSize'][1])) * 8 / 0.5)
    # rounds_results['AverageWorkLoad'].append((AverageWorkLoad / 0.5) / 12)
    # rounds_results['experiments'] += 1
    # number_of_segments = 3
    # compatibility_check(rounds_results, samples_paths_aggregated_statistics, endToEnd_statistics, endToEnd_dfs.keys(), ['A' + str(i) for i in range(num_of_agg_switches)], number_of_segments)
    # # if rounds_results['MaxEpsilonIneqSuccessProb']['R0H0R2H0']['A1'] != 0:
    # #     print(path, np.log(endToEnd_statistics['R0H0R2H0']['A1']['successProbMean']), samples_paths_aggregated_statistics['R0H0R2H0']['A1']['successProbMean'], 
    # #         samples_paths_aggregated_statistics['R0H0R2H0']['A1']['MaxEpsilonSuccessProb'])
              
    # for q in queues_names:
    #     if q[0] == 'T' and q[2] == 'H' and (q[1] == '2' or q[1] == '3'):
    #         rounds_results[q+'Delaystd'].append(samples_dfs[q]['DelayStd'])
    #         rounds_results[q+'DelayMean'].append(samples_dfs[q]['DelayMean'])
    #     if q[0] == 'T' and q[2] == 'A' and (q[1] == '0' or q[1] == '1'):
    #         rounds_results[q+'Delaystd'].append(samples_dfs[q]['DelayStd'])
    #         rounds_results[q+'DelayMean'].append(samples_dfs[q]['DelayMean'])
    #     if q[0] == 'A' and q[2] == 'T' and (q[3] == '2' or q[3] == '3'):
    #         rounds_results[q+'Delaystd'].append(samples_dfs[q]['DelayStd'])
    #         rounds_results[q+'DelayMean'].append(samples_dfs[q]['DelayMean'])
    # return_dict[experiment] = rounds_results

def merge_results(return_dict, merged_results, flows, queues):
    num_of_agg_switches = 2
    for exp in return_dict.keys():
        for q in queues:
            if q[0] == 'T' and q[2] == 'H' and (q[1] == '2' or q[1] == '3'):
                merged_results[q+'Delaystd'] += return_dict[exp][q+'Delaystd']
                merged_results[q+'DelayMean'] += return_dict[exp][q+'DelayMean']
            if q[0] == 'T' and q[2] == 'A' and (q[1] == '0' or q[1] == '1'):
                merged_results[q+'Delaystd'] += return_dict[exp][q+'Delaystd']
                merged_results[q+'DelayMean'] += return_dict[exp][q+'DelayMean']
            if q[0] == 'A' and q[2] == 'T' and (q[3] == '2' or q[3] == '3'):
                merged_results[q+'Delaystd'] += return_dict[exp][q+'Delaystd']
                merged_results[q+'DelayMean'] += return_dict[exp][q+'DelayMean']

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
    
def analyze_all_experiments(rate, steadyStart, steadyEnd, confidenceValue, dir, config, experiments_end=3, ns3_path=__ns3_path, load=None, differentiationDelay=None, errorRate=None):
    if ("delay" in dir) and ("reverse" in dir):
        # remove reverse from dir
        results_folder = 'Results_' + dir.replace("reverse", "forward").replace("delay_", "")
    else:
        results_folder = 'Results_' + dir
    num_of_paths = 2
    if ("delay" in dir) and ("reverse" in dir):
        flows_name = read_data_flowIndicator(ns3_path, rate, results_folder, differentiationDelay=None, errorRate=None, load=load)
        queues_names = read_queues_indicators(ns3_path, rate, results_folder, differentiationDelay=None, errorRate=None, load=load)
    else:
        flows_name = read_data_flowIndicator(ns3_path, rate, results_folder, differentiationDelay=differentiationDelay, errorRate=errorRate, load=load)
        queues_names = read_queues_indicators(ns3_path, rate, results_folder, differentiationDelay=differentiationDelay, errorRate=errorRate, load=load)
    flows_name.sort()
    queues_names.sort()

    rounds_results = prepare_results(flows_name, queues_names, num_of_paths)
    merged_results = prepare_results(flows_name, queues_names, num_of_paths)
    batch_size = 30
    for i in range(int(experiments_end / batch_size) + 1):
        ths = []
        return_dict = multiprocessing.Manager().dict()
        for experiment in range(batch_size * i, min(experiments_end, batch_size * (i + 1))):
            if differentiationDelay is not None and errorRate is not None and ("delay" not in dir):
                if len(os.listdir('{}/scratch/{}/{}/{}/D_{}/f_{}/{}'.format(__ns3_path, results_folder, rate, load, differentiationDelay, errorRate, experiment))) == 0:
                    print(experiment)
                    continue
            else:
                if len(os.listdir('{}/scratch/{}/{}/{}/{}'.format(__ns3_path, results_folder, rate, load, experiment))) == 0:
                    print(experiment)
                    continue
            print("Analyzing experiment: ", experiment)
            ths.append(multiprocessing.Process(target=analyze_single_experiment, args=(return_dict, rate, queues_names, confidenceValue, steadyStart, steadyEnd, rounds_results, results_folder, config, experiment, ns3_path, differentiationDelay, errorRate, load)))
        
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        merge_results(return_dict, merged_results, flows_name, queues_names)
        print("{} joind".format(i))
    # merged_results['AverageWorkLoad'] = sum(merged_results['AverageWorkLoad']) / merged_results['experiments']
    if differentiationDelay is not None and errorRate is not None:
        if differentiationDelay != 0.0:
            os.system('mkdir -p ../Results/results_{}/{}/{}/D_{}/f_{}/'.format(dir, rate, load, differentiationDelay, errorRate))
        with open('../Results/results_{}/{}/{}/D_{}/f_{}/Q_e_m_passive_switch_1.0_{}_{}_to_{}.json'.format(dir, rate, load, differentiationDelay, errorRate, experiments_end, steadyStart, steadyEnd), 'w') as f:
            js.dump(merged_results, f, indent=4)
    else:
        with open('../Results/results_{}/{}/{}/Q_e_m_passive_switch_1.0_{}_{}_to_{}.json'.format(dir, rate, load, experiments_end, steadyStart, steadyEnd), 'w') as f:
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
    steadyStart = convert_to_float(config.get('Settings', 'steadyStart')) * 1e9
    steadyEnd = convert_to_float(config.get('Settings', 'steadyEnd')) * 1e9
    experiments = int(config.get('Settings', 'experiments'))
    experiments = 1
    # serviceRateScales = [float(x) for x in config.get('Settings', 'serviceRateScales').split(',')]
    serviceRateScales = [0.75]
    # loads = [float(x) for x in config.get('Settings', 'load').split(',')]
    loads = [0.2]
    # traffics = config.get('Settings', 'traffic').split(',')
    traffics = ['Google_AllRPC']
    errorRates = [float(x) for x in config.get('Settings', 'errorRate').split(',')]
    differentiationDelays = [float(x) for x in config.get('Settings', 'differentiationDelay').split(',')]
    if "forward" in args.dir:
        for traffic in traffics:
            for rate in serviceRateScales:
                for load in loads:
                    print("\nAnalyzing experiments for traffic {} rate: {} load: {}".format(traffic, rate, load))
                    analyze_all_experiments(rate, steadyStart, steadyEnd, confidenceValue, args.dir + "/" + traffic, config, experiments_end=experiments, ns3_path=__ns3_path, load=load)
                    print("Traffic {} Rate {} {} {} done".format(traffic, rate, load, experiments))
                print("Traffic {} Rate {} done".format(traffic, rate))
            print("Traffic {} done".format(traffic))
    else:
        for traffic in traffics:
            for rate in serviceRateScales:
                for load in loads:
                    for differentiationDelay in differentiationDelays:
                        for errorRate in errorRates:
                            print("\nAnalyzing experiments for rate: ", rate, " load: ", load, " differentiationDelay: ", differentiationDelay, " errorRate: ", errorRate)
                            os.system('mkdir -p ../Results/results_{}/{}/{}/{}/D_{}/f_{}/'.format(args.dir, traffic, rate, load, differentiationDelay, errorRate))
                            analyze_all_experiments(rate, steadyStart, steadyEnd, confidenceValue, args.dir + "/" + traffic, config, experiments_end=experiments, ns3_path=__ns3_path, differentiationDelay=differentiationDelay, errorRate=errorRate, load=load)
                            print("Rate {} load {} with {} and {} done".format(rate, load, differentiationDelay, errorRate))
                    print("Traffic {} Rate {} load {} done".format(traffic, rate, load))
                print("Rate {} done".format(rate))
            print("Traffic {} done".format(traffic))

__main__()