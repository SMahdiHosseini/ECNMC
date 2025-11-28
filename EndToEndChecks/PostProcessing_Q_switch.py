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
# timeAvg_methods = ['poisson_eventAvg', 'eventAvg']
timeAvg_methods = ['linearInterp_timeAvg']
delay_timeAvg_vars = ['event']
# successProb_timeAvg_vars = ['event']
successProb_timeAvg_vars = ['probability']
nonMarkingProb_timeAvg_vars = ['event']
min_sample_size = 15

def check_MaxEpsilon_ineq_delay(endToEnd_statistics, samples_paths_aggregated_statistics):
    if abs(endToEnd_statistics - samples_paths_aggregated_statistics['DelayMean']) / samples_paths_aggregated_statistics['DelayMean'] <= samples_paths_aggregated_statistics['MaxEpsilonDelay']:
        return True
    else:
        return False

def check_MaxEpsilon_ineq_successProb(endToEnd_statistics, samples_paths_aggregated_statistics, number_of_segments):
    if (endToEnd_statistics - samples_paths_aggregated_statistics['SuccessProbMean'] <= (number_of_segments * np.log(1 + samples_paths_aggregated_statistics['MaxEpsilonSuccessProb']))) and (endToEnd_statistics - samples_paths_aggregated_statistics['SuccessProbMean'] >= (number_of_segments * np.log(1 - samples_paths_aggregated_statistics['MaxEpsilonSuccessProb']))):
        return True
    else:
        return False

def check_MaxEpsilon_ineq_nonMarkingProb(endToEnd_statistics, samples_paths_aggregated_statistics, number_of_segments):
    if (endToEnd_statistics - samples_paths_aggregated_statistics['NonMarkingProbMean'] <= (number_of_segments * np.log(1 + samples_paths_aggregated_statistics['MaxEpsilonNonMarkingProb']))) and (endToEnd_statistics - samples_paths_aggregated_statistics['NonMarkingProbMean'] >= (number_of_segments * np.log(1 - samples_paths_aggregated_statistics['MaxEpsilonNonMarkingProb']))):
        return True
    else:
        return False
    
def check_all_delayConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths):
    res = {}
    res['MaxEpsilonIneq'] = {}
    for flow in endToEnd_statistics.keys():
        res['MaxEpsilonIneq'][flow] = {}
        for path in paths:
            res['MaxEpsilonIneq'][flow][path] = {}
            for var_method in endToEnd_statistics[flow]['delay'].keys():
                if var_method != 'event_poisson_eventAvg' and var_method != 'event_eventAvg':
                    res['MaxEpsilonIneq'][flow][path][var_method] = check_MaxEpsilon_ineq_delay(endToEnd_statistics[flow]['delay'][var_method][path], samples_paths_aggregated_statistics[flow][path])
                else:
                    e = samples_paths_aggregated_statistics[flow][path]['DelayMean'] * samples_paths_aggregated_statistics[flow][path]['MaxEpsilonDelay'] # u * epsilon
                    e += endToEnd_statistics[flow]['delay'][var_method][path][1] * confidenceValue / np.sqrt(endToEnd_statistics[flow]['sampleSize']['delay'][path])
                    if (endToEnd_statistics[flow]['sampleSize']['delay'][path] < min_sample_size):
                        res['MaxEpsilonIneq'][flow][path][var_method] = False
                        continue
                    res['MaxEpsilonIneq'][flow][path][var_method] = (abs(endToEnd_statistics[flow]['delay'][var_method][path][0] - samples_paths_aggregated_statistics[flow][path]['DelayMean']) <= e)
    return res

def check_all_successProbConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths, number_of_segments):
    res = {}
    res['MaxEpsilonIneq'] = {}
    for flow in endToEnd_statistics.keys():
        res['MaxEpsilonIneq'][flow] = {}
        for path in paths:
            res['MaxEpsilonIneq'][flow][path] = {}
            for var_method in endToEnd_statistics[flow]['successProb'].keys():
                if var_method != 'event_poisson_eventAvg' and var_method != 'probability_poisson_eventAvg' and var_method != 'event_eventAvg' and var_method != 'probability_eventAvg':
                    res['MaxEpsilonIneq'][flow][path][var_method] = check_MaxEpsilon_ineq_successProb(np.log(endToEnd_statistics[flow]['successProb'][var_method][path]), samples_paths_aggregated_statistics[flow][path], number_of_segments)
                else:
                    epsp = (endToEnd_statistics[flow]['successProb'][var_method][path][1] * confidenceValue) / (np.sqrt(endToEnd_statistics[flow]['sampleSize']['successProb'][path]) * endToEnd_statistics[flow]['successProb'][var_method][path][0])
                    if (endToEnd_statistics[flow]['sampleSize']['successProb'][path] < min_sample_size):
                        res['MaxEpsilonIneq'][flow][path][var_method] = False
                        continue
                    e2e_p = np.log(endToEnd_statistics[flow]['successProb'][var_method][path][0])
                    if (e2e_p - samples_paths_aggregated_statistics[flow][path]['SuccessProbMean'] <= (number_of_segments * np.log(1 + samples_paths_aggregated_statistics[flow][path]['MaxEpsilonSuccessProb']) - np.log(1 - epsp))) and (e2e_p - samples_paths_aggregated_statistics[flow][path]['SuccessProbMean'] >= (number_of_segments * np.log(1 - samples_paths_aggregated_statistics[flow][path]['MaxEpsilonSuccessProb']) - np.log(1 + epsp))):
                        res['MaxEpsilonIneq'][flow][path][var_method] = True
                    else:
                        res['MaxEpsilonIneq'][flow][path][var_method] = False
    return res

def check_all_nonMarkingProbConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths, number_of_segments):
    res = {}
    res['MaxEpsilonIneq'] = {}
    for flow in endToEnd_statistics.keys():
        res['MaxEpsilonIneq'][flow] = {}
        for path in paths:
            res['MaxEpsilonIneq'][flow][path] = {}
            for var_method in endToEnd_statistics[flow]['nonMarkingProb'].keys():
                if var_method != 'event_poisson_eventAvg' and var_method != 'probability_poisson_eventAvg' and var_method != 'event_eventAvg' and var_method != 'probability_eventAvg':
                    res['MaxEpsilonIneq'][flow][path][var_method] = check_MaxEpsilon_ineq_nonMarkingProb(np.log(endToEnd_statistics[flow]['nonMarkingProb'][var_method][path]), samples_paths_aggregated_statistics[flow][path], number_of_segments)
                else:
                    epsp = (endToEnd_statistics[flow]['nonMarkingProb'][var_method][path][1] * confidenceValue) / (np.sqrt(endToEnd_statistics[flow]['sampleSize']['nonMarkingProb'][path]) * endToEnd_statistics[flow]['nonMarkingProb'][var_method][path][0])
                    if (endToEnd_statistics[flow]['sampleSize']['nonMarkingProb'][path] < min_sample_size):
                        res['MaxEpsilonIneq'][flow][path][var_method] = False
                        continue
                    e2e_p = np.log(endToEnd_statistics[flow]['nonMarkingProb'][var_method][path][0])
                    if (e2e_p - samples_paths_aggregated_statistics[flow][path]['NonMarkingProbMean'] <= (number_of_segments * np.log(1 + samples_paths_aggregated_statistics[flow][path]['MaxEpsilonNonMarkingProb']) - np.log(1 - epsp))) and (e2e_p - samples_paths_aggregated_statistics[flow][path]['NonMarkingProbMean'] >= (number_of_segments * np.log(1 - samples_paths_aggregated_statistics[flow][path]['MaxEpsilonNonMarkingProb']) - np.log(1 + epsp))):
                        res['MaxEpsilonIneq'][flow][path][var_method] = True
                    else:
                        res['MaxEpsilonIneq'][flow][path][var_method] = False          
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
    rounds_results['expSuccessDelay'] = {}
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
        rounds_results['expSuccessDelay'][flow] = []
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
    nonMarkingProb_results = check_all_nonMarkingProbConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths, number_of_segments)
    for flow in flows_name:
        for path in paths:
            for var_method in rounds_results['MaxEpsilonIneqDelay'].keys():
                if ('poisson_eventAvg' in var_method) and (endToEnd_statistics[flow]['sampleSize']['delay'][path] < min_sample_size):
                    continue
                rounds_results['MaxEpsilonIneqDelay'][var_method][flow][path][1] += 1
                if delay_results['MaxEpsilonIneq'][flow][path][var_method]:
                    rounds_results['MaxEpsilonIneqDelay'][var_method][flow][path][0]['WOBias'] += 1

            for var_method in rounds_results['MaxEpsilonIneqSuccessProb'].keys():
                if ('poisson_eventAvg' in var_method) and (endToEnd_statistics[flow]['sampleSize']['successProb'][path] < min_sample_size):
                    continue
                rounds_results['MaxEpsilonIneqSuccessProb'][var_method][flow][path][1] += 1
                if successProb_results['MaxEpsilonIneq'][flow][path][var_method]:
                    rounds_results['MaxEpsilonIneqSuccessProb'][var_method][flow][path][0]['WOBias'] += 1
            
            for var_method in rounds_results['MaxEpsilonIneqNonMarkingProb'].keys():
                if ('poisson_eventAvg' in var_method) and (endToEnd_statistics[flow]['sampleSize']['nonMarkingProb'][path] < min_sample_size):
                    continue
                rounds_results['MaxEpsilonIneqNonMarkingProb'][var_method][flow][path][1] += 1
                if nonMarkingProb_results['MaxEpsilonIneq'][flow][path][var_method]:
                    rounds_results['MaxEpsilonIneqNonMarkingProb'][var_method][flow][path][0]['WOBias'] += 1

            
def analyze_single_experiment(return_dict, rate, queues_names, confidenceValue, steadyStart, steadyEnd, rounds_results, results_folder, config, experiment=0, ns3_path=__ns3_path, differentiationDelay=None, errorRate=None, load=None):
    hostToTorLinkRate = convert_to_float(config.get('Settings', 'hostToTorLinkRate')) * 1e-3
    torToAggLinkRate = convert_to_float(config.get('Settings', 'torToAggLinkRate')) * rate * 1e-3
    switchSrcREDQueueDiscMaxSize = convert_to_float(config.get('Settings', 'switchSrcREDQueueDiscMaxSize'))
    switchREDQueueDiscMaxSize = convert_to_float(config.get('DCSim', 'switchREDQueueDiscMaxSize'))
    linkDelay = convert_to_float(config.get('Settings', 'hostToTorLinkDelay')) * 1e6
    passiveProbe = False if config.get('Settings', 'PassiveProbe') == "0" else True
    num_of_paths = 1 # this is the numnber of paths we want to consider for each flow, not the actual number of paths in the network
    nHosts = 24
    paths = range(num_of_paths)
    # if (experiment == 0):
    plot_queuingDelay_time(__ns3_path, results_folder, str(rate) + "/" + str(load), experiment, 'PoissonSampler_queueSize', steadyStart, steadyEnd, range(1), [hostToTorLinkRate, torToAggLinkRate, torToAggLinkRate, hostToTorLinkRate], switchREDQueueDiscMaxSize * 0.2)
    # endToEndStats = calculate_offline_computations_DC(__ns3_path, rate, 'EndToEnd_packets', str(experiment), results_folder, steadyStart, steadyEnd, "SentTime", nHosts, True, "IsReceived", [hostToTorLinkRate], [linkDelay, linkDelay, linkDelay, linkDelay], [0], differentiationDelay=differentiationDelay, errorRate=errorRate, load=load, passiveProbe=passiveProbe)
    endToEndStats = calculate_offline_computations_on_switch(__ns3_path, results_folder, rate, experiment, 'PoissonSampler_queueSize', steadyStart, steadyEnd, paths, [hostToTorLinkRate, torToAggLinkRate, torToAggLinkRate, hostToTorLinkRate], load, queues_names)
    # samples_dfs = calculate_offline_computations_DC(__ns3_path, rate, 'PoissonSampler_events', str(experiment), results_folder, endToEndStats['R0H0R2H0']['first'][0], endToEndStats['R0H0R2H0']['last'][0], "Time", nHosts, linkRates=[hostToTorLinkRate, torToAggLinkRate, torToAggLinkRate, hostToTorLinkRate], swtichDstREDQueueDiscMaxSize=[switchSrcREDQueueDiscMaxSize, switchREDQueueDiscMaxSize], differentiationDelay=differentiationDelay, errorRate=errorRate, load=load)
    samples_dfs = calculate_offline_computations_DC(__ns3_path, rate, 'PoissonSampler_events', str(experiment), results_folder, steadyStart, steadyEnd, "Time", nHosts, linkRates=[hostToTorLinkRate, torToAggLinkRate, torToAggLinkRate, hostToTorLinkRate], swtichDstREDQueueDiscMaxSize=[switchSrcREDQueueDiscMaxSize, switchREDQueueDiscMaxSize], differentiationDelay=differentiationDelay, errorRate=errorRate, load=load)

    averageDropProb = calculate_drop_rate_DC(samples_dfs)

    # samples_paths_statistics
    samples_paths_aggregated_statistics = {}
    for flow in queues_names:
        samples_paths_aggregated_statistics[flow] = {}
        for path in paths:
            samples_paths_aggregated_statistics[flow][path] = {}

            samples_paths_aggregated_statistics[flow][path]['DelayMean'] = samples_dfs[flow]['DelayMean']            
            samples_paths_aggregated_statistics[flow][path]['MaxEpsilonDelay'] = calc_epsilon(confidenceValue, samples_dfs[flow])
            # print(flow, path, samples_paths_aggregated_statistics[flow][path]['DelayMean'], samples_paths_aggregated_statistics[flow][path]['MaxEpsilonDelay'])

            samples_paths_aggregated_statistics[flow][path]['SuccessProbMean'] = np.log(samples_dfs[flow]['SuccessProbMean'])            
            samples_paths_aggregated_statistics[flow][path]['MaxEpsilonSuccessProb'] = calc_epsilon_loss(confidenceValue, samples_dfs[flow])
            # print(flow, path, samples_paths_aggregated_statistics[flow][path]['SuccessProbMean'], samples_paths_aggregated_statistics[flow][path]['MaxEpsilonSuccessProb'])

            samples_paths_aggregated_statistics[flow][path]['NonMarkingProbMean'] = np.log(samples_dfs[flow]['NonMarkingProbMean'])
            samples_paths_aggregated_statistics[flow][path]['MaxEpsilonNonMarkingProb'] = calc_epsilon_marking(confidenceValue, samples_dfs[flow])
            # print(flow, path, samples_paths_aggregated_statistics[flow][path]['NonMarkingProbMean'], samples_paths_aggregated_statistics[flow][path]['MaxEpsilonNonMarkingProb'])

    AverageWorkLoad = 0
    for flow in queues_names:
        for path in paths:
            for var_method in rounds_results['EndToEndDelayMean'].keys():
                if ('poisson_eventAvg' in var_method) and (endToEndStats[flow]['sampleSize']['delay'][path] < min_sample_size):
                    continue
                else:
                    rounds_results['EndToEndDelayMean'][var_method][flow][path][0].append(endToEndStats[flow]['delay'][var_method][path])
                    rounds_results['EndToEndDelayMean'][var_method][flow][path][1] = 1
            for var_method in rounds_results['EndToEndSuccessProb'].keys():
                if ('poisson_eventAvg' in var_method) and (endToEndStats[flow]['sampleSize']['successProb'][path] < min_sample_size):
                    continue
                else:
                    rounds_results['EndToEndSuccessProb'][var_method][flow][path][0].append(endToEndStats[flow]['successProb'][var_method][path])
                    rounds_results['EndToEndSuccessProb'][var_method][flow][path][1] = 1
            for var_method in rounds_results['EndToEndNonMarkingProb'].keys():
                if ('poisson_eventAvg' in var_method) and (endToEndStats[flow]['sampleSize']['nonMarkingProb'][path] < min_sample_size):
                    continue
                else:
                    rounds_results['EndToEndNonMarkingProb'][var_method][flow][path][0].append(endToEndStats[flow]['nonMarkingProb'][var_method][path])
                    rounds_results['EndToEndNonMarkingProb'][var_method][flow][path][1] = 1
            
            rounds_results['maxEpsilonDelay'][flow][path].append(samples_paths_aggregated_statistics[flow][path]['MaxEpsilonDelay'])
            rounds_results['maxEpsilonSuccessProb'][flow][path].append(samples_paths_aggregated_statistics[flow][path]['MaxEpsilonSuccessProb'])
            rounds_results['maxEpsilonNonMarkingProb'][flow][path].append(samples_paths_aggregated_statistics[flow][path]['MaxEpsilonNonMarkingProb'])
            rounds_results['EndToEndSampleSizeDelay'][flow][path].append(endToEndStats[flow]['sampleSize']['delay'][path])
            rounds_results['EndToEndSampleSizeSuccess'][flow][path].append(endToEndStats[flow]['sampleSize']['successProb'][path])
            rounds_results['EndToEndSampleSizeMarking'][flow][path].append(endToEndStats[flow]['sampleSize']['nonMarkingProb'][path])
            rounds_results['totalPckts'][flow][path].append(endToEndStats[flow]['totalPckts'][path])
            rounds_results['InterArrivals'][flow][path].append(endToEndStats[flow]['InterArrivals'][path])
            AverageWorkLoad += (endToEndStats[flow]['workload'][path])
    
        rounds_results['workLoad'][flow][path].append(endToEndStats[flow]['workload'][path])
        rounds_results['RTT'][flow][path].append(endToEndStats[flow]['RTT'][path])
    rounds_results['AverageWorkLoad'].append(AverageWorkLoad / len(endToEndStats.keys()))
    rounds_results['experiments'] += 1
    number_of_segments = 3
    compatibility_check(rounds_results, samples_paths_aggregated_statistics, endToEndStats, endToEndStats.keys(), range(num_of_paths), number_of_segments)
    rounds_results['DropRate'].append(averageDropProb)
    for flow in endToEndStats.keys():
        rounds_results['expSuccessDelay'][flow].append((experiment, rounds_results['MaxEpsilonIneqDelay']["event_linearInterp_timeAvg"][flow][0][0]['WOBias'], rounds_results['MaxEpsilonIneqDelay']["event_linearInterp_timeAvg"][flow][0][1]))
    for q in queues_names:
        rounds_results[q+'Delaystd'].append(samples_dfs[q]['DelayStd'])
        rounds_results[q+'DelayMean'].append(samples_dfs[q]['DelayMean'])
        rounds_results[q+'LastDelaystd'].append(samples_dfs[q]['LastDelayStd'])
        rounds_results[q+'LastDelayMean'].append(samples_dfs[q]['LastDelayMean'])
        rounds_results[q+'SuccessProbStd'].append(samples_dfs[q]['SuccessProbStd'])
        rounds_results[q+'SuccessProbMean'].append(samples_dfs[q]['SuccessProbMean'])
        rounds_results[q+'LastSuccessProbStd'].append(samples_dfs[q]['LastSuccessProbStd'])
        rounds_results[q+'LastSuccessProbMean'].append(samples_dfs[q]['LastSuccessProbMean'])
        rounds_results[q+'NonMarkingProbStd'].append(samples_dfs[q]['NonMarkingProbStd'])
        rounds_results[q+'NonMarkingProbMean'].append(samples_dfs[q]['NonMarkingProbMean'])
        rounds_results[q+'LastNonMarkingProbStd'].append(samples_dfs[q]['LastNonMarkingProbStd'])
        rounds_results[q+'LastNonMarkingProbMean'].append(samples_dfs[q]['LastNonMarkingProbMean'])
        rounds_results[q+'SampleSize'].append(samples_dfs[q]['sampleSize'])
        rounds_results[q+'InterArrivals'].append(samples_dfs[q]['InterArrivals'])
        rounds_results[q+'Occupancy'].append(samples_dfs[q]['Occupancy'])
        rounds_results[q+'PacktsInQueue'].append(samples_dfs[q]['PacktsInQueue'])
        rounds_results[q+'EmptyFrac'].append(samples_dfs[q]['EmptyFrac'])
        rounds_results[q+'GT1PktsFrac'].append(samples_dfs[q]['GT1PktsFrac'])
    return_dict[experiment] = rounds_results

def merge_results(return_dict, merged_results, flows, queues, num_of_paths):
    for exp in return_dict.keys():
        for q in queues:
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
            merged_results[q+'SampleSize'] += return_dict[exp][q+'SampleSize']
            merged_results[q+'InterArrivals'] += return_dict[exp][q+'InterArrivals']
            merged_results[q+'Occupancy'] += return_dict[exp][q+'Occupancy']
            merged_results[q+'PacktsInQueue'] += return_dict[exp][q+'PacktsInQueue']
            merged_results[q+'EmptyFrac'] += return_dict[exp][q+'EmptyFrac']
            merged_results[q+'GT1PktsFrac'] += return_dict[exp][q+'GT1PktsFrac']

    for flow in flows:
        for i in range(num_of_paths):
            for exp in return_dict.keys():

                for var_method in merged_results['MaxEpsilonIneqDelay'].keys():
                    merged_results['MaxEpsilonIneqDelay'][var_method][flow][i][1] += return_dict[exp]['MaxEpsilonIneqDelay'][var_method][flow][i][1]
                    merged_results['MaxEpsilonIneqLastDelay'][var_method][flow][i][1] += return_dict[exp]['MaxEpsilonIneqLastDelay'][var_method][flow][i][1]
                    merged_results['EndToEndDelayMean'][var_method][flow][i][1] += return_dict[exp]['EndToEndDelayMean'][var_method][flow][i][1]

                    merged_results['MaxEpsilonIneqDelay'][var_method][flow][i][0]['WBias'] += return_dict[exp]['MaxEpsilonIneqDelay'][var_method][flow][i][0]['WBias']
                    merged_results['MaxEpsilonIneqLastDelay'][var_method][flow][i][0]['WBias'] += return_dict[exp]['MaxEpsilonIneqLastDelay'][var_method][flow][i][0]['WBias']
                    merged_results['MaxEpsilonIneqDelay'][var_method][flow][i][0]['WOBias'] += return_dict[exp]['MaxEpsilonIneqDelay'][var_method][flow][i][0]['WOBias']
                    merged_results['MaxEpsilonIneqLastDelay'][var_method][flow][i][0]['WOBias'] += return_dict[exp]['MaxEpsilonIneqLastDelay'][var_method][flow][i][0]['WOBias']

                    merged_results['EndToEndDelayMean'][var_method][flow][i][0] += return_dict[exp]['EndToEndDelayMean'][var_method][flow][i][0]

                for var_method in merged_results['MaxEpsilonIneqSuccessProb'].keys():                    
                    merged_results['MaxEpsilonIneqSuccessProb'][var_method][flow][i][1] += return_dict[exp]['MaxEpsilonIneqSuccessProb'][var_method][flow][i][1]
                    merged_results['MaxEpsilonIneqLastSuccessProb'][var_method][flow][i][1] += return_dict[exp]['MaxEpsilonIneqLastSuccessProb'][var_method][flow][i][1]
                    merged_results['EndToEndSuccessProb'][var_method][flow][i][1] += return_dict[exp]['EndToEndSuccessProb'][var_method][flow][i][1]

                    merged_results['MaxEpsilonIneqSuccessProb'][var_method][flow][i][0]['WBias'] += return_dict[exp]['MaxEpsilonIneqSuccessProb'][var_method][flow][i][0]['WBias']
                    merged_results['MaxEpsilonIneqLastSuccessProb'][var_method][flow][i][0]['WBias'] += return_dict[exp]['MaxEpsilonIneqLastSuccessProb'][var_method][flow][i][0]['WBias']
                    merged_results['MaxEpsilonIneqSuccessProb'][var_method][flow][i][0]['WOBias'] += return_dict[exp]['MaxEpsilonIneqSuccessProb'][var_method][flow][i][0]['WOBias']
                    merged_results['MaxEpsilonIneqLastSuccessProb'][var_method][flow][i][0]['WOBias'] += return_dict[exp]['MaxEpsilonIneqLastSuccessProb'][var_method][flow][i][0]['WOBias']

                    merged_results['EndToEndSuccessProb'][var_method][flow][i][0] += return_dict[exp]['EndToEndSuccessProb'][var_method][flow][i][0]

                for var_method in merged_results['MaxEpsilonIneqNonMarkingProb'].keys():
                    merged_results['MaxEpsilonIneqNonMarkingProb'][var_method][flow][i][1] += return_dict[exp]['MaxEpsilonIneqNonMarkingProb'][var_method][flow][i][1]
                    merged_results['MaxEpsilonIneqLastNonMarkingProb'][var_method][flow][i][1] += return_dict[exp]['MaxEpsilonIneqLastNonMarkingProb'][var_method][flow][i][1]
                    merged_results['EndToEndNonMarkingProb'][var_method][flow][i][1] += return_dict[exp]['EndToEndNonMarkingProb'][var_method][flow][i][1]

                    merged_results['MaxEpsilonIneqNonMarkingProb'][var_method][flow][i][0]['WBias'] += return_dict[exp]['MaxEpsilonIneqNonMarkingProb'][var_method][flow][i][0]['WBias']
                    merged_results['MaxEpsilonIneqLastNonMarkingProb'][var_method][flow][i][0]['WBias'] += return_dict[exp]['MaxEpsilonIneqLastNonMarkingProb'][var_method][flow][i][0]['WBias']
                    merged_results['MaxEpsilonIneqNonMarkingProb'][var_method][flow][i][0]['WOBias'] += return_dict[exp]['MaxEpsilonIneqNonMarkingProb'][var_method][flow][i][0]['WOBias']
                    merged_results['MaxEpsilonIneqLastNonMarkingProb'][var_method][flow][i][0]['WOBias'] += return_dict[exp]['MaxEpsilonIneqLastNonMarkingProb'][var_method][flow][i][0]['WOBias']
                    merged_results['EndToEndNonMarkingProb'][var_method][flow][i][0] += return_dict[exp]['EndToEndNonMarkingProb'][var_method][flow][i][0]

                merged_results['TrafficsComptDelay']['event_poisson_eventAvg'][i][0] += return_dict[exp]['TrafficsComptDelay']['event_poisson_eventAvg'][i][0]
                merged_results['TrafficsComptDelay']['event_poisson_eventAvg'][i][1] += return_dict[exp]['TrafficsComptDelay']['event_poisson_eventAvg'][i][1]

                merged_results['maxEpsilonDelay'][flow][i] += return_dict[exp]['maxEpsilonDelay'][flow][i]
                merged_results['maxEpsilonLastDelay'][flow][i] += return_dict[exp]['maxEpsilonLastDelay'][flow][i]
                merged_results['maxEpsilonSuccessProb'][flow][i] += return_dict[exp]['maxEpsilonSuccessProb'][flow][i]
                merged_results['maxEpsilonLastSuccessProb'][flow][i] += return_dict[exp]['maxEpsilonLastSuccessProb'][flow][i]
                merged_results['maxEpsilonNonMarkingProb'][flow][i] += return_dict[exp]['maxEpsilonNonMarkingProb'][flow][i]
                merged_results['maxEpsilonLastNonMarkingProb'][flow][i] += return_dict[exp]['maxEpsilonLastNonMarkingProb'][flow][i]
                merged_results['workLoad'][flow][i] += return_dict[exp]['workLoad'][flow][i]
                merged_results['RTT'][flow][i] += return_dict[exp]['RTT'][flow][i]
                merged_results['EndToEndSampleSizeDelay'][flow][i] += return_dict[exp]['EndToEndSampleSizeDelay'][flow][i]
                merged_results['EndToEndSampleSizeSuccess'][flow][i] += return_dict[exp]['EndToEndSampleSizeSuccess'][flow][i]
                merged_results['EndToEndSampleSizeMarking'][flow][i] += return_dict[exp]['EndToEndSampleSizeMarking'][flow][i]
                merged_results['totalPckts'][flow][i] += return_dict[exp]['totalPckts'][flow][i]
                merged_results['InterArrivals'][flow][i] += return_dict[exp]['InterArrivals'][flow][i]
                merged_results['DelayBias'][flow][i] += return_dict[exp]['DelayBias'][flow][i]
                merged_results['SuccessProbBias'][flow][i] += return_dict[exp]['SuccessProbBias'][flow][i]
                merged_results['NonMarkingProbBias'][flow][i] += return_dict[exp]['NonMarkingProbBias'][flow][i]
                merged_results['ActiveFractionOfAll']['Packets'][flow][i] += return_dict[exp]['ActiveFractionOfAll']['Packets'][flow][i]
                merged_results['ActiveFractionOfAll']['Bytes'][flow][i] += return_dict[exp]['ActiveFractionOfAll']['Bytes'][flow][i]
                merged_results['ActiveFractionOfTagged']['Packets'][flow][i] += return_dict[exp]['ActiveFractionOfTagged']['Packets'][flow][i]
                merged_results['ActiveFractionOfTagged']['Bytes'][flow][i] += return_dict[exp]['ActiveFractionOfTagged']['Bytes'][flow][i]

    for exp in return_dict.keys():
        for flow in flows:
            merged_results['expSuccessDelay'][flow] += return_dict[exp]['expSuccessDelay'][flow]
        merged_results['experiments'] += return_dict[exp]['experiments']
        merged_results['DropRate'] += return_dict[exp]['DropRate']
        merged_results['AverageWorkLoad'] += return_dict[exp]['AverageWorkLoad']
    
def analyze_all_experiments(rate, steadyStart, steadyEnd, confidenceValue, dir, config, experiments_end=3, ns3_path=__ns3_path, load=None, differentiationDelay=None, errorRate=None):
    if ("delay" in dir) and ("reverse" in dir):
        # remove reverse from dir
        results_folder = 'Results_' + dir.replace("reverse", "forward").replace("delay_", "")
    else:
        results_folder = 'Results_' + dir
    num_of_paths = 1
    if ("delay" in dir) and ("reverse" in dir):
        flows_name = read_data_flowIndicator(ns3_path, rate, results_folder, differentiationDelay=None, errorRate=None, load=load)
        queues_names = read_queues_indicators(ns3_path, rate, results_folder, differentiationDelay=None, errorRate=None, load=load)
    else:
        flows_name = read_data_flowIndicator(ns3_path, rate, results_folder, differentiationDelay=differentiationDelay, errorRate=errorRate, load=load)
        queues_names = read_queues_indicators(ns3_path, rate, results_folder, differentiationDelay=differentiationDelay, errorRate=errorRate, load=load)
    queues_names = ["T0A0", "A0T2", "T2H3"]
    flows_name.sort()
    queues_names.sort()

    rounds_results = prepare_results(queues_names, queues_names, num_of_paths)
    merged_results = prepare_results(queues_names, queues_names, num_of_paths)
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
        merge_results(return_dict, merged_results, queues_names, queues_names, num_of_paths)
        print("{} joind".format(i))
    merged_results['AverageWorkLoad'] = sum(merged_results['AverageWorkLoad']) / merged_results['experiments']
    if differentiationDelay is not None and errorRate is not None:
        if differentiationDelay != 0.0:
            os.system('mkdir -p ../Results/results_{}/{}/{}/D_{}/f_{}/'.format(dir, rate, load, differentiationDelay, errorRate))
        with open('../Results/results_{}/{}/{}/D_{}/f_{}/Q_switch_1.0_{}_{}_to_{}.json'.format(dir, rate, load, differentiationDelay, errorRate, experiments_end, steadyStart, steadyEnd), 'w') as f:
            js.dump(merged_results, f, indent=4)
    else:
        with open('../Results/results_{}/{}/{}/Q_switch_1.0_{}_{}_to_{}.json'.format(dir, rate, load, experiments_end, steadyStart, steadyEnd), 'w') as f:
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
    # steadyStart = 0.05 * 1e9
    experiments = int(config.get('Settings', 'experiments'))
    experiments = 1
    serviceRateScales = [float(x) for x in config.get('Settings', 'serviceRateScales').split(',')]
    # serviceRateScales = [0.5, 1.0]
    loads = [float(x) for x in config.get('Settings', 'load').split(',')]
    # loads = [0.25, 0.7, 0.95]
    traffics = config.get('Settings', 'traffic').split(',')
    # traffics = ['Google_AllRPC', 'Facebook_HadoopDist_All', '']
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