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
import pickle
import multiprocessing
import argparse
import time

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
min_sample_size = 30
DelayConsistencyGaurantee = 0.40 # we can tolerate up to 40% difference between the end-to-end delay and the sum of per-segment delays, with 95% confidence

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
                if (endToEnd_statistics[flow]['sampleSize']['delay'][path] < min_sample_size):
                    res['MaxEpsilonIneq'][flow][path][var_method] = False
                    continue
                if var_method != 'event_poisson_eventAvg' and var_method != 'event_eventAvg':
                    res['MaxEpsilonIneq'][flow][path][var_method] = check_MaxEpsilon_ineq_delay(endToEnd_statistics[flow]['delay'][var_method][path], samples_paths_aggregated_statistics[flow][path])
                else:
                    e = samples_paths_aggregated_statistics[flow][path]['DelayMean'] * samples_paths_aggregated_statistics[flow][path]['MaxEpsilonDelay'] # u * epsilon
                    # e += endToEnd_statistics[flow]['delay'][var_method][path][1] * confidenceValue # using e2e samples std as e2e std
                    e += confidenceValue * samples_paths_aggregated_statistics[flow][path]['e2eDelayStd'] / np.sqrt(endToEnd_statistics[flow]['sampleSize']['delay'][path]) # using sum of stds as e2e std
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
                    # epsp = (endToEnd_statistics[flow]['successProb'][var_method][path][1] * confidenceValue) / (endToEnd_statistics[flow]['successProb'][var_method][path][0]) # using e2e samples std as e2e std
                    epsp = (samples_paths_aggregated_statistics[flow][path]['e2eSuccessProbStd'] * confidenceValue) / (endToEnd_statistics[flow]['successProb'][var_method][path][0] * np.sqrt(endToEnd_statistics[flow]['sampleSize']['successProb'][path])) # using sum of stds as e2e std
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
                    # epsp = (endToEnd_statistics[flow]['nonMarkingProb'][var_method][path][1] * confidenceValue) / (endToEnd_statistics[flow]['nonMarkingProb'][var_method][path][0]) # using e2e samples std as e2e std
                    epsp = (samples_paths_aggregated_statistics[flow][path]['e2eNonMarkingProbStd'] * confidenceValue) / (endToEnd_statistics[flow]['nonMarkingProb'][var_method][path][0] * np.sqrt(endToEnd_statistics[flow]['sampleSize']['nonMarkingProb'][path])) # using sum of stds as e2e std
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
    rounds_results['EndToEndSubsamplingErrorDelay'] = {}
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
    rounds_results['e2eVsSwitchCCFpercntg'] = {}
    rounds_results['e2eVsSwitchMaxCCF'] = {}
    rounds_results['e2eCorrArrivals'] = {}
    rounds_results['MinimumE2ESampleSizeDelay'] = {}
    rounds_results['MinimumE2ESampleSizeSuccessProb'] = {}
    rounds_results['MinimumE2ESampleSizeNonMarkingProb'] = {}
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
        rounds_results[q+'MinimumDelayBias'] = []

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
        rounds_results['EndToEndSubsamplingErrorDelay'][flow] = {}
        rounds_results['EndToEndSampleSizeSuccess'][flow] = {}
        rounds_results['EndToEndSampleSizeMarking'][flow] = {}
        rounds_results['e2eVsSwitchCCFpercntg'][flow] = {}
        rounds_results['e2eVsSwitchMaxCCF'][flow] = {}
        rounds_results['e2eCorrArrivals'][flow] = {}
        rounds_results['MinimumE2ESampleSizeDelay'][flow] = {}
        rounds_results['MinimumE2ESampleSizeSuccessProb'][flow] = {}
        rounds_results['MinimumE2ESampleSizeNonMarkingProb'][flow] = {}
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
            rounds_results['EndToEndSubsamplingErrorDelay'][flow][i] = []
            rounds_results['EndToEndSampleSizeSuccess'][flow][i] = []
            rounds_results['EndToEndSampleSizeMarking'][flow][i] = []
            rounds_results['totalPckts'][flow][i] = []
            rounds_results['InterArrivals'][flow][i] = []
            rounds_results['DelayBias'][flow][i] = []
            rounds_results['e2eVsSwitchCCFpercntg'][flow][i] = []
            rounds_results['e2eVsSwitchMaxCCF'][flow][i] = []
            rounds_results['e2eCorrArrivals'][flow][i] = []
            rounds_results['MinimumE2ESampleSizeDelay'][flow][i] = []
            rounds_results['MinimumE2ESampleSizeSuccessProb'][flow][i] = []
            rounds_results['MinimumE2ESampleSizeNonMarkingProb'][flow][i] = []
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

            
def analyze_single_experiment(return_dict, rate, queues_names, confidenceValue, steadyStart, steadyEnd, rounds_results, results_folder, config, experiment=0, ns3_path=__ns3_path, differentiationDelay=None, errorRate=None, load=None, flow_names=[], queue_names=[]):
    hostToTorLinkRate = convert_to_float(config.get('Settings', 'hostToTorLinkRate')) * 1e-3
    torToAggLinkRate = convert_to_float(config.get('Settings', 'torToAggLinkRate')) * rate * 1e-3
    switchSrcREDQueueDiscMaxSize = convert_to_float(config.get('Settings', 'switchSrcREDQueueDiscMaxSize'))
    switchREDQueueDiscMaxSize = convert_to_float(config.get('DCSim', 'switchREDQueueDiscMaxSize')) * rate
    linkDelay = convert_to_float(config.get('Settings', 'hostToTorLinkDelay')) * 1e6
    passiveProbe = False if config.get('Settings', 'PassiveProbe') == "0" else True
    num_of_paths = 1 # this is the numnber of paths we want to consider for each flow, not the actual number of paths in the network
    number_of_segments = 3
    nHosts = 24
    paths = range(num_of_paths)
    # endToEndStats = calculate_offline_computations_DC(__ns3_path, rate, 'EndToEnd_packets', str(experiment), results_folder, steadyStart, steadyEnd, "SentTime", nHosts, True, "IsReceived", [hostToTorLinkRate], [linkDelay, linkDelay, linkDelay, linkDelay], [0], differentiationDelay=differentiationDelay, errorRate=errorRate, load=load, passiveProbe=passiveProbe, flow_names=flow_names)
    samples_dfs = calculate_offline_computations_DC(__ns3_path, rate, 'PoissonSampler_events', str(experiment), results_folder, steadyStart, steadyEnd, "Time", nHosts, 
                                                    linkRates=[hostToTorLinkRate, torToAggLinkRate, torToAggLinkRate, hostToTorLinkRate], 
                                                    linkDelays=[linkDelay, linkDelay, linkDelay, linkDelay],
                                                    swtichDstREDQueueDiscMaxSize=[switchSrcREDQueueDiscMaxSize, switchREDQueueDiscMaxSize], 
                                                    differentiationDelay=differentiationDelay, errorRate=errorRate, load=load, queue_names=queue_names)

    averageDropProb = calculate_drop_rate_DC(samples_dfs)

    # samples_paths_statistics
    samples_paths_aggregated_statistics = {}
    for flow in flow_names:
        samples_paths_aggregated_statistics[flow] = {}
        for path in paths:
            samples_paths_aggregated_statistics[flow][path] = {}
            samples_paths_aggregated_statistics[flow][path]['DelayMean'] = sum([samples_dfs['T' + flow[1] + 'A' + str(path)]['DelayMean'], 
                                                                               samples_dfs['A' + str(path) + 'T' + flow[5]]['DelayMean'],
                                                                               samples_dfs['T' + flow[5] + 'H' + flow[7]]['DelayMean']])
            print("Delay Mean:", samples_paths_aggregated_statistics[flow][path]['DelayMean'])
            samples_paths_aggregated_statistics[flow][path]['MaxEpsilonDelay'] = max([calc_epsilon(confidenceValue, samples_dfs['T' + flow[1] + 'A' + str(path)]),
                                                                                      calc_epsilon(confidenceValue, samples_dfs['A' + str(path) + 'T' + flow[5]]),
                                                                                      calc_epsilon(confidenceValue, samples_dfs['T' + flow[5] + 'H' + flow[7]])])
            samples_paths_aggregated_statistics[flow][path]['e2eDelayStd'] = sum([samples_dfs['T' + flow[1] + 'A' + str(path)]['DelayStd'],
                                                                                  samples_dfs['A' + str(path) + 'T' + flow[5]]['DelayStd'],
                                                                                  samples_dfs['T' + flow[5] + 'H' + flow[7]]['DelayStd']])
            samples_paths_aggregated_statistics[flow][path]['MinimumE2ESampleSizeDelay'] = calc_min_e2e_samples(confidenceValue, DelayConsistencyGaurantee, samples_paths_aggregated_statistics[flow][path], metric='Delay')
            # print(flow, path, samples_paths_aggregated_statistics[flow][path]['DelayMean'], samples_paths_aggregated_statistics[flow][path]['MaxEpsilonDelay'])
            samples_paths_aggregated_statistics[flow][path]['SuccessProbMean'] = sum([np.log(samples_dfs['T' + flow[1] + 'A' + str(path)]['SuccessProbMean']),
                                                                                      np.log(samples_dfs['A' + str(path) + 'T' + flow[5]]['SuccessProbMean']),
                                                                                      np.log(samples_dfs['T' + flow[5] + 'H' + flow[7]]['SuccessProbMean'])])
            
            samples_paths_aggregated_statistics[flow][path]['MaxEpsilonSuccessProb'] = max([calc_epsilon_loss(confidenceValue, samples_dfs['T' + flow[1] + 'A' + str(path)]),
                                                                                     calc_epsilon_loss(confidenceValue, samples_dfs['A' + str(path) + 'T' + flow[5]]),
                                                                                     calc_epsilon_loss(confidenceValue, samples_dfs['T' + flow[5] + 'H' + flow[7]])])
            ## TODO: fix the variance calculation for probabilities
            samples_paths_aggregated_statistics[flow][path]['e2eSuccessProbStd'] = sum([samples_dfs['T' + flow[1] + 'A' + str(path)]['SuccessProbStd'],
                                                                                        samples_dfs['A' + str(path) + 'T' + flow[5]]['SuccessProbStd'],
                                                                                        samples_dfs['T' + flow[5] + 'H' + flow[7]]['SuccessProbStd']])
            samples_paths_aggregated_statistics[flow][path]['MinimumE2ESampleSizeSuccessProb'] = calc_min_e2e_samples_prob(confidenceValue, DelayConsistencyGaurantee, samples_paths_aggregated_statistics[flow][path], number_of_segments, metric='SuccessProb')
            # print(flow, path, samples_paths_aggregated_statistics[flow][path]['SuccessProbMean'], samples_paths_aggregated_statistics[flow][path]['MaxEpsilonSuccessProb'])

            samples_paths_aggregated_statistics[flow][path]['NonMarkingProbMean'] = sum([np.log(samples_dfs['T' + flow[1] + 'A' + str(path)]['NonMarkingProbMean']),
                                                                                        np.log(samples_dfs['A' + str(path) + 'T' + flow[5]]['NonMarkingProbMean']),
                                                                                        np.log(samples_dfs['T' + flow[5] + 'H' + flow[7]]['NonMarkingProbMean'])])
            samples_paths_aggregated_statistics[flow][path]['MaxEpsilonNonMarkingProb'] = max([calc_epsilon_marking(confidenceValue, samples_dfs['T' + flow[1] + 'A' + str(path)]),
                                                                                               calc_epsilon_marking(confidenceValue, samples_dfs['A' + str(path) + 'T' + flow[5]]),
                                                                                               calc_epsilon_marking(confidenceValue, samples_dfs['T' + flow[5] + 'H' + flow[7]])])
            samples_paths_aggregated_statistics[flow][path]['e2eNonMarkingProbStd'] = sum([samples_dfs['T' + flow[1] + 'A' + str(path)]['NonMarkingProbStd'],
                                                                                           samples_dfs['A' + str(path) + 'T' + flow[5]]['NonMarkingProbStd'],
                                                                                           samples_dfs['T' + flow[5] + 'H' + flow[7]]['NonMarkingProbStd']])
            samples_paths_aggregated_statistics[flow][path]['MinimumE2ESampleSizeNonMarkingProb'] = calc_min_e2e_samples_prob(confidenceValue, DelayConsistencyGaurantee, samples_paths_aggregated_statistics[flow][path], number_of_segments, metric='NonMarkingProb')
            # print(flow, path, samples_paths_aggregated_statistics[flow][path]['NonMarkingProbMean'], samples_paths_aggregated_statistics[flow][path]['MaxEpsilonNonMarkingProb'])
    delay_bias_results = {}
    if queue_names:
        average_packet_size = samples_dfs[queue_names[0]]['avgPacktSize']
        for queue_name in queue_names:
            queue_stats = samples_dfs[queue_name]
            delay_bias_results[queue_name + 'bias'] = 0
            delay_bias_results[queue_name + 'e2e_samples_queue_delay_mean'] = queue_stats['DelayMean']
            delay_bias_results[queue_name + 'poisson_samples_queue_delay_mean'] = queue_stats['DelayMean']
            delay_bias_results[queue_name + 'poisson_prob_non_empty'] = 1 - queue_stats['EmptyFrac'] / 100
            delay_bias_results[queue_name + 'error_bound'] = 0
        # delay_bias_results = compute_bias_based_on_average_packet_size(
        #     delay_bias_results,
        #     average_packet_size,
        #     queue_names,
        #     [hostToTorLinkRate, torToAggLinkRate, torToAggLinkRate, hostToTorLinkRate]
        # )
        for flow in flow_names:
            for path in paths:
                samples_paths_aggregated_statistics[flow][path]['DelayBias'] = sum([
                    delay_bias_results['T' + flow[1] + 'A' + str(path) + 'bias'],
                    delay_bias_results['A' + str(path) + 'T' + flow[5] + 'bias'],
                    delay_bias_results['T' + flow[5] + 'H' + flow[7] + 'bias'],
                ])
        for queue_name in queue_names:
            rounds_results[queue_name+'MinimumDelayBias'].append(delay_bias_results[queue_name + 'bias'])
    # This pass also writes the three-way delay CDF comparison for each flow/path.
    endToEndStats = calculate_offline_computations_DC(__ns3_path, rate, 'EndToEnd_packets', str(experiment), results_folder, steadyStart, steadyEnd, "SentTime", nHosts, True, "IsReceived",
                                                      [hostToTorLinkRate, torToAggLinkRate, torToAggLinkRate, hostToTorLinkRate], [linkDelay, linkDelay, linkDelay, linkDelay],
                                                      [switchREDQueueDiscMaxSize, switchREDQueueDiscMaxSize, switchSrcREDQueueDiscMaxSize], differentiationDelay=differentiationDelay,
                                                      errorRate=errorRate, load=load, passiveProbe=passiveProbe, flow_names=flow_names, samples_paths_aggregated_statistics=samples_paths_aggregated_statistics, queue_names=queue_names)

    AverageWorkLoad = 0
    for flow in flow_names:
        for path in paths:
        #     for var_method in rounds_results['EndToEndDelayMean'].keys():
        #         rounds_results['EndToEndDelayMean'][var_method][flow][path][0].append(endToEndStats[flow]['delay'][var_method][path])
        #         rounds_results['EndToEndDelayMean'][var_method][flow][path][1] = 1
            # for var_method in rounds_results['EndToEndSuccessProb'].keys():
        #         rounds_results['EndToEndSuccessProb'][var_method][flow][path][0].append(endToEndStats[flow]['successProb'][var_method][path])
        #         rounds_results['EndToEndSuccessProb'][var_method][flow][path][1] = 1
            # for var_method in rounds_results['EndToEndNonMarkingProb'].keys():
        #         rounds_results['EndToEndNonMarkingProb'][var_method][flow][path][0].append(endToEndStats[flow]['nonMarkingProb'][var_method][path])
        #         rounds_results['EndToEndNonMarkingProb'][var_method][flow][path][1] = 1
            
            rounds_results['maxEpsilonDelay'][flow][path].append(samples_paths_aggregated_statistics[flow][path]['MaxEpsilonDelay'])
            rounds_results['maxEpsilonSuccessProb'][flow][path].append(samples_paths_aggregated_statistics[flow][path]['MaxEpsilonSuccessProb'])
            rounds_results['maxEpsilonNonMarkingProb'][flow][path].append(samples_paths_aggregated_statistics[flow][path]['MaxEpsilonNonMarkingProb'])
            rounds_results['MinimumE2ESampleSizeDelay'][flow][path].append(samples_paths_aggregated_statistics[flow][path]['MinimumE2ESampleSizeDelay'])
            rounds_results['MinimumE2ESampleSizeSuccessProb'][flow][path].append(samples_paths_aggregated_statistics[flow][path]['MinimumE2ESampleSizeSuccessProb'])
            rounds_results['MinimumE2ESampleSizeNonMarkingProb'][flow][path].append(samples_paths_aggregated_statistics[flow][path]['MinimumE2ESampleSizeNonMarkingProb'])
            # rounds_results['EndToEndSampleSizeDelay'][flow][path].append(endToEndStats[flow]['sampleSize']['delay'][path])
            # rounds_results['EndToEndSubsamplingErrorDelay'][flow][path].append(endToEndStats[flow]['subSamplingError']['delay'][path])
            # rounds_results['EndToEndSampleSizeSuccess'][flow][path].append(endToEndStats[flow]['sampleSize']['successProb'][path])
            # rounds_results['EndToEndSampleSizeMarking'][flow][path].append(endToEndStats[flow]['sampleSize']['nonMarkingProb'][path])
            # rounds_results['totalPckts'][flow][path].append(endToEndStats[flow]['totalPckts'][path])
            # rounds_results['InterArrivals'][flow][path].append(endToEndStats[flow]['InterArrivals'][path])
            # rounds_results['e2eVsSwitchCCFpercntg'][flow][path].append(endToEndStats[flow]['Corr'][path]['e2eVsSwitchCCFpercntg'])
            # rounds_results['e2eVsSwitchMaxCCF'][flow][path].append(endToEndStats[flow]['Corr'][path]['e2eVsSwitchMaxCCF'])
            # rounds_results['e2eCorrArrivals'][flow][path].append(endToEndStats[flow]['Corr'][path]['e2eCorrArrivals'])
            rounds_results['DelayBias'][flow][path].append(samples_paths_aggregated_statistics[flow][path]['DelayBias'])
            # rounds_results['SuccessProbBias'][flow][path].append(endToEndStats[flow]['bias']['successProb'][path])
            # rounds_results['NonMarkingProbBias'][flow][path].append(endToEndStats[flow]['bias']['nonMarkingProb'][path])
            # rounds_results['ActiveFractionOfAll']['Packets'][flow][path].append(endToEndStats[flow]['ActiveFractionOfAll']['Packets'])
            # rounds_results['ActiveFractionOfAll']['Bytes'][flow][path].append(endToEndStats[flow]['ActiveFractionOfAll']['Bytes'])
            # rounds_results['ActiveFractionOfTagged']['Packets'][flow][path].append(endToEndStats[flow]['ActiveFractionOfTagged']['Packets'])
            # rounds_results['ActiveFractionOfTagged']['Bytes'][flow][path].append(endToEndStats[flow]['ActiveFractionOfTagged']['Bytes'])
            # AverageWorkLoad += (endToEndStats[flow]['workload'][path])
    
    #     # rounds_results['workLoad'][flow][path].append(endToEndStats[flow]['workload'][path])
    #     # rounds_results['RTT'][flow][path].append(endToEndStats[flow]['RTT'][path])
    # rounds_results['AverageWorkLoad'].append(AverageWorkLoad / len(endToEndStats.keys()))
    # rounds_results['experiments'] += 1
    # number_of_segments = 3
    # compatibility_check(rounds_results, samples_paths_aggregated_statistics, endToEndStats, endToEndStats.keys(), range(num_of_paths), number_of_segments)
    # rounds_results['expSuccessDelay'].append((experiment, rounds_results['MaxEpsilonIneqDelay']["event_poisson_eventAvg"][flow][0][0]['WOBias'], rounds_results['MaxEpsilonIneqDelay']["event_poisson_eventAvg"][flow][0][1]))
 
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

def merge_results(return_dict, merged_results, flows, queues, num_of_paths, experiments):
    for exp in experiments:
        merged_results['expSuccessDelay'] += return_dict[exp]['expSuccessDelay']
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
            merged_results[q+'MinimumDelayBias'] += return_dict[exp][q+'MinimumDelayBias']

    for flow in flows:
        for i in range(num_of_paths):
            for exp in experiments:
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

                merged_results['MinimumE2ESampleSizeDelay'][flow][i] += return_dict[exp]['MinimumE2ESampleSizeDelay'][flow][i]
                merged_results['MinimumE2ESampleSizeSuccessProb'][flow][i] += return_dict[exp]['MinimumE2ESampleSizeSuccessProb'][flow][i]
                merged_results['MinimumE2ESampleSizeNonMarkingProb'][flow][i] += return_dict[exp]['MinimumE2ESampleSizeNonMarkingProb'][flow][i]
                merged_results['maxEpsilonDelay'][flow][i] += return_dict[exp]['maxEpsilonDelay'][flow][i]
                merged_results['maxEpsilonLastDelay'][flow][i] += return_dict[exp]['maxEpsilonLastDelay'][flow][i]
                merged_results['maxEpsilonSuccessProb'][flow][i] += return_dict[exp]['maxEpsilonSuccessProb'][flow][i]
                merged_results['maxEpsilonLastSuccessProb'][flow][i] += return_dict[exp]['maxEpsilonLastSuccessProb'][flow][i]
                merged_results['maxEpsilonNonMarkingProb'][flow][i] += return_dict[exp]['maxEpsilonNonMarkingProb'][flow][i]
                merged_results['maxEpsilonLastNonMarkingProb'][flow][i] += return_dict[exp]['maxEpsilonLastNonMarkingProb'][flow][i]
                merged_results['workLoad'][flow][i] += return_dict[exp]['workLoad'][flow][i]
                merged_results['RTT'][flow][i] += return_dict[exp]['RTT'][flow][i]
                merged_results['EndToEndSampleSizeDelay'][flow][i] += return_dict[exp]['EndToEndSampleSizeDelay'][flow][i]
                merged_results['EndToEndSubsamplingErrorDelay'][flow][i] += return_dict[exp]['EndToEndSubsamplingErrorDelay'][flow][i]
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
                merged_results['e2eVsSwitchCCFpercntg'][flow][i] += return_dict[exp]['e2eVsSwitchCCFpercntg'][flow][i]
                merged_results['e2eVsSwitchMaxCCF'][flow][i] += return_dict[exp]['e2eVsSwitchMaxCCF'][flow][i]
                merged_results['e2eCorrArrivals'][flow][i] += return_dict[exp]['e2eCorrArrivals'][flow][i]

    for exp in experiments:
        merged_results['experiments'] += return_dict[exp]['experiments']
        merged_results['DropRate'] += return_dict[exp]['DropRate']
        merged_results['AverageWorkLoad'] += return_dict[exp]['AverageWorkLoad']
    
def analyze_all_experiments(rate, steadyStart, steadyEnd, confidenceValue, dir, config, experiments_end=3, ns3_path=__ns3_path, load=None, differentiationDelay=None, errorRate=None):
    # if ("delay" in dir) and ("reverse" in dir):
    #     # remove reverse from dir
    #     results_folder = 'Results_' + dir.replace("reverse", "forward").replace("delay_", "")
    # else:
    results_folder = 'Results_' + dir
    num_of_paths = 1
    # if ("delay" in dir) and ("reverse" in dir):
    #     flows_name = read_data_flowIndicator(ns3_path, rate, results_folder, differentiationDelay=None, errorRate=None, load=load)
    #     queues_names = read_queues_indicators(ns3_path, rate, results_folder, differentiationDelay=None, errorRate=None, load=load)
    # else:
    flows_name = read_data_flowIndicator(ns3_path, rate, results_folder, differentiationDelay=differentiationDelay, errorRate=errorRate, load=load)
    queues_names = read_queues_indicators(ns3_path, rate, results_folder, differentiationDelay=differentiationDelay, errorRate=errorRate, load=load)
    flows_name = ['R0H0R2H3']
    queues_names = ["T0A0", "A0T2", "T2H3"]
    flows_name.sort()
    queues_names.sort()

    rounds_results = prepare_results(flows_name, queues_names, num_of_paths)
    merged_results = prepare_results(flows_name, queues_names, num_of_paths)
    batch_size = 1
    for i in range(int(experiments_end / batch_size) + 1):
        ths = []
        exps = []
        return_dict = multiprocessing.Manager().dict()
        for experiment in range(batch_size * i, min(experiments_end, batch_size * (i + 1))):
            if differentiationDelay is not None and errorRate is not None:
                if len(os.listdir('{}/scratch/{}/{}/{}/D_{}/f_{}/{}'.format(__ns3_path, results_folder, rate, load, differentiationDelay, errorRate, experiment))) == 0:
                    print(experiment)
                    continue
            else:
                if len(os.listdir('{}/scratch/{}/{}/{}/{}'.format(__ns3_path, results_folder, rate, load, experiment))) == 0:
                    print(experiment)
                    continue
            print("Analyzing experiment: ", experiment)
            exps.append(experiment)
            ths.append(multiprocessing.Process(target=analyze_single_experiment, args=(return_dict, rate, queues_names, confidenceValue, steadyStart, steadyEnd, rounds_results, results_folder, config, experiment, ns3_path, differentiationDelay, errorRate, load, flows_name, queues_names)))
        
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        merge_results(return_dict, merged_results, flows_name, queues_names, num_of_paths, exps)
        print("{} joind".format(i))
    # merged_results['AverageWorkLoad'] = sum(merged_results['AverageWorkLoad']) / merged_results['experiments']
    # if errorRate is not None:
    #     os.system('mkdir -p ../Results/results_{}/{}/{}/D_{}/f_{}/'.format(dir, rate, load, differentiationDelay, errorRate))
    #     with open('../Results/results_{}/{}/{}/D_{}/f_{}/non_maxE_delay_window_devision_sampling_99percentNonEmpty.0_{}_{}_to_{}.json'.format(dir, rate, load, differentiationDelay, errorRate, experiments_end, steadyStart, steadyEnd), 'w') as f:
    #         js.dump(merged_results, f, indent=4)
    # else:
    #     # with open('../Results/results_{}/{}/{}/non_maxE_delay_window_devision_sampling_99percentNonEmpty.0_{}_{}_to_{}.json'.format(dir, rate, load, experiments_end, steadyStart, steadyEnd), 'w') as f:
    #     with open('../Results/results_{}/{}/{}/temp.0_{}_{}_to_{}.json'.format(dir, rate, load, experiments_end, steadyStart, steadyEnd), 'w') as f:
    #         js.dump(merged_results, f, indent=4)

# main function
def __main__():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dir",
                    required=True,
                    dest="dir",
                    help="The directory of the results",
                    default="")
    parser.add_argument("--emd-vs-flows",
                    action="store_true",
                    dest="emd_vs_flows",
                    help="Run the EMD-vs-number-of-flows analysis (run_emd_vs_flows_experiment) "
                         "for each traffic/rate/load/experiment instead of the standard "
                         "analyze_all_experiments sweep. Only used in the 'forward' branch.")
    parser.add_argument("--flow-name", dest="flow_name", default="R0H0R2H3",
                    help="TCP flow to analyze when --emd-vs-flows is set")
    parser.add_argument("--path", dest="path", type=int, default=0,
                    help="Path index to analyze when --emd-vs-flows is set")
    parser.add_argument("--num-runs", dest="num_runs", type=int, default=10,
                    help="Number of repeated Poisson-sampling runs when --emd-vs-flows is set")
    parser.add_argument("--num-poisson-observations", dest="num_poisson_observations", type=int, default=9000,
                    help="Poisson observations per run when --emd-vs-flows is set")
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=10,
                    help="Parallel workers across runs when --emd-vs-flows is set")
    parser.add_argument("--delay-cdf-sample-interval-ns", dest="delay_cdf_sample_interval_ns", type=float, default=90,
                    help="Ground-truth CDF Poisson sampling interval in ns when --emd-vs-flows is set "
                         "(smaller = more ground-truth samples = more accurate but slower; "
                         "90ns gives ~1M samples over a 90ms steady period)")
    parser.add_argument("--flow-count-step", dest="flow_count_step", type=int, default=1,
                    help="Evaluate the flow-count sweep every N flows instead of every flow when "
                         "--emd-vs-flows is set (e.g. 3 evaluates k=1,4,7,... instead of every k) -- "
                         "cuts the dominant per-run cost (find_samples_path, called once per k per "
                         "run) roughly by this factor, at the cost of a coarser sweep")
    parser.add_argument("--aggregate-emd-vs-flows",
                    action="store_true",
                    dest="aggregate_emd_vs_flows",
                    help="Aggregate every experiment's already-computed run_emd_vs_flows_experiment "
                         "results (found under scratch/Results_<dir>/<traffic>/<rate>/<load>/<experiment>/) "
                         "instead of computing anything new. For each traffic/rate/load this saves the "
                         "combined plots/pickle/text under scratch/ECNMC/Results/results_<dir>/<traffic>/"
                         "<rate>/<load>/, and for each rate it additionally saves cross-traffic comparison "
                         "plots (EMD vs load, one boxplot family per traffic, one plot per flow-count k) "
                         "under scratch/ECNMC/Results/results_<dir>/emd_vs_load_by_traffic_<methods><gt>/<rate>/ "
                         "(tagged with the subsampling method(s) and ground-truth method used). Takes "
                         "precedence over --emd-vs-flows. Only used in the 'forward' branch.")
    parser.add_argument("--subsampling-method", dest="subsampling_methods", nargs='+',
                    default=["find_samples_path"], metavar="METHOD",
                    choices=list(POISSON_SUBSAMPLING_METHODS.keys()),
                    help="Which Poisson-adaptive subsampling algorithm(s) to evaluate as 'sampled' "
                         "comparison series, with --emd-vs-flows or --aggregate-emd-vs-flows. Pass several "
                         "(e.g. --subsampling-method find_samples_path find_samples_path_intensity) to "
                         "evaluate them all in the same run -- on identical packets, flows, ground truth and "
                         "per-run switch statistics, so the algorithms are compared with no extra run-to-run "
                         "noise between them, and each becomes its own plotted family. Always part of the "
                         "output filename tag (several joined with '+') so different configurations' outputs "
                         "for the same traffic/rate/load/experiment don't collide.")
    parser.add_argument("--groundtruth-method", dest="groundtruth_methods", nargs='+',
                    default=["simultaneous"], metavar="METHOD",
                    choices=list(GROUNDTRUTH_METHODS.keys()),
                    help="Which ground-truth path-delay CDF the EMDs are measured against, with "
                         "--emd-vs-flows or --aggregate-emd-vs-flows: 'simultaneous' observes every queue on "
                         "the path at the same instant, 'path_observation' releases a probe that waits out "
                         "each queue's delay before observing the next one (what an actual packet "
                         "experiences, see BiasCalculation_DC.py's path_observation sampling). Pass both to "
                         "run the whole sweep once per ground truth. Anything other than 'simultaneous' adds "
                         "its own tag to the output filenames, so existing 'simultaneous' results keep the "
                         "names they already have.")

    parser.add_argument("--all-flows-only",
                    action="store_true",
                    dest="all_flows_only",
                    help="Skip the flow-count sweep and evaluate only all flows on the path -- i.e. "
                         "every received e2e packet -- with --emd-vs-flows or "
                         "--aggregate-emd-vs-flows. This is the headline configuration (the same one "
                         "the k='max' plots single out) and by far the cheapest to run, since the "
                         "per-run subsampling searches drop from one per flow count to one. Adds its "
                         "own filename tag, so it never overwrites a swept run's outputs; the "
                         "cross-traffic plots then cover only k='max', a fixed-k plot being "
                         "meaningless when each combination contributes a single, possibly different, "
                         "flow total.")
    parser.add_argument("--no-chi-squared-test",
                    action="store_false",
                    dest="run_chi_squared_test",
                    help="Skip the multi-lag chi-squared independence test when checking whether the "
                         "non-Poissonized families' (all packets, rate-matched uniform) sampling "
                         "instants look Poisson. Anderson-Darling is free; chi-squared costs roughly a "
                         "second per family per flow count per run, so it dominates the runtime of "
                         "--emd-vs-flows. With it off, only the Anderson-Darling split plots are "
                         "produced and the 'ad_chi' ones are skipped.")
    parser.add_argument("--delay-percentile", dest="delay_percentiles", nargs='+', type=float,
                    default=list(DEFAULT_DELAY_PERCENTILES), metavar="Q",
                    help="Which delay percentiles to report the tail-shape error at, with "
                         "--emd-vs-flows: the signed 'ground-truth p_q minus family p_q', absolute (ns) "
                         "and relative to the ground truth's own p_q, for all packets, every "
                         "Poisson-adaptive method and every rate-matched uniform baseline. The EMD is a "
                         "single number for the whole distribution and can hide a misplaced tail, which "
                         "is the part delay SLOs are written against. Default p90 and p99.")

    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read('../Results/results_{}/Parameters.config'.format(args.dir))
    steadyStart = convert_to_float(config.get('Settings', 'steadyStart')) * 1e9
    # steadyStart = 0.08 * 1e9
    steadyEnd = convert_to_float(config.get('Settings', 'steadyEnd')) * 1e9
    # steadyEnd = 0.015 * 1e9
    experiments = int(config.get('Settings', 'experiments'))
    experiments = 30
    experiments = 1
    serviceRateScales = [float(x) for x in config.get('Settings', 'serviceRateScales').split(',')]
    # serviceRateScales = [0.5]
    loads = [float(x) for x in config.get('Settings', 'load').split(',')]
    loads = [0.5, 0.6, 0.7, 0.8, 0.95]
    loads = [0.0, 0.125]
    traffics = config.get('Settings', 'traffic').split(',')
    traffics = ["Google_AllRPC", "Google_SearchRPC", "Facebook_HadoopDist_All"]
    traffics = ["Fabricated_Heavy_Middle"]
    errorRates = [float(x) for x in config.get('Settings', 'errorRate').split(',')]
    # errorRates = [0.1, 0.3, 0.5, 0.7, 0.9]
    # errorRates = [0.1]
    differentiationDelays = [float(x) for x in config.get('Settings', 'differentiationDelay').split(',')]
    # differentiationDelays = [5.0]
    # devide steady period into smaller parts
    numOfSteadyParts = 1
    for start in range(int(steadyStart), int(steadyEnd), int((steadyEnd - steadyStart) / numOfSteadyParts)):
        print("Steady period: {} to {}".format(start, start + int((steadyEnd - steadyStart) / numOfSteadyParts)))
        if "forward" in args.dir:
            if args.aggregate_emd_vs_flows:
                for rate in serviceRateScales:
                    for groundtruth_method in args.groundtruth_methods:
                        aggregate_emd_vs_flows_across_traffics_and_loads(
                            __ns3_path, args.dir, traffics, rate, loads,
                            start, start + int((steadyEnd - steadyStart) / numOfSteadyParts),
                            flow_name=args.flow_name, path=args.path,
                            subsampling_methods=args.subsampling_methods,
                            groundtruth_method=groundtruth_method,
                            all_flows_only=args.all_flows_only,
                        )
                continue
            for traffic in traffics:
                for rate in serviceRateScales:
                    for load in loads:
                        if args.emd_vs_flows:
                            print("\nRunning EMD-vs-flows analysis for traffic {} rate: {} load: {}".format(traffic, rate, load))
                            for groundtruth_method in args.groundtruth_methods:
                                for experiment in range(experiments):
                                    print("Running EMD-vs-flows analysis for traffic {} rate: {} load: {} experiment {} "
                                          "(subsampling: {}, ground truth: {})".format(
                                              traffic, rate, load, experiment,
                                              ", ".join(args.subsampling_methods), groundtruth_method))
                                    run_emd_vs_flows_experiment(
                                        rate, start, start + int((steadyEnd - steadyStart) / numOfSteadyParts), confidenceValue,
                                        'Results_' + args.dir + "/" + traffic, config, experiment=experiment, ns3_path=__ns3_path, load=load,
                                        flow_name=args.flow_name, path=args.path, num_runs=args.num_runs,
                                        num_poisson_observations=args.num_poisson_observations, num_workers=args.num_workers,
                                        delay_cdf_sample_interval_ns=args.delay_cdf_sample_interval_ns,
                                        flow_count_step=args.flow_count_step,
                                        subsampling_methods=args.subsampling_methods,
                                        groundtruth_method=groundtruth_method,
                                        all_flows_only=args.all_flows_only,
                                        delay_percentiles=args.delay_percentiles,
                                        run_chi_squared_test=args.run_chi_squared_test,
                                    )
                            print("Traffic {} Rate {} {} {} EMD-vs-flows done".format(traffic, rate, load, experiments))
                        else:
                            print("\nAnalyzing experiments for traffic {} rate: {} load: {}".format(traffic, rate, load))
                            analyze_all_experiments(rate, start, start + int((steadyEnd - steadyStart) / numOfSteadyParts), confidenceValue, args.dir + "/" + traffic, config, experiments_end=experiments, ns3_path=__ns3_path, load=load)
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
                                analyze_all_experiments(rate, start, start + int((steadyEnd - steadyStart) / numOfSteadyParts), confidenceValue, args.dir + "/" + traffic, config, experiments_end=experiments, ns3_path=__ns3_path, differentiationDelay=differentiationDelay, errorRate=errorRate, load=load)
                                print("Rate {} load {} with {} and {} done".format(rate, load, differentiationDelay, errorRate))
                        print("Traffic {} Rate {} load {} done".format(traffic, rate, load))
                    print("Rate {} done".format(rate))
                print("Traffic {} done".format(traffic))

def run_emd_vs_flows_experiment(rate, steadyStart, steadyEnd, confidenceValue, results_folder, config, experiment=0, ns3_path=__ns3_path, load=None, flow_name='R0H0R2H3', queue_names=None, path=0, delay_cdf_sample_interval_ns=90, num_runs=100, num_poisson_observations=9000, pass_threshold=0.9, num_workers=1, emd_y_max=None, mean_diff_y_limit=None, flow_count_step=1, all_flows_only=False, subsampling_methods='find_samples_path', groundtruth_method='simultaneous', delay_percentiles=DEFAULT_DELAY_PERCENTILES, run_chi_squared_test=True):
    """Reconstruct the network queuing delay CDF once (ground truth), then repeat `num_runs` times: draw
    `num_poisson_observations` fresh Poisson-process observation instants at the path's switches, derive the
    per-segment aggregated delay statistics from them, and grow the set of considered TCP flows of `flow_name`
    one at a time, comparing the EMD of the all-packet CDF against one Poisson-adaptive subsample per entry in
    `subsampling_methods`; for each of those methods, a systematic uniform subsample drawing exactly as
    many of the considered flows' packets as that method retained (the rate-matched baseline, see
    Utils.matched_uniform_target_count); and an ideal Poisson probe at the minimum required sample size
    and at each method's own sample size (Utils.construct_oracle_poisson_delays) -- the ceiling any
    Poissonization scheme is trying to reach, since its instants are Poisson by construction and carry
    no selection bias. Saves, under
    `<results_folder>/<rate>/<load>/<experiment>/<steady_tag>/<config_tag>/` (`<steady_tag>` =
    Utils.steady_window_tag(steadyStart, steadyEnd), `<config_tag>` =
    emd_vs_flows_file_tag(subsampling_methods, groundtruth_method, all_flows_only) -- both
    folder levels, not filename infixes, so filenames stay short and re-analyzing the same raw
    experiment over a different steady window or config never collides with or overwrites an
    earlier one):
      - `<flow_name>_path_<path>_emd_vs_num_flows_boxplot.png`: EMD distribution
        across runs, one boxplot family per subsampling method, plus a `..._normalized.png`
        twin of the same plot in units of the mean ground-truth delay.
      - `<flow_name>_path_<path>_delay_mean_diff_boxplot.png`: the signed
        switch-vs-packet mean delay difference underlying the consistency check, same per-method breakdown.
      - `<flow_name>_path_<path>_<quantity>_<test>_split.png`, one per quantity
        (EMD, normalized EMD, mean difference) x test ('ad', 'ad_chi'): the non-Poissonized
        families (all packets, each rate-matched uniform) with their runs split by whether
        that run's own sampling instants passed the Poisson-ness test. `run_chi_squared_test`
        turns off the chi-squared half, which dominates the cost (~1s per family per flow
        count per run); the 'ad_chi' plots are then skipped.
      - `<flow_name>_path_<path>_p<q>_diff_boxplot.png` and `..._p<q>_reldiff_boxplot.png`,
        one pair per percentile in `delay_percentiles`: the signed tail-shape error
        `ground-truth p<q> - family p<q>`, absolute (ns) and relative to the ground truth's own
        p<q>, for all packets, every Poisson-adaptive method and every rate-matched uniform.
      - `<flow_name>_path_<path>_delay_cdf_one_run.png`: the ground-truth delay CDF
        against every method's actual delay CDF from one concrete Poisson realization (not an EMD summary
        across runs).
      - `<flow_name>_path_<path>_emd_vs_num_flows_results.pkl`: the full underlying
        results dict (raw and normalized EMDs both).
      - `<flow_name>_path_<path>_emd_vs_num_flows_results.txt`: a human-readable
        per-flow-count summary.

    `subsampling_methods` is one Utils.POISSON_SUBSAMPLING_METHODS key or a list of them -- pass several
    (e.g. ['find_samples_path', 'find_samples_path_intensity']) to evaluate the algorithms within a single
    run, on identical packets, flows, ground truth and per-run switch statistics, so the comparison between
    them carries no extra run-to-run noise. `groundtruth_method` (one of Utils.GROUNDTRUTH_METHODS) selects
    what the EMDs are measured against: 'simultaneous' (every queue observed at one instant) or
    'path_observation' (a probe that waits out each queue's delay before observing the next, i.e. what a real
    packet experiences). Both choices go into `<config_tag>`, so different configurations for the same
    traffic/rate/load/experiment/steady-window never collide.

    Set `all_flows_only` to skip the flow-count sweep and evaluate only k = all flows on the path
    (every received e2e packet) -- the headline configuration and much the cheapest to run, since
    the per-run subsampling searches drop from one per k to one. It adds its own `<config_tag>`
    component, so such a run never overwrites a swept run's outputs for the same combination.

    Both boxplots color each flow-count's box/point by whether at least `pass_threshold` of the runs'
    consistency check passed there. Returns the underlying per-flow-count, per-run results.
    """
    if queue_names is None:
        queue_names = ["T0A0", "A0T2", "T2H3"]
    subsampling_methods = normalize_subsampling_methods(subsampling_methods)
    hostToTorLinkRate = convert_to_float(config.get('Settings', 'hostToTorLinkRate')) * 1e-3
    torToAggLinkRate = convert_to_float(config.get('Settings', 'torToAggLinkRate')) * rate * 1e-3
    linkDelay = convert_to_float(config.get('Settings', 'hostToTorLinkDelay')) * 1e6
    linkRates = [hostToTorLinkRate, torToAggLinkRate, torToAggLinkRate, hostToTorLinkRate]
    linkDelays = [linkDelay, linkDelay, linkDelay, linkDelay]

    results = compute_emd_vs_num_tcp_flows_multi_run(
        ns3_path, results_folder, rate, load, experiment, flow_name, queue_names, linkDelays, linkRates,
        steadyStart, steadyEnd, confidenceValue, DelayConsistencyGaurantee,
        num_runs=num_runs, num_poisson_observations=num_poisson_observations,
        min_sample_size=min_sample_size, delay_cdf_sample_interval_ns=delay_cdf_sample_interval_ns, path=path,
        num_workers=num_workers, flow_count_step=flow_count_step, all_flows_only=all_flows_only,
        subsampling_methods=subsampling_methods, groundtruth_method=groundtruth_method,
        delay_percentiles=delay_percentiles, run_chi_squared_test=run_chi_squared_test,
    )

    # Steady window and (subsampling/GT/all-flows) config each get their own folder level
    # instead of a filename infix -- keeps filenames short and lets the same raw experiment be
    # re-analyzed over a different window, or with a different config, without collision.
    output_dir = '{}/scratch/{}/{}/{}/{}/{}/{}/'.format(
        ns3_path, results_folder, rate, load, experiment,
        steady_window_tag(steadyStart, steadyEnd),
        emd_vs_flows_file_tag(subsampling_methods, groundtruth_method, all_flows_only))
    os.makedirs(output_dir, exist_ok=True)
    file_prefix = '{}{}_path_{}'.format(output_dir, flow_name, path)
    run_desc = '{} runs x {} Poisson obs'.format(num_runs, num_poisson_observations)
    gt_desc = groundtruth_method_label(groundtruth_method)

    plot_emd_vs_num_flows_boxplot(
        results, file_prefix + '_emd_vs_num_flows_boxplot.png', pass_threshold=pass_threshold,
        title='EMD vs number of TCP flows ({}): {}, path {}\n{}'.format(run_desc, flow_name, path, gt_desc),
        y_max=emd_y_max,
    )
    # The same plot in normalized units (EMD / mean ground-truth delay): raw EMD grows with
    # the delay level the offered load itself drives, so only this one is comparable across loads.
    plot_emd_vs_num_flows_boxplot(
        results, file_prefix + '_emd_vs_num_flows_boxplot_normalized.png', pass_threshold=pass_threshold,
        title='Normalized EMD vs number of TCP flows ({}): {}, path {}\n{}'.format(run_desc, flow_name, path, gt_desc),
        normalized=True,
    )
    plot_mean_diff_vs_num_flows(
        results, file_prefix + '_delay_mean_diff_boxplot.png', pass_threshold=pass_threshold,
        title='Switch vs. packet mean delay difference ({}): {}, path {}'.format(run_desc, flow_name, path),
        y_limit=mean_diff_y_limit,
    )
    for q in results['delay_percentiles']:
        plot_percentile_diff_vs_num_flows(
            results, q, '{}_p{}_diff_boxplot.png'.format(file_prefix, q), relative=False,
            title='p{} error (ground truth - family) vs number of TCP flows ({}): {}, path {}\n{}'.format(
                q, run_desc, flow_name, path, gt_desc),
        )
        plot_percentile_diff_vs_num_flows(
            results, q, '{}_p{}_reldiff_boxplot.png'.format(file_prefix, q), relative=True,
            title='Relative p{} error (ground truth - family) vs number of TCP flows ({}): {}, path {}\n{}'.format(
                q, run_desc, flow_name, path, gt_desc),
        )
    # The two families that never had to pass a Poisson-ness test -- all packets and each
    # rate-matched uniform subset -- with their runs split by whether their own sampling
    # instants passed. One plot per (test, quantity): AD alone, and AD + chi-squared.
    for test_name in POISSON_TEST_NAMES:
        for quantity, quantity_desc in (('emd', 'EMD'), ('emd_normalized', 'Normalized EMD'),
                                         ('mean_diff', 'Switch vs. packet mean delay difference')):
            plot_poisson_test_split_vs_num_flows(
                results, '{}_{}_{}_split.png'.format(file_prefix, quantity, test_name),
                test_name=test_name, quantity=quantity,
                title='{} split by Poisson-ness of the sampling instants: {}, path {}\n{}\n{} | {}'.format(
                    quantity_desc, flow_name, path, poisson_test_label(test_name), run_desc, gt_desc),
            )
    plot_one_run_delay_cdfs(
        results, file_prefix + '_delay_cdf_one_run.png',
        title='Delay CDF comparison, one Poisson realization: {}, path {}\n{}'.format(flow_name, path, gt_desc),
    )
    with open(file_prefix + '_emd_vs_num_flows_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    save_emd_vs_flows_results_text(results, file_prefix + '_emd_vs_num_flows_results.txt')

    return results


def aggregate_emd_vs_flows_across_experiments(ns3_path, dir_name, traffic, rate, load, steadyStart, steadyEnd,
                                               flow_name='R0H0R2H3',
                                               path=0, pass_threshold=0.9, emd_y_max=None, mean_diff_y_limit=None,
                                               subsampling_methods='find_samples_path',
                                               groundtruth_method='simultaneous',
                                               all_flows_only=False):
    """Load every experiment's run_emd_vs_flows_experiment output for the same
    traffic/rate/load/steady-window (each under
    scratch/Results_<dir_name>/<traffic>/<rate>/<load>/<experiment>/<steady_tag>/<config_tag>/,
    discovered by scanning <load>/ for experiment subfolders), combine them via
    aggregate_emd_vs_flows_results, and save the aggregated plots/pickle/text under
    scratch/ECNMC/Results/results_<dir_name>/<traffic>/<rate>/<load>/<steady_tag>/<config_tag>/
    -- `<steady_tag>` = Utils.steady_window_tag(steadyStart, steadyEnd), `<config_tag>` =
    emd_vs_flows_file_tag(subsampling_methods, groundtruth_method, all_flows_only), both as
    their own folder levels (not filename infixes) so filenames stay short and a different
    steady window or configuration for the same traffic/rate/load never collides.

    `steadyStart`/`steadyEnd` (ns) and `subsampling_methods`/`groundtruth_method`/
    `all_flows_only` together select which run_emd_vs_flows_experiment output to look for and
    must match the values that experiment run was computed with.

    Falls back to the pre-2026-09-09 flat path (`<experiment>/<flow_name>_path_<path>_<config_tag>_...`,
    no steady-window folder) for any experiment not found at the current nested path, so results
    computed before that restructuring are still picked up without having to re-run
    run_emd_vs_flows_experiment on them; prints a note when this happens.

    Returns the aggregated results dict, or None if no experiment's results pickle was found
    (e.g. run_emd_vs_flows_experiment hasn't been run yet for this traffic/rate/load/window/configuration).
    """
    steady_tag = steady_window_tag(steadyStart, steadyEnd)
    config_tag = emd_vs_flows_file_tag(subsampling_methods, groundtruth_method, all_flows_only)
    per_experiment_base = '{}/scratch/Results_{}/{}/{}/{}'.format(ns3_path, dir_name, traffic, rate, load)
    file_suffix = '{}/{}/{}_path_{}_emd_vs_num_flows_results.pkl'.format(steady_tag, config_tag, flow_name, path)
    # Pre-2026-09-09 layout: no <steady_tag>/<config_tag>/ folder nesting, config_tag was a
    # filename infix instead. Fall back to it so results computed before that restructuring
    # are still discoverable without having to re-run run_emd_vs_flows_experiment on them --
    # there is no ambiguity in doing so, since a legacy run predates steady-window tagging
    # entirely (it only ever wrote one, whatever steadyStart/steadyEnd its config used).
    legacy_file_suffix = '{}_path_{}_{}_emd_vs_num_flows_results.pkl'.format(flow_name, path, config_tag)

    results_list = []
    legacy_hits = 0
    if os.path.isdir(per_experiment_base):
        for entry in sorted(os.listdir(per_experiment_base)):
            pkl_path = '{}/{}/{}'.format(per_experiment_base, entry, file_suffix)
            if not os.path.isfile(pkl_path):
                legacy_path = '{}/{}/{}'.format(per_experiment_base, entry, legacy_file_suffix)
                if os.path.isfile(legacy_path):
                    pkl_path = legacy_path
                    legacy_hits += 1
                else:
                    continue
            with open(pkl_path, 'rb') as f:
                r = pickle.load(f)
            r['experiment'] = int(entry) if entry.isdigit() else entry
            results_list.append(r)
    if legacy_hits:
        print("Note: {} of {} experiment result(s) for {} rate={} load={} were found at the "
              "pre-2026-09-09 flat path (no steady-window folder) rather than under {} -- "
              "re-run run_emd_vs_flows_experiment for this combination to move them to the "
              "current layout.".format(legacy_hits, len(results_list), traffic, rate, load, file_suffix))

    if not results_list:
        print("No experiment results found for {} rate={} load={} window={} tag={} under {}".format(
            traffic, rate, load, steady_tag, config_tag, per_experiment_base))
        return None

    aggregated = aggregate_emd_vs_flows_results(results_list)
    print("Aggregating {} rate={} load={} window={} tag={}: {} experiment(s) {}".format(
        traffic, rate, load, steady_tag, config_tag, aggregated['num_experiments'], aggregated['experiments']))

    output_dir = '{}/scratch/ECNMC/Results/results_{}/{}/{}/{}/{}/{}/'.format(
        ns3_path, dir_name, traffic, rate, load, steady_tag, config_tag)
    os.makedirs(output_dir, exist_ok=True)
    file_prefix = '{}{}_path_{}'.format(output_dir, flow_name, path)
    run_desc = '{} experiment(s) x {} Poisson obs'.format(aggregated['num_experiments'], aggregated['num_poisson_observations'])
    gt_desc = groundtruth_method_label(aggregated.get('groundtruth_method', groundtruth_method))

    plot_emd_vs_num_flows_boxplot(
        aggregated, file_prefix + '_emd_vs_num_flows_boxplot.png', pass_threshold=pass_threshold,
        title='EMD vs number of TCP flows, aggregated ({}): {}, path {}\n{}'.format(run_desc, flow_name, path, gt_desc),
        y_max=emd_y_max,
    )
    plot_emd_vs_num_flows_boxplot(
        aggregated, file_prefix + '_emd_vs_num_flows_boxplot_normalized.png', pass_threshold=pass_threshold,
        title='Normalized EMD vs number of TCP flows, aggregated ({}): {}, path {}\n{}'.format(
            run_desc, flow_name, path, gt_desc),
        normalized=True,
    )
    plot_mean_diff_vs_num_flows(
        aggregated, file_prefix + '_delay_mean_diff_boxplot.png', pass_threshold=pass_threshold,
        title='Switch vs. packet mean delay difference, aggregated ({}): {}, path {}'.format(run_desc, flow_name, path),
        y_limit=mean_diff_y_limit,
    )
    for q in aggregated['delay_percentiles']:
        plot_percentile_diff_vs_num_flows(
            aggregated, q, '{}_p{}_diff_boxplot.png'.format(file_prefix, q), relative=False,
            title='p{} error (ground truth - family), aggregated ({}): {}, path {}\n{}'.format(
                q, run_desc, flow_name, path, gt_desc),
        )
        plot_percentile_diff_vs_num_flows(
            aggregated, q, '{}_p{}_reldiff_boxplot.png'.format(file_prefix, q), relative=True,
            title='Relative p{} error (ground truth - family), aggregated ({}): {}, path {}\n{}'.format(
                q, run_desc, flow_name, path, gt_desc),
        )
    # The two families that never had to pass a Poisson-ness test -- all packets and each
    # rate-matched uniform subset -- with their runs split by whether their own sampling
    # instants passed. One plot per (test, quantity): AD alone, and AD + chi-squared.
    for test_name in POISSON_TEST_NAMES:
        for quantity, quantity_desc in (('emd', 'EMD'), ('emd_normalized', 'Normalized EMD'),
                                         ('mean_diff', 'Switch vs. packet mean delay difference')):
            plot_poisson_test_split_vs_num_flows(
                aggregated, '{}_{}_{}_split.png'.format(file_prefix, quantity, test_name),
                test_name=test_name, quantity=quantity,
                title='{} split by Poisson-ness of the sampling instants: {}, path {}\n{}\n{} | {}'.format(
                    quantity_desc, flow_name, path, poisson_test_label(test_name), run_desc, gt_desc),
            )
    plot_one_run_delay_cdfs(
        aggregated, file_prefix + '_delay_cdf_one_run.png',
        title='Delay CDF comparison, one Poisson realization (experiment {}): {}, path {}\n{}'.format(
            aggregated['experiments'][0], flow_name, path, gt_desc),
    )
    with open(file_prefix + '_emd_vs_num_flows_results.pkl', 'wb') as f:
        pickle.dump(aggregated, f)
    save_emd_vs_flows_results_text(aggregated, file_prefix + '_emd_vs_num_flows_results.txt')

    return aggregated


def aggregate_emd_vs_flows_across_traffics_and_loads(ns3_path, dir_name, traffics, rate, loads,
                                                       steadyStart, steadyEnd,
                                                       flow_name='R0H0R2H3', path=0, pass_threshold=0.9,
                                                       subsampling_methods='find_samples_path',
                                                       groundtruth_method='simultaneous',
                                                       all_flows_only=False):
    """For a fixed `rate`, aggregate every traffic x load combination (each first
    aggregated across its own experiments via aggregate_emd_vs_flows_across_experiments,
    which also writes that combination's own per-traffic/load plots as a side effect) into
    cross-traffic comparison plots: for every flow-count k seen in any combination, one plot
    of EMD vs. load with all-packets and every Poisson-adaptive subsampling method together
    -- as one boxplot cluster per traffic per load (see plot_emd_vs_load_by_traffic). For
    every Poisson-adaptive method, a second set of plots (same k values) puts that method,
    its own rate-matched uniform baseline, and the ideal Poisson probe at the same sample
    count together (poisson_vs_uniform_vs_ideal_load_plot_series) -- so the group reads as a
    verdict on the selection rule (vs. uniform) and on how much of the remaining error is
    finite-sample noise vs. selection bias (vs. the ideal probe). A third set pairs each
    method against its ideal probe alone plus the minimum-required-sample-size probe
    (sampled_vs_oracle_load_plot_series), and a fourth compares all-packets against the ideal
    probe(s) directly (all_packets_vs_oracle_load_plot_series) -- the ceiling any subsampling
    scheme could reach, independent of a particular sampler's own imperfections. Every one of
    these plots is written twice: in raw nanoseconds and, as a `..._normalized.png` twin, in
    units of the mean ground-truth delay -- the latter being the version actually comparable
    across the loads on the x-axis. Also saves, for each of these plot kinds, one additional
    plot using each combination's own maximum flow count instead of a fixed k (k='max' in
    plot_emd_vs_load_by_traffic), to compare "all the flows we have" per traffic/load even
    though the exact max count can differ across combinations.

    Also saves, for every percentile the results carry (delay_percentiles, default p90/p99),
    the signed tail-shape error `ground-truth p_q - family p_q` vs. load -- absolute (ns) and
    relative to the ground truth's own p_q -- for each of the same comparison groupings. These
    are written only at k='max' (all available flows), since that is the headline flow count
    and emitting them per k as well would multiply the plot count several-fold for little
    extra insight.

    Also saves, per Poisson-adaptive method and per k (plus one all-flows 'max' variant), a
    plot of that method's consistency-check pass rate itself vs. load, one line per traffic
    (see plot_pass_rate_vs_load_by_traffic) -- unlike the EMD plots above, this is the
    success rate, not the EMD distribution.

    Saved under
    scratch/ECNMC/Results/results_<dir_name>/emd_vs_load_by_traffic/<steady_tag>/<config_tag>/<rate>/,
    where <steady_tag> = Utils.steady_window_tag(steadyStart, steadyEnd) and <config_tag> =
    emd_vs_flows_file_tag(subsampling_methods, groundtruth_method, all_flows_only) -- each its
    own folder level -- so different steady windows or subsampling/GT configurations for the
    same dir_name land in separate folders, and within <rate>/ each of the four comparison
    kinds above gets its own subfolder (all_vs_poisson/, poisson_vs_uniform_vs_ideal/<method>/,
    poisson_vs_ideal/<method>/, all_vs_ideal/, pass_rate/<method>/) so filenames only need
    `<flow_name>_path_<path>_...` plus the k/normalized/percentile suffix, not the whole
    comparison description.

    Returns the {(traffic, load): aggregated_results} dict used to build the plots, or None
    if no traffic/load combination had any experiment results to aggregate.
    """
    subsampling_methods = normalize_subsampling_methods(subsampling_methods)
    results_by_traffic_load = {}
    for traffic in traffics:
        for load in loads:
            aggregated = aggregate_emd_vs_flows_across_experiments(
                ns3_path, dir_name, traffic, rate, load, steadyStart, steadyEnd,
                flow_name=flow_name, path=path,
                pass_threshold=pass_threshold, subsampling_methods=subsampling_methods,
                groundtruth_method=groundtruth_method, all_flows_only=all_flows_only,
            )
            if aggregated is not None:
                results_by_traffic_load[(traffic, load)] = aggregated

    if not results_by_traffic_load:
        print("No aggregated results available for rate={} to build cross-traffic/load plots".format(rate))
        return None

    all_k = sorted(set().union(*(set(r['num_flows']) for r in results_by_traffic_load.values())))

    # Steady window and (subsampling/GT/all-flows) config each get their own folder level, same
    # convention as aggregate_emd_vs_flows_across_experiments, so different windows/configs for
    # the same dir_name never collide and filenames don't need to spell either one out.
    steady_tag = steady_window_tag(steadyStart, steadyEnd)
    config_tag = emd_vs_flows_file_tag(subsampling_methods, groundtruth_method, all_flows_only)
    rate_dir = '{}/scratch/ECNMC/Results/results_{}/emd_vs_load_by_traffic/{}/{}/{}/'.format(
        ns3_path, dir_name, steady_tag, config_tag, rate)
    gt_desc = groundtruth_method_label(groundtruth_method)

    # Each comparison kind gets its own subfolder under rate_dir (see docstring), so a
    # filename only ever needs '<flow_name>_path_<path>' plus a k/normalized/percentile
    # suffix -- not the whole comparison description as well.
    plot_kinds = [(all_packets_vs_sampled_load_plot_series(subsampling_methods), 'all_vs_poisson',
                    'all packets vs. Poisson-adaptive subsample(s)')]
    # One three-series plot per method: that method, its rate-matched uniform baseline, and
    # the ideal Poisson probe, all at ~the same sample count -- so the plot answers both "does
    # the selection rule beat blind uniform sampling" and "how much of what's left is
    # finite-sample noise vs. selection bias" together.
    for method in subsampling_methods:
        plot_kinds.append((poisson_vs_uniform_vs_ideal_load_plot_series(method),
                            'poisson_vs_uniform_vs_ideal/{}'.format(method),
                            '{} vs. its rate-matched uniform baseline vs. the ideal Poisson probe '
                            '(all ~equal sample size)'.format(method)))
    # And each method against the ideal Poisson probe at its own sample count alone (plus the
    # probe at the minimum required sample size): the gap between the method and its own ideal
    # probe is the part of its error that having few samples does not explain.
    for method in subsampling_methods:
        plot_kinds.append((sampled_vs_oracle_load_plot_series(method),
                            'poisson_vs_ideal/{}'.format(method),
                            '{} vs. the ideal Poisson probe at the same sample budget'.format(method)))
    # All packets vs. the ideal Poisson probe: the theoretical ceiling a real sampler could
    # reach, independent of any particular sampler's own selection-rule imperfections -- reads
    # alongside the all-packets-vs-Poisson-adaptive plot above to separate "what subsampling
    # costs in principle" from "what this particular sampler costs beyond that".
    plot_kinds.append((all_packets_vs_oracle_load_plot_series(subsampling_methods),
                        'all_vs_ideal',
                        'all packets vs. the ideal Poisson probe(s)'))

    # Raw nanoseconds and the load-comparable normalized twin of every plot below.
    emd_variants = [(False, '', 'EMD'), (True, '_normalized', 'Normalized EMD')]

    percentiles = next(iter(results_by_traffic_load.values())).get('delay_percentiles') or []

    # An all-flows-only run has one k per combination -- and different combinations can have
    # different flow totals -- so a fixed-k plot would show only the combos that happen to
    # match. Only the k='max' plots ("all the flows we have") are meaningful there.
    fixed_k_values = [] if all_flows_only else all_k

    for series_specs, subfolder, kind_desc in plot_kinds:
        kind_dir = '{}{}/'.format(rate_dir, subfolder)
        os.makedirs(kind_dir, exist_ok=True)
        kind_prefix = '{}{}_path_{}'.format(kind_dir, flow_name, path)
        for normalized, norm_suffix, emd_desc in emd_variants:
            for k in fixed_k_values:
                plot_emd_vs_load_by_traffic(
                    results_by_traffic_load, k, '{}_k{}{}.png'.format(kind_prefix, k, norm_suffix),
                    pass_threshold=pass_threshold, series_specs=series_specs, normalized=normalized,
                    title='{} vs load by traffic, {} considered flows: {}, path {}, rate {}\n{}\n{}'.format(
                        emd_desc, k, flow_name, path, rate, kind_desc, gt_desc),
                )
            plot_emd_vs_load_by_traffic(
                results_by_traffic_load, 'max', '{}_kmax{}.png'.format(kind_prefix, norm_suffix),
                pass_threshold=pass_threshold, series_specs=series_specs, normalized=normalized,
                title='{} vs load by traffic, all considered flows: {}, path {}, rate {}\n{}\n{}'.format(
                    emd_desc, flow_name, path, rate, kind_desc, gt_desc),
            )
        # Percentile (tail-shape) error vs load, all available flows only.
        for q in percentiles:
            for kind, kind_suffix, desc in (
                    ('percentile_diff', '_p{}_diff'.format(q), 'p{} error (ns)'.format(q)),
                    ('percentile_reldiff', '_p{}_reldiff'.format(q), 'Relative p{} error'.format(q))):
                plot_emd_vs_load_by_traffic(
                    results_by_traffic_load, 'max', '{}_kmax{}.png'.format(kind_prefix, kind_suffix),
                    pass_threshold=pass_threshold, series_specs=series_specs, metric=(kind, q),
                    title='{} (ground truth - family) vs load by traffic, all considered flows: {}, path {}, rate {}\n{}\n{}'.format(
                        desc, flow_name, path, rate, kind_desc, gt_desc),
                )

    for method in subsampling_methods:
        pass_rate_dir = '{}pass_rate/{}/'.format(rate_dir, method)
        os.makedirs(pass_rate_dir, exist_ok=True)
        pass_rate_prefix = '{}{}_path_{}'.format(pass_rate_dir, flow_name, path)
        for k in fixed_k_values:
            plot_pass_rate_vs_load_by_traffic(
                results_by_traffic_load, k, '{}_k{}.png'.format(pass_rate_prefix, k),
                series_key=('sampled', method), pass_threshold=pass_threshold,
                title='Consistency pass rate vs load by traffic, {} considered flows: {}, path {}, rate {}\n{}\n{}'.format(
                    k, flow_name, path, rate, method, gt_desc),
            )
        plot_pass_rate_vs_load_by_traffic(
            results_by_traffic_load, 'max', '{}_kmax.png'.format(pass_rate_prefix),
            series_key=('sampled', method), pass_threshold=pass_threshold,
            title='Consistency pass rate vs load by traffic, all considered flows: {}, path {}, rate {}\n{}\n{}'.format(
                flow_name, path, rate, method, gt_desc),
        )

    print("Saved {} cross-traffic/load plot kinds x {} EMD variants x {} k values (plus one all-flows plot each), "
          "plus {} percentile-error plots per kind ({} percentile(s) x absolute/relative, all-flows only), "
          "plus {} pass-rate-vs-load plots per method x {} method(s) (plus one all-flows plot each), to {}".format(
        len(plot_kinds), len(emd_variants), len(fixed_k_values), 2 * len(percentiles), len(percentiles),
        len(fixed_k_values), len(subsampling_methods), rate_dir))
    return results_by_traffic_load


if __name__ == "__main__":
    __main__()