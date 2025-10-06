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

timeAvg_methods = ['rightCont_timeAvg', 'leftCont_timeAvg', 'linearInterp_timeAvg', 'poisson_eventAvg', 'eventAvg']
# timeAvg_methods = ['rightCont_timeAvg', 'leftCont_timeAvg', 'linearInterp_timeAvg']
# timeAvg_methods = ['linearInterp_timeAvg']
delay_timeAvg_vars = ['event']
successProb_timeAvg_vars = ['event', 'probability']
# successProb_timeAvg_vars = ['probability']
nonMarkingProb_timeAvg_vars = ['event']
min_sample_size = 15
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
    
def check_all_delayConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths, last="", bias=0):
    res = {}
    res['MaxEpsilonIneq'] = {}
    for flow in endToEnd_statistics.keys():
        res['MaxEpsilonIneq'][flow] = {}
        for path in paths:
            res['MaxEpsilonIneq'][flow][path] = {}
            for var_method in endToEnd_statistics[flow]['delay'].keys():
                if var_method != 'event_poisson_eventAvg' and var_method != 'event_eventAvg':
                    res['MaxEpsilonIneq'][flow][path][var_method] = check_MaxEpsilon_ineq_delay(endToEnd_statistics[flow]['delay'][var_method][path], samples_paths_aggregated_statistics[flow][path], last)
                else:
                    # e = (samples_paths_aggregated_statistics[flow][path][last + 'DelayMean'] * samples_paths_aggregated_statistics[flow][path]['MaxEpsilon' + last + 'Delay']) + endToEnd_statistics[flow]['delay'][var_method][path][1] * confidenceValue
                    sigma = (samples_paths_aggregated_statistics[flow][path][last + 'DelayMean'] * samples_paths_aggregated_statistics[flow][path]['MaxEpsilon' + last + 'Delay']) / confidenceValue
                    # sigma_e = endToEnd_statistics[flow]['delay'][var_method][path][1]
                    if (endToEnd_statistics[flow]['sampleSize']['delay'][path] < min_sample_size):
                        res['MaxEpsilonIneq'][flow][path][var_method] = False
                        continue
                    sigma_e = sigma * np.sqrt(samples_paths_aggregated_statistics[flow][path]['SampleSize']) / np.sqrt(endToEnd_statistics[flow]['sampleSize']['delay'][path])
                    # e = confidenceValue * np.sqrt((sigma**2) + (sigma_e**2))
                    if bias:
                        e = confidenceValue * np.sqrt((sigma**2) + (sigma_e**2)) + endToEnd_statistics[flow]['bias']['delay'][path]
                    else:
                        e = confidenceValue * np.sqrt((sigma**2) + (sigma_e**2))
                    res['MaxEpsilonIneq'][flow][path][var_method] = (abs(endToEnd_statistics[flow]['delay'][var_method][path][0] - samples_paths_aggregated_statistics[flow][path][last + 'DelayMean']) <= e)
    return res

def check_all_successProbConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths, number_of_segments, last="", bias=0):
    res = {}
    res['MaxEpsilonIneq'] = {}
    for flow in endToEnd_statistics.keys():
        res['MaxEpsilonIneq'][flow] = {}
        for path in paths:
            res['MaxEpsilonIneq'][flow][path] = {}
            for var_method in endToEnd_statistics[flow]['successProb'].keys():
                if var_method != 'event_poisson_eventAvg' and var_method != 'probability_poisson_eventAvg' and var_method != 'event_eventAvg' and var_method != 'probability_eventAvg':
                    res['MaxEpsilonIneq'][flow][path][var_method] = check_MaxEpsilon_ineq_successProb(np.log(endToEnd_statistics[flow]['successProb'][var_method][path]), samples_paths_aggregated_statistics[flow][path], number_of_segments, last)
                else:
                    # epsp = (endToEnd_statistics[flow]['successProb'][var_method][path][1] * confidenceValue) / endToEnd_statistics[flow]['successProb'][var_method][path][0]
                    # eps = samples_paths_aggregated_statistics[flow][path]['MaxEpsilon' + last + 'SuccessProb']
                    # e = samples_paths_aggregated_statistics[flow][path][last + 'SuccessProbMean'] - np.log(endToEnd_statistics[flow]['successProb'][var_method][path][0])
                    # res['MaxEpsilonIneq'][flow][path][var_method] = ((e <= (np.log(1+epsp) - np.log(1-eps))) and (e >= (np.log(1-epsp) - np.log(1+eps))))
                    sigma = (np.exp(samples_paths_aggregated_statistics[flow][path][last + 'SuccessProbMean']) * samples_paths_aggregated_statistics[flow][path]['MaxEpsilon' + last + 'SuccessProb']) / confidenceValue
                    # sigma_e = endToEnd_statistics[flow]['successProb'][var_method][path][1]
                    if (endToEnd_statistics[flow]['sampleSize']['successProb'][path] < min_sample_size):
                        res['MaxEpsilonIneq'][flow][path][var_method] = False
                        continue
                    sigma_e = sigma * np.sqrt(samples_paths_aggregated_statistics[flow][path]['SampleSize']) / np.sqrt(endToEnd_statistics[flow]['sampleSize']['successProb'][path])
                    # e = confidenceValue * np.sqrt((sigma**2) + (sigma_e**2))
                    e = confidenceValue * np.sqrt((sigma**2) + (sigma_e**2)) + bias
                    res['MaxEpsilonIneq'][flow][path][var_method] = (abs(np.exp(samples_paths_aggregated_statistics[flow][path][last + 'SuccessProbMean']) - endToEnd_statistics[flow]['successProb'][var_method][path][0]) <= e)
    return res

def check_all_nonMarkingProbConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths, number_of_segments, bias=0):
    res = {}
    res['MaxEpsilonIneq'] = {}
    for flow in endToEnd_statistics.keys():
        res['MaxEpsilonIneq'][flow] = {}
        for path in paths:
            res['MaxEpsilonIneq'][flow][path] = {}
            for var_method in endToEnd_statistics[flow]['nonMarkingProb'].keys():
                if var_method != 'event_poisson_eventAvg' and var_method != 'event_eventAvg':
                    res['MaxEpsilonIneq'][flow][path][var_method] = check_MaxEpsilon_ineq_nonMarkingProb(np.log(endToEnd_statistics[flow]['nonMarkingProb'][var_method][path]), samples_paths_aggregated_statistics[flow][path], number_of_segments)
                else:
                    # epsp = (endToEnd_statistics[flow]['nonMarkingProb'][var_method][path][1] * confidenceValue) / endToEnd_statistics[flow]['nonMarkingProb'][var_method][path][0]
                    # eps = samples_paths_aggregated_statistics[flow][path]['MaxEpsilonNonMarkingProb']
                    # e = samples_paths_aggregated_statistics[flow][path]['NonMarkingProbMean'] - np.log(endToEnd_statistics[flow]['nonMarkingProb'][var_method][path][0])
                    # res['MaxEpsilonIneq'][flow][path][var_method] = ((e <= (np.log(1+epsp) - np.log(1-eps))) and (e >= (np.log(1-epsp) - np.log(1+eps))))
                    sigma = (np.exp(samples_paths_aggregated_statistics[flow][path]['NonMarkingProbMean']) * samples_paths_aggregated_statistics[flow][path]['MaxEpsilonNonMarkingProb']) / confidenceValue
                    # sigma_e = endToEnd_statistics[flow]['nonMarkingProb'][var_method][path][1]
                    if (endToEnd_statistics[flow]['sampleSize']['nonMarkingProb'][path] < min_sample_size):
                        res['MaxEpsilonIneq'][flow][path][var_method] = False
                        continue
                    sigma_e = sigma * np.sqrt(samples_paths_aggregated_statistics[flow][path]['SampleSize']) / np.sqrt(endToEnd_statistics[flow]['sampleSize']['nonMarkingProb'][path])
                    # e = confidenceValue * np.sqrt((sigma**2) + (sigma_e**2))
                    e = confidenceValue * np.sqrt((sigma**2) + (sigma_e**2)) + bias
                    res['MaxEpsilonIneq'][flow][path][var_method] = (abs(np.exp(samples_paths_aggregated_statistics[flow][path]['NonMarkingProbMean']) - endToEnd_statistics[flow]['nonMarkingProb'][var_method][path][0]) <= e)
    return res

def check_all_lastNonMarkingProbConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths, number_of_segments, bias=0):
    res = {}
    res['MaxEpsilonIneq'] = {}
    for flow in endToEnd_statistics.keys():
        res['MaxEpsilonIneq'][flow] = {}
        for path in paths:
            res['MaxEpsilonIneq'][flow][path] = {}
            for var_method in endToEnd_statistics[flow]['nonMarkingProb'].keys():
                if var_method != 'event_poisson_eventAvg' and var_method != 'event_eventAvg':
                    res['MaxEpsilonIneq'][flow][path][var_method] = check_MaxEpsilon_ineq_lastNonMarkingProb(np.log(endToEnd_statistics[flow]['nonMarkingProb'][var_method][path]), samples_paths_aggregated_statistics[flow][path], number_of_segments)
                else:
                    # epsp = (endToEnd_statistics[flow]['nonMarkingProb'][var_method][path][1] * confidenceValue) / endToEnd_statistics[flow]['nonMarkingProb'][var_method][path][0]
                    # eps = samples_paths_aggregated_statistics[flow][path]['MaxEpsilonLastNonMarkingProb']
                    # e = samples_paths_aggregated_statistics[flow][path]['LastNonMarkingProbMean'] - np.log(endToEnd_statistics[flow]['nonMarkingProb'][var_method][path][0])
                    # res['MaxEpsilonIneq'][flow][path][var_method] = ((e <= (np.log(1+epsp) - np.log(1-eps))) and (e >= (np.log(1-epsp) - np.log(1+eps))))
                    sigma = (np.exp(samples_paths_aggregated_statistics[flow][path]['LastNonMarkingProbMean']) * samples_paths_aggregated_statistics[flow][path]['MaxEpsilonLastNonMarkingProb']) / confidenceValue
                    if (endToEnd_statistics[flow]['sampleSize']['nonMarkingProb'][path] < min_sample_size):
                        res['MaxEpsilonIneq'][flow][path][var_method] = False
                        continue
                    sigma_e = sigma * np.sqrt(samples_paths_aggregated_statistics[flow][path]['SampleSize']) / np.sqrt(endToEnd_statistics[flow]['sampleSize']['nonMarkingProb'][path])
                    # e = confidenceValue * np.sqrt((sigma**2) + (sigma_e**2))
                    e = confidenceValue * np.sqrt((sigma**2) + (sigma_e**2)) + bias
                    res['MaxEpsilonIneq'][flow][path][var_method] = (abs(np.exp(samples_paths_aggregated_statistics[flow][path]['LastNonMarkingProbMean']) - endToEnd_statistics[flow]['nonMarkingProb'][var_method][path][0]) <= e)

    return res

def prepare_results(flows, queues, num_of_paths):
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
        for i in range(num_of_paths):
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

    return rounds_results

def check_traffic_delayConsistency(endToEnd_statistics, paths, samples_paths_aggregated_statistics):
    delay_results = {}
    delay_results['TrafficsComptDelay'] = {}
    # print(endToEnd_statistics['A0D0'])
    for path in paths:
        delay_results['TrafficsComptDelay'][path] = {}
        for var_method in endToEnd_statistics['A0D0']['delay'].keys():
            if (('poisson_eventAvg' not in var_method)):
                continue
            if endToEnd_statistics['A0D0']['sampleSize']['delay'][path] < min_sample_size or endToEnd_statistics['C0D0']['sampleSize']['delay'][path] < min_sample_size:
                delay_results['TrafficsComptDelay'][path][var_method] = False
                continue
            # sigma_m = endToEnd_statistics['A0D0']['delay'][var_method][path][1] / np.sqrt(endToEnd_statistics['A0D0']['sampleSize']['delay'][path])
            # sigma_c = endToEnd_statistics['C0D0']['delay'][var_method][path][1] / np.sqrt(endToEnd_statistics['C0D0']['sampleSize']['delay'][path])
            sigma = (samples_paths_aggregated_statistics['A0D0'][path]['DelayMean'] * samples_paths_aggregated_statistics['A0D0'][path]['MaxEpsilonDelay']) / confidenceValue
            sigma_m = sigma / np.sqrt(endToEnd_statistics['A0D0']['sampleSize']['delay'][path])
            sigma_c = sigma / np.sqrt(endToEnd_statistics['C0D0']['sampleSize']['delay'][path])
            
            e = confidenceValue * np.sqrt((sigma_m**2) + (sigma_c**2))
            delay_results['TrafficsComptDelay'][path][var_method] = (abs(endToEnd_statistics['A0D0']['delay'][var_method][path][0] - endToEnd_statistics['C0D0']['delay'][var_method][path][0]) <= e)

    return delay_results

def traffics_compatibility_check(rounds_results, endToEnd_statistics, paths, samples_paths_aggregated_statistics):
    delay_results = check_traffic_delayConsistency(endToEnd_statistics, paths, samples_paths_aggregated_statistics)

    for path in paths:
        for var_method in rounds_results['TrafficsComptDelay'].keys():
            if ('poisson_eventAvg' in var_method) and ((endToEnd_statistics['A0D0']['sampleSize']['delay'][path] < min_sample_size) or (endToEnd_statistics['C0D0']['sampleSize']['delay'][path] < min_sample_size)):
                continue
            if (('poisson_eventAvg' not in var_method)):
                continue
            rounds_results['TrafficsComptDelay'][var_method][path][1] += 1
            if delay_results['TrafficsComptDelay'][path][var_method]:
                rounds_results['TrafficsComptDelay'][var_method][path][0] += 1
 

def compatibility_check(rounds_results, samples_paths_aggregated_statistics, endToEnd_statistics, flows_name, paths, number_of_segments, biasCalculator):
    # End to End and Persegment Compatibility Check
    delay_results = check_all_delayConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths, bias=1.0)
    delay_results_noBias = check_all_delayConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths, bias=None)
    # lastDelay_results = check_all_delayConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths, 'Last', biasCalculator.GTBias['QueuingDelay'][1.0][0])
    # lastDelay_results_noBias = check_all_delayConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths, 'Last', 0)
    # successProb_results = check_all_successProbConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths, number_of_segments, bias=biasCalculator.GTBias['DropProb'][1.0][0])
    successProb_results_noBias = check_all_successProbConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths, number_of_segments, bias=0)
    # lastSuccessProb_results = check_all_successProbConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths, number_of_segments, 'Last', bias=biasCalculator.GTBias['DropProb'][1.0][0])
    # lastSuccessProb_results_noBias = check_all_successProbConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths, number_of_segments, 'Last', bias=0)
    # nonMarkingProb_results = check_all_nonMarkingProbConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths, number_of_segments, biasCalculator.GTBias['MarkingProb'][1.0][0])
    nonMarkingProb_results_noBias = check_all_nonMarkingProbConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths, number_of_segments, 0)
    # lastNonMarkingProb_results = check_all_lastNonMarkingProbConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths, number_of_segments, biasCalculator.GTBias['MarkingProb'][1.0][0])
    # lastNonMarkingProb_results_noBias = check_all_lastNonMarkingProbConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths, number_of_segments, 0)

    for flow in flows_name:
        for path in paths:
            for var_method in rounds_results['MaxEpsilonIneqDelay'].keys():
                if ('poisson_eventAvg' in var_method) and (endToEnd_statistics[flow]['sampleSize']['delay'][path] < min_sample_size):
                    continue
                rounds_results['MaxEpsilonIneqDelay'][var_method][flow][path][1] += 1
                # rounds_results['MaxEpsilonIneqLastDelay'][var_method][flow][path][1] += 1
                if delay_results['MaxEpsilonIneq'][flow][path][var_method]:
                    rounds_results['MaxEpsilonIneqDelay'][var_method][flow][path][0]['WBias'] += 1
                # if lastDelay_results['MaxEpsilonIneq'][flow][path][var_method]:
                    # rounds_results['MaxEpsilonIneqLastDelay'][var_method][flow][path][0]['WBias'] += 1
                if delay_results_noBias['MaxEpsilonIneq'][flow][path][var_method]:
                    rounds_results['MaxEpsilonIneqDelay'][var_method][flow][path][0]['WOBias'] += 1
                # if lastDelay_results_noBias['MaxEpsilonIneq'][flow][path][var_method]:
                #     rounds_results['MaxEpsilonIneqLastDelay'][var_method][flow][path][0]['WOBias'] += 1

            for var_method in rounds_results['MaxEpsilonIneqSuccessProb'].keys():
                if ('poisson_eventAvg' in var_method) and (endToEnd_statistics[flow]['sampleSize']['successProb'][path] < min_sample_size):
                    continue
                rounds_results['MaxEpsilonIneqSuccessProb'][var_method][flow][path][1] += 1
            #     rounds_results['MaxEpsilonIneqLastSuccessProb'][var_method][flow][path][1] += 1
            #     if successProb_results['MaxEpsilonIneq'][flow][path][var_method]:
            #         rounds_results['MaxEpsilonIneqSuccessProb'][var_method][flow][path][0]['WBias'] += 1
            #     if lastSuccessProb_results['MaxEpsilonIneq'][flow][path][var_method]:
            #         rounds_results['MaxEpsilonIneqLastSuccessProb'][var_method][flow][path][0]['WBias'] += 1
                if successProb_results_noBias['MaxEpsilonIneq'][flow][path][var_method]:
                    rounds_results['MaxEpsilonIneqSuccessProb'][var_method][flow][path][0]['WOBias'] += 1
            #     if lastSuccessProb_results_noBias['MaxEpsilonIneq'][flow][path][var_method]:
            #         rounds_results['MaxEpsilonIneqLastSuccessProb'][var_method][flow][path][0]['WOBias'] += 1
            
            for var_method in rounds_results['MaxEpsilonIneqNonMarkingProb'].keys():
                if ('poisson_eventAvg' in var_method) and (endToEnd_statistics[flow]['sampleSize']['nonMarkingProb'][path] < min_sample_size):
                    continue
                rounds_results['MaxEpsilonIneqNonMarkingProb'][var_method][flow][path][1] += 1
            #     rounds_results['MaxEpsilonIneqLastNonMarkingProb'][var_method][flow][path][1] += 1
            #     if nonMarkingProb_results['MaxEpsilonIneq'][flow][path][var_method]:
            #         rounds_results['MaxEpsilonIneqNonMarkingProb'][var_method][flow][path][0]['WBias'] += 1
            #     if lastNonMarkingProb_results['MaxEpsilonIneq'][flow][path][var_method]:
            #         rounds_results['MaxEpsilonIneqLastNonMarkingProb'][var_method][flow][path][0]['WBias'] += 1
                if nonMarkingProb_results_noBias['MaxEpsilonIneq'][flow][path][var_method]:
                    rounds_results['MaxEpsilonIneqNonMarkingProb'][var_method][flow][path][0]['WOBias'] += 1
            #     if lastNonMarkingProb_results_noBias['MaxEpsilonIneq'][flow][path][var_method]:
            #         rounds_results['MaxEpsilonIneqLastNonMarkingProb'][var_method][flow][path][0]['WOBias'] += 1

            
def analyze_single_experiment(return_dict, rate, queues_names, confidenceValue, steadyStart, steadyEnd, rounds_results, results_folder, config, experiment=0, ns3_path=__ns3_path, differentiationDelay=None, errorRate=None, load=None):
    srcHostToSwitchLinkRate = convert_to_float(config.get('SingleQueue', 'srcHostToSwitchLinkRate')) * 1e-3
    bottleneckLinkRate = convert_to_float(config.get('SingleQueue', 'bottleneckLinkRate')) * rate * 1e-3
    linkDelay = convert_to_float(config.get('Settings', 'hostToTorLinkDelay')) * 1e6
    swtichDstREDQueueDiscMaxSize = convert_to_float(config.get('Settings', 'swtichDstREDQueueDiscMaxSize'))
    passiveProbe = False if config.get('Settings', 'PassiveProbe') == "0" else True
    num_of_paths = 1
    nHosts = 2
    paths = range(num_of_paths)
    # if differentiationDelay is not None and errorRate is not None:
    #     biasCalculator = BiasCalculator(results_folder, str(rate) + "/D_" + str(differentiationDelay) + "/f_" + str(errorRate), [experiment], steadyStart, steadyEnd, rounds_results, bottleneckLinkRate)
    # else:
    #     biasCalculator = BiasCalculator(results_folder, str(rate) + "/" + str(load), [experiment], steadyStart, steadyEnd, rounds_results, bottleneckLinkRate)
    # biasCalculator.calculateBias(['MarkingProb', 'DropProb', 'QueuingDelay', 'LastMarkingProb'])
    endToEndStats = calculate_offline_computations(__ns3_path, rate, 'EndToEnd_packets', str(experiment), results_folder, steadyStart, steadyEnd, "SentTime", nHosts, True, "IsReceived", [srcHostToSwitchLinkRate, bottleneckLinkRate], [linkDelay, linkDelay], swtichDstREDQueueDiscMaxSize, differentiationDelay=differentiationDelay, errorRate=errorRate, load=load, passiveProbe=passiveProbe)
    # endToEndStats = calculate_offline_computations_on_switch(__ns3_path, results_folder, rate, experiment, 'PoissonSampler_queueSize', steadyStart, steadyEnd, paths, bottleneckLinkRate, load)
    # plot_queuingDelay_distribution(__ns3_path, results_folder, str(rate) + "/" + str(load), experiment, 'PoissonSampler_queueSize', steadyStart, steadyEnd, paths, bottleneckLinkRate)
    # calculate_offline_computations(__ns3_path, rate, 'EndToEnd_markings', str(experiment), results_folder, endToEndStats['A0D0']['first'][0], endToEndStats['A0D0']['last'][0], "Time", linksRates=[srcHostToSwitchLinkRate, bottleneckLinkRate], linkDelays=[linkDelay, linkDelay], stats=endToEndStats)
    samplesSats = calculate_offline_computations(__ns3_path, rate, 'PoissonSampler_events', str(experiment), results_folder, endToEndStats['A0D0']['first'][0], endToEndStats['A0D0']['last'][0], "Time", nHosts, linksRates=[bottleneckLinkRate], swtichDstREDQueueDiscMaxSize=swtichDstREDQueueDiscMaxSize, differentiationDelay=differentiationDelay, errorRate=errorRate, load=load)
    # mixingRate = computeMixingRate(__ns3_path, results_folder, str(rate) + "/" + str(load), experiment, 'PoissonSampler_queueSize', endToEndStats['A0D0']['first'][0], endToEndStats['A0D0']['last'][0], [srcHostToSwitchLinkRate, bottleneckLinkRate], [linkDelay, linkDelay])
    # rounds_results['DropRate'].append(calculate_avgDrop_rate_offline(endToEndStats, paths))
    # samples_paths_statistics
    samples_paths_aggregated_statistics = {}
    for flow in endToEndStats.keys():
        samples_paths_aggregated_statistics[flow] = {}
        for path in paths:
            samples_paths_aggregated_statistics[flow][path] = {}
            samples_paths_aggregated_statistics[flow][path]['SampleSize'] = samplesSats['SD0']['sampleSize']
            samples_paths_aggregated_statistics[flow][path]['DelayMean'] = samplesSats['SD0']['DelayMean']
            samples_paths_aggregated_statistics[flow][path]['MaxEpsilonDelay'] = calc_epsilon(confidenceValue, samplesSats['SD0'])

            # samples_paths_aggregated_statistics[flow][path]['LastDelayMean'] = samplesSats['SD0']['LastDelayMean']
            # samples_paths_aggregated_statistics[flow][path]['MaxEpsilonLastDelay'] = calc_epsilon(confidenceValue, samplesSats['SD0'], "Last")
            
            samples_paths_aggregated_statistics[flow][path]['SuccessProbMean'] = np.log(samplesSats['SD0']['SuccessProbMean'])
            samples_paths_aggregated_statistics[flow][path]['MaxEpsilonSuccessProb'] = calc_epsilon_loss(confidenceValue, samplesSats['SD0'])
            
            # samples_paths_aggregated_statistics[flow][path]['LastSuccessProbMean'] = np.log(samplesSats['SD0']['LastSuccessProbMean'])
            # samples_paths_aggregated_statistics[flow][path]['MaxEpsilonLastSuccessProb'] = calc_epsilon_loss(confidenceValue, samplesSats['SD0'], "Last")

            samples_paths_aggregated_statistics[flow][path]['NonMarkingProbMean'] = np.log(samplesSats['SD0']['NonMarkingProbMean'])
            samples_paths_aggregated_statistics[flow][path]['MaxEpsilonNonMarkingProb'] = calc_epsilon_marking(confidenceValue, samplesSats['SD0'])

            # samples_paths_aggregated_statistics[flow][path]['LastNonMarkingProbMean'] = np.log(samplesSats['SD0']['LastNonMarkingProbMean'])
            # samples_paths_aggregated_statistics[flow][path]['MaxEpsilonLastNonMarkingProb'] = calc_epsilon_marking(confidenceValue, samplesSats['SD0'], last="Last")

    # endToEnd_statistics
    AverageWorkLoad = 0
    for flow in endToEndStats.keys():
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
            # rounds_results['maxEpsilonLastDelay'][flow][path].append(samples_paths_aggregated_statistics[flow][path]['MaxEpsilonLastDelay'])
            rounds_results['maxEpsilonSuccessProb'][flow][path].append(samples_paths_aggregated_statistics[flow][path]['MaxEpsilonSuccessProb'])
            # rounds_results['maxEpsilonLastSuccessProb'][flow][path].append(samples_paths_aggregated_statistics[flow][path]['MaxEpsilonLastSuccessProb'])
            rounds_results['maxEpsilonNonMarkingProb'][flow][path].append(samples_paths_aggregated_statistics[flow][path]['MaxEpsilonNonMarkingProb'])
            # rounds_results['maxEpsilonLastNonMarkingProb'][flow][path].append(samples_paths_aggregated_statistics[flow][path]['MaxEpsilonLastNonMarkingProb'])
            rounds_results['EndToEndSampleSizeDelay'][flow][path].append(endToEndStats[flow]['sampleSize']['delay'][path])
            rounds_results['EndToEndSampleSizeSuccess'][flow][path].append(endToEndStats[flow]['sampleSize']['successProb'][path])
            rounds_results['EndToEndSampleSizeMarking'][flow][path].append(endToEndStats[flow]['sampleSize']['nonMarkingProb'][path])
            rounds_results['totalPckts'][flow][path].append(endToEndStats[flow]['totalPckts'][path])
            rounds_results['InterArrivals'][flow][path].append(endToEndStats[flow]['InterArrivals'][path])
            rounds_results['DelayBias'][flow][path].append(endToEndStats[flow]['bias']['delay'][path])
            rounds_results['SuccessProbBias'][flow][path].append(endToEndStats[flow]['bias']['successProb'][path])
            rounds_results['NonMarkingProbBias'][flow][path].append(endToEndStats[flow]['bias']['nonMarkingProb'][path])
            AverageWorkLoad += (endToEndStats[flow]['workload'][path])
    
        rounds_results['workLoad'][flow][path].append(endToEndStats[flow]['workload'][path])
        rounds_results['RTT'][flow][path].append(endToEndStats[flow]['RTT'][path])
    rounds_results['AverageWorkLoad'].append(AverageWorkLoad / len(endToEndStats.keys()))
    rounds_results['experiments'] += 1
    number_of_segments = 1
    compatibility_check(rounds_results, samples_paths_aggregated_statistics, endToEndStats, endToEndStats.keys(), range(num_of_paths), number_of_segments, None)
    rounds_results['expSuccessDelay'].append((experiment, rounds_results['MaxEpsilonIneqDelay']["event_poisson_eventAvg"]["A0D0"][0][0]['WOBias'], rounds_results['MaxEpsilonIneqDelay']["event_poisson_eventAvg"]["A0D0"][0][1]))
    # traffics_compatibility_check(rounds_results, endToEndStats, range(num_of_paths), samples_paths_aggregated_statistics)
              
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
        rounds_results[q+'SampleSize'].append(samplesSats[q]['sampleSize'])
        rounds_results[q+'InterArrivals'].append(samplesSats[q]['InterArrivals'])
        rounds_results[q+'Occupancy'].append(samplesSats[q]['Occupancy'])
        rounds_results[q+'PacktsInQueue'].append(samplesSats[q]['PacktsInQueue'])
        rounds_results[q+'EmptyFrac'].append(samplesSats[q]['EmptyFrac'])
        rounds_results[q+'GT1PktsFrac'].append(samplesSats[q]['GT1PktsFrac'])
        # rounds_results[q+'mixingRate'].append(mixingRate[q]['SigneChangeRate'])
        # rounds_results[q+'mixingSignalAvg'].append(mixingRate[q]['SignalAvg'])
        # rounds_results[q+'mixingDelayDiff'].append(mixingRate[q]['DelayDiff'])
        # rounds_results[q+'mixingRateMonly'].append(mixingRate[q]['SigneChangeRateMOnly'])
        # rounds_results[q+'mixingRatePoisson'].append(mixingRate[q]['SigneChangeRatePoisson'])
        # rounds_results[q+'mixingRateE2EPoisson'].append(mixingRate[q]['SigneChangeRateE2EPoisson'])
        # rounds_results[q+'mixingRatePoissonEventAvg'].append(mixingRate[q]['SigneChangeRatePoissonEventAvg'])

    # print("delay std:", rounds_results[q+'Delaystd'])
    # print(experiment, ":", rounds_results['TrafficsComptDelay'], "A0D0: delay", endToEndStats['A0D0']['delay'], " std: " , samplesSats['A0D0']['DelayStd'], "C0D0: delay", endToEndStats['C0D0']['delay'], " std: ", samplesSats['C0D0']['DelayStd'])
    # print(experiment, ":", rounds_results['MaxEpsilonIneqDelay'], samples_paths_aggregated_statistics['A0D0'][0]['SampleSize'], samplesSats['SD0']['PacktsInQueue'], samplesSats['SD0']["BytesInQueue"], samplesSats[q]['GT1PktsFrac'], samplesSats[q]['EmptyFrac'], mixingRate[q]['SigneChangeRate'])
    # print(experiment, ":", samples_paths_aggregated_statistics['A0D0'][0]['DelayMean'])
    # print(experiment, ":", endToEndStats['A0D0']['delay'])
    # print(experiment, ":", mixingRate)
    return_dict[experiment] = rounds_results

def merge_results(return_dict, merged_results, flows, queues, num_of_paths):
    for exp in return_dict.keys():
        merged_results['expSuccessDelay'] += return_dict[exp]['expSuccessDelay']
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
            merged_results[q+'SampleSize'] += return_dict[exp][q+'SampleSize']
            merged_results[q+'InterArrivals'] += return_dict[exp][q+'InterArrivals']
            merged_results[q+'Occupancy'] += return_dict[exp][q+'Occupancy']
            merged_results[q+'PacktsInQueue'] += return_dict[exp][q+'PacktsInQueue']
            merged_results[q+'EmptyFrac'] += return_dict[exp][q+'EmptyFrac']
            merged_results[q+'GT1PktsFrac'] += return_dict[exp][q+'GT1PktsFrac']
            merged_results[q+'mixingRate'] += return_dict[exp][q+'mixingRate']
            merged_results[q+'mixingSignalAvg'] += return_dict[exp][q+'mixingSignalAvg']
            merged_results[q+'mixingRateMonly'] += return_dict[exp][q+'mixingRateMonly']
            merged_results[q+'mixingRatePoisson'] += return_dict[exp][q+'mixingRatePoisson']
            merged_results[q+'mixingRateE2EPoisson'] += return_dict[exp][q+'mixingRateE2EPoisson']
            merged_results[q+'mixingRatePoissonEventAvg'] += return_dict[exp][q+'mixingRatePoissonEventAvg']
            merged_results[q+'mixingDelayDiff'] += return_dict[exp][q+'mixingDelayDiff']

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
    num_of_paths = 1

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
    batch_size = 50
    ks_final_dict = {}
    for i in range(int(experiments_end / batch_size) + 1):
        ths = []
        return_dict = multiprocessing.Manager().dict()
        ks_dict = multiprocessing.Manager().dict()
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
            paths = range(num_of_paths)
            bottleneckLinkRate = convert_to_float(config.get('SingleQueue', 'bottleneckLinkRate')) * rate * 1e-3
            srcHostToSwitchLinkRate = convert_to_float(config.get('SingleQueue', 'srcHostToSwitchLinkRate')) * 1e-3
            LinkDelay = convert_to_float(config.get('Settings', 'hostToTorLinkDelay')) * 1e6
            # steadyStart_plot = convert_to_float(config.get('Settings', 'steadyStart')) * 1e9
            # steadyEnd_plot = convert_to_float(config.get('Settings', 'steadyEnd')) * 1e9
            steadyStart_plot = 0.3 * 1e9
            steadyEnd_plot = 0.8 * 1e9
            # ths.append(multiprocessing.Process(target=plot_queuingDelay_distribution, args=(__ns3_path, results_folder, str(rate) + "/" + str(load), experiment, 'PoissonSampler_queueSize', steadyStart_plot, steadyEnd_plot, paths, bottleneckLinkRate, False)))
            # ths.append(multiprocessing.Process(target=plot_queuingDelay_distribution, args=(__ns3_path, results_folder, str(rate) + "/" + str(load), experiment, 'PoissonSampler_queueSize', steadyStart_plot, steadyEnd_plot, paths, [srcHostToSwitchLinkRate, bottleneckLinkRate], [LinkDelay, LinkDelay], ks_dict)))
            # ths.append(multiprocessing.Process(target=plot_interarrival_distribution, args=(__ns3_path, results_folder, str(rate) + "/" + str(load), experiment, 'PoissonSampler_queueSize', steadyStart_plot, steadyEnd_plot, False)))
            # ths.append(multiprocessing.Process(target=plot_interarrival_distribution, args=(__ns3_path, results_folder, str(rate) + "/" + str(load), experiment, 'PoissonSampler_queueSize', steadyStart_plot, steadyEnd_plot, True)))
            # ths.append(multiprocessing.Process(target=plot_queuingDelay_time, args=(__ns3_path, results_folder, str(rate) + "/" + str(load), experiment, 'PoissonSampler_queueSize', steadyStart_plot, steadyEnd_plot, paths, bottleneckLinkRate)))
            ths.append(multiprocessing.Process(target=analyze_single_experiment, args=(return_dict, rate, queues_names, confidenceValue, steadyStart, steadyEnd, rounds_results, results_folder, config, experiment, ns3_path, differentiationDelay, errorRate, load)))
        
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        ks_final_dict.update(ks_dict)
        merge_results(return_dict, merged_results, flows_name, queues_names, num_of_paths)
        print("{} joind".format(i))
    # print("KS final dict:", ks_final_dict)
    merged_results['AverageWorkLoad'] = sum(merged_results['AverageWorkLoad']) / merged_results['experiments']
    if differentiationDelay is not None and errorRate is not None:
        if differentiationDelay != 0.0:
            os.system('mkdir -p ../Results/results_{}/{}/{}/D_{}/f_{}/'.format(dir, rate, load, differentiationDelay, errorRate))
        with open('../Results/results_{}/{}/{}/D_{}/f_{}/Q_e_m_activePassive_WBiasDelayOnly_switch_1.0_{}_{}_to_{}.json'.format(dir, rate, load, differentiationDelay, errorRate, experiments_end, steadyStart, steadyEnd), 'w') as f:
            js.dump(merged_results, f, indent=4)
    else:
        with open('../Results/results_{}/{}/{}/Q_e_m_activePassive_WBiasDelayOnly_switch_1.0_{}_{}_to_{}.json'.format(dir, rate, load, experiments_end, steadyStart, steadyEnd), 'w') as f:
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
    # steadyStart = convert_to_float(config.get('Settings', 'steadyStart')) * 1e9
    # steadyEnd = convert_to_float(config.get('Settings', 'steadyEnd')) * 1e9
    steadyStart = 0.3 * 1e9
    steadyEnd = 0.8 * 1e9
    experiments = int(config.get('Settings', 'experiments'))
    # if "forward" in args.dir:
    serviceRateScales = [float(x) for x in config.get('Settings', 'serviceRateScales').split(',')]
    loads = [float(x) for x in config.get('Settings', 'load').split(',')]
    traffics = config.get('Settings', 'traffic').split(',')
    # serviceRateScales = [0.5]
    traffics = ["Facebook_HadoopDist_All", "FacebookKeyValue_Sampled"]
    # traffics = ["Google_AllRPC", "Fabricated_Heavy_Head", "Fabricated_Heavy_Middle", "Google_SearchRPC", "Facebook_HadoopDist_All", "FacebookKeyValue_Sampled"]
    # loads = [0.05]
    # elif "param" in args.dir:
    #     serviceRateScales = [float(x) for x in config.get('Settings', 'sampleRateScales').split(',')]
    # else:
    #     serviceRateScales = [float(x) for x in config.get('Settings', 'errorRateScale').split(',')]
    # experiments = 1
    errorRates = [float(x) for x in config.get('Settings', 'errorRate').split(',')]
    # errorRates = [0.001]
    differentiationDelays = [float(x) for x in config.get('Settings', 'differentiationDelay').split(',')]
    # differentiationDelays = [0.5]
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