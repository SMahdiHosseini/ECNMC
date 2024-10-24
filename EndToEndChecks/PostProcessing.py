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
sample_rate = 0.05
confidenceValue = 1.96 # 95% confidence interval
        
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
            res['MaxEpsilonIneq'][flow][path] = check_MaxEpsilon_ineq_successProb(np.log(endToEnd_statistics[flow][path]['successProbMean']), samples_paths_aggregated_statistics[flow][path], number_of_segments)
    return res

def prepare_results(flows, queues, num_of_agg_switches):
    rounds_results = {}
    rounds_results['MaxEpsilonIneqDelay'] = {}
    rounds_results['MaxEpsilonIneqSuccessProb'] = {}
    rounds_results['EndToEndDelayMean'] = {}
    rounds_results['EndToEndDelayStd'] = {}
    rounds_results['EndToEndSuccessProb'] = {}
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
        rounds_results['MaxEpsilonIneqSuccessProb'][flow] = {}
        rounds_results['EndToEndDelayMean'][flow] = {}
        rounds_results['EndToEndDelayStd'][flow] = {}
        rounds_results['EndToEndSuccessProb'][flow] = {}
        rounds_results['maxEpsilonDelay'][flow] = {}
        rounds_results['maxEpsilonSuccessProb'][flow] = {}
        rounds_results['errors'][flow] = {}
        rounds_results['workLoad'][flow] = {}

        for i in range(num_of_agg_switches):
            rounds_results['MaxEpsilonIneqDelay'][flow]['A' + str(i)] = 0
            rounds_results['MaxEpsilonIneqSuccessProb'][flow]['A' + str(i)] = 0
            rounds_results['EndToEndDelayMean'][flow]['A' + str(i)] = []
            rounds_results['EndToEndDelayStd'][flow]['A' + str(i)] = []
            rounds_results['EndToEndSuccessProb'][flow]['A' + str(i)] = []
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
            if successProb_results['MaxEpsilonIneq'][flow][path]:
                rounds_results['MaxEpsilonIneqSuccessProb'][flow][path] += 1

def analyze_single_experiment(return_dict, rate, queues_names, confidenceValue, rounds_results, results_folder, experiment=0, ns3_path=__ns3_path):
    num_of_agg_switches = 2
    paths = ['A' + str(i) for i in range(num_of_agg_switches)]
    endToEnd_dfs = read_online_computations(__ns3_path, rate, 'EndToEnd', str(experiment), results_folder)
    samples_dfs = read_online_computations(__ns3_path, rate, 'PoissonSampler', str(experiment), results_folder)
    # switches_dfs = read_data(__ns3_path, 0.2, 1.2, rate, 'Switch', 'IsSent', 'ReceiveTime', str(experiment), True, results_folder)
    # # # add delay columns which is ReceiveTime - SentTime and print the delay mean 9of the rows that path is 0 only for switch "T0"
    # for switch in switches_dfs.keys():
    #     switches_dfs[switch]['Delay'] = switches_dfs[switch]['SentTime'] - switches_dfs[switch]['ReceiveTime']
    #     if switch == 'T0':
    #         print(get_statistics(switches_dfs[switch][switches_dfs[switch]['path'] == 0], timeAvg=True)['timeAvg'])
    #         print(switches_dfs[switch][switches_dfs[switch]['path'] == 0]['Delay'].mean())
    #         # print(switches_dfs[switch][switches_dfs[switch]['SourceIp'] == '10.1.1.1']['Delay'].mean())
    rounds_results['DropRate'].append(calculate_drop_rate_online(endToEnd_dfs, paths))

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
            endToEnd_statistics[flow][path]['successProbMean'] = endToEnd_dfs[flow]['successProbMean'][int(path[1])]            

            rounds_results['EndToEndSuccessProb'][flow][path].append(endToEnd_dfs[flow]['successProbMean'][int(path[1])])
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
    compatibility_check(rounds_results, samples_paths_aggregated_statistics, endToEnd_statistics, endToEnd_dfs.keys(), ['A' + str(i) for i in range(num_of_agg_switches)], number_of_segments)
    if rounds_results['MaxEpsilonIneqSuccessProb']['R0H0R2H0']['A1'] != 0:
        print(path, np.log(endToEnd_statistics['R0H0R2H0']['A1']['successProbMean']), samples_paths_aggregated_statistics['R0H0R2H0']['A1']['successProbMean'], 
            samples_paths_aggregated_statistics['R0H0R2H0']['A1']['MaxEpsilonSuccessProb'])
              
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
                merged_results['MaxEpsilonIneqSuccessProb'][flow]['A' + str(i)] += return_dict[exp]['MaxEpsilonIneqSuccessProb'][flow]['A' + str(i)]
                merged_results['EndToEndDelayMean'][flow]['A' + str(i)] += return_dict[exp]['EndToEndDelayMean'][flow]['A' + str(i)]
                merged_results['EndToEndDelayStd'][flow]['A' + str(i)] += return_dict[exp]['EndToEndDelayStd'][flow]['A' + str(i)]
                merged_results['EndToEndSuccessProb'][flow]['A' + str(i)] += return_dict[exp]['EndToEndSuccessProb'][flow]['A' + str(i)]
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
    for i in range(int(experiments_end / 10) + 1):
        ths = []
        return_dict = multiprocessing.Manager().dict()
        for experiment in range(10 * i, min(experiments_end, 10 * (i + 1))):
            if len(os.listdir('{}/scratch/{}/{}/{}'.format(__ns3_path, results_folder, rate, experiment))) == 0:
                print(experiment)
                continue
            print("Analyzing experiment: ", experiment)
            ths.append(multiprocessing.Process(target=analyze_single_experiment, args=(return_dict, rate, queues_names, confidenceValue, rounds_results, results_folder, experiment, ns3_path)))
        
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        merge_results(return_dict, merged_results, flows_name, queues_names)
        print("{} joind".format(i))
    merged_results['AverageWorkLoad'] = sum(merged_results['AverageWorkLoad']) / merged_results['experiments']
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
    if args.dir == "forward":
        serviceRateScales = [float(x) for x in config.get('Settings', 'serviceRateScales').split(',')]
    else:
        serviceRateScales = [float(x) for x in config.get('Settings', 'errorRateScale').split(',')]
    serviceRateScales = [0.89]
    # serviceRateScales = [0.91, 0.93, 0.95, 0.97, 0.99, 1.01, 1.03, 1.05]
    # serviceRateScales = [float(x) for x in config.get('Settings', 'serviceRateScales').split(',')]
    experiments = 30

    for rate in serviceRateScales:
        print("\nAnalyzing experiments for rate: ", rate)
        analyze_all_experiments(rate, steadyStart, steadyEnd, confidenceValue, args.dir, experiments_end=experiments, ns3_path=__ns3_path)
        print("Rate {} {} done".format(rate, experiments))

__main__()