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
sample_rates = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0]
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

    return res

def read_data_naive(__ns3_path, rate, segment, experiment, results_folder, paths):
    file_paths = glob.glob('{}/scratch/{}/{}/{}/*_{}_packets.csv'.format(__ns3_path, results_folder, rate, experiment, segment))
    dfs = {}
    for file_path in file_paths:
        df_name = file_path.split('/')[-1].split('_')[0]
        dfs[df_name] = {}
        dfs[df_name]['timeAverage'] = {}
        dfs[df_name]['DelayStd'] = {}
        df = pd.read_csv(file_path)
        df = df[df['receivedTime'] != -1]
        df = df.reset_index(drop=True)
        df['Delay'] = abs(df['receivedTime'] - df['sentTime'])
        # remove the transmission delay
        df['Delay'] = df['Delay'] - (df['size'] * 16000 *((1 / 300) + (1 / (945 * rate))) )
        for path in paths:
            temp = df[df['path'] == int(path[1])]
            dfs[df_name]['timeAverage'][int(path[1])] = np.mean(temp['Delay'])
            dfs[df_name]['DelayStd'][int(path[1])] = np.std(temp['Delay'])
        
    return dfs

def prepare_results(flows, queues, num_of_agg_switches):
    rounds_results = {}
    rounds_results['MaxEpsilonIneqDelay'] = {}
    rounds_results['MaxEpsilonIneqSuccessProb'] = {}
    rounds_results['MaxEpsilonIneqSuccessProb']['E2E_eventAvg'] = {}
    rounds_results['EndToEndDelayMean'] = {}
    rounds_results['EndToEndDelayStd'] = {}
    rounds_results['EndToEndSuccessProb'] = {}
    rounds_results['EndToEndSuccessProb']['E2E_eventAvg'] = {}
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
        rounds_results['EndToEndDelayMean'][flow] = {}
        rounds_results['EndToEndDelayStd'][flow] = {}
        rounds_results['EndToEndSuccessProb']['E2E_eventAvg'][flow] = {}
        rounds_results['maxEpsilonDelay'][flow] = {}
        rounds_results['maxEpsilonSuccessProb'][flow] = {}
        rounds_results['errors'][flow] = {}
        rounds_results['workLoad'][flow] = {}

        for i in range(num_of_agg_switches):
            rounds_results['MaxEpsilonIneqDelay'][flow]['A' + str(i)] = 0
            rounds_results['MaxEpsilonIneqSuccessProb']['E2E_eventAvg'][flow]['A' + str(i)] = 0
            rounds_results['EndToEndDelayMean'][flow]['A' + str(i)] = []
            rounds_results['EndToEndDelayStd'][flow]['A' + str(i)] = []
            rounds_results['EndToEndSuccessProb']['E2E_eventAvg'][flow]['A' + str(i)] = []
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
            
def analyze_single_experiment(return_dict, rate, queues_names, confidenceValue, rounds_results, results_folder, experiment=0, ns3_path=__ns3_path):
    num_of_agg_switches = 2
    paths = ['A' + str(i) for i in range(num_of_agg_switches)]
    endToEnd_dfs = read_online_computations(__ns3_path, rate, 'EndToEnd', str(experiment), results_folder)
    samples_dfs = read_online_computations(__ns3_path, rate, 'PoissonSampler', str(experiment), results_folder)
    endToEnd_dfs_delay = read_data_naive(__ns3_path, rate, 'EndToEnd', str(experiment), results_folder, paths)
    rounds_results['DropRate'].append(calculate_drop_rate_online(endToEnd_dfs, paths))

    # samples_paths_statistics
    samples_paths_aggregated_statistics = {}
    for flow in endToEnd_dfs.keys():
        samples_paths_aggregated_statistics[flow] = {}
        for path in paths:
            total_sent = 0
            for f in endToEnd_dfs.keys():
                if f[1] == flow[1]:
                    total_sent += endToEnd_dfs[f]['sentPackets'][int(path[1])]

            samples_paths_aggregated_statistics[flow][path] = {}
            samples_paths_aggregated_statistics[flow][path]['DelayMean'] = sum([samples_dfs['R' + flow[1] + 'H' + flow[3]]['DelayMean'],
                                                                                samples_dfs['T' + flow[1] + path]['DelayMean'], 
                                                                                samples_dfs[path + 'T' + flow[5]]['DelayMean'],
                                                                                samples_dfs['T' + flow[5] + 'H' + flow[7]]['DelayMean']])
            
            samples_paths_aggregated_statistics[flow][path]['MaxEpsilonDelay'] = max([calc_epsilon(confidenceValue, samples_dfs['R' + flow[1] + 'H' + flow[3]]),
                                                                                      calc_epsilon(confidenceValue, samples_dfs['T' + flow[1] + path]),
                                                                                      calc_epsilon(confidenceValue, samples_dfs[path + 'T' + flow[5]]),
                                                                                      calc_epsilon(confidenceValue, samples_dfs['T' + flow[5] + 'H' + flow[7]])])
            
            successProbMean = samples_dfs['T' + flow[1] + path]['GTSampleSize']  / total_sent
            samples_paths_aggregated_statistics[flow][path]['successProbMean'] = np.log(successProbMean)
            
            samples_paths_aggregated_statistics[flow][path]['MaxEpsilonSuccessProb'] = (confidenceValue * np.sqrt((successProbMean * (1 - successProbMean)) / samples_dfs['T' + flow[1] + path]['GTSampleSize'])) / (successProbMean)
    # endToEnd_statistics
    endToEnd_statistics = {}
    AverageWorkLoad = 0
    for flow in endToEnd_dfs.keys():
        endToEnd_statistics[flow] = {}
        for path in paths:
            endToEnd_statistics[flow][path] = {}
            endToEnd_statistics[flow][path]['DelayMean'] = endToEnd_dfs_delay[flow]['timeAverage'][int(path[1])]
            endToEnd_statistics[flow][path]['successProbMean'] = {}
            endToEnd_statistics[flow][path]['successProbMean']['E2E_eventAvg'] = endToEnd_dfs[flow]['successProbMean'][int(path[1])]   

            # if (flow == 'R0H1R2H1' and path == 'A1'):
            #     # print(flow, path, endToEnd_statistics[flow][path]['successProbMean']['E2E_eventAvg'], samples_paths_aggregated_statistics[flow][path]['successProbMean'], samples_paths_aggregated_statistics[flow][path]['MaxEpsilonSuccessProb'])
            #     print(flow, path, endToEnd_statistics[flow][path]['DelayMean'], samples_paths_aggregated_statistics[flow][path]['DelayMean'], samples_paths_aggregated_statistics[flow][path]['MaxEpsilonDelay'])

            rounds_results['EndToEndSuccessProb']['E2E_eventAvg'][flow][path].append(endToEnd_dfs[flow]['successProbMean'][int(path[1])])
            rounds_results['EndToEndDelayMean'][flow][path].append(endToEnd_dfs_delay[flow]['timeAverage'][int(path[1])])
            rounds_results['EndToEndDelayStd'][flow][path].append(endToEnd_dfs_delay[flow]['DelayStd'][int(path[1])])
            rounds_results['maxEpsilonDelay'][flow][path].append(samples_paths_aggregated_statistics[flow][path]['MaxEpsilonDelay'])
            rounds_results['maxEpsilonSuccessProb'][flow][path].append(samples_paths_aggregated_statistics[flow][path]['MaxEpsilonSuccessProb'])
            rounds_results['errors'][flow][path].append(abs((samples_paths_aggregated_statistics[flow][path]['DelayMean'] - endToEnd_statistics[flow][path]['DelayMean']) / samples_paths_aggregated_statistics[flow][path]['DelayMean']))
            AverageWorkLoad += (endToEnd_dfs[flow]['receivedPackets'][int(path[1])] * endToEnd_dfs[flow]['averagePacketSize'][int(path[1])] * 8)
    
        rounds_results['workLoad'][flow][path].append(((endToEnd_dfs[flow]['receivedPackets'][0] * endToEnd_dfs[flow]['averagePacketSize'][0]) + (endToEnd_dfs[flow]['receivedPackets'][1] * endToEnd_dfs[flow]['averagePacketSize'][1])) * 8 / 0.5)
    rounds_results['AverageWorkLoad'].append((AverageWorkLoad / 0.5) / 12)
    rounds_results['experiments'] += 1
    number_of_segments = 3
    compatibility_check(rounds_results, samples_paths_aggregated_statistics, endToEnd_statistics, endToEnd_dfs.keys(), ['A' + str(i) for i in range(num_of_agg_switches)], number_of_segments)
              
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
                merged_results['EndToEndDelayMean'][flow]['A' + str(i)] += return_dict[exp]['EndToEndDelayMean'][flow]['A' + str(i)]
                merged_results['EndToEndDelayStd'][flow]['A' + str(i)] += return_dict[exp]['EndToEndDelayStd'][flow]['A' + str(i)]
                merged_results['EndToEndSuccessProb']['E2E_eventAvg'][flow]['A' + str(i)] += return_dict[exp]['EndToEndSuccessProb']['E2E_eventAvg'][flow]['A' + str(i)]

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
            ths.append(multiprocessing.Process(target=analyze_single_experiment, args=(return_dict, rate, queues_names, confidenceValue, rounds_results, results_folder, experiment, ns3_path)))
        
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        merge_results(return_dict, merged_results, flows_name, queues_names)
        print("{} joind".format(i))
    merged_results['AverageWorkLoad'] = sum(merged_results['AverageWorkLoad']) / merged_results['experiments']
    with open('../Results/results_{}/{}/naive_{}_{}_{}_to_{}.json'.format(dir, rate, results_folder, experiments_end, steadyStart, steadyEnd), 'w') as f:
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
    serviceRateScales = [0.79, 0.81, 0.83, 0.85, 0.87, 0.89, 0.91, 0.93, 0.95, 0.97, 0.99, 1.0, 1.01, 1.03, 1.05]
    # serviceRateScales = [0.79]
    # serviceRateScales = [1.0, 1.01, 1.03, 1.05]
    # serviceRateScales = [0.91, 0.93, 0.95, 0.97, 0.99, 1.01, 1.03, 1.05]
    # serviceRateScales = [float(x) for x in config.get('Settings', 'serviceRateScales').split(',')]
    # experiments = 1

    for rate in serviceRateScales:
        print("\nAnalyzing experiments for rate: ", rate)
        analyze_all_experiments(rate, steadyStart, steadyEnd, confidenceValue, args.dir, experiments_end=experiments, ns3_path=__ns3_path)
        print("Rate {} {} done".format(rate, experiments))

__main__()