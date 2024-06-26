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

# __ns3_path = os.popen('locate "ns-3.41" | grep /ns-3.41$').read().splitlines()[0]
__ns3_path = "/home/shossein/ns-allinone-3.41/ns-3.41"
sample_rate = 0.05
confidenceValue = 1.96 # 95% confidence interval
        
def check_MaxEpsilon_ineq(endToEnd_statistics, samples_paths_aggregated_statistics, number_of_segments):
    if (endToEnd_statistics - samples_paths_aggregated_statistics['successProbMean'] <= (number_of_segments * np.log(1 + samples_paths_aggregated_statistics['MaxEpsilon']))) and (endToEnd_statistics - samples_paths_aggregated_statistics['successProbMean'] >= (number_of_segments * np.log(1 - samples_paths_aggregated_statistics['MaxEpsilon']))):
        return True
    else:
        return False
    
def check_all_delayConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths, number_of_segments):
    res = {}
    res['MaxEpsilonIneqPackets'] = {}
    for flow in endToEnd_statistics.keys():
        res['MaxEpsilonIneqPackets'][flow] = {}
        for path in paths:
            res['MaxEpsilonIneqPackets'][flow][path] = check_MaxEpsilon_ineq(np.log(endToEnd_statistics[flow][path]['successProbMeanPackets']), samples_paths_aggregated_statistics[flow][path], number_of_segments)
    return res

def prepare_results(flows, queues, num_of_agg_switches):
    rounds_results = {}

    rounds_results['MaxEpsilonIneqPackets'] = {}
    rounds_results['DropRate'] = []
    rounds_results['EndToEndSuccessProbPackets'] = {}
    rounds_results['maxEpsilon'] = {}


    for flow in flows:
        rounds_results['MaxEpsilonIneqPackets'][flow] = {}
        rounds_results['EndToEndSuccessProbPackets'][flow] = {}
        rounds_results['maxEpsilon'][flow] = {}

        for i in range(num_of_agg_switches):
            rounds_results['MaxEpsilonIneqPackets'][flow]['A' + str(i)] = 0
            rounds_results['EndToEndSuccessProbPackets'][flow]['A' + str(i)] = []
            rounds_results['maxEpsilon'][flow]['A' + str(i)] = []

    rounds_results['experiments'] = 0
    return rounds_results

def compatibility_check(rounds_results, samples_paths_aggregated_statistics, endToEnd_statistics, flows_name, paths, number_of_segments):
    results = check_all_delayConsistency(endToEnd_statistics, samples_paths_aggregated_statistics, paths, number_of_segments)

    for flow in flows_name:
        for path in paths:
            if results['MaxEpsilonIneqPackets'][flow][path]:
                rounds_results['MaxEpsilonIneqPackets'][flow][path] += 1
            

def analyze_single_experiment(rate, steadyStart, steadyEnd, confidenceValue, rounds_results, queues_names, results_folder, experiment=0, ns3_path=__ns3_path):
    num_of_agg_switches = 2
    paths = ['A' + str(i) for i in range(num_of_agg_switches)]
    endToEnd_dfs = read_data(__ns3_path, steadyStart, steadyEnd, rate, 'EndToEnd', 'IsReceived', 'SentTime', str(experiment), True, results_folder, False)
    samples_dfs = read_data(__ns3_path, steadyStart, steadyEnd, rate, 'PoissonSampler', 'IsDeparted', 'SampleTime', str(experiment), False, results_folder)
    start_dfs = read_data(__ns3_path, steadyStart, steadyEnd + 0.5, rate, 'start', 'IsSent', 'ReceiveTime', str(experiment), True, results_folder)

    # integrate the switch data with the endToEnd data
    for start_df in start_dfs.keys():
        start_dfs[start_df] = start_dfs[start_df].drop(columns=['ECN'])

    # print_traffic_rate(endToEnd_dfs)
    rounds_results['DropRate'].append(calculate_drop_rate(__ns3_path, steadyStart, steadyEnd, rate, ['EndToEnd'], 'IsReceived', 'SentTime', str(experiment), results_folder))

    clear_data_from_outliers_in_time(endToEnd_dfs, {}, start_dfs)

    # for flow in endToEnd_dfs.keys():
    #     endToEnd_dfs[flow] = pd.merge(endToEnd_dfs[flow], start_dfs[flow[0:4]].drop(columns=['ReceiveTime', 'SentTime', 'IsReceived']), on=['SourceIp', 'SourcePort', 'DestinationIp', 'DestinationPort', 'PayloadSize', 'SequenceNb', 'Id'], how='inner')

    # samples switches statistics
    samples_switches_statistics = {}
    for sample_df in samples_dfs.keys():
        # inline if statement
        samples_dfs[sample_df]['MarkingProb'] = samples_dfs[sample_df]['MarkingProb'].apply(lambda x: 1.0 if x > 1.0 else x)
        samples_switches_statistics[sample_df] = get_loss_statistics(samples_dfs[sample_df])
        
    # samples_paths_statistics
    samples_paths_aggregated_statistics = {}
    for flow in endToEnd_dfs.keys():
        samples_paths_aggregated_statistics[flow] = {}
        for path in paths:
            samples_paths_aggregated_statistics[flow][path] = {}
            # get the sum of the logaritm of means 
            samples_paths_aggregated_statistics[flow][path]['successProbMean'] = sum([np.log(samples_switches_statistics['T' + flow[1] + path]['successProbMean']),
                                                                                   np.log(samples_switches_statistics[path + 'T' + flow[5]]['successProbMean']),
                                                                                   np.log(samples_switches_statistics['T' + flow[5] + 'H' + flow[7]]['successProbMean'])])
            
            samples_paths_aggregated_statistics[flow][path]['MaxEpsilon'] = max([calc_epsilon_loss(confidenceValue, samples_switches_statistics['T' + flow[1] + path]),
                                                                                 calc_epsilon_loss(confidenceValue, samples_switches_statistics[path + 'T' + flow[5]]),
                                                                                 calc_epsilon_loss(confidenceValue, samples_switches_statistics['T' + flow[5] + 'H' + flow[7]])])

    # endToEnd_statistics
    endToEnd_statistics = {}
    for flow in endToEnd_dfs.keys():
        endToEnd_statistics[flow] = {}
        for path in paths:
            # remove A from the path
            temp = endToEnd_dfs[flow][endToEnd_dfs[flow]['Path'] == int(path[1])]
            endToEnd_statistics[flow][path] = get_endToEd_loss_statistics(temp)
            rounds_results['EndToEndSuccessProbPackets'][flow][path].append(endToEnd_statistics[flow][path]['successProbMeanPackets'])
            rounds_results['maxEpsilon'][flow][path].append(samples_paths_aggregated_statistics[flow][path]['MaxEpsilon'])

    rounds_results['experiments'] += 1
    number_of_segments = 3
    compatibility_check(rounds_results, samples_paths_aggregated_statistics, endToEnd_statistics, endToEnd_dfs.keys(), ['A' + str(i) for i in range(num_of_agg_switches)], number_of_segments)

def analyze_all_experiments(rate, steadyStart, steadyEnd, confidenceValue, experiments_start=0, experiments_end=3, ns3_path=__ns3_path):
    # results_folder = 'Results_forward'
    # results_folder = 'Results_reverse_loss_2'
    results_folder = 'Results_reverse_delay_1'
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

    with open('../results_postProcessing_reverse_delay_1/{}/loss_{}_{}_{}_to_{}.json'.format(rate, results_folder, experiments_end, steadyStart, steadyEnd), 'w') as f:
    # with open('../results_postProcessing/{}/loss_{}_{}_{}_{}_to_{}.json'.format(1.0, rate, results_folder, experiments_end, steadyStart, steadyEnd), 'w') as f:
        # config = configparser.ConfigParser()
        # config.read('../Parameters.config')
        # errorRate = convert_to_float(config.get('Settings', 'errorRate')) * rate
        # rounds_results['ErrorRate'] = errorRate
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
    serviceRateScales = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    # experiments = 10

    for rate in serviceRateScales:
        print("\nAnalyzing experiments for rate: ", rate)
        analyze_all_experiments(rate, steadyStart, steadyEnd, confidenceValue, experiments_start=0, experiments_end=experiments, ns3_path=__ns3_path)
        print("Rate {} {} done".format(rate, experiments))

__main__()