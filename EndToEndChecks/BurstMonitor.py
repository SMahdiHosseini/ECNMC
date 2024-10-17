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
__ns3_path = "/media/experiments/ns-allinone-3.41_Microburst_test/ns-3.41"

def prepare_results(queues):
    rounds_results = {}
    for q in queues:
        # if q[0] == 'T' and q[2] == 'H' and (q[1] == '2' or q[1] == '3'):
        #     rounds_results[q+'BurstDurations'] = []
        if q[0] == 'T' and q[2] == 'A' and (q[1] == '0' or q[1] == '1'):
            rounds_results[q+'BurstDurations'] = []
        
    rounds_results['DropRate'] = []
    rounds_results['experiments'] = 0

    return rounds_results

def calculate_burst_durations(df):
    burst_durations = []
    burst = False
    for index, row in df.iterrows():
        if row['isHot'] == 1:
            if burst == False:
                burst = True
                burst_durations.append(1)
            else:
                burst_durations[-1] += 1
        else:
            burst = False
    return burst_durations

def plot_burst_durations_cdf(all_burst_durations, rate, dir, tag):
    all_burst_durations.sort()
    # convert to microseconds and plot the CDF
    all_burst_durations = [x * 25 for x in all_burst_durations]
    plt.rcParams['lines.linewidth'] = 10
    y = np.arange(len(all_burst_durations)) / float(len(all_burst_durations) - 1)
    plt.plot(all_burst_durations, y, label=tag)
    plt.legend()
    plt.xlabel('Burst Durations(us)')
    plt.ylabel('CDF')
    plt.title('CDF of Burst Durations for all queues')
    plt.savefig('../results_{}/{}/CDF.png'.format(dir, rate))
    plt.close()

def analyze_single_experiment(return_dict, rate, queues_names, rounds_results, results_folder, experiment=0, ns3_path=__ns3_path):
    num_of_agg_switches = 2
    paths = ['A' + str(i) for i in range(num_of_agg_switches)]
    endToEnd_dfs = read_online_computations(__ns3_path, rate, 'EndToEnd', str(experiment), results_folder)
    bursts_dfs = read_burst_samples(__ns3_path, rate, 'BurstMonitor', str(experiment), results_folder)
    rounds_results['DropRate'].append(calculate_drop_rate_online(endToEnd_dfs, paths))

    # iterate over bursts dataframes and calculate the burst durations. a Burst duration is unbroken sequence of samples that has columns "isHotThroughputUtilization" equal to 1
    # first calculate the burst durations for each queue
    for q in queues_names:
        # if q[0] == 'T' and q[2] == 'H' and (q[1] == '2' or q[1] == '3'):
        #     rounds_results[q+'BurstDurations'].append(calculate_burst_durations(bursts_dfs[q]))
        if q[0] == 'T' and q[2] == 'A' and (q[1] == '0' or q[1] == '1'):
            rounds_results[q+'BurstDurations'].append(calculate_burst_durations(bursts_dfs[q]))

    rounds_results['experiments'] += 1
    return_dict[experiment] = rounds_results

def merge_results(return_dict, merged_results, queues):
    for exp in return_dict.keys():
        for q in queues:
            # if q[0] == 'T' and q[2] == 'H' and (q[1] == '2' or q[1] == '3'):
            #     merged_results[q+'BurstDurations'] += return_dict[exp][q+'BurstDurations']
            if q[0] == 'T' and q[2] == 'A' and (q[1] == '0' or q[1] == '1'):
                merged_results[q+'BurstDurations'] += return_dict[exp][q+'BurstDurations']

    for exp in return_dict.keys():
        merged_results['experiments'] += return_dict[exp]['experiments']
        merged_results['DropRate'] += return_dict[exp]['DropRate']
    
def analyze_all_experiments(rate, steadyStart, steadyEnd, dir, experiments_end=3, ns3_path=__ns3_path):
    results_folder = 'Results_' + dir

    queues_names = read_queues_indicators(ns3_path, rate, results_folder)
    queues_names.sort()

    rounds_results = prepare_results(queues_names)
    merged_results = prepare_results(queues_names)
    for i in range(int(experiments_end / 10) + 1):
        ths = []
        return_dict = multiprocessing.Manager().dict()
        for experiment in range(10 * i, min(experiments_end, 10 * (i + 1))):
            if len(os.listdir('{}/scratch/{}/{}/{}'.format(__ns3_path, results_folder, rate, experiment))) == 0:
                print(experiment)
                continue
            print("Analyzing experiment: ", experiment)
            ths.append(multiprocessing.Process(target=analyze_single_experiment, args=(return_dict, rate, queues_names, rounds_results, results_folder, experiment, ns3_path)))
        
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        merge_results(return_dict, merged_results, queues_names)
        print("{} joind".format(i))
    
    with open('../results_{}/{}/res_{}_{}_{}_to_{}.json'.format(dir, rate, results_folder, experiments_end, steadyStart, steadyEnd), 'w') as f:
        js.dump(merged_results, f, indent=4)
    
    all_burst_durations_TA = []
    all_burst_durations_TH = []
    # aggregate the results of all queues together and plot the CDF
    for q in queues_names:
        # if q[0] == 'T' and q[2] == 'H' and (q[1] == '2' or q[1] == '3'):
        #     all_burst_durations = []
        #     for burst_durations in rounds_results[q+'BurstDurations']:
        #         all_burst_durations += burst_durations
        #     if len(all_burst_durations) == 0:
        #         continue
        #     plot_burst_durations_cdf(all_burst_durations)
        if q[0] == 'T' and q[2] == 'A' and (q[1] == '0' or q[1] == '1'):
            all_burst_durations_TA += [item for sublist in merged_results[q+'BurstDurations'] for item in sublist]
        
    plot_burst_durations_cdf(all_burst_durations_TA, rate, dir, 'TA')

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
    # serviceRateScales = [0.90]
    # serviceRateScales = [0.91, 0.93, 0.95, 0.97, 0.99, 1.01, 1.03, 1.05]
    experiments = 1

    for rate in serviceRateScales:
        print("\nAnalyzing experiments for rate: ", rate)
        analyze_all_experiments(rate, steadyStart, steadyEnd, args.dir, experiments_end=experiments, ns3_path=__ns3_path)
        print("Rate {} {} done".format(rate, experiments))

__main__()