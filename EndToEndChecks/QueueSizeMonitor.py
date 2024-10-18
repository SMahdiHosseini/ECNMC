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

def prepare_results(queues):
    rounds_results = {}
    for q in queues:
        # if q[0] == 'T' and q[2] == 'H' and (q[1] == '2' or q[1] == '3'):
        #     rounds_results[q+'BurstDurations'] = []
        if q[0] == 'T' and q[2] == 'A' and (q[1] == '0' or q[1] == '1'):
            rounds_results[q+'queuesieze'] = []
        
    rounds_results['DropRate'] = []
    rounds_results['experiments'] = 0

    return rounds_results

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
            rounds_results[q+'queuesieze'] = bursts_dfs[q]['queueSize'].tolist()[12000:32000]

    rounds_results['experiments'] += 1
    return_dict[experiment] = rounds_results

def merge_results(return_dict, merged_results, queues):
    for exp in return_dict.keys():
        for q in queues:
            # if q[0] == 'T' and q[2] == 'H' and (q[1] == '2' or q[1] == '3'):
            #     merged_results[q+'BurstDurations'] += return_dict[exp][q+'BurstDurations']
            if q[0] == 'T' and q[2] == 'A' and (q[1] == '0' or q[1] == '1'):
                merged_results[q+'queuesieze'] += return_dict[exp][q+'queuesieze']

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
    
    return merged_results['T0A0queuesieze'], merged_results['T0A1queuesieze'], merged_results['T1A0queuesieze'], merged_results['T1A1queuesieze']

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
    serviceRateScales = [float(x) for x in config.get('Settings', 'serviceRateScales').split(',')]
    # serviceRateScales = [0.90]
    # serviceRateScales = [0.91, 0.93, 0.95, 0.97, 0.99, 1.01, 1.03, 1.05]
    selectedRates = [0.79, 0.81, 0.83, 0.85, 0.87, 0.89, 0.91, 0.93, 0.95, 0.97, 0.99]
    colormap = plt.cm.gist_ncar
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(selectedRates)))))
    experiments = 1
    fig1, (ax1, ax2) = plt.subplots(1, 2)
    # fig2, ax2 = plt.subplots()
    # fig3, ax3 = plt.subplots()
    # fig4, ax4 = plt.subplots()

    for rate in selectedRates:
        print("\nAnalyzing experiments for rate: ", rate)
        T0A0, T0A1, T1A0, T1A1 = analyze_all_experiments(rate, steadyStart, steadyEnd, args.dir, experiments_end=experiments, ns3_path=__ns3_path)
        print("Rate {} {} done".format(rate, experiments))
        # plot the queue size over time, the x axis is the time and the y axis is the queue size. for the x axis, multiply the index by 25 and add 300000   
        ax1.plot([i * 25 + 300000 for i in range(len(T0A0))], T0A0, label='Utilization: {}'.format(round(6 * 300 / (2 * 945 * rate), 3)))
        ax2.plot([i * 25 + 300000 for i in range(len(T0A1))], T0A1, label='Utilization: {}'.format(round(6 * 300 / (2 * 945 * rate), 3)))


                 
    # incerease the font size of the legend
    plt.rcParams.update({'legend.fontsize': 'large'})
    # put the legend outside the plot
    # ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax2.legend(loc='upper left', bbox_to_anchor=(0.5, 0.5))
    # ax3.legend()
    # ax4.legend()
    ax1.set_xlabel('time(us)')
    ax1.set_ylabel('Accumulated Drops')
    ax1.title.set_text('T0A0')
    ax2.set_xlabel('time(us)')
    ax2.set_ylabel('Accumulated Drops')
    ax2.title.set_text('T0A1')
    # ax3.set_xlabel('time')
    # ax3.set_ylabel('Queue Size')
    # ax3.title.set_text('T1A0')
    # ax4.set_xlabel('time')
    # ax4.set_ylabel('Queue Size')
    # ax4.title.set_text('T1A1')
    # .title('CDF of Burst Durations for all queues')
    plt.savefig('../results_{}/queueSize.png'.format(args.dir, rate))
    plt.close()


__main__()