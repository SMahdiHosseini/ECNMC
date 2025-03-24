import argparse
import configparser
import os
import json as js
import matplotlib.pyplot as plt
import numpy as np

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
def readResults(results_dir, serviceRateScales, results_dir_file):
    results = {}
    flows = ['A0D0']
    paths = ["0"]
    for rate in serviceRateScales:
        results[rate] = {}
        for file in os.listdir('../Results/results_' + results_dir + '/'+str(rate)+'/'):
            if file.find(results_dir_file) != -1:
                temp = {}
                if file.endswith('.json'):
                    with open('../Results/results_' + results_dir + '/'+str(rate)+'/'+file) as f:
                        temp = js.load(f)
                else:   
                    continue
                for flow in flows:
                    for path in paths:
                        results[rate]['MaxEpsilonIneqDelay'] = {}
                        results[rate]['MaxEpsilonIneqSuccessProb'] = {}
                        results[rate]['MaxEpsilonIneqNonMarkingProb'] = {}
                        results[rate]['MaxEpsilonIneqLastNonMarkingProb'] = {}
                        # results[rate]['SD0Delaystd'] = temp['SD0Delaystd']
                        # results[rate]['SD0DelayMean'] = temp['SD0DelayMean']

                        for var_method in temp['MaxEpsilonIneqDelay'].keys():
                            results[rate]['MaxEpsilonIneqDelay'][var_method] = temp['MaxEpsilonIneqDelay'][var_method][flow][path] / temp['experiments'] * 100
                        
                        for var_method in temp['MaxEpsilonIneqSuccessProb'].keys():
                            results[rate]['MaxEpsilonIneqSuccessProb'][var_method] = temp['MaxEpsilonIneqSuccessProb'][var_method][flow][path] / temp['experiments'] * 100
                        
                        for var_method in temp['MaxEpsilonIneqNonMarkingProb'].keys():
                            results[rate]['MaxEpsilonIneqNonMarkingProb'][var_method] = temp['MaxEpsilonIneqNonMarkingProb'][var_method][flow][path] / temp['experiments'] * 100

                        for var_method in temp['MaxEpsilonIneqLastNonMarkingProb'].keys():
                            results[rate]['MaxEpsilonIneqLastNonMarkingProb'][var_method] = temp['MaxEpsilonIneqLastNonMarkingProb'][var_method][flow][path] / temp['experiments'] * 100
    return results, flows, paths

def plot_boxplot(results, serviceRateScales, results_dir, results_dir_file, metric):
    plt.figure(figsize=(8, 6))
    data = [results[rate][metric] for rate in serviceRateScales]
    plt.boxplot(data, patch_artist=True, widths=0.3)
    plt.xticks(range(1, len(serviceRateScales) + 1), serviceRateScales)
    plt.xlabel("Rate (from high to low congestion)")
    plt.ylabel("{} of Delay".format(metric))
    plt.title("{} of Delay vs Rate".format(metric))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(f"../Results/results_{results_dir}/{results_dir_file}_{metric}_vs_Rate.png")
    plt.close()

def plot_success_per_rate(results, flows, paths, rates, results_dir, results_dir_file):
    for metric in set(k for r in results.values() for k in r.keys()):
        plt.figure(figsize=(8, 6))
        
        sub_keys = set(k for r in results.values() if metric in r for k in r[metric].keys())
        sub_keys = sorted(sub_keys)
        i = 0
        for sub_key in sub_keys:
            y_values = [results[rate].get(metric, {}).get(sub_key, np.nan) for rate in rates]
            plt.plot(rates, y_values, marker='o', label=sub_key, color=colors[i], linewidth=1, markersize=4)
            i += 1
        
        plt.ylim(-5, 110)
        plt.yticks(np.arange(0, 101, 10))
        plt.xlabel("Rate (from high to low congestion)")
        plt.ylabel("Success Rate (%)")
        plt.title(f"{metric}")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.savefig(f"../Results/results_{results_dir}/{results_dir_file}_{metric}_vs_Rate.png")
        plt.close()

def __main__():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dir",
                    required=True,
                    dest="dir",
                    help="The directory of the results",
                   default="")
    # parser.add_argument("--file",
    #                 required=True,
    #                 dest="file",
    #                 help="The file of the results",
    #                default="")
    
    args = parser.parse_args()
    results_dir = args.dir
    # results_dir_file = args.file
    results_dir_file = "Q_e_m_WBias"
    config = configparser.ConfigParser()
    config.read('../Results/results_{}/Parameters.config'.format(args.dir))
    serviceRateScales = [float(x) for x in config.get('Settings', 'serviceRateScales').split(',')]
    results, flows, paths = readResults(results_dir, serviceRateScales, results_dir_file)
    plot_success_per_rate(results, flows, paths, serviceRateScales, results_dir, results_dir_file)
    # plot_boxplot(results, serviceRateScales, results_dir, results_dir_file, metric='SD0Delaystd')
    # plot_boxplot(results, serviceRateScales, results_dir, results_dir_file, metric='SD0DelayMean')
__main__()
