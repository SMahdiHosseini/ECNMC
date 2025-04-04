import argparse
import configparser
import os
import json as js
import matplotlib.pyplot as plt
import numpy as np

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k']
def readResults(results_dir, serviceRateScales, results_dir_file):
    results = {}
    dropRate = {}
    flows = ['A0D0']
    paths = ["0"]
    for rate in serviceRateScales:
        results[rate] = {}
        dropRate[rate] = {}
        for file in os.listdir('../Results/results_' + results_dir + '/'+str(rate)+'/'):
            if file.find(results_dir_file) != -1:
                temp = {}
                if file.endswith('.json'):
                    with open('../Results/results_' + results_dir + '/'+str(rate)+'/'+file) as f:
                        temp = js.load(f)
                else:   
                    continue
                dropRate[rate] = np.mean(temp['DropRate'])
                for flow in flows:
                    for path in paths:
                        results[rate]['Delay'] = {}
                        results[rate]['LastDelay'] = {}
                        results[rate]['SuccessProb'] = {}
                        results[rate]['LastSuccessProb'] = {}
                        results[rate]['NonMarkingProb'] = {}
                        results[rate]['LastNonMarkingProb'] = {}
                        # results[rate]['SD0DelayStd'] = temp['SD0Delaystd']
                        # results[rate]['SD0DelayMean'] = temp['SD0DelayMean']
                        # results[rate]['SD0SuccessProbStd'] = temp['SD0SuccessProbStd']
                        # results[rate]['SD0SuccessProbMean'] = temp['SD0SuccessProbMean']
                        # results[rate]['SD0NonMarkingProbStd'] = temp['SD0NonMarkingProbStd']
                        # results[rate]['SD0NonMarkingProbMean'] = temp['SD0NonMarkingProbMean']

                        for var_method in temp['MaxEpsilonIneqDelay'].keys():
                            if var_method == 'event_eventAvg':
                                continue
                            results[rate]['Delay'][var_method] = temp['MaxEpsilonIneqDelay'][var_method][flow][path] / temp['experiments'] * 100
                            results[rate]['LastDelay'][var_method] = temp['MaxEpsilonIneqLastDelay'][var_method][flow][path] / temp['experiments'] * 100
                        
                        for var_method in temp['MaxEpsilonIneqSuccessProb'].keys():
                            if var_method == 'event_eventAvg' or var_method == 'probability_eventAvg':
                                continue
                            results[rate]['SuccessProb'][var_method] = temp['MaxEpsilonIneqSuccessProb'][var_method][flow][path] / temp['experiments'] * 100
                            results[rate]['LastSuccessProb'][var_method] = temp['MaxEpsilonIneqLastSuccessProb'][var_method][flow][path] / temp['experiments'] * 100
                            
                        for var_method in temp['MaxEpsilonIneqNonMarkingProb'].keys():
                            if var_method == 'event_eventAvg':
                                continue
                            results[rate]['NonMarkingProb'][var_method] = temp['MaxEpsilonIneqNonMarkingProb'][var_method][flow][path] / temp['experiments'] * 100

                        for var_method in temp['MaxEpsilonIneqLastNonMarkingProb'].keys():
                            if var_method == 'event_eventAvg':
                                continue
                            results[rate]['LastNonMarkingProb'][var_method] = temp['MaxEpsilonIneqLastNonMarkingProb'][var_method][flow][path] / temp['experiments'] * 100
    return results, flows, paths, dropRate

def plot_CV_perRate(results, serviceRateScales, results_dir, results_dir_file, metric='Delay'):
    plt.figure(figsize=(8, 6)) 
    data = [np.mean(np.array(results[rate]['SD0' + metric + 'Std']) / np.array(results[rate]['SD0' + metric + 'Mean'])) for rate in serviceRateScales]
    plt.scatter(serviceRateScales, data, marker='o', label=metric, color='b', linewidth=1)
    plt.ylim(-0.05, max(data) * (1.05))
    plt.xlabel("Rate (from high to low congestion)")
    plt.ylabel("CV of {}".format(metric))
    plt.title("CV of {} vs Rate".format(metric))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(f"../Results/results_{results_dir}/CV_{results_dir_file}_{metric}_vs_Rate.png")

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
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, fancybox=True, shadow=True, prop={'size': 6})
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.savefig(f"../Results/results_{results_dir}/{results_dir_file}_{metric}_vs_Rate.png")
        plt.close()

def plot_success_per_dropRates(results, flows, paths, rates, results_dir, results_dir_file, DropRates):
    for metric in set(k for r in results.values() for k in r.keys()):
        plt.figure(figsize=(10, 9))
        
        sub_keys = set(k for r in results.values() if metric in r for k in r[metric].keys())
        sub_keys = sorted(sub_keys)
        i = 0
        for sub_key in sub_keys:
            y_values = [results[rate].get(metric, {}).get(sub_key, np.nan) for rate in rates]
            plt.plot(rates, y_values, marker='o', label=sub_key, color=colors[i], linewidth=1, markersize=4)
            i += 1

        xtick_labels = [f"{rate:.2f} ({drop:.3f})" for rate, drop in zip(rates, DropRates)]
        plt.xticks(rates, labels=xtick_labels, rotation=45, size=10)

        plt.ylim(-5, 110)
        plt.yticks(np.arange(0, 101, 10))
        plt.xlabel("Rate (from high to low congestion)(Drop Rate)")
        plt.ylabel("Success Rate (%)")
        plt.title(f"{metric}")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, fancybox=True, shadow=True, prop={'size': 6})
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.savefig(f"../Results/results_{results_dir}/{results_dir_file}_{metric}_vs_DropRate.png")
        plt.close()

def __main__():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dir",
                    required=True,
                    dest="dir",
                    help="The directory of the results",
                   default="")
    parser.add_argument("--IsForward",
                    required=True, 
                    dest="IsForward",
                    help="If the experiment is the straitforward experiment or the reverse experiment!", 
                    type=int,
                    default=1)
    
    args = parser.parse_args()
    results_dir = args.dir
    # results_dir_file = args.file
    results_dir_file = "Q_e_m_forward"
    config = configparser.ConfigParser()
    config.read('../Results/results_{}/Parameters.config'.format(args.dir))
    if args.IsForward == 1:
        RateScales = [float(x) for x in config.get('Settings', 'serviceRateScales').split(',')]
    else:
        RateScales = [float(x) for x in config.get('Settings', 'errorRateScale').split(',')]
    # print(RateScales)
    # serviceRateScales = [0.75, 0.80, 0.85, 0.90, 0.95, 1.0, 1.05]
    results, flows, paths, DropRates = readResults(results_dir, RateScales, results_dir_file)
    plot_success_per_rate(results, flows, paths, RateScales, results_dir, results_dir_file)
    plot_success_per_dropRates(results, flows, paths, RateScales, results_dir, results_dir_file, DropRates.values())
    # plot_CV_perRate(results, serviceRateScales, results_dir, results_dir_file, metric='Delay')
    # plot_CV_perRate(results, serviceRateScales, results_dir, results_dir_file, metric='SuccessProb')
    # plot_CV_perRate(results, serviceRateScales, results_dir, results_dir_file, metric='NonMarkingProb')
    # plot_boxplot(results, serviceRateScales, results_dir, results_dir_file, metric='SD0Delaystd')
    # plot_boxplot(results, serviceRateScales, results_dir, results_dir_file, metric='SD0DelayMean')
__main__()
