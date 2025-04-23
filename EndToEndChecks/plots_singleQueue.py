import argparse
import configparser
import os
import json as js
import matplotlib.pyplot as plt
import numpy as np

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k']
def readResults(results_dir, serviceRateScales, results_dir_file, selectedVarMethods, differentiationDelay=0, errorRate=0):
    results = {}
    dropRate = {}
    sampleSizes = {}
    CVS = {}
    flows = ['A0D0']
    paths = ["0"]
    for rate in serviceRateScales:
        results[rate] = {}
        dropRate[rate] = {}
        sampleSizes[rate] = {}
        CVS[rate] = {}
        if differentiationDelay == 0 and errorRate == 0:
            rate_dir = str(rate)
        else:
            rate_dir = str(rate) + "/D_" + str(differentiationDelay) + "/f_" + str(errorRate)
        for file in os.listdir('../Results/results_' + results_dir + '/' + rate_dir):
            if file.find(results_dir_file) != -1:
                temp = {}
                if file.endswith('.json'):
                    with open('../Results/results_' + results_dir + '/' + rate_dir + '/'+file) as f:
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
                        CVS[rate]['DelayCV'] = np.mean([temp['SD0Delaystd'][i] / temp['SD0DelayMean'][i] for i in range(temp['experiments'])])
                        # CVS[rate]['LastDelayCV'] = np.mean([temp['SD0LastDelaystd'][i] / temp['SD0LastDelayMean'][i] for i in range(temp['experiments'])])
                        CVS[rate]['SuccessProbCV'] = np.mean([temp['SD0SuccessProbStd'][i] / temp['SD0SuccessProbMean'][i] for i in range(temp['experiments'])])
                        # CVS[rate]['LastSuccessProbCV'] = np.mean([temp['SD0LastSuccessProbStd'][i] / temp['SD0LastSuccessProbMean'][i] for i in range(temp['experiments'])])
                        CVS[rate]['NonMarkingProbCV'] = np.mean([temp['SD0NonMarkingProbStd'][i] / temp['SD0NonMarkingProbMean'][i] for i in range(temp['experiments'])])
                        # CVS[rate]['LastNonMarkingProbCV'] = np.mean([temp['SD0LastNonMarkingProbStd'][i] / temp['SD0LastNonMarkingProbMean'][i] for i in range(temp['experiments'])])
                        # CVS[rate]['SubSamplesDelayCV'] = np.mean([temp['EndToEndDelayMean']['event_poisson_eventAvg'][flow][path][i][1] / temp['EndToEndDelayMean']['event_poisson_eventAvg'][flow][path][i][0] * np.sqrt(temp['EndToEndSampleSizeDelay'][flow][path][i]) for i in range(temp['experiments'])])
                        # CVS[rate]['SubSamplesSuccessProbCV'] = np.mean([temp['EndToEndSuccessProb']['event_poisson_eventAvg'][flow][path][i][1] / temp['EndToEndSuccessProb']['event_poisson_eventAvg'][flow][path][i][0] * np.sqrt(temp['EndToEndSampleSizeSuccess'][flow][path][i]) for i in range(temp['experiments'])])
                        # CVS[rate]['SubSamplesNonMarkingProbCV'] = np.mean([temp['EndToEndNonMarkingProb']['event_poisson_eventAvg'][flow][path][i][1] / temp['EndToEndNonMarkingProb']['event_poisson_eventAvg'][flow][path][i][0] * np.sqrt(temp['EndToEndSampleSizeMarking'][flow][path][i]) for i in range(temp['experiments'])])
                        # sampleSizes[rate]['SampleSizeDelay'] = np.mean([temp['EndToEndSampleSizeDelay'][flow][path][i] for i in range(temp['experiments'])])


                        for var_method in temp['MaxEpsilonIneqDelay'].keys():
                            if var_method not in selectedVarMethods:
                                continue
                            results[rate]['Delay'][var_method] = temp['MaxEpsilonIneqDelay'][var_method][flow][path] / temp['experiments'] * 100
                            results[rate]['LastDelay'][var_method] = temp['MaxEpsilonIneqLastDelay'][var_method][flow][path] / temp['experiments'] * 100
                        
                        for var_method in temp['MaxEpsilonIneqSuccessProb'].keys():
                            if var_method not in selectedVarMethods:
                                continue
                            results[rate]['SuccessProb'][var_method] = temp['MaxEpsilonIneqSuccessProb'][var_method][flow][path] / temp['experiments'] * 100
                            results[rate]['LastSuccessProb'][var_method] = temp['MaxEpsilonIneqLastSuccessProb'][var_method][flow][path] / temp['experiments'] * 100
                            
                        for var_method in temp['MaxEpsilonIneqNonMarkingProb'].keys():
                            if var_method not in selectedVarMethods:
                                continue
                            results[rate]['NonMarkingProb'][var_method] = temp['MaxEpsilonIneqNonMarkingProb'][var_method][flow][path] / temp['experiments'] * 100

                        for var_method in temp['MaxEpsilonIneqLastNonMarkingProb'].keys():
                            if var_method not in selectedVarMethods:
                                continue
                            results[rate]['LastNonMarkingProb'][var_method] = temp['MaxEpsilonIneqLastNonMarkingProb'][var_method][flow][path] / temp['experiments'] * 100
    return results, flows, paths, dropRate, CVS, sampleSizes

def plot_CV_perRate(serviceRateScales, results_dir, results_dir_file, CVS, DropRates):
    oversub_ratios = [1 / r if r != 0 else np.nan for r in serviceRateScales]
    for metric in set(k for r in CVS.values() for k in r.keys()):
        print(f"Plotting {metric}...")
        plt.figure(figsize=(20, 14))
        ax = plt.gca()
        data = [CVS[rate][metric] for rate in serviceRateScales]
        plt.scatter(oversub_ratios, data, marker='o', label=metric, color='b', linewidth=1)
        # Primary x-axis: Oversubscription ratios
        ax.set_xticks(oversub_ratios)
        ax.set_xticklabels([f"{alpha:.2f}" for alpha in oversub_ratios], rotation=45, fontsize=15)
        ax.set_xlabel("Oversubscription Ratio (α)", fontsize=20)

        # Y-axis
        plt.ylim(-0.05, max(data) * (1.05))
        ax.set_yticks(np.arange(-0.05, max(data) * (1.05), 0.05))
        ax.set_ylabel(f"{metric}", fontsize=20)

        # Secondary x-axis (top): Drop rates
        ax_top = ax.secondary_xaxis('top')
        ax_top.set_xticks(oversub_ratios)
        ax_top.set_xticklabels([f"{drop*100:.4f}%" for drop in DropRates], rotation=90, fontsize=15)
        ax_top.set_xlabel("Drop Rate", fontsize=20)

        plt.title("{} vs Rate".format(metric))
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.savefig(f"../Results/results_{results_dir}/{results_dir_file}_{metric}_vs_Rate.png")

def plot_SampleSize_perRate(serviceRateScales, results_dir, results_dir_file, sampleSizes, DropRates):
    oversub_ratios = [1 / r if r != 0 else np.nan for r in serviceRateScales]
    for metric in set(k for r in sampleSizes.values() for k in r.keys()):
        print(f"Plotting {metric}...")
        plt.figure(figsize=(20, 14))
        ax = plt.gca()
        data = [sampleSizes[rate][metric] for rate in serviceRateScales]
        plt.scatter(oversub_ratios, data, marker='o', label=metric, color='b', linewidth=1)
        # Primary x-axis: Oversubscription ratios
        ax.set_xticks(oversub_ratios)
        ax.set_xticklabels([f"{alpha:.2f}" for alpha in oversub_ratios], rotation=45, fontsize=15)
        ax.set_xlabel("Oversubscription Ratio (α)", fontsize=20)

        # Y-axis
        plt.ylim(min(data) * 0.95, max(data) * (1.05))
        ax.set_yticks(np.arange(min(data) * 0.95, max(data) * (1.05), 5))
        ax.set_ylabel(f"{metric}", fontsize=20)

        # Secondary x-axis (top): Drop rates
        ax_top = ax.secondary_xaxis('top')
        ax_top.set_xticks(oversub_ratios)
        ax_top.set_xticklabels([f"{drop*100:.4f}%" for drop in DropRates], rotation=90, fontsize=15)
        ax_top.set_xlabel("Drop Rate", fontsize=20)

        plt.title("{} vs Rate".format(metric))
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.savefig(f"../Results/results_{results_dir}/{results_dir_file}_{metric}_vs_Rate.png")

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

def plot_success_per_dropRates(results, rates, results_dir, results_dir_file, DropRates):
    oversub_ratios = [1 / r if r != 0 else np.nan for r in rates]
    for metric in set(k for r in results.values() for k in r.keys()):
        print(f"Plotting {metric}...")
        plt.figure(figsize=(20, 14))
        ax = plt.gca()
        # Prepare and sort sub_keys
        sub_keys = set(k for r in results.values() if metric in r for k in r[metric].keys())
        sub_keys = sorted(sub_keys)
        i = 0
        for sub_key in sub_keys:
            y_values = [results[rate].get(metric, {}).get(sub_key, np.nan) for rate in rates]
            plt.plot(oversub_ratios, y_values, marker='o', label=sub_key,
                    color=colors[i], linewidth=1, markersize=4)
            i += 1

        # Primary x-axis: Oversubscription ratios
        ax.set_xticks(oversub_ratios)
        ax.set_xticklabels([f"{alpha:.2f}" for alpha in oversub_ratios], rotation=45, fontsize=15)
        ax.set_xlabel("Oversubscription Ratio (α)", fontsize=20)

        # Y-axis
        ax.set_ylim(-5, 110)
        ax.set_yticks(np.arange(0, 101, 10))
        ax.set_ylabel("Success Rate (%)", fontsize=20)

        # Secondary x-axis (top): Drop rates
        ax_top = ax.secondary_xaxis('top')
        ax_top.set_xticks(oversub_ratios)
        ax_top.set_xticklabels([f"{drop*100:.4f}%" for drop in DropRates], rotation=90, fontsize=15)
        ax_top.set_xlabel("Drop Rate", fontsize=20)

        # Plot title and legend
        plt.title(f"{metric} success rate vs Oversubscription", fontsize=20)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, fancybox=True, shadow=True, prop={'size': 10})
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.subplots_adjust(left=0.05, right=0.95)
        plt.savefig(f"../Results/results_{results_dir}/{results_dir_file}_{metric}_vs_DropRate.png")
        plt.close()

def analyse_reverse_exp(results_dir, results_dir_file, rateScales, differentiationDelays, errorRates, selectedVarMethods, type):
    results = {}
    DropRates = {}
    for differentiationDelay in differentiationDelays:
        results[differentiationDelay] = {}
        DropRates[differentiationDelay] = {}
        for errorRate in errorRates:
            results[differentiationDelay][errorRate] = {}
            DropRates[differentiationDelay][errorRate] = {}
            results[differentiationDelay][errorRate], flows, paths, DropRates[differentiationDelay][errorRate], CVS, sampleSizes = readResults(results_dir, rateScales, results_dir_file, selectedVarMethods, differentiationDelay, errorRate)
    selectedRates = rateScales
    plot_success_vs_errorRate(results, differentiationDelays, selectedRates, results_dir, results_dir_file, selectedVarMethods[0], type)
        
def plot_success_vs_errorRate(results, differentiationDelays, rates, results_dir, results_dir_file, selectedVarMethods, type):
    for differentiationDelay in differentiationDelays:
        for metric in set(
            k for error_dict in results[differentiationDelay].values()
              for rate_dict in error_dict.values()
              for k in rate_dict.keys()
        ):
            print(f"Plotting {metric} for differentiationDelay={differentiationDelay}...")
            plt.figure(figsize=(20, 14))
            ax = plt.gca()
            ax.set_prop_cycle(color=['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d','#17becf', '#9edae5'])
            # One line per rate
            # Prepare and sort sub_keys
            for i, rate in enumerate(rates):
                error_rate_list = sorted(results[differentiationDelay].keys())
                oversub_ratio = 1 / rate if rate != 0 else np.nan
                y_values = [
                    100 - results[differentiationDelay][errorRate]
                           .get(rate, {})
                           .get(metric, {})
                           .get(selectedVarMethods, np.nan)
                    for errorRate in error_rate_list
                ]
                # print(f"Rate: {rate}, Error Rate: {error_rate_list}, Y Values: {y_values}")
                plt.plot(error_rate_list, y_values, marker='o', label=f"α={oversub_ratio:.2f}", linewidth=1, markersize=4)

            # x-axis: Error rate
            ax.set_xticks(error_rate_list)
            ax.set_xticklabels([f"{e * 100:.3f}%" for e in error_rate_list], rotation=45, fontsize=15)
            if type == 'loss':
                ax.set_xlabel("Silent Packet Drop Rate(%)", fontsize=20)
            else:
                ax.set_xlabel("Fraction of Packets with Extra Delay(%)", fontsize=20)

            # Y-axis: Success rate
            ax.set_ylim(-5, 110)
            ax.set_yticks(np.arange(0, 101, 10))
            ax.set_ylabel("Inconsistency Success Rate (%)", fontsize=20)

            # Title and legend
            plt.title(f"{metric} vs Error Rate (Differentiation Delay = {differentiationDelay})", fontsize=22)
            plt.legend(loc='lower right', ncol=4, fancybox=True, shadow=True, prop={'size': 10}, title="Oversubscription Ratio (α)")
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.subplots_adjust(left=0.05, right=0.95)
            plt.savefig(f"../Results/results_{results_dir}/{results_dir_file}_{metric}_vs_ErrorRate_Delay_{differentiationDelay}.png")
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
    parser.add_argument("--type",
                    required=False,
                    dest="type",
                    help="If the reverse experiment is the loss or delay experiment!",
                    type=str,
                    default="loss")
    args = parser.parse_args()
    results_dir = args.dir
    # results_dir_file = args.file
    results_dir_file = "Q_e_m_WBias_sigma_subsampling"
    config = configparser.ConfigParser()
    config.read('../Results/results_{}/Parameters.config'.format(args.dir))
    rateScales = [float(x) for x in config.get('Settings', 'serviceRateScales').split(',')]
    # experiments = 1
    errorRates = [float(x) for x in config.get('Settings', 'errorRate').split(',')]
    # errorRates = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    differentiationDelays = [float(x) for x in config.get('Settings', 'differentiationDelay').split(',')]
    # differentiationDelays = [0.35]
    selectedVarMethods = ['event_poisson_eventAvg']
    # print(RateScales)
    # serviceRateScales = [0.75, 0.80, 0.85, 0.90, 0.95, 1.0, 1.05]
    if args.IsForward == 1:
        results, flows, paths, DropRates, CVS, sampleSizes = readResults(results_dir, rateScales, results_dir_file, selectedVarMethods)
        # plot_success_per_rate(results, flows, paths, RateScales, results_dir, results_dir_file)
        plot_success_per_dropRates(results, rateScales, results_dir, results_dir_file, DropRates.values())
        # plot_CV_perRate(rateScales, results_dir, results_dir_file, CVS, DropRates.values())
        # plot_SampleSize_perRate(rateScales, results_dir, results_dir_file, sampleSizes, DropRates.values())
        # plot_CV_perRate(results, serviceRateScales, results_dir, results_dir_file, metric='SuccessProb')
        # plot_CV_perRate(results, serviceRateScales, results_dir, results_dir_file, metric='NonMarkingProb')
        # plot_boxplot(results, serviceRateScales, results_dir, results_dir_file, metric='SD0Delaystd')
        # plot_boxplot(results, serviceRateScales, results_dir, results_dir_file, metric='SD0DelayMean')
    else:
        analyse_reverse_exp(results_dir, results_dir_file, rateScales, differentiationDelays, errorRates, selectedVarMethods, args.type)
        

__main__()
