import argparse
import configparser
import os
import json as js
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import numpy as np
import itertools

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k']
def readResults(results_dir, serviceRateScales, results_dir_file, selectedVarMethods, differentiationDelay=0, errorRate=0, load=0, traffic=''):
    results = {}
    dropRate = {}
    sampleSizes = {}
    CVS = {}
    workload = {}
    flows = ['A0D0']
    paths = ["0"]
    for rate in serviceRateScales:
        results[rate] = {}
        dropRate[rate] = {}
        sampleSizes[rate] = {}
        workload[rate] = {}
        CVS[rate] = {}
        if differentiationDelay == 0 and errorRate == 0:
            rate_dir = traffic + "/" + str(rate) + "/" + str(load)
        else:
            rate_dir = traffic + "/" + str(rate) + "/" + str(load) + "/D_" + str(differentiationDelay) + "/f_" + str(errorRate)
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
                        workload[rate] = temp['AverageWorkLoad']
                        CVS[rate]['DelayCV'] = np.mean([temp['SD0Delaystd'][i] / temp['SD0DelayMean'][i] if temp['SD0DelayMean'][i] != 0 else 0 for i in range(temp['experiments'])])
                        # CVS[rate]['LastDelayCV'] = np.mean([temp['SD0LastDelaystd'][i] / temp['SD0LastDelayMean'][i] for i in range(temp['experiments'])])
                        CVS[rate]['SuccessProbCV'] = np.mean([temp['SD0SuccessProbStd'][i] / temp['SD0SuccessProbMean'][i] for i in range(temp['experiments'])])
                        # CVS[rate]['LastSuccessProbCV'] = np.mean([temp['SD0LastSuccessProbStd'][i] / temp['SD0LastSuccessProbMean'][i] for i in range(temp['experiments'])])
                        CVS[rate]['NonMarkingProbCV'] = np.mean([temp['SD0NonMarkingProbStd'][i] / temp['SD0NonMarkingProbMean'][i] for i in range(temp['experiments'])])
                        # CVS[rate]['LastNonMarkingProbCV'] = np.mean([temp['SD0LastNonMarkingProbStd'][i] / temp['SD0LastNonMarkingProbMean'][i] for i in range(temp['experiments'])])
                        # CVS[rate]['SubSamplesDelayCV'] = np.mean([temp['EndToEndDelayMean']['event_poisson_eventAvg'][flow][path][i][1] / temp['EndToEndDelayMean']['event_poisson_eventAvg'][flow][path][i][0] * np.sqrt(temp['EndToEndSampleSizeDelay'][flow][path][i]) for i in range(temp['experiments'])])
                        # CVS[rate]['SubSamplesSuccessProbCV'] = np.mean([temp['EndToEndSuccessProb']['event_poisson_eventAvg'][flow][path][i][1] / temp['EndToEndSuccessProb']['event_poisson_eventAvg'][flow][path][i][0] * np.sqrt(temp['EndToEndSampleSizeSuccess'][flow][path][i]) for i in range(temp['experiments'])])
                        # CVS[rate]['SubSamplesNonMarkingProbCV'] = np.mean([temp['EndToEndNonMarkingProb']['event_poisson_eventAvg'][flow][path][i][1] / temp['EndToEndNonMarkingProb']['event_poisson_eventAvg'][flow][path][i][0] * np.sqrt(temp['EndToEndSampleSizeMarking'][flow][path][i]) for i in range(temp['experiments'])])
                        sampleSizes[rate]['SampleSizeDelay'] = np.mean([temp['EndToEndSampleSizeDelay'][flow][path][i] for i in range(temp['experiments'])])

                        if len(selectedVarMethods) == 0:
                            selectedVarMethods = list(temp['MaxEpsilonIneqDelay'].keys()) + list(temp['MaxEpsilonIneqSuccessProb'].keys()) + list(temp['MaxEpsilonIneqNonMarkingProb'].keys())
                        for var_method in temp['MaxEpsilonIneqDelay'].keys():
                            if var_method not in selectedVarMethods:
                                continue
                            results[rate]['Delay'][var_method] = temp['MaxEpsilonIneqDelay'][var_method][flow][path][0] / temp['MaxEpsilonIneqDelay'][var_method][flow][path][1] * 100 if temp['MaxEpsilonIneqDelay'][var_method][flow][path][1] != 0 else None
                            results[rate]['LastDelay'][var_method] = temp['MaxEpsilonIneqLastDelay'][var_method][flow][path][0] / temp['MaxEpsilonIneqLastDelay'][var_method][flow][path][1] * 100 if temp['MaxEpsilonIneqLastDelay'][var_method][flow][path][1] != 0 else None
                        
                        for var_method in temp['MaxEpsilonIneqSuccessProb'].keys():
                            if var_method not in selectedVarMethods:
                                continue
                            results[rate]['SuccessProb'][var_method] = temp['MaxEpsilonIneqSuccessProb'][var_method][flow][path][0] /temp['MaxEpsilonIneqSuccessProb'][var_method][flow][path][1] * 100 if temp['MaxEpsilonIneqSuccessProb'][var_method][flow][path][1] != 0 else None
                            results[rate]['LastSuccessProb'][var_method] = temp['MaxEpsilonIneqLastSuccessProb'][var_method][flow][path][0] / temp['MaxEpsilonIneqLastSuccessProb'][var_method][flow][path][1] * 100 if temp['MaxEpsilonIneqLastSuccessProb'][var_method][flow][path][1] != 0 else None
                            
                        for var_method in temp['MaxEpsilonIneqNonMarkingProb'].keys():
                            if var_method not in selectedVarMethods:
                                continue
                            results[rate]['NonMarkingProb'][var_method] = temp['MaxEpsilonIneqNonMarkingProb'][var_method][flow][path][0] / temp['MaxEpsilonIneqNonMarkingProb'][var_method][flow][path][1] * 100 if temp['MaxEpsilonIneqNonMarkingProb'][var_method][flow][path][1] != 0 else None

                        for var_method in temp['MaxEpsilonIneqLastNonMarkingProb'].keys():
                            if var_method not in selectedVarMethods:
                                continue
                            results[rate]['LastNonMarkingProb'][var_method] = temp['MaxEpsilonIneqLastNonMarkingProb'][var_method][flow][path][0] / temp['MaxEpsilonIneqLastNonMarkingProb'][var_method][flow][path][1] * 100 if temp['MaxEpsilonIneqLastNonMarkingProb'][var_method][flow][path][1] != 0 else None
    return results, flows, paths, dropRate, CVS, sampleSizes, workload

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

def plot_droprate_vs_load(traffic_list, loads, rates, results_dir, DropRates):
    print("Generating Drop Rate vs Load subplots (subplot per ratio, shape per traffic)...")

    # Compute oversubscription ratios
    oversub_ratios = [1 / r if r != 0 else np.nan for r in rates]
    oversub_ratio_map = dict(zip(rates, oversub_ratios))

    # Assign marker shapes to each traffic
    marker_styles = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'H', '8']
    marker_map = {traffic: marker_styles[i % len(marker_styles)] for i, traffic in enumerate(traffic_list)}

    # Assign colors to traffic (consistent across subplots)
    cmap = get_cmap('tab10') if len(traffic_list) <= 10 else get_cmap('tab20')
    color_map = {traffic: cmap(i / len(traffic_list)) for i, traffic in enumerate(traffic_list)}

    num_ratios = len(rates)
    fig, axs = plt.subplots(nrows=num_ratios, ncols=1, figsize=(20, 7 * num_ratios), sharex=True)

    if num_ratios == 1:
        axs = [axs]  # wrap in list for consistency

    for idx, rate in enumerate(rates):
        alpha = oversub_ratio_map[rate]
        ax = axs[idx]
        for traffic in traffic_list:
            marker = marker_map[traffic]
            color = color_map[traffic]
            x_vals, y_vals = [], []

            for load in sorted(loads):
                drop_val = DropRates[traffic][load].get(rate, np.nan)
                if not np.isnan(drop_val):
                    x_vals.append(load)
                    y_vals.append(drop_val * 100)  # convert to %

            if x_vals:
                ax.plot(
                    x_vals, y_vals,
                    linestyle='-',
                    marker=marker,
                    color=color,
                    markerfacecolor='none',  # hollow
                    markeredgecolor=color,
                    markersize=8,
                    linewidth=1.5,
                    label=traffic
                )

        ax.set_title(f"Drop Rate vs Load (α = {alpha:.2f})", fontsize=18)
        ax.set_ylabel("Drop Rate (%)", fontsize=14)
        ax.set_ylim(bottom=-0.05, top=5)
        ax.set_yticks(np.arange(-0.05, 5.05, 0.1))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}%"))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend(loc='best', fontsize=10, fancybox=True, shadow=True)

    axs[-1].set_xlabel("Offered Load", fontsize=16)
    axs[-1].set_xticks(sorted(loads))
    axs[-1].set_xticklabels([f"{l:.2f}" for l in sorted(loads)], fontsize=12)

    plt.suptitle("Drop Rate vs Load per Oversubscription Ratio", fontsize=24)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"../Results/results_{results_dir}/DropRate_vs_Load_Subplots.png")
    plt.close()


def plot_forward_success_per_loads_traffic(results, loads, rates, results_dir, results_dir_file, selectedVarMethod):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.cm import get_cmap

    print("Generating forward success rate subplots per load (1 per α, shape per traffic)...")

    oversub_ratio_map = {r: 1 / r if r != 0 else np.nan for r in rates}
    traffic_list = list(results.keys())

    # Assign distinct marker per traffic
    marker_styles = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'H', '8']
    marker_map = {traffic: marker_styles[i % len(marker_styles)] for i, traffic in enumerate(traffic_list)}

    # Assign color per traffic (consistent across subplots)
    cmap = get_cmap('tab10') if len(traffic_list) <= 10 else get_cmap('tab20')
    color_map = {traffic: cmap(i / len(traffic_list)) for i, traffic in enumerate(traffic_list)}

    # Determine all relevant metrics
    all_metrics = set(
        k for traffic in results.values()
          for load_dict in traffic.values()
          for rate_dict in load_dict.values()
          for k in rate_dict.keys()
    )

    for metric in all_metrics:
        if len(selectedVarMethod) == 0:
            if 'success' in metric.lower():
                selectedVarMethod_ = 'probability_linearInterp_timeAvg'
            else:
                selectedVarMethod_ = 'event_linearInterp_timeAvg'
        else:
            selectedVarMethod_ = selectedVarMethod[0]

        num_ratios = len(rates)
        fig, axs = plt.subplots(nrows=num_ratios, ncols=1, figsize=(20, 5 * num_ratios), sharex=True)

        if num_ratios == 1:
            axs = [axs]  # make iterable if only one subplot

        for ax_idx, rate in enumerate(rates):
            alpha = oversub_ratio_map[rate]
            ax = axs[ax_idx]
            for traffic in traffic_list:
                marker = marker_map[traffic]
                color = color_map[traffic]
                x_vals = []
                y_vals = []
                for load in sorted(loads):
                    try:
                        val = results[traffic][load].get(rate, {}).get(metric, {}).get(selectedVarMethod_, np.nan)
                        if not np.isnan(val):
                            x_vals.append(load)
                            y_vals.append(val)
                    except:
                        continue

                if x_vals:
                    ax.plot(
                        x_vals, y_vals,
                        linestyle='-',
                        marker=marker,
                        color=color,
                        markerfacecolor='none',  # hollow
                        markeredgecolor=color,
                        markersize=8,
                        linewidth=1.5,
                        label=traffic
                    )

            ax.set_title(f"{metric} Success Rate vs Load (α = {alpha:.2f})", fontsize=18)
            ax.set_ylabel("Success Rate (%)", fontsize=14)
            ax.set_ylim(-5, 110)
            ax.set_yticks(np.arange(0, 101, 10))
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.legend(loc='best', fontsize=10, fancybox=True, shadow=True)

        axs[-1].set_xlabel("Offered Load", fontsize=16)
        axs[-1].set_xticks(sorted(loads))
        axs[-1].set_xticklabels([f"{l:.2f}" for l in sorted(loads)], fontsize=12)

        plt.suptitle(f"{metric} Success Rate vs Offered Load per Oversubscription Ratio", fontsize=24)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"../Results/results_{results_dir}/{results_dir_file}_{metric}_SuccessRate_vs_Load_Subplots.png")
        plt.close()


def plot_success_vs_loads(results, loads, rates, results_dir, results_dir_file, selectedVarMethod):
    for metric in set(
        k for load_dict in results.values()
          for rate_dict in load_dict.values()
          for k in rate_dict.keys()
    ):
        print(f"Plotting {metric} vs Load...")
        plt.figure(figsize=(20, 14))
        ax = plt.gca()

        # One line per rate
        for i, rate in enumerate(rates):
            y_values = [
                results[load].get(rate, {})
                              .get(metric, {})
                              .get(selectedVarMethod, np.nan)
                for load in loads
            ]
            plt.plot(loads, y_values, marker='o', label=f"α={1/rate:.2f}" if rate != 0 else "α=NaN",
                     color=colors[i % len(colors)], linewidth=1, markersize=4)

        # Primary x-axis: Load values
        ax.set_xticks(loads)
        ax.set_xticklabels([f"{l:.2f}" for l in loads], rotation=45, fontsize=15)
        ax.set_xlabel("Load", fontsize=20)

        # Y-axis: Success rate
        ax.set_ylim(-5, 110)
        ax.set_yticks(np.arange(0, 101, 10))
        ax.set_ylabel("Success Rate (%)", fontsize=20)

        # Secondary x-axis (top): Drop rates per load
        drop_rates_per_load = [
            np.mean([results[load]['DropRate'][rate] for rate in rates if rate in results[load]]) 
            if 'DropRate' in results[load] else np.nan
            for load in loads
        ]
        ax_top = ax.secondary_xaxis('top')
        ax_top.set_xticks(loads)
        ax_top.set_xticklabels([f"{drop*100:.4f}%" if not np.isnan(drop) else "NaN" for drop in drop_rates_per_load],
                               rotation=90, fontsize=15)
        ax_top.set_xlabel("Drop Rate", fontsize=20)

        # Plot title and legend
        plt.title(f"{metric} Success Rate vs Offered Load", fontsize=22)
        plt.legend(loc='lower right', ncol=4, fancybox=True, shadow=True, prop={'size': 10}, title="Oversubscription Ratio (α)")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.subplots_adjust(left=0.05, right=0.95)
        plt.savefig(f"../Results/results_{results_dir}/{results_dir_file}_{metric}_vs_Load.png")
        plt.close()


def analyse_forward_exp(results_dir, results_dir_file, rateScales, loads, selectedVarMethods, traffics):
    results = {}
    DropRates = {}
    workload = {}
    for traffic in traffics:
        results[traffic] = {}
        DropRates[traffic] = {}
        workload[traffic] = {}
        for load in loads:
            results[traffic][load] = {}
            DropRates[traffic][load] = {}
            workload[traffic][load] = {}
            results[traffic][load], flows, paths, DropRates[traffic][load], CVS, sampleSizes, workload[traffic][load] = readResults(results_dir, rateScales, results_dir_file, selectedVarMethods, load=load, traffic=traffic)
            # results[traffic][load]['DropRate'] = DropRates[traffic][load]
    selectedRates = rateScales
    plot_forward_success_per_loads_traffic(results, loads, selectedRates, results_dir, results_dir_file, selectedVarMethods)
    # plot_droprate_vs_load(results, loads, selectedRates, results_dir, DropRates)

def analyse_reverse_exp(results_dir, results_dir_file, rateScales, differentiationDelays, errorRates, selectedVarMethods, type, traffics):
    results = {}
    DropRates = {}
    for differentiationDelay in differentiationDelays:
        results[differentiationDelay] = {}
        DropRates[differentiationDelay] = {}
        for errorRate in errorRates:
            results[differentiationDelay][errorRate] = {}
            DropRates[differentiationDelay][errorRate] = {}
            results[differentiationDelay][errorRate], flows, paths, DropRates[differentiationDelay][errorRate], CVS, sampleSizes, _ = readResults(results_dir, rateScales, results_dir_file, selectedVarMethods, differentiationDelay, errorRate)
    selectedRates = rateScales
    plot_success_vs_errorRate(list(results.keys()), differentiationDelays, selectedRates, results_dir, results_dir_file, selectedVarMethods[0], type)
        
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
    results_dir_file = "Q_thinning_0.5"
    config = configparser.ConfigParser()
    config.read('../Results/results_{}/Parameters.config'.format(args.dir))
    rateScales = [float(x) for x in config.get('Settings', 'serviceRateScales').split(',')]
    loads = [float(x) for x in config.get('Settings', 'load').split(',')]
    traffics = config.get('Settings', 'traffic').split(',')
    # traffics = ["Google_AllRPC", "Fabricated_Heavy_Head", "Fabricated_Heavy_Middle", "Google_SearchRPC", "Facebook_HadoopDist_All"]
    # experiments = 1
    errorRates = [float(x) for x in config.get('Settings', 'errorRate').split(',')]
    # errorRates = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    differentiationDelays = [float(x) for x in config.get('Settings', 'differentiationDelay').split(',')]
    # differentiationDelays = [0.35]
    # selectedVarMethods = ['event_poisson_eventAvg']
    selectedVarMethods = []
    # print(RateScales)
    # serviceRateScales = [0.75, 0.80, 0.85, 0.90, 0.95, 1.0, 1.05]
    if args.IsForward == 1:
        analyse_forward_exp(results_dir, results_dir_file, rateScales, loads, selectedVarMethods, traffics)
        # results, flows, paths, DropRates, CVS, sampleSizes = readResults(results_dir, rateScales, results_dir_file, selectedVarMethods)
        # # plot_success_per_rate(results, flows, paths, RateScales, results_dir, results_dir_file)
        # plot_success_per_dropRates(results, rateScales, results_dir, results_dir_file, DropRates.values())
        # plot_CV_perRate(rateScales, results_dir, results_dir_file, CVS, DropRates.values())
        # plot_SampleSize_perRate(rateScales, results_dir, results_dir_file, sampleSizes, DropRates.values())
        # plot_CV_perRate(results, serviceRateScales, results_dir, results_dir_file, metric='SuccessProb')
        # plot_CV_perRate(results, serviceRateScales, results_dir, results_dir_file, metric='NonMarkingProb')
        # plot_boxplot(results, serviceRateScales, results_dir, results_dir_file, metric='SD0Delaystd')
        # plot_boxplot(results, serviceRateScales, results_dir, results_dir_file, metric='SD0DelayMean')
    else:
        analyse_reverse_exp(results_dir, results_dir_file, rateScales, differentiationDelays, errorRates, selectedVarMethods, args.type, traffics)
        

__main__()
