import argparse
import configparser
import os
import json as js
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import numpy as np

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k']
flows = ["T0A0", "A0T2", "T2H0"]
switches = ["T0A0", "A0T2", "T2H0"]
def readResults(results_dir, serviceRateScales, results_dir_file, selectedVarMethods, differentiationDelay=0, errorRate=0, load=0, traffic=''):
    e2e_results_WOBias = {}
    e2e_delay = {}
    e2e_error = {}
    dropRate = {}
    switch_errors_bounds = {}
    switch_stds = {}
    switch_delay = {}
    switch_queueOccupancy = {}
    switch_packtsInQueue = {}
    switch_emptyFrac = {}
    for rate in serviceRateScales:
        e2e_results_WOBias[rate] = {}
        e2e_delay[rate] = {}
        e2e_error[rate] = {}
        dropRate[rate] = {}
        switch_delay[rate] = {}
        switch_queueOccupancy[rate] = {}
        switch_packtsInQueue[rate] = {}
        switch_emptyFrac[rate] = {}
        switch_stds[rate] = {}
        switch_errors_bounds[rate] = {}
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
                print('../Results/results_' + results_dir + '/' + rate_dir + '/'+file)
                dropRate[rate] = np.mean(temp['DropRate'])
                for flow in flows:
                    e2e_delay[rate][flow] = np.mean([temp['EndToEndDelayMean']['event_linearInterp_timeAvg'][flow]["0"][0][i] for i in range(temp['EndToEndDelayMean']['event_linearInterp_timeAvg'][flow]["0"][1])])

                    e2e_results_WOBias[rate][flow] = {}
                    e2e_results_WOBias[rate][flow]['Delay'] = {}
                    e2e_results_WOBias[rate][flow]['SuccessProb'] = {}
                    e2e_results_WOBias[rate][flow]['NonMarkingProb'] = {}
                    e2e_error[rate][flow] = {}
                    e2e_error[rate][flow]['Delay'] = {}
                    e2e_error[rate][flow]['SuccessProb'] = {}
                    e2e_error[rate][flow]['NonMarkingProb'] = {}

                    if len(selectedVarMethods) == 0:
                        selectedVarMethods = list(temp['MaxEpsilonIneqDelay'].keys()) + list(temp['MaxEpsilonIneqSuccessProb'].keys()) + list(temp['MaxEpsilonIneqNonMarkingProb'].keys())
                    for var_method in temp['MaxEpsilonIneqDelay'].keys():
                        if var_method not in selectedVarMethods:
                            continue
                        e2e_results_WOBias[rate][flow]['Delay'][var_method] = temp['MaxEpsilonIneqDelay'][var_method][flow]["0"][0]['WOBias'] / temp['MaxEpsilonIneqDelay'][var_method][flow]["0"][1] * 100 if temp['MaxEpsilonIneqDelay'][var_method][flow]["0"][1] != 0 else None
                        e2e_error[rate][flow]['Delay'] = np.mean([abs(temp[flow + 'DelayMean'][i] - temp['EndToEndDelayMean'][var_method][flow]["0"][0][i]) for i in range(temp['EndToEndDelayMean'][var_method][flow]["0"][1])])
                    for var_method in temp['MaxEpsilonIneqSuccessProb'].keys():
                        if var_method not in selectedVarMethods:
                            print(f"Skipping var_method {var_method} for SuccessProb as it is not in selectedVarMethods")
                            continue
                        e2e_results_WOBias[rate][flow]['SuccessProb'][var_method] = temp['MaxEpsilonIneqSuccessProb'][var_method][flow]["0"][0]['WOBias'] /temp['MaxEpsilonIneqSuccessProb'][var_method][flow]["0"][1] * 100 if temp['MaxEpsilonIneqSuccessProb'][var_method][flow]["0"][1] != 0 else None
                        e2e_error[rate][flow]['SuccessProb'] = np.mean([abs(temp[flow + 'SuccessProbMean'][i] - temp['EndToEndSuccessProb'][var_method][flow]["0"][0][i]) for i in range(temp['EndToEndSuccessProb'][var_method][flow]["0"][1])])
                    for var_method in temp['MaxEpsilonIneqNonMarkingProb'].keys():
                        if var_method not in selectedVarMethods:
                            continue
                        e2e_results_WOBias[rate][flow]['NonMarkingProb'][var_method] = temp['MaxEpsilonIneqNonMarkingProb'][var_method][flow]["0"][0]['WOBias'] / temp['MaxEpsilonIneqNonMarkingProb'][var_method][flow]["0"][1] * 100 if temp['MaxEpsilonIneqNonMarkingProb'][var_method][flow]["0"][1] != 0 else None
                        e2e_error[rate][flow]['NonMarkingProb'] = np.mean([abs(temp[flow + 'NonMarkingProbMean'][i] - temp['EndToEndNonMarkingProb'][var_method][flow]["0"][0][i]) for i in range(temp['EndToEndNonMarkingProb'][var_method][flow]["0"][1])])
                for switch in switches:
                    switch_delay[rate][switch] = np.mean(temp[switch + 'DelayMean'])


                    switch_errors_bounds[rate][switch] = {}
                    switch_errors_bounds[rate][switch]['Delay'] = np.mean([(temp[switch + 'Delaystd'][i] * np.sqrt((1 / temp[switch + 'SampleSize'][i]) + (1 / temp['EndToEndSampleSizeDelay'][flow]["0"][i]))) / temp[switch + 'DelayMean'][i] for i in range(temp['experiments']) if  temp['EndToEndSampleSizeDelay'][flow]["0"][i] != 0])
                    switch_errors_bounds[rate][switch]['SuccessProb'] = np.mean([(temp[switch + 'SuccessProbStd'][i] * np.sqrt((1 / temp[switch + 'SampleSize'][i]) + (1 / temp['EndToEndSampleSizeSuccess'][flow]["0"][i]))) / temp[switch + 'SuccessProbMean'][i] for i in range(temp['experiments']) if  temp['EndToEndSampleSizeSuccess'][flow]["0"][i] != 0])
                    switch_errors_bounds[rate][switch]['NonMarkingProb'] = np.mean([(temp[switch + 'NonMarkingProbStd'][i] * np.sqrt((1 / temp[switch + 'SampleSize'][i]) + (1 / temp['EndToEndSampleSizeMarking'][flow]["0"][i]))) / temp[switch + 'NonMarkingProbMean'][i] for i in range(temp['experiments']) if  temp['EndToEndSampleSizeMarking'][flow]["0"][i] != 0])
                    switch_stds[rate][switch] = {}
                    switch_stds[rate][switch]['Delay'] = np.sqrt(sum(temp[switch + 'Delaystd'][i] ** 2 for i in range(len(temp[switch + 'Delaystd'])))) / len(temp[switch + 'Delaystd'])
                    switch_stds[rate][switch]['SuccessProb'] = np.sqrt(sum(temp[switch + 'SuccessProbStd'][i] ** 2 for i in range(len(temp[switch + 'SuccessProbStd'])))) / len(temp[switch + 'SuccessProbStd'])
                    switch_stds[rate][switch]['NonMarkingProb'] = np.sqrt(sum(temp[switch + 'NonMarkingProbStd'][i] ** 2 for i in range(len(temp[switch + 'NonMarkingProbStd'])))) / len(temp[switch + 'NonMarkingProbStd'])
                    switch_queueOccupancy[rate][switch] = np.mean(temp[switch + 'Occupancy'])
                    switch_packtsInQueue[rate][switch] = np.mean(temp[switch + 'PacktsInQueue'])
                    switch_emptyFrac[rate][switch] = np.mean(temp[switch + 'EmptyFrac'])

    res = {}
    res['e2e_results_WOBias'] = e2e_results_WOBias
    res['e2e_delay'] = e2e_delay
    res['e2e_error'] = e2e_error
    res['dropRate'] = dropRate
    res['switch_stds'] = switch_stds
    res['switch_errors_bounds'] = switch_errors_bounds
    res['switch_delay'] = switch_delay
    res['switch_queueOccupancy'] = switch_queueOccupancy
    res['switch_packtsInQueue'] = switch_packtsInQueue
    res['switch_emptyFrac'] = switch_emptyFrac
    return res

def plot_droprate_vs_load(traffic_list, loads, rates, results_dir, DropRates, results_dir_file):
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
        ax.set_ylim(bottom=-0.05, top=2)
        ax.set_yticks(np.arange(-0.05, 2.05, 0.1))
        ax.tick_params(axis='y', labelsize=12)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}%"))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend(loc='best', fontsize=10, fancybox=True, shadow=True)

    axs[-1].set_xlabel("Offered Load", fontsize=16)
    axs[-1].set_xticks(sorted(loads))
    axs[-1].set_xticklabels([f"{l:.2f}" for l in sorted(loads)], fontsize=12)

    plt.suptitle("Drop Rate vs Load per Oversubscription Ratio", fontsize=24)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"../Results/results_{results_dir}/{results_dir_file}/DropRate_vs_Load_Subplots.png")
    plt.close()


def plot_forward_success_per_loads_traffic(results, loads, rates, results_dir, results_dir_file, selectedVarMethod, biasTag, flow):
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

        if 'success' in metric.lower():
            selectedVarMethod_ = 'probability_linearInterp_timeAvg'
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
        os.system('mkdir -p ../Results/results_' + results_dir + '/' + results_dir_file + '/' + flow)
        plt.savefig(f"../Results/results_{results_dir}/{results_dir_file}/{flow}/{flow}_{biasTag}_{metric}_SuccessRate_vs_Load_Subplots.png")
        plt.close()

def plot_metric_per_loads_traffic_with_std(traffic_list, metric, metric_std, loads, rates, results_dir, results_dir_file, label):
    print(f"Generating {label} Rate vs Load subplots (subplot per ratio, shape per traffic)...")

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
    max_y = 0
    min_y = 0
    for idx, rate in enumerate(rates):
        alpha = oversub_ratio_map[rate]
        ax = axs[idx]
        for traffic in traffic_list:
            marker = marker_map[traffic]
            color = color_map[traffic]
            x_vals, y_vals, error_valas = [], [], []

            for load in sorted(loads):
                metric_val = metric[traffic][load].get(rate, np.nan)
                metric_std_val = metric_std[traffic][load].get(rate, np.nan)
                if not np.isnan(metric_val):
                    x_vals.append(load)
                    y_vals.append(metric_val)  # convert to %
                    error_valas.append(metric_std_val)

            if x_vals:
                if max(y_vals) > max_y:
                    max_y = max(y_vals)
                if min_y == 0:
                    min_y = min(y_vals)
                if min(y_vals) < min_y:
                    min_y = min(y_vals)
                ax.errorbar(
                    x_vals, y_vals, error_valas,
                    linestyle='-',
                    marker=marker,
                    color=color,
                    markerfacecolor='none',  # hollow
                    markeredgecolor=color,
                    markersize=8,
                    linewidth=1.5,
                    label=traffic
                )

        ax.set_title(f"{label} vs Load (α = {alpha:.2f})", fontsize=18)
        ax.set_ylabel(f"{label}", fontsize=14)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend(loc='best', fontsize=10, fancybox=True, shadow=True)
        ax.axhline(36000 * alpha, color='red', linestyle='--', linewidth=1, label='Trshold')
        ax.text(0.02, 0.95, f'Trshold = {36000 * alpha:.2f}', transform=ax.transAxes, color='red', fontsize=12, verticalalignment='top')

        ax.axhline(20000 * alpha, color='blue', linestyle='--', linewidth=1, label='1 MSS')
        ax.text(0.02, 0.90, f'1 MSS = {20000 * alpha:.2f}', transform=ax.transAxes, color='blue', fontsize=12, verticalalignment='top')
    # max_y = 0.003
    # min_y = -10000
    # print(f"max_y: {max_y}, min_y: {min_y}")
    for ax in axs:
        ax.set_ylim(bottom=-0.05 * min_y, top=1.05 * max_y)
        # ax.set_ylim(bottom=-0.05, top=0.5 * 1e6)
        ax.set_yticks(np.arange(-0.05 * min_y, 1.05 * max_y, (1.05 * max_y + 0.05 * min_y) / 20))
        # ax.set_yticks(np.arange(0, 0.5 * 1e6, 0.5 * 1e6 / 20))
        ax.tick_params(axis='y', labelsize=12)

    axs[-1].set_xlabel("Load", fontsize=16)
    axs[-1].set_xticks(sorted(loads))
    axs[-1].set_xticklabels([f"{l:.2f}" for l in sorted(loads)], fontsize=12)

    plt.suptitle(f"{label} vs Load per Oversubscription Ratio", fontsize=24)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"../Results/results_{results_dir}/{results_dir_file}/{label}_withSTD_vs_Load_Subplots.png")
    plt.close()
    
def plot_metric_per_loads_traffic(traffic_list, metric, loads, rates, results_dir, results_dir_file, label, flow):
    print(f"Generating {label} Rate vs Load subplots (subplot per ratio, shape per traffic)...")

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
    max_y = -np.inf
    min_y = np.inf
    for idx, rate in enumerate(rates):
        alpha = oversub_ratio_map[rate]
        ax = axs[idx]
        for traffic in traffic_list:
            marker = marker_map[traffic]
            color = color_map[traffic]
            x_vals, y_vals = [], []

            for load in sorted(loads):
                metric_val = metric[traffic][load].get(rate, np.nan)
                if not np.isnan(metric_val):
                    x_vals.append(load)
                    y_vals.append(metric_val)  # convert to %

            if x_vals:
                if max(y_vals) > max_y:
                    max_y = max(y_vals)
                if min_y == 0:
                    min_y = min(y_vals)
                if min(y_vals) < min_y:
                    min_y = min(y_vals)
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

        ax.set_title(f"{label} vs Load (α = {alpha:.2f})", fontsize=18)
        ax.set_ylabel(f"{label}", fontsize=14)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend(loc='best', fontsize=10, fancybox=True, shadow=True)

    # max_y = 0.003
    # min_y = -10000
    # print(f"max_y: {max_y}, min_y: {min_y}")
    for ax in axs:
        ax.set_ylim(bottom=0.05 * min_y, top=1.05 * max_y)
        # ax.set_ylim(bottom=-0.5, top=0.5)
        ax.set_yticks(np.arange(0.05 * min_y, 1.05 * max_y, (1.05 * max_y - 0.05 * min_y) / 20))
        # ax.set_yticks(np.arange(-0.5, 0.5, 1.0 / 20))
        ax.tick_params(axis='y', labelsize=12)

    axs[-1].set_xlabel("Load", fontsize=16)
    axs[-1].set_xticks(sorted(loads))
    axs[-1].set_xticklabels([f"{l:.2f}" for l in sorted(loads)], fontsize=12)

    plt.suptitle(f"{label} vs Load per Oversubscription Ratio", fontsize=24)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.system('mkdir -p ../Results/results_' + results_dir + '/' + results_dir_file + '/' + flow)
    plt.savefig(f"../Results/results_{results_dir}/{results_dir_file}/{flow}/{flow}_{label}_vs_Load_Subplots.png")
    plt.close()

def analyse_forward_exp(results_dir, results_dir_file, rateScales, loads, selectedVarMethods, traffics):
    e2e_results_WOBias = {}
    e2e_delay = {}
    e2e_error = {}
    e2e_error_all = {}
    e2e_error_all['delay'] = {}
    e2e_error_all['success'] = {}
    e2e_error_all['nonMarking'] = {}
    dropRate = {}
    switch_error_bounds = {}
    switch_error_bounds_all = {}
    switch_error_bounds_all['delay'] = {}
    switch_error_bounds_all['success'] = {}
    switch_error_bounds_all['nonMarking'] = {}
    switch_delay = {}
    switch_queueOccupancy = {}
    switch_packtsInQueue = {}
    switch_emptyFrac = {}
    switch_stds = {}
    switch_stds_all = {}
    switch_stds_all['delay'] = {}
    switch_stds_all['success'] = {}
    switch_stds_all['nonMarking'] = {}
    for traffic in traffics:
        e2e_results_WOBias[traffic] = {}
        dropRate[traffic] = {}
        e2e_delay[traffic] = {}
        switch_delay[traffic] = {}
        switch_queueOccupancy[traffic] = {}
        switch_packtsInQueue[traffic] = {}
        switch_emptyFrac[traffic] = {}
        switch_stds[traffic] = {}
        switch_stds_all['delay'][traffic] = {}
        switch_stds_all['success'][traffic] = {}
        switch_stds_all['nonMarking'][traffic] = {}
        e2e_error[traffic] = {}
        e2e_error_all['delay'][traffic] = {}
        e2e_error_all['success'][traffic] = {}
        e2e_error_all['nonMarking'][traffic] = {}
        switch_error_bounds[traffic] = {}
        switch_error_bounds_all['delay'][traffic] = {}
        switch_error_bounds_all['success'][traffic] = {}
        switch_error_bounds_all['nonMarking'][traffic] = {}
        for load in loads:
            e2e_results_WOBias[traffic][load] = {}
            dropRate[traffic][load] = {}
            e2e_delay[traffic][load] = {}
            switch_delay[traffic][load] = {}
            switch_queueOccupancy[traffic][load] = {}
            switch_packtsInQueue[traffic][load] = {}
            switch_emptyFrac[traffic][load] = {}
            switch_stds[traffic][load] = {}
            switch_stds_all['delay'][traffic][load] = {}
            switch_stds_all['success'][traffic][load] = {}
            switch_stds_all['nonMarking'][traffic][load] = {}
            e2e_error[traffic][load] = {}
            e2e_error_all['delay'][traffic][load] = {}
            e2e_error_all['success'][traffic][load] = {}
            e2e_error_all['nonMarking'][traffic][load] = {}
            switch_error_bounds[traffic][load] = {}
            switch_error_bounds_all['delay'][traffic][load] = {}
            switch_error_bounds_all['success'][traffic][load] = {}
            switch_error_bounds_all['nonMarking'][traffic][load] = {}
            res = readResults(results_dir, rateScales, results_dir_file, selectedVarMethods, load=load, traffic=traffic)
            e2e_results_WOBias[traffic][load] = res['e2e_results_WOBias']
            e2e_delay[traffic][load] = res['e2e_delay']
            e2e_error[traffic][load] = res['e2e_error']
            dropRate[traffic][load] = res['dropRate']
            switch_stds[traffic][load] = res['switch_stds']
            switch_error_bounds[traffic][load] = res['switch_errors_bounds']
            switch_delay[traffic][load] = res['switch_delay']
            switch_queueOccupancy[traffic][load] = res['switch_queueOccupancy']
            switch_packtsInQueue[traffic][load] = res['switch_packtsInQueue']
            switch_emptyFrac[traffic][load] = res['switch_emptyFrac']

    selectedRates = rateScales
    traffics = list(e2e_results_WOBias.keys())
    for flow in flows:
        e2e_results_WOBias_flow = {}
        e2e_delay_flow = {}
        switch_delay_flow = {}
        switch_queueOccupancy_flow = {}
        switch_packtsInQueue_flow = {}
        switch_emptyFrac_flow = {}
        for traffic in traffics:
            e2e_results_WOBias_flow[traffic] = {}
            e2e_delay_flow[traffic] = {}
            switch_delay_flow[traffic] = {}
            switch_queueOccupancy_flow[traffic] = {}
            switch_packtsInQueue_flow[traffic] = {}
            switch_emptyFrac_flow[traffic] = {}
            for load in loads:
                e2e_results_WOBias_flow[traffic][load] = {}
                e2e_delay_flow[traffic][load] = {}
                switch_delay_flow[traffic][load] = {}
                switch_queueOccupancy_flow[traffic][load] = {}
                switch_packtsInQueue_flow[traffic][load] = {}
                switch_emptyFrac_flow[traffic][load] = {}
                for rate in rateScales: 
                    # for the groundtruth
                    e2e_error_all['delay'][traffic][load][rate] = e2e_error[traffic][load][rate][flow]['Delay']
                    e2e_error_all['success'][traffic][load][rate] = e2e_error[traffic][load][rate][flow]['SuccessProb']
                    e2e_error_all['nonMarking'][traffic][load][rate] = e2e_error[traffic][load][rate][flow]['NonMarkingProb']
                    e2e_delay_flow[traffic][load][rate] = e2e_delay[traffic][load][rate][flow]
                    e2e_results_WOBias_flow[traffic][load][rate] = e2e_results_WOBias[traffic][load][rate][flow]
                    # for switches
                    switch_error_bounds_all['delay'][traffic][load][rate] = switch_error_bounds[traffic][load][rate][flow]['Delay']
                    switch_error_bounds_all['success'][traffic][load][rate] = switch_error_bounds[traffic][load][rate][flow]['SuccessProb']
                    switch_error_bounds_all['nonMarking'][traffic][load][rate] = switch_error_bounds[traffic][load][rate][flow]['NonMarkingProb']
                    switch_stds_all['delay'][traffic][load][rate] = switch_stds[traffic][load][rate][flow]['Delay']
                    switch_stds_all['success'][traffic][load][rate] = switch_stds[traffic][load][rate][flow]['SuccessProb']
                    switch_stds_all['nonMarking'][traffic][load][rate] = switch_stds[traffic][load][rate][flow]['NonMarkingProb']
                    switch_delay_flow[traffic][load][rate] = switch_delay[traffic][load][rate][flow]
                    switch_queueOccupancy_flow[traffic][load][rate] = switch_queueOccupancy[traffic][load][rate][flow]
                    switch_packtsInQueue_flow[traffic][load][rate] = switch_packtsInQueue[traffic][load][rate][flow]
                    switch_emptyFrac_flow[traffic][load][rate] = switch_emptyFrac[traffic][load][rate][flow]
        # for the groundtruth
        plot_metric_per_loads_traffic(traffics, e2e_error_all['delay'], loads, selectedRates, results_dir, results_dir_file, 'ABS Error of Delay', flow)
        plot_metric_per_loads_traffic(traffics, e2e_error_all['success'], loads, selectedRates, results_dir, results_dir_file, 'ABS Error of Success Probability', flow)
        plot_metric_per_loads_traffic(traffics, e2e_error_all['nonMarking'], loads, selectedRates, results_dir, results_dir_file, 'ABS Error of Non Marking Probability', flow)
        plot_metric_per_loads_traffic(traffics, e2e_delay_flow, loads, selectedRates, results_dir, results_dir_file, 'End-to-End Delay(ns)', flow)
        plot_forward_success_per_loads_traffic(e2e_results_WOBias_flow, loads, selectedRates, results_dir, results_dir_file, selectedVarMethods, 'WithoutBias', flow)

        # for switches
        plot_metric_per_loads_traffic(traffics, switch_error_bounds_all['delay'], loads, selectedRates, results_dir, results_dir_file, 'Error bounds of Delay(ns)', flow)
        plot_metric_per_loads_traffic(traffics, switch_error_bounds_all['success'], loads, selectedRates, results_dir, results_dir_file, 'Error bounds of Success Probability', flow)
        plot_metric_per_loads_traffic(traffics, switch_error_bounds_all['nonMarking'], loads, selectedRates, results_dir, results_dir_file, 'Error bounds of Non Marking Probability', flow)
        plot_metric_per_loads_traffic(traffics, switch_stds_all['delay'], loads, selectedRates, results_dir, results_dir_file, 'STD of Delay(ns)', flow)
        plot_metric_per_loads_traffic(traffics, switch_stds_all['success'], loads, selectedRates, results_dir, results_dir_file, 'STD of Success Probability', flow)
        plot_metric_per_loads_traffic(traffics, switch_stds_all['nonMarking'], loads, selectedRates, results_dir, results_dir_file, 'STD of Non Marking Probability', flow)
        plot_metric_per_loads_traffic(traffics, switch_delay_flow, loads, selectedRates, results_dir, results_dir_file, 'Switch Delay(ns)', flow)
        plot_metric_per_loads_traffic(traffics, switch_queueOccupancy_flow, loads, selectedRates, results_dir, results_dir_file, 'Queue Occupancy(%)', flow)
        plot_metric_per_loads_traffic(traffics, switch_packtsInQueue_flow, loads, selectedRates, results_dir, results_dir_file, '#Packets in Queue', flow)
        plot_metric_per_loads_traffic(traffics, switch_emptyFrac_flow, loads, selectedRates, results_dir, results_dir_file, 'Empty Fraction', flow)
    plot_droprate_vs_load(dropRate, loads, selectedRates, results_dir, dropRate, results_dir_file)
         
def __main__():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dir",
                    required=True,
                    dest="dir",
                    help="The directory of the results",
                   default="")
    args = parser.parse_args()
    results_dir = args.dir
    start = 1.3 * 1e9
    end = 1.8 * 1e9
    results_dir_file = "Q_switch_1.0_30_{}_to_{}".format(start, end)
    config = configparser.ConfigParser()
    config.read('../Results/results_{}/Parameters.config'.format(args.dir))
    rateScales = [float(x) for x in config.get('Settings', 'serviceRateScales').split(',')]
    loads = [float(x) for x in config.get('Settings', 'load').split(',')]
    traffics = config.get('Settings', 'traffic').split(',')
    # traffics = ["Google_AllRPC", "Fabricated_Heavy_Head", "Fabricated_Heavy_Middle", "Google_SearchRPC", "Facebook_HadoopDist_All"]
    selectedVarMethods = ['event_linearInterp_timeAvg', 'probability_linearInterp_timeAvg']
    os.system('mkdir -p ../Results/results_' + results_dir + '/' + results_dir_file)
    analyse_forward_exp(results_dir, results_dir_file, rateScales, loads, selectedVarMethods, traffics)
        

__main__()
