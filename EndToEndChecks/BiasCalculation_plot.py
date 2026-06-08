import argparse
import configparser
import os
import json as js
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from Utils import compute_average_packet_size

confidenceValue = 1.96 # 95% confidence interval
maxError = 0.40
estimated_bias_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
per_queue_size_thresholds_bytes = [100, 250, 500, 750, 1000, 1250, 1500, 1750]
per_queue_size_thresholds_packets = [0.25, 0.5, 0.75, 1.0, 1.25]
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k']
def readResults(results_dict, results_dir, serviceRateScale, results_dir_file, differentiationDelay=0, errorRate=0, load=0, traffic='', queues=[]):
    rate_dir = traffic + "/" + str(serviceRateScale) + "/" + str(load)
    for file in os.listdir('../Results/results_' + results_dir + '/' + rate_dir):
        if file.find(results_dir_file) != -1:
            temp = {}
            if file.endswith('.json'):
                with open('../Results/results_' + results_dir + '/' + rate_dir + '/'+file) as f:
                    temp = js.load(f)
            else:
                continue
            print('../Results/results_' + results_dir + '/' + rate_dir + '/'+file)
            e2e_vs_sum_consistency_check_filtered_estimated_bias = []
            for idx in range(len(estimated_bias_thresholds)):
                e2e_vs_sum_consistency_check_filtered_estimated_bias.append([])
            e2e_vs_sum_consistency_check_filtered_queue_size_packets = []
            for idx in range(len(per_queue_size_thresholds_packets)):
                e2e_vs_sum_consistency_check_filtered_queue_size_packets.append([])
            e2e_vs_sum_consistency_check_filtered_queue_size_bytes = []
            for idx in range(len(per_queue_size_thresholds_bytes)):
                e2e_vs_sum_consistency_check_filtered_queue_size_bytes.append([])
            e2e_vs_sum_consistency_check_with_estimated_bias_added_to_bound = []
            for i in range(len(temp['experiment'])):
                results_dict['e2e_vs_sum_error_bound'][traffic][load][serviceRateScale].append(temp['e2e_vs_sum_error_bound'][i])
                results_dict['e2e_vs_sum_relative_error_bound'][traffic][load][serviceRateScale].append(temp['e2e_vs_sum_error_bound'][i] / temp['sum_poisson_samples_queue_delay_mean'][i])
                results_dict['e2e_vs_sum_abs_error'][traffic][load][serviceRateScale].append(abs(temp['sum_poisson_samples_queue_delay_mean'][i] - temp['e2e_poisson_samples_queue_delay_mean'][i]))
                results_dict['e2e_vs_sum_relative_error'][traffic][load][serviceRateScale].append(abs(temp['sum_poisson_samples_queue_delay_mean'][i] - temp['e2e_poisson_samples_queue_delay_mean'][i]) / temp['sum_poisson_samples_queue_delay_mean'][i])
                results_dict['e2e_vs_sum_estimated_bias'][traffic][load][serviceRateScale].append(sum([temp[queue_name+'bias'][i] for queue_name in queues]))
                results_dict['e2e_total_queuing_delay_time'][traffic][load][serviceRateScale].append(temp['e2e_poisson_samples_queue_delay_mean'][i])
                results_dict['e2e_total_queuing_delay_packets'][traffic][load][serviceRateScale].append(sum([temp[queue_name+'NPkts'][i] for queue_name in queues]))
                results_dict['e2e_total_queuing_delay_bytes'][traffic][load][serviceRateScale].append(sum([temp[queue_name+'NBytes'][i] for queue_name in queues]))
                success_prob_error = temp['e2e_poisson_samples_queue_success_prob_mean'][i] - temp['sum_poisson_samples_queue_success_prob_mean'][i]
                nonmarking_prob_error = temp['e2e_poisson_samples_queue_nonmarking_prob_mean'][i] - temp['sum_poisson_samples_queue_nonmarking_prob_mean'][i]
                results_dict['e2e_vs_sum_success_prob_error'][traffic][load][serviceRateScale].append(success_prob_error)
                results_dict['e2e_vs_sum_success_prob_abs_error'][traffic][load][serviceRateScale].append(abs(success_prob_error))
                results_dict['e2e_vs_sum_success_prob_error_bound_upper'][traffic][load][serviceRateScale].append(temp['e2e_vs_sum_error_success_prob_bound'][i][0])
                results_dict['e2e_vs_sum_success_prob_error_bound_lower'][traffic][load][serviceRateScale].append(temp['e2e_vs_sum_error_success_prob_bound'][i][1])
                results_dict['e2e_vs_sum_success_prob_e2e_mean'][traffic][load][serviceRateScale].append(temp['e2e_poisson_samples_queue_success_prob_mean'][i])
                results_dict['e2e_vs_sum_success_prob_sum_mean'][traffic][load][serviceRateScale].append(temp['sum_poisson_samples_queue_success_prob_mean'][i])
                results_dict['e2e_vs_sum_nonmarking_prob_error'][traffic][load][serviceRateScale].append(nonmarking_prob_error)
                results_dict['e2e_vs_sum_nonmarking_prob_abs_error'][traffic][load][serviceRateScale].append(abs(nonmarking_prob_error))
                results_dict['e2e_vs_sum_nonmarking_prob_error_bound_upper'][traffic][load][serviceRateScale].append(temp['e2e_vs_sum_error_nonmarking_prob_bound'][i][0])
                results_dict['e2e_vs_sum_nonmarking_prob_error_bound_lower'][traffic][load][serviceRateScale].append(temp['e2e_vs_sum_error_nonmarking_prob_bound'][i][1])
                results_dict['e2e_vs_sum_nonmarking_prob_e2e_mean'][traffic][load][serviceRateScale].append(temp['e2e_poisson_samples_queue_nonmarking_prob_mean'][i])
                results_dict['e2e_vs_sum_nonmarking_prob_sum_mean'][traffic][load][serviceRateScale].append(temp['sum_poisson_samples_queue_nonmarking_prob_mean'][i])
                e2e_vs_sum_consistency_check_with_estimated_bias_added_to_bound.append(abs(temp['sum_poisson_samples_queue_delay_mean'][i] - temp['e2e_poisson_samples_queue_delay_mean'][i]) <= (temp['e2e_vs_sum_error_bound'][i] + results_dict['e2e_vs_sum_estimated_bias'][traffic][load][serviceRateScale][-1]))
                for idx, trsh in enumerate(estimated_bias_thresholds):
                    if results_dict['e2e_vs_sum_estimated_bias'][traffic][load][serviceRateScale][-1] < trsh * temp['e2e_vs_sum_error_bound'][i]:
                        e2e_vs_sum_consistency_check_filtered_estimated_bias[idx].append(temp['e2e_vs_sum_consistent'][i])

                queue_size_filtered_packets = [False for i in range(len(per_queue_size_thresholds_packets))]
                queue_size_filtered_bytes = [False for i in range(len(per_queue_size_thresholds_bytes))]
                for queue_name in queues:
                    results_dict['queue_error_bound'][queue_name][traffic][load][serviceRateScale].append(temp[queue_name+'error_bound'][i])
                    results_dict['queue_relative_error_bound'][queue_name][traffic][load][serviceRateScale].append(temp[queue_name+'error_bound'][i] / temp[queue_name+'poisson_samples_queue_delay_mean'][i])
                    results_dict['queue_abs_error'][queue_name][traffic][load][serviceRateScale].append(abs(temp[queue_name+'poisson_samples_queue_delay_mean'][i] - temp[queue_name+'e2e_samples_queue_delay_mean'][i]))
                    results_dict['queue_error'][queue_name][traffic][load][serviceRateScale].append(temp[queue_name+'e2e_samples_queue_delay_mean'][i] - temp[queue_name+'poisson_samples_queue_delay_mean'][i])
                    results_dict['queue_relative_error'][queue_name][traffic][load][serviceRateScale].append(abs(temp[queue_name+'poisson_samples_queue_delay_mean'][i] - temp[queue_name+'e2e_samples_queue_delay_mean'][i]) / temp[queue_name+'poisson_samples_queue_delay_mean'][i])
                    results_dict['queue_estimated_bias'][queue_name][traffic][load][serviceRateScale].append(temp[queue_name+'bias'][i])
                    results_dict['queue_relative_error_bound_subtracted_relative_error'][queue_name][traffic][load][serviceRateScale].append(results_dict['queue_relative_error_bound'][queue_name][traffic][load][serviceRateScale][-1] - results_dict['queue_relative_error'][queue_name][traffic][load][serviceRateScale][-1])
                    results_dict['queue_relative_error_bound_subtracted_relative_error_after_bias'][queue_name][traffic][load][serviceRateScale].append(results_dict['queue_relative_error_bound'][queue_name][traffic][load][serviceRateScale][-1] - (abs(temp[queue_name+'poisson_samples_queue_delay_mean'][i] + temp[queue_name+'bias'][i] - temp[queue_name+'e2e_samples_queue_delay_mean'][i]) / temp[queue_name+'poisson_samples_queue_delay_mean'][i]))
                    results_dict['queue_delay_time'][queue_name][traffic][load][serviceRateScale].append(temp[queue_name+'poisson_samples_queue_delay_mean'][i])
                    results_dict['queue_delay_packets'][queue_name][traffic][load][serviceRateScale].append(temp[queue_name+'NPkts'][i])
                    results_dict['queue_delay_bytes'][queue_name][traffic][load][serviceRateScale].append(temp[queue_name+'NBytes'][i])                  
                    queue_success_prob_error = temp[queue_name+'e2e_samples_queue_success_prob_mean'][i] - temp[queue_name+'poisson_samples_queue_success_prob_mean'][i]
                    queue_nonmarking_prob_error = temp[queue_name+'e2e_samples_queue_nonmarking_prob_mean'][i] - temp[queue_name+'poisson_samples_queue_nonmarking_prob_mean'][i]
                    results_dict['queue_success_prob_error_bound'][queue_name][traffic][load][serviceRateScale].append(temp[queue_name+'success_prob_error_bound'][i])
                    results_dict['queue_success_prob_error'][queue_name][traffic][load][serviceRateScale].append(queue_success_prob_error)
                    results_dict['queue_success_prob_abs_error'][queue_name][traffic][load][serviceRateScale].append(abs(queue_success_prob_error))
                    results_dict['queue_e2e_success_prob_mean'][queue_name][traffic][load][serviceRateScale].append(temp[queue_name+'e2e_samples_queue_success_prob_mean'][i])
                    results_dict['queue_poisson_success_prob_mean'][queue_name][traffic][load][serviceRateScale].append(temp[queue_name+'poisson_samples_queue_success_prob_mean'][i])
                    results_dict['queue_nonmarking_prob_error_bound'][queue_name][traffic][load][serviceRateScale].append(temp[queue_name+'nonmarking_prob_error_bound'][i])
                    results_dict['queue_nonmarking_prob_error'][queue_name][traffic][load][serviceRateScale].append(queue_nonmarking_prob_error)
                    results_dict['queue_nonmarking_prob_abs_error'][queue_name][traffic][load][serviceRateScale].append(abs(queue_nonmarking_prob_error))
                    results_dict['queue_e2e_nonmarking_prob_mean'][queue_name][traffic][load][serviceRateScale].append(temp[queue_name+'e2e_samples_queue_nonmarking_prob_mean'][i])
                    results_dict['queue_poisson_nonmarking_prob_mean'][queue_name][traffic][load][serviceRateScale].append(temp[queue_name+'poisson_samples_queue_nonmarking_prob_mean'][i])
                    for idx, trsh in enumerate(per_queue_size_thresholds_packets):
                        if temp[queue_name+'NPkts'][i] < trsh:
                            queue_size_filtered_packets[idx] = True
                    for idx, trsh in enumerate(per_queue_size_thresholds_bytes):
                        if temp[queue_name+'NBytes'][i] < trsh:
                            queue_size_filtered_bytes[idx] = True
                for idx, trsh in enumerate(per_queue_size_thresholds_packets):
                    if not queue_size_filtered_packets[idx]:
                        e2e_vs_sum_consistency_check_filtered_queue_size_packets[idx].append(temp['e2e_vs_sum_consistent'][i])
                for idx, trsh in enumerate(per_queue_size_thresholds_bytes):
                    if not queue_size_filtered_bytes[idx]:
                        e2e_vs_sum_consistency_check_filtered_queue_size_bytes[idx].append(temp['e2e_vs_sum_consistent'][i])

            results_dict['e2e_vs_sum_relative_error_bound_subtracted_relative_error'][traffic][load][serviceRateScale] = [results_dict['e2e_vs_sum_relative_error_bound'][traffic][load][serviceRateScale][i] - results_dict['e2e_vs_sum_relative_error'][traffic][load][serviceRateScale][i] for i in range(len(results_dict['e2e_vs_sum_relative_error_bound'][traffic][load][serviceRateScale]))]
            results_dict['e2e_vs_sum_relative_error_bound_subtracted_relative_error_after_bias'][traffic][load][serviceRateScale] = [results_dict['e2e_vs_sum_relative_error_bound'][traffic][load][serviceRateScale][i] - abs(temp['sum_poisson_samples_queue_delay_mean'][i] + results_dict['e2e_vs_sum_estimated_bias'][traffic][load][serviceRateScale][i] - temp['e2e_poisson_samples_queue_delay_mean'][i]) / temp['sum_poisson_samples_queue_delay_mean'][i] for i in range(len(results_dict['e2e_vs_sum_relative_error_bound'][traffic][load][serviceRateScale]))]
            results_dict['e2e_vs_sum_consistency_check'][traffic][load][serviceRateScale] = sum(temp['e2e_vs_sum_consistent']) / len(temp['experiment']) * 100
            results_dict['e2e_vs_sum_consistency_check_with_estimated_bias'][traffic][load][serviceRateScale] = sum(temp['e2e_vs_sum_consistent_with_bias']) / len(temp['experiment']) * 100
            results_dict['e2e_vs_sum_consistency_check_with_estimated_bias_added_to_bound'][traffic][load][serviceRateScale] = sum(e2e_vs_sum_consistency_check_with_estimated_bias_added_to_bound) / len(temp['experiment']) * 100
            results_dict['e2e_vs_sum_success_prob_consistency_check'][traffic][load][serviceRateScale] = sum(temp['e2e_vs_sum_consistent_success_prob']) / len(temp['experiment']) * 100
            results_dict['e2e_vs_sum_nonmarking_prob_consistency_check'][traffic][load][serviceRateScale] = sum(temp['e2e_vs_sum_consistent_nonmarking_prob']) / len(temp['experiment']) * 100
            for idx in range(len(estimated_bias_thresholds)):
                results_dict['e2e_vs_sum_consistency_check_filtered_estimated_bias'][idx][traffic][load][serviceRateScale] = sum(e2e_vs_sum_consistency_check_filtered_estimated_bias[idx]) / len(e2e_vs_sum_consistency_check_filtered_estimated_bias[idx]) * 100 if len(e2e_vs_sum_consistency_check_filtered_estimated_bias[idx]) > 0 else np.nan
            for idx in range(len(per_queue_size_thresholds_packets)):
                results_dict['e2e_vs_sum_consistency_check_filtered_queue_size_packets'][idx][traffic][load][serviceRateScale] = sum(e2e_vs_sum_consistency_check_filtered_queue_size_packets[idx]) / len(e2e_vs_sum_consistency_check_filtered_queue_size_packets[idx]) * 100 if len(e2e_vs_sum_consistency_check_filtered_queue_size_packets[idx]) > 0 else np.nan
            for idx in range(len(per_queue_size_thresholds_bytes)):
                results_dict['e2e_vs_sum_consistency_check_filtered_queue_size_bytes'][idx][traffic][load][serviceRateScale] = sum(e2e_vs_sum_consistency_check_filtered_queue_size_bytes[idx]) / len(e2e_vs_sum_consistency_check_filtered_queue_size_bytes[idx]) * 100 if len(e2e_vs_sum_consistency_check_filtered_queue_size_bytes[idx]) > 0 else np.nan
            for queue_name in queues:
                results_dict['queue_consistency_check'][queue_name][traffic][load][serviceRateScale] = sum(temp[queue_name+'e2e_vs_poisson_consistent']) / len(temp['experiment']) * 100
                results_dict['queue_consistency_check_with_estimated_bias_added_to_Poisson_mean'][queue_name][traffic][load][serviceRateScale] = sum(temp[queue_name+'e2e_vs_poisson_consistent_with_bias']) / len(temp['experiment']) * 100
                consistency_check_with_estimated_bias_added_to_bound = [abs(temp[queue_name+'e2e_samples_queue_delay_mean'][i] - temp[queue_name+'poisson_samples_queue_delay_mean'][i]) <= (temp[queue_name+'error_bound'][i] + temp[queue_name+'bias'][i]) for i in range(len(temp['experiment']))]
                results_dict['queue_consistency_check_with_estimated_bias_added_to_bound'][queue_name][traffic][load][serviceRateScale] = sum(consistency_check_with_estimated_bias_added_to_bound) / len(temp['experiment']) * 100
                results_dict['queue_success_prob_consistency_check'][queue_name][traffic][load][serviceRateScale] = sum(temp[queue_name+'e2e_vs_poisson_consistent_success_prob']) / len(temp['experiment']) * 100
                results_dict['queue_nonmarking_prob_consistency_check'][queue_name][traffic][load][serviceRateScale] = sum(temp[queue_name+'e2e_vs_poisson_consistent_nonmarking_prob']) / len(temp['experiment']) * 100

def plot_forward_success_per_loads_traffic(results, loads, rates, results_dir, results_dir_file, biasTag):
    print("Generating forward success rate subplots per load (1 per α, shape per traffic)...")

    oversub_ratio_map = {r: 1 / r if r != 0 else np.nan for r in rates}
    traffic_list = list(results.keys())

    # Assign distinct marker per traffic
    marker_styles = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'H', '8']
    marker_map = {traffic: marker_styles[i % len(marker_styles)] for i, traffic in enumerate(traffic_list)}

    # Assign color per traffic (consistent across subplots)
    cmap = get_cmap('tab10') if len(traffic_list) <= 10 else get_cmap('tab20')
    color_map = {traffic: cmap(i / len(traffic_list)) for i, traffic in enumerate(traffic_list)}

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
                    val = results[traffic][load].get(rate, np.nan)
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

        ax.set_title(f"Success Rate vs Load (α = {alpha:.2f})", fontsize=18)
        ax.set_ylabel("Success Rate (%)", fontsize=14)
        ax.set_ylim(-5, 110)
        ax.set_yticks(np.arange(0, 101, 10))
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend(loc='best', fontsize=10, fancybox=True, shadow=True)

    axs[-1].set_xlabel("Offered Load", fontsize=16)
    axs[-1].set_xticks(sorted(loads))
    axs[-1].set_xticklabels([f"{l:.2f}" for l in sorted(loads)], fontsize=12)

    plt.suptitle(f" Success Rate vs Offered Load per Oversubscription Ratio", fontsize=24)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"../Results/results_{results_dir}/{results_dir_file}/{biasTag}_SuccessRate_vs_Load_Subplots.png")
    plt.close()


def plot_metric_per_loads_traffic_boxplot(traffic_list, metric, loads, rates, results_dir, results_dir_file, label):
    print(f"Generating {label} Rate vs Load BOXplots (subplot per ratio, box per traffic)...")

    # Compute oversubscription ratios
    oversub_ratios = [1 / r if r != 0 else np.nan for r in rates]
    oversub_ratio_map = dict(zip(rates, oversub_ratios))

    # Colors (one per traffic, consistent across subplots)
    cmap = get_cmap('tab10') if len(traffic_list) <= 10 else get_cmap('tab20')
    color_map = {traffic: cmap(i / len(traffic_list)) for i, traffic in enumerate(traffic_list)}

    num_ratios = len(rates)
    fig, axs = plt.subplots(nrows=num_ratios, ncols=1,
                            figsize=(20, 7 * num_ratios), sharex=True)
    if num_ratios == 1:
        axs = [axs]

    max_y = -np.inf
    min_y = np.inf

    loads_sorted = sorted(loads)
    x_base = np.arange(len(loads_sorted))  # base positions for loads
    box_width = 0.8 / max(len(traffic_list), 1)  # share 0.8 of space among traffics

    for idx, rate in enumerate(rates):
        alpha = oversub_ratio_map[rate]
        ax = axs[idx]

        # For each traffic, collect data (list of lists) and positions
        for t_idx, traffic in enumerate(traffic_list):
            data = []
            positions = []

            offset = (t_idx - (len(traffic_list) - 1) / 2.0) * box_width

            for i, load in enumerate(loads_sorted):
                values = metric[traffic][load].get(rate, [])
                if values is None:
                    values = []

                # Ensure it's a list/array
                values = list(values)
                if len(values) == 0:
                    continue  # no box for this (traffic, load, rate)

                data.append(values)
                pos = x_base[i] + offset
                positions.append(pos)

                # Update global min/max
                v_min = min(values)
                v_max = max(values)
                if v_max > max_y:
                    max_y = v_max
                if v_min < min_y:
                    min_y = v_min

            if len(data) == 0:
                continue

            # Color boxes by traffic
            color = color_map[traffic]

            bp = ax.boxplot(
                data,
                positions=positions,
                widths=box_width * 0.9,
                patch_artist=True,
                manage_ticks=False,
                showmeans=True,
                meanprops={
                    "marker": "o",
                    "markerfacecolor": color,
                    "markeredgecolor": color,
                    "markersize": 7,
                    "markeredgewidth": 1.5
                }
            )

            for patch in bp['boxes']:
                # patch.set_facecolor(color)
                patch.set_facecolor('none')
                patch.set_edgecolor(color)
                patch.set_linewidth(2)

            for element in ['whiskers', 'caps', 'medians']:
                for item in bp[element]:
                    item.set_color(color)
                    item.set_linewidth(2)

        ax.set_title(f"{label} vs Load (α = {alpha:.2f})", fontsize=18)
        ax.set_ylabel(label, fontsize=14)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Legend (one entry per traffic)
        handles = [
            Line2D([0], [0], linestyle='none', marker='s',
                   markersize=10, markerfacecolor=color_map[t],
                   markeredgecolor=color_map[t])
            for t in traffic_list
        ]
        ax.legend(handles, traffic_list, loc='best',
                  fontsize=10, fancybox=True, shadow=True)

    # If nothing got plotted, avoid crash
    if not np.isfinite(max_y) or not np.isfinite(min_y):
        max_y, min_y = 1.0, 0.0

    span = max_y - min_y
    padding = 0.05 * span if span > 0 else (0.05 if max_y == 0 else 0.05 * abs(max_y))

    for ax in axs:
        # Same y-scale across subplots.
        bottom = min_y - padding
        top = max_y + padding
        ax.set_ylim(bottom=bottom, top=top)
        ax.tick_params(axis='y', labelsize=12)

        # X axis: ticks at base positions (loads)
        ax.set_xticks(x_base)
        ax.set_xticklabels([f"{l:.2f}" for l in loads_sorted], fontsize=12)

    axs[-1].set_xlabel("Load", fontsize=16)

    plt.suptitle(f"{label} vs Load per Oversubscription Ratio (boxplots)", fontsize=24)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"../Results/results_{results_dir}/{results_dir_file}/{label}_vs_Load_Subplots_Boxplot.png")
    plt.close()

def prepare_results_dict(results_dir, results_dir_file, rateScales, loads, traffics, queues):
    results = {}
    e2e_vs_sum_error_bound = {}
    e2e_vs_sum_success_prob_error = {}
    e2e_vs_sum_success_prob_abs_error = {}
    e2e_vs_sum_success_prob_error_bound_upper = {}
    e2e_vs_sum_success_prob_error_bound_lower = {}
    e2e_vs_sum_success_prob_e2e_mean = {}
    e2e_vs_sum_success_prob_sum_mean = {}
    e2e_vs_sum_nonmarking_prob_error = {}
    e2e_vs_sum_nonmarking_prob_abs_error = {}
    e2e_vs_sum_nonmarking_prob_error_bound_upper = {}
    e2e_vs_sum_nonmarking_prob_error_bound_lower = {}
    e2e_vs_sum_nonmarking_prob_e2e_mean = {}
    e2e_vs_sum_nonmarking_prob_sum_mean = {}
    e2e_vs_sum_relative_error_bound = {}
    e2e_vs_sum_abs_error = {}
    e2e_vs_sum_relative_error = {}
    e2e_vs_sum_estimated_bias = {}
    e2e_vs_sum_relative_error_bound_subtracted_relative_error = {}
    e2e_vs_sum_relative_error_bound_subtracted_relative_error_after_bias = {}
    e2e_vs_sum_consistency_check_filtered_estimated_bias = []
    for trsh in estimated_bias_thresholds:
        e2e_vs_sum_consistency_check_filtered_estimated_bias.append({})
    e2e_vs_sum_consistency_check_filtered_queue_size_packets = []
    for trsh in per_queue_size_thresholds_packets:
        e2e_vs_sum_consistency_check_filtered_queue_size_packets.append({})
    e2e_vs_sum_consistency_check_filtered_queue_size_bytes = []
    for trsh in per_queue_size_thresholds_bytes:
        e2e_vs_sum_consistency_check_filtered_queue_size_bytes.append({})
    e2e_vs_sum_consistency_check = {}
    e2e_vs_sum_consistency_check_with_estimated_bias = {}
    e2e_vs_sum_consistency_check_with_estimated_bias_added_to_bound = {}
    e2e_vs_sum_success_prob_consistency_check = {}
    e2e_vs_sum_nonmarking_prob_consistency_check = {}
    e2e_total_queuing_delay_time = {}
    e2e_total_queuing_delay_packets = {}
    e2e_total_queuing_delay_bytes = {}
    queue_error_bound = {}
    queue_relative_error_bound = {}
    queue_abs_error = {}
    queue_error = {}
    queue_relative_error = {}
    queue_estimated_bias = {}
    queue_relative_error_bound_subtracted_relative_error = {}
    queue_relative_error_bound_subtracted_relative_error_after_bias = {}
    queue_consistency_check = {}
    queue_consistency_check_with_estimated_bias_added_to_Poisson_mean = {}
    queue_consistency_check_with_estimated_bias_added_to_bound = {}
    queue_success_prob_error_bound = {}
    queue_success_prob_error = {}
    queue_success_prob_abs_error = {}
    queue_success_prob_consistency_check = {}
    queue_e2e_success_prob_mean = {}
    queue_poisson_success_prob_mean = {}
    queue_nonmarking_prob_error_bound = {}
    queue_nonmarking_prob_error = {}
    queue_nonmarking_prob_abs_error = {}
    queue_nonmarking_prob_consistency_check = {}
    queue_e2e_nonmarking_prob_mean = {}
    queue_poisson_nonmarking_prob_mean = {}
    queue_delay_time = {}
    queue_delay_packets = {}
    queue_delay_bytes = {}

    for queue_name in queues:
        queue_error_bound[queue_name] = {}
        queue_relative_error_bound[queue_name] = {}
        queue_abs_error[queue_name] = {}
        queue_error[queue_name] = {}
        queue_relative_error[queue_name] = {}
        queue_estimated_bias[queue_name] = {}
        queue_relative_error_bound_subtracted_relative_error[queue_name] = {}
        queue_relative_error_bound_subtracted_relative_error_after_bias[queue_name] = {}
        queue_consistency_check[queue_name] = {}
        queue_consistency_check_with_estimated_bias_added_to_Poisson_mean[queue_name] = {}
        queue_consistency_check_with_estimated_bias_added_to_bound[queue_name] = {}
        queue_success_prob_error_bound[queue_name] = {}
        queue_success_prob_error[queue_name] = {}
        queue_success_prob_abs_error[queue_name] = {}
        queue_success_prob_consistency_check[queue_name] = {}
        queue_e2e_success_prob_mean[queue_name] = {}
        queue_poisson_success_prob_mean[queue_name] = {}
        queue_nonmarking_prob_error_bound[queue_name] = {}
        queue_nonmarking_prob_error[queue_name] = {}
        queue_nonmarking_prob_abs_error[queue_name] = {}
        queue_nonmarking_prob_consistency_check[queue_name] = {}
        queue_e2e_nonmarking_prob_mean[queue_name] = {}
        queue_poisson_nonmarking_prob_mean[queue_name] = {}
        queue_delay_time[queue_name] = {}
        queue_delay_packets[queue_name] = {}
        queue_delay_bytes[queue_name] = {}

    for traffic in traffics:
        e2e_vs_sum_error_bound[traffic] = {}
        e2e_vs_sum_success_prob_error[traffic] = {}
        e2e_vs_sum_success_prob_abs_error[traffic] = {}
        e2e_vs_sum_success_prob_error_bound_upper[traffic] = {}
        e2e_vs_sum_success_prob_error_bound_lower[traffic] = {}
        e2e_vs_sum_success_prob_e2e_mean[traffic] = {}
        e2e_vs_sum_success_prob_sum_mean[traffic] = {}
        e2e_vs_sum_nonmarking_prob_error[traffic] = {}
        e2e_vs_sum_nonmarking_prob_abs_error[traffic] = {}
        e2e_vs_sum_nonmarking_prob_error_bound_upper[traffic] = {}
        e2e_vs_sum_nonmarking_prob_error_bound_lower[traffic] = {}
        e2e_vs_sum_nonmarking_prob_e2e_mean[traffic] = {}
        e2e_vs_sum_nonmarking_prob_sum_mean[traffic] = {}
        e2e_vs_sum_relative_error_bound[traffic] = {}
        e2e_vs_sum_abs_error[traffic] = {}
        e2e_vs_sum_relative_error[traffic] = {}
        e2e_vs_sum_estimated_bias[traffic] = {}
        e2e_vs_sum_relative_error_bound_subtracted_relative_error[traffic] = {}
        e2e_vs_sum_relative_error_bound_subtracted_relative_error_after_bias[traffic] = {}
        for idx in range(len(estimated_bias_thresholds)):
            e2e_vs_sum_consistency_check_filtered_estimated_bias[idx][traffic] = {}
        for idx in range(len(per_queue_size_thresholds_packets)):
            e2e_vs_sum_consistency_check_filtered_queue_size_packets[idx][traffic] = {}
        for idx in range(len(per_queue_size_thresholds_bytes)):
            e2e_vs_sum_consistency_check_filtered_queue_size_bytes[idx][traffic] = {}
        e2e_vs_sum_consistency_check[traffic] = {}
        e2e_vs_sum_consistency_check_with_estimated_bias[traffic] = {}
        e2e_vs_sum_consistency_check_with_estimated_bias_added_to_bound[traffic] = {}
        e2e_vs_sum_success_prob_consistency_check[traffic] = {}
        e2e_vs_sum_nonmarking_prob_consistency_check[traffic] = {}
        e2e_total_queuing_delay_time[traffic] = {}
        e2e_total_queuing_delay_packets[traffic] = {}
        e2e_total_queuing_delay_bytes[traffic] = {}
        for queue_name in queues:
            queue_error_bound[queue_name][traffic] = {}
            queue_relative_error_bound[queue_name][traffic] = {}
            queue_abs_error[queue_name][traffic] = {}
            queue_error[queue_name][traffic] = {}
            queue_relative_error[queue_name][traffic] = {}
            queue_estimated_bias[queue_name][traffic] = {}
            queue_relative_error_bound_subtracted_relative_error[queue_name][traffic] = {}
            queue_relative_error_bound_subtracted_relative_error_after_bias[queue_name][traffic] = {}
            queue_consistency_check[queue_name][traffic] = {}
            queue_consistency_check_with_estimated_bias_added_to_Poisson_mean[queue_name][traffic] = {}
            queue_consistency_check_with_estimated_bias_added_to_bound[queue_name][traffic] = {}
            queue_success_prob_error_bound[queue_name][traffic] = {}
            queue_success_prob_error[queue_name][traffic] = {}
            queue_success_prob_abs_error[queue_name][traffic] = {}
            queue_success_prob_consistency_check[queue_name][traffic] = {}
            queue_e2e_success_prob_mean[queue_name][traffic] = {}
            queue_poisson_success_prob_mean[queue_name][traffic] = {}
            queue_nonmarking_prob_error_bound[queue_name][traffic] = {}
            queue_nonmarking_prob_error[queue_name][traffic] = {}
            queue_nonmarking_prob_abs_error[queue_name][traffic] = {}
            queue_nonmarking_prob_consistency_check[queue_name][traffic] = {}
            queue_e2e_nonmarking_prob_mean[queue_name][traffic] = {}
            queue_poisson_nonmarking_prob_mean[queue_name][traffic] = {}
            queue_delay_time[queue_name][traffic] = {}
            queue_delay_packets[queue_name][traffic] = {}
            queue_delay_bytes[queue_name][traffic] = {}
        for load in loads:
                e2e_vs_sum_error_bound[traffic][load] = {}
                e2e_vs_sum_success_prob_error[traffic][load] = {}
                e2e_vs_sum_success_prob_abs_error[traffic][load] = {}
                e2e_vs_sum_success_prob_error_bound_upper[traffic][load] = {}
                e2e_vs_sum_success_prob_error_bound_lower[traffic][load] = {}
                e2e_vs_sum_success_prob_e2e_mean[traffic][load] = {}
                e2e_vs_sum_success_prob_sum_mean[traffic][load] = {}
                e2e_vs_sum_nonmarking_prob_error[traffic][load] = {}
                e2e_vs_sum_nonmarking_prob_abs_error[traffic][load] = {}
                e2e_vs_sum_nonmarking_prob_error_bound_upper[traffic][load] = {}
                e2e_vs_sum_nonmarking_prob_error_bound_lower[traffic][load] = {}
                e2e_vs_sum_nonmarking_prob_e2e_mean[traffic][load] = {}
                e2e_vs_sum_nonmarking_prob_sum_mean[traffic][load] = {}
                e2e_vs_sum_abs_error[traffic][load] = {}
                e2e_vs_sum_relative_error[traffic][load] = {}
                e2e_vs_sum_relative_error_bound[traffic][load] = {}
                e2e_vs_sum_estimated_bias[traffic][load] = {}
                e2e_vs_sum_relative_error_bound_subtracted_relative_error[traffic][load] = {}
                e2e_vs_sum_relative_error_bound_subtracted_relative_error_after_bias[traffic][load] = {}
                for idx in range(len(estimated_bias_thresholds)):
                    e2e_vs_sum_consistency_check_filtered_estimated_bias[idx][traffic][load] = {}
                for idx in range(len(per_queue_size_thresholds_packets)):
                    e2e_vs_sum_consistency_check_filtered_queue_size_packets[idx][traffic][load] = {}
                for idx in range(len(per_queue_size_thresholds_bytes)):
                    e2e_vs_sum_consistency_check_filtered_queue_size_bytes[idx][traffic][load] = {}
                e2e_vs_sum_consistency_check[traffic][load] = {}
                e2e_vs_sum_consistency_check_with_estimated_bias[traffic][load] = {}
                e2e_vs_sum_consistency_check_with_estimated_bias_added_to_bound[traffic][load] = {}
                e2e_vs_sum_success_prob_consistency_check[traffic][load] = {}
                e2e_vs_sum_nonmarking_prob_consistency_check[traffic][load] = {}
                e2e_total_queuing_delay_time[traffic][load] = {}
                e2e_total_queuing_delay_packets[traffic][load] = {}
                e2e_total_queuing_delay_bytes[traffic][load] = {}
                for queue_name in queues:
                    queue_error_bound[queue_name][traffic][load] = {}
                    queue_relative_error_bound[queue_name][traffic][load] = {}
                    queue_abs_error[queue_name][traffic][load] = {}
                    queue_error[queue_name][traffic][load] = {}
                    queue_relative_error[queue_name][traffic][load] = {}
                    queue_estimated_bias[queue_name][traffic][load] = {}
                    queue_relative_error_bound_subtracted_relative_error[queue_name][traffic][load] = {}
                    queue_relative_error_bound_subtracted_relative_error_after_bias[queue_name][traffic][load] = {}
                    queue_consistency_check[queue_name][traffic][load] = {}
                    queue_consistency_check_with_estimated_bias_added_to_Poisson_mean[queue_name][traffic][load] = {}
                    queue_consistency_check_with_estimated_bias_added_to_bound[queue_name][traffic][load] = {}
                    queue_success_prob_error_bound[queue_name][traffic][load] = {}
                    queue_success_prob_error[queue_name][traffic][load] = {}
                    queue_success_prob_abs_error[queue_name][traffic][load] = {}
                    queue_success_prob_consistency_check[queue_name][traffic][load] = {}
                    queue_e2e_success_prob_mean[queue_name][traffic][load] = {}
                    queue_poisson_success_prob_mean[queue_name][traffic][load] = {}
                    queue_nonmarking_prob_error_bound[queue_name][traffic][load] = {}
                    queue_nonmarking_prob_error[queue_name][traffic][load] = {}
                    queue_nonmarking_prob_abs_error[queue_name][traffic][load] = {}
                    queue_nonmarking_prob_consistency_check[queue_name][traffic][load] = {}
                    queue_e2e_nonmarking_prob_mean[queue_name][traffic][load] = {}
                    queue_poisson_nonmarking_prob_mean[queue_name][traffic][load] = {}
                    queue_delay_time[queue_name][traffic][load] = {}
                    queue_delay_packets[queue_name][traffic][load] = {}
                    queue_delay_bytes[queue_name][traffic][load] = {}
                for rate in rateScales:
                    e2e_vs_sum_error_bound[traffic][load][rate] = []
                    e2e_vs_sum_success_prob_error[traffic][load][rate] = []
                    e2e_vs_sum_success_prob_abs_error[traffic][load][rate] = []
                    e2e_vs_sum_success_prob_error_bound_upper[traffic][load][rate] = []
                    e2e_vs_sum_success_prob_error_bound_lower[traffic][load][rate] = []
                    e2e_vs_sum_success_prob_e2e_mean[traffic][load][rate] = []
                    e2e_vs_sum_success_prob_sum_mean[traffic][load][rate] = []
                    e2e_vs_sum_nonmarking_prob_error[traffic][load][rate] = []
                    e2e_vs_sum_nonmarking_prob_abs_error[traffic][load][rate] = []
                    e2e_vs_sum_nonmarking_prob_error_bound_upper[traffic][load][rate] = []
                    e2e_vs_sum_nonmarking_prob_error_bound_lower[traffic][load][rate] = []
                    e2e_vs_sum_nonmarking_prob_e2e_mean[traffic][load][rate] = []
                    e2e_vs_sum_nonmarking_prob_sum_mean[traffic][load][rate] = []
                    e2e_vs_sum_relative_error_bound[traffic][load][rate] = []
                    e2e_vs_sum_abs_error[traffic][load][rate] = []
                    e2e_vs_sum_relative_error[traffic][load][rate] = []
                    e2e_vs_sum_estimated_bias[traffic][load][rate] = []
                    e2e_vs_sum_relative_error_bound_subtracted_relative_error[traffic][load][rate] = []
                    e2e_vs_sum_relative_error_bound_subtracted_relative_error_after_bias[traffic][load][rate] = []
                    for idx in range(len(estimated_bias_thresholds)):
                        e2e_vs_sum_consistency_check_filtered_estimated_bias[idx][traffic][load][rate] = np.nan
                    for idx in range(len(per_queue_size_thresholds_packets)):
                        e2e_vs_sum_consistency_check_filtered_queue_size_packets[idx][traffic][load][rate] = np.nan
                    for idx in range(len(per_queue_size_thresholds_bytes)):
                        e2e_vs_sum_consistency_check_filtered_queue_size_bytes[idx][traffic][load][rate] = np.nan
                    e2e_vs_sum_consistency_check[traffic][load][rate] = np.nan
                    e2e_vs_sum_consistency_check_with_estimated_bias[traffic][load][rate] = np.nan
                    e2e_vs_sum_consistency_check_with_estimated_bias_added_to_bound[traffic][load][rate] = np.nan
                    e2e_vs_sum_success_prob_consistency_check[traffic][load][rate] = np.nan
                    e2e_vs_sum_nonmarking_prob_consistency_check[traffic][load][rate] = np.nan
                    e2e_total_queuing_delay_time[traffic][load][rate] = []
                    e2e_total_queuing_delay_packets[traffic][load][rate] = []
                    e2e_total_queuing_delay_bytes[traffic][load][rate] = []
                    for queue_name in queues:
                        queue_error_bound[queue_name][traffic][load][rate] = []
                        queue_relative_error_bound[queue_name][traffic][load][rate] = []
                        queue_abs_error[queue_name][traffic][load][rate] = []
                        queue_error[queue_name][traffic][load][rate] = []
                        queue_relative_error[queue_name][traffic][load][rate] = []
                        queue_estimated_bias[queue_name][traffic][load][rate] = []
                        queue_relative_error_bound_subtracted_relative_error[queue_name][traffic][load][rate] = []
                        queue_relative_error_bound_subtracted_relative_error_after_bias[queue_name][traffic][load][rate] = []
                        queue_consistency_check[queue_name][traffic][load][rate] = np.nan
                        queue_consistency_check_with_estimated_bias_added_to_Poisson_mean[queue_name][traffic][load][rate] = np.nan
                        queue_consistency_check_with_estimated_bias_added_to_bound[queue_name][traffic][load][rate] = np.nan
                        queue_success_prob_error_bound[queue_name][traffic][load][rate] = []
                        queue_success_prob_error[queue_name][traffic][load][rate] = []
                        queue_success_prob_abs_error[queue_name][traffic][load][rate] = []
                        queue_success_prob_consistency_check[queue_name][traffic][load][rate] = np.nan
                        queue_e2e_success_prob_mean[queue_name][traffic][load][rate] = []
                        queue_poisson_success_prob_mean[queue_name][traffic][load][rate] = []
                        queue_nonmarking_prob_error_bound[queue_name][traffic][load][rate] = []
                        queue_nonmarking_prob_error[queue_name][traffic][load][rate] = []
                        queue_nonmarking_prob_abs_error[queue_name][traffic][load][rate] = []
                        queue_nonmarking_prob_consistency_check[queue_name][traffic][load][rate] = np.nan
                        queue_e2e_nonmarking_prob_mean[queue_name][traffic][load][rate] = []
                        queue_poisson_nonmarking_prob_mean[queue_name][traffic][load][rate] = []
                        queue_delay_time[queue_name][traffic][load][rate] = []
                        queue_delay_packets[queue_name][traffic][load][rate] = []
                        queue_delay_bytes[queue_name][traffic][load][rate] = []

    results['e2e_vs_sum_error_bound'] = e2e_vs_sum_error_bound
    results['e2e_vs_sum_success_prob_error'] = e2e_vs_sum_success_prob_error
    results['e2e_vs_sum_success_prob_abs_error'] = e2e_vs_sum_success_prob_abs_error
    results['e2e_vs_sum_success_prob_error_bound_upper'] = e2e_vs_sum_success_prob_error_bound_upper
    results['e2e_vs_sum_success_prob_error_bound_lower'] = e2e_vs_sum_success_prob_error_bound_lower
    results['e2e_vs_sum_success_prob_e2e_mean'] = e2e_vs_sum_success_prob_e2e_mean
    results['e2e_vs_sum_success_prob_sum_mean'] = e2e_vs_sum_success_prob_sum_mean
    results['e2e_vs_sum_nonmarking_prob_error'] = e2e_vs_sum_nonmarking_prob_error
    results['e2e_vs_sum_nonmarking_prob_abs_error'] = e2e_vs_sum_nonmarking_prob_abs_error
    results['e2e_vs_sum_nonmarking_prob_error_bound_upper'] = e2e_vs_sum_nonmarking_prob_error_bound_upper
    results['e2e_vs_sum_nonmarking_prob_error_bound_lower'] = e2e_vs_sum_nonmarking_prob_error_bound_lower
    results['e2e_vs_sum_nonmarking_prob_e2e_mean'] = e2e_vs_sum_nonmarking_prob_e2e_mean
    results['e2e_vs_sum_nonmarking_prob_sum_mean'] = e2e_vs_sum_nonmarking_prob_sum_mean
    results['e2e_vs_sum_relative_error_bound'] = e2e_vs_sum_relative_error_bound
    results['e2e_vs_sum_abs_error'] = e2e_vs_sum_abs_error
    results['e2e_vs_sum_relative_error'] = e2e_vs_sum_relative_error
    results['e2e_vs_sum_estimated_bias'] = e2e_vs_sum_estimated_bias
    results['e2e_vs_sum_relative_error_bound_subtracted_relative_error'] = e2e_vs_sum_relative_error_bound_subtracted_relative_error
    results['e2e_vs_sum_relative_error_bound_subtracted_relative_error_after_bias'] = e2e_vs_sum_relative_error_bound_subtracted_relative_error_after_bias
    results['e2e_vs_sum_consistency_check_filtered_estimated_bias'] = e2e_vs_sum_consistency_check_filtered_estimated_bias
    results['e2e_vs_sum_consistency_check_filtered_queue_size_packets'] = e2e_vs_sum_consistency_check_filtered_queue_size_packets
    results['e2e_vs_sum_consistency_check_filtered_queue_size_bytes'] = e2e_vs_sum_consistency_check_filtered_queue_size_bytes
    results['e2e_vs_sum_consistency_check'] = e2e_vs_sum_consistency_check
    results['e2e_vs_sum_consistency_check_with_estimated_bias'] = e2e_vs_sum_consistency_check_with_estimated_bias
    results['e2e_vs_sum_consistency_check_with_estimated_bias_added_to_bound'] = e2e_vs_sum_consistency_check_with_estimated_bias_added_to_bound
    results['e2e_vs_sum_success_prob_consistency_check'] = e2e_vs_sum_success_prob_consistency_check
    results['e2e_vs_sum_nonmarking_prob_consistency_check'] = e2e_vs_sum_nonmarking_prob_consistency_check
    results['e2e_total_queuing_delay_time'] = e2e_total_queuing_delay_time
    results['e2e_total_queuing_delay_packets'] = e2e_total_queuing_delay_packets
    results['e2e_total_queuing_delay_bytes'] = e2e_total_queuing_delay_bytes
    results['queue_error_bound'] = queue_error_bound
    results['queue_relative_error_bound'] = queue_relative_error_bound
    results['queue_abs_error'] = queue_abs_error
    results['queue_relative_error'] = queue_relative_error
    results['queue_estimated_bias'] = queue_estimated_bias
    results['queue_consistency_check'] = queue_consistency_check
    results['queue_relative_error_bound_subtracted_relative_error'] = queue_relative_error_bound_subtracted_relative_error
    results['queue_relative_error_bound_subtracted_relative_error_after_bias'] = queue_relative_error_bound_subtracted_relative_error_after_bias
    results['queue_consistency_check_with_estimated_bias_added_to_Poisson_mean'] = queue_consistency_check_with_estimated_bias_added_to_Poisson_mean
    results['queue_consistency_check_with_estimated_bias_added_to_bound'] = queue_consistency_check_with_estimated_bias_added_to_bound
    results['queue_success_prob_error_bound'] = queue_success_prob_error_bound
    results['queue_success_prob_error'] = queue_success_prob_error
    results['queue_success_prob_abs_error'] = queue_success_prob_abs_error
    results['queue_success_prob_consistency_check'] = queue_success_prob_consistency_check
    results['queue_e2e_success_prob_mean'] = queue_e2e_success_prob_mean
    results['queue_poisson_success_prob_mean'] = queue_poisson_success_prob_mean
    results['queue_nonmarking_prob_error_bound'] = queue_nonmarking_prob_error_bound
    results['queue_nonmarking_prob_error'] = queue_nonmarking_prob_error
    results['queue_nonmarking_prob_abs_error'] = queue_nonmarking_prob_abs_error
    results['queue_nonmarking_prob_consistency_check'] = queue_nonmarking_prob_consistency_check
    results['queue_e2e_nonmarking_prob_mean'] = queue_e2e_nonmarking_prob_mean
    results['queue_poisson_nonmarking_prob_mean'] = queue_poisson_nonmarking_prob_mean
    results['queue_delay_time'] = queue_delay_time
    results['queue_delay_packets'] = queue_delay_packets
    results['queue_delay_bytes'] = queue_delay_bytes
    results['queue_error'] = queue_error
    return results

def analyse_forward_exp(results_dir, results_dir_file, rateScales, loads, traffics, queues):
    results = prepare_results_dict(results_dir, results_dir_file, rateScales, loads, traffics, queues)
    for traffic in traffics:
        for load in loads:
            for rate in rateScales:
                readResults(results, results_dir, rate, results_dir_file, load=load, traffic=traffic, queues=queues)
                # import pprint
                # pp = pprint.PrettyPrinter(indent=4)
                # pp.pprint(results)

    plot_metric_per_loads_traffic_boxplot(traffics, results['e2e_vs_sum_error_bound'], loads, rateScales, results_dir, results_dir_file, "e2e vs sum error bound (ns)")
    plot_metric_per_loads_traffic_boxplot(traffics, results['e2e_vs_sum_relative_error_bound'], loads, rateScales, results_dir, results_dir_file, "e2e vs sum relative error bound")
    plot_metric_per_loads_traffic_boxplot(traffics, results['e2e_vs_sum_abs_error'], loads, rateScales, results_dir, results_dir_file, "e2e vs sum Absolute Error (ns)")
    plot_metric_per_loads_traffic_boxplot(traffics, results['e2e_vs_sum_relative_error'], loads, rateScales, results_dir, results_dir_file, "e2e vs sum Relative Error")
    plot_metric_per_loads_traffic_boxplot(traffics, results['e2e_vs_sum_estimated_bias'], loads, rateScales, results_dir, results_dir_file, "e2e vs sum Estimated Bias (ns)")
    plot_metric_per_loads_traffic_boxplot(traffics, results['e2e_vs_sum_relative_error_bound_subtracted_relative_error'], loads, rateScales, results_dir, results_dir_file, "e2e vs sum Relative Error Bound Subtracted Relative Error")
    plot_metric_per_loads_traffic_boxplot(traffics, results['e2e_vs_sum_relative_error_bound_subtracted_relative_error_after_bias'], loads, rateScales, results_dir, results_dir_file, "e2e vs sum Relative Error Bound Subtracted Relative Error After Bias")
    plot_forward_success_per_loads_traffic(results['e2e_vs_sum_consistency_check'], loads, rateScales, results_dir, results_dir_file, "consistency_check")
    plot_forward_success_per_loads_traffic(results['e2e_vs_sum_consistency_check_with_estimated_bias'], loads, rateScales, results_dir, results_dir_file, "consistency_check_with_estimated_bias")
    plot_forward_success_per_loads_traffic(results['e2e_vs_sum_consistency_check_with_estimated_bias_added_to_bound'], loads, rateScales, results_dir, results_dir_file, "consistency_check_with_estimated_bias_added_to_bound")
    plot_metric_per_loads_traffic_boxplot(traffics, results['e2e_vs_sum_success_prob_e2e_mean'], loads, rateScales, results_dir, results_dir_file, "e2e vs sum Success Prob E2E Mean")
    plot_metric_per_loads_traffic_boxplot(traffics, results['e2e_vs_sum_success_prob_sum_mean'], loads, rateScales, results_dir, results_dir_file, "e2e vs sum Success Prob Sum Mean")
    plot_metric_per_loads_traffic_boxplot(traffics, results['e2e_vs_sum_success_prob_error'], loads, rateScales, results_dir, results_dir_file, "e2e vs sum Success Prob Error")
    plot_metric_per_loads_traffic_boxplot(traffics, results['e2e_vs_sum_success_prob_abs_error'], loads, rateScales, results_dir, results_dir_file, "e2e vs sum Success Prob Absolute Error")
    plot_metric_per_loads_traffic_boxplot(traffics, results['e2e_vs_sum_success_prob_error_bound_upper'], loads, rateScales, results_dir, results_dir_file, "e2e vs sum Success Prob Upper Error Bound")
    plot_metric_per_loads_traffic_boxplot(traffics, results['e2e_vs_sum_success_prob_error_bound_lower'], loads, rateScales, results_dir, results_dir_file, "e2e vs sum Success Prob Lower Error Bound")
    plot_forward_success_per_loads_traffic(results['e2e_vs_sum_success_prob_consistency_check'], loads, rateScales, results_dir, results_dir_file, "success_prob_consistency_check")
    plot_metric_per_loads_traffic_boxplot(traffics, results['e2e_vs_sum_nonmarking_prob_e2e_mean'], loads, rateScales, results_dir, results_dir_file, "e2e vs sum Non-Marking Prob E2E Mean")
    plot_metric_per_loads_traffic_boxplot(traffics, results['e2e_vs_sum_nonmarking_prob_sum_mean'], loads, rateScales, results_dir, results_dir_file, "e2e vs sum Non-Marking Prob Sum Mean")
    plot_metric_per_loads_traffic_boxplot(traffics, results['e2e_vs_sum_nonmarking_prob_error'], loads, rateScales, results_dir, results_dir_file, "e2e vs sum Non-Marking Prob Error")
    plot_metric_per_loads_traffic_boxplot(traffics, results['e2e_vs_sum_nonmarking_prob_abs_error'], loads, rateScales, results_dir, results_dir_file, "e2e vs sum Non-Marking Prob Absolute Error")
    plot_metric_per_loads_traffic_boxplot(traffics, results['e2e_vs_sum_nonmarking_prob_error_bound_upper'], loads, rateScales, results_dir, results_dir_file, "e2e vs sum Non-Marking Prob Upper Error Bound")
    plot_metric_per_loads_traffic_boxplot(traffics, results['e2e_vs_sum_nonmarking_prob_error_bound_lower'], loads, rateScales, results_dir, results_dir_file, "e2e vs sum Non-Marking Prob Lower Error Bound")
    plot_forward_success_per_loads_traffic(results['e2e_vs_sum_nonmarking_prob_consistency_check'], loads, rateScales, results_dir, results_dir_file, "nonmarking_prob_consistency_check")
    plot_metric_per_loads_traffic_boxplot(traffics, results['e2e_total_queuing_delay_time'], loads, rateScales, results_dir, results_dir_file, f"e2e Total Queuing Delay Time (ns)")
    plot_metric_per_loads_traffic_boxplot(traffics, results['e2e_total_queuing_delay_packets'], loads, rateScales, results_dir, results_dir_file, f"e2e Total Queuing Delay Packets")
    plot_metric_per_loads_traffic_boxplot(traffics, results['e2e_total_queuing_delay_bytes'], loads, rateScales, results_dir, results_dir_file, f"e2e Total Queuing Delay Bytes")
    for idx, trsh in enumerate(estimated_bias_thresholds):
        plot_forward_success_per_loads_traffic(results['e2e_vs_sum_consistency_check_filtered_estimated_bias'][idx], loads, rateScales, results_dir, results_dir_file, f"consistency_check_filtered_by_estimated_bias_{trsh}")
    for idx, trsh in enumerate(per_queue_size_thresholds_packets):
        plot_forward_success_per_loads_traffic(results['e2e_vs_sum_consistency_check_filtered_queue_size_packets'][idx], loads, rateScales, results_dir, results_dir_file, f"consistency_check_filtered_by_queue_size_{trsh}pkts")
    for idx, trsh in enumerate(per_queue_size_thresholds_bytes):
        plot_forward_success_per_loads_traffic(results['e2e_vs_sum_consistency_check_filtered_queue_size_bytes'][idx], loads, rateScales, results_dir, results_dir_file, f"consistency_check_filtered_by_queue_size_{trsh}bytes")

    for queue_name in queues:
        plot_metric_per_loads_traffic_boxplot(traffics, results['queue_error_bound'][queue_name], loads, rateScales, results_dir, results_dir_file, f"{queue_name} Error Bound (ns)")
        plot_metric_per_loads_traffic_boxplot(traffics, results['queue_relative_error_bound'][queue_name], loads, rateScales, results_dir, results_dir_file, f"{queue_name} Relative Error Bound")
        plot_metric_per_loads_traffic_boxplot(traffics, results['queue_abs_error'][queue_name], loads, rateScales, results_dir, results_dir_file, f"{queue_name} Absolute Error (ns)")
        plot_metric_per_loads_traffic_boxplot(traffics, results['queue_error'][queue_name], loads, rateScales, results_dir, results_dir_file, f"{queue_name} Error (ns)")
        plot_metric_per_loads_traffic_boxplot(traffics, results['queue_relative_error'][queue_name], loads, rateScales, results_dir, results_dir_file, f"{queue_name} Relative Error")
        plot_metric_per_loads_traffic_boxplot(traffics, results['queue_estimated_bias'][queue_name], loads, rateScales, results_dir, results_dir_file, f"{queue_name} Estimated Bias (ns)")
        plot_metric_per_loads_traffic_boxplot(traffics, results['queue_relative_error_bound_subtracted_relative_error'][queue_name], loads, rateScales, results_dir, results_dir_file, f"{queue_name} Relative Error Bound Subtracted Relative Error")
        plot_metric_per_loads_traffic_boxplot(traffics, results['queue_relative_error_bound_subtracted_relative_error_after_bias'][queue_name], loads, rateScales, results_dir, results_dir_file, f"{queue_name} Relative Error Bound Subtracted Relative Error After Bias")
        plot_forward_success_per_loads_traffic(results['queue_consistency_check'][queue_name], loads, rateScales, results_dir, results_dir_file, f"{queue_name} Consistency Check")
        plot_forward_success_per_loads_traffic(results['queue_consistency_check_with_estimated_bias_added_to_bound'][queue_name], loads, rateScales, results_dir, results_dir_file, f"{queue_name} Consistency Check with Estimated Bias Added to Bound")
        plot_forward_success_per_loads_traffic(results['queue_consistency_check_with_estimated_bias_added_to_Poisson_mean'][queue_name], loads, rateScales, results_dir, results_dir_file, f"{queue_name} Consistency Check with Estimated Bias Added to Poisson Mean")
        plot_metric_per_loads_traffic_boxplot(traffics, results['queue_e2e_success_prob_mean'][queue_name], loads, rateScales, results_dir, results_dir_file, f"{queue_name} E2E Success Prob Mean")
        plot_metric_per_loads_traffic_boxplot(traffics, results['queue_poisson_success_prob_mean'][queue_name], loads, rateScales, results_dir, results_dir_file, f"{queue_name} Poisson Success Prob Mean")
        plot_metric_per_loads_traffic_boxplot(traffics, results['queue_success_prob_error_bound'][queue_name], loads, rateScales, results_dir, results_dir_file, f"{queue_name} Success Prob Error Bound")
        plot_metric_per_loads_traffic_boxplot(traffics, results['queue_success_prob_error'][queue_name], loads, rateScales, results_dir, results_dir_file, f"{queue_name} Success Prob Error")
        plot_metric_per_loads_traffic_boxplot(traffics, results['queue_success_prob_abs_error'][queue_name], loads, rateScales, results_dir, results_dir_file, f"{queue_name} Success Prob Absolute Error")
        plot_forward_success_per_loads_traffic(results['queue_success_prob_consistency_check'][queue_name], loads, rateScales, results_dir, results_dir_file, f"{queue_name} Success Prob Consistency Check")
        plot_metric_per_loads_traffic_boxplot(traffics, results['queue_e2e_nonmarking_prob_mean'][queue_name], loads, rateScales, results_dir, results_dir_file, f"{queue_name} E2E Non-Marking Prob Mean")
        plot_metric_per_loads_traffic_boxplot(traffics, results['queue_poisson_nonmarking_prob_mean'][queue_name], loads, rateScales, results_dir, results_dir_file, f"{queue_name} Poisson Non-Marking Prob Mean")
        plot_metric_per_loads_traffic_boxplot(traffics, results['queue_nonmarking_prob_error_bound'][queue_name], loads, rateScales, results_dir, results_dir_file, f"{queue_name} Non-Marking Prob Error Bound")
        plot_metric_per_loads_traffic_boxplot(traffics, results['queue_nonmarking_prob_error'][queue_name], loads, rateScales, results_dir, results_dir_file, f"{queue_name} Non-Marking Prob Error")
        plot_metric_per_loads_traffic_boxplot(traffics, results['queue_nonmarking_prob_abs_error'][queue_name], loads, rateScales, results_dir, results_dir_file, f"{queue_name} Non-Marking Prob Absolute Error")
        plot_forward_success_per_loads_traffic(results['queue_nonmarking_prob_consistency_check'][queue_name], loads, rateScales, results_dir, results_dir_file, f"{queue_name} Non-Marking Prob Consistency Check")
        plot_metric_per_loads_traffic_boxplot(traffics, results['queue_delay_time'][queue_name], loads, rateScales, results_dir, results_dir_file, f"{queue_name} Queue Delay Time (ns)")
        plot_metric_per_loads_traffic_boxplot(traffics, results['queue_delay_packets'][queue_name], loads, rateScales, results_dir, results_dir_file, f"{queue_name} Queue Delay Packets")
        plot_metric_per_loads_traffic_boxplot(traffics, results['queue_delay_bytes'][queue_name], loads, rateScales, results_dir, results_dir_file, f"{queue_name} Queue Delay Bytes")

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
    steadyStart = 0.010 * 1e9
    steadyEnd = 0.100 * 1e9
    queues = ["T0A0", "A0T2", "T2H3"]
    config = configparser.ConfigParser()
    config.read('../Results/results_{}/Parameters.config'.format(args.dir))
    rateScales = [float(x) for x in config.get('Settings', 'serviceRateScales').split(',')]
    loads = [float(x) for x in config.get('Settings', 'load').split(',')]
    traffics = config.get('Settings', 'traffic').split(',')
    # experiments = 1
    errorRates = [float(x) for x in config.get('Settings', 'errorRate').split(',')]
    differentiationDelays = [float(x) for x in config.get('Settings', 'differentiationDelay').split(',')]
    traffics = ["Google_AllRPC","Fabricated_Heavy_Head","Fabricated_Heavy_Middle","Google_SearchRPC", "Facebook_HadoopDist_All"]
    # traffics = ["Google_SearchRPC"]
    loads = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.95]
    # loads = [0.95]
    numOfSteadyParts = 1
    for start in range(int(steadyStart), int(steadyEnd), int((steadyEnd - steadyStart) / numOfSteadyParts)):
        print("Steady period: {} to {}".format(start, start + int((steadyEnd - steadyStart) / numOfSteadyParts)))
        # results_dir_file = "delay_minimum_bias_e2e_0.01subsampling_vs_switch_poisson.0_30_{}_to_{}".format(start, start + int((steadyEnd - steadyStart) / numOfSteadyParts))
        results_dir_file = "delay_minimum_bias_e2e_vs_switch_poisson.0_None_30_to_10000000"
        if args.IsForward == 1:
            os.system('mkdir -p ../Results/results_' + results_dir + '/' + results_dir_file)
            analyse_forward_exp(results_dir, results_dir_file, rateScales, loads, traffics, queues)
        else:
            print("Reverse experiment is not implemented yet!")

__main__()
