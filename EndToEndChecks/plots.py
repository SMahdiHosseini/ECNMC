import configparser
import os
import json as js
import matplotlib.pyplot as plt
import numpy as np
import glob
import sys

# results_dir = 'reverse_1'
# results_dir = 'highRate_reverse_loss_1'
# results_dir = 'highRate_reverse_loss_2'
# results_dir = 'reverse_delay_1'
# results_dir = 'reverse_delay_2'
# results_dir = 'forward'
def read_data_flowIndicator(results_dict):
    return list(results_dict['MaxEpsilonIneqDelay'].keys())

def plot_reverse_loss_1(results, flows, errorRates, key, sample_rate=None):
    fig = plt.figure()
    fig.set_size_inches(7, 6)
    ax1 = fig.add_subplot(111)
    X = [round(r * 0.001 * 100, 3) for r in errorRates]
    for flow in flows:
        if sample_rate is None:
            if flow == 'R0H0R2H0':
                ax1.plot(X, [100 - max(value['MaxEpsilonIneqSuccessProb'][key][flow]['A0'], value['MaxEpsilonIneqSuccessProb'][key][flow]['A1']) for value in results.values()], 'r^-', markersize=10)
            if flow == 'R0H1R2H1':
                ax1.plot(X, [100 - max(value['MaxEpsilonIneqSuccessProb'][key][flow]['A0'], value['MaxEpsilonIneqSuccessProb'][key][flow]['A1']) for value in results.values()], 'bX-', markersize=10)
        else:
            if flow == 'R0H0R2H0':
                ax1.plot(X, [100 - max(value['MaxEpsilonIneqSuccessProb'][key][sample_rate][flow]['A0'], value['MaxEpsilonIneqSuccessProb'][key][sample_rate][flow]['A1']) for value in results.values()], 'r^-', markersize=10)
            if flow == 'R0H1R2H1':
                ax1.plot(X, [100 - max(value['MaxEpsilonIneqSuccessProb'][key][sample_rate][flow]['A0'], value['MaxEpsilonIneqSuccessProb'][key][sample_rate][flow]['A1']) for value in results.values()], 'bX-', markersize=10)

    ax1.legend(['Flow 1', 'Flow 2'], fontsize=15, loc='lower right')
    ax1.set_xlabel('Silent Packet Drop Rate (%)', fontsize=25, labelpad=-1)
    ax1.set_ylabel('Success Rate (%)', fontsize=25, labelpad=-10)
    ax1.tick_params(axis='y', labelsize=20)
    ax1.set_xticks(X[0::3])
    ax1.set_xticklabels(X[0::3], fontsize=17)
    ax1.set_ylim(0, 110)
    plt.setp(ax1.spines.values(), lw=1.5, color='black')
    # plt.grid(True)
    # plt.title('success rate of detecting the silent packet drop for different silent drop rates')
    plt.savefig('../Results/results_{}/{}_success_perDropRate_{}.pdf'.format(results_dir, check, key + ('_' + sample_rate if sample_rate is not None else '')))
    plt.clf()

def plot_boxplot_loss(utilizationFactors, droprates_xticks, data, key):
    fig = plt.figure()
    fig.set_size_inches(7, 6)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    
    bp = ax1.boxplot(data)
    for box in bp['boxes']:
        box.set(linewidth=2)
    for whisker in bp['whiskers']:
        whisker.set_linewidth(2)
    for cap in bp['caps']:
        cap.set_linewidth(2)
    for median in bp['medians']:
        median.set_linewidth(2)

    ax1.set_ylim(-10, 110)
    ax1.set_xlabel('Congestion Factor', fontsize=25, labelpad=-1)
    ax1.set_ylabel('Success Rate (%)', fontsize=25, labelpad=-10)
    ax1.tick_params(axis='y', labelsize=20)
    # x = selectedUtils
    x = utilizationFactors.copy()
    x.reverse()
    x = [str(x[k]) if k % 2 == 0 else '' for k in range(len(x))]
    y = [i+1 for i in range(len(x))]
    # y = [y[1], y[3], y[5], y[7], y[9], y[11], y[13]]
    ax1.set_xticks(y, x)
    ax1.set_xticklabels(x, fontsize=17)
    ax1.set_ylim(0, 110)
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(y)
    # print(queueSizesStd)
    # x = [droprates_xticks[0], droprates_xticks[2], droprates_xticks[4], droprates_xticks[6], droprates_xticks[8], droprates_xticks[10]]
    # x = queueSizesStd
    x = droprates_xticks.copy()
    x.reverse()
    ax2.set_xticklabels(x, fontsize=5)
    ax2.set_xlabel('Drop Rate', fontsize=15, labelpad=10)
    # plt.axvline(x = 11, color = 'gray', label = 'axvline - full height', linestyle='--', fillstyle='top')
    # plt.annotate('Loss:{}%'.format(congestion_utils[3][1] * 100), xy=(10.5, plt.ylim()[1]), xytext=(9.5, plt.ylim()[1] + 1), fontsize=15)
    plt.title(key)
    plt.setp(ax1.spines.values(), lw=1.5, color='black')
    plt.savefig('../Results/results_{}/{}_{}_{}_boxPLot.pdf'.format(results_dir, results_dir_file, check, key))
    plt.clf()

def prepare_results(flows, sample_rates):
    rounds_results = {}
    # num_of_paths = 2
    num_of_paths = 1

    rounds_results['MaxEpsilonIneqDelay'] = {}
    rounds_results['MaxEpsilonIneqSuccessProb'] = {}
    rounds_results['MaxEpsilonIneqSuccessProb']['E2E_eventAvg'] = {}
    rounds_results['MaxEpsilonIneqSuccessProb']['sentTime_est'] = {}
    rounds_results['MaxEpsilonIneqSuccessProb']['poisson_sentTime_est'] = {}

    rounds_results['MaxEpsilonIneqNonMarkingProb'] = {}
    rounds_results['MaxEpsilonIneqNonMarkingProb']['E2E_eventAvg'] = {}
    rounds_results['MaxEpsilonIneqNonMarkingProb']['sentTime_est_events'] = {}
    rounds_results['MaxEpsilonIneqNonMarkingProb']['sentTime_est_probs'] = {}
    rounds_results['MaxEpsilonIneqNonMarkingProb']['possion_est_probs'] = {}
    for sample_rate in sample_rates:
        rounds_results['MaxEpsilonIneqSuccessProb']['poisson_sentTime_est'][sample_rate] = {}
    rounds_results['DropRate'] = []


    for flow in flows:
        rounds_results['MaxEpsilonIneqDelay'][flow] = {}
        rounds_results['MaxEpsilonIneqSuccessProb']['E2E_eventAvg'][flow] = {}
        rounds_results['MaxEpsilonIneqSuccessProb']['sentTime_est'][flow] = {}
        rounds_results['MaxEpsilonIneqNonMarkingProb']['E2E_eventAvg'][flow] = {}
        rounds_results['MaxEpsilonIneqNonMarkingProb']['sentTime_est_events'][flow] = {}
        rounds_results['MaxEpsilonIneqNonMarkingProb']['sentTime_est_probs'][flow] = {}
        rounds_results['MaxEpsilonIneqNonMarkingProb']['possion_est_probs'] [flow]= {}
        for sample_rate in sample_rates:
            rounds_results['MaxEpsilonIneqSuccessProb']['poisson_sentTime_est'][sample_rate][flow] = {}
        for i in range(num_of_paths):
            rounds_results['MaxEpsilonIneqDelay'][flow]['A' + str(i)] = 0
            rounds_results['MaxEpsilonIneqSuccessProb']['E2E_eventAvg'][flow]['A' + str(i)] = 0
            rounds_results['MaxEpsilonIneqSuccessProb']['sentTime_est'][flow]['A' + str(i)] = 0
            for sample_rate in sample_rates:
                rounds_results['MaxEpsilonIneqSuccessProb']['poisson_sentTime_est'][sample_rate][flow]['A' + str(i)] = 0
        rounds_results['MaxEpsilonIneqNonMarkingProb']['E2E_eventAvg'][flow]['A' + str(i)] = 0
        rounds_results['MaxEpsilonIneqNonMarkingProb']['sentTime_est_events'][flow]['A' + str(i)] = 0
        rounds_results['MaxEpsilonIneqNonMarkingProb']['sentTime_est_probs'][flow]['A' + str(i)] = 0
        rounds_results['MaxEpsilonIneqNonMarkingProb']['possion_est_probs'] [flow]['A' + str(i)] = 0
    rounds_results['experiments'] = 0
    return rounds_results

__ns3_path = "/media/experiments/ns-allinone-3.41/ns-3.41"



config = configparser.ConfigParser()
config.read('../Parameters.config')
serviceRateScales = [float(x) for x in config.get('Settings', 'serviceRateScales').split(',')]
serviceRateScales = [0.79, 0.81, 0.83, 0.85, 0.87, 0.89, 0.91, 0.93, 0.95, 0.99, 1.0, 1.01, 1.03, 1.05]
# serviceRateScales = [0.89]
# serviceRateScales = [0.85, 0.87, 0.89, 0.91, 0.93, 0.95, 0.97, 0.99, 1.01]
errorRates = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 28.0, 30.0]
# errorRates = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 10.0, 12.0, 14.0, 15.0]
# errorRates = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
# utilizationFactors = [round(6 * 300 / (r * 2 * 945), 3) for r in serviceRateScales]
utilizationFactors = [round(2 * 300 / (r * 2 * 300), 3) for r in serviceRateScales]
print("utilizationFactors: ", utilizationFactors)
# selectedUtils = [utilizationFactors[1], utilizationFactors[3], utilizationFactors[5], utilizationFactors[7], utilizationFactors[9], utilizationFactors[11], utilizationFactors[13]]
# print("serviceRateScales: ", serviceRateScales)
check = sys.argv[1]
results_dir = sys.argv[2]
# results_dir_file = 'naive'
# results_dir_file = 'delay_Results_forward_diff_loss_est_and_pckt_CDF'
results_dir_file = 'receiverF'
results = {}
flows = []
congestion_utils = []
# error_success = {}
# utilizationFactors = []
sample_rates = []
num_of_paths = 1
for rate in serviceRateScales:
# for rate in errorRates:
    # if rate >= 6.0:
    #     results[rate] = prepare_results(flows)
    #     for flow in flows:
    #         for i in range(2):
    #             results[rate]['MaxEpsilonIneqDelay'][flow]['A' + str(i)] = 0
    #             results[rate]['MaxEpsilonIneqSuccessProb'][flow]['A' + str(i)] = 0
    #     continue
    for file in os.listdir('../Results/results_' + results_dir + '/'+str(rate)+'/'):
        temp = {}
        if file.find(results_dir_file) != -1:
            with open('../Results/results_' + results_dir + '/'+str(rate)+'/'+file) as f:
                temp = js.load(f)
            flows = read_data_flowIndicator(temp)
            results[rate] = prepare_results(flows, temp['MaxEpsilonIneqSuccessProb']['poisson_sentTime_est'].keys() if 'poisson_sentTime_est' in temp['MaxEpsilonIneqSuccessProb'] else [])
            sample_rates = temp['MaxEpsilonIneqSuccessProb']['poisson_sentTime_est'].keys()
            # x = round(6 * 300 / (rate * 2 * 945), 3)
            x = round(2 * 1000 / (rate * 2 * 1000), 3)
            dropRate = 0
            for flow in flows:
                for i in range(num_of_paths):
                    results[rate]['MaxEpsilonIneqDelay'][flow]['A' + str(i)] = temp['MaxEpsilonIneqDelay'][flow]['A' + str(i)] / temp['experiments'] * 100
                    print(flow, rate, results[rate]['MaxEpsilonIneqDelay'][flow]['A' + str(i)])
                    results[rate]['MaxEpsilonIneqSuccessProb']['E2E_eventAvg'][flow]['A' + str(i)] = temp['MaxEpsilonIneqSuccessProb']['E2E_eventAvg'][flow]['A' + str(i)] / temp['experiments'] * 100
                    results[rate]['MaxEpsilonIneqSuccessProb']['sentTime_est'][flow]['A' + str(i)] = temp['MaxEpsilonIneqSuccessProb']['sentTime_est'][flow]['A' + str(i)] / temp['experiments'] * 100
                    for sample_rate in sample_rates:
                        results[rate]['MaxEpsilonIneqSuccessProb']['poisson_sentTime_est'][sample_rate][flow]['A' + str(i)] = temp['MaxEpsilonIneqSuccessProb']['poisson_sentTime_est'][sample_rate][flow]['A' + str(i)] / temp['experiments'] * 100
                results[rate]['MaxEpsilonIneqNonMarkingProb']['E2E_eventAvg'][flow]['A' + str(i)] = temp['MaxEpsilonIneqNonMarkingProb']['E2E_eventAvg'][flow]['A' + str(i)] / temp['experiments'] * 100
                results[rate]['MaxEpsilonIneqNonMarkingProb']['sentTime_est_events'][flow]['A' + str(i)] = temp['MaxEpsilonIneqNonMarkingProb']['sentTime_est_events'][flow]['A' + str(i)] / temp['experiments'] * 100
                results[rate]['MaxEpsilonIneqNonMarkingProb']['sentTime_est_probs'][flow]['A' + str(i)] = temp['MaxEpsilonIneqNonMarkingProb']['sentTime_est_probs'][flow]['A' + str(i)] / temp['experiments'] * 100
                results[rate]['MaxEpsilonIneqNonMarkingProb']['possion_est_probs'][flow]['A' + str(i)] = temp['MaxEpsilonIneqNonMarkingProb']['possion_est_probs'][flow]['A' + str(i)] / temp['experiments'] * 100
                # just for the reverse_loss_2
                if flow != 'R0H0R2H0' and flow != 'R0H1R2H1':
                    dropRate += sum([1.0 - np.mean(temp['EndToEndSuccessProb']['E2E_eventAvg'][flow]['A' + str(i)]) for i in range(num_of_paths)])
            results[rate]['experiments'] = temp['experiments']

            results[rate]['DropRate'] = temp['DropRate']
            # results[rate]['DropRate'] = dropRate
            # utilizationFactors.append(round(6 * temp['AverageWorkLoad'] / (rate * 2 * 945000000), 3))
            # congestion_utils.append([round(6 * 300 / (rate * 2 * 945), 3), round(np.mean(temp['DropRate']), 3), round(6 * temp['AverageWorkLoad'] / (rate * 2 * 945000000), 3)])
            congestion_utils.append([round(2 * 1000 / (rate * 2 * 1000), 3), round(np.mean(temp['DropRate']), 3), round(2 * temp['AverageWorkLoad'] / (rate * 2 * 100000000), 3)])
# print(error_success)
print(congestion_utils)
# print([(a, np.mean(b)) for a,b in utils_success.items()])
droprates_mean = [np.mean(value['DropRate']) for value in results.values()]
droprates_xticks = [round(x, 3) for x in droprates_mean]

# plot for forward experiment -- BoxPLots(loss):
if check == 'marking':
    for key in results[0.79]['MaxEpsilonIneqNonMarkingProb'].keys():
        data = [[] for i in range(len(utilizationFactors))]
        for flow in flows:
            # if flow != 'R0H0R2H0' and flow != 'R0H1R2H1':
            for i in range(len(utilizationFactors))[::-1]:
                data[len(utilizationFactors) - i - 1].append(max([results[serviceRateScales[i]]['MaxEpsilonIneqNonMarkingProb'][key][flow]['A' + str(j)] for j in range(num_of_paths)]))
        plot_boxplot_loss(utilizationFactors, droprates_xticks, data, key)


# plot for forward experiment -- BoxPLots(loss):
if check == 'loss':
    for key in results[0.79]['MaxEpsilonIneqSuccessProb'].keys():
        if key != 'poisson_sentTime_est':
            data = [[] for i in range(len(utilizationFactors))]
            for flow in flows:
                # if flow != 'R0H0R2H0' and flow != 'R0H1R2H1':
                for i in range(len(utilizationFactors))[::-1]:
                    data[len(utilizationFactors) - i - 1].append(max([results[serviceRateScales[i]]['MaxEpsilonIneqSuccessProb'][key][flow]['A' + str(j)] for j in range(num_of_paths)]))
            plot_boxplot_loss(utilizationFactors, droprates_xticks, data, key)
        if key == 'poisson_sentTime_est':
            for sample_rate in sample_rates:
                data = [[] for i in range(len(utilizationFactors))]
                for flow in flows:
                    # if flow != 'R0H0R2H0' and flow != 'R0H1R2H1':
                    for i in range(len(utilizationFactors))[::-1]:
                        data[len(utilizationFactors) - i - 1].append(max(results[serviceRateScales[i]]['MaxEpsilonIneqSuccessProb'][key][sample_rate][flow]['A' + str(j)] for j in range(num_of_paths)))
                plot_boxplot_loss(utilizationFactors, droprates_xticks, data, key + '_' + sample_rate)
    
# plot for forward experiment -- BoxPLots(delay): 
if check == 'delay':
    data = [[] for i in range(len(utilizationFactors))]
    for flow in flows:
        # if flow != 'R0H0R2H0' and flow != 'R0H1R2H1':
        for i in range(len(utilizationFactors))[::-1]:
            # print(flow, serviceRateScales[i], max(results[serviceRateScales[i]]['MaxEpsilonIneqDelay'][flow]['A0'], results[serviceRateScales[i]]['MaxEpsilonIneqDelay'][flow]['A1']))
            data[len(utilizationFactors) - i - 1].append(max(results[serviceRateScales[i]]['MaxEpsilonIneqDelay'][flow]['A' + str(j)] for j in range(num_of_paths)))
    plot_boxplot_loss(utilizationFactors, droprates_xticks, data, 'delay')

# # plot for reverse experiment loss_1:s
# for key in results[1.0]['MaxEpsilonIneqSuccessProb'].keys():
#     if key != 'poisson_sentTime_est':
#         plot_reverse_loss_1(results, flows, errorRates, key)
#     if key == 'poisson_sentTime_est':
#         for sample_rate in sample_rates:
#             plot_reverse_loss_1(results, flows, errorRates, key, sample_rate)

# # plot for reverse experiment loss_2:
# fig = plt.figure()
# fig.set_size_inches(20, 6)
# ax1 = fig.add_subplot(111)
# ax2 = ax1.twiny()
# for flow in flows:
#     if flow == 'R0H0R2H0':
#         ax1.plot(utilizationFactors, [100 - min(value['MaxEpsilonIneqSuccessProb'][flow]['A0'], value['MaxEpsilonIneqSuccessProb'][flow]['A1']) for value in results.values()], 'ro-')
#     if flow == 'R0H1R2H1':
#         ax1.plot(utilizationFactors, [100 - min(value['MaxEpsilonIneqSuccessProb'][flow]['A0'], value['MaxEpsilonIneqSuccessProb'][flow]['A1']) for value in results.values()], 'bo-')
# ax1.legend(['ToR1S1 -> ToR3S1', 'ToR1S2 -> ToR3S2'], fontsize=15)
# ax1.set_xlabel('Utilization Factor', fontsize=25, labelpad=-1)
# ax1.set_ylabel('Success Rate (%)', fontsize=25, labelpad=-10)
# ax1.set_xticks(utilizationFactors)
# ax1.set_xticklabels(utilizationFactors, fontsize=20)
# ax1.tick_params(axis='y', labelsize=20)
# ax1.set_ylim(0, 110)

# ax2.set_xlim(ax1.get_xlim())
# ax2.set_xticks(utilizationFactors)
# ax2.set_xticklabels(droprates_xticks, fontsize=17)
# ax2.set_xlabel('Congestion Loss Rate', fontsize=23, labelpad=10)
# plt.grid(True)
# plt.title('success rate of detecting the silent packet drop for different utulization factors')
# plt.setp(ax1.spines.values(), lw=1.5, color='black')
# plt.savefig('../results_{}/{}_success_perDropRate.pdf'.format(results_dir, check))
# plt.clf()

# # plot for reverse experiment delay_1:
# fig = plt.figure()
# fig.set_size_inches(7, 6)
# ax1 = fig.add_subplot(111)
# X = [round(r * 0.05, 3) for r in errorRates]
# for flow in flows:
#     if flow == 'R0H0R2H0':
#         ax1.plot(X, [100 - max(value['MaxEpsilonIneq'][flow]['A0'], value['MaxEpsilonIneq'][flow]['A1']) for value in results.values()], 'ro-')
#     if flow == 'R0H1R2H1':
#         ax1.plot(X, [100 - max(value['MaxEpsilonIneq'][flow]['A0'], value['MaxEpsilonIneq'][flow]['A1']) for value in results.values()], 'bo-')
# ax1.legend(['ToR1S1 -> ToR3S1', 'ToR1S2 -> ToR3S2'], fontsize=15)
# ax1.set_xlabel('Delay Prioritization Rate', fontsize=25, labelpad=-1)
# ax1.set_ylabel('Success Rate (%)', fontsize=25, labelpad=-10)
# ax1.tick_params(axis='y', labelsize=20)
# ax1.set_xticks(X[1::2])
# ax1.set_xticklabels(X[1::2], fontsize=20)
# plt.setp(ax1.spines.values(), lw=1.5, color='black')
# # plt.grid(True)
# # plt.title('success rate of detecting the silent packet drop for different silent drop rates')
# plt.savefig('../results_postProcessing_{}/{}_success_perDropRate.pdf'.format(results_dir, check))
# plt.clf()

# # plot for reverse experiment delay_2:
# fig = plt.figure()
# fig.set_size_inches(10, 6)
# ax1 = fig.add_subplot(111)
# ax2 = ax1.twiny()
# for flow in flows:
#     if flow == 'R0H0R2H0' or flow == 'R0H1R2H1':
#         ax1.plot(utilizationFactors, [100 - value['MaxEpsilonIneq'][flow]['A0'] for value in results.values()], 'o-')
# ax1.legend(['ToR0H0 -> ToR2H0', 'ToR0H1 -> ToR2H1'])
# ax1.set_xlabel('Utilization Factor', fontsize=20)
# ax1.set_ylabel('Success Rate (%)', fontsize=20)
# ax1.set_xticks(utilizationFactors[1::2])
# ax1.set_xticklabels(utilizationFactors[1::2], fontsize=15)
# ax1.tick_params(axis='y', labelsize=15)
# ax1.set_ylim(-10, 110)

# ax2.set_xlim(ax1.get_xlim())
# ax2.set_xticks(utilizationFactors)
# ax2.set_xticklabels(droprates_xticks, fontsize=10)
# ax2.set_xlabel('Average Netwrok Loss Rate', fontsize=20, labelpad=10)
# # plt.grid(True)
# # plt.title('success rate of detecting the silent packet drop for different utulization factors')
# plt.savefig('../results_postProcessing_{}/{}_success_perDropRate.pdf'.format(results_dir, check))
# plt.clf()

# # claculate Coefficient of Variation for each flow for each service rate scale
# for key in results.keys():
#     results[key]['EndToEndCV'] = {}
#     for flow in flows:
#         results[key]['EndToEndCV'][flow] = [results[key]['EndToEndStd'][flow][i] / results[key]['EndToEndMean'][flow][i] for i in range(len(results[key]['EndToEndMean'][flow]))]

# # plot the EndToEndStd per flow per service rate scale. The end to end std is a list, thus we need to have error bars
# for flow in flows:
#     plt.errorbar(droprates_mean, [np.mean(value['EndToEndStd'][flow]) for value in results.values()], yerr=[np.std(value['EndToEndStd'][flow]) for value in results.values()], xerr=droprates_std, fmt='-o')
# plt.legend(flows)
# plt.xticks(droprates_xticks)
# plt.xlabel('Drop Rate')
# plt.ylabel('End To End Delay Std')
# plt.title('End To End Delay Std for Different Drop Rates')
# plt.savefig('results/EndToEndStd.png')
# plt.clf()

# # plot the EndToEndMean per flow per service rate scale. The end to end mean is a list, thus we need to have error bars
# for flow in flows:
#     plt.errorbar(droprates_mean, [np.mean(value['EndToEndMean'][flow]) for value in results.values()], yerr=[np.std(value['EndToEndMean'][flow]) for value in results.values()], xerr=droprates_std, fmt='-o')
# plt.legend(flows)
# plt.xticks(droprates_xticks)
# plt.xlabel('Drop Rate')
# plt.ylabel('End To End Delay Mean')
# plt.title('End To End Delay Mean for Different Drop Rates')
# plt.savefig('results/EndToEndMean.png')
# plt.clf()

# # plot the EndToEndSkew per flow per service rate scale. The end to end skew is a list, thus we need to have error bars
# for flow in flows:
#     plt.errorbar(droprates_mean, [np.mean(value['EndToEndSkew'][flow]) for value in results.values()], yerr=[np.std(value['EndToEndSkew'][flow]) for value in results.values()], xerr=droprates_std, fmt='-o')
# plt.legend(flows)
# plt.xticks(droprates_xticks)
# plt.xlabel('Drop Rate')
# plt.ylabel('End To End Delay Skew')
# plt.title('End To End Delay Skew for Different Drop Rates')
# plt.savefig('results/EndToEndSkew.png')
# plt.clf()

# # plot the EndToEndCV per flow per service rate scale. The end to end cv is a list, thus we need to have error bars
# for flow in flows:
#     plt.errorbar(droprates_mean, [np.mean(value['EndToEndCV'][flow]) for value in results.values()], yerr=[np.std(value['EndToEndCV'][flow]) for value in results.values()], xerr=droprates_std, fmt='-o')
# plt.legend(flows)
# plt.xticks(droprates_xticks)
# plt.xlabel('Drop Rate')
# plt.ylabel('End To End Delay CV')
# plt.title('End To End Delay CV for Different Drop Rates')
# plt.savefig('results/EndToEndCV.png')
# plt.clf()

# # # plot the Dominant Assumption Success Rate per flow per service rate scale
# # for flow in flows:
# #     plt.errorbar(droprates_mean, [value['Overall']['samples']['DominantAssumption'][flow] for value in results.values()], xerr=droprates_std, fmt='-o')
# # plt.legend(flows)
# # plt.xticks(droprates_xticks)
# # plt.xlabel('Drop Rate')
# # plt.ylabel('Dominant Assumption Success Rate (%)')
# # plt.title('Dominant Assumption Success Rate for Different Drop Rates')
# # plt.savefig('results/DominantAssumption_perServiceRateScale.png')
# # plt.clf()

# # # plot the Dominant Assumption Success Rate per flow per std. The end to end std is a list, thus we need to have error bars
# # for flow in flows:
# #     plt.errorbar([np.mean(value['EndToEndStd'][flow]) for value in results.values()], [value['Overall']['samples']['DominantAssumption'][flow] for value in results.values()], xerr=[np.std(value['EndToEndStd'][flow]) for value in results.values()], fmt='-o')
# # plt.legend(flows)
# # plt.xlabel('End To End Delay Std')
# # plt.ylabel('Dominant Assumption Success Rate (%)')
# # plt.title('Dominant Assumption Success Rate for Different End To End Delay Stds')
# # plt.savefig('results/DominantAssumption_perEndToEndStd.png')
# # plt.clf()

# # # plot the Dominant Assumption Success Rate per flow per skew. The end to end skew is a list, thus we need to have error bars
# # for flow in flows:
# #     plt.errorbar([np.mean(value['EndToEndSkew'][flow]) for value in results.values()], [value['Overall']['samples']['DominantAssumption'][flow] for value in results.values()], xerr=[np.std(value['EndToEndSkew'][flow]) for value in results.values()], fmt='o')
# # plt.legend(flows)
# # plt.xlabel('End To End Delay Skew')
# # plt.ylabel('Dominant Assumption Success Rate (%)')
# # plt.title('Dominant Assumption Success Rate for Different End To End Delay Skews')
# # plt.savefig('results/DominantAssumption_perEndToEndSkew.png')
# # plt.clf()

# # # plot the Dominant Assumption Success Rate per flow per CV. The end to end cv is a list, thus we need to have error bars
# # for flow in flows:
# #     plt.errorbar([np.mean(value['EndToEndCV'][flow]) for value in results.values()], [value['Overall']['samples']['DominantAssumption'][flow] for value in results.values()], xerr=[np.std(value['EndToEndCV'][flow]) for value in results.values()], fmt='o')
# # plt.legend(flows)
# # plt.xlabel('End To End Delay CV')
# # plt.ylabel('Dominant Assumption Success Rate (%)')
# # plt.title('Dominant Assumption Success Rate for Different End To End Delay CVs')
# # plt.savefig('results/DominantAssumption_perEndToEndCV.png')
# # plt.clf()

# # # plot the Dominant Assumption Success Rate per flow per EndToEndStd_sumstdi. The EndToEndStd_sumstdi is a list, thus we need to have error bars
# # for flow in flows:
# #     plt.errorbar([np.mean(np.divide(value['EndToEndStd'][flow], (np.array(value['T0std']) + np.array(value['T1std'][flow])))) for value in results.values()], [value['Overall']['samples']['DominantAssumption'][flow] for value in results.values()], xerr=[np.std(np.divide(value['EndToEndStd'][flow], (np.array(value['T0std']) + np.array(value['T1std'][flow])))) for value in results.values()], fmt='-o')
# # plt.legend(flows)
# # plt.xlabel('EndToEndStd_sumstdi')
# # plt.ylabel('Dominant Assumption Success Rate (%)')
# # plt.title('Dominant Assumption Success Rate for Different EndToEndStd sum of std_i')
# # plt.savefig('results/DominantAssumption_EndToEndStd_sumstdi.png')
# # plt.clf()

# # plot EndToEndStd_sumstdi per flow per service rate scale 
# for flow in flows:
#     plt.errorbar(droprates_mean, 
#                  [np.mean(np.divide(value['EndToEndStd'][flow], (np.array(value['T0std']) + np.array(value['T1std'][flow])))) for value in results.values()], 
#                  yerr=[np.std(np.divide(value['EndToEndStd'][flow], (np.array(value['T0std']) + np.array(value['T1std'][flow])))) for value in results.values()], 
#                     xerr=droprates_std, fmt='-o')
# plt.legend(flows)
# plt.xticks(droprates_xticks)
# plt.xlabel('Drop Rate')
# plt.ylabel('$ \dfrac{\sigma_{e2e}}{\sum \sigma_{Ti}} $')
# plt.title('$ \dfrac{\sigma_{e2e}^2}{\sum \sigma_{Ti}^2} $ for Different Drop Rates')
# plt.savefig('results/EndToEndStd_sumstdi.png')
# plt.clf()

# # plot EndToEndStd2_sumstdi2 per flow per service rate scale 
# for flow in flows:
#     plt.errorbar(droprates_mean, 
#                  [np.mean(np.divide(np.array(value['EndToEndStd'][flow]) ** 2, (np.array(value['T0std']) ** 2 + np.array(value['T1std'][flow]) ** 2))) for value in results.values()], 
#                  yerr=[np.std(np.divide(np.array(value['EndToEndStd'][flow]) ** 2, (np.array(value['T0std']) ** 2 + np.array(value['T1std'][flow]) ** 2))) for value in results.values()], 
#                     xerr=droprates_std, fmt='-o')
# plt.legend(flows)
# plt.xticks(droprates_xticks)
# plt.xlabel('Drop Rate')
# plt.ylabel('$ \dfrac{\sigma_{e2e}^2}{\sum \sigma_{Ti}^2} $')
# plt.title('$ \dfrac{\sigma_{e2e}^2}{\sum \sigma_{Ti}^2} $ for Different Drop Rate')
# plt.savefig('results/EndToEndStd2_sumstdi2.png')
# plt.clf()

# # plot T0std_T1std per flow per service rate scale 
# for flow in flows:
#     plt.errorbar(droprates_mean, [np.mean(np.divide(np.array(value['T0std']), np.array(value['T1std'][flow]))) for value in results.values()], 
#                  yerr=[np.std(np.divide(np.array(value['T0std']), np.array(value['T1std'][flow]))) for value in results.values()], xerr=droprates_std, fmt='-o')
# plt.legend(flows)
# plt.xticks(droprates_xticks)
# plt.xlabel('Drop Rate')
# plt.ylabel('$ \dfrac{\sigma_{T0}}{\sigma_{T1}} $')
# plt.title('$ \dfrac{\sigma_{T0}}{\sigma_{T1}} $ for Different Drop Rates')
# plt.savefig('results/T0std_T1std.png')
# plt.clf()

# # plot EndToEndStd_std0 per flow per drop rate
# for flow in flows:
#     plt.errorbar(droprates_mean, [np.mean(np.divide(np.array(value['EndToEndStd'][flow]), np.array(value['T0std']))) for value in results.values()], 
#                  yerr=[np.std(np.divide(np.array(value['EndToEndStd'][flow]), np.array(value['T0std']))) for value in results.values()], xerr=droprates_std, fmt='-o')
# plt.legend(flows)
# plt.xticks(droprates_xticks)
# plt.xlabel('Drop Rate')
# plt.ylabel('$ \dfrac{\sigma_{e2e}}{\sigma_{T0}} $')
# plt.title('$ \dfrac{\sigma_{e2e}}{\sigma_{T0}} $ for Different Drop Rates')
# plt.savefig('results/EndToEndStd_std0.png')
# plt.clf()

# # plot DominantAssumption per flow per drop rate
# for flow in flows:
#     plt.errorbar(droprates_mean, [value['Overall']['samples']['DominantAssumption'][flow] / value['experiments'] * 100 for value in results.values()], xerr=droprates_std, fmt='-o')
# plt.legend(flows)
# plt.xlabel('Drop Rate')
# plt.ylabel('Dominant Assumption Success Rate (%)')
# plt.title('Dominant Assumption Success Rate for Different Drop Rates')
# plt.savefig('results/DominantAssumption_perServiceRateScale.png')
# plt.clf()

# # plot the RelaxedDominantAssumption Success Rate per flow drop rate
# for flow in flows:
#     plt.errorbar(droprates_mean, [value['Overall']['samples']['RelaxedDominantAssumption'][flow] / value['experiments'] * 100 for value in results.values()], xerr=droprates_std, fmt='-o')
# plt.legend(flows)
# plt.xlabel('Drop Rate')
# plt.ylabel('Relaxed Dominant Assumption Success Rate (%)')
# plt.title('Relaxed Dominant Assumption Success Rate for Different Drop Rates')
# plt.savefig('results/RelaxedDominantAssumption_perServiceRateScale.png')
# plt.clf()

# # plot the DominantAssumption Success Rate per EndToEndStd_std0
# for flow in flows:
#     plt.errorbar([np.mean(np.divide(np.array(value['EndToEndStd'][flow]), np.array(value['T0std']))) for value in results.values()], 
#                  [value['Overall']['samples']['DominantAssumption'][flow] / value['experiments'] * 100 for value in results.values()], 
#                  xerr=[np.std(np.divide(np.array(value['EndToEndStd'][flow]), np.array(value['T0std']))) for value in results.values()], fmt='o')
# plt.legend(flows)
# plt.xlabel('$ \dfrac{\sigma_{e2e}}{\sigma_{T0}} $')
# plt.ylabel('Dominant Assumption Success Rate (%)')
# plt.title('Dominant Assumption Success Rate for Different $ \dfrac{\sigma_{e2e}}{\sigma_{T0}} $ ')
# plt.savefig('results/DominantAssumption_perEndToEndStd_std0.png')
# plt.clf()

# # plot the DominantAssumption Success Rate per EndToEndStd_std0 splitted
# for flow in flows:
#     plt.errorbar([np.mean(endToEndStd_std0[flow][i]) for i in range(len(endToEndStd_std0[flow]))],
#                  [dominantAssumption[flow][i] for i in range(len(dominantAssumption[flow]))], 
#                  xerr=[np.std(endToEndStd_std0[flow][i]) for i in range(len(endToEndStd_std0[flow]))], fmt='o')
# plt.legend(flows)
# plt.xlabel('$ \dfrac{\sigma_{e2e}}{\sigma_{T0}} $')
# plt.ylabel('Dominant Assumption Success Rate (%)')
# plt.title('Dominant Assumption Success Rate for Different $ \dfrac{\sigma_{e2e}}{\sigma_{T0}} $')
# plt.savefig('results/DominantAssumption_perEndToEndStd_std0_splitted.png')
# plt.clf()

# # plot T0Ineq success rate per flow per drop rate
# # for flow in flows:
# #     plt.errorbar(droprates_mean, [value['T0Ineq'][flow] for value in results.values()], xerr=droprates_std, fmt='-o')
# # plt.legend(flows)
# # plt.xticks(droprates_xticks)
# # plt.xlabel('Drop Rate')
# # plt.ylabel('T0Ineq Success Rate (%)')
# # plt.title('T0Ineq Success Rate for Different Drop Rates')
# # plt.savefig('results/T0Ineq.png')
# # plt.clf()

# # plot T0Ineq success rate per flow per drop rate
# # for flow in flows:
# #     plt.errorbar(droprates_mean, [value['T0Ineq'][flow] for value in results.values()], xerr=droprates_std, fmt='-o')
# # plt.legend(flows)
# # plt.xticks(droprates_xticks)
# # plt.xlabel('Drop Rate')
# # plt.ylabel('T0Ineq Success Rate (%)')
# # plt.title('T0Ineq Success Rate for Different Drop Rates')
# # plt.savefig('results/T0Ineq.png')
# # plt.clf()