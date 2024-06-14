import configparser
import os
import json as js
import matplotlib.pyplot as plt
import numpy as np
import glob


def read_data_flowIndicator(__ns3_path, rate):
    flows_name = []
    file_paths = glob.glob('{}/scratch/Results/{}/0/*_EndToEnd.csv'.format(__ns3_path, rate))
    for file_path in file_paths:
        flows_name.append(file_path.split('/')[-1].split('_')[0])
    return flows_name

def prepare_results(flows):
    rounds_results = {}
    rounds_results['Overall'] = {}
    rounds_results['PerTrafficStream'] = {}
    rounds_results['Overall']['samples'] = {}
    rounds_results['Overall']['samples']['DominantAssumption'] = {}
    rounds_results['Overall']['samples']['General'] = {}
    rounds_results['Overall']['samples']['RelaxedDominantAssumption'] = {}
    rounds_results['Overall']['samples']['Basic'] = {}
    rounds_results['Overall']['samples']['Basic_timeAvg'] = {}
    rounds_results['ANOVA'] = {}
    rounds_results['Kruskal'] = {}
    rounds_results['ANOVA']['samples'] = 0
    rounds_results['Kruskal']['samples'] = 0
    rounds_results['ANOVA']['groundtruth'] = 0
    rounds_results['Kruskal']['groundtruth'] = 0
    rounds_results['EndToEndMean'] = {}
    rounds_results['EndToEndStd'] = {}
    rounds_results['EndToEndSkew'] = {}
    rounds_results['DropRate'] = []
    rounds_results['T0std'] = []
    rounds_results['T1std'] = {}
    rounds_results['T0Ineq'] = {}
    rounds_results['T0IneqMaxEpsilon'] = {}
    # rounds_results['T0IeqHighRate'] = {}
    # rounds_results['T0IneqRegular'] = {}
    rounds_results['T0IneqRemovedZeroes'] = {}
    rounds_results['T1Ineq'] = {}
    rounds_results['T1IneqMaxEpsilon'] = {}
    # rounds_results['T1IeqHighRate'] = {}
    # rounds_results['T1IneqRegular'] = {}
    rounds_results['T1IneqRemovedZeroes'] = {}
    for flow in flows:
        rounds_results['Overall']['samples']['DominantAssumption'][flow] = 0
        rounds_results['Overall']['samples']['General'][flow] = 0
        rounds_results['Overall']['samples']['RelaxedDominantAssumption'][flow] = 0
        rounds_results['Overall']['samples']['Basic'][flow] = 0
        rounds_results['Overall']['samples']['Basic_timeAvg'][flow] = 0
        rounds_results['EndToEndMean'][flow] = []
        rounds_results['EndToEndStd'][flow] = []
        rounds_results['EndToEndSkew'][flow] = []
        rounds_results['T1std'][flow] = []
        rounds_results['T0Ineq'][flow] = 0
        # rounds_results['T0IeqHighRate'][flow] = 0
        # rounds_results['T0IneqRegular'][flow] = 0
        rounds_results['T0IneqRemovedZeroes'][flow] = 0
        rounds_results['T0IneqMaxEpsilon'][flow] = 0
        rounds_results['T1Ineq'][flow] = 0
        # rounds_results['T1IeqHighRate'][flow] = 0
        # rounds_results['T1IneqRegular'][flow] = 0
        rounds_results['T1IneqRemovedZeroes'][flow] = 0
        rounds_results['T1IneqMaxEpsilon'][flow] = 0


    rounds_results['experiments'] = 0
    return rounds_results

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (40, 20)
plt.rcParams.update({
    "lines.color": "black",
    "patch.edgecolor": "black",
    "text.color": "black",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "grid.color": "gray",
    "figure.facecolor": "white",
    "figure.edgecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "font.size": 30,
    "xtick.labelsize":30,
    "ytick.labelsize":30,
    "lines.linewidth":3.,
    "legend.fontsize": 30,
    })

__ns3_path = os.popen('locate "ns-3.41" | grep /ns-3.41$').read().splitlines()[0]


config = configparser.ConfigParser()
config.read('Parameters.config')
serviceRateScales = [float(x) for x in config.get('Settings', 'serviceRateScales').split(',')]
# serviceRateScales = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1]
# print("serviceRateScales: ", serviceRateScales)


results = {}
endToEndStd_std0 = {}
dominantAssumption = {}
flows = read_data_flowIndicator(__ns3_path, serviceRateScales[0])
for flow in flows:
    dominantAssumption[flow] = []
    endToEndStd_std0[flow] = []

for rate in serviceRateScales:
    # print("rate: ", rate)
    results[rate] = prepare_results(flows)
    # iterarat over all json files in results/'+str(rate)+'/'
    for file in os.listdir('results/'+str(rate)+'/'):
        temp = {}
        if file.endswith('.json'):
            # print("file: ", file)
            with open('results/'+str(rate)+'/'+file) as f:
                temp = js.load(f)
            results[rate]['ANOVA']['samples'] += temp['ANOVA']['samples']
            results[rate]['Kruskal']['samples'] += temp['Kruskal']['samples']
            results[rate]['ANOVA']['groundtruth'] += temp['ANOVA']['groundtruth']
            results[rate]['Kruskal']['groundtruth'] += temp['Kruskal']['groundtruth']

            for flow in flows:
                results[rate]['Overall']['samples']['DominantAssumption'][flow] += temp['Overall']['samples']['DominantAssumption'][flow]
                results[rate]['Overall']['samples']['General'][flow] += temp['Overall']['samples']['General'][flow]
                results[rate]['Overall']['samples']['RelaxedDominantAssumption'][flow] += temp['Overall']['samples']['RelaxedDominantAssumption'][flow]
                results[rate]['Overall']['samples']['Basic'][flow] += temp['Overall']['samples']['Basic'][flow]
                results[rate]['Overall']['samples']['Basic_timeAvg'][flow] += temp['Overall']['samples']['Basic_timeAvg'][flow]
                results[rate]['EndToEndMean'][flow] += temp['EndToEndMean'][flow]
                results[rate]['EndToEndStd'][flow] += temp['EndToEndStd'][flow]
                results[rate]['EndToEndSkew'][flow] += temp['EndToEndSkew'][flow]
                results[rate]['T1std'][flow] += temp['T1std'][flow]
                results[rate]['T0Ineq'][flow] += temp['T0Ineq'][flow]
                results[rate]['T1Ineq'][flow] += temp['T1Ineq'][flow]

                endToEndStd_std0[flow].append(np.divide(np.array(temp['EndToEndStd'][flow]), np.array(temp['T0std'])))
                dominantAssumption[flow].append(temp['Overall']['samples']['DominantAssumption'][flow] / temp['experiments'] * 100)
            
            results[rate]['DropRate'] += temp['DropRate']
            results[rate]['T0std'] += temp['T0std']
            results[rate]['experiments'] += temp['experiments']

    # with open('results/'+str(rate)+'/'+str(rate)+'_results.json') as f:
    #     results[rate] = js.load(f)


droprates_mean = [np.mean(value['DropRate']) for value in results.values()]
droprates_std = [np.std(value['DropRate']) for value in results.values()]
# # 30 poin beween min and max with 2 decimal points
# droprates_xticks = np.linspace(np.min(droprates_mean), np.max(droprates_mean), 35)
# droprates_xticks = np.round(droprates_xticks, 2)
droprates_xticks = [round(x, 3) for x in droprates_mean]

# plot drop rate over the different service rate scales
plt.errorbar(list(results.keys()), droprates_mean, yerr=droprates_std)
plt.xlabel('Service Rate Scale')
plt.ylabel('Drop Rate')
plt.title('Drop Rate for Different Service Rate Scales')
plt.savefig('results/DropRate.png')
plt.clf()

# plot the ANOVA and Kruskal success rate per service rate scale
plt.errorbar(droprates_mean, [value['ANOVA']['groundtruth'] / value['experiments'] * 100 for value in results.values()], xerr=droprates_std, color='b')
plt.errorbar(droprates_mean, [value['Kruskal']['groundtruth'] / value['experiments'] * 100 for value in results.values()], xerr=droprates_std, color='r')
plt.legend(['ANOVA', 'Kruskal'])
plt.xticks(droprates_xticks)
plt.xlabel('Drop Rate')
plt.ylabel('ANOVA and Kruska success rate (%)')
plt.title('ANOVA and Kruska Results for Different Drop Rates')
plt.savefig('results/ANOVA_Kruskal_groundtruth.png')
plt.clf()

plt.errorbar(droprates_mean, [value['ANOVA']['samples'] / value['experiments'] * 100 for value in results.values()], xerr=droprates_std, color='b')
plt.errorbar(droprates_mean, [value['Kruskal']['samples'] / value['experiments'] * 100 for value in results.values()], xerr=droprates_std, color='r')
plt.legend(['ANOVA', 'Kruskal'])
plt.xticks(droprates_xticks)
plt.xlabel('Drop Rate')
plt.ylabel('ANOVA and Kruska success rate (%)')
plt.title('ANOVA and Kruska Results for Different Drop Rates')
plt.savefig('results/ANOVA_Kruskal_samples.png')
plt.clf()

flows = results[list(results.keys())[0]]['EndToEndSkew']
# plot overall samples general success rate per drop rate
for flow in flows:
    plt.errorbar(droprates_mean, [value['Overall']['samples']['General'][flow] for value in results.values()], xerr=droprates_std, fmt='-o')
plt.legend(flows)
plt.xticks(droprates_xticks)
plt.xlabel('Drop Rate')
plt.ylabel('General Success Rate (%)')
plt.title('General Success Rate for Different Drop Rates')
plt.savefig('results/General_perDropRate.png')
plt.clf()

# claculate Coefficient of Variation for each flow for each service rate scale
for key in results.keys():
    results[key]['EndToEndCV'] = {}
    for flow in flows:
        results[key]['EndToEndCV'][flow] = [results[key]['EndToEndStd'][flow][i] / results[key]['EndToEndMean'][flow][i] for i in range(len(results[key]['EndToEndMean'][flow]))]

# plot the EndToEndStd per flow per service rate scale. The end to end std is a list, thus we need to have error bars
for flow in flows:
    plt.errorbar(droprates_mean, [np.mean(value['EndToEndStd'][flow]) for value in results.values()], yerr=[np.std(value['EndToEndStd'][flow]) for value in results.values()], xerr=droprates_std, fmt='-o')
plt.legend(flows)
plt.xticks(droprates_xticks)
plt.xlabel('Drop Rate')
plt.ylabel('End To End Delay Std')
plt.title('End To End Delay Std for Different Drop Rates')
plt.savefig('results/EndToEndStd.png')
plt.clf()

# plot the EndToEndMean per flow per service rate scale. The end to end mean is a list, thus we need to have error bars
for flow in flows:
    plt.errorbar(droprates_mean, [np.mean(value['EndToEndMean'][flow]) for value in results.values()], yerr=[np.std(value['EndToEndMean'][flow]) for value in results.values()], xerr=droprates_std, fmt='-o')
plt.legend(flows)
plt.xticks(droprates_xticks)
plt.xlabel('Drop Rate')
plt.ylabel('End To End Delay Mean')
plt.title('End To End Delay Mean for Different Drop Rates')
plt.savefig('results/EndToEndMean.png')
plt.clf()

# plot the EndToEndSkew per flow per service rate scale. The end to end skew is a list, thus we need to have error bars
for flow in flows:
    plt.errorbar(droprates_mean, [np.mean(value['EndToEndSkew'][flow]) for value in results.values()], yerr=[np.std(value['EndToEndSkew'][flow]) for value in results.values()], xerr=droprates_std, fmt='-o')
plt.legend(flows)
plt.xticks(droprates_xticks)
plt.xlabel('Drop Rate')
plt.ylabel('End To End Delay Skew')
plt.title('End To End Delay Skew for Different Drop Rates')
plt.savefig('results/EndToEndSkew.png')
plt.clf()

# plot the EndToEndCV per flow per service rate scale. The end to end cv is a list, thus we need to have error bars
for flow in flows:
    plt.errorbar(droprates_mean, [np.mean(value['EndToEndCV'][flow]) for value in results.values()], yerr=[np.std(value['EndToEndCV'][flow]) for value in results.values()], xerr=droprates_std, fmt='-o')
plt.legend(flows)
plt.xticks(droprates_xticks)
plt.xlabel('Drop Rate')
plt.ylabel('End To End Delay CV')
plt.title('End To End Delay CV for Different Drop Rates')
plt.savefig('results/EndToEndCV.png')
plt.clf()

# # plot the Dominant Assumption Success Rate per flow per service rate scale
# for flow in flows:
#     plt.errorbar(droprates_mean, [value['Overall']['samples']['DominantAssumption'][flow] for value in results.values()], xerr=droprates_std, fmt='-o')
# plt.legend(flows)
# plt.xticks(droprates_xticks)
# plt.xlabel('Drop Rate')
# plt.ylabel('Dominant Assumption Success Rate (%)')
# plt.title('Dominant Assumption Success Rate for Different Drop Rates')
# plt.savefig('results/DominantAssumption_perServiceRateScale.png')
# plt.clf()

# # plot the Dominant Assumption Success Rate per flow per std. The end to end std is a list, thus we need to have error bars
# for flow in flows:
#     plt.errorbar([np.mean(value['EndToEndStd'][flow]) for value in results.values()], [value['Overall']['samples']['DominantAssumption'][flow] for value in results.values()], xerr=[np.std(value['EndToEndStd'][flow]) for value in results.values()], fmt='-o')
# plt.legend(flows)
# plt.xlabel('End To End Delay Std')
# plt.ylabel('Dominant Assumption Success Rate (%)')
# plt.title('Dominant Assumption Success Rate for Different End To End Delay Stds')
# plt.savefig('results/DominantAssumption_perEndToEndStd.png')
# plt.clf()

# # plot the Dominant Assumption Success Rate per flow per skew. The end to end skew is a list, thus we need to have error bars
# for flow in flows:
#     plt.errorbar([np.mean(value['EndToEndSkew'][flow]) for value in results.values()], [value['Overall']['samples']['DominantAssumption'][flow] for value in results.values()], xerr=[np.std(value['EndToEndSkew'][flow]) for value in results.values()], fmt='o')
# plt.legend(flows)
# plt.xlabel('End To End Delay Skew')
# plt.ylabel('Dominant Assumption Success Rate (%)')
# plt.title('Dominant Assumption Success Rate for Different End To End Delay Skews')
# plt.savefig('results/DominantAssumption_perEndToEndSkew.png')
# plt.clf()

# # plot the Dominant Assumption Success Rate per flow per CV. The end to end cv is a list, thus we need to have error bars
# for flow in flows:
#     plt.errorbar([np.mean(value['EndToEndCV'][flow]) for value in results.values()], [value['Overall']['samples']['DominantAssumption'][flow] for value in results.values()], xerr=[np.std(value['EndToEndCV'][flow]) for value in results.values()], fmt='o')
# plt.legend(flows)
# plt.xlabel('End To End Delay CV')
# plt.ylabel('Dominant Assumption Success Rate (%)')
# plt.title('Dominant Assumption Success Rate for Different End To End Delay CVs')
# plt.savefig('results/DominantAssumption_perEndToEndCV.png')
# plt.clf()

# # plot the Dominant Assumption Success Rate per flow per EndToEndStd_sumstdi. The EndToEndStd_sumstdi is a list, thus we need to have error bars
# for flow in flows:
#     plt.errorbar([np.mean(np.divide(value['EndToEndStd'][flow], (np.array(value['T0std']) + np.array(value['T1std'][flow])))) for value in results.values()], [value['Overall']['samples']['DominantAssumption'][flow] for value in results.values()], xerr=[np.std(np.divide(value['EndToEndStd'][flow], (np.array(value['T0std']) + np.array(value['T1std'][flow])))) for value in results.values()], fmt='-o')
# plt.legend(flows)
# plt.xlabel('EndToEndStd_sumstdi')
# plt.ylabel('Dominant Assumption Success Rate (%)')
# plt.title('Dominant Assumption Success Rate for Different EndToEndStd sum of std_i')
# plt.savefig('results/DominantAssumption_EndToEndStd_sumstdi.png')
# plt.clf()

# plot EndToEndStd_sumstdi per flow per service rate scale 
for flow in flows:
    plt.errorbar(droprates_mean, 
                 [np.mean(np.divide(value['EndToEndStd'][flow], (np.array(value['T0std']) + np.array(value['T1std'][flow])))) for value in results.values()], 
                 yerr=[np.std(np.divide(value['EndToEndStd'][flow], (np.array(value['T0std']) + np.array(value['T1std'][flow])))) for value in results.values()], 
                    xerr=droprates_std, fmt='-o')
plt.legend(flows)
plt.xticks(droprates_xticks)
plt.xlabel('Drop Rate')
plt.ylabel('$ \dfrac{\sigma_{e2e}}{\sum \sigma_{Ti}} $')
plt.title('$ \dfrac{\sigma_{e2e}^2}{\sum \sigma_{Ti}^2} $ for Different Drop Rates')
plt.savefig('results/EndToEndStd_sumstdi.png')
plt.clf()

# plot EndToEndStd2_sumstdi2 per flow per service rate scale 
for flow in flows:
    plt.errorbar(droprates_mean, 
                 [np.mean(np.divide(np.array(value['EndToEndStd'][flow]) ** 2, (np.array(value['T0std']) ** 2 + np.array(value['T1std'][flow]) ** 2))) for value in results.values()], 
                 yerr=[np.std(np.divide(np.array(value['EndToEndStd'][flow]) ** 2, (np.array(value['T0std']) ** 2 + np.array(value['T1std'][flow]) ** 2))) for value in results.values()], 
                    xerr=droprates_std, fmt='-o')
plt.legend(flows)
plt.xticks(droprates_xticks)
plt.xlabel('Drop Rate')
plt.ylabel('$ \dfrac{\sigma_{e2e}^2}{\sum \sigma_{Ti}^2} $')
plt.title('$ \dfrac{\sigma_{e2e}^2}{\sum \sigma_{Ti}^2} $ for Different Drop Rate')
plt.savefig('results/EndToEndStd2_sumstdi2.png')
plt.clf()

# plot T0std_T1std per flow per service rate scale 
for flow in flows:
    plt.errorbar(droprates_mean, [np.mean(np.divide(np.array(value['T0std']), np.array(value['T1std'][flow]))) for value in results.values()], 
                 yerr=[np.std(np.divide(np.array(value['T0std']), np.array(value['T1std'][flow]))) for value in results.values()], xerr=droprates_std, fmt='-o')
plt.legend(flows)
plt.xticks(droprates_xticks)
plt.xlabel('Drop Rate')
plt.ylabel('$ \dfrac{\sigma_{T0}}{\sigma_{T1}} $')
plt.title('$ \dfrac{\sigma_{T0}}{\sigma_{T1}} $ for Different Drop Rates')
plt.savefig('results/T0std_T1std.png')
plt.clf()

# plot EndToEndStd_std0 per flow per drop rate
for flow in flows:
    plt.errorbar(droprates_mean, [np.mean(np.divide(np.array(value['EndToEndStd'][flow]), np.array(value['T0std']))) for value in results.values()], 
                 yerr=[np.std(np.divide(np.array(value['EndToEndStd'][flow]), np.array(value['T0std']))) for value in results.values()], xerr=droprates_std, fmt='-o')
plt.legend(flows)
plt.xticks(droprates_xticks)
plt.xlabel('Drop Rate')
plt.ylabel('$ \dfrac{\sigma_{e2e}}{\sigma_{T0}} $')
plt.title('$ \dfrac{\sigma_{e2e}}{\sigma_{T0}} $ for Different Drop Rates')
plt.savefig('results/EndToEndStd_std0.png')
plt.clf()

# plot DominantAssumption per flow per drop rate
for flow in flows:
    plt.errorbar(droprates_mean, [value['Overall']['samples']['DominantAssumption'][flow] / value['experiments'] * 100 for value in results.values()], xerr=droprates_std, fmt='-o')
plt.legend(flows)
plt.xlabel('Drop Rate')
plt.ylabel('Dominant Assumption Success Rate (%)')
plt.title('Dominant Assumption Success Rate for Different Drop Rates')
plt.savefig('results/DominantAssumption_perServiceRateScale.png')
plt.clf()

# plot the RelaxedDominantAssumption Success Rate per flow drop rate
for flow in flows:
    plt.errorbar(droprates_mean, [value['Overall']['samples']['RelaxedDominantAssumption'][flow] / value['experiments'] * 100 for value in results.values()], xerr=droprates_std, fmt='-o')
plt.legend(flows)
plt.xlabel('Drop Rate')
plt.ylabel('Relaxed Dominant Assumption Success Rate (%)')
plt.title('Relaxed Dominant Assumption Success Rate for Different Drop Rates')
plt.savefig('results/RelaxedDominantAssumption_perServiceRateScale.png')
plt.clf()

# plot the DominantAssumption Success Rate per EndToEndStd_std0
for flow in flows:
    plt.errorbar([np.mean(np.divide(np.array(value['EndToEndStd'][flow]), np.array(value['T0std']))) for value in results.values()], 
                 [value['Overall']['samples']['DominantAssumption'][flow] / value['experiments'] * 100 for value in results.values()], 
                 xerr=[np.std(np.divide(np.array(value['EndToEndStd'][flow]), np.array(value['T0std']))) for value in results.values()], fmt='o')
plt.legend(flows)
plt.xlabel('$ \dfrac{\sigma_{e2e}}{\sigma_{T0}} $')
plt.ylabel('Dominant Assumption Success Rate (%)')
plt.title('Dominant Assumption Success Rate for Different $ \dfrac{\sigma_{e2e}}{\sigma_{T0}} $ ')
plt.savefig('results/DominantAssumption_perEndToEndStd_std0.png')
plt.clf()

# plot the DominantAssumption Success Rate per EndToEndStd_std0 splitted
for flow in flows:
    plt.errorbar([np.mean(endToEndStd_std0[flow][i]) for i in range(len(endToEndStd_std0[flow]))],
                 [dominantAssumption[flow][i] for i in range(len(dominantAssumption[flow]))], 
                 xerr=[np.std(endToEndStd_std0[flow][i]) for i in range(len(endToEndStd_std0[flow]))], fmt='o')
plt.legend(flows)
plt.xlabel('$ \dfrac{\sigma_{e2e}}{\sigma_{T0}} $')
plt.ylabel('Dominant Assumption Success Rate (%)')
plt.title('Dominant Assumption Success Rate for Different $ \dfrac{\sigma_{e2e}}{\sigma_{T0}} $')
plt.savefig('results/DominantAssumption_perEndToEndStd_std0_splitted.png')
plt.clf()

# plot T0Ineq success rate per flow per drop rate
# for flow in flows:
#     plt.errorbar(droprates_mean, [value['T0Ineq'][flow] for value in results.values()], xerr=droprates_std, fmt='-o')
# plt.legend(flows)
# plt.xticks(droprates_xticks)
# plt.xlabel('Drop Rate')
# plt.ylabel('T0Ineq Success Rate (%)')
# plt.title('T0Ineq Success Rate for Different Drop Rates')
# plt.savefig('results/T0Ineq.png')
# plt.clf()

# plot T0Ineq success rate per flow per drop rate
# for flow in flows:
#     plt.errorbar(droprates_mean, [value['T0Ineq'][flow] for value in results.values()], xerr=droprates_std, fmt='-o')
# plt.legend(flows)
# plt.xticks(droprates_xticks)
# plt.xlabel('Drop Rate')
# plt.ylabel('T0Ineq Success Rate (%)')
# plt.title('T0Ineq Success Rate for Different Drop Rates')
# plt.savefig('results/T0Ineq.png')
# plt.clf()