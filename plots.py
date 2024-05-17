import configparser
import os
import json as js
import matplotlib.pyplot as plt
import numpy as np
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
# serviceRateScales = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35]
print("serviceRateScales: ", serviceRateScales)


results = {}
for rate in serviceRateScales:
    with open('results/'+str(rate)+'/'+str(rate)+'_results.json') as f:
        results[rate] = js.load(f)


plt.plot(results.keys(), [value['ANOVA']['groundtruth'] for value in results.values()], 'b')
plt.plot(results.keys(), [value['Kruskal']['groundtruth'] for value in results.values()], 'r')
plt.legend(['ANOVA', 'Kruskal'])
plt.xticks(list(results.keys()))
plt.xlabel('Service Rate Scale')
plt.ylabel('ANOVA and Kruska success rate (%)')
plt.title('ANOVA and Kruska Results for Different Service Rate Scales')
plt.savefig('results/ANOVA_Kruskal_groundtruth.png')
plt.clf()

plt.plot(results.keys(), [value['ANOVA']['samples'] for value in results.values()], 'b')
plt.plot(results.keys(), [value['Kruskal']['samples'] for value in results.values()], 'r')
plt.legend(['ANOVA', 'Kruskal'])
plt.xticks(list(results.keys()))
plt.xlabel('Service Rate Scale')
plt.ylabel('ANOVA and Kruska success rate (%)')
plt.title('ANOVA and Kruska Results for Different Service Rate Scales')
plt.savefig('results/ANOVA_Kruskal_samples.png')
plt.clf()

flows = results[list(results.keys())[0]]['EndToEndSkew']
# claculate Coefficient of Variation for each flow for each service rate scale
for key in results.keys():
    results[key]['EndToEndCV'] = {}
    for flow in flows:
        results[key]['EndToEndCV'][flow] = [results[key]['EndToEndStd'][flow][i] / results[key]['EndToEndMean'][flow][i] for i in range(len(results[key]['EndToEndMean'][flow]))]

# plot the EndToEndStd per flow per service rate scale. The end to end std is a list, thus we need to have error bars
for flow in flows:
    plt.errorbar(list(results.keys()), [np.mean(value['EndToEndStd'][flow]) for value in results.values()], yerr=[np.std(value['EndToEndStd'][flow]) for value in results.values()], fmt='-o')
plt.legend(flows)
plt.xticks(list(results.keys()))
plt.xlabel('Service Rate Scale')
plt.ylabel('End To End Delay Std')
plt.title('End To End Delay Std for Different Service Rate Scales')
plt.savefig('results/EndToEndStd.png')
plt.clf()

# plot the EndToEndMean per flow per service rate scale. The end to end mean is a list, thus we need to have error bars
for flow in flows:
    plt.errorbar(list(results.keys()), [np.mean(value['EndToEndMean'][flow]) for value in results.values()], yerr=[np.std(value['EndToEndMean'][flow]) for value in results.values()], fmt='-o')
plt.legend(flows)
plt.xticks(list(results.keys()))
plt.xlabel('Service Rate Scale')
plt.ylabel('End To End Delay Mean')
plt.title('End To End Delay Mean for Different Service Rate Scales')
plt.savefig('results/EndToEndMean.png')
plt.clf()

# plot the EndToEndSkew per flow per service rate scale. The end to end skew is a list, thus we need to have error bars
for flow in flows:
    plt.errorbar(list(results.keys()), [np.mean(value['EndToEndSkew'][flow]) for value in results.values()], yerr=[np.std(value['EndToEndSkew'][flow]) for value in results.values()], fmt='-o')
plt.legend(flows)
plt.xticks(list(results.keys()))
plt.xlabel('Service Rate Scale')
plt.ylabel('End To End Delay Skew')
plt.title('End To End Delay Skew for Different Service Rate Scales')
plt.savefig('results/EndToEndSkew.png')
plt.clf()

# plot the EndToEndCV per flow per service rate scale. The end to end cv is a list, thus we need to have error bars
for flow in flows:
    plt.errorbar(list(results.keys()), [np.mean(value['EndToEndCV'][flow]) for value in results.values()], yerr=[np.std(value['EndToEndCV'][flow]) for value in results.values()], fmt='-o')
plt.legend(flows)
plt.xticks(list(results.keys()))
plt.xlabel('Service Rate Scale')
plt.ylabel('End To End Delay CV')
plt.title('End To End Delay CV for Different Service Rate Scales')
plt.savefig('results/EndToEndCV.png')
plt.clf()

# plot the Dominant Assumption Success Rate per flow per service rate scale
for flow in flows:
    plt.plot(list(results.keys()), [value['Overall']['samples']['DominantAssumption'][flow] for value in results.values()], '-o')
plt.legend(flows)
plt.xlabel('Service Rate Scale')
plt.ylabel('Dominant Assumption Success Rate (%)')
plt.title('Dominant Assumption Success Rate for Different Service Rate Scales')
plt.savefig('results/DominantAssumption_perServiceRateScale.png')
plt.clf()

# plot the Dominant Assumption Success Rate per flow per std. The end to end std is a list, thus we need to have error bars
for flow in flows:
    plt.errorbar([np.mean(value['EndToEndStd'][flow]) for value in results.values()], [value['Overall']['samples']['DominantAssumption'][flow] for value in results.values()], xerr=[np.std(value['EndToEndStd'][flow]) for value in results.values()], fmt='-o')
plt.legend(flows)
plt.xlabel('End To End Delay Std')
plt.ylabel('Dominant Assumption Success Rate (%)')
plt.title('Dominant Assumption Success Rate for Different End To End Delay Stds')
plt.savefig('results/DominantAssumption_perEndToEndStd.png')
plt.clf()

# plot the Dominant Assumption Success Rate per flow per skew. The end to end skew is a list, thus we need to have error bars
for flow in flows:
    plt.errorbar([np.mean(value['EndToEndSkew'][flow]) for value in results.values()], [value['Overall']['samples']['DominantAssumption'][flow] for value in results.values()], xerr=[np.std(value['EndToEndSkew'][flow]) for value in results.values()], fmt='o')
plt.legend(flows)
plt.xlabel('End To End Delay Skew')
plt.ylabel('Dominant Assumption Success Rate (%)')
plt.title('Dominant Assumption Success Rate for Different End To End Delay Skews')
plt.savefig('results/DominantAssumption_perEndToEndSkew.png')
plt.clf()

# plot the Dominant Assumption Success Rate per flow per CV. The end to end cv is a list, thus we need to have error bars
for flow in flows:
    plt.errorbar([np.mean(value['EndToEndCV'][flow]) for value in results.values()], [value['Overall']['samples']['DominantAssumption'][flow] for value in results.values()], xerr=[np.std(value['EndToEndCV'][flow]) for value in results.values()], fmt='o')
plt.legend(flows)
plt.xlabel('End To End Delay CV')
plt.ylabel('Dominant Assumption Success Rate (%)')
plt.title('Dominant Assumption Success Rate for Different End To End Delay CVs')
plt.savefig('results/DominantAssumption_perEndToEndCV.png')
plt.clf()

# plot the Dominant Assumption Success Rate per flow per EndToEndStd_sumstdi. The EndToEndStd_sumstdi is a list, thus we need to have error bars
for flow in flows:
    plt.errorbar([np.mean(value['EndToEndStd_sumstdi'][flow]) for value in results.values()], [value['Overall']['samples']['DominantAssumption'][flow] for value in results.values()], xerr=[np.std(value['EndToEndStd_sumstdi'][flow]) for value in results.values()], fmt='-o')
plt.legend(flows)
plt.xlabel('EndToEndStd_sumstdi')
plt.ylabel('Dominant Assumption Success Rate (%)')
plt.title('Dominant Assumption Success Rate for Different EndToEndStd_sumstdi')
plt.savefig('results/DominantAssumption_EndToEndStd_sumstdi.png')
plt.clf()

# plot EndToEndStd_sumstdi per flow per service rate scale 
for flow in flows:
    plt.errorbar(list(results.keys()), [np.mean(value['EndToEndStd_sumstdi'][flow]) for value in results.values()], yerr=[np.std(value['EndToEndStd_sumstdi'][flow]) for value in results.values()], fmt='-o')
plt.legend(flows)
plt.xticks(list(results.keys()))
plt.xlabel('Service Rate Scale')
plt.ylabel('EndToEndStd_sumstdi')
plt.title('EndToEndStd_sumstdi for Different Service Rate Scales')
plt.savefig('results/EndToEndStd_sumstdi.png')
plt.clf()

# plot EndToEndStd2_sumstdi2 per flow per service rate scale 
for flow in flows:
    plt.errorbar(list(results.keys()), [np.mean(value['EndToEndStd2_sumstdi2'][flow]) for value in results.values()], yerr=[np.std(value['EndToEndStd2_sumstdi2'][flow]) for value in results.values()], fmt='-o')
plt.legend(flows)
plt.xticks(list(results.keys()))
plt.xlabel('Service Rate Scale')
plt.ylabel('EndToEndStd2_sumstdi2')
plt.title('EndToEndStd2_sumstdi2 for Different Service Rate Scales')
plt.savefig('results/EndToEndStd2_sumstdi2.png')
plt.clf()

# plot T0std_T1std per flow per service rate scale 
for flow in flows:
    plt.errorbar(list(results.keys()), [np.mean(value['T0std_T1std'][flow]) for value in results.values()], yerr=[np.std(value['T0std_T1std'][flow]) for value in results.values()], fmt='-o')
plt.legend(flows)
plt.xticks(list(results.keys()))
plt.xlabel('Service Rate Scale')
plt.ylabel('T0std_T1std')
plt.title('T0std_T1std for Different Service Rate Scales')
plt.savefig('results/T0std_T1std.png')
plt.clf()

# plot the RelaxedDominantAssumption Success Rate per flow per service rate scale
for flow in flows:
    plt.plot(list(results.keys()), [value['Overall']['samples']['RelaxedDominantAssumption'][flow] for value in results.values()], '-o')
plt.legend(flows)
plt.xlabel('Service Rate Scale')
plt.ylabel('Relaxed Dominant Assumption Success Rate (%)')
plt.title('Relaxed Dominant Assumption Success Rate for Different Service Rate Scales')
plt.savefig('results/RelaxedDominantAssumption_perServiceRateScale.png')
plt.clf()