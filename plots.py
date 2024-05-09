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
print("serviceRateScales: ", serviceRateScales)


results = {}
for rate in serviceRateScales:
    with open('results/'+str(rate)+'/'+str(rate)+'_results.json') as f:
        results[rate] = js.load(f)


plt.plot(results.keys(), [value['ANOVA'] for value in results.values()], 'b')
plt.plot(results.keys(), [value['Kruskal'] for value in results.values()], 'r')
plt.legend(['ANOVA', 'Kruskal'])
plt.xticks(list(results.keys()))
plt.xlabel('Service Rate Scale')
plt.ylabel('ANOVA and Kruska success rate (%)')
plt.title('ANOVA and Kruska Results for Different Service Rate Scales')
plt.savefig('results/ANOVA_Kruskal.png')
plt.clf()

flows = results[list(results.keys())[0]]['EndToEndSkew']

# plot the sqrt of EndToEndStd 
for flow in flows:
    plt.plot(results.keys(), [value['EndToEndStd'][flow] / value['EndToEndMean'][flow] for value in results.values()])
plt.legend(flows)
plt.xticks(list(results.keys()))
plt.xlabel('Service Rate Scale')
plt.ylabel('EndToEndStd')
plt.title('EndToEndStd for Different Service Rate Scales')
plt.savefig('results/EndToEndStd.png')
plt.clf()

# for flow in flows:
#     plt.plot([value['EndToEndStd'][flow] / value['EndToEndMean'][flow] for value in results.values()], [value['Overall']['samples']['DominantAssumption'][flow] for value in results.values()], 'o')

# plt.legend(flows)
# plt.xlabel('End to End std')
# plt.ylabel('Dominant Assumption Success Rate (%)')
# plt.title('Dominant Assumption Success Rate for Different End to End std')
# plt.savefig('results/DominantAssumption.png')