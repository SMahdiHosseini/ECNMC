import configparser
import os
import json as js
import matplotlib.pyplot as plt

__ns3_path = os.popen('locate "ns-3.41" | grep /ns-3.41$').read().splitlines()[0]


config = configparser.ConfigParser()
config.read('Parameters.config')
serviceRateScales = [float(x) for x in config.get('Settings', 'serviceRateScales').split(',')]
print("serviceRateScales: ", serviceRateScales)


results = {}
for rate in serviceRateScales:
    # read the jason file in results/rate/rate_results.json
    with open('results/'+str(rate)+'/'+str(rate)+'_results.json') as f:
        results[rate] = js.load(f)

# for each rate we have an ANOVA result(a number). plot the anove results that the x axis is the rate and y axis is the ANOVA result
plt.plot(results.keys(), [value['ANOVA'] / 100 for value in results.values()])
plt.plot(results.keys(), [value['Kruskal'] / 100 for value in results.values()], 'r')
plt.legend(['ANOVA', 'Kruskal'])
plt.xticks(list(results.keys()))
plt.xlabel('Service Rate Scale')
plt.ylabel('ANOVA and Kruska success rate (%)')
plt.title('ANOVA and Kruska Results for Different Service Rate Scales')
plt.show()