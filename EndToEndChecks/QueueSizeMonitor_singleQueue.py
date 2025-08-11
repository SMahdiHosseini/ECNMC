from Utils import *
from BiasCalculation import *
import pandas as pd
import glob
import configparser
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import anderson
from scipy.stats import f_oneway, kruskal
import json as js
import multiprocessing
import argparse

# __ns3_path = os.popen('locate "ns-3.41" | grep /ns-3.41$').read().splitlines()[0]
__ns3_path = "/media/experiments/ns-allinone-3.41/ns-3.41"
errorRate = 0.05
difference = 1.30
# sample_rate = 0.30
sample_rates = [0.5]
confidenceValue = 1.96 # 95% confidence interval
propagationDelay = 50000

timeAvg_methods = ['rightCont_timeAvg', 'leftCont_timeAvg', 'linearInterp_timeAvg', 'poisson_eventAvg', 'eventAvg']
# timeAvg_methods = ['rightCont_timeAvg', 'leftCont_timeAvg', 'linearInterp_timeAvg']
delay_timeAvg_vars = ['event']
successProb_timeAvg_vars = ['event', 'probability']
# successProb_timeAvg_vars = ['probability']
nonMarkingProb_timeAvg_vars = ['event']
min_sample_size = 15

def prepare_results(flows, queues, num_of_paths):
    rounds_results = {}
    for q in queues:
        rounds_results[q+'Occupancy'] = []
        rounds_results[q+'PacktsInQueue'] = []
        rounds_results[q+'EmptyFrac'] = []
        rounds_results[q+'GT1PktsFrac'] = []
    return rounds_results


            
def analyze_single_experiment(return_dict, rate, queues_names, confidenceValue, steadyStart, steadyEnd, rounds_results, results_folder, config, experiment=0, ns3_path=__ns3_path, differentiationDelay=None, errorRate=None, load=None):
    bottleneckLinkRate = convert_to_float(config.get('SingleQueue', 'bottleneckLinkRate')) * rate * 1e-3
    swtichDstREDQueueDiscMaxSize = convert_to_float(config.get('Settings', 'swtichDstREDQueueDiscMaxSize'))
    num_of_paths = 1
    samplesSats = calculate_offline_computations(__ns3_path, rate, 'PoissonSampler_events', str(experiment), results_folder, steadyStart, steadyEnd, "Time", linksRates=[bottleneckLinkRate], swtichDstREDQueueDiscMaxSize=swtichDstREDQueueDiscMaxSize, differentiationDelay=differentiationDelay, errorRate=errorRate, load=load)
              
    for q in queues_names:
        # if q[0] == 'S' and q[1] == 'D':
        rounds_results[q+'Occupancy'].append(samplesSats[q]['Occupancy'])
        rounds_results[q+'PacktsInQueue'].append(samplesSats[q]['PacktsInQueue'])
        rounds_results[q+'EmptyFrac'].append(samplesSats[q]['EmptyFrac'])
        rounds_results[q+'GT1PktsFrac'].append(samplesSats[q]['GT1PktsFrac'])

    return_dict[experiment] = rounds_results

def merge_results(return_dict, merged_results, flows, queues, num_of_paths):
    for exp in return_dict.keys():
        for q in queues:
            # if q[0] == 'S' and q[1] == 'D':
            merged_results[q+'Occupancy'] += return_dict[exp][q+'Occupancy']
            merged_results[q+'PacktsInQueue'] += return_dict[exp][q+'PacktsInQueue']
            merged_results[q+'EmptyFrac'] += return_dict[exp][q+'EmptyFrac']
            merged_results[q+'GT1PktsFrac'] += return_dict[exp][q+'GT1PktsFrac']
    
def analyze_all_experiments(rate, steadyStart, steadyEnd, confidenceValue, dir, config, experiments_end=3, ns3_path=__ns3_path, load=None, differentiationDelay=None, errorRate=None):
    results_folder = 'Results_' + dir
    num_of_paths = 1
    flows_name = read_data_flowIndicator(ns3_path, rate, results_folder, differentiationDelay=differentiationDelay, errorRate=errorRate, load=load)
    flows_name.sort()

    queues_names = read_queues_indicators(ns3_path, rate, results_folder, differentiationDelay=differentiationDelay, errorRate=errorRate, load=load)
    queues_names.sort()

    rounds_results = prepare_results(flows_name, queues_names, num_of_paths)
    merged_results = prepare_results(flows_name, queues_names, num_of_paths)
    batch_size = 50
    for i in range(int(experiments_end / batch_size) + 1):
        ths = []
        return_dict = multiprocessing.Manager().dict()
        for experiment in range(batch_size * i, min(experiments_end, batch_size * (i + 1))):
            if differentiationDelay is not None and errorRate is not None:
                if len(os.listdir('{}/scratch/{}/{}/{}/D_{}/f_{}/{}'.format(__ns3_path, results_folder, rate, load, differentiationDelay, errorRate, experiment))) == 0:
                    print(experiment)
                    continue
            else:
                if len(os.listdir('{}/scratch/{}/{}/{}/{}'.format(__ns3_path, results_folder, rate, load, experiment))) == 0:
                    print(experiment)
                    continue
            print("Analyzing experiment: ", experiment)
            ths.append(multiprocessing.Process(target=analyze_single_experiment, args=(return_dict, rate, queues_names, confidenceValue, steadyStart, steadyEnd, rounds_results, results_folder, config, experiment, ns3_path, differentiationDelay, errorRate, load)))
        
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        merge_results(return_dict, merged_results, flows_name, queues_names, num_of_paths)
        print("{} joind".format(i))
    if differentiationDelay is not None and errorRate is not None:
        with open('../Results/results_{}/{}/{}/D_{}/f_{}/Q_e_m_e2e_5RTT_notall_switch_1.0_{}_{}_to_{}.json'.format(dir, rate, load, differentiationDelay, errorRate, experiments_end, steadyStart, steadyEnd), 'r') as f:
            temp = js.load(f)
            with open('../Results/results_{}/{}/{}/D_{}/f_{}/Q_e_m_e2e_5RTT_notall_switch_1.0_{}_{}_to_{}.json'.format(dir, rate, load, differentiationDelay, errorRate, experiments_end, steadyStart, steadyEnd), 'w') as ff:
                temp.update(merged_results)
                js.dump(temp, ff, indent=4)
    else:
        with open('../Results/results_{}/{}/{}/Q_e_m_e2e_5RTT_notall_switch_1.0_{}_{}_to_{}.json'.format(dir, rate, load, experiments_end, steadyStart, steadyEnd), 'r') as f:
            temp = js.load(f)
            with open('../Results/results_{}/{}/{}/Q_e_m_e2e_5RTT_notall_switch_1.0_{}_{}_to_{}.json'.format(dir, rate, load, experiments_end, steadyStart, steadyEnd), 'w') as ff:
                temp.update(merged_results)
                js.dump(temp, ff, indent=4)

# main function
def __main__():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dir",
                    required=True,
                    dest="dir",
                    help="The directory of the results",
                    default="")

    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read('../Results/results_{}/Parameters.config'.format(args.dir))
    # steadyStart = convert_to_float(config.get('Settings', 'steadyStart')) * 1e9
    # steadyEnd = convert_to_float(config.get('Settings', 'steadyEnd')) * 1e9
    steadyStart = 0.3 * 1e9
    steadyEnd = 0.8 * 1e9
    experiments = int(config.get('Settings', 'experiments'))
    # if "forward" in args.dir:
    serviceRateScales = [float(x) for x in config.get('Settings', 'serviceRateScales').split(',')]
    loads = [float(x) for x in config.get('Settings', 'load').split(',')]
    traffics = config.get('Settings', 'traffic').split(',')
    # serviceRateScales = [1.0]  
    # traffics = ["Google_SearchRPC"]
    # traffics = ["Google_AllRPC", "Fabricated_Heavy_Head", "Fabricated_Heavy_Middle", "Google_SearchRPC", "Facebook_HadoopDist_All", "FacebookKeyValue_Sampled"]
    # loads = [0.8]
    # elif "param" in args.dir:
    #     serviceRateScales = [float(x) for x in config.get('Settings', 'sampleRateScales').split(',')]
    # else:
    #     serviceRateScales = [float(x) for x in config.get('Settings', 'errorRateScale').split(',')]
    # experiments = 1
    errorRates = [float(x) for x in config.get('Settings', 'errorRate').split(',')]
    # errorRates = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    differentiationDelays = [float(x) for x in config.get('Settings', 'differentiationDelay').split(',')]
    # differentiationDelays = [0.35]
    if "forward" in args.dir:
        for traffic in traffics:
            for rate in serviceRateScales:
                for load in loads:
                    print("\nAnalyzing experiments for traffic {} rate: {} load: {}".format(traffic, rate, load))
                    analyze_all_experiments(rate, steadyStart, steadyEnd, confidenceValue, args.dir + "/" + traffic, config, experiments_end=experiments, ns3_path=__ns3_path, load=load)
                    print("Traffic {} Rate {} {} {} done".format(traffic, rate, load, experiments))
                print("Traffic {} Rate {} done".format(traffic, rate))
            print("Traffic {} done".format(traffic))
    else:
        for differentiationDelay in differentiationDelays:
            for errorRate in errorRates:
                for rate in serviceRateScales:
                    print("\nAnalyzing experiments for rate: ", rate)
                    analyze_all_experiments(rate, steadyStart, steadyEnd, confidenceValue, args.dir, config, experiments_end=experiments, ns3_path=__ns3_path, differentiationDelay=differentiationDelay, errorRate=errorRate)
                    print("Rate {} with {} and {} done".format(rate, differentiationDelay, errorRate))

__main__()