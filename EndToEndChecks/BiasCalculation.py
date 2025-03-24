from Utils import *
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


__ns3_path = "/media/experiments/ns-allinone-3.41/ns-3.41"
steps = 0.10

class BiasCalculator:
    def __init__(self, results_folder, rate, experiments, steadyStart, steadyEnd, output_dir, linkRate):
        self.results_folder = results_folder
        self.rate = rate
        self.experiments = experiments
        self.steadyStart = steadyStart
        self.steadyEnd = steadyEnd
        self.output_dir = output_dir
        # self.fractionsFactor = [round(f, 2) for f in np.arange(0.1, 1.55, steps)]
        self.fractionsFactor = [1.0]
        self.trafficFractions = {x: [] for x in self.fractionsFactor}
        self.GTBias = {}
        self.GTByPacketsBias = {}
        self.observabilityError = {}
        self.linkRate = linkRate

    def selectFlows(self, Q):
        # for each fractionFactor, select the fraction of flows that will be used
        selectedFlows = {}
        for fractionFactor in self.fractionsFactor:
            measurementFlows = Q[Q["Label"].str.contains("10.1.1.1", na=False)]['Label'].unique()
            if fractionFactor < 1.0:
                selectedFlows[fractionFactor] = list(np.random.choice(measurementFlows, int(len(measurementFlows) * fractionFactor), replace=False))
            elif fractionFactor == 1.0:
                selectedFlows[fractionFactor] = list(measurementFlows)
            else:
                ctFlows = Q[Q["Label"].str.contains("10.2.1.1", na=False)]['Label'].unique()
                f = fractionFactor - 1
                selectedFlows[fractionFactor] = list(measurementFlows) + list(np.random.choice(ctFlows, int(len(ctFlows) * f), replace=False))
        return selectedFlows

    def calculateTrafficFractions(self, selectedFlows, exp):
        for fractionFactor in self.fractionsFactor:
            self.trafficFractions[fractionFactor].append(self.calculateTrafficFraction(selectedFlows[fractionFactor], exp))

    def calculateTrafficFraction(self, selectedFlows, exp):
        allPackets = read_and_prune_data(self.results_folder, self.rate, exp, 'S0_Switch', self.steadyStart, self.steadyEnd)
        allPackets["SourceCombo"] = allPackets["SourceIp"] + ":" + allPackets["SourcePort"].astype(str)
        selectedFromAllPackets = allPackets[allPackets["SourceCombo"].isin(selectedFlows)].sort_values(by=['ReceiveTime'], ignore_index=True)
        allPackets = allPackets[allPackets['ReceiveTime'] >= selectedFromAllPackets.iloc[0]['ReceiveTime']]
        allPackets = allPackets[allPackets['ReceiveTime'] <= selectedFromAllPackets.iloc[-1]['ReceiveTime']]
        return (np.sum(selectedFromAllPackets['PayloadSize']) / np.sum(allPackets['PayloadSize']))
    

    def calculateSingleExpBias(self, Q, Q_e, metric, selectedFlows, poisson):
        GTBias = {}
        GTByPacketsBias = {}
        observationError = {}
        GTBias[metric] = {}
        GTByPacketsBias[metric] = {}
        observationError[metric] = {}
        for fractionFactor in self.fractionsFactor:
            if fractionFactor == 1.0:
                Q_m = Q[Q['Label'].str.contains('10.1.1.1', na=False)]
                # Q_e_m = Q_e[Q_e['Label'].str.contains('10.1.1.1', na=False)]
            else:
                Q_m = Q[Q['Label'].isin(selectedFlows[fractionFactor])]
                # Q_e_m = Q_e[Q_e['Label'].isin(selectedFlows[fractionFactor])]
            GTBias[metric][fractionFactor] = calculateSingleBias(Q, Q_m, metric, poisson, self.linkRate)
            # GTByPacketsBias[metric][fractionFactor] = calculateSingleBias(Q_e, Q_e_m, metric, poisson, self.linkRate)
            GTByPacketsBias[metric][fractionFactor] = None
            # observationError[metric][fractionFactor] = calculateSingleObservabilityError(Q_m, Q_e_m, metric)
            observationError[metric][fractionFactor] = None

        return GTBias, GTByPacketsBias, observationError
    
    def calculateBias(self, metrics):
        GTBias = {y: {x: [] for x in self.fractionsFactor} for y in metrics}
        GTByPacketsBias = {y: {x: [] for x in self.fractionsFactor} for y in metrics}
        observabilityError = {y: {x: [] for x in self.fractionsFactor} for y in metrics}
        for exp in self.experiments:
            Q = read_and_prune_data(self.results_folder, self.rate, exp, 'SD0_PoissonSampler_queueSize', self.steadyStart, self.steadyEnd)
            Q_e = read_and_prune_data(self.results_folder, self.rate, exp, 'SD0_PoissonSampler_queueSizeByPackets', self.steadyStart, self.steadyEnd)
            poisson = read_and_prune_data(self.results_folder, self.rate, exp, 'SD0_PoissonSampler_events', self.steadyStart, self.steadyEnd)
            selectedFlows = self.selectFlows(Q)
            self.calculateTrafficFractions(selectedFlows, exp)
            for metric in metrics:
                _GTBias, _GTByPacketsBias, _observabilityError = self.calculateSingleExpBias(Q, Q_e, metric, selectedFlows, poisson)
                for fractionFactor in self.fractionsFactor:
                    GTBias[metric][fractionFactor].append(_GTBias[metric][fractionFactor])
                    GTByPacketsBias[metric][fractionFactor].append(_GTByPacketsBias[metric][fractionFactor])
                    observabilityError[metric][fractionFactor].append(_observabilityError[metric][fractionFactor])

        self.GTBias = GTBias
        self.GTByPacketsBias = GTByPacketsBias
        self.observabilityError = observabilityError

    def scatterPLotBias(self, metric, GT):
        traffic_values = []
        bias_values = []
        if GT == "Qe":
            GTBias = self.GTByPacketsBias
        elif GT == "Q":
            GTBias = self.GTBias
        else:
            GTBias = self.observabilityError
        
        for i, key in enumerate(self.fractionsFactor):
            traffic_values.extend(self.trafficFractions[key])
            bias_values.extend(GTBias[metric][key])
        
        fig, ax = plt.subplots()
        ax.scatter(traffic_values, bias_values)
        ax.set_xlabel("Traffic Fraction")
        
        if "Q" in GT:
            ax.set_title("Bias of {} between {} and {}_m".format(metric, GT, GT))
            if "Queue" in metric: 
                ax.set_ylabel("Bias (Bytes)")
            else:
                ax.set_ylabel("Bias")
        else:
            ax.set_title("Observability Error of {} (Q_m Q_e_m)".format(metric))
            if "Queue" in metric: 
                ax.set_ylabel("Observability Error (Bytes)")
            else:
                ax.set_ylabel("Observability Error")
        
        plt.savefig('../Results/results_{}/{}/scatter_{}_{}_{}_m.pdf'.format(self.output_dir, self.rate, metric, GT, GT))
        plt.clf()
            
# def calculateSingleBias_totalSize(Q, Q_m, metric, linkRate):
#     Q_m = Q_m.sort_values(by=['Time', 'TotalQueueSize'], ascending=[True, True]).reset_index(drop=True)
#     Q = Q.sort_values(by=['Time', 'TotalQueueSize'], ascending=[True, True]).reset_index(drop=True)
#     Q = Q[Q['Time'] >= Q_m.iloc[0]['Time']]
#     Q = Q[Q['Time'] <= Q_m.iloc[-1]['Time']]
#     Q_m = manipulate_for_delay_Q_m(Q_m, linkRate)
#     Q = manipulate_for_delay_Q(Q, linkRate)

def calculateSingleBias(Q, Q_m, metric, poisson, linkRate):
    Q_m = Q_m.sort_values(by=['Time'], ignore_index=True)
    Q = Q.sort_values(by=['Time'], ignore_index=True)
    Q = Q[Q['Time'] >= Q_m.iloc[0]['Time']]
    Q = Q[Q['Time'] <= Q_m.iloc[-1]['Time']]
    poisson = poisson.sort_values(by=['Time'], ignore_index=True)
    poisson = poisson[poisson['Time'] >= Q_m.iloc[0]['Time']]
    poisson = poisson[poisson['Time'] <= Q_m.iloc[-1]['Time']]

    Q_times = Q['Time'].values
    Q_values = Q[metric].values
    if metric != 'QueuingDelay':
        Q_values = 1 - Q_values
    poisson_var = poisson[metric].values.std() / np.sqrt(len(poisson))
    Q_rightCont_time_average = np.sum(Q_values[:-1] * np.diff(Q_times)) / (Q_times[-1] - Q_times[0])
    Q_leftCont_time_average = np.sum(Q_values[1:] * np.diff(Q_times)) / (Q_times[-1] - Q_times[0])
    if metric == 'QueuingDelay':
        _, Q_linearInterp_time_average = manipulate_for_delay_Q(Q, linkRate)

    Q_m_times = Q_m['Time'].values
    Q_m_values = Q_m[metric].values
    if metric != 'QueuingDelay':
        Q_m_values = 1 - Q_m_values
    Q_m_rightCont_time_average = np.sum(Q_m_values[:-1] * np.diff(Q_m_times)) / (Q_m_times[-1] - Q_m_times[0])
    Q_m_leftCont_time_average = np.sum(Q_m_values[1:] * np.diff(Q_m_times)) / (Q_m_times[-1] - Q_m_times[0])
    if metric == 'QueuingDelay':
        _, Q_m_linearInterp_time_average = manipulate_for_delay_Q_m(Q_m, linkRate)

    if metric == 'QueuingDelay':
        diff = abs(Q_linearInterp_time_average - Q_m_linearInterp_time_average)
    else:
        diff = min(abs(Q_rightCont_time_average - Q_m_rightCont_time_average), abs(Q_leftCont_time_average - Q_m_leftCont_time_average))
    # if poisson_var == 0:
    #     return 0
    # else:
    #     return diff / poisson_var
    return diff

def calculateSingleObservabilityError(Q, Q_m, metric):
    Q_m = Q_m.sort_values(by=['Time'], ignore_index=True)
    Q = Q.sort_values(by=['Time'], ignore_index=True)

    Q_times = Q['Time'].values
    Q_values = Q[metric].values
    Q_rightCont_time_average = np.sum(Q_values[:-1] * np.diff(Q_times)) / (Q_times[-1] - Q_times[0])

    Q_m_times = Q_m['Time'].values
    Q_m_values = Q_m[metric].values
    Q_m_rightCont_time_average = np.sum(Q_m_values[:-1] * np.diff(Q_m_times)) / (Q_m_times[-1] - Q_m_times[0])

    return abs(Q_rightCont_time_average - Q_m_rightCont_time_average)


def plotBias(metric, GTBiases, fractionFactor, serviceRateScales, output_dir):
    bias_values = []
    f_labels = []
    f_positions = []
    y_max = 0
    for rate in serviceRateScales:
        bias_values.append(GTBiases[rate][metric][fractionFactor])
        y_max = max(y_max, max(GTBiases[rate][metric][fractionFactor]))
        f_labels.append(rate)
        f_positions.append(rate * 100)

    fig, ax1 = plt.subplots()
    box = ax1.boxplot(bias_values, positions=f_positions, patch_artist=True, showmeans=True, meanline=True, medianprops=dict(color="red", linewidth=2))
    for box_patch in box['boxes']:
        # increase the gap between the boxes
        box_patch.set(facecolor='none', edgecolor='black', linewidth=2)
    ax1.set_xticks(f_positions)
    ax1.set_xticklabels(f_labels)
    ax1.set_xlabel("Rate (from high to low congestion)")
    ax1.set_ylabel("Bias / samples STD")
    ax1.set_title("Bias samples Variance of {} between Q and Q_m".format(metric))
    plt.ylim(0, 1.05 * y_max)
    plt.yticks(np.arange(0, 1.05 * y_max, 0.1 * y_max))
    plt.savefig('../Results/results_{}/{}_Bias_Q_Q_m.pdf'.format(output_dir, metric))
    plt.clf()

def read_and_prune_data(results_folder, rate, experiment, segment, steadyStart, steadyEnd):
    path = '{}/scratch/{}/{}/{}/{}.csv'.format(__ns3_path, results_folder, rate, experiment, segment)
    full_df = pd.read_csv(path)
    if 'SentTime' in full_df.columns:
        full_df = full_df.rename(columns={'SentTime': 'Time'})
    full_df = full_df[full_df['Time'] >= steadyStart]
    full_df = full_df[full_df['Time'] <= steadyEnd]
    return full_df


# def __main__():
#     parser=argparse.ArgumentParser()
#     parser.add_argument("--dir",
#                     required=True,
#                     dest="dir",
#                     help="The directory of the results",
#                     default="")
    
#     args = parser.parse_args()
#     config = configparser.ConfigParser()
#     config.read('../Results/results_{}/Parameters.config'.format(args.dir))
#     steadyStart = convert_to_float(config.get('Settings', 'steadyStart')) * 1e9
#     steadyEnd = convert_to_float(config.get('Settings', 'steadyEnd')) * 1e9
#     experiments = int(config.get('Settings', 'experiments'))
#     serviceRateScales = [float(x) for x in config.get('Settings', 'serviceRateScales').split(',')]
#     # serviceRateScales = [0.79, 0.81]
#     results_folder = 'Results_' + args.dir
#     biases = {}
#     for rate in serviceRateScales:
#         linkRate = convert_to_float(config.get('SingleQueue', 'bottleneckLinkRate')) * rate * 1e-3
#         biasCalculator = BiasCalculator(results_folder, rate, experiments, steadyStart, steadyEnd, args.dir, linkRate)
#         biasCalculator.calculateBias(['QueueSize', 'MarkingProb', 'DropProb', 'QueuingDelay'])
#         biases[rate] = biasCalculator.GTBias
#         print('Rate {} Done with Link Rate: {}'.format(rate, linkRate))
#     plotBias('QueueSize', biases, 1.0, serviceRateScales, args.dir)
#     plotBias('MarkingProb', biases, 1.0, serviceRateScales, args.dir)
#     plotBias('DropProb', biases, 1.0, serviceRateScales, args.dir)
#     plotBias('QueuingDelay', biases, 1.0, serviceRateScales, args.dir)
    # biasCalculator = BiasCalculator(results_folder, serviceRateScales[0], experiments, steadyStart, steadyEnd, args.dir)
    # biasCalculator.scatterPLotBias('QueueSize', 'Q')
    # biasCalculator.scatterPLotBias('MarkingProb', 'Q')
    # biasCalculator.scatterPLotBias('DropProb', 'Q')
    # biasCalculator.scatterPLotBias('TotalQueueSize', 'Q')
    # biasCalculator.scatterPLotBias('QueueSize', 'Qe')
    # biasCalculator.scatterPLotBias('QueueSize', 'ObservabilityError')
    # biasCalculator.scatterPLotBias('MarkingProb', 'ObservabilityError')
    # print('Q Done')

    # biasCalculator.calculateBias('TotalQueueSize')
    # biasCalculator.scatterPLotBias('TotalQueueSize', 'Q')
    # biasCalculator.scatterPLotBias('TotalQueueSize', 'Qe')
    # print('TotalQueueSize Done')

    # biasCalculator.calculateBias('MarkingProb')
    # biasCalculator.scatterPLotBias('MarkingProb', 'Q')
    # biasCalculator.scatterPLotBias('MarkingProb', 'Qe')
    # print('MarkingProb Done')

    # biasCalculator.calculateBias(['DropProb'])
    # biasCalculator.scatterPLotBias('DropProb', 'Q')
    # biasCalculator.scatterPLotBias('DropProb', 'Qe')
    # print('DropProb Done')
# __main__()