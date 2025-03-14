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
    def __init__(self, results_folder, rate, experiments, steadyStart, steadyEnd, output_dir):
        self.results_folder = results_folder
        self.rate = rate
        self.experiments = experiments
        self.steadyStart = steadyStart
        self.steadyEnd = steadyEnd
        self.output_dir = output_dir
        self.fractionsFactor = [round(f, 2) for f in np.arange(0.1, 1.55, steps)]
        # self.fractionsFactor = [0.01]
        self.trafficFractions = {x: [] for x in self.fractionsFactor}
        self.GTBias = {}
        self.GTByPacketsBias = {}
        self.observabilityError = {}

    def selectFlows(self, Q):
        # for each fractionFactor, select the fraction of flows that will be used
        selectedFlows = {}
        for fractionFactor in self.fractionsFactor:
            measurementFlows = Q[Q["Label"].str.contains("10.1.1.1", na=False)]['Label'].unique()
            if fractionFactor <= 1.0:
                selectedFlows[fractionFactor] = list(np.random.choice(measurementFlows, int(len(measurementFlows) * fractionFactor), replace=False))
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
    

    def calculateSingleExpBias(self, Q, Q_e, metric, selectedFlows):
        GTBias = {}
        GTByPacketsBias = {}
        observationError = {}
        GTBias[metric] = {}
        GTByPacketsBias[metric] = {}
        observationError[metric] = {}
        for fractionFactor in self.fractionsFactor:
            Q_m = Q[Q['Label'].isin(selectedFlows[fractionFactor])]
            Q_e_m = Q_e[Q_e['Label'].isin(selectedFlows[fractionFactor])]

            GTBias[metric][fractionFactor] = calculateSingleBias(Q, Q_m, metric)
            GTByPacketsBias[metric][fractionFactor] = calculateSingleBias(Q_e, Q_e_m, metric)
            observationError[metric][fractionFactor] = calculateSingleObservabilityError(Q_m, Q_e_m, metric)

        return GTBias, GTByPacketsBias, observationError
    
    def calculateBias(self, metrics):
        GTBias = {y: {x: [] for x in self.fractionsFactor} for y in metrics}
        GTByPacketsBias = {y: {x: [] for x in self.fractionsFactor} for y in metrics}
        observabilityError = {y: {x: [] for x in self.fractionsFactor} for y in metrics}
        for exp in range(self.experiments):
            Q = read_and_prune_data(self.results_folder, self.rate, exp, 'SD0_PoissonSampler_queueSize', self.steadyStart, self.steadyEnd)
            Q_e = read_and_prune_data(self.results_folder, self.rate, exp, 'SD0_PoissonSampler_queueSizeByPackets', self.steadyStart, self.steadyEnd)
            selectedFlows = self.selectFlows(Q)
            self.calculateTrafficFractions(selectedFlows, exp)
            for metric in metrics:
                _GTBias, _GTByPacketsBias, _observabilityError = self.calculateSingleExpBias(Q, Q_e, metric, selectedFlows)
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

    def plotBias(self, metric, GT):
        bias_values = []
        f_labels = []
        f_positions = []
        if GT == "Qe":
            GTBias = self.GTByPacketsBias
        else:
            GTBias = self.GTBias
        
        for i, key in enumerate(self.fractionsFactor):
            bias_values.append(GTBias[metric][key])
            f_labels.append(str(key * 100))
            f_positions.append(round(np.mean(self.trafficFractions[key]) * 100, 3))

        fig, ax1 = plt.subplots()
        box = ax1.boxplot(bias_values, positions=f_positions, patch_artist=True, showmeans=True, meanline=True, medianprops=dict(color="red", linewidth=2))
        for box_patch in box['boxes']:
            box_patch.set(facecolor='none', edgecolor='black', linewidth=2)
        ax1.set_xlabel("Measurement over Total Traffic (%)")
        ax1.set_ylabel("Bias (Bytes)")
        ax1.set_title("Bias of {} between {} and {}_m".format(metric, GT, GT))
        # show the median of boxplot

        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(f_positions)
        ax2.set_xticklabels(f_labels)
        ax2.set_xlabel("fraction of measurement traffic (%)")
        plt.savefig('../Results/results_{}/{}/{}_{}_{}_m.pdf'.format(self.output_dir, self.rate, metric, GT, GT))
        plt.clf()
            

def calculateSingleBias(Q, Q_m, metric):
    Q_m = Q_m.sort_values(by=['Time'], ignore_index=True)
    Q = Q.sort_values(by=['Time'], ignore_index=True)
    Q = Q[Q['Time'] >= Q_m.iloc[0]['Time']]
    Q = Q[Q['Time'] <= Q_m.iloc[-1]['Time']]

    Q_times = Q['Time'].values
    Q_values = Q[metric].values
    Q_rightCont_time_average = np.sum(Q_values[:-1] * np.diff(Q_times)) / (Q_times[-1] - Q_times[0])

    Q_m_times = Q_m['Time'].values
    Q_m_values = Q_m[metric].values
    Q_m_rightCont_time_average = np.sum(Q_m_values[:-1] * np.diff(Q_m_times)) / (Q_m_times[-1] - Q_m_times[0])

    return abs(Q_rightCont_time_average - Q_m_rightCont_time_average)

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




def read_and_prune_data(results_folder, rate, experiment, segment, steadyStart, steadyEnd):
    path = '{}/scratch/{}/{}/{}/{}.csv'.format(__ns3_path, results_folder, rate, experiment, segment)
    full_df = pd.read_csv(path)
    if 'SentTime' in full_df.columns:
        full_df = full_df.rename(columns={'SentTime': 'Time'})
    full_df = full_df[full_df['Time'] >= steadyStart]
    full_df = full_df[full_df['Time'] <= steadyEnd]
    return full_df


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
    steadyStart = convert_to_float(config.get('Settings', 'steadyStart')) * 1e9
    steadyEnd = convert_to_float(config.get('Settings', 'steadyEnd')) * 1e9
    experiments = int(config.get('Settings', 'experiments'))
    serviceRateScales = [float(x) for x in config.get('Settings', 'serviceRateScales').split(',')]
    results_folder = 'Results_' + args.dir
    biasCalculator = BiasCalculator(results_folder, serviceRateScales[0], experiments, steadyStart, steadyEnd, args.dir)
    biasCalculator.calculateBias(['QueueSize', 'MarkingProb'])
    # biasCalculator.scatterPLotBias('QueueSize', 'Q')
    # biasCalculator.scatterPLotBias('QueueSize', 'Qe')
    biasCalculator.scatterPLotBias('QueueSize', 'ObservabilityError')
    biasCalculator.scatterPLotBias('MarkingProb', 'ObservabilityError')
    print('QueueSize Done')

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
__main__()