import argparse
import configparser
import os
import json as js
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import numpy as np
import itertools

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k']
def readResults(results_dir, serviceRateScales, results_dir_file, selectedVarMethods, differentiationDelay=0, errorRate=0, load=0, traffic=''):
    results = {}
    results_WOBias = {}
    results_WOBias_mixingRate_filter = {}
    results_WOBias_packetsInQueue_filter = {}
    results_WOBias_mixingRate_packetsInQueue_filter = {}
    dropRate = {}
    sampleSizes = {}
    CVS = {}
    bias = {}
    erors = {}
    e2e_samples_rtt = {}
    avgRtt = {}
    avgInterArrivals= {}
    e2e_delay = {}
    e2e_stds = {}
    stds = {}
    switch_samples_rtt = {}
    switch_delay = {}
    switch_nonMarking = {}
    queueOccupancy = {}
    PacktsInQueue = {}
    GT1PktsFrac = {}
    EmptyFrac = {}
    ks_statistic = {}
    ks_statisticMean = {}
    mixingRate = {}
    mixingRateTimeAvg = {}
    mixingSignalAvg = {}
    mixingDelayDiff = {}
    mixingRateMonly = {}
    mixingRatePoisson = {}
    mixingRateE2EPoisson = {}
    workload = {}
    totalPkts = {}
    pcktsRatio = {}
    stdsRatios = {}
    flows = ['A0D0']
    paths = ["0"]
    for rate in serviceRateScales:
        results[rate] = {}
        results_WOBias[rate] = {}
        results_WOBias_mixingRate_filter[rate] = {}
        results_WOBias_packetsInQueue_filter[rate] = {}
        results_WOBias_mixingRate_packetsInQueue_filter[rate] = {}
        dropRate[rate] = {}
        sampleSizes[rate] = {}
        workload[rate] = {}
        totalPkts[rate] = {}
        pcktsRatio[rate] = {}
        CVS[rate] = {}
        bias[rate] = {}
        erors[rate] = {}
        e2e_samples_rtt[rate] = {}
        e2e_delay[rate] = {}
        e2e_stds[rate] = {}
        switch_samples_rtt[rate] = {}
        switch_delay[rate] = {}
        switch_nonMarking[rate] = {}
        queueOccupancy[rate] = {}
        PacktsInQueue[rate] = {}
        GT1PktsFrac[rate] = {}
        EmptyFrac[rate] = {}
        ks_statistic[rate] = {}
        ks_statisticMean[rate] = {}
        mixingRate[rate] = {}
        mixingRateTimeAvg[rate] = {}
        mixingSignalAvg[rate] = {}
        mixingDelayDiff[rate] = {}
        mixingRateMonly[rate] = {}
        mixingRatePoisson[rate] = {}
        mixingRateE2EPoisson[rate] = {}
        stds[rate] = {}
        stdsRatios[rate] = {}
        avgRtt[rate] = {}
        avgInterArrivals[rate] = {}
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
                    for path in paths:
                        results[rate]['Delay'] = {}
                        results[rate]['LastDelay'] = {}
                        results[rate]['SuccessProb'] = {}
                        results[rate]['LastSuccessProb'] = {}
                        results[rate]['NonMarkingProb'] = {}
                        results[rate]['LastNonMarkingProb'] = {}
                        results_WOBias[rate]['Delay'] = {}
                        results_WOBias[rate]['LastDelay'] = {}
                        results_WOBias[rate]['SuccessProb'] = {}
                        results_WOBias[rate]['LastSuccessProb'] = {}
                        results_WOBias[rate]['NonMarkingProb'] = {}
                        results_WOBias[rate]['LastNonMarkingProb'] = {}
                        results_WOBias_mixingRate_filter[rate]['Delay'] = {}
                        results_WOBias_packetsInQueue_filter[rate]['Delay'] = {}
                        results_WOBias_mixingRate_packetsInQueue_filter[rate]['Delay'] = {}
                        workload[rate] = np.mean(temp['workLoad'][flow][path]) * 1e3
                        totalPkts[rate] = np.mean(temp['totalPckts'][flow][path])
                        pcktsRatio[rate] = np.mean([temp['EndToEndSampleSizeDelay'][flow][path][i] / temp['totalPckts'][flow][path][i] for i in range(temp['experiments'])])
                        CVS[rate]['DelayCV'] = np.mean([temp['SD0Delaystd'][i] / temp['SD0DelayMean'][i] if temp['SD0DelayMean'][i] != 0 else 0 for i in range(temp['experiments'])])
                        # CVS[rate]['LastDelayCV'] = np.mean([temp['SD0LastDelaystd'][i] / temp['SD0LastDelayMean'][i] for i in range(temp['experiments'])])
                        CVS[rate]['SuccessProbCV'] = np.mean([temp['SD0SuccessProbStd'][i] / temp['SD0SuccessProbMean'][i] for i in range(temp['experiments'])])
                        # CVS[rate]['LastSuccessProbCV'] = np.mean([temp['SD0LastSuccessProbStd'][i] / temp['SD0LastSuccessProbMean'][i] for i in range(temp['experiments'])])
                        CVS[rate]['NonMarkingProbCV'] = np.mean([temp['SD0NonMarkingProbStd'][i] / temp['SD0NonMarkingProbMean'][i] for i in range(temp['experiments'])])
                        bias[rate]['delay'] = np.mean(temp['DelayBias'][flow][path])
                        bias[rate]['success'] = np.mean(temp['SuccessProbBias'][flow][path])
                        bias[rate]['nonMarking'] = np.mean(temp['NonMarkingProbBias'][flow][path])
                        # erors[rate]['delay'] = np.mean([abs(temp['SD0DelayMean'][i] - temp['EndToEndDelayMean']['event_poisson_eventAvg'][flow][path][0][i][0]) for i in range(temp['EndToEndDelayMean']['event_poisson_eventAvg'][flow][path][1])])
                        # erors[rate]['success'] = np.mean([abs(temp['SD0SuccessProbMean'][i] - temp['EndToEndSuccessProb']['event_poisson_eventAvg'][flow][path][0][i][0]) for i in range(temp['EndToEndSuccessProb']['event_poisson_eventAvg'][flow][path][1])])
                        # erors[rate]['nonMarking'] = np.mean([abs(temp['SD0NonMarkingProbMean'][i] - temp['EndToEndNonMarkingProb']['event_poisson_eventAvg'][flow][path][0][i][0]) for i in range(temp['EndToEndNonMarkingProb']['event_poisson_eventAvg'][flow][path][1])])
                        erors[rate]['delay'] = np.mean([temp['SD0DelayMean'][i] - temp['EndToEndDelayMean']['event_poisson_eventAvg'][flow][path][0][i][0] for i in range(temp['EndToEndDelayMean']['event_poisson_eventAvg'][flow][path][1])])
                        erors[rate]['success'] = np.mean([temp['SD0SuccessProbMean'][i] - temp['EndToEndSuccessProb']['event_poisson_eventAvg'][flow][path][0][i][0] for i in range(temp['EndToEndSuccessProb']['event_poisson_eventAvg'][flow][path][1])])
                        erors[rate]['nonMarking'] = np.mean([temp['SD0NonMarkingProbMean'][i] - temp['EndToEndNonMarkingProb']['event_poisson_eventAvg'][flow][path][0][i][0] for i in range(temp['EndToEndNonMarkingProb']['event_poisson_eventAvg'][flow][path][1])])
                        # CVS[rate]['LastNonMarkingProbCV'] = np.mean([temp['SD0LastNonMarkingProbStd'][i] / temp['SD0LastNonMarkingProbMean'][i] for i in range(temp['experiments'])])
                        # CVS[rate]['SubSamplesDelayCV'] = np.mean([temp['EndToEndDelayMean']['event_poisson_eventAvg'][flow][path][i][1] / temp['EndToEndDelayMean']['event_poisson_eventAvg'][flow][path][i][0] * np.sqrt(temp['EndToEndSampleSizeDelay'][flow][path][i]) for i in range(temp['experiments'])])
                        # CVS[rate]['SubSamplesSuccessProbCV'] = np.mean([temp['EndToEndSuccessProb']['event_poisson_eventAvg'][flow][path][i][1] / temp['EndToEndSuccessProb']['event_poisson_eventAvg'][flow][path][i][0] * np.sqrt(temp['EndToEndSampleSizeSuccess'][flow][path][i]) for i in range(temp['experiments'])])
                        # CVS[rate]['SubSamplesNonMarkingProbCV'] = np.mean([temp['EndToEndNonMarkingProb']['event_poisson_eventAvg'][flow][path][i][1] / temp['EndToEndNonMarkingProb']['event_poisson_eventAvg'][flow][path][i][0] * np.sqrt(temp['EndToEndSampleSizeMarking'][flow][path][i]) for i in range(temp['experiments'])])
                        stds[rate]['delay'] = np.sqrt(sum(temp['SD0Delaystd'][i] ** 2 for i in range(len(temp['SD0Delaystd'])))) / len(temp['SD0Delaystd'])
                        stds[rate]['success'] = np.sqrt(sum(temp['SD0SuccessProbStd'][i] ** 2 for i in range(len(temp['SD0SuccessProbStd'])))) / len(temp['SD0SuccessProbStd'])
                        stds[rate]['nonMarking'] = np.sqrt(sum(temp['SD0NonMarkingProbStd'][i] ** 2 for i in range(len(temp['SD0NonMarkingProbStd'])))) / len(temp['SD0NonMarkingProbStd'])
                        sampleSizes[rate] = np.mean(temp['EndToEndSampleSizeDelay'][flow][path])
                        e2e_samples_rtt[rate] = np.mean([temp['RTT'][flow][path][i] / temp['InterArrivals'][flow][path][i] for i in range(temp['experiments']) if (str(temp['InterArrivals'][flow][path][i]) != 'nan' and temp['InterArrivals'][flow][path][i] != 0)])
                        avgRtt[rate] = np.mean(temp['RTT'][flow][path])
                        avgInterArrivals[rate] = np.nanmean(temp['InterArrivals'][flow][path])
                        e2e_delay[rate] = np.mean([temp['EndToEndDelayMean']['event_poisson_eventAvg'][flow][path][0][i][0] for i in range(temp['EndToEndDelayMean']['event_poisson_eventAvg'][flow][path][1])])
                        e2e_stds[rate]['delay'] = np.mean([temp['EndToEndDelayMean']['event_poisson_eventAvg'][flow][path][0][i][1] for i in range(temp['EndToEndDelayMean']['event_poisson_eventAvg'][flow][path][1])])
                        e2e_stds[rate]['success'] = np.mean([temp['EndToEndSuccessProb']['event_poisson_eventAvg'][flow][path][0][i][1] for i in range(temp['EndToEndSuccessProb']['event_poisson_eventAvg'][flow][path][1])])
                        e2e_stds[rate]['nonMarking'] = np.mean([temp['EndToEndNonMarkingProb']['event_poisson_eventAvg'][flow][path][0][i][1] for i in range(temp['EndToEndNonMarkingProb']['event_poisson_eventAvg'][flow][path][1])])
                        stdsRatios[rate]['delay'] = np.mean([temp['EndToEndDelayMean']['event_poisson_eventAvg'][flow][path][0][i][1] / temp['SD0Delaystd'][i] if temp['SD0Delaystd'][i] != 0 else 1 for i in range(temp['EndToEndDelayMean']['event_poisson_eventAvg'][flow][path][1])])
                        stdsRatios[rate]['success'] = np.mean([temp['EndToEndSuccessProb']['event_poisson_eventAvg'][flow][path][0][i][1] / temp['SD0SuccessProbStd'][i] if temp['SD0SuccessProbStd'][i] != 0 else 1 for i in range(temp['EndToEndSuccessProb']['event_poisson_eventAvg'][flow][path][1])])
                        stdsRatios[rate]['nonMarking'] = np.mean([temp['EndToEndNonMarkingProb']['event_poisson_eventAvg'][flow][path][0][i][1] / temp['SD0NonMarkingProbStd'][i] if temp['SD0NonMarkingProbStd'][i] != 0 else 1 for i in range(temp['EndToEndNonMarkingProb']['event_poisson_eventAvg'][flow][path][1])])
                        switch_samples_rtt[rate] = np.mean([temp['RTT'][flow][path][i] / temp['SD0InterArrivals'][i] for i in range(temp['experiments']) if temp['SD0InterArrivals'][i] != 0])
                        switch_delay[rate] = np.mean(temp['SD0DelayMean'])
                        switch_nonMarking[rate] = np.mean(temp['SD0NonMarkingProbMean'])
                        queueOccupancy[rate] = np.mean(temp['SD0Occupancy'])
                        PacktsInQueue[rate] = np.mean(temp['SD0PacktsInQueue'])
                        EmptyFrac[rate] = np.mean(temp['SD0EmptyFrac'])
                        GT1PktsFrac[rate] = np.mean(temp['SD0GT1PktsFrac'])
                        # ks_statistic[rate] = np.mean(temp['SD0ks_statistic'])
                        # ks_statisticMean[rate] = np.mean(temp['SD0ks_statisticMean'])
                        mixingRate[rate] = np.mean(temp['SD0mixingRate'])
                        # mixingRateTimeAvg[rate] = np.mean(temp['SD0mixingRateTimeAvg'])
                        mixingSignalAvg[rate] = abs(np.mean(temp['SD0mixingSignalAvg']))
                        mixingDelayDiff[rate] = abs(np.mean(temp['SD0mixingDelayDiff']))
                        mixingRateMonly[rate] = np.mean(temp['SD0mixingRateMonly'])
                        mixingRatePoisson[rate] = np.mean(temp['SD0mixingRatePoisson'])
                        # mixingRateE2EPoisson[rate] = np.mean(temp['SD0mixingRateE2EPoisson'])
                        if len(selectedVarMethods) == 0:
                            selectedVarMethods = list(temp['MaxEpsilonIneqDelay'].keys()) + list(temp['MaxEpsilonIneqSuccessProb'].keys()) + list(temp['MaxEpsilonIneqNonMarkingProb'].keys())
                        for var_method in temp['MaxEpsilonIneqDelay'].keys():
                            if var_method not in selectedVarMethods:
                                continue
                            # results[rate]['Delay'][var_method] = temp['MaxEpsilonIneqDelay'][var_method][flow][path][0] / temp['MaxEpsilonIneqDelay'][var_method][flow][path][1] * 100 if temp['MaxEpsilonIneqDelay'][var_method][flow][path][1] != 0 else None
                            # results[rate]['LastDelay'][var_method] = temp['MaxEpsilonIneqLastDelay'][var_method][flow][path][0] / temp['MaxEpsilonIneqLastDelay'][var_method][flow][path][1] * 100 if temp['MaxEpsilonIneqLastDelay'][var_method][flow][path][1] != 0 else None
                            results[rate]['Delay'][var_method] = temp['MaxEpsilonIneqDelay'][var_method][flow][path][0]['WBias'] / temp['MaxEpsilonIneqDelay'][var_method][flow][path][1] * 100 if temp['MaxEpsilonIneqDelay'][var_method][flow][path][1] != 0 else None
                            results[rate]['LastDelay'][var_method] = temp['MaxEpsilonIneqLastDelay'][var_method][flow][path][0]['WBias'] / temp['MaxEpsilonIneqLastDelay'][var_method][flow][path][1] * 100 if temp['MaxEpsilonIneqLastDelay'][var_method][flow][path][1] != 0 else None
                            results_WOBias[rate]['Delay'][var_method] = temp['MaxEpsilonIneqDelay'][var_method][flow][path][0]['WOBias'] / temp['MaxEpsilonIneqDelay'][var_method][flow][path][1] * 100 if temp['MaxEpsilonIneqDelay'][var_method][flow][path][1] != 0 else None
                            results_WOBias[rate]['LastDelay'][var_method] = temp['MaxEpsilonIneqLastDelay'][var_method][flow][path][0]['WOBias'] / temp['MaxEpsilonIneqLastDelay'][var_method][flow][path][1] * 100 if temp['MaxEpsilonIneqLastDelay'][var_method][flow][path][1] != 0 else None
                            # WOBias_mixingRate_filter = []
                            # WOBias_packetsInQueue_filter = []
                            # WOBias_mixingRate_packetsInQueue_filter = []

                            # exp = 0
                            # exp_wenoughSamples = 0
                            # mixingrate_trsh = 0.10
                            # packetsInQueue_trsh = 1.0
                            # samples_trsh = 1
                            # while (exp < temp['experiments']):
                            #     if (temp['EndToEndSampleSizeDelay'][flow][path][exp] >= 15):
                            #         if(temp['SD0mixingRate'][exp] > mixingrate_trsh and temp['SD0PacktsInQueue'][exp] > packetsInQueue_trsh):
                            #                 WOBias_mixingRate_packetsInQueue_filter.append(abs(temp['EndToEndDelayMean']['event_poisson_eventAvg'][flow][path][0][exp_wenoughSamples][0] - temp['SD0DelayMean'][exp]) < (1.96 * temp['SD0Delaystd'][exp] * np.sqrt(1 / temp['EndToEndSampleSizeDelay'][flow][path][exp] + 1 / temp['SD0SampleSize'][exp])))                                        
                            #         if(temp['SD0mixingRate'][exp] > mixingrate_trsh):
                            #             WOBias_mixingRate_filter.append(abs(temp['EndToEndDelayMean']['event_poisson_eventAvg'][flow][path][0][exp_wenoughSamples][0] - temp['SD0DelayMean'][exp]) < (1.96 * temp['SD0Delaystd'][exp] * np.sqrt(1 / temp['EndToEndSampleSizeDelay'][flow][path][exp] + 1 / temp['SD0SampleSize'][exp])))
                            #         if(temp['SD0PacktsInQueue'][exp] > packetsInQueue_trsh):
                            #             WOBias_packetsInQueue_filter.append(abs(temp['EndToEndDelayMean']['event_poisson_eventAvg'][flow][path][0][exp_wenoughSamples][0] - temp['SD0DelayMean'][exp]) < (1.96 * temp['SD0Delaystd'][exp] * np.sqrt(1 / temp['EndToEndSampleSizeDelay'][flow][path][exp] + 1 / temp['SD0SampleSize'][exp])))
                            #         exp_wenoughSamples += 1
                            #     exp += 1
                            # results_WOBias_mixingRate_filter[rate]['Delay'][var_method] = np.mean(WOBias_mixingRate_filter) * 100 if len(WOBias_mixingRate_filter) >= samples_trsh else None
                            # results_WOBias_packetsInQueue_filter[rate]['Delay'][var_method] = np.mean(WOBias_packetsInQueue_filter) * 100 if len(WOBias_packetsInQueue_filter) >= samples_trsh else None
                            # results_WOBias_mixingRate_packetsInQueue_filter[rate]['Delay'][var_method] = np.mean(WOBias_mixingRate_packetsInQueue_filter) * 100 if len(WOBias_mixingRate_packetsInQueue_filter) >= samples_trsh else None
                            # print(len(WOBias_mixingRate_occupancy_filter))
                        for var_method in temp['MaxEpsilonIneqSuccessProb'].keys():
                            if var_method not in selectedVarMethods:
                                continue
                            # results[rate]['SuccessProb'][var_method] = temp['MaxEpsilonIneqSuccessProb'][var_method][flow][path][0] /temp['MaxEpsilonIneqSuccessProb'][var_method][flow][path][1] * 100 if temp['MaxEpsilonIneqSuccessProb'][var_method][flow][path][1] != 0 else None
                            # results[rate]['LastSuccessProb'][var_method] = temp['MaxEpsilonIneqLastSuccessProb'][var_method][flow][path][0] / temp['MaxEpsilonIneqLastSuccessProb'][var_method][flow][path][1] * 100 if temp['MaxEpsilonIneqLastSuccessProb'][var_method][flow][path][1] != 0 else None
                            results[rate]['SuccessProb'][var_method] = temp['MaxEpsilonIneqSuccessProb'][var_method][flow][path][0]['WBias'] /temp['MaxEpsilonIneqSuccessProb'][var_method][flow][path][1] * 100 if temp['MaxEpsilonIneqSuccessProb'][var_method][flow][path][1] != 0 else None
                            results[rate]['LastSuccessProb'][var_method] = temp['MaxEpsilonIneqLastSuccessProb'][var_method][flow][path][0]['WBias'] / temp['MaxEpsilonIneqLastSuccessProb'][var_method][flow][path][1] * 100 if temp['MaxEpsilonIneqLastSuccessProb'][var_method][flow][path][1] != 0 else None
                            results_WOBias[rate]['SuccessProb'][var_method] = temp['MaxEpsilonIneqSuccessProb'][var_method][flow][path][0]['WOBias'] /temp['MaxEpsilonIneqSuccessProb'][var_method][flow][path][1] * 100 if temp['MaxEpsilonIneqSuccessProb'][var_method][flow][path][1] != 0 else None
                            results_WOBias[rate]['LastSuccessProb'][var_method] = temp['MaxEpsilonIneqLastSuccessProb'][var_method][flow][path][0]['WOBias'] / temp['MaxEpsilonIneqLastSuccessProb'][var_method][flow][path][1] * 100 if temp['MaxEpsilonIneqLastSuccessProb'][var_method][flow][path][1] != 0 else None
                            
                        for var_method in temp['MaxEpsilonIneqNonMarkingProb'].keys():
                            if var_method not in selectedVarMethods:
                                continue
                            # results[rate]['NonMarkingProb'][var_method] = temp['MaxEpsilonIneqNonMarkingProb'][var_method][flow][path][0] / temp['MaxEpsilonIneqNonMarkingProb'][var_method][flow][path][1] * 100 if temp['MaxEpsilonIneqNonMarkingProb'][var_method][flow][path][1] != 0 else None
                            results[rate]['NonMarkingProb'][var_method] = temp['MaxEpsilonIneqNonMarkingProb'][var_method][flow][path][0]['WBias'] / temp['MaxEpsilonIneqNonMarkingProb'][var_method][flow][path][1] * 100 if temp['MaxEpsilonIneqNonMarkingProb'][var_method][flow][path][1] != 0 else None
                            results_WOBias[rate]['NonMarkingProb'][var_method] = temp['MaxEpsilonIneqNonMarkingProb'][var_method][flow][path][0]['WOBias'] / temp['MaxEpsilonIneqNonMarkingProb'][var_method][flow][path][1] * 100 if temp['MaxEpsilonIneqNonMarkingProb'][var_method][flow][path][1] != 0 else None

                        for var_method in temp['MaxEpsilonIneqLastNonMarkingProb'].keys():
                            if var_method not in selectedVarMethods:
                                continue
                            # results[rate]['LastNonMarkingProb'][var_method] = temp['MaxEpsilonIneqLastNonMarkingProb'][var_method][flow][path][0] / temp['MaxEpsilonIneqLastNonMarkingProb'][var_method][flow][path][1] * 100 if temp['MaxEpsilonIneqLastNonMarkingProb'][var_method][flow][path][1] != 0 else None
                            results[rate]['LastNonMarkingProb'][var_method] = temp['MaxEpsilonIneqLastNonMarkingProb'][var_method][flow][path][0]['WBias'] / temp['MaxEpsilonIneqLastNonMarkingProb'][var_method][flow][path][1] * 100 if temp['MaxEpsilonIneqLastNonMarkingProb'][var_method][flow][path][1] != 0 else None
                            results_WOBias[rate]['LastNonMarkingProb'][var_method] = temp['MaxEpsilonIneqLastNonMarkingProb'][var_method][flow][path][0]['WBias'] / temp['MaxEpsilonIneqLastNonMarkingProb'][var_method][flow][path][1] * 100 if temp['MaxEpsilonIneqLastNonMarkingProb'][var_method][flow][path][1] != 0 else None
    res = {}
    res['results'] = results
    res['results_WOBias'] = results_WOBias
    res['results_WOBias_mixingRate_filter'] = results_WOBias_mixingRate_filter
    res['results_WOBias_packetsInQueue_filter'] = results_WOBias_packetsInQueue_filter
    res['results_WOBias_mixingRate_packetsInQueue_filter'] = results_WOBias_mixingRate_packetsInQueue_filter
    res['dropRate'] = dropRate
    res['CVS'] = CVS
    res['sampleSizes'] = sampleSizes
    res['workload'] = workload
    res['e2e_samples_rtt'] = e2e_samples_rtt
    res['switch_samples_rtt'] = switch_samples_rtt
    res['stds'] = stds
    res['totalPkts'] = totalPkts
    res['pcktsRatio'] = pcktsRatio
    res['errors'] = erors
    res['bias'] = bias
    res['switch_delay'] = switch_delay
    res['switch_nonMarking'] = switch_nonMarking
    res['queueOccupancy'] = queueOccupancy
    res['PacktsInQueue'] = PacktsInQueue
    res['EmptyFrac'] = EmptyFrac
    res['GT1PktsFrac'] = GT1PktsFrac
    res['ks_statistic'] = ks_statistic
    res['ks_statisticMean'] = ks_statisticMean
    res['mixingRate'] = mixingRate
    res['mixingRateTimeAvg'] = mixingRateTimeAvg
    res['mixingSignalAvg'] = mixingSignalAvg
    res['mixingDelayDiff'] = mixingDelayDiff
    res['mixingRateMonly'] = mixingRateMonly
    res['mixingRatePoisson'] = mixingRatePoisson
    res['mixingRateE2EPoisson'] = mixingRateE2EPoisson
    res['e2e_delay'] = e2e_delay
    res['e2e_stds'] = e2e_stds
    res['stdsRatios'] = stdsRatios
    res['avgRtt'] = avgRtt
    res['avgInterArrivals'] = avgInterArrivals
    return res

def plot_CV_perRate(serviceRateScales, results_dir, results_dir_file, CVS, DropRates):
    oversub_ratios = [1 / r if r != 0 else np.nan for r in serviceRateScales]
    for metric in set(k for r in CVS.values() for k in r.keys()):
        print(f"Plotting {metric}...")
        plt.figure(figsize=(20, 14))
        ax = plt.gca()
        data = [CVS[rate][metric] for rate in serviceRateScales]
        plt.scatter(oversub_ratios, data, marker='o', label=metric, color='b', linewidth=1)
        # Primary x-axis: Oversubscription ratios
        ax.set_xticks(oversub_ratios)
        ax.set_xticklabels([f"{alpha:.2f}" for alpha in oversub_ratios], rotation=45, fontsize=15)
        ax.set_xlabel("Oversubscription Ratio (α)", fontsize=20)

        # Y-axis
        plt.ylim(-0.05, max(data) * (1.05))
        ax.set_yticks(np.arange(-0.05, max(data) * (1.05), 0.05))
        ax.set_ylabel(f"{metric}", fontsize=20)

        # Secondary x-axis (top): Drop rates
        ax_top = ax.secondary_xaxis('top')
        ax_top.set_xticks(oversub_ratios)
        ax_top.set_xticklabels([f"{drop*100:.4f}%" for drop in DropRates], rotation=90, fontsize=15)
        ax_top.set_xlabel("Drop Rate", fontsize=20)

        plt.title("{} vs Rate".format(metric))
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.savefig(f"../Results/results_{results_dir}/{results_dir_file}_{metric}_vs_Rate.png")

def plot_SampleSize_perRate(serviceRateScales, results_dir, results_dir_file, sampleSizes, DropRates):
    oversub_ratios = [1 / r if r != 0 else np.nan for r in serviceRateScales]
    for metric in set(k for r in sampleSizes.values() for k in r.keys()):
        print(f"Plotting {metric}...")
        plt.figure(figsize=(20, 14))
        ax = plt.gca()
        data = [sampleSizes[rate][metric] for rate in serviceRateScales]
        plt.scatter(oversub_ratios, data, marker='o', label=metric, color='b', linewidth=1)
        # Primary x-axis: Oversubscription ratios
        ax.set_xticks(oversub_ratios)
        ax.set_xticklabels([f"{alpha:.2f}" for alpha in oversub_ratios], rotation=45, fontsize=15)
        ax.set_xlabel("Oversubscription Ratio (α)", fontsize=20)

        # Y-axis
        plt.ylim(min(data) * 0.95, max(data) * (1.05))
        ax.set_yticks(np.arange(min(data) * 0.95, max(data) * (1.05), 5))
        ax.set_ylabel(f"{metric}", fontsize=20)

        # Secondary x-axis (top): Drop rates
        ax_top = ax.secondary_xaxis('top')
        ax_top.set_xticks(oversub_ratios)
        ax_top.set_xticklabels([f"{drop*100:.4f}%" for drop in DropRates], rotation=90, fontsize=15)
        ax_top.set_xlabel("Drop Rate", fontsize=20)

        plt.title("{} vs Rate".format(metric))
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.savefig(f"../Results/results_{results_dir}/{results_dir_file}_{metric}_vs_Rate.png")

def plot_boxplot(results, serviceRateScales, results_dir, results_dir_file, metric):
    plt.figure(figsize=(8, 6))
    data = [results[rate][metric] for rate in serviceRateScales]
    plt.boxplot(data, patch_artist=True, widths=0.3)
    plt.xticks(range(1, len(serviceRateScales) + 1), serviceRateScales)
    plt.xlabel("Rate (from high to low congestion)")
    plt.ylabel("{} of Delay".format(metric))
    plt.title("{} of Delay vs Rate".format(metric))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(f"../Results/results_{results_dir}/{results_dir_file}_{metric}_vs_Rate.png")
    plt.close()

def plot_success_per_rate(results, flows, paths, rates, results_dir, results_dir_file):
    for metric in set(k for r in results.values() for k in r.keys()):
        plt.figure(figsize=(8, 6))
        
        sub_keys = set(k for r in results.values() if metric in r for k in r[metric].keys())
        sub_keys = sorted(sub_keys)
        i = 0
        for sub_key in sub_keys:
            y_values = [results[rate].get(metric, {}).get(sub_key, np.nan) for rate in rates]
            plt.plot(rates, y_values, marker='o', label=sub_key, color=colors[i], linewidth=1, markersize=4)
            i += 1
        
        plt.ylim(-5, 110)
        plt.yticks(np.arange(0, 101, 10))
        plt.xlabel("Rate (from high to low congestion)")
        plt.ylabel("Success Rate (%)")
        plt.title(f"{metric}")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, fancybox=True, shadow=True, prop={'size': 6})
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.savefig(f"../Results/results_{results_dir}/{results_dir_file}_{metric}_vs_Rate.png")
        plt.close()

def plot_success_per_dropRates(results, rates, results_dir, results_dir_file, DropRates):
    oversub_ratios = [1 / r if r != 0 else np.nan for r in rates]
    for metric in set(k for r in results.values() for k in r.keys()):
        print(f"Plotting {metric}...")
        plt.figure(figsize=(20, 14))
        ax = plt.gca()
        # Prepare and sort sub_keys
        sub_keys = set(k for r in results.values() if metric in r for k in r[metric].keys())
        sub_keys = sorted(sub_keys)
        i = 0
        for sub_key in sub_keys:
            y_values = [results[rate].get(metric, {}).get(sub_key, np.nan) for rate in rates]
            plt.plot(oversub_ratios, y_values, marker='o', label=sub_key,
                    color=colors[i], linewidth=1, markersize=4)
            i += 1

        # Primary x-axis: Oversubscription ratios
        ax.set_xticks(oversub_ratios)
        ax.set_xticklabels([f"{alpha:.2f}" for alpha in oversub_ratios], rotation=45, fontsize=15)
        ax.set_xlabel("Oversubscription Ratio (α)", fontsize=20)

        # Y-axis
        ax.set_ylim(-5, 110)
        ax.set_yticks(np.arange(0, 101, 10))
        ax.set_ylabel("Success Rate (%)", fontsize=20)

        # Secondary x-axis (top): Drop rates
        ax_top = ax.secondary_xaxis('top')
        ax_top.set_xticks(oversub_ratios)
        ax_top.set_xticklabels([f"{drop*100:.4f}%" for drop in DropRates], rotation=90, fontsize=15)
        ax_top.set_xlabel("Drop Rate", fontsize=20)

        # Plot title and legend
        plt.title(f"{metric} success rate vs Oversubscription", fontsize=20)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, fancybox=True, shadow=True, prop={'size': 10})
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.subplots_adjust(left=0.05, right=0.95)
        plt.savefig(f"../Results/results_{results_dir}/{results_dir_file}_{metric}_vs_DropRate.png")
        plt.close()

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


def plot_forward_success_per_loads_traffic(results, loads, rates, results_dir, results_dir_file, selectedVarMethod, biasTag):
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
        if len(selectedVarMethod) == 0:
            if 'success' in metric.lower():
                selectedVarMethod_ = 'probability_linearInterp_timeAvg'
            else:
                selectedVarMethod_ = 'event_linearInterp_timeAvg'
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
        plt.savefig(f"../Results/results_{results_dir}/{results_dir_file}/{biasTag}_{metric}_SuccessRate_vs_Load_Subplots.png")
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
    
def plot_metric_per_loads_traffic(traffic_list, metric, loads, rates, results_dir, results_dir_file, label):
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
    plt.savefig(f"../Results/results_{results_dir}/{results_dir_file}/{label}_vs_Load_Subplots.png")
    plt.close()



def plot_success_vs_loads(results, loads, rates, results_dir, results_dir_file, selectedVarMethod):
    for metric in set(
        k for load_dict in results.values()
          for rate_dict in load_dict.values()
          for k in rate_dict.keys()
    ):
        print(f"Plotting {metric} vs Load...")
        plt.figure(figsize=(20, 14))
        ax = plt.gca()

        # One line per rate
        for i, rate in enumerate(rates):
            y_values = [
                results[load].get(rate, {})
                              .get(metric, {})
                              .get(selectedVarMethod, np.nan)
                for load in loads
            ]
            plt.plot(loads, y_values, marker='o', label=f"α={1/rate:.2f}" if rate != 0 else "α=NaN",
                     color=colors[i % len(colors)], linewidth=1, markersize=4)

        # Primary x-axis: Load values
        ax.set_xticks(loads)
        ax.set_xticklabels([f"{l:.2f}" for l in loads], rotation=45, fontsize=15)
        ax.set_xlabel("Load", fontsize=20)

        # Y-axis: Success rate
        ax.set_ylim(-5, 110)
        ax.set_yticks(np.arange(0, 101, 10))
        ax.set_ylabel("Success Rate (%)", fontsize=20)

        # Secondary x-axis (top): Drop rates per load
        drop_rates_per_load = [
            np.mean([results[load]['DropRate'][rate] for rate in rates if rate in results[load]]) 
            if 'DropRate' in results[load] else np.nan
            for load in loads
        ]
        ax_top = ax.secondary_xaxis('top')
        ax_top.set_xticks(loads)
        ax_top.set_xticklabels([f"{drop*100:.4f}%" if not np.isnan(drop) else "NaN" for drop in drop_rates_per_load],
                               rotation=90, fontsize=15)
        ax_top.set_xlabel("Drop Rate", fontsize=20)

        # Plot title and legend
        plt.title(f"{metric} Success Rate vs Offered Load", fontsize=22)
        plt.legend(loc='lower right', ncol=4, fancybox=True, shadow=True, prop={'size': 10}, title="Oversubscription Ratio (α)")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.subplots_adjust(left=0.05, right=0.95)
        plt.savefig(f"../Results/results_{results_dir}/{results_dir_file}_{metric}_vs_Load.png")
        plt.close()


def analyse_forward_exp(results_dir, results_dir_file, rateScales, loads, selectedVarMethods, traffics):
    results = {}
    results_WOBias = {}
    results_WOBias_mixingRate_filter = {}
    results_WOBias_packetsInQueue_filter = {}
    results_WOBias_mixingRate_packetsInQueue_filter = {}
    DropRates = {}
    workload = {}
    totalPkts = {}
    e2e_samples_rtt = {}
    e2e_delay = {}
    e2e_stds = {}
    e2e_stds_all = {}
    e2e_stds_all['delay'] = {}
    e2e_stds_all['success'] = {}
    e2e_stds_all['nonMarking'] = {}
    switch_samples_rtt = {}
    switch_delay = {}
    switch_nonMarking = {}
    queueOccupancy = {}
    PacktsInQueue = {}
    EmptyFrac = {}
    GT1PktsFrac = {}
    ks_statistic = {}
    ks_statisticMean = {}
    mixingRate = {}
    mixingRateTimeAvg = {}
    mixingSignalAvg = {}
    mixingDelayDiff = {}
    mixingRateMonly = {}
    mixingRatePoisson = {}
    mixingRateE2EPoisson = {}
    sampleSizes = {}
    pcktsRatio = {}
    CVS = {}
    CVS_all = {}
    CVS_all['delay'] = {}
    CVS_all['success'] = {}
    CVS_all['nonMarking'] = {}
    bias = {}
    bias_all = {}
    bias_all['delay'] = {}
    bias_all['success'] = {}
    bias_all['nonMarking'] = {}
    stds = {}
    stds_all = {}
    stds_all['delay'] = {}
    stds_all['success'] = {}
    stds_all['nonMarking'] = {}
    errors = {}
    errors_all = {}
    errors_all['delay'] = {}
    errors_all['success'] = {}
    errors_all['nonMarking'] = {}
    stdsRatios = {}
    stdsRatios_all = {}
    stdsRatios_all['delay'] = {}
    stdsRatios_all['success'] = {}
    stdsRatios_all['nonMarking'] = {}
    avgRtt = {}
    avgInterArrivals = {}
    for traffic in traffics:
        results[traffic] = {}
        results_WOBias[traffic] = {}
        results_WOBias_mixingRate_filter[traffic] = {}
        results_WOBias_packetsInQueue_filter[traffic] = {}
        results_WOBias_mixingRate_packetsInQueue_filter[traffic] = {}
        DropRates[traffic] = {}
        workload[traffic] = {}
        totalPkts[traffic] = {}
        e2e_samples_rtt[traffic] = {}
        e2e_delay[traffic] = {}
        e2e_stds[traffic] = {}
        e2e_stds_all['delay'][traffic] = {}
        e2e_stds_all['success'][traffic] = {}
        e2e_stds_all['nonMarking'][traffic] = {}
        switch_samples_rtt[traffic] = {}
        switch_delay[traffic] = {}
        switch_nonMarking[traffic] = {}
        queueOccupancy[traffic] = {}
        PacktsInQueue[traffic] = {}
        EmptyFrac[traffic] = {}
        GT1PktsFrac[traffic] = {}
        ks_statistic[traffic] = {}
        ks_statisticMean[traffic] = {}
        sampleSizes[traffic] = {}
        mixingRate[traffic] = {}
        mixingRateTimeAvg[traffic] = {}
        mixingSignalAvg[traffic] = {}
        mixingDelayDiff[traffic] = {}
        mixingRateMonly[traffic] = {}
        mixingRatePoisson[traffic] = {}
        mixingRateE2EPoisson[traffic] = {}
        CVS[traffic] = {}
        CVS_all['delay'][traffic] = {}
        CVS_all['success'][traffic] = {}
        CVS_all['nonMarking'][traffic] = {}
        bias[traffic] = {}
        bias_all['delay'][traffic] = {}
        bias_all['success'][traffic] = {}
        bias_all['nonMarking'][traffic] = {}
        stds[traffic] = {}
        stds_all['delay'][traffic] = {}
        stds_all['success'][traffic] = {}
        stds_all['nonMarking'][traffic] = {}
        errors[traffic] = {}
        errors_all['delay'][traffic] = {}
        errors_all['success'][traffic] = {}
        errors_all['nonMarking'][traffic] = {}
        stdsRatios[traffic] = {}
        stdsRatios_all['delay'][traffic] = {}
        stdsRatios_all['success'][traffic] = {}
        stdsRatios_all['nonMarking'][traffic] = {}
        pcktsRatio[traffic] = {}
        avgRtt[traffic] = {}
        avgInterArrivals[traffic] = {}
        for load in loads:
            results[traffic][load] = {}
            results_WOBias[traffic][load] = {}
            results_WOBias_mixingRate_filter[traffic][load] = {}
            results_WOBias_packetsInQueue_filter[traffic][load] = {}
            results_WOBias_mixingRate_packetsInQueue_filter[traffic][load] = {}
            DropRates[traffic][load] = {}
            workload[traffic][load] = {}
            totalPkts[traffic][load] = {}
            e2e_samples_rtt[traffic][load] = {}
            e2e_delay[traffic][load] = {}
            e2e_stds[traffic][load] = {}
            e2e_stds_all['delay'][traffic][load] = {}
            e2e_stds_all['success'][traffic][load] = {}
            e2e_stds_all['nonMarking'][traffic][load] = {}
            switch_samples_rtt[traffic][load] = {}
            switch_delay[traffic][load] = {}
            switch_nonMarking[traffic][load] = {}
            queueOccupancy[traffic][load] = {}
            PacktsInQueue[traffic][load] = {}
            EmptyFrac[traffic][load] = {}
            GT1PktsFrac[traffic][load] = {}
            ks_statistic[traffic][load] = {}
            ks_statisticMean[traffic][load] = {}
            mixingRate[traffic][load] = {}
            mixingRateTimeAvg[traffic][load] = {}
            mixingSignalAvg[traffic][load] = {}
            mixingDelayDiff[traffic][load] = {}
            mixingRateMonly[traffic][load] = {}
            mixingRatePoisson[traffic][load] = {}
            mixingRateE2EPoisson[traffic][load] = {}
            sampleSizes[traffic][load] = {}
            CVS[traffic][load] = {}
            CVS_all['delay'][traffic][load] = {}
            CVS_all['success'][traffic][load] = {}
            CVS_all['nonMarking'][traffic][load] = {}
            bias[traffic][load] = {}
            bias_all['delay'][traffic][load] = {}
            bias_all['success'][traffic][load] = {}
            bias_all['nonMarking'][traffic][load] = {}
            stds[traffic][load] = {}
            stds_all['delay'][traffic][load] = {}
            stds_all['success'][traffic][load] = {}
            stds_all['nonMarking'][traffic][load] = {}
            errors[traffic][load] = {}
            errors_all['delay'][traffic][load] = {}
            errors_all['success'][traffic][load] = {}
            errors_all['nonMarking'][traffic][load] = {}
            stdsRatios[traffic][load] = {}
            stdsRatios_all['delay'][traffic][load] = {}
            stdsRatios_all['success'][traffic][load] = {}
            stdsRatios_all['nonMarking'][traffic][load] = {}
            pcktsRatio[traffic][load] = {}
            avgInterArrivals[traffic][load] = {}
            res = readResults(results_dir, rateScales, results_dir_file, selectedVarMethods, load=load, traffic=traffic)
            results[traffic][load] = res['results']
            results_WOBias[traffic][load] = res['results_WOBias']
            results_WOBias_mixingRate_filter[traffic][load] = res['results_WOBias_mixingRate_filter']
            results_WOBias_packetsInQueue_filter[traffic][load] = res['results_WOBias_packetsInQueue_filter']
            results_WOBias_mixingRate_packetsInQueue_filter[traffic][load] = res['results_WOBias_mixingRate_packetsInQueue_filter']
            DropRates[traffic][load] = res['dropRate']
            CVS[traffic][load] = res['CVS']
            sampleSizes[traffic][load] = res['sampleSizes']
            workload[traffic][load] = res['workload']
            e2e_samples_rtt[traffic][load] = res['e2e_samples_rtt']
            switch_samples_rtt[traffic][load] = res['switch_samples_rtt']
            stds[traffic][load] = res['stds']
            totalPkts[traffic][load] = res['totalPkts']
            errors[traffic][load] = res['errors']
            bias[traffic][load] = res['bias']
            switch_delay[traffic][load] = res['switch_delay']
            switch_nonMarking[traffic][load] = res['switch_nonMarking']
            queueOccupancy[traffic][load] = res['queueOccupancy']
            PacktsInQueue[traffic][load] = res['PacktsInQueue']
            EmptyFrac[traffic][load] = res['EmptyFrac']
            GT1PktsFrac[traffic][load] = res['GT1PktsFrac']
            ks_statistic[traffic][load] = res['ks_statistic']
            ks_statisticMean[traffic][load] = res['ks_statisticMean']
            mixingRate[traffic][load] = res['mixingRate']
            mixingRateTimeAvg[traffic][load] = res['mixingRateTimeAvg']
            mixingSignalAvg[traffic][load] = res['mixingSignalAvg']
            mixingDelayDiff[traffic][load] = res['mixingDelayDiff']
            mixingRateMonly[traffic][load] = res['mixingRateMonly']
            mixingRatePoisson[traffic][load] = res['mixingRatePoisson']
            mixingRateE2EPoisson[traffic][load] = res['mixingRateE2EPoisson']
            e2e_delay[traffic][load] = res['e2e_delay']
            pcktsRatio[traffic][load] = res['pcktsRatio']
            e2e_stds[traffic][load] = res['e2e_stds']
            stdsRatios[traffic][load] = res['stdsRatios']
            avgRtt[traffic][load] = res['avgRtt']
            avgInterArrivals[traffic][load] = res['avgInterArrivals']
            for rate in rateScales:
                CVS_all['delay'][traffic][load][rate] = CVS[traffic][load][rate]['DelayCV']
                CVS_all['success'][traffic][load][rate] = CVS[traffic][load][rate]['SuccessProbCV']
                CVS_all['nonMarking'][traffic][load][rate] = CVS[traffic][load][rate]['NonMarkingProbCV']

                stds_all['delay'][traffic][load][rate] = stds[traffic][load][rate]['delay']
                stds_all['success'][traffic][load][rate] = stds[traffic][load][rate]['success']
                stds_all['nonMarking'][traffic][load][rate] = stds[traffic][load][rate]['nonMarking']

                errors_all['delay'][traffic][load][rate] = errors[traffic][load][rate]['delay']
                errors_all['success'][traffic][load][rate] = errors[traffic][load][rate]['success']
                errors_all['nonMarking'][traffic][load][rate] = errors[traffic][load][rate]['nonMarking']

                e2e_stds_all['delay'][traffic][load][rate] = e2e_stds[traffic][load][rate]['delay']
                e2e_stds_all['success'][traffic][load][rate] = e2e_stds[traffic][load][rate]['success']
                e2e_stds_all['nonMarking'][traffic][load][rate] = e2e_stds[traffic][load][rate]['nonMarking']

                stdsRatios_all['delay'][traffic][load][rate] = stdsRatios[traffic][load][rate]['delay']
                stdsRatios_all['success'][traffic][load][rate] = stdsRatios[traffic][load][rate]['success']
                stdsRatios_all['nonMarking'][traffic][load][rate] = stdsRatios[traffic][load][rate]['nonMarking']
                bias_all['delay'][traffic][load][rate] = bias[traffic][load][rate]['delay']
                bias_all['success'][traffic][load][rate] = bias[traffic][load][rate]['success']
                bias_all['nonMarking'][traffic][load][rate] = bias[traffic][load][rate]['nonMarking']
                
            # results[traffic][load]['DropRate'] = DropRates[traffic][load]
    selectedRates = rateScales
    plot_metric_per_loads_traffic(list(results.keys()), avgInterArrivals, loads, selectedRates, results_dir, results_dir_file, 'Avg Inter Arrival Time(ns)')
    plot_metric_per_loads_traffic(list(results.keys()), e2e_samples_rtt, loads, selectedRates, results_dir, results_dir_file, '#end-to-end samples per RTT')
    plot_metric_per_loads_traffic(list(results.keys()), switch_samples_rtt, loads, selectedRates, results_dir, results_dir_file, '#switch samples per RTT')
    plot_metric_per_loads_traffic(list(results.keys()), CVS_all['delay'], loads, selectedRates, results_dir, results_dir_file, 'CV of Delay')
    plot_metric_per_loads_traffic(list(results.keys()), CVS_all['success'], loads, selectedRates, results_dir, results_dir_file, 'CV of Success Probability')
    plot_metric_per_loads_traffic(list(results.keys()), CVS_all['nonMarking'], loads, selectedRates, results_dir, results_dir_file, 'CV of Non Marking Probability')
    plot_metric_per_loads_traffic(list(results.keys()), errors_all['delay'], loads, selectedRates, results_dir, results_dir_file, 'Absolute Error of Delay(ns)')
    # plot_metric_per_loads_traffic(list(results.keys()), bias_all['delay'], loads, selectedRates, results_dir, results_dir_file, 'Bias of Delay(ns)')
    plot_metric_per_loads_traffic(list(results.keys()), e2e_stds_all['delay'], loads, selectedRates, results_dir, results_dir_file, 'End-to-end STD of Delay(ns)')
    plot_metric_per_loads_traffic(list(results.keys()), stdsRatios_all['delay'], loads, selectedRates, results_dir, results_dir_file, 'End-to-end STD of Delay(ns) over Switch STD of Delay(ns)')
    plot_metric_per_loads_traffic(list(results.keys()), switch_delay, loads, selectedRates, results_dir, results_dir_file, 'Switch Delay(ns)')
    plot_metric_per_loads_traffic(list(results.keys()), switch_nonMarking, loads, selectedRates, results_dir, results_dir_file, 'Switch Non Marking Probability')
    plot_metric_per_loads_traffic(list(results.keys()), queueOccupancy, loads, selectedRates, results_dir, results_dir_file, 'Queue Occupancy(%)')
    plot_metric_per_loads_traffic(list(results.keys()), PacktsInQueue, loads, selectedRates, results_dir, results_dir_file, '#Packets in Queue')
    plot_metric_per_loads_traffic(list(results.keys()), EmptyFrac, loads, selectedRates, results_dir, results_dir_file, 'Empty Fraction')
    plot_metric_per_loads_traffic(list(results.keys()), GT1PktsFrac, loads, selectedRates, results_dir, results_dir_file, 'Fraction of Loads with >1 Packet')
    # plot_metric_per_loads_traffic(list(results.keys()), ks_statistic, loads, selectedRates, results_dir, results_dir_file, 'KS Statistic')
    # plot_metric_per_loads_traffic(list(results.keys()), ks_statisticMean, loads, selectedRates, results_dir, results_dir_file, 'KS Statistic Mean')
    # plot_metric_per_loads_traffic(list(results.keys()), mixingRate, loads, selectedRates, results_dir, results_dir_file, 'Mixing Rate')
    # plot_metric_per_loads_traffic(list(results.keys()), mixingRateTimeAvg, loads, selectedRates, results_dir, results_dir_file, 'Mixing Rate Time Avg')
    # plot_metric_per_loads_traffic(list(results.keys()), mixingSignalAvg, loads, selectedRates, results_dir, results_dir_file, 'Mixing Signal Avg')
    # plot_metric_per_loads_traffic(list(results.keys()), mixingDelayDiff, loads, selectedRates, results_dir, results_dir_file, 'Mixing Delay Difference(ns)')
    # plot_metric_per_loads_traffic(list(results.keys()), mixingRateMonly, loads, selectedRates, results_dir, results_dir_file, 'Mixing Rate M only')
    # plot_metric_per_loads_traffic(list(results.keys()), mixingRatePoisson, loads, selectedRates, results_dir, results_dir_file, 'Mixing Rate Poisson')
    # plot_metric_per_loads_traffic(list(results.keys()), mixingRateE2EPoisson, loads, selectedRates, results_dir, results_dir_file, 'Mixing Rate E2E Poisson')
    plot_metric_per_loads_traffic(list(results.keys()), e2e_delay, loads, selectedRates, results_dir, results_dir_file, 'End-to-End Delay(ns)')
    plot_metric_per_loads_traffic(list(results.keys()), pcktsRatio, loads, selectedRates, results_dir, results_dir_file, '#end-to-end Packets Ratio')
    plot_metric_per_loads_traffic(list(results.keys()), stds_all['delay'], loads, selectedRates, results_dir, results_dir_file, 'STD of Delay(ns)')
    plot_metric_per_loads_traffic(list(results.keys()), stds_all['success'], loads, selectedRates, results_dir, results_dir_file, 'STD of Success Probability')
    plot_metric_per_loads_traffic(list(results.keys()), stds_all['nonMarking'], loads, selectedRates, results_dir, results_dir_file, 'STD of Non Marking Probability')
    plot_metric_per_loads_traffic(list(results.keys()), workload, loads, selectedRates, results_dir, results_dir_file, 'Workload(Mbps)')
    plot_metric_per_loads_traffic(list(results.keys()), totalPkts, loads, selectedRates, results_dir, results_dir_file, '#end-to-end Packets')
    plot_metric_per_loads_traffic(list(results.keys()), sampleSizes, loads, selectedRates, results_dir, results_dir_file, '#Delay Samples')
    plot_metric_per_loads_traffic(list(results.keys()), avgRtt, loads, selectedRates, results_dir, results_dir_file, 'Avg RTT(ns)')
    plot_forward_success_per_loads_traffic(results, loads, selectedRates, results_dir, results_dir_file, selectedVarMethods, 'WithBias')
    plot_forward_success_per_loads_traffic(results_WOBias, loads, selectedRates, results_dir, results_dir_file, selectedVarMethods, 'WithoutBias')
    # plot_forward_success_per_loads_traffic(results_WOBias_mixingRate_filter, loads, selectedRates, results_dir, results_dir_file, selectedVarMethods, 'WithoutBias_MixingRateFilter')
    # plot_forward_success_per_loads_traffic(results_WOBias_packetsInQueue_filter, loads, selectedRates, results_dir, results_dir_file, selectedVarMethods, 'WithoutBias_PacketsInQueueFilter')
    # plot_forward_success_per_loads_traffic(results_WOBias_mixingRate_packetsInQueue_filter, loads, selectedRates, results_dir, results_dir_file, selectedVarMethods, 'WithoutBias_MixingRate_PacketsInQueueFilter')
    plot_droprate_vs_load(list(results.keys()), loads, selectedRates, results_dir, DropRates, results_dir_file)
    plot_metric_per_loads_traffic_with_std(list(results.keys()), switch_delay, stds_all['delay'], loads, selectedRates, results_dir, results_dir_file, 'Switch Delay(ns)')
def analyse_reverse_exp(results_dir, results_dir_file, rateScales, differentiationDelays, errorRates, selectedVarMethods, type, traffics):
    results = {}
    DropRates = {}
    for differentiationDelay in differentiationDelays:
        results[differentiationDelay] = {}
        DropRates[differentiationDelay] = {}
        for errorRate in errorRates:
            results[differentiationDelay][errorRate] = {}
            DropRates[differentiationDelay][errorRate] = {}
            results[differentiationDelay][errorRate], flows, paths, DropRates[differentiationDelay][errorRate], CVS, sampleSizes, _ = readResults(results_dir, rateScales, results_dir_file, selectedVarMethods, differentiationDelay, errorRate)
    selectedRates = rateScales
    plot_success_vs_errorRate(list(results.keys()), differentiationDelays, selectedRates, results_dir, results_dir_file, selectedVarMethods[0], type)
        
def plot_success_vs_errorRate(results, differentiationDelays, rates, results_dir, results_dir_file, selectedVarMethods, type):
    for differentiationDelay in differentiationDelays:
        for metric in set(
            k for error_dict in results[differentiationDelay].values()
              for rate_dict in error_dict.values()
              for k in rate_dict.keys()
        ):
            print(f"Plotting {metric} for differentiationDelay={differentiationDelay}...")
            plt.figure(figsize=(20, 14))
            ax = plt.gca()
            ax.set_prop_cycle(color=['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d','#17becf', '#9edae5'])
            # One line per rate
            # Prepare and sort sub_keys
            for i, rate in enumerate(rates):
                error_rate_list = sorted(results[differentiationDelay].keys())
                oversub_ratio = 1 / rate if rate != 0 else np.nan
                y_values = [
                    100 - results[differentiationDelay][errorRate]
                           .get(rate, {})
                           .get(metric, {})
                           .get(selectedVarMethods, np.nan)
                    for errorRate in error_rate_list
                ]
                # print(f"Rate: {rate}, Error Rate: {error_rate_list}, Y Values: {y_values}")
                plt.plot(error_rate_list, y_values, marker='o', label=f"α={oversub_ratio:.2f}", linewidth=1, markersize=4)

            # x-axis: Error rate
            ax.set_xticks(error_rate_list)
            ax.set_xticklabels([f"{e * 100:.3f}%" for e in error_rate_list], rotation=45, fontsize=15)
            if type == 'loss':
                ax.set_xlabel("Silent Packet Drop Rate(%)", fontsize=20)
            else:
                ax.set_xlabel("Fraction of Packets with Extra Delay(%)", fontsize=20)

            # Y-axis: Success rate
            ax.set_ylim(-5, 110)
            ax.set_yticks(np.arange(0, 101, 10))
            ax.set_ylabel("Inconsistency Success Rate (%)", fontsize=20)

            # Title and legend
            plt.title(f"{metric} vs Error Rate (Differentiation Delay = {differentiationDelay})", fontsize=22)
            plt.legend(loc='lower right', ncol=4, fancybox=True, shadow=True, prop={'size': 10}, title="Oversubscription Ratio (α)")
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.subplots_adjust(left=0.05, right=0.95)
            plt.savefig(f"../Results/results_{results_dir}/{results_dir_file}_{metric}_vs_ErrorRate_Delay_{differentiationDelay}.png")
            plt.close()
         
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
    # results_dir_file = args.file
    start = 0.3 * 1e9
    end = 0.8 * 1e9
    results_dir_file = "Q_e_m_e2e_5RTT_switch_1.0_100_{}_to_{}".format(start, end)
    config = configparser.ConfigParser()
    config.read('../Results/results_{}/Parameters.config'.format(args.dir))
    rateScales = [float(x) for x in config.get('Settings', 'serviceRateScales').split(',')]
    loads = [float(x) for x in config.get('Settings', 'load').split(',')]
    traffics = config.get('Settings', 'traffic').split(',')
    # traffics = ["Facebook_HadoopDist_All"]
    # experiments = 1
    errorRates = [float(x) for x in config.get('Settings', 'errorRate').split(',')]
    # errorRates = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    differentiationDelays = [float(x) for x in config.get('Settings', 'differentiationDelay').split(',')]
    # differentiationDelays = [0.35]
    selectedVarMethods = ['event_poisson_eventAvg']
    # serviceRateScales = [0.75]
    # traffics = ["Google_AllRPC","Fabricated_Heavy_Head","Fabricated_Heavy_Middle","Google_SearchRPC", "Facebook_HadoopDist_All"]
    # loads = [0.05, 0.07, 0.1, 0.2, 0.3]
    # selectedVarMethods = []
    # print(RateScales)
    # rateScales = [0.5]
    if args.IsForward == 1:
        os.system('mkdir -p ../Results/results_' + results_dir + '/' + results_dir_file)
        analyse_forward_exp(results_dir, results_dir_file, rateScales, loads, selectedVarMethods, traffics)
        # results, flows, paths, DropRates, CVS, sampleSizes = readResults(results_dir, rateScales, results_dir_file, selectedVarMethods)
        # # plot_success_per_rate(results, flows, paths, RateScales, results_dir, results_dir_file)
        # plot_success_per_dropRates(results, rateScales, results_dir, results_dir_file, DropRates.values())
        # plot_CV_perRate(rateScales, results_dir, results_dir_file, CVS, DropRates.values())
        # plot_SampleSize_perRate(rateScales, results_dir, results_dir_file, sampleSizes, DropRates.values())
        # plot_CV_perRate(results, serviceRateScales, results_dir, results_dir_file, metric='SuccessProb')
        # plot_CV_perRate(results, serviceRateScales, results_dir, results_dir_file, metric='NonMarkingProb')
        # plot_boxplot(results, serviceRateScales, results_dir, results_dir_file, metric='SD0Delaystd')
        # plot_boxplot(results, serviceRateScales, results_dir, results_dir_file, metric='SD0DelayMean')
    else:
        analyse_reverse_exp(results_dir, results_dir_file, rateScales, differentiationDelays, errorRates, selectedVarMethods, args.type, traffics)
        

__main__()
