from Utils import *
import configparser
import os
import json as js
import multiprocessing
import argparse

# __ns3_path = os.popen('locate "ns-3.41" | grep /ns-3.41$').read().splitlines()[0]
__ns3_path = "/media/experiments/ns-allinone-3.41/ns-3.41"

def prepare_results(queues):
    rounds_results = {}
    rounds_results['e2e_vs_sum_error_bound'] = []
    rounds_results['e2e_vs_sum_error_success_prob_bound'] = []
    rounds_results['e2e_vs_sum_error_nonmarking_prob_bound'] = []
    rounds_results['sum_poisson_samples_queue_delay_mean'] = []
    rounds_results['e2e_poisson_samples_queue_delay_mean'] = []
    rounds_results['e2e_poisson_samples_queue_delay_std'] = []
    rounds_results['sum_poisson_samples_queue_success_prob_mean'] = []
    rounds_results['sum_poisson_samples_queue_success_prob_pair_covariance'] = []
    rounds_results['sum_poisson_samples_queue_success_prob_triple_covariance'] = []
    rounds_results['e2e_poisson_samples_queue_success_prob_mean'] = []
    rounds_results['e2e_poisson_samples_queue_success_prob_std'] = []
    rounds_results['sum_poisson_samples_queue_nonmarking_prob_mean'] = []
    rounds_results['sum_poisson_samples_queue_nonmarking_prob_pair_covariance'] = []
    rounds_results['sum_poisson_samples_queue_nonmarking_prob_triple_covariance'] = []
    rounds_results['e2e_poisson_samples_queue_nonmarking_prob_mean'] = []
    rounds_results['e2e_poisson_samples_queue_nonmarking_prob_std'] = []
    rounds_results['e2e_vs_sum_consistent'] = []
    rounds_results['e2e_vs_sum_consistent_with_bias'] = []
    rounds_results['e2e_vs_sum_consistent_success_prob'] = []
    rounds_results['e2e_vs_sum_consistent_nonmarking_prob'] = []
    for queue_name in queues:
        rounds_results[queue_name+'e2e_samples_queue_delay_mean'] = []
        rounds_results[queue_name+'e2e_samples_queue_delay_std'] = []
        rounds_results[queue_name+'e2e_samples_queue_delay_count'] = []
        rounds_results[queue_name+'e2e_samples_queue_success_prob_mean'] = []
        rounds_results[queue_name+'e2e_samples_queue_success_prob_std'] = []
        rounds_results[queue_name+'e2e_samples_queue_nonmarking_prob_mean'] = []
        rounds_results[queue_name+'e2e_samples_queue_nonmarking_prob_std'] = []
        rounds_results[queue_name+'poisson_samples_queue_delay_mean'] = []
        rounds_results[queue_name+'poisson_samples_queue_delay_std'] = []
        rounds_results[queue_name+'poisson_samples_queue_success_prob_mean'] = []
        rounds_results[queue_name+'poisson_samples_queue_success_prob_std'] = []
        rounds_results[queue_name+'poisson_samples_queue_nonmarking_prob_mean'] = []
        rounds_results[queue_name+'poisson_samples_queue_nonmarking_prob_std'] = []
        rounds_results[queue_name+'poisson_samples_queue_delay_count'] = []
        rounds_results[queue_name+'poisson_prob_non_empty'] = []
        rounds_results[queue_name+'error_bound'] = []
        rounds_results[queue_name+'success_prob_error_bound'] = []
        rounds_results[queue_name+'nonmarking_prob_error_bound'] = []
        rounds_results[queue_name+'e2e_vs_poisson_consistent'] = []
        rounds_results[queue_name+'e2e_vs_poisson_consistent_with_bias'] = []
        rounds_results[queue_name+'e2e_vs_poisson_consistent_success_prob'] = []
        rounds_results[queue_name+'e2e_vs_poisson_consistent_nonmarking_prob'] = []
        rounds_results[queue_name+'split_ratio'] = []
        rounds_results[queue_name+'bias'] = []
        rounds_results[queue_name+'NPkts'] = []
        rounds_results[queue_name+'NBytes'] = []
    rounds_results['experiment'] = []
    return rounds_results
            
def analyze_single_experiment(return_dict, rate, queues_names, steadyStart, steadyEnd, rounds_results, results_folder, config, 
                              experiment=0, ns3_path=__ns3_path, differentiationDelay=None, errorRate=None, load=None, 
                              flow_names=[], queue_names=[], sampling_factor=None):
    hostToTorLinkRate = convert_to_float(config.get('Settings', 'hostToTorLinkRate')) * 1e-3
    torToAggLinkRate = convert_to_float(config.get('Settings', 'torToAggLinkRate')) * rate * 1e-3
    switchSrcREDQueueDiscMaxSize = convert_to_float(config.get('Settings', 'switchSrcREDQueueDiscMaxSize'))
    switchREDQueueDiscMaxSize = convert_to_float(config.get('DCSim', 'switchREDQueueDiscMaxSize')) * rate
    linkDelay = convert_to_float(config.get('Settings', 'hostToTorLinkDelay')) * 1e6
    passiveProbe = False if config.get('Settings', 'PassiveProbe') == "0" else True
    num_of_paths = 1 # this is the numnber of paths we want to consider for each flow, not the actual number of paths in the network
    nHosts = 24
    paths = range(num_of_paths)

    bias_res = calculate_offline_delay_bias_DC(__ns3_path, rate, experiment, results_folder, steadyStart, steadyEnd, 
                                    linkRates=[hostToTorLinkRate, torToAggLinkRate, torToAggLinkRate, hostToTorLinkRate], 
                                    linkDelays=[linkDelay, linkDelay, linkDelay, linkDelay], 
                                    swtichDstREDQueueDiscMaxSize=[switchSrcREDQueueDiscMaxSize, switchREDQueueDiscMaxSize, switchREDQueueDiscMaxSize, switchSrcREDQueueDiscMaxSize], 
                                    tsh=0.15, differentiationDelay=differentiationDelay, errorRate=errorRate, load=load, 
                                    queue_names=queue_names, flow_names=flow_names, e2e_intervals=10000, sampling_factor=sampling_factor)
    bias_res['experiment'] = experiment
    return_dict[experiment] = bias_res

def merge_results(return_dict, merged_results, queues):
    for exp in sorted(return_dict.keys()):
        for queue_name in queues:
            merged_results[queue_name+'e2e_samples_queue_delay_mean'].append(return_dict[exp][queue_name+'e2e_samples_queue_delay_mean'])
            merged_results[queue_name+'e2e_samples_queue_delay_std'].append(return_dict[exp][queue_name+'e2e_samples_queue_delay_std'])
            merged_results[queue_name+'e2e_samples_queue_delay_count'].append(return_dict[exp][queue_name+'e2e_samples_queue_delay_count'])
            merged_results[queue_name+'e2e_samples_queue_success_prob_mean'].append(return_dict[exp][queue_name+'e2e_samples_queue_success_prob_mean'])
            merged_results[queue_name+'e2e_samples_queue_success_prob_std'].append(return_dict[exp][queue_name+'e2e_samples_queue_success_prob_std'])
            merged_results[queue_name+'e2e_samples_queue_nonmarking_prob_mean'].append(return_dict[exp][queue_name+'e2e_samples_queue_nonmarking_prob_mean'])
            merged_results[queue_name+'e2e_samples_queue_nonmarking_prob_std'].append(return_dict[exp][queue_name+'e2e_samples_queue_nonmarking_prob_std'])
            merged_results[queue_name+'poisson_samples_queue_delay_mean'].append(return_dict[exp][queue_name+'poisson_samples_queue_delay_mean'])
            merged_results[queue_name+'poisson_samples_queue_delay_std'].append(return_dict[exp][queue_name+'poisson_samples_queue_delay_std'])
            merged_results[queue_name+'poisson_samples_queue_delay_count'].append(return_dict[exp][queue_name+'poisson_samples_queue_delay_count'])
            merged_results[queue_name+'poisson_samples_queue_success_prob_mean'].append(return_dict[exp][queue_name+'poisson_samples_queue_success_prob_mean'])
            merged_results[queue_name+'poisson_samples_queue_success_prob_std'].append(return_dict[exp][queue_name+'poisson_samples_queue_success_prob_std'])
            merged_results[queue_name+'poisson_samples_queue_nonmarking_prob_mean'].append(return_dict[exp][queue_name+'poisson_samples_queue_nonmarking_prob_mean'])
            merged_results[queue_name+'poisson_samples_queue_nonmarking_prob_std'].append(return_dict[exp][queue_name+'poisson_samples_queue_nonmarking_prob_std'])
            merged_results[queue_name+'poisson_prob_non_empty'].append(return_dict[exp][queue_name+'poisson_prob_non_empty'])
            merged_results[queue_name+'error_bound'].append(return_dict[exp][queue_name+'error_bound'])
            merged_results[queue_name+'success_prob_error_bound'].append(return_dict[exp][queue_name+'success_prob_error_bound'])
            merged_results[queue_name+'nonmarking_prob_error_bound'].append(return_dict[exp][queue_name+'nonmarking_prob_error_bound'])
            merged_results[queue_name+'e2e_vs_poisson_consistent'].append(return_dict[exp][queue_name+'e2e_vs_poisson_consistent'])
            merged_results[queue_name+'e2e_vs_poisson_consistent_success_prob'].append(return_dict[exp][queue_name+'e2e_vs_poisson_consistent_success_prob'])
            merged_results[queue_name+'e2e_vs_poisson_consistent_nonmarking_prob'].append(return_dict[exp][queue_name+'e2e_vs_poisson_consistent_nonmarking_prob'])
            merged_results[queue_name+'e2e_vs_poisson_consistent_with_bias'].append(return_dict[exp][queue_name+'e2e_vs_poisson_consistent_with_bias'])
            merged_results[queue_name+'split_ratio'].append(return_dict[exp][queue_name+'split_ratio'])
            merged_results[queue_name+'bias'].append(return_dict[exp][queue_name+'bias'])
            merged_results[queue_name+'NPkts'].append(return_dict[exp][queue_name+'NPkts'])
            merged_results[queue_name+'NBytes'].append(return_dict[exp][queue_name+'NBytes'])
        
        merged_results['e2e_vs_sum_error_bound'].append(return_dict[exp]['e2e_vs_sum_error_bound'])
        merged_results['e2e_vs_sum_error_success_prob_bound'].append(return_dict[exp]['e2e_vs_sum_error_success_prob_bound'])
        merged_results['e2e_vs_sum_error_nonmarking_prob_bound'].append(return_dict[ exp]['e2e_vs_sum_error_nonmarking_prob_bound'])
        merged_results['sum_poisson_samples_queue_delay_mean'].append(return_dict[exp]['sum_poisson_samples_queue_delay_mean'])
        merged_results['sum_poisson_samples_queue_success_prob_mean'].append(return_dict[exp]['sum_poisson_samples_queue_success_prob_mean'])
        merged_results['sum_poisson_samples_queue_success_prob_pair_covariance'].append(return_dict[exp]['sum_poisson_samples_queue_success_prob_pair_covariance'])
        merged_results['sum_poisson_samples_queue_success_prob_triple_covariance'].append(return_dict[exp]['sum_poisson_samples_queue_success_prob_triple_covariance'])
        merged_results['sum_poisson_samples_queue_nonmarking_prob_mean'].append(return_dict[exp]['sum_poisson_samples_queue_nonmarking_prob_mean'])
        merged_results['sum_poisson_samples_queue_nonmarking_prob_pair_covariance'].append(return_dict[exp]['sum_poisson_samples_queue_nonmarking_prob_pair_covariance'])
        merged_results['sum_poisson_samples_queue_nonmarking_prob_triple_covariance'].append(return_dict[exp]['sum_poisson_samples_queue_nonmarking_prob_triple_covariance'])
        merged_results['e2e_poisson_samples_queue_delay_mean'].append(return_dict[exp]['e2e_poisson_samples_queue_delay_mean'])
        merged_results['e2e_poisson_samples_queue_success_prob_mean'].append(return_dict[exp]['e2e_poisson_samples_queue_success_prob_mean'])
        merged_results['e2e_poisson_samples_queue_nonmarking_prob_mean'].append(return_dict[exp]['e2e_poisson_samples_queue_nonmarking_prob_mean'])
        merged_results['e2e_poisson_samples_queue_delay_std'].append(return_dict[exp]['e2e_poisson_samples_queue_delay_std'])
        merged_results['e2e_poisson_samples_queue_success_prob_std'].append(return_dict[exp]['e2e_poisson_samples_queue_success_prob_std'])
        merged_results['e2e_poisson_samples_queue_nonmarking_prob_std'].append(return_dict[exp]['e2e_poisson_samples_queue_nonmarking_prob_std'])
        merged_results['e2e_vs_sum_consistent'].append(return_dict[exp]['e2e_vs_sum_consistent'])
        merged_results['e2e_vs_sum_consistent_success_prob'].append(return_dict[exp]['e2e_vs_sum_consistent_success_prob'])
        merged_results['e2e_vs_sum_consistent_nonmarking_prob'].append(return_dict[exp]['e2e_vs_sum_consistent_nonmarking_prob'])
        merged_results['e2e_vs_sum_consistent_with_bias'].append(return_dict[exp]['e2e_vs_sum_consistent_with_bias'])
        merged_results['experiment'].append(return_dict[exp]['experiment'])

def analyze_all_experiments(rate, steadyStart, steadyEnd, dir, config, experiments_end=3, ns3_path=__ns3_path, load=None, differentiationDelay=None, errorRate=None):
    # if ("delay" in dir) and ("reverse" in dir):
    #     # remove reverse from dir
    #     results_folder = 'Results_' + dir.replace("reverse", "forward").replace("delay_", "")
    # else:
    results_folder = 'Results_' + dir
    flows_name = ['R0H0R2H3']
    queues_names = ["T0A0", "A0T2", "T2H3"]
    flows_name.sort()
    queues_names.sort()
    # for sampling_factor in [0.8, 0.4, 0.2, 0.1, 0.05, 0.025, 0.01]:
    for sampling_factor in [None]:
        rounds_results = prepare_results(queues_names)
        merged_results = prepare_results(queues_names)
        batch_size = 30
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
                ths.append(multiprocessing.Process(target=analyze_single_experiment, args=(return_dict, rate, queues_names, steadyStart, steadyEnd, rounds_results, results_folder, config, experiment, ns3_path, differentiationDelay, errorRate, load, flows_name, queues_names, sampling_factor)))
            
            for th in ths:
                th.start()
            for th in ths:
                th.join()
            merge_results(return_dict, merged_results, queues_names)
            print("{} joind".format(i))
        if errorRate is not None:
            os.system('mkdir -p ../Results/results_{}/{}/{}/D_{}/f_{}/'.format(dir, rate, load, differentiationDelay, errorRate))
            with open('../Results/results_{}/{}/{}/D_{}/f_{}/delay_minimum_bias_e2e_vs_switch_poisson.0_{}_{}_to_{}.json'.format(dir, rate, load, differentiationDelay, errorRate, sampling_factor, experiments_end, steadyStart, steadyEnd), 'w') as f:
                js.dump(merged_results, f, indent=4)
        else:
            with open('../Results/results_{}/{}/{}/delay_minimum_bias_e2e_vs_switch_poisson.0_{}_{}_to_{}.json'.format(dir, rate, load, sampling_factor, experiments_end, steadyStart, steadyEnd), 'w') as f:
                js.dump(merged_results, f, indent=4)

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
    steadyStart = convert_to_float(config.get('Settings', 'steadyStart')) * 1e9
    # steadyStart = 0.08 * 1e9
    steadyEnd = convert_to_float(config.get('Settings', 'steadyEnd')) * 1e9
    # steadyEnd = 0.015 * 1e9
    experiments = int(config.get('Settings', 'experiments'))
    # experiments = 1
    serviceRateScales = [float(x) for x in config.get('Settings', 'serviceRateScales').split(',')]
    # serviceRateScales = [0.5]
    loads = [float(x) for x in config.get('Settings', 'load').split(',')]
    loads = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.95]
    # loads = [0.7]
    traffics = config.get('Settings', 'traffic').split(',')
    traffics = ["Google_AllRPC","Fabricated_Heavy_Head","Fabricated_Heavy_Middle","Google_SearchRPC", "Facebook_HadoopDist_All"]
    # traffics = ["Google_AllRPC"]
    errorRates = [float(x) for x in config.get('Settings', 'errorRate').split(',')]
    # errorRates = [0.1, 0.3, 0.5, 0.7, 0.9]
    # errorRates = [0.1]
    differentiationDelays = [float(x) for x in config.get('Settings', 'differentiationDelay').split(',')]
    # differentiationDelays = [5.0]
    # devide steady period into smaller parts
    numOfSteadyParts = 1
    for start in range(int(steadyStart), int(steadyEnd), int((steadyEnd - steadyStart) / numOfSteadyParts)):
        print("Steady period: {} to {}".format(start, start + int((steadyEnd - steadyStart) / numOfSteadyParts)))
        if "forward" in args.dir:
            for traffic in traffics:
                for rate in serviceRateScales:
                    for load in loads:
                        print("\nAnalyzing experiments for traffic {} rate: {} load: {}".format(traffic, rate, load))
                        analyze_all_experiments(rate, start, start + int((steadyEnd - steadyStart) / numOfSteadyParts), args.dir + "/" + traffic, config, experiments_end=experiments, ns3_path=__ns3_path, load=load)
                        print("Traffic {} Rate {} {} {} done".format(traffic, rate, load, experiments))
                    print("Traffic {} Rate {} done".format(traffic, rate))
                print("Traffic {} done".format(traffic))
        else:
            for traffic in traffics:
                for rate in serviceRateScales:
                    for load in loads:
                        for differentiationDelay in differentiationDelays:
                            for errorRate in errorRates:
                                print("\nAnalyzing experiments for rate: ", rate, " load: ", load, " differentiationDelay: ", differentiationDelay, " errorRate: ", errorRate)
                                os.system('mkdir -p ../Results/results_{}/{}/{}/{}/D_{}/f_{}/'.format(args.dir, traffic, rate, load, differentiationDelay, errorRate))
                                analyze_all_experiments(rate, start, start + int((steadyEnd - steadyStart) / numOfSteadyParts), args.dir + "/" + traffic, config, experiments_end=experiments, ns3_path=__ns3_path, differentiationDelay=differentiationDelay, errorRate=errorRate, load=load)
                                print("Rate {} load {} with {} and {} done".format(rate, load, differentiationDelay, errorRate))
                        print("Traffic {} Rate {} load {} done".format(traffic, rate, load))
                    print("Rate {} done".format(rate))
                print("Traffic {} done".format(traffic))

__main__()