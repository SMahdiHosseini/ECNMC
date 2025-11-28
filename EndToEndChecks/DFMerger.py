import argparse
import configparser
from Utils import *
import os
import multiprocessing

__ns3_path = "/media/experiments/ns-allinone-3.41/ns-3.41"

def merge_single_experiment(return_dict, rate, results_folder, config, experiment=0, ns3_path=__ns3_path, differentiationDelay=None, errorRate=None, load=None):
    segments = ["PoissonSampler_queueSize", "PoissonSampler_events", "EndToEnd_packets", "SwitchMonitor"]
    for segment in segments:
        file_paths = glob.glob('{}/scratch/{}/{}/{}/{}/*_{}.csv'.format(__ns3_path, results_folder, rate, load, experiment, segment))
        # There are a lot mutiple file with the format of namex_time_PoissonSampler_queueSize.csv with different namex and time, I want to merge all the files with the same namex into one file
        # and if there is already a file with the same namex, but with different time, we have already merged them before, so skip them
        file_dict = {}
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            name_parts = file_name.split('_')
            namex = '_'.join(name_parts[:-3])  # Get the part before the last three parts
            if namex == '':
                continue
            if namex not in file_dict:
                file_dict[namex] = []
            file_dict[namex].append(file_path)

        for namex, paths in file_dict.items():
            output_file = '{}/scratch/{}/{}/{}/{}/{}_{}.csv'.format(__ns3_path, results_folder, rate, load, experiment, namex, segment)
            with open(output_file, 'w') as outfile:
                for i, path in enumerate(paths):
                    with open(path, 'r') as infile:
                        if i != 0:
                            next(infile)  # Skip header for all but the first file
                        for line in infile:
                            outfile.write(line)
                    # delete the individual file after merging
                    os.remove(path)


def merge_all_experiments(rate, dir, config, experiments_end=3, ns3_path=__ns3_path, load=None, differentiationDelay=None, errorRate=None):
    if ("delay" in dir) and ("reverse" in dir):
        # remove reverse from dir
        results_folder = 'Results_' + dir.replace("reverse", "forward").replace("delay_", "")
    else:
        results_folder = 'Results_' + dir

    batch_size = 5
    for i in range(int(experiments_end / batch_size) + 1):
        ths = []
        return_dict = multiprocessing.Manager().dict()
        for experiment in range(batch_size * i, min(experiments_end, batch_size * (i + 1))):
            if differentiationDelay is not None and errorRate is not None and ("delay" not in dir):
                if len(os.listdir('{}/scratch/{}/{}/{}/D_{}/f_{}/{}'.format(__ns3_path, results_folder, rate, load, differentiationDelay, errorRate, experiment))) == 0:
                    print(experiment)
                    continue
            else:
                if len(os.listdir('{}/scratch/{}/{}/{}/{}'.format(__ns3_path, results_folder, rate, load, experiment))) == 0:
                    print(experiment)
                    continue
            print("Analyzing experiment: ", experiment)
            ths.append(multiprocessing.Process(target=merge_single_experiment, args=(return_dict, rate, results_folder, config, experiment, ns3_path, differentiationDelay, errorRate, load)))
        
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        print("{} joind".format(i))

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
    experiments = 1
    serviceRateScales = [float(x) for x in config.get('Settings', 'serviceRateScales').split(',')]
    # serviceRateScales = [0.5]
    loads = [float(x) for x in config.get('Settings', 'load').split(',')]
    # loads = [0.4]
    traffics = config.get('Settings', 'traffic').split(',')
    # traffics = ['Google_SearchRPC']
    errorRates = [float(x) for x in config.get('Settings', 'errorRate').split(',')]
    differentiationDelays = [float(x) for x in config.get('Settings', 'differentiationDelay').split(',')]
    if "forward" in args.dir:
        for traffic in traffics:
            for rate in serviceRateScales:
                for load in loads:
                    print("\nAnalyzing experiments for traffic {} rate: {} load: {}".format(traffic, rate, load))
                    merge_all_experiments(rate, args.dir + "/" + traffic, config, experiments_end=experiments, ns3_path=__ns3_path, load=load)
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
                            merge_all_experiments(rate, args.dir + "/" + traffic, config, experiments_end=experiments, ns3_path=__ns3_path, differentiationDelay=differentiationDelay, errorRate=errorRate, load=load)
                            print("Rate {} load {} with {} and {} done".format(rate, load, differentiationDelay, errorRate))
                    print("Traffic {} Rate {} load {} done".format(traffic, rate, load))
                print("Rate {} done".format(rate))
            print("Traffic {} done".format(traffic))

__main__()