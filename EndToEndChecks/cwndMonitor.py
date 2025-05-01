from Utils import *
import pandas as pd
import configparser
import matplotlib.pyplot as plt
import numpy as np
import argparse

# __ns3_path = os.popen('locate "ns-3.41" | grep /ns-3.41$').read().splitlines()[0]
__ns3_path = "/media/experiments/ns-allinone-3.41/ns-3.41"
# __ns3_path = '/Users/shossein/Documents/NAL/Flwo-Path_Consistency/ns-allinone-3.41/ns-3.41'

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
    # steadyStart = 400000000
    # steadyEnd = 500000000
    experiments = int(config.get('Settings', 'experiments'))
    serviceRateScales = [float(x) for x in config.get('Settings', 'serviceRateScales').split(',')]
    loads = [float(x) for x in config.get('Settings', 'load').split(',')]
    # serviceRateScales = [0.77]
    # serviceRateScales = [0.77, 0.79, 0.81, 0.83, 0.85, 0.87, 0.89, 0.91, 0.93, 0.95, 0.97, 0.99]
    # serviceRateScales = [1.01, 1.03, 1.05, 1.07]
    colormap = plt.cm.gist_ncar
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(serviceRateScales)))))
    plt.figure(figsize=(20, 10))
    experiments = 1
    for rate in serviceRateScales:
        for load in loads:
            print("\nAnalyzing experiments for rate: ", rate)
            full_df = pd.read_csv('{}/scratch/Results_{}/{}/{}/0/50001_cwnd.csv'.format(__ns3_path, args.dir, rate, load))
            full_df.columns = ['time', 'cwnd']
            # full_df = full_df[full_df['time'] > steadyStart]
            # full_df = full_df[full_df['time'] < steadyEnd]
            full_df['cwnd'] = full_df['cwnd'] / 1448
            print("Rate {} {} done".format(rate, experiments))
            if rate not in serviceRateScales:
                continue
            # plot the cwnd over time
            plt.plot(full_df['time'], full_df['cwnd'], label='Rate {}'.format(rate))
        
    plt.ylim(0)
    plt.xlabel('Time (s)')
    plt.ylabel('Congestion Window (MSS)')
    plt.title('Congestion Window over Time for rate {}'.format(rate))
    plt.legend()
    plt.savefig('../Results/results_{}/{}_to_{}_cwnd.png'.format(args.dir, min(serviceRateScales), max(serviceRateScales)))
    plt.close()


__main__()