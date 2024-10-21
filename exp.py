import os
import time
import configparser
import threading
import argparse
# __ns3_path = os.popen('locate "ns-3.41" | grep /ns-3.41$').read().splitlines()[0]
__ns3_path = "/media/experiments/ns-allinone-3.41/ns-3.41"

class ExperimentConfig:
    def __init__(self):
        self.host_to_tor_link_rate = "10Mbps"
        self.host_to_tor_cross_traffic_rate = "10Mbps"
        self.tor_to_agg_link_rate = "100Mbps"
        self.agg_to_core_link_rate = "100Mbps"
        self.host_to_tor_link_delay = "3ms"
        self.tor_to_agg_link_delay = "3ms"
        self.agg_to_core_link_delay = "3ms"
        self.pct_paced_back = 0.8
        self.app_data_rate = "20Mbps"
        self.duration = "10s"
        self.sampleRate="10.0"
        self.experiments="100"
        self.errorRate="0.002"
        self.steadyStart="3"
        self.steadyEnd="18"
        self.serviceRateScales=[]

    def read_config_file(self, config_file):
        config = configparser.ConfigParser()
        config.read('Parameters.config')
        self.host_to_tor_link_rate = config.get('Settings', 'hostToTorLinkRate')
        self.host_to_tor_cross_traffic_rate = config.get('Settings', 'hostToTorCrossTrafficRate')
        self.tor_to_agg_link_rate = config.get('Settings', 'torToAggLinkRate')
        self.agg_to_core_link_rate = config.get('Settings', 'aggToCoreLinkRate')
        self.host_to_tor_link_delay = config.get('Settings', 'hostToTorLinkDelay')
        self.tor_to_agg_link_delay = config.get('Settings', 'torToAggLinkDelay')
        self.agg_to_core_link_delay = config.get('Settings', 'aggToCoreLinkDelay')
        self.pct_paced_back = config.getfloat('Settings', 'pctPacedBack')
        self.app_data_rate = config.get('Settings', 'appDataRate')
        self.duration = config.get('Settings', 'duration')
        self.sampleRate = config.get('Settings', 'sampleRate')
        self.experiments = config.get('Settings', 'experiments')
        self.errorRate = config.get('Settings', 'errorRate')
        self.serviceRateScales = [float(x) for x in config.get('Settings', 'serviceRateScales').split(',')]
        self.errorRateScale = [float(x) for x in config.get('Settings', 'errorRateScale').split(',')]
        self.steadyStart = config.get('Settings', 'steadyStart')
        self.steadyEnd = config.get('Settings', 'steadyEnd')




def get_ns3_path(): return __ns3_path

def rebuild_project():
    os.system('{}/ns3 build'.format(get_ns3_path()))

def run_forward_experiment(exp):
    expConfig = ExperimentConfig()
    expConfig.read_config_file('Parameters.config')
    os.system('mkdir -p {}/scratch/ECNMC/results_forward/'.format(get_ns3_path()))
    for rate in expConfig.serviceRateScales:
        exp_tor_to_agg_link_rate = "{}Mbps".format(round(float(expConfig.tor_to_agg_link_rate.split('M')[0]) * rate, 1))
        exp_errorRate = "{}".format(float(expConfig.errorRate) * expConfig.errorRateScale[0])
        for i in exp:
            os.system('mkdir -p {}/scratch/ECNMC/results_forward/{}'.format(get_ns3_path(), i + 1))
            os.system(
                '{}/ns3 run \'DatacenterSimulation '.format(get_ns3_path()) +
                '--hostToTorLinkRate={} '.format(expConfig.host_to_tor_link_rate) +
                '--hostToTorLinkRateCrossTraffic={} '.format(expConfig.host_to_tor_cross_traffic_rate) +
                '--torToAggLinkRate={} '.format(exp_tor_to_agg_link_rate) +
                '--aggToCoreLinkRate={} '.format(expConfig.agg_to_core_link_rate) +
                '--hostToTorLinkDelay={} '.format(expConfig.host_to_tor_link_delay) +
                '--torToAggLinkDelay={} '.format(expConfig.tor_to_agg_link_delay) +
                '--aggToCoreLinkDelay={} '.format(expConfig.agg_to_core_link_delay) +
                '--pctPacedBack={} '.format(expConfig.pct_paced_back) +
                '--appDataRate={} '.format(expConfig.app_data_rate) +
                '--duration={} '.format(expConfig.duration) +
                '--sampleRate={} '.format(expConfig.sampleRate) +
                '--experiment={} '.format(i + 1) +
                '--errorRate={} '.format(exp_errorRate) +
                '--trafficStartTime={} '.format(i * float(expConfig.duration)) +
                '--trafficStopTime={} '.format((i + 1) * float(expConfig.duration)) +
                '--steadyStartTime={} '.format(expConfig.steadyStart) +
                '--steadyStopTime={} '.format(expConfig.steadyEnd) +
                '--dirName=' + 'forward' +
                '\' > {}/scratch/ECNMC/results_forward/result_{}.txt'.format(get_ns3_path(), i)
            )
    
            os.system('mkdir -p {}/scratch/Results_forward/{}/{}'.format(get_ns3_path(), rate, i))
            os.system('mv {}/scratch/ECNMC/results_forward/{}/*.csv {}/scratch/Results_forward/{}/{}'.format(get_ns3_path(), i + 1, get_ns3_path(), rate, i))
            os.system('mkdir -p {}/scratch/ECNMC/results_forward/{}'.format(get_ns3_path(), rate))
            print('\tExperiment {} done'.format(i))
        print('Rate {} done'.format(rate))

def run_reverse_experiment(exp):
    expConfig = ExperimentConfig()
    expConfig.read_config_file('Parameters.config')
    os.system('mkdir -p {}/scratch/ECNMC/results_reverse/'.format(get_ns3_path()))
    for rate in expConfig.errorRateScale:
        exp_tor_to_agg_link_rate = "{}Mbps".format(round(float(expConfig.tor_to_agg_link_rate.split('M')[0]) * expConfig.serviceRateScales[0], 1))
        exp_errorRate = "{}".format(float(expConfig.errorRate) * rate)
        for i in exp:
            os.system('mkdir -p {}/scratch/ECNMC/results_reverse/{}'.format(get_ns3_path(), i + 1))
            os.system(
                '{}/ns3 run \'DatacenterSimulation '.format(get_ns3_path()) +
                '--hostToTorLinkRate={} '.format(expConfig.host_to_tor_link_rate) +
                '--hostToTorLinkRateCrossTraffic={} '.format(expConfig.host_to_tor_cross_traffic_rate) +
                '--torToAggLinkRate={} '.format(exp_tor_to_agg_link_rate) +
                '--aggToCoreLinkRate={} '.format(expConfig.agg_to_core_link_rate) +
                '--hostToTorLinkDelay={} '.format(expConfig.host_to_tor_link_delay) +
                '--torToAggLinkDelay={} '.format(expConfig.tor_to_agg_link_delay) +
                '--aggToCoreLinkDelay={} '.format(expConfig.agg_to_core_link_delay) +
                '--pctPacedBack={} '.format(expConfig.pct_paced_back) +
                '--appDataRate={} '.format(expConfig.app_data_rate) +
                '--duration={} '.format(expConfig.duration) +
                '--sampleRate={} '.format(expConfig.sampleRate) +
                '--experiment={} '.format(i + 1) +
                '--errorRate={} '.format(exp_errorRate) +
                '--trafficStartTime={} '.format(i * float(expConfig.duration)) +
                '--trafficStopTime={} '.format((i + 1) * float(expConfig.duration)) +
                '--steadyStartTime={} '.format(expConfig.steadyStart) +
                '--steadyStopTime={} '.format(expConfig.steadyEnd) +
                '--dirName=' + 'reverse' +
                '\' > {}/scratch/ECNMC/results_reverse/result_{}.txt'.format(get_ns3_path(), i)
            )
    
            os.system('mkdir -p {}/scratch/Results_reverse/{}/{}'.format(get_ns3_path(), rate, i))
            os.system('mv {}/scratch/ECNMC/results_reverse/{}/*.csv {}/scratch/Results_reverse/{}/{}'.format(get_ns3_path(), i + 1, get_ns3_path(), rate, i))
            os.system('mkdir -p {}/scratch/ECNMC/results_reverse/{}'.format(get_ns3_path(), rate))
            print('\tExperiment {} done'.format(i))
        print('Rate {} done'.format(rate))

def run_burst_experiment(exp, rate):
    expConfig = ExperimentConfig()
    expConfig.read_config_file('Parameters.config')
    os.system('mkdir -p {}/scratch/ECNMC/results_burst/'.format(get_ns3_path()))
    exp_tor_to_agg_link_rate = "{}Mbps".format(round(float(expConfig.tor_to_agg_link_rate.split('M')[0]) * rate, 1))
    for i in exp:
        os.system('mkdir -p {}/scratch/ECNMC/results_burst/{}'.format(get_ns3_path(), i + 1))
        os.system(
            '{}/ns3 run \'DatacenterSimulation '.format(get_ns3_path()) +
            '--hostToTorLinkRate={} '.format(expConfig.host_to_tor_link_rate) +
            '--hostToTorLinkRateCrossTraffic={} '.format(expConfig.host_to_tor_cross_traffic_rate) +
            '--torToAggLinkRate={} '.format(exp_tor_to_agg_link_rate) +
            '--aggToCoreLinkRate={} '.format(expConfig.agg_to_core_link_rate) +
            '--hostToTorLinkDelay={} '.format(expConfig.host_to_tor_link_delay) +
            '--torToAggLinkDelay={} '.format(expConfig.tor_to_agg_link_delay) +
            '--aggToCoreLinkDelay={} '.format(expConfig.agg_to_core_link_delay) +
            '--pctPacedBack={} '.format(expConfig.pct_paced_back) +
            '--appDataRate={} '.format(expConfig.app_data_rate) +
            '--duration={} '.format(expConfig.duration) +
            '--sampleRate={} '.format(expConfig.sampleRate) +
            '--experiment={} '.format(i + 1) +
            '--errorRate={} '.format(expConfig.errorRate) +
            '--trafficStartTime={} '.format(i * float(expConfig.duration)) +
            '--trafficStopTime={} '.format((i + 1) * float(expConfig.duration)) +
            '--steadyStartTime={} '.format(expConfig.steadyStart) +
            '--steadyStopTime={} '.format(expConfig.steadyEnd) +
            '--dirName=' + 'burst' +
            '\' > {}/scratch/ECNMC/results_burst/result_{}.txt'.format(get_ns3_path(), i)
        )

        os.system('mkdir -p {}/scratch/Results_burst/{}/{}'.format(get_ns3_path(), rate, 0))
        os.system('mv {}/scratch/ECNMC/results_burst/{}/*.csv {}/scratch/Results_burst/{}/{}'.format(get_ns3_path(), i + 1, get_ns3_path(), rate, 0))
        os.system('mkdir -p {}/scratch/ECNMC/results_burst/{}'.format(get_ns3_path(), rate))
        print('\tExperiment {} done'.format(i))
        print('Rate {} done'.format(rate))

# def run_experiment(exp):
#     expConfig = ExperimentConfig()
#     expConfig.read_config_file('Parameters.config')
#     # os.system('mkdir -p {}/scratch/ECNMC/results/'.format(get_ns3_path()))
#     for rate in expConfig.errorRateScale:
#     # for rate in expConfig.serviceRateScales:
#         # exp_tor_to_agg_link_rate = "{}Mbps".format(round(float(expConfig.tor_to_agg_link_rate.split('M')[0]) * rate, 1))
#         exp_tor_to_agg_link_rate = "{}Mbps".format(round(float(expConfig.tor_to_agg_link_rate.split('M')[0]), 1))
#         exp_errorRate = "{}".format(float(expConfig.errorRate) * rate)
#         # exp_errorRate = "{}".format(float(expConfig.errorRate))
#         for i in exp:
#             os.system('mkdir -p {}/scratch/ECNMC/results/{}'.format(get_ns3_path(), i + 1))
#             os.system(
#                 '{}/ns3 run \'DatacenterSimulation '.format(get_ns3_path()) +
#                 '--hostToTorLinkRate={} '.format(expConfig.host_to_tor_link_rate) +
#                 '--hostToTorLinkRateCrossTraffic={} '.format(expConfig.host_to_tor_cross_traffic_rate) +
#                 '--torToAggLinkRate={} '.format(exp_tor_to_agg_link_rate) +
#                 '--aggToCoreLinkRate={} '.format(expConfig.agg_to_core_link_rate) +
#                 '--hostToTorLinkDelay={} '.format(expConfig.host_to_tor_link_delay) +
#                 '--torToAggLinkDelay={} '.format(expConfig.tor_to_agg_link_delay) +
#                 '--aggToCoreLinkDelay={} '.format(expConfig.agg_to_core_link_delay) +
#                 '--pctPacedBack={} '.format(expConfig.pct_paced_back) +
#                 '--appDataRate={} '.format(expConfig.app_data_rate) +
#                 '--duration={} '.format(expConfig.duration) +
#                 '--sampleRate={} '.format(expConfig.sampleRate) +
#                 '--experiment={} '.format(i + 1) +
#                 '--errorRate={}'.format(exp_errorRate) +
#                 '\' > {}/scratch/ECNMC/results/result.txt'.format(get_ns3_path())
#             )
    
#             os.system('mkdir -p {}/scratch/Results/{}/{}'.format(get_ns3_path(), rate, i))
#             os.system('mv {}/scratch/ECNMC/results/{}/*.csv {}/scratch/Results/{}/{}'.format(get_ns3_path(), i + 1, get_ns3_path(), rate, i))
#             os.system('mkdir -p {}/scratch/ECNMC/results_postProcessing/{}'.format(get_ns3_path(), rate))
#             print('\tExperiment {} done'.format(i))
#         print('Rate {} done'.format(rate))

# main
parser=argparse.ArgumentParser()
parser.add_argument("--IsForward",
                    required=True, 
                    dest="IsForward",
                    help="If the experiment is the straitforward experiment or the reverse experiment!", 
                    type=int,
                    default=1)

parser.add_argument("--IsTest",
                    required=True,
                    dest="IsTest",
                    help="If the experiment is the test experiment(just 1 to see if everything works) or not (runnig all the experiments)!", 
                    type=int,
                    default=1)

args = parser.parse_args()
args.IsForward = int(args.IsForward)
args.IsTest = bool(args.IsTest)
# rebuild_project()
if (args.IsForward == 1):
    if (args.IsTest):
        run_forward_experiment([0])
    else:
        expConfig = ExperimentConfig()
        expConfig.read_config_file('Parameters.config')
        expConfig.experiments = int(expConfig.experiments)
        ths = []
        numOfThs = 30
        for th in range(numOfThs):
            ths.append(threading.Thread(target=run_forward_experiment, args=([i for i in range(int(th * expConfig.experiments / numOfThs), int((th + 1) * expConfig.experiments / numOfThs))], )))

        for th in ths:
            th.start()

        for th in ths:
            th.join()
elif(args.IsForward == 0):
    if (args.IsTest):
        run_reverse_experiment([0])
    else:
        expConfig = ExperimentConfig()
        expConfig.read_config_file('Parameters.config')
        expConfig.experiments = int(expConfig.experiments)
        ths = []
        numOfThs = 30
        for th in range(numOfThs):
            ths.append(threading.Thread(target=run_reverse_experiment, args=([i for i in range(int(th * expConfig.experiments / numOfThs), int((th + 1) * expConfig.experiments / numOfThs))], )))

        for th in ths:
            th.start()

        for th in ths:
            th.join()
else:
    expConfig = ExperimentConfig()
    expConfig.read_config_file('Parameters.config')
    ths = []
    numOfThs = len(expConfig.serviceRateScales)
    # numOfThs = 1
    for th in range(numOfThs):
        ths.append(threading.Thread(target=run_burst_experiment, args=([th], expConfig.serviceRateScales[th], )))

    for th in ths:
        th.start()

    for th in ths:
        th.join()

