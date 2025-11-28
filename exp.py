import os
import time
import configparser
import threading
import argparse
from enum import Enum
import subprocess
import random
import psutil
# __ns3_path = os.popen('locate "ns-3.41" | grep /ns-3.41$').read().splitlines()[0]
__ns3_path = "/media/experiments/ns-allinone-3.41/ns-3.41"
# __ns3_path = '/Users/shossein/Documents/NAL/Flwo-Path_Consistency/ns-allinone-3.41/ns-3.41'

class ReverseType(Enum):
    Delay = 1
    Loss = 2

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
        self.trafficStartTime = "1.0"
        self.trafficStopTime = "10.0"
        self.sampleRate="10.0"
        self.experiments="100"
        self.steadyStart="3"
        self.steadyEnd="18"
        self.serviceRateScales=[]
        self.load=[]
        self.errorRate=[]
        self.differentiationDelay=[]
        self.swtichDstREDQueueDiscMaxSize = "10KB"
        self.switchSrcREDQueueDiscMaxSize = "6KB"
        self.switchSrcREDQueueDiscMaxSize = "15KB"
        self.switchREDQueueDiscMaxSize = "90KB"
        self.switchTXMaxSize = "1p"
        self.MinTh = "0.15"
        self.MaxTh = "0.15"
        self.traffic = []
        self.isDifferentating = False
        self.silentPacketDrop = False
        self.NagleIsEnabled = False
        self.ActiveProbeIsEnabled = False
        self.PassiveProbeIsEnabled = False
        self.probeInterval = "1ms"
        self.incastMessageSize = 10000
        self.incastFactor = 6
        self.incastperiod = "50us"

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
        self.NagleIsEnabled = config.getboolean('Settings', 'Nagle')
        self.ActiveProbeIsEnabled = config.getboolean('Settings', 'ActiveProbe')
        self.PassiveProbeIsEnabled = config.getboolean('Settings', 'PassiveProbe')
        self.app_data_rate = config.get('Settings', 'appDataRate')
        self.duration = config.get('Settings', 'duration')
        self.trafficStartTime = config.get('Settings', 'trafficStartTime')
        self.trafficStopTime = config.get('Settings', 'trafficStopTime')
        self.sampleRate = config.get('Settings', 'sampleRate')
        self.sampleRateScales = [float(x) for x in config.get('Settings', 'sampleRateScales').split(',')]
        self.experiments = config.get('Settings', 'experiments')
        self.serviceRateScales = [float(x) for x in config.get('Settings', 'serviceRateScales').split(',')]
        self.load = [float(x) for x in config.get('Settings', 'load').split(',')]
        self.errorRate = [float(x) for x in config.get('Settings', 'errorRate').split(',')]
        self.differentiationDelay = [float(x) for x in config.get('Settings', 'differentiationDelay').split(',')]
        self.steadyStart = config.get('Settings', 'steadyStart')
        self.steadyEnd = config.get('Settings', 'steadyEnd')
        self.srcHostToSwitchLinkRate = config.get('SingleQueue', 'srcHostToSwitchLinkRate')
        self.bottleneckLinkRate = config.get('SingleQueue', 'bottleneckLinkRate')
        self.ctHostToSwitchLinkRate = config.get('SingleQueue', 'ctHostToSwitchLinkRate')
        self.swtichDstREDQueueDiscMaxSize = config.get('Settings', 'swtichDstREDQueueDiscMaxSize')
        self.switchSrcREDQueueDiscMaxSize = config.get('Settings', 'switchSrcREDQueueDiscMaxSize')
        self.switchREDQueueDiscMaxSize = config.get('DCSim', 'switchREDQueueDiscMaxSize')
        self.incastMessageSize = config.getint('DCSim', 'incastMessageSize')
        self.incastFactor = config.getint('DCSim', 'incastFactor')
        self.incastperiod = config.get('DCSim', 'incastperiod')
        self.switchTXMaxSize = config.get('Settings', 'switchTXMaxSize')
        self.MinTh = config.get('Settings', 'MinTh')
        self.MaxTh = config.get('Settings', 'MaxTh')
        self.traffic = config.get('Settings', 'traffic').split(',')
        self.probeInterval = config.get('Settings', 'probeInterval')




def get_ns3_path(): return __ns3_path

def rebuild_project():
    os.system('{}/ns3 build'.format(get_ns3_path()))

def run_ns3_with_timeout(base_cmd, output_file, timeout_seconds=180, initial_seed=1):
    seed = initial_seed or int(time.time())
    attempt = 1

    while attempt <= 1:
        print(f"Attempt {attempt}: Running simulation with seed {seed}")

        # Add the seed to the command
        full_cmd = f"{base_cmd} --seed={seed} \' > {output_file}"

        try:
            # Start the process
            process = subprocess.Popen(full_cmd, shell=True)
            proc = psutil.Process(process.pid)

            # Wait for the timeout
            try:
                process.wait(timeout=timeout_seconds)
                print(f"Simulation completed successfully with seed {seed}")
                break  # success
            except subprocess.TimeoutExpired:
                print(f"Timeout expired for seed {seed}. Killing process tree...")
                for child in proc.children(recursive=True):
                    child.kill()
                proc.kill()
                time.sleep(1)  # Give the system a moment to reclaim resources

        except Exception as e:
            print(f"Error running simulation with seed {seed}: {e}")

        # Retry with a new seed
        seed = int(time.time()) + random.randint(0, 10000)
        attempt += 1

def run_forward_experiment(exp, singleQueue=False):
    expConfig = ExperimentConfig()
    expConfig.read_config_file('Parameters.config')
    expConfig.isDifferentating = False
    expConfig.silentPacketDrop = False
    os.system('mkdir -p {}/scratch/ECNMC/Results/results_forward/'.format(get_ns3_path()))
    # copy Parameters.config to the results folder
    os.system('cp Parameters.config {}/scratch/ECNMC/Results/results_forward/'.format(get_ns3_path()))
    for traffic in expConfig.traffic:
        for rate in expConfig.serviceRateScales:
            exp_tor_to_agg_link_rate = "{}Mbps".format(round(float(expConfig.tor_to_agg_link_rate.split('M')[0]) * rate, 1))
            exp_bottleNeckLinkRate = "{}Mbps".format(round(float(expConfig.bottleneckLinkRate.split('M')[0]) * rate, 1))
            # exp_errorRate = "{}".format(float(expConfig.errorRate) * expConfig.errorRateScale[0])
            exp_errorRate = "0.0"
            for load in expConfig.load:
                for i in exp:
                    os.system('mkdir -p {}/scratch/ECNMC/Results/results_forward/{}'.format(get_ns3_path(), i + 1))
                    if singleQueue:
                        # NS_LOG="DefaultSimulatorImpl=*" 
                        cmd = ('{}/ns3 run \'DatacenterSimulation '.format(get_ns3_path()) +
                            '{} '.format(singleQueue) +
                            '--srcHostToSwitchLinkRate={} '.format(expConfig.srcHostToSwitchLinkRate) +
                            '--ctHostToSwitchLinkRate={} '.format(expConfig.ctHostToSwitchLinkRate) +
                            '--hostToSwitchLinkDelay={} '.format(expConfig.host_to_tor_link_delay) +
                            '--bottleneckLinkRate={} '.format(exp_bottleNeckLinkRate) +
                            '--load={} '.format(load) +
                            '--pctPacedBack={} '.format(expConfig.pct_paced_back) +
                            '--duration={} '.format(expConfig.duration) +
                            '--sampleRate={} '.format(expConfig.sampleRate) +
                            '--experiment={} '.format(i + 1) +
                            '--errorRate={} '.format(exp_errorRate) +
                            '--trafficStartTime={} '.format(expConfig.trafficStartTime) +
                            '--trafficStopTime={} '.format(expConfig.trafficStopTime) +
                            '--steadyStartTime={} '.format(expConfig.steadyStart) +
                            '--steadyStopTime={} '.format(expConfig.steadyEnd) +
                            '--swtichDstREDQueueDiscMaxSize={} '.format(expConfig.swtichDstREDQueueDiscMaxSize) +
                            '--switchSrcREDQueueDiscMaxSize={} '.format(expConfig.switchSrcREDQueueDiscMaxSize) +
                            '--switchTXMaxSize={} '.format(expConfig.switchTXMaxSize) +
                            '--minTh={} '.format(expConfig.MinTh) +
                            '--maxTh={} '.format(expConfig.MaxTh) +
                            '--dirName=' + 'forward ' +
                            '--traffic={} '.format(traffic) +
                            '--Nagle={} '.format(expConfig.NagleIsEnabled) +
                            '--ActiveProbe={} '.format(expConfig.ActiveProbeIsEnabled) +
                            '--PassiveProbe={} '.format(expConfig.PassiveProbeIsEnabled) +
                            '--differentiationDelay={} '.format(expConfig.differentiationDelay[0]) +
                            '--isDifferentating={} '.format(expConfig.isDifferentating) +
                            '--silentPacketDrop={} '.format(expConfig.silentPacketDrop) + 
                            '--probeInterval={} '.format(expConfig.probeInterval)
                        )
                    else:
                        cmd = (
                            '{}/ns3 run \'DatacenterSimulation '.format(get_ns3_path()) +
                            '{} '.format(singleQueue) +
                            '--hostToTorLinkRate={} '.format(expConfig.host_to_tor_link_rate) +
                            '--hostToTorLinkRateCrossTraffic={} '.format(expConfig.host_to_tor_cross_traffic_rate) +
                            '--torToAggLinkRate={} '.format(exp_tor_to_agg_link_rate) +
                            '--aggToCoreLinkRate={} '.format(expConfig.agg_to_core_link_rate) +
                            '--hostToTorLinkDelay={} '.format(expConfig.host_to_tor_link_delay) +
                            '--torToAggLinkDelay={} '.format(expConfig.tor_to_agg_link_delay) +
                            '--aggToCoreLinkDelay={} '.format(expConfig.agg_to_core_link_delay) +
                            '--load={} '.format(load) +
                            '--pctPacedBack={} '.format(expConfig.pct_paced_back) +
                            '--duration={} '.format(expConfig.duration) +
                            '--sampleRate={} '.format(expConfig.sampleRate) +
                            '--experiment={} '.format(i + 1) +
                            '--errorRate={} '.format(exp_errorRate) +
                            '--trafficStartTime={} '.format(expConfig.trafficStartTime) +
                            '--trafficStopTime={} '.format(expConfig.trafficStopTime) +
                            '--steadyStartTime={} '.format(expConfig.steadyStart) +
                            '--steadyStopTime={} '.format(expConfig.steadyEnd) +
                            '--switchREDQueueDiscMaxSize={} '.format(expConfig.switchREDQueueDiscMaxSize) +
                            '--switchSrcREDQueueDiscMaxSize={} '.format(expConfig.switchSrcREDQueueDiscMaxSize) +
                            '--switchTXMaxSize={} '.format(expConfig.switchTXMaxSize) +
                            '--minTh={} '.format(expConfig.MinTh) +
                            '--maxTh={} '.format(expConfig.MaxTh) +
                            '--dirName=' + 'forward ' +
                            '--traffic={} '.format(traffic) +
                            '--Nagle={} '.format(expConfig.NagleIsEnabled) +
                            '--ActiveProbe={} '.format(expConfig.ActiveProbeIsEnabled) +
                            '--PassiveProbe={} '.format(expConfig.PassiveProbeIsEnabled) +
                            '--differentiationDelay={} '.format(expConfig.differentiationDelay[0]) +
                            '--isDifferentating={} '.format(expConfig.isDifferentating) +
                            '--silentPacketDrop={} '.format(expConfig.silentPacketDrop) +
                            '--probeInterval={} '.format(expConfig.probeInterval) +
                            '--incastMessageSize={} '.format(expConfig.incastMessageSize) +
                            '--incastFactor={} '.format(expConfig.incastFactor) +
                            '--incastperiod={} '.format(expConfig.incastperiod)
                        )
                    output_file = '{}/scratch/ECNMC/Results/results_forward/result_{}.txt'.format(get_ns3_path(), i)
                    run_ns3_with_timeout(cmd, output_file, timeout_seconds=36000, initial_seed=i + 1)
                    os.system('mkdir -p {}/scratch/Results_forward/{}/{}/{}/{}'.format(get_ns3_path(), traffic, rate, load, i))
                    os.system('mv {}/scratch/ECNMC/Results/results_forward/{}/*.csv {}/scratch/Results_forward/{}/{}/{}/{}'.format(get_ns3_path(), i + 1, get_ns3_path(), traffic, rate, load, i))
                    # os.system('mv {}/scratch/ECNMC/Results/*_cwnd.csv {}/scratch/Results_forward/{}/{}'.format(get_ns3_path(), get_ns3_path(), rate, i))
                    os.system('mkdir -p {}/scratch/ECNMC/Results/results_forward/{}/{}/{}'.format(get_ns3_path(), traffic, rate, load))
                    print('\tExperiment {} with rate {} and load {} done'.format(i, rate, load))
                print('traffic {} Rate {} , load {} done'.format(traffic, rate, load))
            print('Rate {} done'.format(rate))
        print('Traffic {} done'.format(traffic))

def run_reverse_experiment(exp, singleQueue=False, type=ReverseType.Delay):
    expConfig = ExperimentConfig()
    expConfig.read_config_file('Parameters.config')
    type_name = 'delay' if type == ReverseType.Delay else 'loss'
    if type == ReverseType.Delay:
        expConfig.isDifferentating = True
        expConfig.silentPacketDrop = False
    else:
        expConfig.isDifferentating = False
        expConfig.silentPacketDrop = True
    os.system('mkdir -p {}/scratch/ECNMC/Results/results_reverse_{}/'.format(get_ns3_path(), type_name))
    os.system('cp Parameters.config {}/scratch/ECNMC/Results/results_reverse_{}/'.format(get_ns3_path(), type_name))
    for traffic in expConfig.traffic:
        for CRate in expConfig.serviceRateScales:
            for load in expConfig.load:
                for DiffRate in expConfig.differentiationDelay: 
                    for errorRate in expConfig.errorRate:
                        exp_tor_to_agg_link_rate = "{}Mbps".format(round(float(expConfig.tor_to_agg_link_rate.split('M')[0]) * CRate, 1))
                        exp_bottleNeckLinkRate = "{}Mbps".format(round(float(expConfig.bottleneckLinkRate.split('M')[0]) * CRate, 1))
                        for i in exp:
                            os.system('mkdir -p {}/scratch/ECNMC/Results/results_reverse_{}/{}'.format(get_ns3_path(), type_name, i + 1))
                            if singleQueue:
                                cmd = (
                                    '{}/ns3 run \'DatacenterSimulation '.format(get_ns3_path()) +
                                    '{} '.format(singleQueue) +
                                    '--srcHostToSwitchLinkRate={} '.format(expConfig.srcHostToSwitchLinkRate) +
                                    '--ctHostToSwitchLinkRate={} '.format(expConfig.ctHostToSwitchLinkRate) +
                                    '--hostToSwitchLinkDelay={} '.format(expConfig.host_to_tor_link_delay) +
                                    '--bottleneckLinkRate={} '.format(exp_bottleNeckLinkRate) +
                                    '--pctPacedBack={} '.format(expConfig.pct_paced_back) +
                                    '--duration={} '.format(expConfig.duration) +
                                    '--sampleRate={} '.format(expConfig.sampleRate) +
                                    '--load={} '.format(load) +
                                    '--experiment={} '.format(i + 1) +
                                    '--errorRate={} '.format(errorRate) +
                                    '--trafficStartTime={} '.format(i * float(expConfig.duration)) +
                                    '--trafficStopTime={} '.format((i + 1) * float(expConfig.duration)) +
                                    '--steadyStartTime={} '.format(expConfig.steadyStart) +
                                    '--steadyStopTime={} '.format(expConfig.steadyEnd) +
                                    '--swtichDstREDQueueDiscMaxSize={} '.format(expConfig.swtichDstREDQueueDiscMaxSize) +
                                    '--switchSrcREDQueueDiscMaxSize={} '.format(expConfig.switchSrcREDQueueDiscMaxSize) +
                                    '--switchTXMaxSize={} '.format(expConfig.switchTXMaxSize) +
                                    '--minTh={} '.format(expConfig.MinTh) +
                                    '--maxTh={} '.format(expConfig.MaxTh) +
                                    '--dirName=' + 'reverse ' +
                                    '--traffic={} '.format(traffic) +
                                    '--Nagle={} '.format(expConfig.NagleIsEnabled) +
                                    '--ActiveProbe={} '.format(expConfig.ActiveProbeIsEnabled) +
                                    '--PassiveProbe={} '.format(expConfig.PassiveProbeIsEnabled) +
                                    '--probeInterval={} '.format(expConfig.probeInterval) +
                                    '--differentiationDelay={} '.format(DiffRate) +
                                    '--isDifferentating={} '.format(expConfig.isDifferentating) +
                                    '--silentPacketDrop={} '.format(expConfig.silentPacketDrop) + 
                                    '--dirName=' + 'reverse_{} '.format(type_name)
                                )
                            else:
                                cmd = (
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
                                    '--errorRate={} '.format(errorRate) +
                                    '--trafficStartTime={} '.format(i * float(expConfig.duration)) +
                                    '--trafficStopTime={} '.format((i + 1) * float(expConfig.duration)) +
                                    '--steadyStartTime={} '.format(expConfig.steadyStart) +
                                    '--steadyStopTime={} '.format(expConfig.steadyEnd) +
                                    '--dirName=' + 'reverse_{} '.format(type_name)
                                )
                            output_file = '{}/scratch/ECNMC/Results/results_reverse_{}/result_{}.txt'.format(get_ns3_path(), type_name, i)
                            run_ns3_with_timeout(cmd, output_file, timeout_seconds=600, initial_seed=i + 1)
                            os.system('mkdir -p {}/scratch/Results_reverse_{}/{}/{}/{}/D_{}/f_{}/{}'.format(get_ns3_path(), type_name, traffic, CRate, load, DiffRate, errorRate, i))
                            os.system('mv {}/scratch/ECNMC/Results/results_reverse_{}/{}/*.csv {}/scratch/Results_reverse_{}/{}/{}/{}/D_{}/f_{}/{}'.format(get_ns3_path(), type_name, i + 1, get_ns3_path(), type_name, traffic, CRate, load, DiffRate, errorRate, i))
                            os.system('mkdir -p {}/scratch/ECNMC/Results/results_reverse_{}/{}/{}/D_{}/f_{}'.format(get_ns3_path(), type_name, traffic, CRate, load, DiffRate, errorRate))
                            print('\tExperiment {} with {} rate {} load {} and diff {} with fraction {} done'.format(i, traffic, CRate, load, DiffRate, errorRate))
                        print('traffic {} Rate {} load {}, diff {} with fraction {} done'.format(traffic, CRate, load, DiffRate, errorRate))
                print('Rate {} load {} done'.format(CRate, load))
            print('Rate {} done'.format(CRate))
        print('Traffic {} done'.format(traffic))

def run_param_experiments(exp):
    expConfig = ExperimentConfig()
    expConfig.read_config_file('Parameters.config')
    os.system('mkdir -p {}/scratch/ECNMC/Results/results_params/'.format(get_ns3_path()))
    for rate in expConfig.sampleRateScales:
        exp_tor_to_agg_link_rate = "{}Mbps".format(round(float(expConfig.tor_to_agg_link_rate.split('M')[0]) * expConfig.serviceRateScales[0], 1))
        exp_errorRate = "{}".format(float(expConfig.errorRate))
        exp_sampleRate = "{}".format(float(expConfig.sampleRate) * rate)
        for i in exp:
            os.system('mkdir -p {}/scratch/ECNMC/Results/results_params/{}'.format(get_ns3_path(), i + 1))
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
                '--sampleRate={} '.format(exp_sampleRate) +
                '--experiment={} '.format(i + 1) +
                '--errorRate={} '.format(exp_errorRate) +
                '--trafficStartTime={} '.format(i * float(expConfig.duration)) +
                '--trafficStopTime={} '.format((i + 1) * float(expConfig.duration)) +
                '--steadyStartTime={} '.format(expConfig.steadyStart) +
                '--steadyStopTime={} '.format(expConfig.steadyEnd) +
                '--dirName=' + 'params' +
                '\' > {}/scratch/ECNMC/Results/results_params/result_{}.txt'.format(get_ns3_path(), i)
            )
    
            os.system('mkdir -p {}/scratch/Results_params/{}/{}'.format(get_ns3_path(), rate, i))
            os.system('mv {}/scratch/ECNMC/Results/results_params/{}/*.csv {}/scratch/Results_params/{}/{}'.format(get_ns3_path(), i + 1, get_ns3_path(), rate, i))
            os.system('mkdir -p {}/scratch/ECNMC/Results/results_params/{}'.format(get_ns3_path(), rate))
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
            '\' > {}/scratch/ECNMC/Results/results_burst/result_{}.txt'.format(get_ns3_path(), i)
        )

        os.system('mkdir -p {}/scratch/Results_burst/{}/{}'.format(get_ns3_path(), rate, 0))
        os.system('mv {}/scratch/ECNMC/Results/results_burst/{}/*.csv {}/scratch/Results_burst/{}/{}'.format(get_ns3_path(), i + 1, get_ns3_path(), rate, 0))
        os.system('mkdir -p {}/scratch/ECNMC/Results/results_burst/{}'.format(get_ns3_path(), rate))
        print('\tExperiment {} done'.format(i))
        print('Rate {} done'.format(rate))

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

parser.add_argument("--IsSingleQueue",
                    required=False,
                    dest="IsSingleQueue",
                    help="If the experiment is the single queue experiment or not!", 
                    type=int,
                    default=0)

parser.add_argument("--ReverseType",
                    required=False,
                    dest="reverseType",
                    help="In case of reverse experiment, if the experiment is for delayy differentiation or silent packet drop!",
                    type=int,
                    default=1)

args = parser.parse_args()
args.IsForward = int(args.IsForward)
args.IsTest = bool(args.IsTest)
args.IsSingleQueue = bool(args.IsSingleQueue)
args.reverseType = ReverseType(int(args.reverseType))
# rebuild_project()
if (args.IsForward == 1):
    if (args.IsTest):
        run_forward_experiment([0], args.IsSingleQueue)
    else:
        expConfig = ExperimentConfig()
        expConfig.read_config_file('Parameters.config')
        expConfig.experiments = int(expConfig.experiments)
        ths = []
        numOfThs = 15
        for th in range(numOfThs):
            ths.append(threading.Thread(target=run_forward_experiment, args=([i for i in range(int(th * expConfig.experiments / numOfThs), int((th + 1) * expConfig.experiments / numOfThs))], args.IsSingleQueue, )))

        for th in ths:
            th.start()

        for th in ths:
            th.join()
elif(args.IsForward == 0):
    if (args.IsTest):
        run_reverse_experiment([0], args.IsSingleQueue, args.reverseType)
    else:
        expConfig = ExperimentConfig()
        expConfig.read_config_file('Parameters.config')
        expConfig.experiments = int(expConfig.experiments)
        ths = []
        numOfThs = 30
        for th in range(numOfThs):
            ths.append(threading.Thread(target=run_reverse_experiment, args=([i for i in range(int(th * expConfig.experiments / numOfThs), int((th + 1) * expConfig.experiments / numOfThs))], args.IsSingleQueue, args.reverseType, )))

        for th in ths:
            th.start()

        for th in ths:
            th.join()
elif(args.IsForward == 2):
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

elif(args.IsForward == 3):
    if (args.IsTest):
        run_param_experiments([0])
    else:
        expConfig = ExperimentConfig()
        expConfig.read_config_file('Parameters.config')
        expConfig.experiments = int(expConfig.experiments)
        ths = []
        numOfThs = 30
        for th in range(numOfThs):
            ths.append(threading.Thread(target=run_param_experiments, args=([i for i in range(int(th * expConfig.experiments / numOfThs), int((th + 1) * expConfig.experiments / numOfThs))], )))

        for th in ths:
            th.start()

        for th in ths:
            th.join()

