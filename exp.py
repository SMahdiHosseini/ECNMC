import os
import time
import configparser

__ns3_path = os.popen('locate "ns-3.41" | grep /ns-3.41$').read().splitlines()[0]

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
        self.serviceRateScales = [float(x) for x in config.get('Settings', 'serviceRateScales').split(',')]




def get_ns3_path(): return __ns3_path

def rebuild_project():
    os.system('{}/ns3 build'.format(get_ns3_path()))

def run_experiment():
    expConfig = ExperimentConfig()
    expConfig.read_config_file('Parameters.config')
    os.system('mkdir -p {}/scratch/ECNMC/results/'.format(get_ns3_path()))
    for rate in expConfig.serviceRateScales:
        exp_tor_to_agg_link_rate = "{}Mbps".format(round(float(expConfig.tor_to_agg_link_rate.split('M')[0]) * rate, 1))
        for i in range(int(expConfig.experiments)):
            os.system(
                '{}/ns3 run \'N4_datacenter_switch '.format(get_ns3_path()) +
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
                '\' > {}/scratch/ECNMC/results/result.txt'.format(get_ns3_path())
            )
    
            os.system('mkdir -p {}/scratch/Results/{}/{}'.format(get_ns3_path(), rate, i))
            os.system('mv {}/scratch/ECNMC/results/*.csv {}/scratch/Results/{}/{}'.format(get_ns3_path(), get_ns3_path(), rate, i))
            os.system('mkdir -p {}/scratch/ECNMC/results/{}'.format(get_ns3_path(), rate))
            print('\tExperiment {} done'.format(i))
        print('Rate {} done'.format(rate))

# main
# rebuild_project()
run_experiment()