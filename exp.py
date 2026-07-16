import os
import time
import csv
import configparser
import threading
import argparse
from datetime import datetime
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

def monitor_simulation_memory(process, output_csv, stop_event, interval=0.5):
    """Write aggregate memory usage for an ns-3 launcher and its process tree."""
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    start = time.monotonic()
    rss_samples = []
    vms_samples = []
    elapsed_samples = []
    known_processes = {}

    with open(output_csv, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "timestamp", "elapsed_sec", "rss_MB", "vms_MB",
            "cpu_percent", "threads",
        ])

        while not stop_event.is_set():
            try:
                root = psutil.Process(process.pid)
                discovered = [root] + root.children(recursive=True)
                for discovered_proc in discovered:
                    known_processes.setdefault(discovered_proc.pid, discovered_proc)
                processes = [known_processes[proc.pid] for proc in discovered]
            except psutil.NoSuchProcess:
                break

            rss_bytes = 0
            vms_bytes = 0
            cpu_percent = 0.0
            threads = 0
            sampled_pids = set()

            for proc in processes:
                if proc.pid in sampled_pids:
                    continue
                sampled_pids.add(proc.pid)
                try:
                    memory = proc.memory_info()
                    rss_bytes += memory.rss
                    vms_bytes += memory.vms
                    cpu_percent += proc.cpu_percent(interval=None)
                    threads += proc.num_threads()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            rss_mb = rss_bytes / 1024**2
            vms_mb = vms_bytes / 1024**2
            rss_samples.append(rss_mb)
            vms_samples.append(vms_mb)
            elapsed = time.monotonic() - start
            elapsed_samples.append(elapsed)
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                round(elapsed, 3),
                round(rss_mb, 3),
                round(vms_mb, 3),
                round(cpu_percent, 1),
                threads,
            ])
            csv_file.flush()

            if process.poll() is not None:
                break
            stop_event.wait(interval)

    if rss_samples:
        print(
            "Memory monitor: {} samples, peak RSS {:.2f} MB, "
            "average RSS {:.2f} MB, peak VMS {:.2f} MB -> {}".format(
                len(rss_samples), max(rss_samples),
                sum(rss_samples) / len(rss_samples), max(vms_samples),
                output_csv,
            )
        )
    else:
        print("Memory monitor: no samples collected -> {}".format(output_csv))


    if rss_samples:
        plot_output = os.path.splitext(output_csv)[0] + ".png"
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10,5))
            plt.plot(elapsed_samples, rss_samples, linewidth=2, label="RSS")
            plt.plot(elapsed_samples, vms_samples, linewidth=2, label="VMS")

            plt.xlabel("Time (s)")
            plt.ylabel("Memory (MB)")
            plt.title("NS-3 Memory Usage")
            plt.grid(True)
            plt.legend()

            plt.tight_layout()
            plt.savefig(plot_output, dpi=300)

            print(f"CSV saved to  : {output_csv}")
            print(f"Plot saved to : {plot_output}")
        except Exception as error:
            print(
                "Unable to create memory plot {}: {}".format(
                    plot_output, error
                )
            )


def run_ns3_with_timeout(base_cmd, output_file, timeout_seconds=180,
                         initial_seed=1, memory_output_file=None,
                         memory_interval=30):
    seed = initial_seed or int(time.time())
    attempt = 1

    while attempt <= 1:
        print(f"Attempt {attempt}: Running simulation with seed {seed}")
        full_cmd = f"{base_cmd} --seed={seed} \' > {output_file}"

        try:
            process = subprocess.Popen(full_cmd, shell=True)
            proc = psutil.Process(process.pid)
            memory_stop_event = threading.Event()
            monitor_csv = memory_output_file or "{}_memory_usage.csv".format(
                os.path.splitext(output_file)[0]
            )
            memory_thread = threading.Thread(
                target=monitor_simulation_memory,
                args=(process, monitor_csv, memory_stop_event, memory_interval),
                name="ns3-memory-monitor-{}".format(process.pid),
            )
            memory_thread.start()

            try:
                process.wait(timeout=timeout_seconds)
                if process.returncode == 0:
                    print(f"Simulation completed successfully with seed {seed}")
                else:
                    print(
                        f"Simulation with seed {seed} exited with status "
                        f"{process.returncode}"
                    )
                break
            except subprocess.TimeoutExpired:
                print(f"Timeout expired for seed {seed}. Killing process tree...")
                for child in proc.children(recursive=True):
                    child.kill()
                proc.kill()
                time.sleep(1)
            finally:
                memory_stop_event.set()
                memory_thread.join()

        except Exception as e:
            print(f"Error running simulation with seed {seed}: {e}")

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
            exp_switchREDQueueDiscMaxSize = "{}KB".format(round(float(expConfig.switchREDQueueDiscMaxSize.split('K')[0]) * rate, 1))
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
                            '--switchREDQueueDiscMaxSize={} '.format(exp_switchREDQueueDiscMaxSize) +
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
                    memory_output_file = (
                        '{}/scratch/ECNMC/Results/results_forward/{}/memory_usage.csv'
                        .format(get_ns3_path(), i + 1)
                    )
                    run_ns3_with_timeout(
                        cmd, output_file, timeout_seconds=120,
                        initial_seed=i + 1, memory_output_file=memory_output_file)
                    os.system('mkdir -p {}/scratch/Results_forward/{}/{}/{}/{}'.format(get_ns3_path(), traffic, rate, load, i))
                    os.system('mv {}/scratch/ECNMC/Results/results_forward/{}/*.csv {}/scratch/Results_forward/{}/{}/{}/{}'.format(get_ns3_path(), i + 1, get_ns3_path(), traffic, rate, load, i))
                    os.system('mv {}/scratch/ECNMC/Results/results_forward/{}/*.png {}/scratch/Results_forward/{}/{}/{}/{}'.format(get_ns3_path(), i + 1, get_ns3_path(), traffic, rate, load, i))
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
                        exp_switchREDQueueDiscMaxSize = "{}KB".format(round(float(expConfig.switchREDQueueDiscMaxSize.split('K')[0]) * CRate, 1))
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
                                    '--errorRate={} '.format(errorRate) +
                                    '--trafficStartTime={} '.format(expConfig.trafficStartTime) +
                                    '--trafficStopTime={} '.format(expConfig.trafficStopTime) +
                                    '--steadyStartTime={} '.format(expConfig.steadyStart) +
                                    '--steadyStopTime={} '.format(expConfig.steadyEnd) +
                                    '--switchREDQueueDiscMaxSize={} '.format(exp_switchREDQueueDiscMaxSize) +
                                    '--switchSrcREDQueueDiscMaxSize={} '.format(expConfig.switchSrcREDQueueDiscMaxSize) +
                                    '--switchTXMaxSize={} '.format(expConfig.switchTXMaxSize) +
                                    '--minTh={} '.format(expConfig.MinTh) +
                                    '--maxTh={} '.format(expConfig.MaxTh) +
                                    '--dirName=' + 'reverse_{} '.format(type_name) +
                                    '--traffic={} '.format(traffic) +
                                    '--Nagle={} '.format(expConfig.NagleIsEnabled) +
                                    '--ActiveProbe={} '.format(expConfig.ActiveProbeIsEnabled) +
                                    '--PassiveProbe={} '.format(expConfig.PassiveProbeIsEnabled) +
                                    '--differentiationDelay={}ns '.format(DiffRate) +
                                    '--isDifferentating={} '.format(expConfig.isDifferentating) +
                                    '--silentPacketDrop={} '.format(expConfig.silentPacketDrop) +
                                    '--probeInterval={} '.format(expConfig.probeInterval) +
                                    '--incastMessageSize={} '.format(expConfig.incastMessageSize) +
                                    '--incastFactor={} '.format(expConfig.incastFactor) +
                                    '--incastperiod={} '.format(expConfig.incastperiod)
                                )
                            output_file = '{}/scratch/ECNMC/Results/results_reverse_{}/result_{}.txt'.format(get_ns3_path(), type_name, i)
                            memory_output_file = (
                                '{}/scratch/ECNMC/Results/results_reverse_{}/{}/memory_usage.csv'
                                .format(get_ns3_path(), type_name, i + 1)
                            )
                            run_ns3_with_timeout(
                                cmd, output_file, timeout_seconds=72000,
                                initial_seed=i + 1, memory_output_file=memory_output_file)
                            os.system('mkdir -p {}/scratch/Results_reverse_{}/{}/{}/{}/D_{}/f_{}/{}'.format(get_ns3_path(), type_name, traffic, CRate, load, DiffRate, errorRate, i))
                            os.system('mv {}/scratch/ECNMC/Results/results_reverse_{}/{}/*.csv {}/scratch/Results_reverse_{}/{}/{}/{}/D_{}/f_{}/{}'.format(get_ns3_path(), type_name, i + 1, get_ns3_path(), type_name, traffic, CRate, load, DiffRate, errorRate, i))
                            os.system('mv {}/scratch/ECNMC/Results/results_reverse_{}/{}/*.png {}/scratch/Results_reverse_{}/{}/{}/{}/D_{}/f_{}/{}'.format(get_ns3_path(), type_name, i + 1, get_ns3_path(), type_name, traffic, CRate, load, DiffRate, errorRate, i))
                            os.system('mkdir -p {}/scratch/ECNMC/Results/results_reverse_{}/{}/{}/{}/D_{}/f_{}'.format(get_ns3_path(), type_name, traffic, CRate, load, DiffRate, errorRate))
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
                '--sampleRate={} '.format(exp_sampleRate) +
                '--experiment={} '.format(i + 1) +
                '--errorRate={} '.format(exp_errorRate) +
                '--trafficStartTime={} '.format(i * float(expConfig.duration)) +
                '--trafficStopTime={} '.format((i + 1) * float(expConfig.duration)) +
                '--steadyStartTime={} '.format(expConfig.steadyStart) +
                '--steadyStopTime={} '.format(expConfig.steadyEnd) +
                '--dirName=' + 'params '
            )
            output_file = (
                '{}/scratch/ECNMC/Results/results_params/result_{}.txt'
                .format(get_ns3_path(), i)
            )
            memory_output_file = (
                '{}/scratch/ECNMC/Results/results_params/{}/memory_usage.csv'
                .format(get_ns3_path(), i + 1)
            )
            run_ns3_with_timeout(
                cmd, output_file, timeout_seconds=72000,
                initial_seed=i + 1, memory_output_file=memory_output_file)
    
            os.system('mkdir -p {}/scratch/Results_params/{}/{}'.format(get_ns3_path(), rate, i))
            os.system('mv {}/scratch/ECNMC/Results/results_params/{}/*.csv {}/scratch/Results_params/{}/{}'.format(get_ns3_path(), i + 1, get_ns3_path(), rate, i))
            os.system('mv {}/scratch/ECNMC/Results/results_params/{}/*.png {}/scratch/Results_params/{}/{}'.format(get_ns3_path(), i + 1, get_ns3_path(), rate, i))
            os.system('mkdir -p {}/scratch/ECNMC/Results/results_params/{}'.format(get_ns3_path(), rate))
            print('\tExperiment {} done'.format(i))
        print('Rate {} done'.format(rate))

def run_burst_experiment(exp, rate):
    expConfig = ExperimentConfig()
    expConfig.read_config_file('Parameters.config')
    os.system('mkdir -p {}/scratch/ECNMC/Results/results_burst/'.format(get_ns3_path()))
    exp_tor_to_agg_link_rate = "{}Mbps".format(round(float(expConfig.tor_to_agg_link_rate.split('M')[0]) * rate, 1))
    for i in exp:
        os.system('mkdir -p {}/scratch/ECNMC/Results/results_burst/{}'.format(get_ns3_path(), i + 1))
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
            '--errorRate={} '.format(expConfig.errorRate) +
            '--trafficStartTime={} '.format(i * float(expConfig.duration)) +
            '--trafficStopTime={} '.format((i + 1) * float(expConfig.duration)) +
            '--steadyStartTime={} '.format(expConfig.steadyStart) +
            '--steadyStopTime={} '.format(expConfig.steadyEnd) +
            '--dirName=' + 'burst '
        )
        output_file = (
            '{}/scratch/ECNMC/Results/results_burst/result_{}.txt'
            .format(get_ns3_path(), i)
        )
        memory_output_file = (
            '{}/scratch/ECNMC/Results/results_burst/{}/memory_usage.csv'
            .format(get_ns3_path(), i + 1)
        )
        run_ns3_with_timeout(
            cmd, output_file, timeout_seconds=72000,
            initial_seed=i + 1, memory_output_file=memory_output_file)

        os.system('mkdir -p {}/scratch/Results_burst/{}/{}'.format(get_ns3_path(), rate, 0))
        os.system('mv {}/scratch/ECNMC/Results/results_burst/{}/*.csv {}/scratch/Results_burst/{}/{}'.format(get_ns3_path(), i + 1, get_ns3_path(), rate, 0))
        os.system('mv {}/scratch/ECNMC/Results/results_burst/{}/*.png {}/scratch/Results_burst/{}/{}'.format(get_ns3_path(), i + 1, get_ns3_path(), rate, 0))
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
        numOfThs = 2
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
        numOfThs = 2
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

