import configparser
import time
from PostProcessing import run_emd_vs_flows_experiment
from Utils import convert_to_float

ns3_path = "/media/experiments/ns-allinone-3.41/ns-3.41"
rate = 0.5
load = 0.8
experiment = 0
results_folder = 'Results_forward_DCW_DC24Servers_WOIncast/Google_AllRPC'
flow_name = 'R0H0R2H3'
confidenceValue = 1.96
num_runs = 10
num_poisson_observations = 9000
num_workers = 10
uniform_sample_strides = (10, 100)

config = configparser.ConfigParser()
config.read('{}/scratch/ECNMC/Results/results_forward_DCW_DC24Servers_WOIncast/Parameters.config'.format(ns3_path))
steadyStart = convert_to_float(config.get('Settings', 'steadyStart')) * 1e9
steadyEnd = convert_to_float(config.get('Settings', 'steadyEnd')) * 1e9

t0 = time.time()
results = run_emd_vs_flows_experiment(
    rate, steadyStart, steadyEnd, confidenceValue, results_folder, config,
    experiment=experiment, ns3_path=ns3_path, load=load, flow_name=flow_name,
    num_runs=num_runs, num_poisson_observations=num_poisson_observations,
    num_workers=num_workers, uniform_sample_strides=uniform_sample_strides,
)
elapsed = time.time() - t0

print("Elapsed seconds:", elapsed)
print("Total TCP flows on path 0:", results['total_flows'])
print("num_flows:", results['num_flows'])
print("emd_all_packets:", ["{:.1f}".format(v) for v in results['emd_all_packets']])
print("num_runs with valid EMD (sampled):", [len(v) for v in results['emd_sampled_packets_by_run']])
print("pass_rate_all_packets:", ["{:.2f}".format(v) for v in results['pass_rate_all_packets']])
print("pass_rate_sampled:", ["{:.2f}".format(v) for v in results['pass_rate_sampled']])
print("mean_diff_all_packets sizes:", [len(v) for v in results['mean_diff_all_packets_by_run']])
print("mean_diff_sampled sizes:", [len(v) for v in results['mean_diff_sampled_by_run']])
for stride in results['uniform_sample_strides']:
    print("uniform 1-in-{} num_runs with valid EMD:".format(stride), [len(v) for v in results['emd_uniform_packets_by_run'][stride]])
    print("uniform 1-in-{} pass_rate:".format(stride), ["{:.2f}".format(v) for v in results['pass_rate_uniform'][stride]])
