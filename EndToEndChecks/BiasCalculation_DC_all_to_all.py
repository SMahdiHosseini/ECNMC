"""Offline bias calculation for every observed datacenter flow and ECMP path.

The optimized simulation writes one ``RiHj_ALL_EndToEnd_packets.csv`` file per
source server.  This script discovers the destinations and path IDs in those
files, maps them to the queues traversed by the path, repeats the stochastic
bias calculation, and averages the repetitions.

The JSON layout keeps the metric names used by ``BiasCalculation_DC.py`` and
adds two levels below every metric::

    metric -> flow name -> path id -> [average for experiment 0, ...]

Inter-rack paths use ``TiAx, AxTk, TkHl``.  Same-rack paths do not visit an
aggregation switch and therefore use only ``TiHl``.
"""

from __future__ import annotations

import argparse
import configparser
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable
import random
import numpy as np
import pandas as pd

from Utils import (
    PacketCDF,
    calculate_offline_delay_bias_DC,
    compute_average_packet_size,
    convert_to_float,
    find_queue_size_at_time,
)


NS3_PATH = Path(__file__).resolve().parents[3]
SOURCE_FILE_RE = re.compile(r"^(R(?P<rack>\d+)H(?P<host>\d+))_ALL_EndToEnd_packets\.csv$")
HOST_RE = re.compile(r"^R(?P<rack>\d+)H(?P<host>\d+)$")
REQUIRED_E2E_COLUMNS = {"SourceIp", "DestinationIp", "Path", "PayloadSize"}


def host_coordinates(host_name: str) -> tuple[int, int]:
    match = HOST_RE.fullmatch(host_name)
    if match is None:
        raise ValueError(f"Invalid host name: {host_name}")
    return int(match.group("rack")), int(match.group("host"))


def queues_for_flow(source: str, destination: str, path_id: int) -> list[str]:
    """Return queues in traversal order for one flow/path."""
    source_rack, _ = host_coordinates(source)
    destination_rack, destination_host = host_coordinates(destination)
    if source_rack == destination_rack:
        return [f"T{source_rack}H{destination_host}"]
    return [
        f"T{source_rack}A{path_id}",
        f"A{path_id}T{destination_rack}",
        f"T{destination_rack}H{destination_host}",
    ]


def _read_source_ip(file_path: Path) -> str | None:
    frame = pd.read_csv(file_path, usecols=["SourceIp"], nrows=1)
    if frame.empty:
        return None
    return str(frame.iloc[0]["SourceIp"])


def build_ip_to_host(experiment_dir: Path) -> dict[str, str]:
    """Build the address mapping from monitor files rather than assuming IPs."""
    mapping: dict[str, str] = {}
    for file_path in sorted(experiment_dir.glob("*_ALL_EndToEnd_packets.csv")):
        match = SOURCE_FILE_RE.fullmatch(file_path.name)
        if match is None:
            continue
        source_ip = _read_source_ip(file_path)
        if source_ip is not None:
            mapping[source_ip] = match.group(1)
    return mapping


def discover_flow_paths(
    experiment_dir: Path,
    max_flows: int | None = None,
    randomize: bool = False,
) -> dict[tuple[str, str, int], int]:
    """Discover observed flow/path pairs and their packet counts.

    ``max_flows`` limits distinct source/destination pairs, not individual
    paths. All observed paths belonging to a selected flow are retained.
    """
    ip_to_host = build_ip_to_host(experiment_dir)
    discovered: dict[tuple[str, str, int], int] = {}

    for file_path in sorted(experiment_dir.glob("*_ALL_EndToEnd_packets.csv")):
        match = SOURCE_FILE_RE.fullmatch(file_path.name)
        if match is None:
            continue
        source = match.group(1)
        frame = pd.read_csv(file_path, usecols=list(REQUIRED_E2E_COLUMNS))
        missing = REQUIRED_E2E_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"{file_path} is missing columns: {sorted(missing)}")

        grouped = frame.groupby(["DestinationIp", "Path"], sort=True).size()
        for (destination_ip, path_id), packet_count in grouped.items():
            destination = ip_to_host.get(str(destination_ip))
            if destination is None:
                raise ValueError(
                    f"Destination IP {destination_ip} in {file_path} has no matching "
                    "RiHj_ALL_EndToEnd_packets.csv source file"
                )
            discovered[(source, destination, int(path_id))] = int(packet_count)

        if max_flows is not None:
            flow_pairs = sorted({(src, dst) for src, dst, _ in discovered})
            if len(flow_pairs) >= max_flows:
                if randomize:
                    selected = set(random.sample(flow_pairs, max_flows))
                else:
                    selected = set(flow_pairs[:max_flows])
                return {
                    key: count
                    for key, count in discovered.items()
                    if key[:2] in selected
                }

    return discovered


def _mean_value(values: Iterable[Any]) -> Any:
    """Average scalar or fixed-shape list metrics and return JSON-native data."""
    values = list(values)
    if not values:
        raise ValueError("Cannot average an empty collection")
    first = values[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        array = np.asarray(values, dtype=float)
        if np.isnan(array).all():
            return np.full(array.shape[1:], np.nan).tolist()
        return np.nanmean(array, axis=0).tolist()
    if isinstance(first, (bool, int, float, np.number)):
        array = np.asarray(values, dtype=float)
        if np.isnan(array).all():
            return float("nan")
        return float(np.nanmean(array))
    if all(value == first for value in values):
        return first
    raise TypeError(f"Metric has non-numeric, non-constant values: {values[:3]}")


def average_repetitions(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("No repetition results to average")
    keys = set(results[0])
    if any(set(result) != keys for result in results[1:]):
        raise ValueError("Bias repetitions returned different metric sets")
    return {key: _mean_value(result[key] for result in results) for key in sorted(keys)}


class DirectQueueCalculator:
    """One-queue equivalent of the legacy three-queue calculation.

    The legacy utility assumes exactly three queues while computing pair/triple
    covariance.  Same-rack traffic has one direct ToR-to-host queue, so this
    small adapter produces the same metric family with zero cross-queue terms.
    """

    def __init__(self, experiment_dir: Path, traffic: str):
        self.experiment_dir = experiment_dir
        self.packet_cdf = PacketCDF()
        cdf_path = NS3_PATH / "scratch" / "ECNMC" / "DCWorkloads" / f"packet_size_cdf_{traffic}.csv"
        self.packet_cdf.load_cdf_data(str(cdf_path))
        self._queue_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _sample_sizes(self, queue_name: str, times: np.ndarray, link_rate: float) -> np.ndarray:
        if queue_name not in self._queue_cache:
            file_path = self.experiment_dir / f"{queue_name}_PoissonSampler_queueSize.csv"
            frame = pd.read_csv(file_path, usecols=["Time", "TotalQueueSize"])
            frame = frame.sort_values(["Time", "TotalQueueSize"], ascending=[True, False])
            self._queue_cache[queue_name] = (
                frame["Time"].to_numpy(dtype=float),
                frame["TotalQueueSize"].to_numpy(dtype=float),
            )
        trace_times, trace_sizes = self._queue_cache[queue_name]
        return find_queue_size_at_time(trace_times, trace_sizes, times, link_rate)

    def calculate(
        self,
        queue_name: str,
        steady_start: float,
        steady_end: float,
        link_rate: float,
        link_delay: float,
        queue_capacity: float,
        threshold_fraction: float,
        interval: float,
    ) -> dict[str, Any]:
        count = int((steady_end - steady_start) // interval)
        times = np.asarray(
            np.cumsum(np.random.exponential(interval, size=count)) + steady_start,
            dtype=np.int64,
        )
        if not len(times):
            raise ValueError("The steady interval is too short to produce samples")

        sizes = self._sample_sizes(queue_name, times.astype(float), link_rate)
        valid = np.isfinite(sizes)
        sizes = sizes[valid]
        if not len(sizes):
            raise ValueError(f"No valid queue samples for {queue_name}")

        delays = sizes * 8 / link_rate
        nonmarking = (sizes < queue_capacity * threshold_fraction).astype(float)
        success = np.asarray(
            [1.0 - self.packet_cdf.calculate_probability_greater_than(queue_capacity - size) for size in sizes]
        )
        prefix = queue_name
        result: dict[str, Any] = {}

        # With a single queue, path-observer and independent Poisson samples are
        # the same samples (the legacy distinction starts at the next queue).
        for tag in ("e2e", "poisson"):
            result[prefix + tag + "_samples_queue_delay_mean"] = np.mean(delays)
            result[prefix + tag + "_samples_queue_delay_std"] = np.std(delays)
            result[prefix + tag + "_samples_queue_delay_count"] = len(delays)
            result[prefix + tag + "_samples_queue_success_prob_mean"] = np.mean(success)
            result[prefix + tag + "_samples_queue_success_prob_std"] = np.std(success)
            result[prefix + tag + "_samples_queue_nonmarking_prob_mean"] = np.mean(nonmarking)
            result[prefix + tag + "_samples_queue_nonmarking_prob_std"] = np.std(nonmarking)

        delay_error = 2 * 1.96 * np.std(delays) / math.sqrt(len(delays))
        success_error = 2 * 1.96 * np.std(success) / math.sqrt(len(success))
        nonmarking_error = 2 * 1.96 * np.std(nonmarking) / math.sqrt(len(nonmarking))
        result[prefix + "poisson_prob_non_empty"] = np.mean(sizes > 0)
        result[prefix + "error_bound"] = delay_error
        result[prefix + "success_prob_error_bound"] = success_error
        result[prefix + "nonmarking_prob_error_bound"] = nonmarking_error
        result[prefix + "e2e_vs_poisson_consistent"] = 1
        result[prefix + "e2e_vs_poisson_consistent_with_bias"] = 1
        result[prefix + "e2e_vs_poisson_consistent_success_prob"] = 1
        result[prefix + "e2e_vs_poisson_consistent_nonmarking_prob"] = 1
        result[prefix + "split_ratio"] = 1.0
        result[prefix + "bias"] = 0.0
        average_packet_size = self.packet_cdf.compute_average_packet_size_from_cdf()
        result[prefix + "NPkts"] = np.mean(delays) * link_rate / (average_packet_size * 8)
        result[prefix + "NBytes"] = result[prefix + "NPkts"] * average_packet_size

        result["sum_poisson_samples_queue_delay_mean"] = np.mean(delays)
        result["sum_poisson_samples_queue_success_prob_mean"] = np.mean(success)
        result["sum_poisson_samples_queue_success_prob_pair_covariance"] = 0.0
        result["sum_poisson_samples_queue_success_prob_triple_covariance"] = 0.0
        result["sum_poisson_samples_queue_nonmarking_prob_mean"] = np.mean(nonmarking)
        result["sum_poisson_samples_queue_nonmarking_prob_pair_covariance"] = 0.0
        result["sum_poisson_samples_queue_nonmarking_prob_triple_covariance"] = 0.0
        result["e2e_poisson_samples_queue_delay_mean"] = np.mean(delays)
        result["e2e_poisson_samples_queue_delay_std"] = np.std(delays)
        result["e2e_poisson_samples_queue_success_prob_mean"] = np.mean(success)
        result["e2e_poisson_samples_queue_success_prob_std"] = np.std(success)
        result["e2e_poisson_samples_queue_nonmarking_prob_mean"] = np.mean(nonmarking)
        result["e2e_poisson_samples_queue_nonmarking_prob_std"] = np.std(nonmarking)
        result["e2e_vs_sum_error_bound"] = delay_error
        result["e2e_vs_sum_error_success_prob_bound"] = [success_error, -success_error]
        result["e2e_vs_sum_error_nonmarking_prob_bound"] = [nonmarking_error, -nonmarking_error]
        result["e2e_vs_sum_consistent"] = 1
        result["e2e_vs_sum_consistent_with_bias"] = 1
        result["e2e_vs_sum_consistent_success_prob"] = 1
        result["e2e_vs_sum_consistent_nonmarking_prob"] = 1
        return result


def store_metrics(
    merged: dict[str, dict[str, dict[str, list[Any]]]],
    flow_name: str,
    path_id: int,
    metrics: dict[str, Any],
) -> None:
    path_key = str(path_id)
    for metric, value in metrics.items():
        merged.setdefault(metric, {}).setdefault(flow_name, {}).setdefault(path_key, []).append(value)


def analyze_experiment(
    experiment_dir: Path,
    results_folder: str,
    rate: float,
    load: float,
    experiment: int,
    traffic: str,
    steady_start: float,
    steady_end: float,
    link_rates: list[float],
    link_delays: list[float],
    queue_capacities: list[float],
    repetitions: int,
    sampling_factor: float | None,
    max_flows: int | None = None,
    randomize: bool = False,
) -> dict[tuple[str, int], dict[str, Any]]:
    discovered = discover_flow_paths(experiment_dir, max_flows=max_flows, randomize=randomize)
    if not discovered:
        raise FileNotFoundError(f"No aggregate end-to-end records found in {experiment_dir}")

    direct = DirectQueueCalculator(experiment_dir, traffic)
    average_packet_size = compute_average_packet_size(str(experiment_dir) + "/")
    route_cache: dict[tuple[str, ...], dict[str, Any]] = {}
    flow_results: dict[tuple[str, int], dict[str, Any]] = {}

    for source, destination, path_id in sorted(discovered):
        flow_name = source + destination
        queue_names = queues_for_flow(source, destination, path_id)
        # mahdi: code checked by this line
        cache_key = tuple(queue_names)
        if cache_key not in route_cache:
            runs: list[dict[str, Any]] = []
            for _ in range(repetitions):
                if len(queue_names) == 1:
                    runs.append(
                        direct.calculate(
                            queue_names[0], steady_start, steady_end,
                            link_rates[3], link_delays[3], queue_capacities[3],
                            0.15, 10000,
                        )
                    )
                else:
                    runs.append(
                        calculate_offline_delay_bias_DC(
                            str(NS3_PATH), rate, experiment, results_folder,
                            steady_start, steady_end,
                            linkRates=link_rates,
                            linkDelays=link_delays,
                            swtichDstREDQueueDiscMaxSize=queue_capacities,
                            tsh=0.15,
                            load=load,
                            queue_names=queue_names,
                            flow_names=[flow_name],
                            e2e_intervals=10000,
                            sampling_factor=sampling_factor,
                            average_packet_size=average_packet_size,
                        )
                    )
            route_cache[cache_key] = average_repetitions(runs)
        flow_results[(flow_name, path_id)] = route_cache[cache_key]

    return flow_results


def _parse_csv_filter(value: str | None, cast: Any) -> list[Any] | None:
    if value is None:
        return None
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, help="Result suffix, e.g. forward")
    parser.add_argument("--config", type=Path, help="Override Parameters.config path")
    parser.add_argument("--traffic", help="Comma-separated traffic names")
    parser.add_argument("--rate", help="Comma-separated service-rate scales")
    parser.add_argument("--load", help="Comma-separated offered loads")
    parser.add_argument("--experiments", help="Comma-separated zero-based experiment IDs")
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument(
        "--max-flows", type=int,
        help="Process at most this many source/destination flows, keeping all paths",
    )
    parser.add_argument("--randomize", action="store_true", help="Randomly select flows if --max-flows is set")
    parser.add_argument("--sampling-factor", type=float)
    parser.add_argument("--output-name", default="delay_minimum_bias_e2e_vs_switch_poisson_all_to_all.json")
    args = parser.parse_args()

    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    if args.max_flows is not None and args.max_flows < 1:
        parser.error("--max-flows must be at least 1")

    result_config_dir = NS3_PATH / "scratch" / "ECNMC" / "Results" / f"results_{args.dir}"
    config_path = args.config or (result_config_dir / "Parameters.config")
    config = configparser.ConfigParser()
    if not config.read(config_path):
        raise FileNotFoundError(f"Could not read configuration: {config_path}")

    steady_start = convert_to_float(config.get("Settings", "steadyStart")) * 1e9
    steady_end = convert_to_float(config.get("Settings", "steadyEnd")) * 1e9
    rates = _parse_csv_filter(args.rate, float) or [
        float(item) for item in config.get("Settings", "serviceRateScales").split(",")
    ]
    loads = _parse_csv_filter(args.load, float) or [
        float(item) for item in config.get("Settings", "load").split(",")
    ]
    traffics = _parse_csv_filter(args.traffic, str) or [
        item.strip() for item in config.get("Settings", "traffic").split(",")
    ]
    experiment_ids = _parse_csv_filter(args.experiments, int)
    if experiment_ids is None:
        experiment_ids = list(range(config.getint("Settings", "experiments")))

    host_to_tor_rate = convert_to_float(config.get("Settings", "hostToTorLinkRate")) * 1e-3
    base_tor_to_agg_rate = convert_to_float(config.get("Settings", "torToAggLinkRate")) * 1e-3
    link_delay = convert_to_float(config.get("Settings", "hostToTorLinkDelay")) * 1e6
    host_queue_capacity = convert_to_float(config.get("Settings", "switchSrcREDQueueDiscMaxSize"))
    base_switch_capacity = convert_to_float(config.get("DCSim", "switchREDQueueDiscMaxSize"))

    results_folder = f"Results_{args.dir}"
    for traffic in traffics:
        for rate in rates:
            link_rates = [host_to_tor_rate, base_tor_to_agg_rate * rate,
                          base_tor_to_agg_rate * rate, host_to_tor_rate]
            link_delays = [link_delay] * 4
            queue_capacities = [host_queue_capacity, base_switch_capacity * rate,
                                base_switch_capacity * rate, host_queue_capacity]
            for load in loads:
                merged: dict[str, dict[str, dict[str, list[Any]]]] = {}
                completed_experiments: list[int] = []
                for experiment in experiment_ids:
                    experiment_dir = NS3_PATH / "scratch" / results_folder / traffic / str(rate) / str(load) / str(experiment)
                    if not experiment_dir.is_dir():
                        print(f"Skipping missing experiment directory: {experiment_dir}")
                        continue
                    print(f"Analyzing {traffic}, rate={rate}, load={load}, experiment={experiment}")
                    flow_results = analyze_experiment(
                        experiment_dir, results_folder + "/" + traffic,
                        rate, load, experiment, traffic, steady_start, steady_end,
                        link_rates, link_delays, queue_capacities,
                        args.repetitions, args.sampling_factor, args.max_flows, args.randomize
                    )
                    for (flow_name, path_id), metrics in flow_results.items():
                        store_metrics(merged, flow_name, path_id, metrics)
                    completed_experiments.append(experiment)

                merged["experiment"] = completed_experiments  # type: ignore[assignment]
                output_dir = result_config_dir / traffic / str(rate) / str(load)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / args.output_name
                with output_path.open("w") as output_file:
                    json.dump(merged, output_file, indent=4, allow_nan=True)
                print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
