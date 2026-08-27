"""Actual end-to-end versus reconstructed-switch consistency checks.

For every selected flow and path, this script reads the actual packet outcomes
from ``RiHj_ALL_EndToEnd_packets.csv`` and compares them with switch metrics
estimated at random Poisson times. Switch queue states are reconstructed from
the full ``*_PoissonSampler_queueSize.csv`` event traces; the pre-sampled
``*_PoissonSampler_events.csv`` files are never used.

The output layout is compatible with ``BiasCalculation_DC_all_to_all.py``::

    metric -> flow name -> path id -> [value for experiment 0, ...]
"""

from __future__ import annotations

import argparse
import configparser
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from BiasCalculation_DC_all_to_all import (
    FLOW_RE,
    NS3_PATH,
    average_repetitions,
    host_coordinates,
    infer_topology,
    load_flow_inventory,
    output_filename,
    queues_for_flow,
    select_flow_paths,
    store_metrics,
)
from Utils import (
    calculate_offline_delay_bias_DC,
    compute_average_packet_size,
    convert_to_float,
)


DEFAULT_OUTPUT_NAME = "consistency_e2e_vs_switch_poisson_all_to_all.json"
CONFIDENCE_Z = 1.96
E2E_COLUMNS = [
    "Id",
    "DestinationIp",
    "Path",
    "SentTime",
    "IsReceived",
    "ReceiveTime",
    "transmissionDelay",
    "PayloadSize",
    "ECN",
]


class EndToEndTraceReader:
    """Keep only the currently used source-server packet trace in memory."""

    def __init__(self, experiment_dir: Path):
        self.experiment_dir = experiment_dir
        self._source: str | None = None
        self._frame: pd.DataFrame | None = None

    def flow_path(
        self,
        source: str,
        destination_ip: str,
        path_id: int,
        steady_start: float,
        steady_end: float,
    ) -> pd.DataFrame:
        if source != self._source:
            trace_path = self.experiment_dir / f"{source}_ALL_EndToEnd_packets.csv"
            self._frame = pd.read_csv(trace_path, usecols=E2E_COLUMNS)
            self._source = source

        assert self._frame is not None
        frame = self._frame
        destination = frame["DestinationIp"].astype(str)
        selected = frame[
            (destination == destination_ip)
            & (frame["Path"] == path_id)
            & (frame["SentTime"] >= steady_start)
            & (frame["SentTime"] <= steady_end)
        ].copy()
        if selected.empty:
            raise ValueError(
                f"No packets for {source} -> {destination_ip}, path {path_id}, "
                f"in [{steady_start}, {steady_end}]"
            )
        # Match the legacy postprocessor: if the same packet ID has both a
        # failed and a received record, keep the received outcome.
        selected = selected.sort_values("IsReceived", ascending=False)
        selected = selected.drop_duplicates(subset="Id", keep="first")
        return selected


def _sample_statistics(values: np.ndarray) -> tuple[float, float, int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), 0
    return float(np.mean(values)), float(np.std(values)), int(values.size)


def actual_flow_path_statistics(
    frame: pd.DataFrame,
    direct_path: bool = False,
    link_rates: list[float] | None = None,
    link_delays: list[float] | None = None,
) -> dict[str, Any]:
    """Compute actual delay, delivery, and ECN outcomes for one flow/path."""
    received = frame["IsReceived"].to_numpy(dtype=float) == 1
    elapsed = (
        frame.loc[received, "ReceiveTime"].to_numpy(dtype=float)
        - frame.loc[received, "SentTime"].to_numpy(dtype=float)
    )
    if direct_path:
        if link_rates is None or link_delays is None:
            raise ValueError("link rates and delays are required for a direct path")
        payload = frame.loc[received, "PayloadSize"].to_numpy(dtype=float)
        transmission_delay = (
            link_delays[0] + link_delays[3]
            + payload * 8 / link_rates[0]
            + payload * 8 / link_rates[3]
        )
    else:
        transmission_delay = frame.loc[
            received, "transmissionDelay"
        ].to_numpy(dtype=float)
    # Match the legacy PostProcessing calculation while using the correct
    # two-link baseline for same-rack traffic.
    delay = np.abs(elapsed - transmission_delay)
    success = frame["IsReceived"].to_numpy(dtype=float)
    nonmarking = 1.0 - frame["ECN"].to_numpy(dtype=float)

    delay_mean, delay_std, delay_count = _sample_statistics(delay)
    success_mean, success_std, success_count = _sample_statistics(success)
    nonmarking_mean, nonmarking_std, nonmarking_count = _sample_statistics(nonmarking)
    return {
        "actual_e2e_delay_mean": delay_mean,
        "actual_e2e_delay_std": delay_std,
        "actual_e2e_delay_count": delay_count,
        "actual_e2e_success_probability_mean": success_mean,
        "actual_e2e_success_probability_std": success_std,
        "actual_e2e_success_probability_count": success_count,
        "actual_e2e_nonmarking_probability_mean": nonmarking_mean,
        "actual_e2e_nonmarking_probability_std": nonmarking_std,
        "actual_e2e_nonmarking_probability_count": nonmarking_count,
        "actual_e2e_packet_count": int(len(frame)),
    }


def _standard_error(std: float, count: float) -> float:
    if count <= 0 or not np.isfinite(std):
        return float("nan")
    return float(std / np.sqrt(count))


def _product_standard_error(
    metrics: dict[str, Any], queue_names: list[str], metric_suffix: str
) -> float:
    """Delta-method standard error for a product of queue probabilities."""
    means = np.asarray(
        [metrics[name + metric_suffix + "_mean"] for name in queue_names],
        dtype=float,
    )
    standard_errors = np.asarray(
        [
            _standard_error(
                metrics[name + metric_suffix + "_std"],
                metrics[name + "poisson_samples_queue_delay_count"],
            )
            for name in queue_names
        ],
        dtype=float,
    )
    derivatives = np.asarray(
        [np.prod(np.delete(means, index)) for index in range(len(means))],
        dtype=float,
    )
    return float(np.sqrt(np.sum(np.square(derivatives * standard_errors))))


def add_actual_consistency_metrics(
    switch_metrics: dict[str, Any],
    actual_metrics: dict[str, Any],
    queue_names: list[str],
    confidence_z: float = CONFIDENCE_Z,
) -> dict[str, Any]:
    """Add actual-E2E versus independently sampled switch consistency metrics."""
    result = dict(switch_metrics)
    result.update(actual_metrics)

    switch_delay_mean = float(result["sum_poisson_samples_queue_delay_mean"])
    switch_delay_se = float(np.sqrt(np.sum([
        np.square(_standard_error(
            result[name + "poisson_samples_queue_delay_std"],
            result[name + "poisson_samples_queue_delay_count"],
        ))
        for name in queue_names
    ])))
    actual_delay_se = _standard_error(
        result["actual_e2e_delay_std"], result["actual_e2e_delay_count"]
    )
    delay_bound = confidence_z * (switch_delay_se + actual_delay_se)
    delay_difference = result["actual_e2e_delay_mean"] - switch_delay_mean
    biased_delay_difference = delay_difference - result["total_estimated_bias"]

    result["switch_path_delay_mean"] = switch_delay_mean
    result["actual_e2e_minus_switch_delay"] = delay_difference
    result["actual_e2e_minus_biased_switch_delay"] = biased_delay_difference
    result["actual_e2e_vs_switch_delay_error_bound"] = delay_bound
    result["actual_e2e_vs_switch_delay_consistent"] = int(
        np.isfinite(delay_bound) and abs(delay_difference) <= delay_bound
    )
    result["actual_e2e_vs_switch_delay_consistent_with_bias"] = int(
        np.isfinite(delay_bound) and abs(biased_delay_difference) <= delay_bound
    )

    probability_specs = (
        (
            "success",
            "sum_poisson_samples_queue_success_prob_mean",
            "poisson_samples_queue_success_prob",
            "actual_e2e_success_probability",
        ),
        (
            "nonmarking",
            "sum_poisson_samples_queue_nonmarking_prob_mean",
            "poisson_samples_queue_nonmarking_prob",
            "actual_e2e_nonmarking_probability",
        ),
    )
    for label, switch_key, queue_suffix, actual_prefix in probability_specs:
        switch_mean = float(result[switch_key])
        switch_se = _product_standard_error(result, queue_names, queue_suffix)
        actual_se = _standard_error(
            result[actual_prefix + "_std"], result[actual_prefix + "_count"]
        )
        error_bound = confidence_z * (switch_se + actual_se)
        difference = result[actual_prefix + "_mean"] - switch_mean
        result[f"switch_path_{label}_probability_mean"] = switch_mean
        result[f"actual_e2e_minus_switch_{label}_probability"] = difference
        result[f"actual_e2e_vs_switch_{label}_probability_error_bound"] = error_bound
        result[f"actual_e2e_vs_switch_{label}_probability_consistent"] = int(
            np.isfinite(error_bound) and abs(difference) <= error_bound
        )

    return result


def analyze_experiment(
    experiment_dir: Path,
    results_folder: str,
    rate: float,
    load: float,
    experiment: int,
    steady_start: float,
    steady_end: float,
    link_rates: list[float],
    link_delays: list[float],
    queue_capacities: list[float],
    repetitions: int,
    sampling_factor: float | None,
    max_flows: int | None = None,
    randomize: bool = False,
    start_flow: int = 0,
    specific_flow: str | None = None,
) -> dict[tuple[str, int], dict[str, Any]]:
    ip_to_host, complete_inventory = load_flow_inventory(experiment_dir)
    discovered = select_flow_paths(
        complete_inventory, max_flows, randomize, start_flow, specific_flow
    )
    if not discovered:
        raise FileNotFoundError(f"No selected flow records found in {experiment_dir}")

    host_to_ip = {host: ip for ip, host in ip_to_host.items()}
    number_of_racks, hosts_per_rack = infer_topology(ip_to_host)
    alternative_routes = [number_of_racks - 1, hosts_per_rack]
    average_packet_size = compute_average_packet_size(str(experiment_dir) + "/")
    trace_reader = EndToEndTraceReader(experiment_dir)
    switch_runs_by_route: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    results: dict[tuple[str, int], dict[str, Any]] = {}

    for source, destination, path_id in sorted(discovered):
        print(f"Checking actual consistency for {source} -> {destination}, path {path_id}")
        flow_name = source + destination
        queue_names = queues_for_flow(source, destination, path_id)
        route_key = tuple(queue_names)

        if route_key not in switch_runs_by_route:
            route_capacities = (
                [0, queue_capacities[3]]
                if len(queue_names) == 1 else queue_capacities
            )
            switch_runs_by_route[route_key] = [
                calculate_offline_delay_bias_DC(
                    str(NS3_PATH), rate, experiment, results_folder,
                    steady_start, steady_end,
                    linkRates=link_rates,
                    linkDelays=link_delays,
                    swtichDstREDQueueDiscMaxSize=route_capacities,
                    tsh=0.15,
                    load=load,
                    queue_names=queue_names,
                    flow_names=[flow_name],
                    e2e_intervals=10000,
                    sampling_factor=sampling_factor,
                    average_packet_size=average_packet_size,
                    alternative_routes=alternative_routes,
                    source_rack=host_coordinates(source)[0],
                )
                for _ in range(repetitions)
            ]

        frame = trace_reader.flow_path(
            source, host_to_ip[destination], path_id, steady_start, steady_end
        )
        actual_metrics = actual_flow_path_statistics(
            frame,
            direct_path=len(queue_names) == 1,
            link_rates=link_rates,
            link_delays=link_delays,
        )
        repetition_results = [
            add_actual_consistency_metrics(run, actual_metrics, queue_names)
            for run in switch_runs_by_route[route_key]
        ]
        results[(flow_name, path_id)] = average_repetitions(repetition_results)

    return results


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
    parser.add_argument("--max-flows", type=int, help="Maximum number of complete flows")
    parser.add_argument("--start-flow", type=int, default=0, help="Flows to skip before selection")
    parser.add_argument("--randomize", action="store_true", help="Randomize selection after --start-flow")
    parser.add_argument("--flow", help="One exact flow, e.g. R0H0R2H4")
    parser.add_argument(
        "--save-per-switch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save queue-specific metrics (disable with --no-save-per-switch)",
    )
    parser.add_argument("--sampling-factor", type=float)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    args = parser.parse_args()

    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    if args.max_flows is not None and args.max_flows < 1:
        parser.error("--max-flows must be at least 1")
    if args.start_flow < 0:
        parser.error("--start-flow must be non-negative")
    if args.sampling_factor is not None and not 0 < args.sampling_factor <= 1:
        parser.error("--sampling-factor must be in (0, 1]")
    if args.flow is not None and (
        args.max_flows is not None or args.start_flow != 0 or args.randomize
    ):
        parser.error(
            "--flow cannot be combined with --max-flows, --start-flow, or --randomize"
        )
    if args.flow is not None and FLOW_RE.fullmatch(args.flow) is None:
        parser.error("--flow must have the form RaHbRcHd, for example R0H0R2H4")

    result_config_dir = (
        NS3_PATH / "scratch" / "ECNMC" / "Results" / f"results_{args.dir}"
    )
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

    host_to_tor_rate = convert_to_float(
        config.get("Settings", "hostToTorLinkRate")
    ) * 1e-3
    base_tor_to_agg_rate = convert_to_float(
        config.get("Settings", "torToAggLinkRate")
    ) * 1e-3
    link_delay = convert_to_float(
        config.get("Settings", "hostToTorLinkDelay")
    ) * 1e6
    host_queue_capacity = convert_to_float(
        config.get("Settings", "switchSrcREDQueueDiscMaxSize")
    )
    base_switch_capacity = convert_to_float(
        config.get("DCSim", "switchREDQueueDiscMaxSize")
    )

    results_folder = f"Results_{args.dir}"
    for traffic in traffics:
        for rate in rates:
            link_rates = [
                host_to_tor_rate,
                base_tor_to_agg_rate * rate,
                base_tor_to_agg_rate * rate,
                host_to_tor_rate,
            ]
            link_delays = [link_delay] * 4
            queue_capacities = [
                host_queue_capacity,
                base_switch_capacity * rate,
                base_switch_capacity * rate,
                host_queue_capacity,
            ]
            for load in loads:
                merged: dict[str, dict[str, dict[str, list[Any]]]] = {}
                completed_experiments: list[int] = []
                for experiment in experiment_ids:
                    experiment_dir = (
                        NS3_PATH / "scratch" / results_folder / traffic
                        / str(rate) / str(load) / str(experiment)
                    )
                    if not experiment_dir.is_dir():
                        print(f"Skipping missing experiment directory: {experiment_dir}")
                        continue
                    experiment_results = analyze_experiment(
                        experiment_dir,
                        results_folder + "/" + traffic,
                        rate,
                        load,
                        experiment,
                        steady_start,
                        steady_end,
                        link_rates,
                        link_delays,
                        queue_capacities,
                        args.repetitions,
                        args.sampling_factor,
                        max_flows=args.max_flows,
                        randomize=args.randomize,
                        start_flow=args.start_flow,
                        specific_flow=args.flow,
                    )
                    for (flow_name, path_id), metrics in experiment_results.items():
                        store_metrics(
                            merged,
                            flow_name,
                            path_id,
                            metrics,
                            save_per_switch=args.save_per_switch,
                        )
                    completed_experiments.append(experiment)

                merged["experiment"] = completed_experiments  # type: ignore[assignment]
                output_dir = result_config_dir / traffic / str(rate) / str(load)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / output_filename(args.output_name, args.flow)
                with output_path.open("w") as output_file:
                    json.dump(merged, output_file, indent=4, allow_nan=True)
                print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
