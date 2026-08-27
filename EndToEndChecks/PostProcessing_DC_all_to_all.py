"""Subsampled end-to-end versus reconstructed-switch consistency checks.

For every selected flow and path, this script reads the packet outcomes
from ``RiHj_ALL_EndToEnd_packets.csv`` and compares them with switch metrics
estimated at random Poisson times. Switch queue states are reconstructed from
the full ``*_PoissonSampler_queueSize.csv`` event traces; the pre-sampled
``*_PoissonSampler_events.csv`` files are never used.

Each switch estimate first determines the minimum required E2E sample count
for delay, success probability, and non-marking probability. The corresponding
packet-arrival process is then independently subsampled for each metric before
the PostProcessing consistency inequality is evaluated.

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
    calc_min_e2e_samples,
    calc_min_e2e_samples_prob,
    calculate_offline_delay_bias_DC,
    compute_average_packet_size,
    convert_to_float,
    find_delta_for_empty_prob,
    find_samples_path,
)


DEFAULT_OUTPUT_NAME = "consistency_e2e_vs_switch_poisson_all_to_all.json"
CONFIDENCE_Z = 1.96
MAX_CONSISTENCY_ERROR = 0.40
MINIMUM_CONSISTENCY_SAMPLES = 30
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


def _subsampled_delay_values(
    frame: pd.DataFrame,
    direct_path: bool,
    link_rates: list[float],
    link_delays: list[float],
) -> tuple[pd.DataFrame, np.ndarray]:
    received = frame["IsReceived"].to_numpy(dtype=float) == 1
    received_frame = frame.loc[received].copy()
    elapsed = (
        received_frame["ReceiveTime"].to_numpy(dtype=float)
        - received_frame["SentTime"].to_numpy(dtype=float)
    )
    if direct_path:
        payload = received_frame["PayloadSize"].to_numpy(dtype=float)
        transmission_delay = (
            link_delays[0] + link_delays[3]
            + payload * 8 / link_rates[0]
            + payload * 8 / link_rates[3]
        )
    else:
        transmission_delay = received_frame["transmissionDelay"].to_numpy(dtype=float)
    # Match the legacy PostProcessing calculation while using the correct
    # two-link baseline for same-rack traffic.
    return received_frame, np.abs(elapsed - transmission_delay)


def _subsample_metric(
    times: np.ndarray,
    values: np.ndarray,
    calculated_minimum: int | None,
) -> dict[str, Any]:
    """Return a Poisson-like subsample and whether it meets the requirement."""
    required_minimum = (
        max(MINIMUM_CONSISTENCY_SAMPLES, int(np.ceil(calculated_minimum)))
        if calculated_minimum is not None else None
    )
    order = np.argsort(times, kind="stable")
    sorted_times = np.asarray(times)[order]
    sorted_values = np.asarray(values, dtype=float)[order]
    available_count = int(len(sorted_times))
    if required_minimum is None or required_minimum > available_count:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "count": 0,
            "selected_value_count": 0,
            "available_count": available_count,
            "calculated_minimum": (
                int(calculated_minimum)
                if calculated_minimum is not None else float("nan")
            ),
            "required_minimum": (
                required_minimum if required_minimum is not None else float("nan")
            ),
            "subsampling_succeeded": 0,
            "minimum_requirement_met": 0,
        }
    try:
        sampling_window, _ = find_delta_for_empty_prob(
            sorted_times, p0_max=0.05
        )
    except ValueError:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "count": 0,
            "selected_value_count": 0,
            "available_count": available_count,
            "calculated_minimum": (
                int(calculated_minimum)
                if calculated_minimum is not None else float("nan")
            ),
            "required_minimum": (
                required_minimum if required_minimum is not None else float("nan")
            ),
            "subsampling_succeeded": 0,
            "minimum_requirement_met": 0,
        }
    sampled_times, _ = find_samples_path(
        sorted_times,
        MinimumNumberOfSamples=(required_minimum or 0),
        window=sampling_window,
    )
    sampled_times = np.asarray(sampled_times)
    selected_values = sorted_values[np.isin(sorted_times, sampled_times)]
    mean, std, value_count = _sample_statistics(selected_values)
    sample_count = int(len(sampled_times))
    minimum_requirement_met = int(
        required_minimum is not None
        and sample_count >= required_minimum
        and value_count > 0
    )
    return {
        "mean": mean,
        "std": std,
        "count": sample_count,
        "selected_value_count": value_count,
        "available_count": available_count,
        "calculated_minimum": (
            int(calculated_minimum)
            if calculated_minimum is not None else float("nan")
        ),
        "required_minimum": (
            required_minimum if required_minimum is not None else float("nan")
        ),
        "subsampling_succeeded": int(sample_count > 0 and value_count > 0),
        "minimum_requirement_met": minimum_requirement_met,
    }


def subsampled_flow_path_statistics(
    frame: pd.DataFrame,
    sampling_requirements: dict[str, int | None],
    direct_path: bool,
    link_rates: list[float],
    link_delays: list[float],
) -> dict[str, Any]:
    """Subsample packet arrivals separately for each E2E metric."""
    received_frame, delay = _subsampled_delay_values(
        frame, direct_path, link_rates, link_delays
    )
    success = frame["IsReceived"].to_numpy(dtype=float)
    # As in PostProcessing.py, a dropped packet cannot be a successful
    # non-marked end-to-end observation.
    nonmarking = np.where(
        success == 1,
        1.0 - frame["ECN"].to_numpy(dtype=float),
        0.0,
    )

    sampled = {
        "delay": _subsample_metric(
            received_frame["SentTime"].to_numpy(dtype=float),
            delay,
            sampling_requirements["delay"],
        ),
        "success_probability": _subsample_metric(
            frame["SentTime"].to_numpy(dtype=float),
            success,
            sampling_requirements["success_probability"],
        ),
        "nonmarking_probability": _subsample_metric(
            frame["SentTime"].to_numpy(dtype=float),
            nonmarking,
            sampling_requirements["nonmarking_probability"],
        ),
    }

    result: dict[str, Any] = {
        "total_e2e_packet_count": int(len(frame)),
        "total_e2e_received_packet_count": int(len(received_frame)),
    }
    for metric, statistics in sampled.items():
        prefix = "subsampled_e2e_" + metric
        for name, value in statistics.items():
            result[prefix + "_" + name] = value
    return result


def _relative_epsilon(mean: float, std: float, count: float) -> float:
    if count <= 0 or mean <= 0 or not np.isfinite([mean, std, count]).all():
        return float("inf")
    return float(CONFIDENCE_Z * std / (np.sqrt(count) * mean))


def switch_sampling_requirements(
    metrics: dict[str, Any], queue_names: list[str]
) -> tuple[dict[str, Any], dict[str, int | None]]:
    """Derive the legacy path uncertainty and minimum E2E sample counts."""
    delay_means = np.asarray([
        metrics[name + "poisson_samples_queue_delay_mean"] for name in queue_names
    ], dtype=float)
    delay_stds = np.asarray([
        metrics[name + "poisson_samples_queue_delay_std"] for name in queue_names
    ], dtype=float)
    counts = np.asarray([
        metrics[name + "poisson_samples_queue_delay_count"] for name in queue_names
    ], dtype=float)
    success_means = np.asarray([
        metrics[name + "poisson_samples_queue_success_prob_mean"]
        for name in queue_names
    ], dtype=float)
    success_stds = np.asarray([
        metrics[name + "poisson_samples_queue_success_prob_std"]
        for name in queue_names
    ], dtype=float)
    nonmarking_means = np.asarray([
        metrics[name + "poisson_samples_queue_nonmarking_prob_mean"]
        for name in queue_names
    ], dtype=float)
    nonmarking_stds = np.asarray([
        metrics[name + "poisson_samples_queue_nonmarking_prob_std"]
        for name in queue_names
    ], dtype=float)

    delay_mean = float(np.sum(delay_means))
    if delay_mean == 0 and np.all(delay_stds == 0):
        max_delay_epsilon = 0.0
    else:
        max_delay_epsilon = max(
            _relative_epsilon(mean, std, count)
            for mean, std, count in zip(delay_means, delay_stds, counts)
        )

    def probability_aggregate(means: np.ndarray) -> float:
        if np.any(means <= 0):
            return float("-inf")
        return float(np.sum(np.log(means)))

    aggregate = {
        "DelayMean": delay_mean,
        "MaxEpsilonDelay": max_delay_epsilon,
        "e2eDelayStd": float(np.sum(delay_stds)),
        "SuccessProbMean": probability_aggregate(success_means),
        "MaxEpsilonSuccessProb": max(
            _relative_epsilon(mean, std, count)
            for mean, std, count in zip(success_means, success_stds, counts)
        ),
        "e2eSuccessProbStd": float(np.sum(success_stds)),
        "NonMarkingProbMean": probability_aggregate(nonmarking_means),
        "MaxEpsilonNonMarkingProb": max(
            _relative_epsilon(mean, std, count)
            for mean, std, count in zip(nonmarking_means, nonmarking_stds, counts)
        ),
        "e2eNonMarkingProbStd": float(np.sum(nonmarking_stds)),
    }
    number_of_segments = len(queue_names)
    requirements = {
        "delay": calc_min_e2e_samples(
            CONFIDENCE_Z, MAX_CONSISTENCY_ERROR, aggregate, metric="Delay"
        ),
        "success_probability": calc_min_e2e_samples_prob(
            CONFIDENCE_Z,
            MAX_CONSISTENCY_ERROR,
            aggregate,
            number_of_segments,
            metric="SuccessProb",
        ),
        "nonmarking_probability": calc_min_e2e_samples_prob(
            CONFIDENCE_Z,
            MAX_CONSISTENCY_ERROR,
            aggregate,
            number_of_segments,
            metric="NonMarkingProb",
        ),
    }
    return aggregate, requirements


def add_subsampled_consistency_metrics(
    switch_metrics: dict[str, Any],
    subsampled_metrics: dict[str, Any],
    queue_names: list[str],
    switch_aggregate: dict[str, Any],
    confidence_z: float = CONFIDENCE_Z,
) -> dict[str, Any]:
    """Apply PostProcessing.py inequalities to the E2E subsample estimates."""
    result = dict(switch_metrics)
    result.update(subsampled_metrics)

    switch_delay_mean = float(switch_aggregate["DelayMean"])
    delay_count = result["subsampled_e2e_delay_count"]
    delay_bound = float("nan")
    if result["subsampled_e2e_delay_minimum_requirement_met"] and delay_count > 0:
        delay_bound = (
            switch_delay_mean * switch_aggregate["MaxEpsilonDelay"]
            + confidence_z * switch_aggregate["e2eDelayStd"] / np.sqrt(delay_count)
        )
    delay_difference = result["subsampled_e2e_delay_mean"] - switch_delay_mean
    biased_delay_difference = delay_difference - result["total_estimated_bias"]

    result["switch_path_delay_mean"] = switch_delay_mean
    result["switch_path_max_epsilon_delay"] = switch_aggregate["MaxEpsilonDelay"]
    result["subsampled_e2e_minus_switch_delay"] = delay_difference
    result["subsampled_e2e_minus_biased_switch_delay"] = biased_delay_difference
    result["subsampled_e2e_vs_switch_delay_error_bound"] = delay_bound
    delay_check_performed = int(np.isfinite(delay_bound))
    result["subsampled_e2e_vs_switch_delay_check_performed"] = delay_check_performed
    result["subsampled_e2e_vs_switch_delay_consistent"] = (
        int(abs(delay_difference) <= delay_bound)
        if delay_check_performed else float("nan")
    )
    result["subsampled_e2e_vs_switch_delay_consistent_with_bias"] = (
        int(abs(biased_delay_difference) <= delay_bound)
        if delay_check_performed else float("nan")
    )

    probability_specs = (
        (
            "success",
            "subsampled_e2e_success_probability",
            "SuccessProb",
        ),
        (
            "nonmarking",
            "subsampled_e2e_nonmarking_probability",
            "NonMarkingProb",
        ),
    )
    number_of_segments = len(queue_names)
    for label, subsampled_prefix, legacy_prefix in probability_specs:
        switch_log_mean = switch_aggregate[legacy_prefix + "Mean"]
        switch_mean = float(np.exp(switch_log_mean))
        max_epsilon = switch_aggregate["MaxEpsilon" + legacy_prefix]
        e2e_std = switch_aggregate["e2e" + legacy_prefix + "Std"]
        subsampled_mean = result[subsampled_prefix + "_mean"]
        subsampled_count = result[subsampled_prefix + "_count"]
        epsp = float("nan")
        lower_bound = float("nan")
        upper_bound = float("nan")
        consistent = 0
        if (
            result[subsampled_prefix + "_minimum_requirement_met"]
            and subsampled_mean > 0
            and subsampled_count > 0
            and 0 <= max_epsilon < 1
        ):
            epsp = confidence_z * e2e_std / (
                subsampled_mean * np.sqrt(subsampled_count)
            )
            if 0 <= epsp < 1:
                upper_bound = (
                    number_of_segments * np.log1p(max_epsilon)
                    - np.log1p(-epsp)
                )
                lower_bound = (
                    number_of_segments * np.log1p(-max_epsilon)
                    - np.log1p(epsp)
                )
                log_difference = np.log(subsampled_mean) - switch_log_mean
                consistent = int(lower_bound <= log_difference <= upper_bound)
            else:
                log_difference = float("nan")
        else:
            log_difference = float("nan")
        difference = result[subsampled_prefix + "_mean"] - switch_mean
        result[f"switch_path_{label}_probability_mean"] = switch_mean
        result[f"switch_path_max_epsilon_{label}_probability"] = max_epsilon
        result[f"subsampled_e2e_minus_switch_{label}_probability"] = difference
        result[f"subsampled_e2e_minus_switch_{label}_log_probability"] = log_difference
        result[f"subsampled_e2e_{label}_probability_relative_error"] = epsp
        result[f"subsampled_e2e_vs_switch_{label}_log_probability_bounds"] = [
            upper_bound,
            lower_bound,
        ]
        check_performed = int(np.isfinite([lower_bound, upper_bound]).all())
        result[f"subsampled_e2e_vs_switch_{label}_probability_check_performed"] = (
            check_performed
        )
        result[f"subsampled_e2e_vs_switch_{label}_probability_consistent"] = (
            consistent if check_performed else float("nan")
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
        print(f"Checking subsampled consistency for {source}({host_to_ip.get(source, 'unknown')}) -> {destination}({host_to_ip.get(destination, 'unknown')}), path {path_id}")
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
        repetition_results = []
        for run in switch_runs_by_route[route_key]:
            switch_aggregate, requirements = switch_sampling_requirements(
                run, queue_names
            )
            subsampled_metrics = subsampled_flow_path_statistics(
                frame,
                sampling_requirements=requirements,
                direct_path=len(queue_names) == 1,
                link_rates=link_rates,
                link_delays=link_delays,
            )
            repetition_results.append(
                add_subsampled_consistency_metrics(
                    run, subsampled_metrics, queue_names, switch_aggregate
                )
            )
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
