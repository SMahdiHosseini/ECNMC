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
import random
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from Utils import (
    calculate_offline_delay_bias_DC,
    compute_average_packet_size,
    convert_to_float,
)


NS3_PATH = Path(__file__).resolve().parents[3]
SOURCE_FILE_RE = re.compile(r"^(R(?P<rack>\d+)H(?P<host>\d+))_ALL_EndToEnd_packets\.csv$")
HOST_RE = re.compile(r"^R(?P<rack>\d+)H(?P<host>\d+)$")
REQUIRED_E2E_COLUMNS = {"SourceIp", "DestinationIp", "Path", "PayloadSize"}
FLOW_INVENTORY_FILE = "all_to_all_flow_inventory.json"
FLOW_INVENTORY_VERSION = 1
QUEUE_METRIC_RE = re.compile(r"^(?:T\d+A\d+|A\d+T\d+|T\d+H\d+)")


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


def _source_fingerprint(source_files: list[Path]) -> list[dict[str, int | str]]:
    return [
        {
            "name": path.name,
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in source_files
    ]


def _build_flow_inventory(
    source_files: list[Path],
) -> tuple[dict[str, str], dict[tuple[str, str, int], int]]:
    """Read aggregate CSVs once to build the complete experiment inventory."""
    mapping: dict[str, str] = {}
    for file_path in source_files:
        match = SOURCE_FILE_RE.fullmatch(file_path.name)
        if match is None:
            continue
        source_ip = _read_source_ip(file_path)
        if source_ip is not None:
            mapping[source_ip] = match.group(1)
    discovered: dict[tuple[str, str, int], int] = {}

    for file_path in source_files:
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
            destination = mapping.get(str(destination_ip))
            if destination is None:
                raise ValueError(
                    f"Destination IP {destination_ip} in {file_path} has no matching "
                    "RiHj_ALL_EndToEnd_packets.csv source file"
                )
            discovered[(source, destination, int(path_id))] = int(packet_count)

    return mapping, discovered


def _write_flow_inventory(
    cache_path: Path,
    fingerprint: list[dict[str, int | str]],
    ip_to_host: dict[str, str],
    discovered: dict[tuple[str, str, int], int],
) -> None:
    host_to_ip = {host: ip for ip, host in ip_to_host.items()}
    flow_pairs = sorted({(source, destination) for source, destination, _ in discovered})
    number_of_racks, hosts_per_rack = infer_topology(ip_to_host)
    payload = {
        "version": FLOW_INVENTORY_VERSION,
        "source_files": fingerprint,
        "ip_to_host": ip_to_host,
        "host_to_ip": host_to_ip,
        "topology": {
            "number_of_racks": number_of_racks,
            "hosts_per_rack": hosts_per_rack,
            "alternative_routes": [number_of_racks - 1, hosts_per_rack],
        },
        "existing_flows": [source + destination for source, destination in flow_pairs],
        "flow_to_ips": {
            source + destination: [host_to_ip[source], host_to_ip[destination]]
            for source, destination in flow_pairs
        },
        "flow_paths": [
            [source, destination, path_id, packet_count]
            for (source, destination, path_id), packet_count in sorted(discovered.items())
        ],
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=cache_path.parent,
        prefix=cache_path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as cache_file:
        json.dump(payload, cache_file, separators=(",", ":"))
        temporary_path = Path(cache_file.name)
    temporary_path.chmod(0o644)
    temporary_path.replace(cache_path)


def load_flow_inventory(
    experiment_dir: Path,
) -> tuple[dict[str, str], dict[tuple[str, str, int], int]]:
    """Load a valid inventory cache or build and atomically save it once."""
    source_files = sorted(experiment_dir.glob("*_ALL_EndToEnd_packets.csv"))
    if not source_files:
        raise FileNotFoundError(
            f"No RiHj_ALL_EndToEnd_packets.csv files found in {experiment_dir}"
        )
    fingerprint = _source_fingerprint(source_files)
    cache_path = experiment_dir / FLOW_INVENTORY_FILE

    if cache_path.is_file():
        try:
            with cache_path.open() as cache_file:
                payload = json.load(cache_file)
            if (
                payload.get("version") == FLOW_INVENTORY_VERSION
                and payload.get("source_files") == fingerprint
            ):
                discovered = {
                    (source, destination, int(path_id)): int(packet_count)
                    for source, destination, path_id, packet_count
                    in payload["flow_paths"]
                }
                print(f"Loaded flow inventory: {cache_path}")
                return dict(payload["ip_to_host"]), discovered
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass

    ip_to_host, discovered = _build_flow_inventory(source_files)
    _write_flow_inventory(cache_path, fingerprint, ip_to_host, discovered)
    print(f"Saved flow inventory: {cache_path}")
    return ip_to_host, discovered


def build_ip_to_host(experiment_dir: Path) -> dict[str, str]:
    """Return the cached address-to-host mapping for an experiment."""
    ip_to_host, _ = load_flow_inventory(experiment_dir)
    return ip_to_host


def infer_topology(ip_to_host: dict[str, str]) -> tuple[int, int]:
    """Return (number of racks, hosts per rack) from recorded source hosts."""
    hosts_by_rack: dict[int, set[int]] = {}
    for host_name in ip_to_host.values():
        rack, host = host_coordinates(host_name)
        hosts_by_rack.setdefault(rack, set()).add(host)
    host_counts = {len(hosts) for hosts in hosts_by_rack.values()}
    if len(hosts_by_rack) < 2 or len(host_counts) != 1:
        raise ValueError("Flow inventory does not describe a uniform multi-rack topology")
    return len(hosts_by_rack), host_counts.pop()


def discover_flow_paths(
    experiment_dir: Path,
    max_flows: int | None = None,
    randomize: bool = False,
    start_flow: int = 0,
) -> dict[tuple[str, str, int], int]:
    """Return cached flow/path pairs, optionally selecting distinct flows."""
    _, discovered = load_flow_inventory(experiment_dir)

    return select_flow_paths(discovered, max_flows, randomize, start_flow)


def select_flow_paths(
    discovered: dict[tuple[str, str, int], int],
    max_flows: int | None,
    randomize: bool,
    start_flow: int = 0,
) -> dict[tuple[str, str, int], int]:
    """Select complete flows after a zero-based offset, retaining all paths."""
    if start_flow < 0:
        raise ValueError("start_flow must be non-negative")

    flow_pairs = sorted({(src, dst) for src, dst, _ in discovered})
    eligible_flows = flow_pairs[start_flow:]
    if max_flows is not None:
        if randomize:
            eligible_flows = random.sample(
                eligible_flows, min(max_flows, len(eligible_flows))
            )
        else:
            eligible_flows = eligible_flows[:max_flows]
    selected = set(eligible_flows)
    return {key: count for key, count in discovered.items() if key[:2] in selected}


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


def store_metrics(
    merged: dict[str, dict[str, dict[str, list[Any]]]],
    flow_name: str,
    path_id: int,
    metrics: dict[str, Any],
    save_per_switch: bool = True,
) -> None:
    path_key = str(path_id)
    for metric, value in metrics.items():
        if not save_per_switch and QUEUE_METRIC_RE.match(metric):
            continue
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
    start_flow: int = 0,
) -> dict[tuple[str, int], dict[str, Any]]:
    ip_to_host, all_discovered = load_flow_inventory(experiment_dir)
    discovered = select_flow_paths(all_discovered, max_flows, randomize, start_flow)
    if not discovered:
        raise FileNotFoundError(f"No aggregate end-to-end records found in {experiment_dir}")

    number_of_racks, hosts_per_rack = infer_topology(ip_to_host)
    alternative_routes = [number_of_racks - 1, hosts_per_rack]
    average_packet_size = compute_average_packet_size(str(experiment_dir) + "/")
    route_cache: dict[tuple[str, ...], dict[str, Any]] = {}
    flow_results: dict[tuple[str, int], dict[str, Any]] = {}

    for source, destination, path_id in sorted(discovered):
        print(f"Analyzing flow {source} -> {destination}, path {path_id}")
        flow_name = source + destination
        queue_names = queues_for_flow(source, destination, path_id)
        # mahdi: code checked by this line
        cache_key = tuple(queue_names)
        if cache_key not in route_cache:
            runs: list[dict[str, Any]] = []
            for _ in range(repetitions):
                route_capacities = (
                    [0, queue_capacities[3]]
                    if len(queue_names) == 1 else queue_capacities
                )
                runs.append(
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
    parser.add_argument(
        "--start-flow", type=int, default=0,
        help="Skip this many flows in sorted discovery order before selecting flows",
    )
    parser.add_argument("--randomize", action="store_true", help="Randomly select flows if --max-flows is set")
    parser.add_argument(
        "--save-per-switch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save queue-specific metrics (disable with --no-save-per-switch)",
    )
    parser.add_argument("--sampling-factor", type=float)
    parser.add_argument("--output-name", default="delay_minimum_bias_e2e_vs_switch_poisson_all_to_all.json")
    args = parser.parse_args()

    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    if args.max_flows is not None and args.max_flows < 1:
        parser.error("--max-flows must be at least 1")
    if args.start_flow < 0:
        parser.error("--start-flow must be non-negative")

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
                        args.repetitions, args.sampling_factor,
                        max_flows=args.max_flows,
                        start_flow=args.start_flow,
                        randomize=args.randomize,
                    )
                    for (flow_name, path_id), metrics in flow_results.items():
                        store_metrics(
                            merged, flow_name, path_id, metrics,
                            save_per_switch=args.save_per_switch,
                        )
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
