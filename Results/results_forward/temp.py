#!/usr/bin/env python3
import re
import csv
import sys
import argparse
from pathlib import Path

# Patterns
RE_BYTES = re.compile(
    r'^Bytes in Queue when sending packet with ID:\s*(\d+)\s*is\s*:\s*(\d+)\s*at time:\s*(\d+)\s*$'
)
RE_START = re.compile(
    r'^Start transmitting packet with ID:\s*(\d+)\s*at time:\s*(\d+)\s*$'
)

def parse_args():
    p = argparse.ArgumentParser(
        description="Pair 'Bytes in Queue...' and 'Start transmitting...' lines by ID and compare times."
    )
    p.add_argument("logfile", help="Path to the log file")
    p.add_argument("--tolerance", "-t", type=float, default=1.0,
                   help="Absolute tolerance for comparison (default: 1.0)")
    p.add_argument("--factor", type=float, default=0.3/8,
                   help="Scaling factor applied to delta time (default: 0.3/8)")
    p.add_argument("--output", "-o", default="-",
                   help="CSV output file (default: stdout)")
    p.add_argument("--all", action="store_true",
                   help="If an ID appears multiple times, report all occurrences (default: only first pair).")
    return p.parse_args()

def write_csv(rows, outpath):
    fieldnames = [
        "id", "bytes_reported", "bytes_line_time", "start_time",
        "delta_time", "scaled_value", "abs_diff", "within_tolerance"
    ]
    if outpath == "-" or outpath.lower() == "stdout":
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    else:
        with open(outpath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

def main():
    args = parse_args()
    path = Path(args.logfile)
    print(f"Reading log file: {path}")
    if not path.exists():
        sys.stderr.write(f"Error: file not found: {path}\n")
        sys.exit(1)

    # We support multiple occurrences per ID by keeping lists.
    bytes_seen = {}  # id -> list of (bytes_reported, bytes_time)
    start_seen = {}  # id -> list of start_time
    reported = set() # track (id, index) tuples already emitted if not --all

    rows = []

    def emit_pairs_for_id(pid):
        """Try to pair up entries for a given id and emit rows."""
        b_list = bytes_seen.get(pid, [])
        s_list = start_seen.get(pid, [])
        n = min(len(b_list), len(s_list))
        for i in range(n):
            key = (pid, i)
            if not args.all and key in reported:
                continue
            bytes_reported, btime = b_list[i]
            stime = s_list[i]
            delta = int(stime) - int(btime)
            scaled = delta * args.factor
            diff = abs(scaled - bytes_reported)
            within = diff <= args.tolerance
            rows.append({
                "id": pid,
                "bytes_reported": bytes_reported,
                "bytes_line_time": int(btime),
                "start_time": int(stime),
                "delta_time": delta,
                "scaled_value": scaled,
                "abs_diff": diff,
                "within_tolerance": within,
            })
            reported.add(key)

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")

            m = RE_BYTES.match(line)
            if m:
                pid = int(m.group(1))
                bytes_reported = float(m.group(2))
                bytes_time = int(m.group(3))
                bytes_seen.setdefault(pid, []).append((bytes_reported, bytes_time))
                emit_pairs_for_id(pid)
                continue

            m = RE_START.match(line)
            if m:
                pid = int(m.group(1))
                start_time = int(m.group(2))
                start_seen.setdefault(pid, []).append(start_time)
                emit_pairs_for_id(pid)
                continue

            # Ignore other lines
    filtered = [r for r in rows if not r["within_tolerance"]]
    write_csv(filtered, args.output)

if __name__ == "__main__":
    main()
