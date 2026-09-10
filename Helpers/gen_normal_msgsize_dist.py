#!/usr/bin/env python3
"""Generate a DCWorkloads message-size distribution file for normally distributed messages.

File format (consumed by WorkloadApp::ReadWorkloadFile in
traffic_generator_module/DC_traffic_generator/WorkloadApp.cc):

    line 1     average message size in bytes.  readAvgMsgSize() in DatacenterSimulation.cc
               reads only this line and computeTraffciRate() turns it into the per-host
               message rate for a given `load`, so it must be the TRUE mean of the
               distribution written below -- otherwise the realised offered load is wrong.
    line 2..N  "<size_bytes> <cdf>" pairs, handed one by one to
               ns3::EmpiricalRandomVariable::CDF(value, cdf).

ns-3 runs EmpiricalRandomVariable with Interpolate=false (its default; this project never
overrides it), so the file defines a DISCRETE distribution, not a piecewise-linear one:

    P(V = v_i) = c_i - c_{i-1}      and      P(V = v_1) = c_1

(see EmpiricalRandomVariable::PreSample/DoSampleCDF in src/core/model/random-variable-stream.cc).
The map is keyed by the CDF value, so two lines sharing a cdf silently overwrite each other --
the cdf column here is therefore strictly increasing.  WorkloadApp casts the draw to uint32_t,
so every size written is an integer.

Discretisation: one atom per integer byte on [mean - trunc*sigma, mean + trunc*sigma], using the
midpoint rule (atom v carries the Gaussian mass of (v-0.5, v+0.5]) so the discrete mean lands on
`mean` rather than half a step above it.  The two tails beyond the truncation are absorbed into
the end atoms, which is what the ns-3 sampler does with out-of-range uniforms anyway.

Usage:
    python3 gen_normal_msgsize_dist.py --mean 1448 --cv 0.2 --out ../DCWorkloads/Foo.txt
"""

import argparse
import math
import os

MSS_BYTES = 1448  # ns3::TcpSocket::SegmentSize set in DatacenterSimulation.cc


def phi(z):
    """Standard normal CDF, evaluated through erfc so both tails stay accurate."""
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def build_cdf(mean, sigma, trunc=4.0, step=1, cdf_decimals=12):
    """Return [(size_bytes, cdf), ...] with a strictly increasing cdf ending at exactly 1.0."""
    lo = max(1, int(round(mean - trunc * sigma)))
    hi = int(round(mean + trunc * sigma))
    if hi <= lo:
        raise ValueError("empty support: mean={} sigma={} trunc={}".format(mean, sigma, trunc))

    grid = list(range(lo, hi, step))
    if grid[-1] != hi:
        grid.append(hi)

    points = []
    prev_written = None
    for i, v in enumerate(grid):
        if i == len(grid) - 1:
            c = 1.0
        else:
            # atom v takes the mass up to its upper bin edge; the lower tail is absorbed
            # into the first atom because P(V = v_1) = c_1.
            c = phi((v + 0.5 * step - mean) / sigma)
        c = round(c, cdf_decimals)
        if prev_written is not None and c <= prev_written:
            continue  # equal cdfs would collide in ns-3's std::map keyed by cdf
        points.append((v, c))
        prev_written = c

    # the loop can drop the 1.0 endpoint only if the point before it already rounded to 1.0,
    # which the tail masses here never do; assert rather than silently ship a cdf below 1.
    assert points[-1][1] == 1.0, "last cdf is {}, not 1.0".format(points[-1][1])
    return points


def moments(points):
    """Exact mean/std/segment statistics of the discrete distribution the file defines."""
    prev_c = 0.0
    m1 = m2 = 0.0
    seg1 = seg2 = 0.0
    p_single = 0.0
    for v, c in points:
        p = c - prev_c
        prev_c = c
        m1 += p * v
        m2 += p * v * v
        segs = -(-v // MSS_BYTES)  # ceil(v / MSS)
        seg1 += p * segs
        seg2 += p * segs * segs
        if segs <= 1:
            p_single += p
    return {
        "mean": m1,
        "std": math.sqrt(max(0.0, m2 - m1 * m1)),
        "mean_segments": seg1,
        # size-biased batch size E[B^2]/E[B]: the batch a randomly chosen PACKET belongs to
        "size_biased_segments": seg2 / seg1 if seg1 else float("nan"),
        "p_single_packet": p_single,
        "min": points[0][0],
        "max": points[-1][0],
    }


def write_dist(path, points, mean_bytes):
    with open(path, "w") as f:
        f.write("{:.3f}\n".format(mean_bytes))
        for v, c in points:
            f.write("{} {}\n".format(v, "1.0" if c == 1.0 else "{:.12f}".format(c)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mean", type=float, required=True, help="centre of the normal, in bytes")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sigma", type=float, help="standard deviation in bytes")
    g.add_argument("--cv", type=float, help="coefficient of variation; sigma = cv * mean")
    ap.add_argument("--trunc", type=float, default=4.0,
                    help="support half-width in sigmas (default 4)")
    ap.add_argument("--step", type=int, default=1, help="byte spacing of the atoms (default 1)")
    ap.add_argument("--out", required=True, help="output .txt path")
    args = ap.parse_args()

    sigma = args.sigma if args.sigma is not None else args.cv * args.mean
    points = build_cdf(args.mean, sigma, trunc=args.trunc, step=args.step)
    mom = moments(points)
    write_dist(args.out, points, mom["mean"])

    print("{}: nominal N({:.1f}, {:.1f}^2) truncated to [{}, {}] B, {} atoms".format(
        os.path.basename(args.out), args.mean, sigma, mom["min"], mom["max"], len(points)))
    print("    exact discrete mean {:.3f} B (written on line 1), std {:.3f} B, CV {:.3f}".format(
        mom["mean"], mom["std"], mom["std"] / mom["mean"]))
    print("    P(single packet) {:.4f} | E[segments] {:.3f} | size-biased E[B^2]/E[B] {:.3f}".format(
        mom["p_single_packet"], mom["mean_segments"], mom["size_biased_segments"]))


if __name__ == "__main__":
    main()
