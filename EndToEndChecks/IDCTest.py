from Utils import *
import numpy as np

def mixed_burst_size_sampler(rng):
    u = rng.random()
    if u <= 0.89110434132683:
        return 1       # small messages
    elif u < 0.926257074284752:
        return 2
    elif u < 0.937795399226954:
        return 3
    elif u < 0.951999148490551:
        return 4
    elif u < 0.995923894177891:
        return 70
    else:
        return 500    # large messages

def generate_poisson_timestamps(rate, duration_seconds, seed=None):
    """
    Generate timestamps for a Poisson process with nanosecond precision.

    Parameters
    ----------
    rate : float
        Poisson rate (events per second).
    duration_seconds : float
        Total simulation duration in seconds.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    timestamps_ns : np.ndarray
        Event timestamps in nanoseconds (int64).
    """

    if seed is not None:
        np.random.seed(seed)

    timestamps = []
    t = 0.0

    while True:
        # exponential inter-arrival
        inter_arrival = np.random.exponential(1.0 / rate)
        t += inter_arrival

        if t > duration_seconds:
            break

        timestamps.append(t)

    timestamps = np.array(timestamps)

    # convert to nanoseconds
    timestamps_ns = (timestamps * 1e9).astype(np.int64)

    return timestamps_ns

import numpy as np

def generate_bursty_poisson_timestamps(
    burst_rate,
    duration_seconds,
    mean_burst_size,
    mean_burst_duration_ns,
    seed=None,
):
    """
    Generate bursty event timestamps.

    Bursts arrive according to a Poisson process with rate `burst_rate`
    bursts/second. Each burst contains a random number of packets, and
    packet timestamps are placed uniformly within the burst duration.

    Parameters
    ----------
    burst_rate : float
        Average number of bursts per second.
    duration_seconds : float
        Total simulation duration in seconds.
    mean_burst_size : float
        Mean number of packets per burst. Actual burst size is sampled
        from Poisson(mean_burst_size), with a minimum of 1.
    mean_burst_duration_ns : float
        Mean burst duration in nanoseconds. Actual burst duration is
        sampled from an exponential distribution.
    seed : int, optional
        Random seed.

    Returns
    -------
    timestamps_ns : np.ndarray
        Sorted packet timestamps in nanoseconds (int64).
    """
    rng = np.random.default_rng(seed)

    if burst_rate <= 0:
        raise ValueError("burst_rate must be > 0")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be > 0")
    if mean_burst_duration_ns <= 0:
        raise ValueError("mean_burst_duration_ns must be > 0")

    duration_ns = int(round(duration_seconds * 1e9))

    # Generate burst start times as a Poisson process
    burst_starts_ns = []
    t = 0.0
    while True:
        inter_arrival = rng.exponential(1.0 / burst_rate)  # seconds
        t += inter_arrival
        if t >= duration_seconds:
            break
        burst_starts_ns.append(int(round(t * 1e9)))

    all_timestamps = []

    for start_ns in burst_starts_ns:
        burst_size = max(1, rng.poisson(mean_burst_size))
        burst_duration_ns = max(1, int(round(rng.exponential(mean_burst_duration_ns))))

        # Put packets uniformly inside the burst
        offsets = rng.uniform(0, burst_duration_ns, size=burst_size - 1)
        offsets = np.insert(offsets, 0, 0)
        pkt_times = start_ns + offsets.astype(np.int64)
        # Keep only timestamps inside the observation interval
        pkt_times = pkt_times[pkt_times < duration_ns]
        all_timestamps.append(pkt_times)

    if not all_timestamps:
        return np.array([], dtype=np.int64)

    timestamps_ns = np.sort(np.concatenate(all_timestamps).astype(np.int64))
    return timestamps_ns

import numpy as np

def generate_bursty_poisson_timestamps_wsampler(
    burst_rate,
    duration_seconds,
    burst_size_sampler,
    mean_burst_duration_ns,
    seed=None,
):
    """
    Generate bursty packet timestamps where burst arrivals are Poisson,
    but burst sizes come from an arbitrary user-provided distribution.

    Parameters
    ----------
    burst_rate : float
        Average number of bursts (messages) per second.
    duration_seconds : float
        Total simulation duration in seconds.
    burst_size_sampler : callable
        Function of the form burst_size_sampler(rng) -> int
        returning the number of packets in one burst/message.
    mean_burst_duration_ns : float
        Mean burst duration in nanoseconds. Actual burst duration is
        sampled from an exponential distribution.
    seed : int, optional
        Random seed.

    Returns
    -------
    timestamps_ns : np.ndarray
        Sorted packet timestamps in nanoseconds (int64).
    """
    rng = np.random.default_rng(seed)

    if burst_rate <= 0:
        raise ValueError("burst_rate must be > 0")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be > 0")
    if mean_burst_duration_ns <= 0:
        raise ValueError("mean_burst_duration_ns must be > 0")

    duration_ns = int(round(duration_seconds * 1e9))

    # Poisson burst/message arrival times
    burst_starts_ns = []
    t = 0.0
    while True:
        t += rng.exponential(1.0 / burst_rate)
        if t >= duration_seconds:
            break
        burst_starts_ns.append(int(round(t * 1e9)))

    all_timestamps = []

    for start_ns in burst_starts_ns:
        burst_size = int(burst_size_sampler(rng))
        if burst_size <= 0:
            continue

        burst_duration_ns = max(1, int(round(rng.exponential(mean_burst_duration_ns))))

        # Uniform placement inside the burst duration
        offsets = rng.integers(0, burst_duration_ns + 1, size=burst_size, endpoint=False)
        pkt_times = start_ns + offsets

        pkt_times = pkt_times[pkt_times < duration_ns]
        all_timestamps.append(pkt_times)

    if not all_timestamps:
        return np.array([], dtype=np.int64)

    return np.sort(np.concatenate(all_timestamps).astype(np.int64))

# timestamps = generate_poisson_timestamps(
#     rate=350000,          # 350000 packets/sec
#     duration_seconds=0.09 # 90 ms
# )

# print(f"Generated {len(timestamps)} timestamps.")

# plot_idc_over_delta(timestamps)

# timestamps = generate_bursty_poisson_timestamps(
#     burst_rate=50000,                 # bursts/sec
#     duration_seconds=0.09,          # 90 ms
#     mean_burst_size=2,             # avg packets per burst
#     mean_burst_duration_ns=50,  # avg burst lasts 50 ns
# )

# print(f"Generated {len(timestamps)} bursty timestamps.")

# plot_idc_over_delta(timestamps, t_start=0, duration=0.09*1e9)

timestamps = generate_bursty_poisson_timestamps_wsampler(
    burst_rate=50000,                 # bursts/sec
    duration_seconds=0.09,          # 90 ms
    burst_size_sampler=mixed_burst_size_sampler,
    mean_burst_duration_ns=50000,  # avg burst lasts 50 us
)

print(f"Generated {len(timestamps)} bursty timestamps.")

plot_idc_over_delta(timestamps, t_start=0, duration=0.09*1e9)