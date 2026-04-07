import pytest
import numpy as np
from Utils import sample_increments_of_arrivals
from Utils import crosscorr_qdelay_vs_arrival_increments
from Utils import find_queue_size_at_time

def test_basic_counts():
    arrival_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    T = 2.0
    times_to_sample = np.array([0.0, 1.0, 2.0])

    # [0,2): 0,1 → 2
    # [1,3): 1,2 → 2
    # [2,4): 2,3 → 2
    expected = np.array([2, 2, 2])

    result = sample_increments_of_arrivals(arrival_times, T, times_to_sample)
    assert np.all(result == expected)

def test_no_arrivals_in_window():
    arrival_times = np.array([10.0, 20.0, 30.0])
    T = 5.0
    times_to_sample = np.array([0.0, 1.0, 2.0])

    expected = np.array([0, 0, 0])

    result = sample_increments_of_arrivals(arrival_times, T, times_to_sample)
    assert np.all(result == expected)

def test_duplicate_arrivals():
    arrival_times = np.array([1.0, 1.0, 1.0, 2.0])
    T = 1.0
    times_to_sample = np.array([1.0])

    # [1,2): includes all 1.0's → 3
    expected = np.array([3])

    result = sample_increments_of_arrivals(arrival_times, T, times_to_sample)
    assert np.all(result == expected)

def test_right_boundary_exclusion():
    arrival_times = np.array([0.0, 1.0, 2.0])
    T = 1.0
    times_to_sample = np.array([1.0])

    # window = [1,2)
    # includes 1.0, excludes 2.0
    expected = np.array([1])

    result = sample_increments_of_arrivals(arrival_times, T, times_to_sample)
    assert np.all(result == expected)

def test_non_aligned_sampling_times():
    arrival_times = np.array([1.0, 2.0, 3.0])
    T = 1.0
    times_to_sample = np.array([1.5])

    # [1.5, 2.5): includes 2.0 → 1
    expected = np.array([1])

    result = sample_increments_of_arrivals(arrival_times, T, times_to_sample)
    assert np.all(result == expected)

def test_unsorted_arrivals():
    arrival_times = np.array([3.0, 1.0, 2.0])
    T = 1.0
    times_to_sample = np.array([1.0])

    expected = np.array([1])  # only arrival at 1.0

    result = sample_increments_of_arrivals(arrival_times, T, times_to_sample)
    assert np.all(result == expected)

def test_large_window():
    arrival_times = np.array([1.0, 2.0, 3.0])
    T = 100.0
    times_to_sample = np.array([0.0])

    expected = np.array([3])

    result = sample_increments_of_arrivals(arrival_times, T, times_to_sample)
    assert np.all(result == expected)

def test_empty_arrivals_error():
    with pytest.raises(ValueError):
        sample_increments_of_arrivals([], 1.0, [0.0])

def test_negative_T_error():
    with pytest.raises(ValueError):
        sample_increments_of_arrivals([1.0, 2.0], -1.0, [0.0])

def test_non_1d_input_error():
    with pytest.raises(ValueError):
        sample_increments_of_arrivals([[1.0, 2.0]], 1.0, [0.0])

def test_large_input():
    arrival_times = np.linspace(0, 1e6, 100000)
    times_to_sample = np.linspace(0, 1e6, 1000)
    T = 100.0

    result = sample_increments_of_arrivals(arrival_times, T, times_to_sample)

    assert len(result) == len(times_to_sample)

def test_basic_crosscorr_no_normalization():
    arrival_increments = np.array([1, 2, 3])
    queue_delays = np.array([1, 1, 1])
    times = np.array([0, 1, 2])

    res = crosscorr_qdelay_vs_arrival_increments(
        arrival_increments,
        queue_delays,
        times,
        normalize=False,
        subtract_mean=False,
    )

    # manual correlation
    expected = np.correlate(queue_delays, arrival_increments, mode="full")

    assert np.allclose(res["crosscorr"], expected)

def test_mean_subtraction():
    arrival_increments = np.array([1, 2, 3])
    queue_delays = np.array([4, 5, 6])
    times = np.array([0, 1, 2])

    res = crosscorr_qdelay_vs_arrival_increments(
        arrival_increments,
        queue_delays,
        times,
        normalize=False,
        subtract_mean=True,
    )

    x = queue_delays - queue_delays.mean()
    y = arrival_increments - arrival_increments.mean()
    expected = np.correlate(x, y, mode="full")

    assert np.allclose(res["crosscorr"], expected)

def test_normalization():
    arrival_increments = np.array([1, 2, 3])
    queue_delays = np.array([2, 4, 6])
    times = np.array([0, 1, 2])

    res = crosscorr_qdelay_vs_arrival_increments(
        arrival_increments,
        queue_delays,
        times,
        normalize=True,
        subtract_mean=True,
    )

    # peak correlation should be 1 for perfectly linear signals
    assert np.isclose(np.max(res["crosscorr"]), 1.0)

def test_max_lag():
    arrival_increments = np.array([1, 2, 3, 4])
    queue_delays = np.array([1, 2, 3, 4])
    times = np.array([0, 1, 2, 3])

    res = crosscorr_qdelay_vs_arrival_increments(
        arrival_increments,
        queue_delays,
        times,
        max_lag=1,
        normalize=False,
    )

    assert np.all(np.abs(res["lags"]) <= 1)
    assert len(res["lags"]) == 3  # [-1, 0, 1]

def test_self_correlation_symmetry():
    x = np.array([1, 2, 3, 4])
    times = np.arange(len(x))

    res = crosscorr_qdelay_vs_arrival_increments(
        x,
        x,
        times,
        normalize=False,
        subtract_mean=True,
    )

    corr = res["crosscorr"]

    # autocorrelation should be symmetric
    assert np.allclose(corr, corr[::-1])

def test_zero_variance_error():
    arrival_increments = np.array([1, 1, 1])
    queue_delays = np.array([2, 2, 2])
    times = np.array([0, 1, 2])

    with pytest.raises(ValueError):
        crosscorr_qdelay_vs_arrival_increments(
            arrival_increments,
            queue_delays,
            times,
            normalize=True,
        )

def test_mismatched_lengths():
    with pytest.raises(ValueError):
        crosscorr_qdelay_vs_arrival_increments(
            [1, 2],
            [1, 2, 3],
            [0, 1],
        )


def test_empty_input():
    with pytest.raises(ValueError):
        crosscorr_qdelay_vs_arrival_increments(
            [],
            [],
            [],
        )


def test_non_1d_input():
    with pytest.raises(ValueError):
        crosscorr_qdelay_vs_arrival_increments(
            [[1, 2]],
            [1, 2],
            [0, 1],
        )


def test_negative_max_lag():
    with pytest.raises(ValueError):
        crosscorr_qdelay_vs_arrival_increments(
            [1, 2, 3],
            [1, 2, 3],
            [0, 1, 2],
            max_lag=-1,
        )

def test_against_numpy_random():
    rng = np.random.default_rng(0)

    x = rng.normal(size=50)
    y = rng.normal(size=50)
    times = np.arange(50)

    res = crosscorr_qdelay_vs_arrival_increments(
        y,
        x,
        times,
        normalize=False,
        subtract_mean=False,
    )

    expected = np.correlate(x, y, mode="full")

    assert np.allclose(res["crosscorr"], expected)

def test_empty_times_returns_nans():
    times = np.array([])
    queue_sizes = np.array([])
    target_time = np.array([1.0, 2.0, 3.0])
    link_rate = 8.0

    result = find_queue_size_at_time(times, queue_sizes, target_time, link_rate)
    
    assert np.all(np.isnan(result))


def test_exact_match_returns_queue_size_without_drain():
    times = np.array([10.0, 20.0, 30.0])
    queue_sizes = np.array([100.0, 200.0, 300.0])
    target_time = np.array([20.0])
    link_rate = 8.0  # 1 byte per ns

    result = find_queue_size_at_time(times, queue_sizes, target_time, link_rate)

    assert np.allclose(result, np.array([200.0]))


def test_between_two_samples_applies_drain():
    times = np.array([10.0, 20.0, 30.0])
    queue_sizes = np.array([100.0, 200.0, 300.0])
    target_time = np.array([25.0])
    link_rate = 8.0  # 1 byte per ns

    # last known queue at t=20 is 200
    # 5 ns later -> 5 bytes drained
    expected = np.array([195.0])

    result = find_queue_size_at_time(times, queue_sizes, target_time, link_rate)

    assert np.allclose(result, expected)


def test_drain_is_floored_at_zero():
    times = np.array([10.0])
    queue_sizes = np.array([5.0])
    target_time = np.array([20.0])
    link_rate = 8.0  # 1 byte per ns

    # 10 ns later -> 10 bytes drained, but queue cannot go negative
    expected = np.array([0.0])

    result = find_queue_size_at_time(times, queue_sizes, target_time, link_rate)

    assert np.allclose(result, expected)

def test_before_first_time_returns_nan():
    times = np.array([10.0, 20.0])
    queue_sizes = np.array([100.0, 200.0])
    target_time = np.array([5.0])
    link_rate = 8.0

    result = find_queue_size_at_time(times, queue_sizes, target_time, link_rate)

    assert np.isnan(result[0])


def test_vectorized_targets():
    times = np.array([10.0, 20.0, 40.0])
    queue_sizes = np.array([100.0, 80.0, 40.0])
    target_time = np.array([10.0, 15.0, 20.0, 30.0, 50.0])
    link_rate = 16.0  # 2 bytes per ns

    # t=10 -> 100
    # t=15 -> from t=10: 100 - 5*2 = 90
    # t=20 -> 80
    # t=30 -> from t=20: 80 - 10*2 = 60
    # t=50 -> from t=40: 40 - 10*2 = 20
    expected = np.array([100.0, 90.0, 80.0, 60.0, 20.0])

    result = find_queue_size_at_time(times, queue_sizes, target_time, link_rate)

    assert np.allclose(result, expected)

def test_queue_never_increases():
    times = np.array([10.0])
    queue_sizes = np.array([100.0])
    target_time = np.array([15.0, 20.0])
    link_rate = 8.0

    result = find_queue_size_at_time(times, queue_sizes, target_time, link_rate)

    # Should always be <= original queue size
    assert np.all(result <= 100.0)

def test_queue_non_negative():
    times = np.array([10.0])
    queue_sizes = np.array([5.0])
    target_time = np.array([20.0])
    link_rate = 8.0

    result = find_queue_size_at_time(times, queue_sizes, target_time, link_rate)

    assert np.all(result >= 0.0)

def test_mixed_valid_and_invalid_targets():
    times = np.array([10.0, 20.0])
    queue_sizes = np.array([100.0, 200.0])
    target_time = np.array([5.0, 15.0, 25.0])
    link_rate = 8.0

    result = find_queue_size_at_time(times, queue_sizes, target_time, link_rate)

    assert np.isnan(result[0])      # before first sample
    assert result[1] == 95.0        # 100 - 5
    assert result[2] == 195.0       # 200 - 5

def test_zero_link_rate_means_no_drain():
    times = np.array([10.0, 20.0, 30.0])
    queue_sizes = np.array([100.0, 200.0, 300.0])
    target_time = np.array([15.0, 25.0, 35.0])
    link_rate = 0.0

    # no draining, so just latest known queue sizes
    expected = np.array([100.0, 200.0, 300.0])

    result = find_queue_size_at_time(times, queue_sizes, target_time, link_rate)

    assert np.allclose(result, expected)


def test_multiple_targets_same_interval():
    times = np.array([10.0, 20.0])
    queue_sizes = np.array([50.0, 100.0])
    target_time = np.array([21.0, 22.0, 23.5])
    link_rate = 8.0  # 1 byte per ns

    # all use sample at t=20 with queue 100
    expected = np.array([99.0, 98.0, 96.5])

    result = find_queue_size_at_time(times, queue_sizes, target_time, link_rate)

    assert np.allclose(result, expected)

def test_exact_formula_against_manual_computation():
    times = np.array([5.0, 12.0, 18.0])
    queue_sizes = np.array([40.0, 30.0, 50.0])
    target_time = np.array([6.0, 14.0, 19.0])
    link_rate = 24.0  # 3 bytes per ns

    result = find_queue_size_at_time(times, queue_sizes, target_time, link_rate)

    manual = np.array([
        max(0.0, 40.0 - (6.0 - 5.0) * 3.0),
        max(0.0, 30.0 - (14.0 - 12.0) * 3.0),
        max(0.0, 50.0 - (19.0 - 18.0) * 3.0),
    ])

    assert np.allclose(result, manual)
def run_all_tests():
    """Run all tests defined in this module."""
    return pytest.main([__file__, "-s"])


if __name__ == "__main__":
    raise SystemExit(run_all_tests())