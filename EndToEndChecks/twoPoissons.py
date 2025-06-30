import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
T_total = 500  # Total time
rates = [0.01, 0.05, 0.1, 0.5, 0.9]  # Different Poisson process rates
c = 1.0  # Constant = rate * n_obs

np.random.seed(40)  # For reproducibility

def generate_poisson_observations(rate, T_total):
    n_obs = int(np.ceil(c / rate))  # Number of consecutive observations
    t = 0
    observation_times = []

    while t < T_total:
        # Time until next Poisson event
        t += np.random.exponential(1 / rate)
        if t + n_obs > T_total:
            break
        # Record consecutive integer time observations
        observation_times.extend(np.arange(int(t), int(t) + n_obs))

    return np.array(observation_times)

# Create plots
fig, axes = plt.subplots(len(rates), 1, figsize=(12, 2 * len(rates)), sharex=True)

time = np.arange(T_total)

for i, rate in enumerate(rates):
    obs_p1 = generate_poisson_observations(rate, T_total)
    obs_p2 = generate_poisson_observations(rate, T_total)

    signal_p1 = np.zeros(T_total)
    signal_p2 = np.zeros(T_total)

    signal_p1[obs_p1] = 1
    signal_p2[obs_p2] = 1

    axes[i].plot(obs_p1, [1 for i in range(len(obs_p1))], '|', label='P1', markersize=10, color='blue')
    axes[i].plot(obs_p2, [1 for i in range(len(obs_p2))], '|', label='P2', markersize=10, color='red', alpha=0.6)
    axes[i].set_title(f'Observations with rate λ = {rate:.2f}, n_obs = {int(c / rate)}')
    axes[i].set_ylim(-0.1, 1.1)
    axes[i].legend(loc='upper right')

axes[-1].set_xlabel('Time')
plt.tight_layout()
plt.savefig('two_poissons.png', dpi=300)
plt.close()
