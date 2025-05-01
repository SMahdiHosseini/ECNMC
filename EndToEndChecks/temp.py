# import numpy as np
# # from Utils import *

# list =[]
# print(np.average(list))
# print(np.average([x[0] for x in list]))
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns

# # Example data
# # import matplotlib.pyplot as plt
# import numpy as np
# print(np.random.binomial(n=1, p=0.99, size=100))

# Example data
# f = [1, 2, 3, 4]  # List of f values
# Bias = {
#     1: np.random.normal(0, 1, 20).tolist(),
#     2: np.random.normal(1, 1, 20).tolist(),
#     3: np.random.normal(2, 1, 20).tolist(),
#     4: np.random.normal(3, 1, 20).tolist()
# }
# Traffic = {
#     1: np.random.normal(10, 2, 20).tolist(),
#     2: np.random.normal(20, 2, 20).tolist(),
#     3: np.random.normal(30, 2, 20).tolist(),
#     4: np.random.normal(40, 2, 20).tolist()
# }

# # Convert data into lists for plotting
# traffic_values = []
# bias_values = []
# f_labels = []
# f_positions = []

# for i, key in enumerate(f):
#     traffic_values.append(Traffic[key])
#     bias_values.append(Bias[key])
#     f_labels.append(str(key))
#     f_positions.append(np.mean(Traffic[key]))

# # Create the boxplot
# fig, ax1 = plt.subplots(figsize=(10, 6))
# ax1.boxplot(bias_values, positions=f_positions, widths=5, patch_artist=True)
# ax1.set_xlabel("Traffic")
# ax1.set_ylabel("Bias")
# ax1.set_title("Bias vs Traffic with f on Top X Axis")

# # Create secondary x-axis for f values
# ax2 = ax1.twiny()
# ax2.set_xlim(ax1.get_xlim())
# ax2.set_xticks(f_positions)
# ax2.set_xticklabels(f_labels)
# ax2.set_xlabel("f values")

# plt.grid(True, linestyle='--', alpha=0.5)
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import anderson

def read_data(filename):
    times = []
    sizes = []

    with open(filename, 'r') as f:
        for line in f:
            if ',' not in line:
                continue
            time_str, size_str = line.strip().split(',')
            try:
                times.append(float(time_str))
                sizes.append(float(size_str))
            except ValueError:
                continue

    return np.array(times), np.array(sizes)

def read_actual_cdf(filename):
    x_vals = []
    cdf_vals = []

    with open(filename, 'r') as f:
        i = 0
        for line in f:
            if i == 0:
                i += 1
                continue
            x_str, cdf_str = line.strip().split()
            try:
                x_vals.append(float(x_str))
                cdf_vals.append(float(cdf_str))
            except ValueError:
                continue
    return np.array(x_vals), np.array(cdf_vals)

def check_poisson_process(times):
    sorted_times = np.sort(times)
    inter_arrivals = np.diff(sorted_times)

    normalized = inter_arrivals / np.mean(inter_arrivals)

    result = anderson(normalized, dist='expon')

    print("Anderson-Darling Test Statistic:", result.statistic)
    print("Critical Values:", result.critical_values)
    print("Significance Levels:", result.significance_level)

    for stat, alpha in zip(result.critical_values, result.significance_level):
        if result.statistic < stat:
            print(f"At {alpha}%: Inter-arrivals are exponential ⇒ Poisson process likely.")
        else:
            print(f"At {alpha}%: Inter-arrivals are not exponential ⇒ Not Poisson.")

def plot_size_cdf(sizes, actual_cdf_file=None):
    sorted_sizes = np.sort(sizes)
    cdf_empirical = np.arange(1, len(sizes) + 1) / len(sizes)

    plt.figure(figsize=(8, 6))
    plt.plot(sorted_sizes, cdf_empirical, label="Empirical CDF", marker='.', linestyle='none')

    if actual_cdf_file:
        x_vals, cdf_vals = read_actual_cdf(actual_cdf_file)
        plt.plot(x_vals, cdf_vals, label="Actual CDF", color='orange', linewidth=2)
    
    plt.xscale('log')

    plt.xlabel("Size")
    plt.ylabel("CDF")
    plt.title("Empirical vs Actual CDF of Sizes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("size_cdf.png")

if __name__ == "__main__":
    traffic_file = "temp.txt"        # time,size file
    actual_cdf_file = "../DCWorkloads/Google_AllRPC.txt"       # value,cdf file

    times, sizes = read_data(traffic_file)

    if len(times) < 2:
        print("Not enough data to analyze.")
    else:
        check_poisson_process(times)
        plot_size_cdf(sizes, actual_cdf_file)


