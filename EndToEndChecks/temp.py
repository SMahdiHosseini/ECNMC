import numpy as np
# from Utils import *

list = [
                        6039.702496157652,
                        1960.901693161407,
                        2437.792690346038,
                        1060.3901369474113,
                        14788.471946437978,
                        9157.589038778124,
                        1253.7530814002757,
                        4070.3398736765657,
                        4449.414459906187,
                        5064.826519884737,
                        82063.3462265752,
                        1275.5721678746008,
                        831.7224287331619,
                        77986.6539339941,
                        1259.4903545124303,
                        6734.428571196932,
                        52146.54622993412,
                        2676.6522107444957,
                        1459.8212370468955,
                        2543.952668213181,
                        39660.664034305904,
                        1772.048136289897,
                        3865.1366522715284,
                        5330.488439183394,
                        5787.7347278959405,
                        2474.6961039077037,
                        1111.2591169624066,
                        887.019503879876,
                        69667.69771703363,
                        147151.16650485003,
                        54964.628368270976,
                        1615.998383370973,
                        4395.224575103194,
                        3996.421825220984,
                        1004.0275166796648,
                        36713.56653671433,
                        21304.34390839286,
                        10069.163799419372,
                        11036.632482024816,
                        1174.8944382072432,
                        32773.743434682074,
                        27134.245253503235,
                        101259.9721045294,
                        9511.273880139584,
                        5717.896729288562,
                        12507.79933343261,
                        42977.904478126955,
                        36255.98280956559,
                        32314.126134480928,
                        69274.88599370359,
                        1322.3422291490294,
                        1702.735683667404,
                        1265.8714696082086,
                        45389.61694166758,
                        2968.930671302899,
                        2709.2458219891146,
                        2114.206946667507,
                        25094.999248534787,
                        4265.056088155044,
                        42001.5602401894,
                        30398.11633567862,
                        881.6904649627589,
                        1559.7766307210036,
                        2285.635333404654,
                        2292.2792756591607,
                        33069.80166879228,
                        23716.485070865423,
                        20118.826513506632,
                        16505.918314917413,
                        3198.2642631877247,
                        2268.579529459257,
                        1036.5947168608259,
                        30074.474556908994,
                        32467.54647334835,
                        1544.523896906895,
                        8983.078921086544,
                        8446.35008767299,
                        50071.70875812125,
                        1245.8901048253742,
                        3607.400386467646,
                        3894.144962839852,
                        5968.943402550037,
                        33946.94040286163,
                        838.7935013473921,
                        1844.745420752461,
                        20201.90900254988,
                        15005.774070314643,
                        17313.4474470348,
                        7470.926209046275,
                        4742.327607388407,
                        3663.672464006833,
                        24106.159954824998,
                        12144.051981768227,
                        32666.095074928155,
                        19139.58635288977,
                        3535.6929805980103,
                        117878.47793490169,
                        26957.273886405364,
                        1091.2284600508442,
                        15153.57480665271
                    ]
print(np.average(list))
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


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import anderson

# def read_data(filename):
#     times = []
#     sizes = []

#     with open(filename, 'r') as f:
#         for line in f:
#             if ',' not in line:
#                 continue
#             time_str, size_str = line.strip().split(',')
#             try:
#                 times.append(float(time_str))
#                 sizes.append(float(size_str))
#             except ValueError:
#                 continue

#     return np.array(times), np.array(sizes)

# def read_actual_cdf(filename):
#     x_vals = []
#     cdf_vals = []

#     with open(filename, 'r') as f:
#         i = 0
#         for line in f:
#             if i == 0:
#                 i += 1
#                 continue
#             x_str, cdf_str = line.strip().split()
#             try:
#                 x_vals.append(float(x_str))
#                 cdf_vals.append(float(cdf_str))
#             except ValueError:
#                 continue
#     return np.array(x_vals), np.array(cdf_vals)

# def check_poisson_process(times):
#     sorted_times = np.sort(times)
#     inter_arrivals = np.diff(sorted_times)

#     normalized = inter_arrivals / np.mean(inter_arrivals)

#     result = anderson(normalized, dist='expon')

#     print("Anderson-Darling Test Statistic:", result.statistic)
#     print("Critical Values:", result.critical_values)
#     print("Significance Levels:", result.significance_level)

#     for stat, alpha in zip(result.critical_values, result.significance_level):
#         if result.statistic < stat:
#             print(f"At {alpha}%: Inter-arrivals are exponential ⇒ Poisson process likely.")
#         else:
#             print(f"At {alpha}%: Inter-arrivals are not exponential ⇒ Not Poisson.")

# def plot_size_cdf(sizes, actual_cdf_file=None):
#     sorted_sizes = np.sort(sizes)
#     cdf_empirical = np.arange(1, len(sizes) + 1) / len(sizes)

#     plt.figure(figsize=(8, 6))
#     plt.plot(sorted_sizes, cdf_empirical, label="Empirical CDF", marker='.', linestyle='none')

#     if actual_cdf_file:
#         x_vals, cdf_vals = read_actual_cdf(actual_cdf_file)
#         plt.plot(x_vals, cdf_vals, label="Actual CDF", color='orange', linewidth=2)
    
#     plt.xscale('log')

#     plt.xlabel("Size")
#     plt.ylabel("CDF")
#     plt.title("Empirical vs Actual CDF of Sizes")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig("size_cdf.png")

# if __name__ == "__main__":
#     traffic_file = "temp.txt"        # time,size file
#     actual_cdf_file = "../DCWorkloads/Google_AllRPC.txt"       # value,cdf file

#     times, sizes = read_data(traffic_file)

#     if len(times) < 2:
#         print("Not enough data to analyze.")
#     else:
#         check_poisson_process(times)
#         plot_size_cdf(sizes, actual_cdf_file)


