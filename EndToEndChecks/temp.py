import numpy as np
# from Utils import *
import matplotlib.pyplot as plt
import seaborn as sns
list_1 =  [
        0.05026910928328604,
        0.05720629307112812,
        0.05457712806589786,
        0.05767868073290651,
        0.06309304338131042,
        0.06900012724075948,
        0.05813402057698128,
        0.0842650096357782,
        0.06739679651706264,
        0.06512278311059728,
        0.05631304851775389,
        0.07097805401750656,
        0.06340052833505445,
        0.05180725609502143,
        0.06896461629805317,
        0.07523247099304857,
        0.061211794804956775,
        0.07141656887948274,
        0.0536553450646538,
        0.059771968736968425,
        0.0850926105367115,
        0.0969903381250113,
        0.07222483764526651,
        0.07557172458107443,
        0.06575416384188688,
        0.08873096225504992,
        0.05679713081529292,
        0.057237839126051564,
        0.0630054188484241,
        0.06853558099390397
    ]
# list_2 =  [
#                     ]

print(np.average(list_1))
# print(np.average([x[0] for x in list]))
# res = [((list_2[i][0] - list_1[i])) for i in range(len(list_1))]
# print(np.average(res))
# print(np.std(res))
# print(np.median(res))
# # plot hidtogram of res
# plt.hist(res, bins=30, edgecolor='black', alpha=0.7)
# plt.savefig('diff_hist.png')
# plt.close()
# # plot histogram of list_1 and list_2[i][0]
# plt.hist(list_1, bins=30, edgecolor='red', alpha=0.7, label='switch')
# plt.hist([x[0] for x in list_2], bins=30, edgecolor='blue', alpha=0.7, label='e2e')
# plt.legend()
# plt.savefig('histogram_switchVSe2e.png')
# plt.figure(figsize=(10, 6))
# plt.hist(list, bins=4, edgecolor='black', alpha=0.7)
# plt.title('cdf', fontsize=16)
# plt.xlabel('size', fontsize=16)
# plt.ylabel('CDF', fontsize=16)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.grid()
# plt.show()
# plt.savefig('{}size_cdf.png'.format('WoNagle'))
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

#     with open('../DCWorkloads/' + filename + '.txt', 'r') as f:
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

# def plot_size_cdf(actual_cdf_file):
#     for traffic in actual_cdf_file:
#         x_vals, cdf_vals = read_actual_cdf(traffic)
#         plt.plot(x_vals, cdf_vals, label=traffic, linewidth=2)
    
#     plt.xscale('log')
#     plt.xlabel("Message Size(Bytes)")
#     plt.ylabel("CDF")
#     plt.title("Actual CDF of Message Sizes")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig("size_cdf.png")

# if __name__ == "__main__":
#     traffics = ["Google_AllRPC", "Fabricated_Heavy_Head", "Fabricated_Heavy_Middle", "Google_SearchRPC", "Facebook_HadoopDist_All", "FacebookKeyValue_Sampled"]
#     plot_size_cdf(traffics)

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import interp1d

# def read_packets_cdf(filename):
#     x_vals = []
#     cdf_vals = []

#     with open('../DCWorkloads/packet_size_cdf_' + filename + '.csv', 'r') as f:
#         i = 0
#         for line in f:
#             if i == 0:
#                 i += 1
#                 continue
#             x_str, cdf_str = line.strip().split(',')
#             try:
#                 x_vals.append(float(x_str))
#                 cdf_vals.append(float(cdf_str))
#             except ValueError:
#                 continue
#     return np.array(x_vals), np.array(cdf_vals)

# def plot_size_pdff(traffic):
#     x_common = np.linspace(52, 1500, 500)
#     x_mid = (x_common[:-1] + x_common[1:]) / 2
#     plt.figure(figsize=(10, 5))
#     for t in traffic:
#         x_vals, cdf_vals = read_packets_cdf(t)
#         interp_cdf = interp1d(x_vals, cdf_vals, kind='linear', bounds_error=False, fill_value=(0, 1))
#         cdf_interp = interp_cdf(x_common)
#         pdf = np.diff(cdf_interp) / np.diff(x_common)
#         plt.bar(x_mid, pdf, width=np.diff(x_common), align='center', alpha=0.7, label=t)

#     # plt.xscale('log')
#     plt.title('Histogram (PDF)')
#     plt.ylabel('Density')
#     plt.xlabel("Message Size(Bytes)")
#     plt.title("Actual PDF of Message Sizes")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig("size_pdf.png")



# traffics = ["Google_AllRPC", "Fabricated_Heavy_Head", "Fabricated_Heavy_Middle", "Google_SearchRPC", "Facebook_HadoopDist_All", "FacebookKeyValue_Sampled"]
# plot_size_pdff(traffics)