import numpy as np
# from Utils import *
import matplotlib.pyplot as plt
import seaborn as sns
list =[
        12371.982938229263,
        12793.40508033142,
        4898.096906295302,
        5366.086086569313,
        9313.06806755037,
        7585.11289536145,
        8022.5450406667915,
        62736.414561057216,
        4955.333853631988,
        6251.365802535317,
        45901.47195818339,
        11971.925131047283,
        31591.881168117834,
        3773.243401311544,
        9089.72925800958,
        8050.111532005178,
        197410.57279298588,
        4316.982801851992,
        116315.0580569094,
        4605.21637259923,
        10213.959887093662,
        120739.71683727721,
        10704.910546890222,
        14800.149370151983,
        6146.531093013375,
        4755.622287389111,
        26729.09884828885,
        75338.30117250967,
        46151.54848180795,
        18283.07409294039,
        5593.134585729565,
        7563.211133343782,
        6000.357334069775,
        40846.62763783416,
        26281.081143925825,
        49335.58377614336,
        12762.89877410881,
        79387.71406044479,
        25862.132100452367,
        20332.202518686194,
        70917.91809298896,
        16790.171850887535,
        6141.537761105878,
        55293.7912605721,
        12019.764191562075,
        13438.805281492223,
        128571.56544109917,
        9068.342044692032,
        44921.49100570947,
        117868.24076104174,
        4084.206724326368,
        4439.463212491382,
        7440.625297374018,
        36151.340514014635,
        4787.967423599996,
        6593.468813327937,
        38506.692587356476,
        118783.5965827135,
        16848.06171408086,
        5338.205971365919,
        14215.370309402813,
        30916.176078004828,
        5378.1334924417315,
        16631.70479965408,
        103386.35585468265,
        15535.176770174114,
        7155.22671193432,
        113039.13835461346,
        10669.69284658025,
        50207.67022092103,
        10965.998370850586,
        6263.7709221628675,
        78711.14004528597,
        12922.17157791715,
        6859.316494752466,
        6269.83914857575,
        10255.39269931825,
        31906.776798732353,
        6423.88675977102,
        10638.947953115177,
        4946.279149505372,
        36736.97458366549,
        29334.36893056281,
        5762.008809301824,
        95087.50792553263,
        77162.10369245005,
        16134.129796573105,
        17838.336909698974,
        35092.38975872882,
        5444.4191578380605,
        7202.508352195181,
        120932.37015903201,
        43190.92425641296,
        110567.04248682206,
        4786.281840613913,
        36357.84898144275,
        6394.401648408553,
        113460.85574677946,
        138216.12463585768,
        64718.32397408871
    ]
print(np.average(list))
# print(np.average([x[0] for x in list]))
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