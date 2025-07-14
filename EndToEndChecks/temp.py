import numpy as np
# from Utils import *
import matplotlib.pyplot as plt
import seaborn as sns
list =   [
        82644.02215608465,
        86133.8480772547,
        83003.3365991365,
        84979.14732902347,
        82745.85269262634,
        82072.77274220032,
        83052.61482220006,
        82861.15368037135,
        83501.35043449198,
        84473.48275278999,
        83351.55829858215,
        85781.36785162287,
        85125.48526656447,
        83518.63651563284,
        84130.80087527352,
        83233.11142571618,
        82482.42741535921,
        83757.28044280443,
        83238.8001332445,
        81707.510958456,
        83620.48092369478,
        82103.23131263348,
        83000.62172035869,
        82712.86674391657,
        83144.63926788686,
        81804.4951710591,
        84268.02529084471,
        83076.09160432253,
        83221.31218781219,
        82022.49384539636,
        82670.2128680119,
        82750.97745731808,
        82726.44368996356,
        84287.48979591837,
        83320.07771847899,
        82442.3036450602,
        83594.29802543507,
        83374.68056713928,
        82681.92818136686,
        83376.26176961603,
        84253.38902953586,
        84160.93363651675,
        83517.91695906433,
        84389.20726351351,
        82692.30099337749,
        84426.95926990028,
        83311.38162720906,
        83285.25679053491,
        84332.01991897366,
        82157.66381578948,
        83485.8554961577,
        84858.30169779286,
        82807.06030483764,
        83639.42658630504,
        83077.73798836242,
        83973.93497983871,
        83239.85095753538,
        84212.35620364126,
        82703.64558847873,
        82552.20647505781,
        84431.00540906018,
        83517.45012531328,
        83935.73353494624,
        84564.49119837508,
        84134.80764695974,
        83575.1831103679,
        85010.81765606395,
        84456.62690098006,
        84326.39807627404,
        83021.59445090547,
        81708.96548339604,
        81485.61992499592,
        82687.78828008608,
        83995.63113651647,
        84479.28858833475,
        83468.47578490314,
        82791.6705882353,
        83704.98257956449,
        84237.02696780718,
        85786.4727085479,
        83462.46584265909,
        81524.22932637416,
        81694.89471963381,
        82572.86002313667,
        84898.499490316,
        81671.34052287582,
        86150.45819686262,
        83798.57193158954,
        82851.49544173712,
        82136.01988823143,
        83314.57292882146,
        81713.61471790679,
        82526.89145878077,
        81984.17588187038,
        83878.00671253566,
        82390.19274525969,
        83201.22746208965,
        82074.08492115638,
        82932.3600464576,
        83310.41470735367
    ]
print(np.average(list))
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