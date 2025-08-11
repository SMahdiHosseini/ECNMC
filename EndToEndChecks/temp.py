import numpy as np
# from Utils import *
import matplotlib.pyplot as plt
import seaborn as sns
list =  [
        0.2343729609813389,
        0.2813299232736573,
        0.3482402424692274,
        0.3477079395085066,
        0.23863999488719884,
        0.27237458193979935,
        0.3023790746582545,
        0.26889826405698875,
        0.1925213385720092,
        0.34306441233253376,
        0.23102434612195674,
        0.3288799414348463,
        0.2591298825023817,
        0.27193918378234194,
        0.2962176509621765,
        0.30118025088055367,
        0.26491332146859137,
        0.3534831193533048,
        0.3132126921170055,
        0.2089162074325279,
        0.29195286407461085,
        0.3144622991347342,
        0.38850707479064395,
        0.3310849145811007,
        0.3015636634400596,
        0.2426740516519611,
        0.2491262776129245,
        0.2914610069101678,
        0.2748787584869059,
        0.2694695079029884,
        0.2543648785425101,
        0.2561281605867593,
        0.24202334630350195,
        0.33162415074990387,
        0.24945239015590775,
        0.2213125665601704,
        0.31514164767176817,
        0.24514877102199223,
        0.3005918650044075,
        0.3164882916230788,
        0.32369827756879826,
        0.26410878447395303,
        0.31192592592592594,
        0.3362304444996275,
        0.318646975917642,
        0.3678002977212871,
        0.35982288510836635,
        0.31835961373915167,
        0.3307465366854797,
        0.34081203007518796,
        0.26027499070977334,
        0.2385743962607115,
        0.2566469719350074,
        0.2954838709677419,
        0.2842944862948661,
        0.23418350447167938,
        0.3032110767417882,
        0.3289247454040471,
        0.2531041069723018,
        0.23185124193769727,
        0.2815494940017425,
        0.3030530906903294,
        0.22597782057241517,
        0.24691358024691357,
        0.3160791589363018,
        0.28557989517537963,
        0.28559045142361805,
        0.31317951195941046,
        0.27715736040609135,
        0.32447316249357516,
        0.28736702127659575,
        0.22964619229251612,
        0.3046789286866731,
        0.2557641098915373,
        0.30294639687610936,
        0.3339481268011527,
        0.2149731200637154,
        0.27977089627391744,
        0.3148229948886225,
        0.34639310922359134,
        0.314021067925899,
        0.32960151802656545,
        0.25891738643106743,
        0.2917059674275294,
        0.2454596517506085,
        0.3025732031943212,
        0.3289286440333401,
        0.272984863172023,
        0.3389632723252672,
        0.3007065823726292,
        0.34173197745968004,
        0.30272651810229234,
        0.3108500772797527,
        0.26103244650454,
        0.34054746494066884,
        0.28061477441745164,
        0.26801272299820217,
        0.25663716814159293,
        0.34910845139228136,
        0.33017066108064813
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