# import numpy as np
# # from Utils import *

# list = [
#         0.7408196721311475,
#         0.8218085106382979,
#         0.8154771853585158,
#         0.8160039926800865,
#         0.825157441166722,
#         0.7915331421824928,
#         0.7677536831650389,
#         0.7813772048846676,
#         0.8628158844765343,
#         0.7584037092233813,
#         0.8323758026360257,
#         0.7557695500581105,
#         0.7687221028269136,
#         0.7144049407444499,
#         0.8166917544739923,
#         0.7819548872180451,
#         0.864359059291156,
#         0.8265739012387578,
#         0.6616566466265865,
#         0.79986624310316,
#         0.8629073104845517,
#         0.7896417169208939,
#         0.8934357775179556,
#         0.8284616669450476,
#         0.8720442730169378,
#         0.7632061323112814,
#         0.8042213727771315,
#         0.8113176509294292,
#         0.784845496383958,
#         0.8491009914300118,
#         0.764373546028581,
#         0.8110484406104844,
#         0.7896336316056053,
#         0.8198198198198199,
#         0.7796405115796751,
#         0.785885989010989,
#         0.7975346166835529,
#         0.7739352640545145,
#         0.8054036134593071,
#         0.7768428138673847,
#         0.7354047424366312,
#         0.8247947454844007,
#         0.8785140562248996,
#         0.7831813576494427,
#         0.7140003329448976,
#         0.7985444922262653,
#         0.7994041708043694,
#         0.7614678899082569,
#         0.6750830564784054,
#         0.8023833167825223,
#         0.8088554720133667,
#         0.8208704419425307,
#         0.7742099036673991,
#         0.800033921302578,
#         0.8480927575197446,
#         0.7955599734923791,
#         0.8406117247238742,
#         0.7438678458201236,
#         0.848291646231813,
#         0.8941870261162594,
#         0.8076600033596506,
#         0.6295375435106911,
#         0.745011449133137,
#         0.9024269440316989,
#         0.7083192994274166,
#         0.7446485757626833,
#         0.7437904048996258,
#         0.7203667321545514,
#         0.8750419322375041,
#         0.8297488097192579,
#         0.7531790022823606,
#         0.7403687849851828,
#         0.7858803986710964,
#         0.8386828538167304,
#         0.7469125359499238,
#         0.7347288949897048,
#         0.8609249329758712,
#         0.7801324503311258,
#         0.847887323943662,
#         0.7350883038987004,
#         0.7423435419440746,
#         0.7975661897714192,
#         0.853510498687664,
#         0.723091976516634,
#         0.85351595122766,
#         0.8080690685704799,
#         0.7107001321003963,
#         0.8184579045048085,
#         0.7870385835977952,
#         0.7657443557969784,
#         0.7820641282565131,
#         0.7935383552176918,
#         0.7743871935967984,
#         0.7809778968519758,
#         0.7417173766058147,
#         0.7700643413477819,
#         0.7003682624707064,
#         0.7636911740077039,
#         0.7342259414225941,
#         0.7847095813337746
#     ]
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

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def read_packets_cdf(filename):
    x_vals = []
    cdf_vals = []

    with open('../DCWorkloads/packet_size_cdf_' + filename + '.csv', 'r') as f:
        i = 0
        for line in f:
            if i == 0:
                i += 1
                continue
            x_str, cdf_str = line.strip().split(',')
            try:
                x_vals.append(float(x_str))
                cdf_vals.append(float(cdf_str))
            except ValueError:
                continue
    return np.array(x_vals), np.array(cdf_vals)

def plot_size_pdff(traffic):
    x_common = np.linspace(52, 1500, 500)
    x_mid = (x_common[:-1] + x_common[1:]) / 2
    plt.figure(figsize=(10, 5))
    for t in traffic:
        x_vals, cdf_vals = read_packets_cdf(t)
        interp_cdf = interp1d(x_vals, cdf_vals, kind='linear', bounds_error=False, fill_value=(0, 1))
        cdf_interp = interp_cdf(x_common)
        pdf = np.diff(cdf_interp) / np.diff(x_common)
        plt.bar(x_mid, pdf, width=np.diff(x_common), align='center', alpha=0.7, label=t)

    # plt.xscale('log')
    plt.title('Histogram (PDF)')
    plt.ylabel('Density')
    plt.xlabel("Message Size(Bytes)")
    plt.title("Actual PDF of Message Sizes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("size_pdf.png")



traffics = ["Google_AllRPC", "Fabricated_Heavy_Head", "Fabricated_Heavy_Middle", "Google_SearchRPC", "Facebook_HadoopDist_All", "FacebookKeyValue_Sampled"]
plot_size_pdff(traffics)