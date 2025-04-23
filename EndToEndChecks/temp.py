import numpy as np
# from Utils import *

list =[
                143,
                168,
                154,
                158,
                167,
                168,
                145,
                157,
                167,
                167,
                154,
                153,
                138,
                165,
                159,
                165,
                164,
                152,
                157,
                155,
                152,
                133,
                152,
                169,
                159,
                157,
                168,
                157,
                167,
                153,
                156,
                168,
                156,
                154,
                152,
                120,
                141,
                172,
                157,
                162,
                157,
                154,
                157,
                132,
                157,
                164,
                153,
                152,
                149,
                152,
                152,
                158,
                166,
                158,
                153,
                157,
                160,
                143,
                171,
                169,
                164,
                157,
                161,
                160,
                153,
                153,
                153,
                157,
                158,
                158,
                155,
                141,
                158,
                153,
                120,
                169,
                138,
                131,
                148,
                166,
                164,
                166,
                156,
                133,
                158,
                164,
                163,
                166,
                155,
                158,
                138,
                141,
                168,
                160,
                155,
                157,
                165,
                156,
                165,
                154
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