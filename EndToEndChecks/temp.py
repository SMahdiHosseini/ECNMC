# import numpy as np
# from Utils import *

# list =[
#         58726.0,
#         57869.0,
#         59760.0,
#         53101.0,
#         55998.0,
#         56295.0,
#         52586.0,
#         57323.0,
#         52778.0,
#         54067.0,
#         54025.0,
#         56667.0,
#         55818.0,
#         54449.0,
#         45940.0,
#         56718.0,
#         48120.0,
#         50947.0,
#         54076.0,
#         53534.0,
#         52471.0,
#         52639.0,
#         55321.0,
#         54849.0,
#         52172.0,
#         52726.0,
#         56785.0,
#         52430.0,
#         54954.0,
#         52512.0
#     ]
# print(np.average(list))
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns

# # Example data
# import matplotlib.pyplot as plt
import numpy as np
print(np.random.binomial(n=1, p=0.99, size=100))

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