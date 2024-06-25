import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()

X = np.linspace(0,1,1000)
Y = np.cos(X*20)

ax1.plot(X,Y)
ax1.set_xlabel(r"Original x-axis: $X$")

new_tick_locations = np.array([0.002, 0.002, 0.005, 0.009])

# def tick_function(X):
#     V = 1/(1+X)
#     return ["%.3f" % z for z in V]

ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks([i * 100 for i in new_tick_locations])
ax2.set_xticklabels(new_tick_locations)
ax2.set_xlabel(r"Modified x-axis: $1/(1+X)$")
plt.grid(True)
plt.show()
plt.savefig("temp.png")