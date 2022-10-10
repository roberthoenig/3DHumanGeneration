import matplotlib.pyplot as plt
import numpy as np

data = np.load("distance.npy")


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')


fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, figsize=(9, 4), sharey=True)


ax1.set_title("Example 1")
ax2.set_title("Example 2")
ax3.set_title("Example 3")
ax4.set_title("Example 4")
ax5.set_title("Example 5")


ax1.set_ylabel("Mean pairwise euclidian distance")

ax1.violinplot(data[0, :])
ax2.violinplot(data[1, :])
ax3.violinplot(data[2, :])
ax4.violinplot(data[3, :])
ax5.violinplot(data[4, :])

plt.subplots_adjust(bottom=0.15, wspace=0.05)
plt.show()
plt.savefig()