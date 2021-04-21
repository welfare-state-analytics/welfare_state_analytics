# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
# ---

# %%

# %%

import itertools

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import kmeans, whiten

# %%

# Create 50 datapoints in two clusters a and b

# %matplotlib inline

features = np.array(
    [
        [1.9, 2.3],
        [1.5, 2.5],
        [0.8, 0.6],
        [0.4, 1.8],
        [0.1, 0.1],
        [0.2, 1.8],
        [2.0, 0.5],
        [0.3, 1.5],
        [1.0, 1.0],
    ]
)

whitened = whiten(features)
book = np.array((whitened[0], whitened[2]))

kmeans(whitened, book)

pts = 50
a = np.random.multivariate_normal([0, 0], [[4, 1], [1, 4]], size=pts)
b = np.random.multivariate_normal([30, 10], [[10, 2], [2, 1]], size=pts)
features = np.concatenate((a, b))

# Whiten data
whitened = whiten(features)

# Find 2 clusters in the data
codebook, distortion = kmeans(whitened, 2)

# Plot whitened data and cluster centers in red
plt.scatter(whitened[:, 0], whitened[:, 1])
plt.scatter(codebook[:, 0], codebook[:, 1], c="r")
plt.show()

# %%


def main():
    np.random.seed(1977)
    numvars, numdata = 4, 10
    data = 10 * np.random.random((numvars, numdata))  # pylint: disable=no-member
    fig = scatterplot_matrix(
        data,
        ["mpg", "disp", "drat", "wt"],
        linestyle="none",
        marker="o",
        color="black",
        mfc="none",
    )
    fig.suptitle("Simple Scatterplot Matrix")
    plt.show()


def scatterplot_matrix(data, names, **kwargs):
    """Plots a scatterplot matrix of subplots.  Each row of "data" is plotted
    against other rows, resulting in a nrows by nrows grid of subplots with the
    diagonal subplots labeled with "names".  Additional keyword arguments are
    passed on to matplotlib's "plot" command. Returns the matplotlib figure
    object containg the subplot grid."""
    numvars, _ = data.shape
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(8, 8))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for ax in axes.flat:
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # Set up ticks only on one side for the "edge" subplots...
        if ax.is_first_col():
            ax.yaxis.set_ticks_position("left")
        if ax.is_last_col():
            ax.yaxis.set_ticks_position("right")
        if ax.is_first_row():
            ax.xaxis.set_ticks_position("top")
        if ax.is_last_row():
            ax.xaxis.set_ticks_position("bottom")

    # Plot the data.
    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        for x, y in [(i, j), (j, i)]:
            axes[x, y].plot(data[x], data[y], **kwargs)

    # Label the diagonal subplots...
    for i, label in enumerate(names):
        axes[i, i].annotate(label, (0.5, 0.5), xycoords="axes fraction", ha="center", va="center")

    # Turn on the proper x or y axes ticks.
    for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
        axes[j, i].xaxis.set_visible(True)
        axes[i, j].yaxis.set_visible(True)

    return fig


main()
