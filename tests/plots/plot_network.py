import numpy as np

import matplotlib.pyplot as plt

from deeptime.markov.msm import MarkovStateModel
from deeptime.plots.network import plot_adjacency


def test_sanity():
    X = np.random.uniform(size=(5, 5))
    X /= X.sum(1)[:, None]
    msm = MarkovStateModel(X)
    positions = np.array([[-1, -1], [0, 0], [1.5, 3], [3., 1.5], [-1., 4.]])
    plot_adjacency(msm.transition_matrix, positions=positions)
    plt.show()
