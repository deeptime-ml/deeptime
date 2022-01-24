"""
TRAM on a 1D double well
========================

This example shows how to use the transition-based reweighting analysis method (TRAM) to estimate the free energies
and Markov model of a simple double-well potential, sampled using umbrella sampling.

For more information see the :class:`TRAM <deeptime.markov.msm.tram.TRAM>` estimator and
its respective `TRAM tutorial <../notebooks/tram.ipynb>`__.
"""

import numpy as np
import matplotlib.pyplot as plt

from deeptime.data import tmatrix_metropolis1d
from deeptime.markov.msm import MarkovStateModel, TRAM
from deeptime.clustering import ClusterModel

xs = np.linspace(-1.5, 1.5, num=100)
n_samples = 10000
bias_centers = [-1, -0.5, 0.0, 0.5, 1]


def harmonic(x0, x):
    return 2 * (x - x0) ** 4


def plot_contour_with_colourbar(data, vmin=None, vmax=None):
    if vmin is None:
        vmin = np.min(data)
    if vmax is None:
        vmax = np.max(data)
    fig, (ax1) = plt.subplots(1, figsize=(3.5, 3))
    im = ax1.contourf(data, vmin=vmin, vmax=vmax, levels=50, cmap='jet')
    plt.colorbar(im)
    plt.show()


def get_bias_functions():
    bias_functions = []
    for i, bias_center in enumerate(bias_centers):
        bias = lambda x, x0=bias_center: harmonic(x0, x)
        bias_functions.append(bias)
    return bias_functions


def sample_trajectories(bias_functions):
    trajs = np.zeros((len(bias_centers), n_samples), dtype=np.int32)

    for i, bias in enumerate(bias_functions):
        biased_energies = (xs - 1) ** 4 * (xs + 1) ** 4 - 0.1 * xs + bias(xs)

        biased_energies /= np.max(biased_energies)
        transition_matrix = tmatrix_metropolis1d(biased_energies)

        msm = MarkovStateModel(transition_matrix)
        trajs[i] = msm.simulate(n_steps=n_samples)
    return trajs


if __name__ == "__main__":
    bias_functions = get_bias_functions()
    trajectories = sample_trajectories(bias_functions)

    # move from trajectory over 100 bins back to the space of the xs: (-1.5, 1.5)
    trajectories = trajectories / 100 * 3 - 1.5

    bias_matrices = np.zeros((len(bias_centers), n_samples, len(bias_centers)))
    for i, traj in enumerate(trajectories):
        for j, bias_function in enumerate(bias_functions):
            bias_matrices[i, :, j] = bias_function(traj)

    # discretize the trajectories into two Markov states (centered around the two wells)
    clustering = ClusterModel(cluster_centers=np.asarray([-0.75, 0.75]), metric='euclidean')

    dtrajs = clustering.transform(trajectories.flatten()).reshape((len(bias_matrices), n_samples))

    from tqdm import tqdm
    tram = TRAM(lagtime=1, maxiter=100, maxerr=1e-2, progress=tqdm)

    # For every simulation frame seen in trajectory i and time step t, btrajs[i][t,k] is the
    # bias energy of that frame evaluated in the k'th thermodynamic state (i.e. at the k'th
    # Umbrella/Hamiltonian/temperature).
    model = tram.fit_fetch((dtrajs, bias_matrices))

    plot_contour_with_colourbar(model.biased_conf_energies)
