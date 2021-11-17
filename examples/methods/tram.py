import numpy as np
import matplotlib.pyplot as plt

from deeptime.data import tmatrix_metropolis1d
from deeptime.markov.msm import MarkovStateModel
from deeptime.markov.msm import TRAM

xs = np.linspace(-1.5, 1.5, num=100)


def harmonic(x0, x):
    return 0.1 * (x - x0) ** 2


n_samples = 10000
bias_centers = [-1, -0.5, 0.0, 0.5, 1]

ttrajs = np.asarray([[i] * n_samples for i in range(len(bias_centers))])
trajs = np.zeros((len(bias_centers), n_samples))
bias_matrix = np.zeros((len(bias_centers), n_samples, len(bias_centers)))

bias_functions = []

for i, bias_center in enumerate(bias_centers):
    bias = lambda x, x0=bias_center: harmonic(x0, x)
        # return harmonic(x0=bias_center, x=x)

    bias_functions.append(bias)

    biased_energies = 1 / 8 * (xs - 1) ** 4 * (xs + 1) ** 4 + bias(xs)

    biased_energies /= np.max(biased_energies)
    transition_matrix = tmatrix_metropolis1d(biased_energies)
    msm = MarkovStateModel(transition_matrix)

    traj = msm.simulate(n_steps=n_samples)
    trajs[i] = traj/100 * 3 - 1.5
    # plt.plot(xs, biased_energies, color='C0', label='Energy')
    # plt.plot(xs, biased_energies, marker='x', color='C0')
    # plt.hist(xs[trajs[i]], bins=100, density=True, alpha=.6, color='C1', label='Histogram over visited states')
    # plt.legend()
    # plt.show()

for i, traj in enumerate(trajs):
    for j, bias_function in enumerate(bias_functions):
        bias_matrix[i, :, j] = bias_function(traj)

tram = TRAM(lagtime=1, connectivity="summed_count_matrix", maxiter=100)

from deeptime.clustering import KMeans

estimator = KMeans(
    n_clusters=2,  # place 100 cluster centers
    init_strategy='uniform',  # uniform initialization strategy
    max_iter=0,  # don't actually perform the optimization, just place centers
    fixed_seed=13,
    n_jobs=8
)

clustering = estimator.fit_fetch(trajs.flatten())
dtrajs = clustering.transform(trajs.flatten()).reshape((len(ttrajs), n_samples))

# For every simulation frame seen in trajectory i and time step t, btrajs[i][t,k] is the
# bias energy of that frame evaluated in the k'th thermodynamic state (i.e. at the k'th
# Umbrella/Hamiltonian/temperature).
tram.fit_fetch((ttrajs, dtrajs, bias_matrix))


def plot_contour_with_colourbar(data, vmin=None, vmax=None):
    if vmin is None:
        vmin = np.min(data)
    if vmax is None:
        vmax = np.max(data)
    fig, (ax1) = plt.subplots(1, figsize=(3.5, 3))
    im = ax1.contourf(data, vmin=vmin, vmax=vmax, levels=50, cmap='jet')
    plt.colorbar(im)
    plt.show()

plot_contour_with_colourbar(tram.biased_conf_energies)
plt.plot(tram.therm_energies)
plt.show()