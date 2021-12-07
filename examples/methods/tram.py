import numpy as np
import matplotlib.pyplot as plt

from deeptime.data import tmatrix_metropolis1d
from deeptime.markov.msm import TRAM, MarkovStateModel
from deeptime.clustering import KMeans

xs = np.linspace(-1.5, 1.5, num=100)
n_samples = 10000
bias_centers = [-1, -0.5, 0.0, 0.5, 1]


def harmonic(x0, x):
    return 0.1 * (x - x0) ** 2


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
        biased_energies = 1 / 8 * (xs - 1) ** 6 * (xs + 1) ** 6 + bias(xs)

        biased_energies /= np.max(biased_energies)
        transition_matrix = tmatrix_metropolis1d(biased_energies)

        msm = MarkovStateModel(transition_matrix)
        trajs[i] = msm.simulate(n_steps=n_samples)
    return trajs


def main():
    bias_functions = get_bias_functions()

    therm_state_sequences = np.asarray([[i] * n_samples for i in range(len(bias_centers))])
    trajectories = sample_trajectories(bias_functions)

    [plt.plot(xs, fn(xs), label=f'Bias {i}') for i, fn in enumerate(bias_functions)]
    plt.hist(xs[trajectories.flatten()], bins=100, density=True, alpha=.6, color='C1',
             label='Histogram over visited states')
    plt.legend()
    plt.show()

    # move from trajectory over 100 bins back to the space of the xs: (-1.5, 1.5)
    trajectories = trajectories / 100 * 3 - 1.5

    bias_matrix = np.zeros((len(bias_centers), n_samples, len(bias_centers)))
    for i, traj in enumerate(trajectories):
        for j, bias_function in enumerate(bias_functions):
            bias_matrix[i, :, j] = bias_function(traj)

    tram = TRAM(lagtime=1, connectivity="summed_count_matrix", maxiter=100)

    estimator = KMeans(
        n_clusters=50,  # place 100 cluster centers
        init_strategy='uniform',  # uniform initialization strategy
        max_iter=0,  # don't actually perform the optimization, just place centers
        fixed_seed=13,
        n_jobs=8
    )

    clustering = estimator.fit_fetch(trajectories.flatten())
    markov_state_sequences = clustering.transform(trajectories.flatten()).reshape(
        (len(therm_state_sequences), n_samples))

    # For every simulation frame seen in trajectory i and time step t, btrajs[i][t,k] is the
    # bias energy of that frame evaluated in the k'th thermodynamic state (i.e. at the k'th
    # Umbrella/Hamiltonian/temperature).
    model = tram.fit_fetch((therm_state_sequences, markov_state_sequences, bias_matrix))

    plot_contour_with_colourbar(tram.biased_conf_energies)

    plt.plot(tram.therm_state_energies)
    plt.show()


if __name__ == "__main__":
    main()
