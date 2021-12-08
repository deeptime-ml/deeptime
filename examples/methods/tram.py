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
    trajectories = sample_trajectories(bias_functions)

    # [plt.plot(xs, fn(xs), label=f'Bias {i}') for i, fn in enumerate(bias_functions)]
    # plt.hist(xs[trajectories.flatten()], bins=100, density=True, alpha=.6, color='C1',
    #          label='Histogram over visited states')
    # plt.legend()
    # plt.show()

    # move from trajectory over 100 bins back to the space of the xs: (-1.5, 1.5)
    trajectories = trajectories / 100 * 3 - 1.5

    bias_matrix = np.zeros((len(bias_centers), n_samples, len(bias_centers)))
    for i, traj in enumerate(trajectories):
        for j, bias_function in enumerate(bias_functions):
            bias_matrix[i, :, j] = bias_function(traj)


    estimator = KMeans(
        n_clusters=5,  # place 100 cluster centers
        init_strategy='uniform',  # uniform initialization strategy
        max_iter=0,  # don't actually perform the optimization, just place centers
        fixed_seed=13,
        n_jobs=8
    )

    clustering = estimator.fit_fetch(trajectories.flatten())
    dtrajs = clustering.transform(trajectories.flatten()).reshape(
        (len(bias_matrix), n_samples))



    # dtrajs = np.asarray([[1, 2, 1, 3, 2],[3, 4, 3, 3, 4]])
    # ttrajs = np.asarray([[0, 1, 0, 0, 0],[1, 1, 1, 1, 1]])
    # # ttrajs = [np.asarray([i] * len(dtrajs[i])) for i in range(len(dtrajs))]
    # bias_matrix = np.asarray([np.ones((len(dtrajs[i]), len(dtrajs))) for i in range(len(dtrajs))])
    #
    # bias_matrix[0] *= 0.5
    # bias_matrix[0][:, 0] *= 0.5
    # bias_matrix[1][:, 0] *= 0.5

    tram = TRAM(lagtime=1, connectivity="BAR_variance", maxiter=100)

    # For every simulation frame seen in trajectory i and time step t, btrajs[i][t,k] is the
    # bias energy of that frame evaluated in the k'th thermodynamic state (i.e. at the k'th
    # Umbrella/Hamiltonian/temperature).
    model = tram.fit_fetch((dtrajs, bias_matrix))

    plot_contour_with_colourbar(tram.biased_conf_energies)

    plt.plot(tram.therm_state_energies)
    plt.show()


if __name__ == "__main__":
    main()
