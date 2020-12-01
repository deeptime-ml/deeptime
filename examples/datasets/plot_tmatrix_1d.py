r"""
Metropolis chain in 1D energy landscape
=======================================

Example for :meth:`deeptime.data.tmatrix_metropolis1d`.
"""
import matplotlib.pyplot as plt
import numpy as np
import deeptime as dt

xs = np.linspace(-1.5, 1.5, num=100)
energies = 1/8 * (xs-1)**2 * (xs+1)**2
energies /= np.max(energies)
transition_matrix = dt.data.tmatrix_metropolis1d(energies)
msm = dt.markov.msm.MarkovStateModel(transition_matrix)

traj = msm.simulate(n_steps=1000000)

plt.plot(xs, energies, color='C0', label='Energy')
plt.plot(xs, energies, marker='x', color='C0')
plt.hist(xs[traj], bins=100, density=True, alpha=.6, color='C1', label='Histogram over visited states')
plt.legend()
plt.show()
