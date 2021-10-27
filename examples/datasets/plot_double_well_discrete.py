r"""
Double-well discrete
====================

Showcase use of the :meth:`deeptime.data.double_well_discrete` dataset.
"""
import matplotlib.pyplot as plt
import numpy as np

from deeptime.data import double_well_discrete

dwd = double_well_discrete()
n_states = dwd.analytic_msm.n_states

divides = [40, 45, 50, 55, 60]
dtraj = dwd.dtraj_n(divides)
divides = np.array([0] + divides + [n_states])
f, ax = plt.subplots(1, 1)
f.suptitle("Discrete double well with good\ndiscretization of transition region")
ax.hist(divides[dtraj], divides, density=True, alpha=.5, color='C0', edgecolor='black', label='Empirical distribution')
ax.bar(np.arange(n_states), dwd.analytic_msm.stationary_distribution, color='C1', alpha=.5,
       label='Stationary distribution')
plt.legend()
plt.show()
