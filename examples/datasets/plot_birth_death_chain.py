r"""
Birth-death chain model
=======================

Example for the :meth:`deeptime.data.birth_death_chain` model.
"""

import numpy as np
import matplotlib.pyplot as plt

from deeptime.data import birth_death_chain

n_states = 7
b = 2
q = np.zeros(n_states)
p = np.zeros(n_states)
q[1:] = 0.5
p[0:-1] = 0.5
q[2] = 1.0 - 10 ** (-b)
q[4] = 10 ** (-b)
p[2] = 10 ** (-b)
p[4] = 1.0 - 10 ** (-b)

bd = birth_death_chain(q, p)
dtraj = bd.msm.simulate(100000)

bins = np.arange(0, dtraj.max() + 1.5) - 0.5

fig, ax = plt.subplots()
ax.set_xticks(bins + 0.5)
ax.vlines(bins, ymin=0, ymax=.3, zorder=1, color='black', linestyles='dashed')
ax.hist(dtraj, bins, density=True, alpha=.5, color='C0', label='Empirical distribution')
ax.bar(np.arange(n_states), bd.stationary_distribution, color='C1', alpha=.5, label='Stationary distribution')
ax.set_xlabel('State')
ax.set_ylabel('State population')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=1, fancybox=True, shadow=True)
plt.show()
