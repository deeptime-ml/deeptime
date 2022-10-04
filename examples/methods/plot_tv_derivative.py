r"""
TV Derivative
=============

Total-variation regularized derivative on a noisy function.
"""

import numpy as np
import deeptime.util.diff as diff
import matplotlib.pyplot as plt

noise_variance = .08 * .08
x0 = np.linspace(0, 2.0 * np.pi, 200)
testf = np.sin(x0) + np.random.normal(0.0, np.sqrt(noise_variance), x0.shape)
true_deriv = np.cos(x0)
df_tv = diff.tv_derivative(x0, testf, alpha=0.01, tol=1e-5, verbose=True, fd_window_radius=5)

plt.figure()
plt.plot(x0, np.sin(x0), label=r'$f(x) = \sin(x)$')
plt.plot(x0, testf, label=r'$f(x) + \mathcal{N}(0, \sigma)$', color='C0', alpha=.5)
plt.plot(x0, true_deriv, label=r'$\frac{df}{dx}(x) = \cos(x)$')
plt.plot(x0, np.gradient(testf, x0), label='finite differences', alpha=.5)
plt.plot(x0, df_tv, label=r'$\mathrm{TV}(f(x) + \mathcal{N}(0, \sigma))$, $\alpha = 0.01$')
plt.legend()
