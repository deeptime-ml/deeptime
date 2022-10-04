import numpy as np

import deeptime.util.diff as diff


def test_tv_derivative():
    noise_variance = .08 * .08
    x0 = np.arange(0, 2.0 * np.pi, 0.005)
    testf = np.sin(x0) + np.random.normal(0.0, np.sqrt(noise_variance), x0.shape)
    fd = np.gradient(testf)
    true_deriv = np.cos(x0)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 15))
    plt.plot(x0, testf, label='f(x)')
    plt.plot(x0, true_deriv, label='derivative f')
    plt.plot(x0, fd, label='finite differences')
    for alpha in [.001]:
        tv_deriv0 = diff.tv_derivative(x0, testf, alpha=alpha, tol=1e-6, verbose=True, fd_window_radius=3)
        plt.plot(x0, tv_deriv0, label=f'tv deriv, alpha={alpha:.4f}')
    plt.show()
