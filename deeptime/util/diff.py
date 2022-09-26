import numpy as _np
from scipy import sparse as _sparse

from deeptime.util.data import sliding_window


def finite_difference_coefficients(x_bar, xs, k=1):
    """
    Calculate finite difference coefficients, so that

    f^(k)(x_bar) \approx f(xs) * fd_coeff(xs, window_around(x_bar), k).

    It is required that len(x) > k.

    See https://epubs.siam.org/doi/10.1137/S0036144596322507.
    :param x_bar: the evaluation point of the derivative
    :param xs: n values of f so that x_1 <= ... <= xbar <= ... <= x_n
    :param k: k-th derivative
    :return: a vector with weights w so that w.dot(f(xs)) \approx f^(k)(x_bar)
    """
    n = len(xs)
    if k >= n:
        raise ValueError("length(xs) = {} has to be larger than k = {}".format(len(xs), k))

    if _np.min(xs) > x_bar or _np.max(xs) < x_bar:
        raise ValueError("the grid xs has to be so that min(xs) <= xbar <= max(xs)")

    # change to m = n - 1 to compute coeffs for all derivatives, then output C
    m = k

    c1 = 1
    c4 = xs[0] - x_bar
    C = _np.zeros(shape=(n, m + 1))
    C[0, 0] = 1
    for i in range(1, n):
        mn = min(i, m)
        c2 = 1
        c5 = c4
        c4 = xs[i] - x_bar
        for j in range(0, i):
            c3 = xs[i] - xs[j]
            c2 = c2 * c3
            if j == i - 1:
                for s in range(mn, 0, -1):
                    C[i, s] = c1 * (s * C[i - 1, s - 1] - c5 * C[i - 1, s]) / c2
                C[i, 0] = -c1 * c5 * C[i - 1, 0] / c2
            for s in range(mn, 0, -1):
                C[j, s] = (c4 * C[j, s] - s * C[j, s - 1]) / c3
            C[j, 0] = c4 * C[j, 0] / c3
        c1 = c2
    c = C[:, -1]
    return c


def finite_difference_operator_midpoints(xs, k=1, window_radius=2):
    """

    :param xs:
    :param k:
    :param window_width:
    :return:
    """
    n_nodes = len(xs)
    indices = _np.arange(0, n_nodes, step=1, dtype=int)
    entries_per_row = 2 * window_radius + 1
    data = _np.empty(shape=(entries_per_row * (n_nodes-1),), dtype=xs.dtype)
    row_data = _np.empty_like(data)
    col_data = _np.empty_like(data)

    offset = 0
    for row, window in enumerate(sliding_window(indices, radius=window_radius, fixed_width=True)):
        window_grid = xs[window]
        window_slice = slice(offset, offset + len(window))
        data[window_slice] = finite_difference_coefficients(.5 * (xs[row] + xs[row + 1]), window_grid, k=k)
        row_data[window_slice] = row
        col_data[window_slice] = window

        offset += len(window)
        if row == n_nodes-2:
            break

    return _sparse.csc_matrix((data, (row_data, col_data)), shape=(n_nodes-1, n_nodes))


def _cumtrapz_operator(xs):
    """
    Returns matrix representation of the cumulative trapezoidal rule, i.e. \int_0^x f
    :param xs: grid
    :return: (n-1, n)-matrix
    """
    n = len(xs)
    data = _np.zeros(shape=(int(.5 * (n ** 2 + n) - 1),))
    row_data = _np.empty_like(data)
    col_data = _np.empty_like(data)

    current_row = _np.zeros(1)

    offset = 0
    for row in range(n - 1):
        dx = xs[row + 1] - xs[row]
        current_row[-1] += dx
        current_row = _np.append(current_row, dx)
        data[offset:offset + len(current_row)] = current_row
        row_data[offset:offset + len(current_row)] = row
        col_data[offset:offset + len(current_row)] = _np.array([range(len(current_row))])
        offset += len(current_row)
    data *= .5
    assert (len(data) == offset)
    return _sparse.csc_matrix((data, (row_data, col_data)), shape=(n - 1, n))


def tv_derivative(data, xs, u0=None, alpha=10., maxit=1000, verbose=False, fd_window_radius=5, tol=None):
    data = _np.asarray(data, dtype=_np.float64).squeeze()
    xs = _np.asarray(xs, dtype=_np.float64).squeeze()
    n = data.shape[0]
    assert xs.shape[0] == n, "the grid must have the same dimension as data"

    epsilon = 1e-6

    # grid of midpoints between xs, extrapolating first and last node:
    #
    #    x--|--x--|---x---|-x-|-x
    #
    midpoints = _np.concatenate((
        [xs[0] - .5 * (xs[1] - xs[0])],
        .5 * (xs[1:] + xs[:-1]),
        [xs[-1] + .5 * (xs[-1] - xs[-2])]
    )).squeeze()
    assert midpoints.shape[0] == n + 1

    diff = finite_difference_operator_midpoints(midpoints, k=1, window_radius=fd_window_radius)
    assert diff.shape[0] == n
    assert diff.shape[1] == n + 1

    diff_t = diff.transpose().tocsc()
    assert diff.shape[0] == n
    assert diff.shape[1] == n + 1

    A = _cumtrapz_operator(midpoints)
    AT = A.T
    ATA = AT @ A

    if u0 is None:
        u = _np.concatenate(([0], _np.diff(data), [0]))
    else:
        u = u0
    Aadj_offset = AT * (data[0] - data)

    E_n = _sparse.dia_matrix((n, n), dtype=xs.dtype)
    midpoints_diff = _np.diff(midpoints)

    for ii in range(1, maxit + 1):
        E_n.setdiag(midpoints_diff * (1. / _np.sqrt(_np.diff(u) ** 2.0 + epsilon)))
        L = diff_t * E_n * diff
        g = ATA.dot(u) + Aadj_offset + alpha * L * u

        # solve linear equation.
        s = _np.linalg.solve((alpha * L + ATA).todense().astype(_np.float64), (-g).astype(_np.float64))

        relative_change = _np.linalg.norm(s[0]) / _np.linalg.norm(u)
        if verbose:
            print(f'iteration {ii:4d}: relative change = {relative_change:.3e},'
                  f' gradient norm = {_np.linalg.norm(g):.3e}')

        # Update current solution
        u = u + s
        if tol is not None and relative_change < tol:
            break

    return u
