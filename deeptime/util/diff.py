import numpy as _np
from scipy import sparse as _sparse

from deeptime.util.data import sliding_window


def finite_difference_coefficients(x_bar, xs, k=1):
    r"""Calculates finite difference coefficients. The coefficients are computed so that

    .. math::
        f^{(k)}(\bar{x}) \approx f(x) * \mathrm{fd\_coeff}(x, \mathrm{window\_around}(\bar{x}), k).

    It is required that `len(x) > k`.

    See `here <https://epubs.siam.org/doi/10.1137/S0036144596322507>`_ for details.

    Parameters
    ----------
    x_bar : float
        The evaluation point of the derivative.
    xs : ndarray of float
        :math:`n` grid points so that :math:`x_1 \leq\ldots\leq\bar{x}\leq\ldots\leq x_n`.
    k : int, default=1
        k-th derivative.

    Returns
    -------
    weights : ndarray
        An array `w` with weights so that the dot product :math:`w \cdot f(\mathrm{xs}) \approx f^{(k)}(\bar{x})`
        is approximately the function's `k`-th derivative at :math:`\bar{x}`.
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
    r""" Yields a finite difference operator on midpoints. The midpoints are derived from the `xs` grid, first and
    last node are extrapolated.

    Parameters
    ----------
    xs : ndarray
        Grid points.
    k
    window_radius

    Returns
    -------
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
    r""" Returns matrix representation of the cumulative trapezoidal rule.
    This means that it approximates :math:`\int_0^x f` for values of `f` in the provided grid points.

    Parameters
    ----------
    xs : ndarray
        The grid.

    Returns
    -------
    operator : sparse matrix
        A `(n-1, n)`-shaped matrix where `n` is the number of grid points.
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


def tv_derivative(xs, ys, u0=None, alpha=10., tol=None, maxit=1000, fd_window_radius=5, verbose=False):
    r""" Total-variation regularized derivative. Note that this is currently only implemented for one-dimensional
    functions. See :footcite:`chartrand2011numerical` for theory and algorithmic details.

    .. plot:: examples/plot_tv_derivative.py

    Parameters
    ----------
    xs : ndarray
        Grid points.
    ys : ndarray
        Function values, must be of same length as `xs`.
    u0 : ndarray, optional, default=None
        Initial guess. May be left `None`, in which case the
        `numpy.gradient <https://numpy.org/doc/stable/reference/generated/numpy.gradient.html>`_ with `edge_order=2`
        is used.
    alpha : float, default=10
        Regularization parameter. Is required to be positive.
    tol : float, optional, default=None
        Tolerance on the relative change of the solution update. If given, the algorithm may return early.
    maxit : int, default=1000
        Maximum number of iterations before termination of the algorithm.
    fd_window_radius : int, default=5
        Radius in which the finite differences are computed. For example, a value of `2` means that the local gradient
        at :math:`x_n` is approximated using grid nodes :math:`x_{n-2}, x_{n-1}, x_n, x_{n+1}, x_{n+2}`.
    verbose : bool, default=False
        Print convergence information.

    Returns
    -------
    derivative : ndarray
        The regularized derivative values on given grid nodes.
    """
    assert alpha > 0, "Regularization parameter may only be positive."
    data = _np.asarray(ys, dtype=_np.float64).squeeze()
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
        df = _np.gradient(data, edge_order=2)
        u0 = _np.concatenate(([0], .5 * (df[1:] + df[:-1]), [0]))
    if len(u0) == n:
        u0 = _np.concatenate(([0], .5 * (u0[1:] + u0[:-1]), [0]))
    u = u0
    Aadj_offset = AT * (data[0] - data)

    E_n = _sparse.dia_matrix((n, n), dtype=xs.dtype)
    midpoints_diff = _np.diff(midpoints)

    for ii in range(1, maxit + 1):
        E_n.setdiag(midpoints_diff * (1. / _np.sqrt(_np.diff(u) ** 2.0 + epsilon)))
        L = diff_t * E_n * diff
        g = ATA.dot(u) + Aadj_offset + alpha * L * u

        # solve linear equation.
        s = _np.linalg.solve((alpha * L + ATA).todense().astype(_np.float64), -g.astype(_np.float64))

        relative_change = _np.linalg.norm(s[0]) / _np.linalg.norm(u)
        if verbose:
            print(f'iteration {ii:4d}: relative change = {relative_change:.3e},'
                  f' gradient norm = {_np.linalg.norm(g):.3e}')

        # Update current solution
        u = u + s
        if tol is not None and relative_change < tol:
            break

    return .5 * (u[1:] + u[:-1])  # project back onto grid points
