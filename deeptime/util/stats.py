import math
from typing import Union, Callable, List, Any, Iterable

import numpy as np

from deeptime.util.decorators import plotting_function
from deeptime.util.types import ensure_array


def confidence_interval(data, conf=0.95, remove_nans=False):
    r""" Computes element-wise confidence intervals from a sample of ndarrays

    Given a sample of arbitrarily shaped ndarrays, computes element-wise
    confidence intervals

    Parameters
    ----------
    data : array-like of dimension 1 and 2
        array of numbers or arrays. The first index is used as the sample
        index, the remaining indexes are specific to the array of interest
    conf : float, optional, default = 0.95
        confidence interval
    remove_nans : bool, optional, default=False
        The default leads to a `np.nan` result if there are any `nan` values in `data`. If set to `True`, the
        `np.nan` values are ignored.

    Return
    ------
    lower : ndarray(shape)
        element-wise lower bounds
    upper : ndarray(shape)
        element-wise upper bounds

    """
    if conf < 0 or conf > 1:
        raise ValueError(f'Not a meaningful confidence level: {conf}')

    data = ensure_array(data)

    def _confidence_interval_1d(x):
        """
        Computes the mean and alpha-confidence interval of the given sample set

        Parameters
        ----------
        x : ndarray
            a 1D-array of samples

        Returns
        -------
        (m, l, r) : m is the mean of the data, and (l, r) are the m-alpha/2
            and m+alpha/2 confidence interval boundaries.
        """
        assert x.ndim == 1, x.ndim

        nan = np.isnan(x)
        if remove_nans:
            x = x[np.where(~nan)]
            if len(x) == 0:
                return np.nan, np.nan, np.nan
        else:
            if np.any(nan):
                return np.nan, np.nan, np.nan

        d_min, d_max = np.min(x), np.max(x)

        if np.isclose(d_min, d_max):
            return d_min, d_min, d_max

        m = np.mean(x)
        x = np.sort(x)

        # index of the mean
        im = np.searchsorted(x, m)
        if im == 0 or im == len(x) or (np.isinf(m - x[im - 1]) and np.isinf(x[im] - x[im - 1])):
            pm = im
        else:
            pm = (im - 1) + (m - x[im - 1]) / (x[im] - x[im - 1])
        # left interval boundary
        pl = pm - conf * pm
        left_boundary = _compute_interval_boundary(pl, x)
        # right interval boundary
        pr = pm + conf * (len(x) - im)
        right_boundary = _compute_interval_boundary(pr, x)
        return m, left_boundary, right_boundary

    def _compute_interval_boundary(p, x):
        i1 = max(0, int(math.floor(p)))
        i2 = min(len(x) - 1, int(math.ceil(p)))
        if np.isclose(x[i1], x[i2]):  # catch infs
            return x[i1]
        else:
            return x[i1] + (p - i1) * (x[i2] - x[i1])

    if conf == 1.:
        return np.min(data, axis=0), np.max(data, axis=0)

    if data.ndim == 1:
        mean, lower, upper = _confidence_interval_1d(data)
        return lower, upper
    else:
        lower = np.zeros_like(data[0])
        upper = np.zeros_like(data[0])

        # compute interval for each column

        def _column(arr, indexes):
            """ Returns a column with given indexes from a deep array

            For example, if the array is a matrix and indexes is a single int, will
            return arr[:,indexes]. If the array is an order 3 tensor and indexes is a
            pair of ints, will return arr[:,indexes[0],indexes[1]], etc.

            """
            if arr.ndim == 2 and isinstance(indexes, (int, tuple)):
                if isinstance(indexes, tuple):
                    indexes = indexes[0]
                return arr[:, indexes]
            elif arr.ndim == 3 and len(indexes) == 2:
                return arr[:, indexes[0], indexes[1]]
            else:
                raise NotImplementedError('Only supporting arrays of dimension 2 and 3 as yet.')

        for i in np.ndindex(data[0].shape):
            col = _column(data, i)
            _, lower[i], upper[i] = _confidence_interval_1d(col)

        return lower, upper


def call_member(obj, f: Union[str, Callable], *args, **kwargs):
    """ Calls the specified method, property or attribute of the given object

    Parameters
    ----------
    obj : object
        The object that will be used
    f : str or function
        Name of or reference to method, property or attribute
    *args : list
        list of arguments to pass to f during evaluation
    ** kwargs: dict
        keyword arguments to pass to f during evaluation
    """
    import inspect
    # get function name
    if not isinstance(f, str):
        fname = f.__func__.__name__
    else:
        fname = f
    # get the method ref
    method = getattr(obj, fname)
    # handle cases
    if inspect.ismethod(method):
        return method(*args, **kwargs)

    # attribute or property
    return method


def evaluate_samples(samples: Iterable[Any], quantity: str, delimiter: str = '/', *args, **kwargs) -> List[Any]:
    r"""Evaluate a quantity (like a property, function, attribute) of an iterable of objects and return the result.

    Parameters
    ----------
    samples : list of object
        The samples which contain sought after quantities.
    quantity : str, optional, default=None
        Name of attribute, which will be evaluated on samples. If None, no quantity is evaluated and the samples
        are assumed to already be the quantity that is to be evaluated.
    delimiter : str, optional, default='/'
        Separator to call members of members.
    *args
        pass through
    **kwargs
        pass through

    Returns
    -------
    result : list of any or ndarray
        The collected data, if it can be converted to numpy array then numpy array.
    """
    if quantity is not None and delimiter in quantity:
        qs = quantity.split(delimiter)
        quantity = qs[-1]
        for q in qs[:-1]:
            samples = [call_member(s, q) for s in samples]
    if quantity is not None:
        samples = [call_member(s, quantity, *args, **kwargs) for s in samples]
    try:
        samples = np.asfarray(samples)
    except:
        pass
    return samples


class QuantityStatistics:
    """ Container for statistical quantities computed on samples.

    Parameters
    ----------
    samples: list of ndarrays
        the samples
    store_samples: bool, default=False
        whether to store the samples (array).
    """

    @property
    def R(self):
        r"""Element-wise upper bounds.

        :type: ndarray
        """
        return self._R

    @property
    def L(self):
        r""" Element-wise lower bounds.

        :type: ndarray
        """
        return self._L

    @property
    def std(self):
        r""" Standard deviation along axis=0.

        :type: (n,) ndarray
        """
        return self._std

    @property
    def mean(self):
        r""" Mean along axis=0

        :type: (n,) ndarray
        """
        return self._mean

    @staticmethod
    def gather(samples, quantity=None, store_samples=False, delimiter='/', confidence: float = 0.95, *args, **kwargs):
        r"""Obtain statistics about a sampled quantity. Can also be a chained call, separated by the delimiter.

        Parameters
        ----------
        samples : list of object
            The samples which contain sought after quantities.
        quantity : str, optional, default=None
            Name of attribute, which will be evaluated on samples. If None, no quantity is evaluated and the samples
            are assumed to already be the quantity that is to be evaluated.
        store_samples : bool, optional, default=False
            Whether to store the samples (array).
        delimiter : str, optional, default='/'
            Separator to call members of members.
        confidence : float, optional, default=0.95
            Confidence parameter for the confidence intervals.
        *args
            pass through
        **kwargs
            pass through

        Returns
        -------
        statistics : QuantityStatistics
            The collected statistics.
        """
        if quantity is not None:
            samples = evaluate_samples(samples, quantity=quantity, delimiter=delimiter, *args, **kwargs)
        return QuantityStatistics(samples, quantity=quantity, store_samples=store_samples, confidence=confidence)

    def __init__(self, samples: List[np.ndarray], quantity, confidence=0.95, store_samples=False):
        super().__init__()
        self.quantity = quantity
        samples = np.array(samples)
        if store_samples:
            self.samples = samples
        else:
            self.samples = np.empty(0)
        self._mean = samples.mean(axis=0)
        self._std = samples.std(axis=0)
        self._L, self._R = confidence_interval(samples, conf=confidence)

    def __str__(self):
        return "QuantityStatistics(mean={}, std={})".format(self.mean, self.std)


def _maxlength(X):
    """ Returns the maximum length of signal trajectories X """
    return np.fromiter((map(lambda x: len(x), X)), dtype=int).max()


def statistical_inefficiency(X, truncate_acf=True, mact=1.0):
    r""" Estimates the statistical inefficiency from univariate time series X

    The statistical inefficiency [1]_ is a measure of the correlatedness of samples in a signal.
    Given a signal :math:`{x_t}` with :math:`N` samples and statistical inefficiency :math:`I \in (0,1]`, there are
    only :math:`I \cdot N` effective or uncorrelated samples in the signal. This means that :math:`I \cdot N` should
    be used in order to compute statistical uncertainties. See [2]_ for a review.

    The statistical inefficiency is computed as :math:`I = (2 \tau)^{-1}` using the damped autocorrelation time

    .. math:
        \tau = \frac{1}{2}+\sum_{K=1}^{N} A(k) \left(1-\frac{k}{N}\right)

    where

    .. math:
        A(k) = \frac{\langle x_t x_{t+k} \rangle_t - \langle x^2 \rangle_t}{\mathrm{var}(x)}

    is the autocorrelation function of the signal :math:`{x_t}`, which is computed either for a single or multiple
    trajectories.

    Parameters
    ----------
    X : float array or list of float arrays
        Univariate time series (single or multiple trajectories)
    truncate_acf : bool, optional, default=True
        When the normalized autocorrelation function passes through 0, it is truncated in order to avoid integrating
        random noise

    References
    ----------
    .. [1] Anderson, T. W.: The Statistical Analysis of Time Series (Wiley, New York, 1971)

    .. [2] Janke, W: Statistical Analysis of Simulations: Data Correlations and Error Estimation
        Quantum Simulations of Complex Many-Body Systems: From Theory to Algorithms, Lecture Notes,
        J. Grotendorst, D. Marx, A. Muramatsu (Eds.), John von Neumann Institute for Computing, Juelich
        NIC Series 10, pp. 423-445, 2002.

    """
    # check input
    assert np.ndim(X[0]) == 1, 'Data must be 1-dimensional'
    N = _maxlength(X)  # max length
    # mean-free data
    xflat = np.concatenate(X)
    Xmean = np.mean(xflat)
    X0 = [x - Xmean for x in X]
    # moments
    x2m = np.mean(xflat ** 2)
    del xflat, Xmean
    # integrate damped autocorrelation
    corrsum = 0.0
    for lag in range(N):
        acf = 0.0
        n = 0
        # cache partial sums
        for x in X0:
            Nx = len(x)  # length of this trajectory
            if Nx > lag:  # only use trajectories that are long enough
                prod = x[:Nx - lag] * x[lag:]
                acf += np.sum(prod)
                n += Nx - lag
        acf /= float(n)
        if acf <= 0 and truncate_acf:  # zero autocorrelation. Exit
            break
        elif lag > 0:  # start integrating at lag 1 (effect of lag 0 is contained in the 0.5 below
            corrsum += acf * (1.0 - (float(lag) / float(N)))
    # compute damped correlation time
    corrtime = 0.5 + mact * corrsum / x2m
    # return statistical inefficiency
    return 1.0 / (2 * corrtime)


class EnergyLandscape2d:
    r""" Result of the :meth:`energy2d` method.

    Parameters
    ----------
    x_meshgrid : ndarray
        `(n_bins_x,)`-shaped array with `x` coordinates for the energy landscape.
    y_meshgrid : ndarray
        `(n_bins_y,)`-shaped array with `y` coordinates for the energy landscape.
    energies : ndarray
        `(n_bins_y, n_bins_x)`-shaped array with the estimated energies.
    kbt : float
        The value of :math:`k_BT` in the desired energy unit.

    See Also
    --------
    energy2d, deeptime.plots.plot_energy2d
    """

    def __init__(self, x_meshgrid, y_meshgrid, energies, kbt):
        self.x_meshgrid = x_meshgrid
        self.y_meshgrid = y_meshgrid
        self.energies = energies
        self.kbt = kbt

    @plotting_function()
    def plot(self, ax=None, levels=100, contourf_kws=None, cbar=True,
             cbar_kws=None, cbar_ax=None):
        r""" Plot estimated energy landscape directly. See :func:`deeptime.plots.plot_energy2d`. """
        from deeptime.plots import plot_energy2d
        return plot_energy2d(self, ax=ax, levels=levels, contourf_kws=contourf_kws, cbar=cbar,
                             cbar_kws=cbar_kws, cbar_ax=cbar_ax)


def histogram2d_from_xy(x: np.ndarray, y: np.ndarray, bins=100, weights=None, density=True):
    r"""Computes a histogram from unordered (x, y) pairs. Can optionally apply a
    Parameters
    ----------
    x : ndarray
        Sample x coordinates of shape `(N,)`.
    y : ndarray
        Sample y coordinates of shape `(N,)`.
    bins : int or [int, int], optional, default=100
        Number of histogram bins used in each dimension.
    weights : ndarray, optional, default=None
        Sample weights of shape `(N,)`. By default, all samples have the same weight.
    density : bool, default=True
        Whether to normalize the histogram, producing a discrete probability density.

    Returns
    -------
    x_meshgrid : ndarray
        `(n_bins_x,)`-shaped array with `x` coordinates for the histogram.
    y_meshgrid : ndarray
        `(n_bins_y,)`-shaped array with `y` coordinates for the histogram.
    histogram : ndarray
        `(n_bins_y, n_bins_x)`-shaped array with the estimated histogram values.
    """
    hist, x_edges, y_edges = np.histogram2d(x, y, bins=bins, weights=weights)
    x_meshgrid = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_meshgrid = 0.5 * (y_edges[:-1] + y_edges[1:])

    if density:
        hist /= np.sum(hist)
    return x_meshgrid, y_meshgrid, hist.T


def energy2d(x: np.ndarray, y: np.ndarray, bins=100, kbt: float = 1., weights=None, shift_energy=True):
    r""" Compute a two-dimensional energy landscape based on data arrays `x` and `y`.

    .. plot:: examples/plot_energy_surface.py

    This function assumes that the sampled data follows a Boltzmann distribution

    .. math::
        p(x) \propto e^{-E(x) / k_BT},

    which is a probability distribution over states :math:`x` of a system based on their energy :math:`E(x)`.
    Based on data we estimate :math:`p(x)` as a normalized histogram and subsequently compute
    :math:`E(x)/k_BT = -\log p(x) + C` where :math:`C` is a constant depending on the partition function.

    If possible it is strongly encouraged to set the weights according to a stationary distribution
    (see :meth:`MSM.compute_trajectory_weights <deeptime.markov.msm.MarkovStateModel.compute_trajectory_weights>`).
    Otherwise, the energy landscape may be biased due to finite sampling.

    Parameters
    ----------
    x : ndarray
        Sample x coordinates of shape `(N,)`.
    y : ndarray
        Sample y coordinates of shape `(N,)`.
    bins : int or [int, int], optional, default=100
        Number of histogram bins used in each dimension.
    kbt : float, optional, default=1
        The value of :math:`k_BT` in the desired energy unit. By default, energies are computed in :math:`k_BT`
        (setting `kbt=1.0`). If you want to measure the energy in :math:`\mathrm{kJ}/\mathrm{mol}` at
        :math:`298\;K`, use `kbt=2.479`.
    weights : ndarray, optional, default=None
        Sample weights of shape `(N,)`. By default, all samples have the same weight.
    shift_energy : bool, optional, default=True
        Whether to shift the minimum energy to zero. Defaults to `True`.

    Returns
    -------
    energy_landscape : EnergyLandscape2d
        The estimated energy landscape.

    See Also
    --------
    deeptime.plots.plot_energy2d, EnergyLandscape2d
    """
    x_meshgrid, y_meshgrid, hist = histogram2d_from_xy(x, y, bins=bins, weights=weights)
    energy = np.full_like(hist, fill_value=np.inf)

    nonzero = hist.nonzero()
    energy[nonzero] = -np.log(hist[nonzero])
    if shift_energy:
        energy[nonzero] -= np.min(energy[nonzero])

    energy *= kbt
    return EnergyLandscape2d(x_meshgrid, y_meshgrid, energy, kbt)
