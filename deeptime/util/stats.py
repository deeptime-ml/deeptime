import math
from typing import Union, Callable, List, Any, Iterable

import numpy as np

from deeptime.util.types import ensure_array


def confidence_interval(data, conf=0.95):
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

        if np.any(np.isnan(x)):
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
            mean, lower[i], upper[i] = _confidence_interval_1d(col)

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
