import warnings
from typing import Optional, Union, List, Tuple

import numpy as np

from . import VAMP, CovarianceKoopmanModel, KoopmanBasisTransform
from ..covariance import Covariance, CovarianceModel

try:
    import torch
    import torch.nn as nn
except (ModuleNotFoundError, ImportError):
    warnings.warn("Tried importing VampNets; this only works with a PyTorch installation!")


def sym_inverse(mat, epsilon: float = 1e-6, ret_sqrt=False):
    """ Utility function that returns the inverse of a matrix, with the
    option to return the square root of the inverse matrix.

    Parameters
    ----------
    mat: numpy array with shape [m,m]
        Matrix to be inverted.
    epsilon : float
        Cutoff for eigenvalues.
    ret_sqrt: bool, optional, default = False
        if True, the square root of the inverse matrix is returned instead

    Returns
    -------
    x_inv: numpy array with shape [m,m]
        inverse of the original matrix
    """
    # Calculate eigvalues and eigvectors
    eigval_all, eigvec_all_t = torch.symeig(mat, eigenvectors=True)

    # Filter out eigvalues below threshold and corresponding eigvectors
    mask = eigval_all > epsilon

    eigval = eigval_all[mask]
    eigvec = eigvec_all_t.transpose(0, 1)[mask]
    eigvec_t = eigvec.transpose(0, 1)

    # Build the diagonal matrix with the filtered eigenvalues or square
    # root of the filtered eigenvalues according to the parameter

    if ret_sqrt:
        diag = torch.diag(torch.sqrt(1. / eigval))
    else:
        diag = torch.diag(1. / eigval)

    return torch.chain_matmul(eigvec_t, diag, eigvec)


def koopman_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r""" Computes the Koopman matrix

    .. math:: K = C_{00}^{-1/2}C_{0t}C_{tt}^{-1/2}

    based on data over which the covariance matrices :math:`C_{\cdot\cdot}` are computed.

    Parameters
    ----------
    x : torch.Tensor
        Instantaneous data.
    y : torch.Tensor
        Time-lagged data.

    Returns
    -------
    K : torch.Tensor
        The Koopman matrix.
    """
    c00, c0t, ctt = covariances(x, y, remove_mean=True)
    c00_sqrt_inv = sym_inverse(c00, ret_sqrt=True)
    ctt_sqrt_inv = sym_inverse(ctt, ret_sqrt=True)
    return torch.chain_matmul(c00_sqrt_inv, c0t, ctt_sqrt_inv).t()


def covariances(x: torch.Tensor, y: torch.Tensor, remove_mean: bool = True):
    """Computes instantaneous and time-lagged covariances matrices.

    Parameters
    ----------
    x : (T, n) torch.Tensor
        Instantaneous data.
    y : (T, n) torch.Tensor
        Time-lagged data.
    remove_mean: bool, default=True
        Whether to remove the mean of x and y.

    Returns
    -------
    cov_00 : (n, n) torch.Tensor
        Auto-covariance matrix of x.
    cov_0t : (n, n) torch.Tensor
        Cross-covariance matrix of x and y.
    cov_tt : (n, n) torch.Tensor
        Auto-covariance matrix of y.

    See Also
    --------
    sktime.covariance.Covariance : Estimator yielding these kind of covariance matrices based on raw data.
    """

    assert x.shape == y.shape, "x and y must be of same shape"
    batch_size = x.shape[0]

    if remove_mean:
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)

    # Calculate the cross-covariance
    y_t = y.transpose(0, 1)
    x_t = x.transpose(0, 1)
    cov_01 = 1 / (batch_size - 1) * torch.matmul(x_t, y)
    # Calculate the auto-correlations
    cov_00 = 1 / (batch_size - 1) * torch.matmul(x_t, x)
    cov_11 = 1 / (batch_size - 1) * torch.matmul(y_t, y)

    return cov_00, cov_01, cov_11


def score(data: torch.Tensor, data_lagged: torch.Tensor, method='VAMP2'):
    if method not in score.valid_methods:
        raise ValueError(f"Invalid method '{method}', supported are {score.valid_methods}")
    assert data.shape == data_lagged.shape

    if method == 'VAMP1':
        koopman = koopman_matrix(data, data_lagged)
        vamp_score = torch.norm(koopman, p='nuc')
    elif method == 'VAMP2':
        koopman = koopman_matrix(data, data_lagged)
        vamp_score = torch.square(torch.norm(koopman, p='fro'))
    else:
        raise RuntimeError("This should have been caught earlier.")
    return 1 + vamp_score


score.valid_methods = ('VAMP1', 'VAMP2',)


def loss(data: torch.Tensor, data_lagged: torch.Tensor, method='VAMP2'):
    return -1. * score(data, data_lagged, method=method)


class NumPyAccessor(object):
    def __init__(self, module: nn.Module):
        self._module = module
        self._device = next(self._module.parameters()).device

    def __call__(self, data):
        with torch.no_grad():
            if data.dtype not in (np.float32, np.float64):
                raise ValueError("only supports float and double precision arrays")
            if data.dtype == np.float32:
                self._module.float()
            else:
                self._module.double()
            with torch.no_grad():
                data_tensor = torch.tensor(data, device=self._device, requires_grad=False)
                return self._module(data_tensor).cpu().numpy()


class VAMPNet(VAMP):

    def __init__(self, lagtime: int,
                 lobe: nn.Module,
                 lobe_timelagged: Optional[nn.Module] = None,
                 dim: Optional[int] = None,
                 var_cutoff: Optional[float] = None,
                 scaling: Optional[str] = None,
                 epsilon: float = 1e-6):
        super().__init__(lagtime=lagtime, dim=dim, var_cutoff=var_cutoff, scaling=scaling, epsilon=epsilon)
        self.lagtime = lagtime
        self.lobe = NumPyAccessor(lobe)
        self.lobe_timelagged = self.lobe if lobe_timelagged is None else NumPyAccessor(lobe_timelagged)

    def fit_from_timeseries(self, data: Union[np.ndarray, List[np.ndarray]], weights=None):
        data_feat = self.lobe(data)
        return super().fit_from_timeseries(data_feat, weights=weights)

    def partial_fit(self, data: Tuple[np.ndarray, np.ndarray]):
        data_feat = (self.lobe(data[0]), self.lobe_timelagged(data[1]))
        return super().partial_fit(data_feat)

    def _decompose(self, covariances: CovarianceModel):
        decomposition = self._decomposition(covariances, self.epsilon, self.scaling, self.dim, self.var_cutoff)
        return CovarianceKoopmanModel(
            operator=np.diag(decomposition.singular_values),
            basis_transform_forward=KoopmanBasisTransform(covariances.mean_0, decomposition.left_singular_vecs),
            basis_transform_backward=KoopmanBasisTransform(covariances.mean_t, decomposition.right_singular_vecs),
            feature_transform_forward=self.lobe,
            feature_transform_backward=self.lobe_timelagged,
            rank_0=decomposition.rank0, rank_t=decomposition.rankt,
            dim=self.dim, var_cutoff=self.var_cutoff, cov=covariances, scaling=self.scaling, epsilon=self.epsilon
        )
