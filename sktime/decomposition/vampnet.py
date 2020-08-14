import logging
import warnings
from typing import Optional, Union, List, Callable

import numpy as np

from . import VAMP
from ..base import Transformer, Model, Estimator
from ..data.util import TimeSeriesDataSet

try:
    import torch
    import torch.nn as nn
except (ModuleNotFoundError, ImportError):
    warnings.warn("Tried importing VampNets; this only works with a PyTorch installation!")

logger = logging.getLogger(__name__)


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


valid_score_methods = ('VAMP1', 'VAMP2',)


def score(data: torch.Tensor, data_lagged: torch.Tensor, method='VAMP2'):
    if method not in valid_score_methods:
        raise ValueError(f"Invalid method '{method}', supported are {valid_score_methods}")
    assert data.shape == data_lagged.shape

    if method == 'VAMP1':
        koopman = koopman_matrix(data, data_lagged)
        vamp_score = torch.norm(koopman, p='nuc')
    elif method == 'VAMP2':
        koopman = koopman_matrix(data, data_lagged)
        vamp_score = torch.pow(torch.norm(koopman, p='fro'), 2)
    else:
        raise RuntimeError("This should have been caught earlier.")
    return 1 + vamp_score


def loss(data: torch.Tensor, data_lagged: torch.Tensor, method='VAMP2'):
    return -1. * score(data, data_lagged, method=method)


class MLPLobe(nn.Module):

    def __init__(self, units: List, nonlinearity=nn.ELU, initial_batchnorm: bool = True,
                 output_nonlinearity=lambda: nn.Softmax(dim=1)):
        super().__init__()
        layers = []
        if initial_batchnorm:
            layers.append(nn.BatchNorm1d(units[0]))
        for fan_in, fan_out in zip(units[:-2], units[1:-1]):
            layers.append(nn.Linear(fan_in, fan_out))
            layers.append(nonlinearity())
        layers.append(nn.Linear(units[-2], units[-1]))
        if output_nonlinearity is not None:
            layers.append(output_nonlinearity())
        self._sequential = nn.Sequential(*layers)

    def forward(self, inputs):
        return self._sequential(inputs)


class VAMPNetModel(Model, Transformer):

    def __init__(self, lobe: nn.Module, lobe_timelagged: Optional[nn.Module] = None,
                 dtype=np.float32, device=None, train_scores=None, validation_scores=None):
        super().__init__()
        self._lobe = lobe
        self._lobe_timelagged = lobe_timelagged if lobe_timelagged is not None else lobe

        self._device = device
        self._dtype = dtype

        if self._dtype == np.float32:
            self._lobe = self._lobe.float()
            self._lobe_timelagged = self._lobe_timelagged.float()
        elif self._dtype == np.float64:
            self._lobe = self._lobe.double()
            self._lobe_timelagged = self._lobe_timelagged.double()
        else:
            raise ValueError(f"Unsupported type {dtype}! Only float32 and float64 are allowed.")

        self.train_scores = train_scores
        self.validation_scores = validation_scores

    def transform(self, data, **kwargs):
        self._lobe.eval()
        self._lobe_timelagged.eval()

        with torch.no_grad():
            if not isinstance(data, (list, tuple)):
                data = [data]
            out = []
            for x in data:
                if isinstance(x, torch.Tensor):
                    x = x.to(device=self._device)
                else:
                    x = torch.from_numpy(np.asarray(x, dtype=self._dtype)).to(device=self._device)
                out.append(self._lobe(x).cpu().numpy())
        if isinstance(out, (list, tuple)) and len(out) == 1:
            out = out[0]
        return out


class VAMPNet(Estimator, Transformer):
    _MUTABLE_INPUT_DATA = True

    def __init__(self, lagtime: int, lobe: nn.Module, lobe_timelagged: Optional[nn.Module] = None,
                 device=None, optimizer: Union[str, Callable] = 'Adam', learning_rate: float = 5e-4,
                 score_method: str = 'VAMP2', dtype=np.float32):
        super().__init__()
        self.lagtime = lagtime
        self.dtype = dtype
        self.device = device
        self.lobe = lobe
        self.lobe_timelagged = lobe_timelagged
        self.score_method = score_method
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self._step = 0
        self._train_scores = []
        self._validation_scores = []

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        assert value in (np.float32, np.float64), "only float32 and float64 are supported."
        self._dtype = value

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value: Union[str, Callable]):
        all_params = list(self.lobe.parameters()) + list(self.lobe_timelagged.parameters())
        unique_params = list(set(all_params))
        if isinstance(value, str):
            known_optimizers = {'Adam': torch.optim.Adam, 'SGD': torch.optim.SGD, 'RMSprop': torch.optim.RMSprop}
            if value not in known_optimizers.keys():
                raise ValueError(f"Unknown optimizer type, supported types are {known_optimizers.keys()}. "
                                 f"If desired, you can also pass the class of the "
                                 f"desired optimizer rather than its name.")
            value = known_optimizers[value]
        self._optimizer = value(params=unique_params, lr=self.learning_rate)

    @property
    def score_method(self) -> str:
        return self._score_method

    @score_method.setter
    def score_method(self, value: str):
        if value not in valid_score_methods:
            raise ValueError(f"Tried setting an unsupported scoring method '{value}', "
                             f"available are {valid_score_methods}.")
        self._score_method = value

    @property
    def lobe(self) -> nn.Module:
        return self._lobe

    @lobe.setter
    def lobe(self, value: nn.Module):
        self._lobe = value
        if self.dtype == np.float32:
            self._lobe = self._lobe.float()
        else:
            self._lobe = self._lobe.double()
        self._lobe = self._lobe.to(device=self.device)

    @property
    def lobe_timelagged(self) -> nn.Module:
        return self._lobe_timelagged

    @lobe_timelagged.setter
    def lobe_timelagged(self, value: Optional[nn.Module]):
        if value is None:
            value = self.lobe
        else:
            if self.dtype == np.float32:
                value = value.float()
            else:
                value = value.double()
        self._lobe_timelagged = value
        self._lobe_timelagged = self._lobe_timelagged.to(device=self.device)

    def partial_fit(self, data):
        self.lobe.train()
        self.lobe_timelagged.train()

        assert isinstance(data, (list, tuple)) and len(data) == 2, \
            "Data must be a list or tuple of batches belonging to instantaneous " \
            "and respective time-lagged data."

        batch_0, batch_t = data[0], data[1]

        if isinstance(data[0], np.ndarray):
            batch_0 = torch.from_numpy(data[0].astype(self.dtype)).to(device=self.device)
        if isinstance(data[1], np.ndarray):
            batch_t = torch.from_numpy(data[1].astype(self.dtype)).to(device=self.device)

        self.optimizer.zero_grad()
        x_0 = self.lobe(batch_0)
        x_t = self.lobe_timelagged(batch_t)
        loss_value = loss(x_0, x_t, method=self.score_method)
        loss_value.backward()
        self.optimizer.step()

        self._train_scores.append((self._step, -loss_value.detach().cpu().numpy()))
        self._step += 1

        return self

    def validate(self, validation_data: torch.Tensor) -> float:
        self.lobe.eval()
        self.lobe_timelagged.eval()

        with torch.no_grad():
            val = self.lobe(validation_data)
            val_score = VAMP(lagtime=self.lagtime, epsilon=1e-12) \
                .fit(val.cpu().numpy()).fetch_model().score(score_method=self.score_method)

        return val_score

    def fit(self, data, n_epochs=1, batch_size=512, validation_data=None, **kwargs):
        self._step = 0
        self._train_scores = []
        self._validation_scores = []

        if not isinstance(data, TimeSeriesDataSet):
            if isinstance(data, np.ndarray):
                data = data.astype(self.dtype)
            data = TimeSeriesDataSet(data, lagtime=self.lagtime)
        else:
            assert data.lagtime == self.lagtime, "If fitting with a data set, lagtimes must be compatible."
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)

        validation_loader = None
        if validation_data is not None:
            val_ds = validation_data
            if not isinstance(validation_data, (TimeSeriesDataSet, torch.utils.data.DataSet)):
                val_ds = torch.utils.data.DataSet(validation_data)
            validation_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

        for epoch in range(n_epochs):
            for batch_0, batch_t in data_loader:
                self.partial_fit((batch_0.to(device=self.device), batch_t.to(device=self.device)))
            if validation_loader is not None:
                scores = []
                for val_batch in validation_loader:
                    scores.append(self.validate(val_batch.to(device=self.device)))
                self._validation_scores.append((self._step, np.mean(scores)))
            latest_train_score = self._train_scores[-1][1]
            msg = f"Epoch [{epoch + 1}/{n_epochs}]: Latest training score {latest_train_score:.5f}"
            if validation_data is not None:
                latest_val_score = self._validation_scores[-1][1]
                msg += f", latest validation score {latest_val_score:.5f}"
            logger.debug(msg)
        return self

    def transform(self, data, **kwargs):
        return self.fetch_model().transform(data)

    def fetch_model(self) -> VAMPNetModel:
        return VAMPNetModel(self.lobe, self.lobe_timelagged, dtype=self.dtype, device=self.device,
                            train_scores=np.array(self._train_scores),
                            validation_scores=np.array(self._validation_scores))
