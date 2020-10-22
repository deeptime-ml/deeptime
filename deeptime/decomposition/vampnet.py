from typing import Optional, Union, List, Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ..base import Transformer, Model, Estimator
from ..data.util import TimeSeriesDataset, TimeLaggedDataset


def symeig_reg(mat, epsilon: float = 1e-6, mode='regularize', eigenvectors=True) \
        -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r""" Solves a eigenvector/eigenvalue decomposition for a hermetian matrix also if it is rank deficient.

    Parameters
    ----------
    mat : torch.Tensor
        the hermetian matrix
    epsilon : float, default=1e-6
        Cutoff for eigenvalues.
    mode : str, default='regularize'
        Whether to truncate eigenvalues if they are too small or to regularize them by taking the absolute value
        and adding a small positive constant. :code:`trunc` leads to truncation, :code:`regularize` leads to epsilon
        being added to the eigenvalues after taking the absolute value
    eigenvectors : bool, default=True
        Whether to compute eigenvectors.

    Returns
    -------
    (eigval, eigvec) : Tuple[torch.Tensor, Optional[torch.Tensor]]
        Eigenvalues and -vectors.
    """
    assert mode in sym_inverse.valid_modes, f"Invalid mode {mode}, supported are {sym_inverse.valid_modes}"

    if mode == 'regularize':
        identity = torch.eye(mat.shape[0], dtype=mat.dtype, device=mat.device)
        mat = mat + epsilon * identity

    # Calculate eigvalues and potentially eigvectors
    eigval, eigvec = torch.symeig(mat, eigenvectors=True)

    if eigenvectors:
        eigvec = eigvec.transpose(0, 1)

    if mode == 'trunc':
        # Filter out Eigenvalues below threshold and corresponding Eigenvectors
        mask = eigval > epsilon
        eigval = eigval[mask]
        if eigenvectors:
            eigvec = eigvec[mask]
    elif mode == 'regularize':
        # Calculate eigvalues and eigvectors
        eigval = torch.abs(eigval)
    elif mode == 'clamp':
        eigval = torch.clamp_min(eigval, min=epsilon)

    else:
        raise RuntimeError("Invalid mode! Should have been caught by the assertion.")

    if eigenvectors:
        return eigval, eigvec
    else:
        return eigval, eigvec


def sym_inverse(mat, epsilon: float = 1e-6, return_sqrt=False, mode='regularize'):
    """ Utility function that returns the inverse of a matrix, with the
    option to return the square root of the inverse matrix.

    Parameters
    ----------
    mat: numpy array with shape [m,m]
        Matrix to be inverted.
    epsilon : float
        Cutoff for eigenvalues.
    return_sqrt: bool, optional, default = False
        if True, the square root of the inverse matrix is returned instead
    mode: str, default='trunc'
        Whether to truncate eigenvalues if they are too small or to regularize them by taking the absolute value
        and adding a small positive constant. :code:`trunc` leads to truncation, :code:`regularize` leads to epsilon
        being added to the eigenvalues after taking the absolute value

    Returns
    -------
    x_inv: numpy array with shape [m,m]
        inverse of the original matrix
    """
    eigval, eigvec = symeig_reg(mat, epsilon, mode)

    # Build the diagonal matrix with the filtered eigenvalues or square
    # root of the filtered eigenvalues according to the parameter
    if return_sqrt:
        diag = torch.diag(torch.sqrt(1. / eigval))
    else:
        diag = torch.diag(1. / eigval)

    return torch.chain_matmul(eigvec.t(), diag, eigvec)


sym_inverse.valid_modes = ('trunc', 'regularize', 'clamp')


def koopman_matrix(x: torch.Tensor, y: torch.Tensor, epsilon: float = 1e-6, mode: str = 'trunc',
                   c_xx: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
    r""" Computes the Koopman matrix

    .. math:: K = C_{00}^{-1/2}C_{0t}C_{tt}^{-1/2}

    based on data over which the covariance matrices :math:`C_{\cdot\cdot}` are computed.

    Parameters
    ----------
    x : torch.Tensor
        Instantaneous data.
    y : torch.Tensor
        Time-lagged data.
    epsilon : float, default=1e-6
        Cutoff parameter for small eigenvalues.
    mode : str, default='trunc'
        Regularization mode for Hermetian inverse. See :meth:`sym_inverse`.
    c_xx : tuple of torch.Tensor, optional, default=None
        Tuple containing c00, c0t, ctt if already computed.

    Returns
    -------
    K : torch.Tensor
        The Koopman matrix.
    """
    if c_xx is not None:
        c00, c0t, ctt = c_xx
    else:
        c00, c0t, ctt = covariances(x, y, remove_mean=True)
    c00_sqrt_inv = sym_inverse(c00, return_sqrt=True, epsilon=epsilon, mode=mode)
    ctt_sqrt_inv = sym_inverse(ctt, return_sqrt=True, epsilon=epsilon, mode=mode)
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
    deeptime.covariance.Covariance : Estimator yielding these kind of covariance matrices based on raw numpy arrays
                                     using an online estimation procedure.
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


valid_score_methods = ('VAMP1', 'VAMP2', 'VAMPE')


def score(data: torch.Tensor, data_lagged: torch.Tensor, method='VAMP2', epsilon: float = 1e-6, mode='trunc'):
    r"""Computes the VAMP score based on data and corresponding time-shifted data.

    Parameters
    ----------
    data : torch.Tensor
        (N, d)-dimensional torch tensor
    data_lagged : torch.Tensor
        (N, k)-dimensional torch tensor
    method : str, default='VAMP2'
        The scoring method. See :meth:`score <deeptime.decomposition.CovarianceKoopmanModel.score>` for details.
    epsilon : float, default=1e-6
        Cutoff parameter for small eigenvalues, alternatively regularization parameter.
    mode : str, default='trunc'
        Regularization mode for Hermetian inverse. See :meth:`sym_inverse`.

    Returns
    -------
    score : torch.Tensor
        The score. It contains a contribution of :math:`+1` for the constant singular function since the
        internally estimated Koopman operator is defined on a decorrelated basis set.
    """
    if method not in valid_score_methods:
        raise ValueError(f"Invalid method '{method}', supported are {valid_score_methods}")
    assert data.shape == data_lagged.shape

    if method == 'VAMP1':
        koopman = koopman_matrix(data, data_lagged, epsilon=epsilon, mode=mode)
        vamp_score = torch.norm(koopman, p='nuc')
    elif method == 'VAMP2':
        koopman = koopman_matrix(data, data_lagged, epsilon=epsilon, mode=mode)
        vamp_score = torch.pow(torch.norm(koopman, p='fro'), 2)
    elif method == 'VAMPE':
        c00, c0t, ctt = covariances(data, data_lagged, remove_mean=True)
        c00_sqrt_inv = sym_inverse(c00, epsilon=epsilon, return_sqrt=True, mode=mode)
        ctt_sqrt_inv = sym_inverse(ctt, epsilon=epsilon, return_sqrt=True, mode=mode)
        koopman = torch.chain_matmul(c00_sqrt_inv, c0t, ctt_sqrt_inv).t()

        u, s, v = torch.svd(koopman)
        mask = s > epsilon

        u = torch.mm(c00_sqrt_inv, u[:, mask])
        v = torch.mm(ctt_sqrt_inv, v[:, mask])
        s = s[mask]

        u_t = u.t()
        v_t = v.t()
        s = torch.diag(s)

        vamp_score = torch.trace(
            2. * torch.chain_matmul(s, u_t, c0t, v)
            - torch.chain_matmul(s, u_t, c00, u, s, v_t, ctt, v)
        )
    else:
        raise RuntimeError("This should have been caught earlier.")
    return 1 + vamp_score


def loss(data: torch.Tensor, data_lagged: torch.Tensor, method='VAMP2', epsilon: float = 1e-6, mode: str = 'trunc'):
    r"""Loss function that can be used to train VAMPNets. It evaluates as :math:`-\mathrm{score}`. The score
    is implemented in :meth:`score`."""
    return -1. * score(data, data_lagged, method=method, epsilon=epsilon, mode=mode)


class MLPLobe(nn.Module):
    r""" A multilayer perceptron which can be used as a neural network lobe for VAMPNets. """

    def __init__(self, units: List[int], nonlinearity=nn.ELU, initial_batchnorm: bool = True,
                 output_nonlinearity=lambda: nn.Softmax(dim=1)):
        r"""Instantiates a new lobe.

        Parameters
        ----------
        units : list of integers
            The units of the fully connected layers.
        nonlinearity : callable, default=torch.nn.ELU
            A callable (like a constructor) which yields an instance of a particular activation function.
        initial_batchnorm : bool, default=True
            Whether to use batch normalization before the data enters the rest of the network.
        output_nonlinearity : callable, default=softmax
            The output activation/nonlinearity. If the data decomposes into states, it can make sense to use
            an output activation like softmax which produces a probability distribution over said states.
        """
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
    r"""
    A VAMPNet model which can be fit to data optimizing for one of the implemented VAMP scores.

    See Also
    --------
    VAMPNet : The corresponding estimator.
    """

    def __init__(self, lobe: nn.Module, lobe_timelagged: Optional[nn.Module] = None,
                 dtype=np.float32, device=None):
        r"""Creates a new VAMPNet estimator instance.

        Parameters
        ----------
        lobe : torch.nn.Module
            One of the lobes of the VAMPNet. See also :class:`MLPLobe`.
        lobe_timelagged : torch.nn.Module, optional, default=None
            The timelagged lobe. Can be left None, in which case the lobes are shared.
        dtype : data type, default=np.float32
            The data type for which operations should be performed. Leads to an appropriate cast within fit and
            transform methods.
        device : device, default=None
            The device for the lobe(s). Can be None which defaults to CPU.
        """
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

    def transform(self, data, instantaneous: bool = True, **kwargs):
        r""" Transforms a tensor or array or list thereof using the learnt transformation.

        Parameters
        ----------
        data : array_like
            The input data.
        instantaneous : bool, default=True
            Whether to use the instantaneous lobe or the time-shifted one.
        **kwargs
            Scikit-learn compatibility.

        Returns
        -------
        transform : array_like
            The featurized data.
        """
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
                if instantaneous:
                    out.append(self._lobe(x).cpu().numpy())
                else:
                    out.append(self._lobe_timelagged(x).cpu().numpy())
        if isinstance(out, (list, tuple)) and len(out) == 1:
            out = out[0]
        return out


class VAMPNet(Estimator, Transformer):
    r""" Implementation of VAMPNets :cite:`vnet-mardt2018vampnets` which try to find an optimal featurization of
    data based on a VAMP score :cite:`vnet-wu2020variational` by using neural networks as featurizing transforms
    which are equipped with a loss that is the negative VAMP score. This estimator is also a transformer
    and can be used to transform data into the optimized space. From there it can either be used to estimate
    Markov state models via making assignment probabilities crisp (in case of softmax output distributions) or
    to estimate the Koopman operator using the :class:`VAMP <deeptime.decomposition.VAMP>` estimator.

    See Also
    --------
    deeptime.decomposition.VAMP : Koopman operator estimator which can be applied to transformed data.

    References
    ----------
    .. bibliography:: /references.bib
        :style: unsrt
        :filter: docname in docnames
        :keyprefix: vnet-
    """
    _MUTABLE_INPUT_DATA = True

    def __init__(self, lagtime: int, lobe: nn.Module, lobe_timelagged: Optional[nn.Module] = None,
                 device=None, optimizer: Union[str, Callable] = 'Adam', learning_rate: float = 5e-4,
                 score_method: str = 'VAMP2', score_mode: str = 'regularize', epsilon: float = 1e-6,
                 dtype=np.float32, shuffle: bool = True):
        r""" Creates a new VAMPNet instance.

        Parameters
        ----------
        lagtime : int
            The lagtime under which covariance matrices are estimated.
        lobe : torch.nn.Module
            A neural network module which maps input data to some (potentially) lower-dimensional space.
        lobe_timelagged : torch.nn.Module, optional, default=None
            Neural network module for timelagged data, in case of None the lobes are shared (structure and weights).
        device : torch device, default=None
            The device on which the torch modules are executed.
        optimizer : str or Callable, default='Adam'
            An optimizer which can either be provided in terms of a class reference (like `torch.optim.Adam`) or
            a string (like `'Adam'`). Defaults to Adam.
        learning_rate : float, default=5e-4
            The learning rate of the optimizer.
        score_method : str, default='VAMP2'
            The scoring method which is used for optimization.
        score_mode : str, default='regularize'
            The mode under which inverses of positive semi-definite matrices are estimated. Per default, the matrices
            are perturbed by a small constant added to the diagonal. This makes sure that eigenvalues are not too
            small. For a complete list of modes, see :meth:`sym_inverse`.
        epsilon : float, default=1e-6
            The strength of the regularization under which matrices are inverted. Meaning depends on the score_mode,
            see :meth:`sym_inverse`.
        dtype : dtype, default=np.float32
            The data type of the modules and incoming data.
        shuffle : bool, default=True
            Whether to shuffle data during training after each epoch.
        """
        super().__init__()
        self.lagtime = lagtime
        self.dtype = dtype
        self.device = device
        self.lobe = lobe
        self.lobe_timelagged = lobe_timelagged
        self.score_method = score_method
        self.score_mode = score_mode
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self._step = 0
        self.shuffle = shuffle
        self._epsilon = epsilon

    @property
    def dtype(self):
        r""" The data type under which the estimator operates.

        :getter: Gets the currently set data type.
        :setter: Sets a new data type, must be one of np.float32, np.float64
        :type: numpy data type type
        """
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        assert value in (np.float32, np.float64), "only float32 and float64 are supported."
        self._dtype = value

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        r""" The optimizer that is used.

        :getter: Gets the currently configured optimizer.
        :setter: Sets a new optimizer based on optimizer name (string) or optimizer class (class reference).
        :type: torch.optim.Optimizer
        """
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
    def epsilon(self) -> float:
        r""" Regularization parameter for matrix inverses.

        :getter: Gets the currently set parameter.
        :setter: Sets a new parameter. Must be non-negative.
        :type: float
        """
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float):
        assert value >= 0
        self._epsilon = value

    @property
    def score_method(self) -> str:
        r""" Property which steers the scoring behavior of this estimator.

        :getter: Gets the current score.
        :setter: Sets the score to use.
        :type: str
        """
        return self._score_method

    @score_method.setter
    def score_method(self, value: str):
        if value not in valid_score_methods:
            raise ValueError(f"Tried setting an unsupported scoring method '{value}', "
                             f"available are {valid_score_methods}.")
        self._score_method = value

    @property
    def lobe(self) -> nn.Module:
        r""" The instantaneous lobe of the VAMPNet.

        :getter: Gets the instantaneous lobe.
        :setter: Sets a new lobe.
        :type: torch.nn.Module
        """
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
        r""" The timelagged lobe of the VAMPNet.

        :getter: Gets the timelagged lobe. Can be the same a the instantaneous lobe.
        :setter: Sets a new lobe. Can be None, in which case the instantaneous lobe is shared.
        :type: torch.nn.Module
        """
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

    def partial_fit(self, data, train_score_callback: Callable[[int, torch.Tensor], None] = None):
        r""" Performs a partial fit on data. This does not perform any batching.

        Parameters
        ----------
        data : tuple or list of length 2, containing instantaneous and timelagged data
            The data to train the lobe(s) on.
        train_score_callback : callable, optional, default=None
            An optional callback function which is evaluated after partial fit, containing the current step
            of the training (only meaningful during a :meth:`fit`) and the current score as torch Tensor.

        Returns
        -------
        self : VAMPNet
            Reference to self.
        """
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
        loss_value = loss(x_0, x_t, method=self.score_method, epsilon=self.epsilon, mode=self.score_mode)
        loss_value.backward()
        self.optimizer.step()

        if train_score_callback is not None:
            lval_detached = loss_value.detach()
            train_score_callback(self._step, -lval_detached)
        self._step += 1

        return self

    def validate(self, validation_data: Tuple[torch.Tensor]) -> torch.Tensor:
        r""" Evaluates the currently set lobe(s) on validation data and returns the value of the configured score.

        Parameters
        ----------
        validation_data : Tuple of torch Tensor containing instantaneous and timelagged data
            The validation data.

        Returns
        -------
        score : torch.Tensor
            The value of the score.
        """
        self.lobe.eval()
        self.lobe_timelagged.eval()

        with torch.no_grad():
            val = self.lobe(validation_data[0])
            val_t = self.lobe_timelagged(validation_data[1])
            return score(val, val_t, method=self.score_method, mode=self.score_mode, epsilon=self.epsilon)

    def _set_up_data_loader(self, data, batch_size, shuffle):
        r""" Helper method which yields a data loader from a torch dataset or numpy arrays. """
        if not isinstance(data, (TimeSeriesDataset, Dataset)):
            if isinstance(data, np.ndarray):
                data = data.astype(self.dtype)
            data = TimeLaggedDataset.from_trajectory(lagtime=self.lagtime, data=data)
        return DataLoader(data, batch_size=batch_size, shuffle=shuffle)

    def fit(self, data, n_epochs=1, batch_size=512, validation_data=None,
            train_score_callback: Callable[[int, torch.Tensor], None] = None,
            validation_score_callback: Callable[[int, torch.Tensor], None] = None, **kwargs):
        r""" Fits a VampNet on data.

        Parameters
        ----------
        data : torch.Tensor or Dataset or TimelaggedDataset or numpy array
            The data to use for training.
        n_epochs : int, default=1
            The number of epochs (i.e., passes through the training data) to use for training.
        batch_size : int, default=512
            The batch size to use during training. Should be reasonably large so that covariance matrices have
            enough statistics to be estimated.
        validation_data : torch.Tensor or Dataset or TimeLaggedDataset or numpy array, optional, default=None
            Validation data.
        train_score_callback : callable, optional, default=None
            Callback function which is invoked after each batch and gets as arguments the current training step
            as well as the score (as torch Tensor).
        validation_score_callback : callable, optional, default=None
            Callback function for validation data. Is invoked after each epoch if validation data is given
            and the callback function is not None. Same as the train callback, this gets the 'step' as well as
            the score.
        **kwargs
            Optional keyword arguments for scikit-learn compatibility

        Returns
        -------
        self : VAMPNet
            Reference to self.
        """
        self._step = 0

        # set up loaders
        data_loader = self._set_up_data_loader(data, batch_size=batch_size, shuffle=self.shuffle)
        validation_loader = None
        if validation_data is not None:
            validation_loader = self._set_up_data_loader(validation_data, batch_size=batch_size, shuffle=False)

        # and train
        for epoch in range(n_epochs):
            for batch_0, batch_t in data_loader:
                self.partial_fit((batch_0.to(device=self.device), batch_t.to(device=self.device)),
                                 train_score_callback=train_score_callback)
            if validation_loader is not None and validation_score_callback is not None:
                with torch.no_grad():
                    scores = []
                    for val_batch in validation_loader:
                        scores.append(
                            self.validate((val_batch[0].to(device=self.device), val_batch[1].to(device=self.device)))
                        )
                    mean_score = torch.mean(torch.stack(scores))
                    validation_score_callback(self._step, mean_score)
        return self

    def transform(self, data, instantaneous: bool = True, **kwargs):
        r""" Transforms data through the instantaneous or time-shifted network lobe.

        Parameters
        ----------
        data : numpy array or torch tensor
            The data to transform.
        instantaneous : bool, default=True
            Whether to use the instantaneous lobe or the time-shifted lobe for transformation.
        **kwargs
            Ignored kwargs for api compatibility.

        Returns
        -------
        transform : array_like
            List of numpy array or numpy array containing transformed data.
        """
        return self.fetch_model().transform(data, instantaneous=instantaneous, **kwargs)

    def fetch_model(self) -> VAMPNetModel:
        r""" Yields the current model. """
        return VAMPNetModel(self.lobe, self.lobe_timelagged, dtype=self.dtype, device=self.device)
