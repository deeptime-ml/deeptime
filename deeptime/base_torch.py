from typing import Callable, Union, List, Optional

import numpy as np
import torch


class DLEstimatorMixin:
    r""" Estimator subclass which offers some deep-learning estimators commonly used functionality.
    """

    @property
    def learning_rate(self):
        r""" Sets or yields a learning rate. Note that :meth:`setup_optimizer` should be called on update to propagate
        the changes.

        :type: float
        """
        if not hasattr(self, '_learning_rate'):
            self._learning_rate = 5e-4
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = value

    @property
    def device(self):
        r""" The device on which the estimator's PyTorch module(s) are operating. """
        if not hasattr(self, '_device'):
            self._device = None
        return self._device

    @device.setter
    def device(self, value):
        self._device = value

    @property
    def dtype(self):
        r""" The data type under which the estimator operates.

        :getter: Gets the currently set data type.
        :setter: Sets a new data type, must be one of np.float32, np.float64
        :type: numpy data type type
        """
        if not hasattr(self, '_dtype'):
            self._dtype = np.float32
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        assert value in (np.float32, np.float64), "only float32 and float64 are supported."
        self._dtype = value

    @property
    def optimizer(self) -> Optional[torch.optim.Optimizer]:
        r""" The optimizer that is used.

        :getter: Gets the currently configured optimizer.
        :setter: Sets a new optimizer based on optimizer name (string) or optimizer class (class reference).
        :type: torch.optim.Optimizer
        """
        if not hasattr(self, '_optimizer'):
            self._optimizer = None
        return self._optimizer

    def setup_optimizer(self, kind: Union[str, Callable], parameters: List):
        r""" Initializes a new optimizer on a list of parameters.

        Parameters
        ----------
        kind : str or Callable
            The optimizer.
        parameters : list of parameters
            The parameters
        """
        unique_params = list(set(parameters))
        if isinstance(kind, str):
            known_optimizers = {'Adam': torch.optim.Adam, 'SGD': torch.optim.SGD, 'RMSprop': torch.optim.RMSprop}
            if kind not in known_optimizers.keys():
                raise ValueError(f"Unknown optimizer type, supported types are {known_optimizers.keys()}. "
                                 f"If desired, you can also pass the class of the "
                                 f"desired optimizer rather than its name.")
            kind = known_optimizers[kind]
        self._optimizer = kind(params=unique_params, lr=self.learning_rate)
