from typing import Optional, Union, List

import numpy as np

from ..base import Model, Transformer
from ..numeric import is_diagonal_matrix
from ..util import cached_property


class KoopmanBasisTransform(object):
    r""" Transforms a system's observable

    .. math:: f(x_t) = (\chi^{(1)}(x_t),\ldots,\chi^{(n)})^\top

    to another basis

    .. math:: \tilde f(x_t) = T (f(x_t) - \mu ),

    where :math:`T` is the transformation matrix and :math:`\mu` a constant mean value that is subtracted.
    """

    def __init__(self, mean, transformation_matrix):
        self._mean = mean
        self._transformation_matrix = transformation_matrix

    @cached_property
    def backward_transformation_matrix(self):
        return np.linalg.pinv(self._transformation_matrix)

    def __call__(self, data, inverse=False, dim=None):
        r""" Applies the basis transform to data.

        Parameters
        ----------
        data : (T, n) ndarray
            Data consisting of `T` frames in `n` dimensions.
        inverse : bool, default=False
            Whether to apply the forward or backward operation, i.e., :math:`T (f(x_t) - \mu )` or
            :math:`T^{-1} f(x_t) + \mu`, respectively.
        dim : int or None
            Number of dimensions to restrict to, removes the all but the first :math:`n` basis transformation vectors.
            Can only be not `None` if `inverse` is False.

        Returns
        -------
        transformed_data : (T, k) ndarray
            Transformed data. If :math:`T\in\mathbb{R}^{n\times m}`, we get :math:`k=\min\{m, \mathrm{dim}\}`.
        """
        if not inverse:
            return (data - self._mean[:dim]) @ self._transformation_matrix[:, :dim]
        else:
            if dim is not None:
                raise ValueError("This currently only works for the forward transform.")
            return data @ self.backward_transformation_matrix + self._mean


class IdentityKoopmanBasisTransform(KoopmanBasisTransform):

    def __init__(self):
        super().__init__(0., 1.)

    def backward_transformation_matrix(self):
        return 1.

    def __call__(self, data, inverse=False, dim=None):
        return data


class KoopmanModel(Model, Transformer):
    r""" Model which contains a finite-dimensional Koopman operator (or approximation thereof).
    It describes the temporal evolution of observable space, i.e.,

    .. math:: \mathbb{E}[g(x_{t+\tau}] = K^\top \mathbb{E}[f(x_t)],

    where :math:`K\in\mathbb{R}^{n\times m}` is the Koopman operator, :math:`x_t` the system's state at time :math:`t`,
    and $f$ and $g$ observables of the system's state.
    """

    def __init__(self, operator: np.ndarray,
                 basis_transform_forward: Optional[KoopmanBasisTransform],
                 basis_transform_backward: Optional[KoopmanBasisTransform],
                 output_dimension=None):
        r""" Creates a new Koopman model.

        Parameters
        ----------
        operator : (n, n) ndarray
            Applies the transform :math:`K^\top` in the modified basis.
        basis_transform_forward : KoopmanBasisTransform or None
            Transforms the current state :math:`f(x_t)` to the basis in which the Koopman operator is defined.
            If `None`, this defaults to the identity operation.
        basis_transform_backward : KoopmanBasisTransform or None
            Transforms the future state :math:`g(x_t)` to the basis in which the Koopman operator is defined.
            If `None`, this defaults to the identity operation
        """
        super().__init__()
        if basis_transform_forward is None:
            basis_transform_forward = IdentityKoopmanBasisTransform()
        if basis_transform_backward is None:
            basis_transform_backward = IdentityKoopmanBasisTransform()
        self._operator = np.asarray_chkfinite(operator)
        self._basis_transform_forward = basis_transform_forward
        self._basis_transform_backward = basis_transform_backward
        if output_dimension is not None and not is_diagonal_matrix(self.operator):
            raise ValueError("Output dimension can only be set if the Koopman operator is a diagonal matrix. This"
                             "can be achieved through the VAMP estimator.")
        self._output_dimension = output_dimension

    @property
    def operator(self) -> np.ndarray:
        r""" The operator :math:`K` so that :math:`\mathbb{E}[g(x_{t+\tau}] = K^\top \mathbb{E}[f(x_t)]` in transformed
        bases.
        """
        return self._operator

    @cached_property
    def operator_inverse(self) -> np.ndarray:
        r""" Inverse of the operator :math:`K`, i.e., :math:`K^{-1}`. Potentially also pseudo-inverse. """
        return np.linalg.pinv(self.operator)

    @property
    def basis_transform_forward(self) -> KoopmanBasisTransform:
        r"""Transforms the basis of :math:`\mathbb{E}[f(x_t)]` to the one in which the Koopman operator is defined. """
        return self._basis_transform_forward

    @property
    def basis_transform_backward(self) -> KoopmanBasisTransform:
        r"""Transforms the basis of :math:`\mathbb{E}[g(x_{t+\tau})]` to the one in which
        the Koopman operator is defined. """
        return self._basis_transform_backward

    @property
    def output_dimension(self):
        r"""The output dimension of the :meth:`transform` pass."""
        return self._output_dimension

    def forward(self, trajectory: np.ndarray, components: Optional[Union[int, List[int]]]) -> np.ndarray:
        r""" Applies the forward transform to the trajectory in non-transformed space. Given the Koopman operator
        :math:`\Sigma`, transformations  :math:`V^\top - \mu_t` and :math:`U^\top -\mu_0` for
        bases :math:`f` and :math:`g`, respectively, this is achieved by transforming each frame :math:`X_t` with

        .. math::
            \hat{X}_{t+\tau} = (V^\top)^{-1} \Sigma U^\top (X_t - \mu_0) + \mu_t.

        If the model stems from a :class:`VAMP <sktime.decomposition.VAMP>` estimator, :math:`V` are the left
        singular vectors, :math:`\Sigma` the singular values, and :math:`U` the right singular vectors.

        Parameters
        ----------
        trajectory : (T, n) ndarray
            The input trajectory
        components : int or list of int or None
            Optional arguments for the Koopman operator if appropriate. If the model stems from
            a :class:`VAMP <sktime.decomposition.VAMP>` estimator, these are the component(s) to project onto.
            If None, all processes are taken into account, if list of integer, this sets all singular values
            to zero but the "components"th ones.

        Returns
        -------
        predictions : (T, n) ndarray
            The predicted trajectory.
        """
        if components is not None:
            if not is_diagonal_matrix(self.operator):
                raise ValueError("A subselection of components is only possible if the koopman operator is a diagonal"
                                 "matrix! This can be achieved by using the VAMP estimator, yielding appropriate"
                                 "basis transforms from covariance matrices.")
            operator = np.zeros_like(self.operator)
            if not isinstance(components, (list, tuple)):
                components = [components]
            for ii in components:
                operator[ii, ii] = self.operator[ii]
        else:
            operator = self.operator
        output_trajectory = np.empty_like(trajectory)
        # todo i think this can be vectorized by just not looping over t?
        for t, frame in enumerate(trajectory):
            x = self.basis_transform_forward(trajectory[t])  # map into basis in which K is defined
            x = operator.T @ x  # apply operator, we are now in the modified basis of g
            x = self.basis_transform_backward(x, inverse=True)  # map back to observable basis
            output_trajectory[t] = x
        return output_trajectory

    def transform(self, data, forward=True, propagate=False, **kwargs):
        r"""Projects the data into the Koopman operator basis, possibly discarding nonrelevant dimensions.

        Parameters
        ----------
        data : ndarray(T, n)
            the input data
        forward : bool, default=True
            Whether to use forward or backward transform for projection.
        propagate : bool, default=False
            Whether to propagate the projection with :math:`K^\top`.

        Returns
        -------
        projection : ndarray(T, m)
            The projected data.
            In case of VAMP, if `forward` is True, projection will be on the right singular
            functions. Otherwise, projection will be on the left singular functions.
        """
        if forward:
            transform = self.basis_transform_forward(data, dim=self._output_dimension)
            return self.operator.T @ transform if propagate else transform
        else:
            transform = self.basis_transform_backward(data, dim=self._output_dimension)
            return self.operator_inverse.T @ transform if propagate else transform
