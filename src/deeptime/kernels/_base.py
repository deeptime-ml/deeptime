import numpy as np


class Kernel:
    r""" The base class of all kernels. Provides interfaces to evaluating the kernel on points in state space
    as well as computing the kernel matrix / Gramian.
    """

    def __call__(self, x, y):
        r""" Computes the value of the kernel at two specific points.

        Parameters
        ----------
        x : (d,) ndarray
            X point.
        y : (d,) ndarray
            Y point.

        Returns
        -------
        kxy : float
            The kernel evaluation :math:`\kappa (x, y)`.

        Notes
        -----
        This dispatches to :meth:`_evaluate` which is an interface method intended to be overridden
        by a specific kernel implementation.
        """
        return self._evaluate(x, y)

    def _evaluate(self, x, y) -> float:
        raise NotImplementedError("Not implemented, please provide a subclass!")

    def gram(self, data: np.ndarray) -> np.ndarray:
        r""" Computes the corresponding Gram matrix, see also :meth:`apply`.

        Parameters
        ----------
        data : (T, d) ndarray
            The data array.

        Returns
        -------
        G : (T, T) ndarray
            The kernel Gramian with :code:`G[i, j] = k(x[i], x[j])`.
        """
        return self.apply(data_1=data, data_2=data)

    def apply(self, data_1: np.ndarray, data_2: np.ndarray) -> np.ndarray:
        r""" Applies the kernel to data arrays.

        Given :math:`x\in\mathbb{R}^{T_1 \times d}` and
        :math:`y\in\mathbb{R}^{T_2\times d}`, it yields a kernel matrix

        .. math::

            K = (\kappa(x_i, y_j))_{i,j}\in\mathbb{R}^{T_1\times T_2}.

        Note that this corresponds to the kernel Gramian in case :math:`x = y`.

        Parameters
        ----------
        data_1 : (T_1, d) ndarray
            Data array.
        data_2 : (T_2, d) ndarray
            Data array.

        Returns
        -------
        K : (T_1, T_2) ndarray
            The kernel matrix.
        """
        m = data_1.shape[0]
        n = data_2.shape[0]

        result = np.empty([m, n], dtype=data_1.dtype)
        for i in range(m):
            for j in range(n):
                result[i, j] = self(data_1[i], data_2[j])
        return result

    def __mul__(self, other):
        r""" Multiplication with another kernel.

        Parameters
        ----------
        other : Kernel
            The other kernel.

        Returns
        -------
        result : ProductKernel
            Product of this and the other kernel.
        """
        if not isinstance(other, Kernel):
            raise ValueError("Kernel can only be multiplied with other kernels.")
        return ProductKernel(self, other)

    __rmul__ = __mul__  # no difference here


class ProductKernel(Kernel):

    def __init__(self, k1: Kernel, k2: Kernel):
        self._k1 = k1
        self._k2 = k2

    def _evaluate(self, x, y) -> float:
        return self._k1(x, y) * self._k2(x, y)

    def __str__(self):
        return f"{self._k1} * {self._k2}"


def is_torch_kernel(kernel: Kernel) -> bool:
    r""" Tests whether given kernel supports PyTorch. Note that even if the kernel does not support PyTorch it
    (depending on the case) might still be usable in conjunction with PyTorch code at a performance penalty, e.g.,
    when no gradients are required for the kernel evaluation.

    Parameters
    ----------
    kernel : Kernel
        A kernel instance.

    Returns
    -------
    is_torch_kernel : bool
        True if it supports PyTorch, False otherwise.
    """
    return hasattr(kernel, 'apply_torch')
