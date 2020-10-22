r"""
.. currentmodule: deeptime.kernels

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    Kernel
    GaussianKernel
    GeneralizedGaussianKernel
    LaplacianKernel
    PolynomialKernel


.. rubric:: PyTorch-ready kernels

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    TorchGaussianKernel

.. rubric:: Utils

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    is_torch_kernel
"""

from .base import Kernel, is_torch_kernel
from .kernels import GaussianKernel, GeneralizedGaussianKernel, LaplacianKernel, PolynomialKernel

from ..util.platform import module_available
if module_available("torch"):
    from .kernels_torch import TorchGaussianKernel
del module_available
