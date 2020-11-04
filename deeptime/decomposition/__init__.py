r"""
.. currentmodule: deeptime.decomposition

===============================================================================
Estimators
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    VAMP
    TICA
    DMD
    EDMD
    KernelEDMD
    KernelCCA

===============================================================================
Deep estimators
===============================================================================

Note that usage of these estimators requires a working installation of `PyTorch <https://pytorch.org/>`__.

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    VAMPNet
    TAE
    TVAE

===============================================================================
Models
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    KoopmanModel
    CovarianceKoopmanModel
    KoopmanBasisTransform
    IdentityKoopmanBasisTransform
    DMDModel
    EDMDModel
    KernelEDMDModel
    KernelCCAModel
    VAMPNetModel
    TAEModel
    TVAEModel

===============================================================================
Utils
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    TVAEEncoder
    vampnet.koopman_matrix
    vampnet.sym_inverse
    vampnet.covariances
    vampnet.score
    vampnet.loss

    kvadnet.kvad_score

"""

from .tica import TICA
from .vamp import VAMP
from .koopman import KoopmanBasisTransform, IdentityKoopmanBasisTransform, KoopmanModel, CovarianceKoopmanModel
from .dmd import DMD, DMDModel, EDMD, EDMDModel, KernelEDMD, KernelEDMDModel
from .cca import KernelCCA, KernelCCAModel

from ..util.platform import module_available
if module_available("torch"):
    from .vampnet import VAMPNet, VAMPNetModel
    from .tae import TAE, TAEModel, TVAE, TVAEModel, TVAEEncoder
    from . import vampnet
    from . import kvadnet
del module_available
