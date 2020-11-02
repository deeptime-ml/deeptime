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
    VAMPNet
    TAE
    TVAE
    DMD
    EDMD
    KernelEDMD
    KernelCCA

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
    VAMPNetModel
    DMDModel
    EDMDModel
    KernelEDMDModel
    KernelCCAModel

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
