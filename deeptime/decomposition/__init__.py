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
    DMD
    EDMD

===============================================================================
Models
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    KoopmanModel
    CovarianceKoopmanModel
    EDMDKoopmanModel
    KoopmanBasisTransform
    IdentityKoopmanBasisTransform
    DMDModel

===============================================================================
Utils
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    vampnet.MLPLobe
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
from .dmd import DMD, DMDModel, EDMD, EDMDKoopmanModel
from ..util.platform import module_available

if module_available("torch"):
    from .vampnet import VAMPNet
    from . import vampnet
    from . import kvadnet
del module_available
