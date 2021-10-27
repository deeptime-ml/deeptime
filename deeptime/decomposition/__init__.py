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
    KVAD

===============================================================================
Deep estimators
===============================================================================

Note that usage of these estimators requires a working installation of `PyTorch <https://pytorch.org/>`__.

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    deep.VAMPNet
    deep.TAE
    deep.TVAE

===============================================================================
Models
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    TransferOperatorModel
    CovarianceKoopmanModel
    DMDModel
    EDMDModel

    KernelEDMDModel
    KernelCCAModel
    KVADModel

    deep.VAMPNetModel
    deep.TAEModel
    deep.TVAEModel


===============================================================================
Utils
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    vamp_score
    vamp_score_data
    vamp_score_cv

    cvsplit_trajs

    deep.TVAEEncoder
    deep.koopman_matrix
    deep.sym_inverse
    deep.covariances
    deep.vamp_score
    deep.vampnet_loss
    deep.kvad_score
"""

from ._score import vamp_score, vamp_score_data, vamp_score_cv, cvsplit_trajs
from ._tica import TICA
from ._vamp import VAMP
from ._koopman import TransferOperatorModel, CovarianceKoopmanModel
from ._dmd import DMD, DMDModel, EDMD, EDMDModel, KernelEDMD, KernelEDMDModel
from ._cca import KernelCCA, KernelCCAModel
from ._kvad import KVAD, KVADModel

from ..util.platform import module_available
if module_available("torch"):
    from . import deep
del module_available
