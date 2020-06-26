r"""
.. currentmodule: sktime.decomposition

=============
Koopman model
=============

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    KoopmanModel
    CovarianceKoopmanModel
    KoopmanBasisTransform
    IdentityKoopmanBasisTransform

===============================================================================
TICA
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    TICA
    TICAModel

===============================================================================
VAMP
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    VAMP
"""

from .tica import TICA, TICAModel
from .vamp import VAMP
from .koopman import KoopmanBasisTransform, IdentityKoopmanBasisTransform, KoopmanModel, CovarianceKoopmanModel
