r"""
.. currentmodule: sktime.decomposition

.. rubric:: Koopman model
.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    KoopmanModel
    CovarianceKoopmanModel
    KoopmanBasisTransform
    IdentityKoopmanBasisTransform

.. rubric:: TICA
.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    TICA

.. rubric:: VAMP
.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    VAMP
"""

from .tica import TICA
from .vamp import VAMP
from .koopman import KoopmanBasisTransform, IdentityKoopmanBasisTransform, KoopmanModel, CovarianceKoopmanModel
