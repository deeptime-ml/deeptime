r"""
.. currentmodule: deeptime.markov

===============================================================================
Transition counting and analysis tools
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    pcca
    PCCAModel

    TransitionCountEstimator
    TransitionCountModel

    ReactiveFlux
    reactive_flux

    MembershipsChapmanKolmogorovValidator

===============================================================================
Utilities
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    number_of_states
    count_states
    compute_connected_sets

    compute_dtrajs_effective
    compute_effective_stride

    sample.indices_by_distribution
    sample.compute_index_states
    sample.indices_by_sequence
    sample.indices_by_state
"""

from ._util import number_of_states, count_states, compute_connected_sets, \
    compute_dtrajs_effective, compute_effective_stride
from . import sample
from . import tools  # former msmtools

from ._base import BayesianPosterior, _MSMBaseEstimator, MembershipsChapmanKolmogorovValidator
from ._pcca import pcca, PCCAModel
from ._transition_counting import TransitionCountEstimator, TransitionCountModel

from ._reactive_flux import ReactiveFlux, reactive_flux

from . import msm
from . import hmm
