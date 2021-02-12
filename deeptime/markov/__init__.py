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
    compute_reactive_flux

    score_cv

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
from . import _sample as sample
from . import tools  # former msmtools

from ._base import BayesianPosterior, _MSMBaseEstimator
from ._pcca import pcca, PCCAModel
from ._transition_counting import TransitionCountEstimator, TransitionCountModel

from ._reactive_flux import ReactiveFlux, compute_reactive_flux

from ._base import score_cv
from . import msm
from . import hmm
