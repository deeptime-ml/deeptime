r"""
.. currentmodule: deeptime.plots

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    plot_implied_timescales
    plot_ck_test
    plot_energy2d
    Energy2dPlot

    plot_adjacency
    plot_markov_model
    plot_flux
    Network
"""

from .implied_timescales import plot_implied_timescales
from .chapman_kolmogorov import plot_ck_test
from .energy import plot_energy2d, Energy2dPlot
from .network import Network, plot_adjacency, plot_markov_model, plot_flux
