r"""
.. currentmodule: deeptime.plots

Diagnostic plots for Markovianity and resolving processes.

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    plot_implied_timescales
    plot_ck_test


Plotting two-dimensional landscapes.

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst
    
    plot_energy2d
    Energy2dPlot
    plot_contour2d_from_xyz


Plots depicting networks.

.. autosummary::
    :toctree: generated/
    :template: class_nomodule.rst

    plot_adjacency
    plot_markov_model
    plot_flux
    Network
"""

from .implied_timescales import plot_implied_timescales
from .chapman_kolmogorov import plot_ck_test
from .energy import plot_energy2d, Energy2dPlot
from .network import Network, plot_adjacency, plot_markov_model, plot_flux
from .contour import plot_contour2d_from_xyz
