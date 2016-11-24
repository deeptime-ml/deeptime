from __future__ import absolute_import
__author__ = 'noe'

# import subpackages such that they are available after the main package import
from . import estimators
from . import solvers

# direct imports of important functions/classes to-level API
from .solvers.direct import eig_corr
from .solvers.direct import sort_by_norm
from .solvers.eig_qr.eig_qr import eig_qr

