from ._version import get_versions

__version__ = get_versions()['version']
del get_versions

from . import util
from . import numeric
from . import data
from . import basis
from . import kernels
from . import sindy
from . import clustering
from . import covariance
from . import decomposition
from . import markov


def capi_includes(inc_clustering: bool = False, inc_markov: bool = False, inc_markov_hmm: bool = False,
                  inc_data: bool = False):
    import os
    import sys
    module_path = sys.modules['deeptime'].__path__[0]
    includes = [os.path.join(module_path, 'src', 'include')]  # common headers
    if inc_clustering:
        includes.append(os.path.join(module_path, 'clustering', 'include'))
    if inc_markov:
        includes.append(os.path.join(module_path, 'markov', '_bindings', 'include'))
    if inc_markov_hmm:
        includes.append(os.path.join(module_path, 'markov', 'hmm', '_bindings', 'include'))
    if inc_data:
        includes.append(os.path.join(module_path, 'data', 'include'))
    return includes
