from . import _version
__version__ = _version.get_versions()['version']

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

from .util.platform import module_available
if module_available("matplotlib"):
    from . import plots
del module_available


def capi_includes(inc_clustering: bool = False, inc_markov: bool = False, inc_markov_hmm: bool = False,
                  inc_data: bool = False):
    import os
    import sys
    module_path = sys.modules['deeptime'].__path__[0]
    includes = [os.path.join(module_path, 'src', 'include')]  # common headers
    if inc_clustering:
        includes.append(os.path.join(includes[0], 'clustering'))
    if inc_markov:
        includes.append(os.path.join(includes[0], 'markov'))
    if inc_markov_hmm:
        includes.append(os.path.join(includes[0], 'markov', 'hmm'))
    if inc_data:
        includes.append(os.path.join(includes[0], 'data'))
    return includes
