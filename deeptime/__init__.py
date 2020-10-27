from ._version import get_versions

__version__ = get_versions()['version']
del get_versions

from . import clustering
from . import covariance
from . import data
from . import decomposition
from . import markov
from . import numeric
from . import kernels
from . import basis
from . import sindy


def capi_includes():
    import os
    import sys
    module_path = sys.modules['deeptime'].__path__[0]
    includes = [os.path.join(module_path, *rest) for rest in [
        ('src', 'include'),  # common headers
        ('clustering', 'include'),  # clustering headers
        ('markov', '_bindings', 'include'),  # markov module headers
        ('markov', 'hmm', '_bindings', 'include')  # hmm headers
    ]]
    return includes
