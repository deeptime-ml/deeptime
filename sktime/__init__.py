from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from . import clustering
from . import covariance
from . import data
from . import decomposition
from . import markov
from . import numeric
