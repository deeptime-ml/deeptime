# This file is part of scikit-time
#
# Copyright (c) 2020 AI4Science Group, Freie Universitaet Berlin (GER)
#
# scikit-time is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


from ._version import get_versions

__version__ = get_versions()['version']
del get_versions

from . import clustering
from . import covariance
from . import data
from . import decomposition
from . import markov
from . import numeric


def capi_includes():
    import os
    import sys
    module_path = sys.modules['sktime'].__path__[0]
    includes = [os.path.join(module_path, *rest) for rest in [
        ('src', 'include'),  # common headers
        ('clustering', 'include'),  # clustering headers
        ('markov', '_bindings', 'include'),  # markov module headers
        ('markov', 'hmm', '_bindings', 'include')  # hmm headers
    ]]
    return includes
