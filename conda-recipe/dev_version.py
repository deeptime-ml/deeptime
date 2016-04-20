from __future__ import print_function
from pyemma import __version__ as version

with open('__conda_version__.txt', 'w') as f:
    f.write(version)
