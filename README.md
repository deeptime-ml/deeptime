# deeptime

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0) [![Build Status](https://dev.azure.com/clonker/deeptime/_apis/build/status/deeptime-ml.deeptime?branchName=main)](https://dev.azure.com/clonker/deeptime/_build/latest?definitionId=1&branchName=main) [![codecov](https://codecov.io/gh/deeptime-ml/deeptime/branch/main/graph/badge.svg?token=MgQZqDM4sK)](https://codecov.io/gh/deeptime-ml/deeptime) [![DOI](https://img.shields.io/badge/DOI-10.1088%2F2632--2153%2Fac3de0-blue)](https://doi.org/10.1088/2632-2153/ac3de0)

Releases:

Installation via conda recommended.

|  [![conda-forge](https://img.shields.io/conda/v/conda-forge/deeptime?color=brightgreen&label=conda-forge)](https://github.com/conda-forge/deeptime-feedstock) 	|   [![PyPI](https://badge.fury.io/py/deeptime.svg)](https://pypi.org/project/deeptime)	|
|:-:	|:-:	|
|  `conda install -c conda-forge deeptime` |  `pip install deeptime`  	|

Documentation: [deeptime-ml.github.io](https://deeptime-ml.github.io/).

## Building the latest trunk version of the package:

Using pip with a local clone and pulling dependencies:
```
git clone https://github.com/deeptime-ml/deeptime.git

cd deeptime
pip install -r tests/requirements.txt
pip install -e .
```

Or using pip directly on the remote:
```
pip install git+https://github.com/deeptime-ml/deeptime.git@main
```
