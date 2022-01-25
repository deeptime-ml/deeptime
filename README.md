# deeptime

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0) [![Build Status](https://dev.azure.com/clonker/deeptime/_apis/build/status/deeptime-ml.deeptime?branchName=main)](https://dev.azure.com/clonker/deeptime/_build/latest?definitionId=1&branchName=main) [![codecov](https://codecov.io/gh/deeptime-ml/deeptime/branch/main/graph/badge.svg?token=MgQZqDM4sK)](https://codecov.io/gh/deeptime-ml/deeptime) [![DOI](https://img.shields.io/badge/DOI-10.1088%2F2632--2153%2Fac3de0-blue)](https://doi.org/10.1088/2632-2153/ac3de0)

Deeptime is a general purpose Python library offering various tools to estimate dynamical models 
based on time-series data including conventional linear learning methods, such as Markov State 
Models (MSMs), Hidden Markov Models (HMMs) and Koopman models, as well as kernel and 
deep learning approaches such as VAMPnets and deep MSMs. The library is largely compatible 
with scikit-learn, having a range of Estimator classes for these different models, but in 
contrast to scikit-learn also provides Model classes, e.g., in the case of an MSM, 
which provide a multitude of analysis methods to compute interesting thermodynamic, kinetic 
and dynamical quantities, such as free energies, relaxation times and transition paths.

Releases:

Installation via `conda` recommended, `pip` compiles the library locally.

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
