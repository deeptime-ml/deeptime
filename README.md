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

Installation via `conda` or `pip`. Both provide compiled binaries for Linux, Windows, and MacOS (x86_64 and arm64).

|  [![conda-forge](https://img.shields.io/conda/v/conda-forge/deeptime?color=brightgreen&label=conda-forge)](https://github.com/conda-forge/deeptime-feedstock) 	|   [![PyPI](https://badge.fury.io/py/deeptime.svg)](https://pypi.org/project/deeptime)	|
|:-:	|:-:	|
|  `conda install -c conda-forge deeptime` |  `pip install deeptime`  	|

Documentation: [deeptime-ml.github.io](https://deeptime-ml.github.io/).

## Main components of deeptime

|  <!-- -->  |  <!-- -->  |  <!-- -->  |
|  :---:	|  :---:  |  :---:  |
|  Dimension reduction  |  Deep dimension reduction  |  SINDy  |
|  [![Dimension reduction](https://user-images.githubusercontent.com/1685266/208686380-087687e0-4dfa-4d27-a2a0-957c33566276.png)](https://deeptime-ml.github.io/latest/index_dimreduction.html) |  [![Deep dimension reduction](https://user-images.githubusercontent.com/1685266/208686212-f84f0a5b-a014-49d1-a469-dfa8a661d555.png)](https://deeptime-ml.github.io/latest/index_deepdimreduction.html)  |  [![SINDy](https://user-images.githubusercontent.com/1685266/208684380-d0234430-50fb-4a62-8d97-73ce1ebf2832.png)](https://deeptime-ml.github.io/latest/notebooks/sindy.html)  |
|  Markov state models  |  Hidden Markov models  |  Datasets  |
|  [![MSMs](https://user-images.githubusercontent.com/1685266/208686588-2e8b960b-06b0-4633-93a6-5df1e5b63209.png)](https://deeptime-ml.github.io/latest/index_msm.html) |  [![HMMs](https://user-images.githubusercontent.com/1685266/208683917-ef7acb41-062c-4503-b48d-dc7718779d9a.png)](https://deeptime-ml.github.io/latest/notebooks/hmm.html)  |  [![Datasets](https://user-images.githubusercontent.com/1685266/208684805-45c82242-6a8c-43f1-88b8-add1af4e7438.png)](https://deeptime-ml.github.io/latest/index_datasets.html)  |

## Building the latest trunk version of the package:

Using pip with a local clone and pulling dependencies:
```
git clone https://github.com/deeptime-ml/deeptime.git

cd deeptime
pip install .
```

Or using pip directly on the remote:
```
pip install git+https://github.com/deeptime-ml/deeptime.git@main
```
