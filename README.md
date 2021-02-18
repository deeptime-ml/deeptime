# deeptime

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0) [![Build Status](https://dev.azure.com/clonker/deeptime/_apis/build/status/deeptime-ml.deeptime?branchName=main)](https://dev.azure.com/clonker/deeptime/_build/latest?definitionId=1&branchName=main)  

Releases:

|  [![PyPI](https://badge.fury.io/py/deeptime.svg)](https://pypi.org/project/deeptime) 	|  [![conda-forge](https://img.shields.io/conda/v/conda-forge/deeptime?color=brightgreen&label=conda-forge)](https://github.com/conda-forge/deeptime-feedstock) 	|
|:-:	|:-:	|
|  `pip install deeptime` 	|  `conda install -c conda-forge deeptime` 	|

Documentation: [deeptime-ml.github.io](https://deeptime-ml.github.io/).

### Building the latest trunk version of the package:
```
git clone https://github.com/deeptime-ml/deeptime.git

cd deeptime
git submodule update --init

conda install numpy scipy cython scikit-learn pybind11

python setup.py install
```

or 

```
pip install git+https://github.com/deeptime-ml/deeptime.git@main
```
