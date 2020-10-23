deeptime
========

[![Build Status](https://dev.azure.com/marscher/sktime/_apis/build/status/scikit-time.scikit-time?branchName=master)](https://dev.azure.com/marscher/sktime/_build/latest?definitionId=1&branchName=master)

For usage, please refer to [deeptime-ml.github.io](https://deeptime-ml.github.io/).

Installation instructions until first release is on conda-forge and pip:

```
git clone https://github.com/deeptime-ml/deeptime.git

cd deeptime
git submodule update --init

conda install numpy scipy cython scikit-learn

python setup.py install
```

or 

```
pip install git+https://github.com/deeptime-ml/deeptime.git@master
```
