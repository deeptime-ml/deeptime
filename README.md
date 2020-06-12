scikit-time
===========
For usage, please refer to [scikit-time.github.io](https://scikit-time.github.io/).

Installation instructions until first release is on conda-forge and pip:

```
git clone https://github.com/scikit-time/scikit-time.git

cd scikit-time
git submodule update --init

conda install numpy scipy cython
conda install msmtools -c conda-forge

python setup.py install
```
