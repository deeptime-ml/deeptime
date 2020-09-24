===========
Scikit-time
===========

Scikit-time is a Python library for analysis for time series data. In particular, methods for dimensionality reduction,
clustering, and Markov model estimation are implemented.

The API is similar to that of `scikit-learn <https://scikit-learn.org/>`__ and offers basic compatibility to the tools
offered by scikit-learn.

Get started
-----------
To get yourself started, the major tools offered by this library are covered and largely explained behind
the following links. If you are new to a method and want to find an entry into how and when to use it, we recommend
these.

They are structured following a possible workflow of a timeseries analysis. That is,

    1. reducing the dimension of the dataset if it is very high-dimensional by projecting onto "slow"
       degrees of freedom,
    2. clustering the dataset frame-wise to obtain a discrete time series for which structural transitions
       can be analyzed,
    3. and estimating an msm or hmm based on said discrete time series for further analysis.

.. only:: notebooks

    .. toctree::
        :maxdepth: 3
        :caption: Documentation
        :hidden:

        index_dimreduction
        notebooks/clustering
        index_msm
        notebooks/hmm
        examples/index


.. toctree::
   :caption: API docs
   :maxdepth: 3
   :hidden:

   api/index_base
   api/index_clustering
   api/index_covariance
   api/index_decomposition
   api/index_markov
   api/index_markov_hmm
   api/index_markov_tools
   api/index_basis
   api/index_kernels
   api/index_data
   api/index_numeric

.. toctree::
   :caption: Other
   :maxdepth: 1
   :hidden:

   changelog
   imprint
   license
