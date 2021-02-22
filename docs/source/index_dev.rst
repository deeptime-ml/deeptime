.. _ref-ix-base:

==============
For developers
==============

This document is mainly for aspiring contributors to familiarize themselves with the code structure and underlying API.
When it is planned to add a new estimator / model / transformer, the module `deeptime.base <api/index_base.rst>`__
defines the interfaces, which are described in some more detail subsequently.

Writing a custom estimator
--------------------------
When writing a custom estimator, first one should decide on what is supposed to be estimated. To this end, a
:class:`Model <deeptime.base.Model>` can be implemented. For example, implementing a model which can hold mean values:

.. code-block:: python

    class MeanModel(deeptime.base.Model):

        def __init__(self, mean):
            self._mean = mean

        @property
        def mean(self):
            return self._mean

is nothing more than a dictionary which inherits from :class:`Model`.

Subsequently, an Estimator for :code:`MeanModel` s can be implemented:

.. code-block:: python

    class MeanEstimator(deeptime.base.Estimator):

        def __init__(self, axis=-1):
            super(MeanEstimator, self).__init__()
            self.axis = axis

        def fetch_model(self, data) -> typing.Optional[MeanModel]:
            return self._model

        def fit(self, data):
            self._model = MeanModel(np.mean(data, axis=self.axis))
            return self

Some estimators also offer a :code:`partial_fit`, in which case an existing model is updated.

Now estimator and model can be used:

.. code-block:: python

    data = np.random.normal(size=(100000, 4))
    mean_model = MeanEstimator(axis=-1).fit(data).fetch_model()
    print(mean_model.mean)


Adding transformer capabilities
-------------------------------
Some models have the capability to transform / project data. For example,
:class:`k-means <deeptime.clustering.KMeans>` can be used to transform time series to discrete series of
states by assigning each frame to its respective cluster center.

To add this kind of functionality, one can use the :class:`Transformer` interface and implement the abstract
:meth:`Transformer.transform` method:

.. code-block:: python

    class Projector(deeptime.base.Model, deeptime.base.Transformer):

        def __init__(self, dim):
            self.dim = dim

        def transform(self, data: np.ndarray):
            # projects time series data to "dim"-th dimension
            return data[:, self.dim]

It usually also makes sense to implement the transformer interface for estimators whose models are transformers
by simply calling :code:`self.fetch_model().transform(data)`, i.e., dispatching the transform call to the current model.

Depending on PyTorch
--------------------

If your code depends on pytorch it is no problem to import it at module level (at the top of your implementation file).
To make it accessible to the parent package via `__init__` however, the import should be wrapped into a call to
:func:`module_available <deeptime.util.module_available>` like so

.. code-block:: python

    # ... the init
    from ..util.platform import module_available
    if module_available("torch"):
        from .your_module import MeanEstimator, MeanModel
    del module_available

because there is no hard dependency to PyTorch and functionality should be exposed as available.

Testing your code
-----------------
Tests are designed to be run with `py.test <https://docs.pytest.org/en/stable/>`__ which can be obtained via, e.g., pypi
or conda. All tests (except for doctests) are placed inside the toplevel `tests` directory. The tests directory
is organized in the same way as the deeptime package itself. For example, if you developed a new estimator
:code:`MeanEstimator` in the package :code:`deeptime.some.package`, then tests should go into
:code:`tests/some/package/test_mean_estimator.py`.

To execute the tests a call to :code:`pytest tests/` suffices. To execute doctests,
:code:`pytest --doctest-modules deeptime` can be called.

Documenting the code
--------------------

When documenting your code, `numpydoc style <numpydoc.readthedocs.io>`__ should be used. Going back to the example
of the :code:`MeanEstimator`, this style of documentation would look like the following:

.. code-block:: python

    class MeanEstimator(deeptime.base.Estimator):
        r""" The mean estimator. It estimates the mean using a complicated algorithm
        :cite:`mean-estimator-authorofthecomplicatedalgo1988`.

        Parameters
        ----------
        axis : int, optional, default=-1
            The axis over which to compute the mean. Defaults to -1, which refers to the last axis.

        References
        ----------
        .. bibliography:: /references.bib
            :style: plain
            :filter: docname in docnames
            :keyprefix: mean-estimator-

        See Also
        --------
        MeanModel
        """

        def __init__(self, axis=-1):
            super(MeanEstimator, self).__init__()
            self.axis = axis

        def fetch_model(self, data) -> typing.Optional[MeanModel]:
            r"""Fetches the current model. Can be `None` in case :meth:`fit` was not called yet.

            Returns
            -------
            model : MeanModel or None
                the latest estimated model
            """
            return self._model

        def fit(self, data):
            r""" Performs the estimation.

            Parameters
            ----------
            data : ndarray
                Array over which the mean should be estimated.

            Returns
            -------
            self : MeanEstimator
                Reference to self.
            """
            self._model = MeanModel(np.mean(data, axis=self.axis))
            return self

Note the specific style of using citations. For citations there is a package-global BibTeX file under
:code:`docs/source/references.bib`. These references can then be included into the documentation website
using the citation key as defined in the references file with a unique prefix - in this case :code:`mean-estimator-`.

The documentation website is hosted via GitHub pages, its sources can be found
`here <https://github.com/deeptime-ml/deeptime-ml.github.io>`__. Please see the
`README <https://github.com/deeptime-ml/deeptime/tree/main/docs>`__ on GitHub for instructions on how to build
it.
