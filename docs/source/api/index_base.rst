.. _ref-base:
.. currentmodule:: deeptime.base

=============
deeptime.base
=============
This module contains all base classes of deeptime. They are the important when developing new estimators.

Writing a custom estimator
--------------------------
When writing a custom estimator, first one should decide on what is supposed to be estimated. To this end, a
:class:`Model` can be implemented. For example, implementing a model which can hold mean values:

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

        def __init__(axis=-1):
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

.. currentmodule:: deeptime.base
.. autosummary::
   :toctree: generated/
   :template: class_nomodule.rst

   Model
   Estimator

Adding transformer capabilities
-------------------------------
Some models have the capability to transform / project data. For example,
:class:`k-means <deeptime.clustering.KmeansClustering>` can be used to transform time series to discrete series of
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

.. currentmodule:: deeptime.base
.. autosummary::
   :toctree: generated/
   :template: class_nomodule.rst

   Transformer
