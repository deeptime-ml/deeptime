from typing import Tuple, Any, Dict

from . import _clustering_bindings as _bd


class MetricRegistry:
    r""" Registry of available metrics. Per default this contains only the Euclidean metric.
    If a custom metric is implemented, it can be registered through a call to
    :meth:`register <deeptime.clustering.MetricRegistry.register>`.

    Note that the registry should not be instantiated directly but rather be accessed
    through :data:`metrics <deeptime.clustering.metrics>`.
    """

    def __init__(self):
        self._registered = None
        self.register("euclidean", _bd.EuclideanMetric)

    def register(self, name: str, clazz):
        r""" Adds a new metric to the registry.

        Parameters
        ----------
        name : str
            The name of the metric.
        clazz : class
            Reference to the class of the metric.
        """
        self._mapping[name] = clazz

    @property
    def available(self) -> Tuple[str]:
        r"""List of registered metrics."""
        return tuple(self._mapping.keys())

    @property
    def _mapping(self) -> Dict[str, Any]:
        if self._registered is None:
            self._registered = dict()
        return self._registered

    def __getitem__(self, item: str):
        return self._mapping[item]


r""" Singleton instance of metrics registry. """
metrics = MetricRegistry()
