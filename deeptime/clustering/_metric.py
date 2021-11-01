from typing import Tuple, Any, Dict

from . import _clustering_bindings as _bd


class MetricRegistry:
    r""" Registry of available metrics. Per default this contains only the Euclidean metric.
    If a custom metric is implemented, it can be registered through a call to
    :meth:`register <deeptime.clustering.MetricRegistry.register>`.

    .. note::

        The registry should not be instantiated directly but rather be accessed
        through the :data:`metrics <deeptime.clustering.metrics>` singleton.


    .. rubric:: Adding a new metric

    A new metric may be added by linking against the deeptime clustering c++ library (directory is provided by
    `deeptime.capi_includes(inc_clustering=True)`) and subsequently exposing the clustering algorithms with your custom
    metric like

    .. code-block:: cpp

        #include "register_clustering.h"

        PYBIND11_MODULE(_clustering_bindings, m) {
            m.doc() = "module containing clustering algorithms.";
            auto customModule = m.def_submodule("custom");
            deeptime::clustering::registerClusteringImplementation<Custom>(customModule);
        }

    and registering it with the deeptime library through

    .. code-block:: python

        import deeptime
        import bindings  # this is your compiled extension, rename as appropriate

        deeptime.clustering.metrics.register("custom", bindings.custom)
    """

    def __init__(self):
        self._registered = None
        self.register("euclidean", _bd.euclidean)

    def register(self, name: str, impl):
        r""" Adds a new metric to the registry.

        Parameters
        ----------
        name : str
            The name of the metric.
        impl : module
            Reference to the implementation module.
        """
        self._mapping[name] = impl

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
