# This file is part of scikit-time
#
# Copyright (c) 2020 AI4Science Group, Freie Universitaet Berlin (GER)
#
# scikit-time is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from typing import Tuple, Any, Dict

from . import _clustering_bindings as _bd


class MetricRegistry(object):
    r""" Registry of available metrics. Per default this contains only the Euclidean metric.
    If a custom metric is implemented, it can be registered through a call to
    :meth:`register <sktime.clustering.MetricRegistry.register>`. """

    def __init__(self):
        r""" Creates a new metrics instance. The registry should not be instantiated directly
        but rather be accessed through :data:`metrics <sktime.clustering.metrics>`."""
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
