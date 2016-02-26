# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
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
import six
'''
Created on 15.02.2016

@author: marscher
'''
from pyemma.coordinates.data.featurization.util import (_catch_unhashable,
                                                        _describe_atom,
                                                        hash_top, _hash_numpy_array)
import numpy as np
import mdtraj
from pyemma.coordinates.data.featurization._base import Feature


class CustomFeature(Feature):

    """
    A CustomFeature is the base class for user-defined features. If you want to
    implement a new fancy feature, derive from this class, calculate the quantity
    of interest in the map method and return it as an ndarray.

    If you have defined a map function that should be classed, you don't need to derive a class, but you
    can simply pass a function to the constructor of this class


    Parameters
    ----------
    func : function
        will be invoked with given args and kwargs on mapping traj
    args : list of positional args (optional) passed to func
    kwargs : named arguments (optional) passed to func

    Notes
    -----
    Your passed in function will get a mdtraj.Trajectory object as first argument.

    Examples
    --------
    We define a feature that transforms all coordinates by :math:`1 / x^2`:

    >>> from pyemma.coordinates import source
    >>> from pyemma.datasets import get_bpti_test_data
    >>> inp = get_bpti_test_data()

    Define a function which transforms the coordinates of the trajectory object.
    Note that you need to define the output dimension, which we pass directly in
    the feature construction. The trajectory contains 58 atoms, so the output
    dimension will be 3 * 58 = 174:

    >>> my_feature = CustomFeature(lambda x: (1.0 / x.xyz**2).reshape(-1, 174), dim=174)
    >>> reader = source(inp['trajs'][0], top=inp['top'])

    pass the feature to the featurizer and transform the data

    >>> reader.featurizer.add_custom_feature(my_feature)
    >>> data = reader.get_output()

    """

    def __init__(self, func=None, *args, **kwargs):
        self._func = func
        self._args = args
        self._kwargs = kwargs
        self._dim = kwargs.pop('dim', 0)

    def describe(self):
        return ["CustomFeature calling %s with args %s" % (str(self._func),
                                                           str(self._args) + 
                                                           str(self._kwargs))]

    def transform(self, traj):
        feature = self._func(traj, *self._args, **self._kwargs)
        if not isinstance(feature, np.ndarray):
            raise ValueError("your function should return a NumPy array!")
        return feature

    def __hash__(self):
        hash_value = hash(self._func)
        # if key contains numpy arrays, we hash their data arrays
        key = tuple(list(map(_catch_unhashable, self._args)) + 
                    list(map(_catch_unhashable, sorted(self._kwargs.items()))))
        hash_value ^= hash(key)
        return hash_value


class SelectionFeature(Feature):

    """
    Just provide the cartesian coordinates of a selection of atoms (could be simply all atoms).
    The coordinates are flattened as follows: [x1, y1, z1, x2, y2, z2, ...]

    """
    def __init__(self, top, indexes):
        self.top = top
        self.indexes = np.array(indexes)
        if len(self.indexes) == 0:
            raise ValueError("empty indices")
        self.prefix_label = "ATOM:"

    def describe(self):
        labels = []
        for i in self.indexes:
            labels.append("%s%s x" % 
                          (self.prefix_label, _describe_atom(self.top, i)))
            labels.append("%s%s y" % 
                          (self.prefix_label, _describe_atom(self.top, i)))
            labels.append("%s%s z" % 
                          (self.prefix_label, _describe_atom(self.top, i)))
        return labels

    @property
    def dimension(self):
        return 3 * self.indexes.shape[0]

    def transform(self, traj):
        newshape = (traj.xyz.shape[0], 3 * self.indexes.shape[0])
        return np.reshape(traj.xyz[:, self.indexes, :], newshape)

    def __hash__(self):
        hash_value = hash(self.prefix_label)
        hash_value ^= hash_top(self.top)
        hash_value ^= _hash_numpy_array(self.indexes)

        return hash_value


class MinRmsdFeature(Feature):

    def __init__(self, ref, ref_frame=0, atom_indices=None, topology=None, precentered=False):

        assert isinstance(
            ref_frame, int), "ref_frame has to be of type integer, and not %s" % type(ref_frame)

        # Will be needing the hashed input parameter
        self.__hashed_input__ = hash(ref)

        # Types of inputs
        # 1. Filename+top
        if isinstance(ref, six.string_types):
            # Store the filename
            self.name = ref[:]
            ref = mdtraj.load_frame(ref, ref_frame, top=topology)
            # mdtraj is pretty good handling exceptions, we're not checking for
            # types or anything here

        # 2. md.Trajectory object
        elif isinstance(ref, mdtraj.Trajectory):
            self.name = ref.__repr__()[:]
        else:
            raise TypeError("input reference has to be either a filename or "
                            "a mdtraj.Trajectory object, and not of %s" % type(ref))

        self.ref = ref
        self.ref_frame = ref_frame
        self.atom_indices = atom_indices
        self.precentered = precentered

    def describe(self):
        label = "minrmsd to frame %u of %s" % (self.ref_frame, self.name)
        if self.precentered:
            label += ', precentered=True'
        if self.atom_indices is not None:
            label += ', subset of atoms  '
        return [label]

    @property
    def dimension(self):
        return 1

    def transform(self, traj):
        return np.array(mdtraj.rmsd(traj, self.ref, atom_indices=self.atom_indices), ndmin=2).T

    def __hash__(self):
        hash_value = hash(self.__hashed_input__)
        # TODO: identical md.Trajectory objects have different hashes need a
        # way to differentiate them here
        hash_value ^= hash(self.ref_frame)
        if self.atom_indices is None:
            hash_value ^= _hash_numpy_array(np.arange(self.ref.n_atoms))
        else:
            hash_value ^= _hash_numpy_array(np.array(self.atom_indices))
        hash_value ^= hash(self.precentered)

        return hash_value
