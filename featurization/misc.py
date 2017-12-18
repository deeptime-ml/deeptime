
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
'''
Created on 15.02.2016

@author: marscher
'''
import numpy as np
import mdtraj
from itertools import count
from pyemma.coordinates.data.featurization.util import (_catch_unhashable,
                                                        _describe_atom,
                                                        hash_top, _hash_numpy_array)
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
    _id = count(0)

    def __init__(self, func=None, *args, **kwargs):
        self._func = func
        self._args = args
        self._kwargs = kwargs
        self._dim = kwargs.pop('dim', 0)
        desc = kwargs.pop('description', [])
        if isinstance(desc, str):
            desc = [desc]
        self.id = next(CustomFeature._id)
        if not desc:
            arg_str = "{args}, {kw}" if self._kwargs else "{args}"
            desc = ["CustomFeature[{id}][0] calling {func} with args {arg_str}".format(
                id=self.id,
                func=self._func,
                arg_str=arg_str, args=self._args, kw=self._kwargs)]
            if self.dimension > 1:
                desc.extend(('CustomFeature[{id}][{i}]'.format(id=self.id, i=i) for i in range(1, self.dimension)))
        elif desc and not (len(desc) == self._dim or len(desc) == 1):
            raise ValueError("to avoid confusion, ensure the lengths of 'description' "
                             "list matches dimension - or give a single element which will be repeated."
                             "Input was {}".format(desc))

        if len(desc) == 1:
            desc *= self.dimension

        self._desc = desc

    def describe(self):
        return self._desc

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


class DummyCustomFeature(Feature):
    _serialize_version = 0
    _serialize_fields = ('description', )

    def __init__(self, description):
        self.description = description
        self._warn()

    def transform(self, _):
        self._warn()
        return np.empty_like((0, 0))

    def dimension(self):
        self._warn()
        return 0

    def _warn(self):
        import warnings
        from pyemma.util.exceptions import PyEMMA_UserWarning
        warnings.warn('Please re-add your custom feature again! Description was: {}'.format(self.description[:30]),
                      category=PyEMMA_UserWarning)


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

    def __reduce__(self):
        self._ensure_topfile()
        return SelectionFeature, (self.top.fname, self.indexes)

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
        if isinstance(ref, str):
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


class GroupCOMFeature(Feature):

    def __init__(self, topology, group_definitions, ref_geom=None, image_molecules=False, mass_weighted=True):

        assert ref_geom is None or isinstance(ref_geom, mdtraj.Trajectory), "argument ref_geom has to be either " \
                                                                            "None or and mdtraj.Trajectory, got instead %s"%type(ref_geom)

        self.ref_geom = ref_geom
        self.topology = topology
        self.image_molecules = image_molecules
        self.group_definitions = [np.asarray(gf) for gf in group_definitions]
        self.atom_masses = np.array([aa.element.mass for aa in topology.atoms])

        if mass_weighted:
            self.masses_in_groups = [self.atom_masses[aa_in_rr] for aa_in_rr in self.group_definitions]
        else:
            self.masses_in_groups = [np.ones_like(aa_in_rr) for aa_in_rr in self.group_definitions]

        # Prepare and store the description
        self._describe = []
        for group in self.group_definitions:
            for coor in 'xyz':
                self._describe.append('COM-%s of atom group [%s..%s] '%(coor, group[:3], group[-3:]))
                # TODO consider including the ref_geom and image_molecule arsg here?

        self.__hashed_input__ = hash_top(topology)
        self.__hashed_input__ ^= _hash_numpy_array(np.hstack(self.group_definitions))
        self.__hashed_input__ ^= _hash_numpy_array(np.hstack([len(gd) for gd in self.group_definitions]))
        self.__hashed_input__ ^= hash(tuple((mass_weighted, image_molecules)))
        if ref_geom is not None:
            self.__hashed_input__ ^= _hash_numpy_array(ref_geom.xyz[0])
            # Hashing xyz instead of the top allows for different refs in the same featurizer

    def describe(self):
        return self._describe

    @property
    def dimension(self):
        return 3*len(self.group_definitions)

    def transform(self, traj):
        # TODO: is it possible to avoid copy? Otherwise the trajectory is altered...
        traj_copy = traj[:]
        COM_xyz = []
        if self.ref_geom is not None:
            traj_copy = traj_copy.superpose(self.ref_geom)
        if self.image_molecules:
            traj_copy = traj_copy.image_molecules()
        for aas, mms in zip(self.group_definitions, self.masses_in_groups):
            COM_xyz.append(np.average(traj_copy.xyz[:, aas, ], axis=1, weights=mms))
        return np.hstack(COM_xyz)

    def __hash__(self):
        hash_value = hash(self.__hashed_input__)

        return hash_value

class ResidueCOMFeature(GroupCOMFeature):

    def __init__(self, topology, residue_indices, residue_atoms, scheme, ref_geom=None, image_molecules=False, mass_weighted = True):
        GroupCOMFeature.__init__(self, topology, residue_atoms, mass_weighted=mass_weighted, ref_geom=ref_geom, image_molecules=image_molecules)

        # This are the only extra attributes that residueCOMFeature should have
        self.residue_indices = residue_indices
        self.scheme = scheme

        # Add the scheme to the hash
        self.__hashed_input__ ^= hash(scheme)

        # Overwrite the self._describe attribute, this way the method of the superclass can be used "as is"
        self._describe = []
        for ri in self.residue_indices:
            for coor in 'xyz':
                self._describe.append('%s COM-%s (%s)' % (topology.residue(ri), coor, self.scheme))
                # TODO consider including the ref_geom and image_molecule arsg here?
