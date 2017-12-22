
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
from itertools import count

import mdtraj
import numpy as np

from pyemma.coordinates.data.featurization._base import Feature
from pyemma.coordinates.data.featurization.util import (_describe_atom,
                                                        cmp_traj)


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
    __serialize_version = 0
    __serialize_fields = ('_desc',)

    def __init__(self, fun, dim, description=None, fun_args=(), fun_kwargs=None):
        # we set topology to None here
        self.top = None
        if fun_kwargs is None:
            fun_kwargs = {}
        self._fun = fun
        self._args = fun_args
        self._kwargs = fun_kwargs
        self._dim = dim
        desc = description
        if isinstance(desc, str):
            desc = [desc]
        self.id = next(CustomFeature._id)
        if not desc:
            arg_str = "{args}, {kw}" if self._kwargs else "{args}"
            desc = ["CustomFeature[{id}][0] calling {fun} with args {arg_str}".format(
                id=self.id,
                fun=self._fun,
                arg_str=arg_str, args=self._args, kw=self._kwargs)]
            #if self.dimension > 1:
            #    desc.extend(('CustomFeature[{id}][{i}]'.format(id=self.id, i=i) for i in range(1, self.dimension)))
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
        feature = self._fun(traj, *self._args, **self._kwargs)
        if not isinstance(feature, np.ndarray):
            raise ValueError("your function should return a NumPy array!")
        return feature

    def __eq__(self, other):
        eq = super(CustomFeature, self).__eq__(other)
        if not eq or not isinstance(other, CustomFeature):
            return False
        return self._fun == other._fun and self._args == other._args and self._kwargs == other._kwargs

    def __getstate__(self):
        import warnings
        warnings.warn('We can not save custom functions by now and probably never will. '
                      'Please re-add your custom function after you have restored your Featurizer.')
        return super(CustomFeature, self).__getstate__()

    def __setstate__(self, state):
        super(CustomFeature, self).__setstate__(state)

        def _warn(_):
            raise NotImplementedError('Please re-add your custom feature again! Description was: {}\n'
                                      '>>> featurizer.remove_all_custom_funcs()\n'
                                      '>>> featurizer.add_custom_func(...)'
                                      .format(self.describe()[:30]))
        self._fun = _warn
        self._args = ()
        self._kwargs = {}


class SelectionFeature(Feature):

    """
    Just provide the cartesian coordinates of a selection of atoms (could be simply all atoms).
    The coordinates are flattened as follows: [x1, y1, z1, x2, y2, z2, ...]

    """
    __serialize_version = 0
    __serialize_fields = ('indexes',)
    prefix_label = "ATOM:"

    def __init__(self, top, indexes):
        self.top = top
        self.indexes = np.array(indexes)
        if len(self.indexes) == 0:
            raise ValueError("empty indices")
        self.dimension = 3 * len(indexes)

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

    def transform(self, traj):
        newshape = (traj.xyz.shape[0], 3 * self.indexes.shape[0])
        return np.reshape(traj.xyz[:, self.indexes, :], newshape)

    def __eq__(self, other):
        eq = super(SelectionFeature, self).__eq__(other)
        if not eq or not isinstance(other, SelectionFeature):
            return False
        return np.all(self.indexes == other.indexes)


class MinRmsdFeature(Feature):

    __serialize_version = 0
    __serialize_fields = ('ref', 'ref_frame', 'name', 'precentered', 'atom_indices',)

    def __init__(self, ref, ref_frame=0, atom_indices=None, topology=None, precentered=False):
        self.top = topology

        assert isinstance(
            ref_frame, int), "ref_frame has to be of type integer, and not %s" % type(ref_frame)

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
        self.dimension = 1

    def describe(self):
        label = "minrmsd to frame %u of %s" % (self.ref_frame, self.name)
        if self.precentered:
            label += ', precentered=True'
        if self.atom_indices is not None:
            label += ', subset of atoms  '
        return [label]

    def transform(self, traj):
        return np.array(mdtraj.rmsd(traj, self.ref, atom_indices=self.atom_indices), ndmin=2).T

    def __eq__(self, other):
        if not isinstance(other, MinRmsdFeature):
            return False
        eq = super(MinRmsdFeature, self).__eq__(other)
        return (eq and cmp_traj(self.ref, other.ref)
                and self.ref_frame == other.ref_frame
                and np.all(self.atom_indices == other.atom_indices)
                and self.precentered == other.precentered
                )


class AlignFeature(SelectionFeature):
    __serialize_version = 0
    __serialize_fields = ('ref', 'atom_indices', 'ref_atom_indices', 'in_place')

    prefix_label = 'aligned ATOM:'

    def __init__(self, reference, indexes, atom_indices=None, ref_atom_indices=None, in_place=True):
        super(AlignFeature, self).__init__(top=reference.topology, indexes=indexes)
        self.ref = reference
        self.atom_indices = atom_indices
        self.ref_atom_indices = ref_atom_indices
        self.in_place = in_place

    def __eq__(self, other):
        if not isinstance(other, AlignFeature):
            return False
        return (cmp_traj(self.ref, other.ref)
                and np.all(self.atom_indices == other.atom_indices)
                and self.in_place == other.in_place)

    def transform(self, traj):
        if not self.in_place:
            traj = traj.slice(slice(None), copy=True)
        aligned = traj.superpose(reference=self.ref, atom_indices=self.atom_indices,
                                 ref_atom_indices=self.ref_atom_indices)
        # apply selection
        return super(AlignFeature, self).transform(aligned)


class GroupCOMFeature(Feature):
    __serialize_version = 0
    __serialize_fields = ('ref_geom', 'image_molecules', 'group_definitions', 'atom_masses',
                          'masses_in_groups', '_describe')

    def __init__(self, topology, group_definitions, ref_geom=None, image_molecules=False, mass_weighted=True):
        if not (ref_geom is None or isinstance(ref_geom, mdtraj.Trajectory)):
            raise ValueError("argument ref_geom has to be either None or and mdtraj.Trajectory,"
                             " got instead %s" % type(ref_geom))

        self.ref_geom = ref_geom
        self.top = topology
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
        self.dimension = 3 * len(self.group_definitions)

    def describe(self):
        return self._describe

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

    def __eq__(self, other):
        eq = super(GroupCOMFeature, self).__eq__(other)
        if not eq or not isinstance(other, GroupCOMFeature):
            return False
        return (cmp_traj(self.ref_geom, other.ref_geom) and self.image_molecules == other.image_molecules
                and all(np.array_equal(g1, g2) for g1, g2 in zip(self.group_definitions, other.group_definitions))
                and all(np.array_equal(m1, m2) for m1, m2 in zip(self.masses_in_groups, other.masses_in_groups))
        )


class ResidueCOMFeature(GroupCOMFeature):

    __serialize_version = 0
    __serialize_fields = ('residue_indices', 'scheme')

    def __init__(self, topology, residue_indices, residue_atoms, scheme, ref_geom=None, image_molecules=False,
                 mass_weighted=True):
        super(ResidueCOMFeature, self).__init__(topology, residue_atoms, mass_weighted=mass_weighted, ref_geom=ref_geom,
                                                image_molecules=image_molecules)

        # This are the only extra attributes that residueCOMFeature should have
        self.residue_indices = residue_indices
        self.scheme = scheme

        # Overwrite the self._describe attribute, this way the method of the superclass can be used "as is"
        self._describe = []
        for ri in self.residue_indices:
            for coor in 'xyz':
                self._describe.append('%s COM-%s (%s)' % (topology.residue(ri), coor, self.scheme))
                # TODO consider including the ref_geom and image_molecule arsg here?

    def __eq__(self, other):
        eq = super(ResidueCOMFeature, self).__eq__(other)
        if not eq or not isinstance(other, ResidueCOMFeature):
            return False
        return np.all(self.residue_indices == other.residue_indices) and self.scheme == other.scheme
