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


from __future__ import absolute_import
import mdtraj
from mdtraj.geometry.dihedral import _get_indices_phi, \
    _get_indices_psi, compute_dihedrals, _atom_sequence, CHI1_ATOMS

import numpy as np
import warnings
from itertools import combinations as _combinations, count
from itertools import product as _product
from pyemma.util.types import is_iterable_of_int as _is_iterable_of_int
import functools


from six import PY3
from pyemma.util.types import is_iterable_of_int as _is_iterable_of_int
from pyemma._base.logging import instance_name, create_logger

from pyemma.util.annotators import deprecated
from six.moves import map
from six.moves import range
from six.moves import zip


__author__ = 'Frank Noe, Martin Scherer'
__all__ = ['MDFeaturizer',
           ]


def _get_indices_chi1(traj):
    rids, indices = list(zip(*(_atom_sequence(traj, atoms) for atoms in CHI1_ATOMS)))
    id_sort = np.argsort(np.concatenate(rids))
    if not any(x.size for x in indices):
        return np.empty(shape=(0, 4), dtype=np.int)

    indices = np.vstack(x for x in indices if x.size)[id_sort]
    return id_sort, indices

# this is needed for get_indices functions, since they expect a Trajectory,
# not a Topology
class fake_traj(object):
    def __init__(self, top):
        self.top = top


def _describe_atom(topology, index):
    """
    Returns a string describing the given atom

    :param topology:
    :param index:
    :return:
    """
    #assert isinstance(index, int)
    at = topology.atom(index)
    return "%s %i %s %i" % (at.residue.name, at.residue.index, at.name, at.index)

def _catch_unhashable(x):
    if hasattr(x, '__getitem__'):
        res = list(x)
        for i, value in enumerate(x):
            if isinstance(value, np.ndarray):
                res[i] = _hash_numpy_array(value)
            else:
                res[i] = value
        return tuple(res)
    elif isinstance(x, np.ndarray):
        return _hash_numpy_array(x)

    return x

if PY3:
    def _hash_numpy_array(x):
        hash_value = hash(x.shape)
        hash_value ^= hash(x.strides)
        hash_value ^= hash(x.data.tobytes())
        return hash_value
else:
    def _hash_numpy_array(x):
        writeable = x.flags.writeable
        try:
            x.flags.writeable = False
            hash_value = hash(x.shape)
            hash_value ^= hash(x.strides)
            hash_value ^= hash(x.data)
        finally:
            x.flags.writeable = writeable
        return hash_value

def hash_top(top):
    if not PY3:
        return hash(top)
    else:
        # this is a temporary workaround for py3
        hash_value = hash(top.n_atoms)
        hash_value ^= hash(tuple(top.atoms))
        hash_value ^= hash(tuple(top.residues))
        hash_value ^= hash(tuple(top.bonds))
        return hash_value


def _parse_pairwise_input(indices1, indices2, MDlogger, fname=''):
    r"""For input of pairwise type (distances, inverse distances, contacts) checks the
        type of input the user gave and reformats it so that :py:func:`DistanceFeature`,
        :py:func:`InverseDistanceFeature`, and ContactFeature can work.

        In case the input isn't already a list of distances, this function will:
            - sort the indices1 array
            - check for duplicates within the indices1 array
            - sort the indices2 array
            - check for duplicates within the indices2 array
            - check for duplicates between the indices1 and indices2 array
            - if indices2 is     None, produce a list of pairs of indices in indices1, or
            - if indices2 is not None, produce a list of pairs of (i,j) where i comes from indices1, and j from indices2

        """

    if _is_iterable_of_int(indices1):
        MDlogger.warning('The 1D arrays input for %s have been sorted, and '
                         'index duplicates have been eliminated.\n'
                         'Check the output of describe() to see the actual order of the features' % fname)

        # Eliminate duplicates and sort
        indices1 = np.unique(indices1)

        # Intra-group distances
        if indices2 is None:
            atom_pairs = np.array(list(_combinations(indices1, 2)))

        # Inter-group distances
        elif _is_iterable_of_int(indices2):

            # Eliminate duplicates and sort
            indices2 = np.unique(indices2)

            # Eliminate duplicates between indices1 and indices1
            uniqs = np.in1d(indices2, indices1, invert=True)
            indices2 = indices2[uniqs]
            atom_pairs = np.asarray(list(_product(indices1, indices2)))

    else:
        atom_pairs = indices1

    return atom_pairs

def _parse_groupwise_input(group_definitions, group_pairs, MDlogger, mname=''):
    r"""For input of group type (add_group_mindist), prepare the array of pairs of indices
        and groups so that :py:func:`MinDistanceFeature` can work

        This function will:
            - check the input types
            - sort the 1D arrays of each entry of group_definitions
            - check for duplicates within each group_definition
            - produce the list of pairs for all needed distances
            - produce a list that maps each entry in the pairlist to a given group of distances

    Returns
    --------
        parsed_group_definitions: list
            List of of 1D arrays containing sorted, unique atom indices

        parsed_group_pairs: numpy.ndarray
            (N,2)-numpy array containing pairs of indices that represent pairs
             of groups for which the inter-group distance-pairs will be generated

        distance_pairs: numpy.ndarray
            (M,2)-numpy array with all the distance-pairs needed (regardless of their group)

        group_membership: numpy.ndarray
            (N,2)-numpy array mapping each pair in distance_pairs to their associated group pair

        """

    assert isinstance(group_definitions, list), "group_definitions has to be of type list, not %s"%type(group_definitions)
    # Handle the special case of just one group
    if len(group_definitions) == 1:
        group_pairs = np.array([0,0], ndmin=2)

    # Sort the elements within each group
    parsed_group_definitions = []
    for igroup in group_definitions:
        assert np.ndim(igroup) == 1, "The elements of the groups definition have to be of dim 1, not %u"%np.ndim(igroup)
        parsed_group_definitions.append(np.unique(igroup))

    # Check for group duplicates
    for ii, igroup in enumerate(parsed_group_definitions[:-1]):
        for jj, jgroup in enumerate(parsed_group_definitions[ii+1:]):
            if len(igroup) == len(jgroup):
                assert not np.allclose(igroup, jgroup), "Some group definitions appear to be duplicated, e.g %u and %u"%(ii,ii+jj+1)

    # Create and/or check the pair-list
    if group_pairs == 'all':
        parsed_group_pairs = np.array(list(_combinations(np.arange(len(group_definitions)), 2)))
    else:
        assert isinstance(group_pairs, np.ndarray)
        assert group_pairs.shape[1] == 2
        assert group_pairs.max() <= len(parsed_group_definitions), "Cannot ask for group nr. %u if group_definitions only " \
                                                    "contains %u groups"%(group_pairs.max(), len(parsed_group_definitions))
        assert group_pairs.min() >= 0, "Group pairs contains negative group indices"

        parsed_group_pairs = np.zeros_like(group_pairs, dtype='int')
        for ii, ipair in enumerate(group_pairs):
            if ipair[0] == ipair[1]:
                MDlogger.warning("%s will compute the mindist of group %u with itself. Is this wanted? "%(mname, ipair[0]))
            parsed_group_pairs[ii,:] = np.sort(ipair)

    # Create the large list of distances that will be computed, and an array containing group identfiers
    # of the distances that actually characterize a pair of groups
    distance_pairs = []
    group_membership = np.zeros_like(parsed_group_pairs)
    b = 0
    for ii, pair in enumerate(parsed_group_pairs):
        if pair[0] != pair[1]:
            distance_pairs.append(list(_product(parsed_group_definitions[pair[0]],
                                                        parsed_group_definitions[pair[1]])))
        else:
            distance_pairs.append(list(_combinations(parsed_group_definitions[pair[0]], 2)))

        group_membership[ii,:] = [b, b+len(distance_pairs[ii])]
        b += len(distance_pairs[ii])

    return parsed_group_definitions, parsed_group_pairs, np.vstack(distance_pairs), group_membership

class CustomFeature(object):

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
        self.dimension = kwargs.pop('dim', 0)

    def describe(self):
        return ["CustomFeature calling %s with args %s" % (str(self._func),
                                                           str(self._args) +
                                                           str(self._kwargs))]

    @deprecated
    def map(self, traj):
        r"""Deprecated: use transform(traj)

        """
        return self.transform(traj)

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

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


class SelectionFeature(object):

    """
    Just provide the cartesian coordinates of a selection of atoms (could be simply all atoms).
    The coordinates are flattened as follows: [x1, y1, z1, x2, y2, z2, ...]

    """
    # TODO: Needs an orientation option

    def __init__(self, top, indexes):
        self.top = top
        self.indexes = np.array(indexes)
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

    @deprecated
    def map(self, traj):
        r"""Deprecated: use transform(traj)

        """
        return self.transform(traj)

    def transform(self, traj):
        newshape = (traj.xyz.shape[0], 3 * self.indexes.shape[0])
        return np.reshape(traj.xyz[:, self.indexes, :], newshape)

    def __hash__(self):
        hash_value = hash(self.prefix_label)
        hash_value ^= hash_top(self.top)
        hash_value ^= _hash_numpy_array(self.indexes)

        return hash_value

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


class DistanceFeature(object):

    def __init__(self, top, distance_indexes, periodic=True):
        self.top = top
        self.distance_indexes = np.array(distance_indexes)
        self.prefix_label = "DIST:"
        self.periodic = periodic

    def describe(self):
        labels = ["%s %s - %s" % (self.prefix_label,
                                  _describe_atom(self.top, pair[0]),
                                  _describe_atom(self.top, pair[1]))
                  for pair in self.distance_indexes]
        return labels

    @property
    def dimension(self):
        return self.distance_indexes.shape[0]

    @deprecated
    def map(self, traj):
        r"""Deprecated: use transform(traj)

        """
        return self.transform(traj)

    def transform(self, traj):
        return mdtraj.compute_distances(traj, self.distance_indexes, periodic=self.periodic)

    def __hash__(self):
        hash_value = _hash_numpy_array(self.distance_indexes)
        hash_value ^= hash_top(self.top)
        hash_value ^= hash(self.prefix_label)
        return hash_value

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


class InverseDistanceFeature(DistanceFeature):

    def __init__(self, top, distance_indexes, periodic=True):
        DistanceFeature.__init__(
            self, top, distance_indexes, periodic=periodic)
        self.prefix_label = "INVDIST:"

    @deprecated
    def map(self, traj):
        r"""Deprecated: use transform(traj)

        """
        return self.transform(traj)

    def transform(self, traj):
        return 1.0 / mdtraj.compute_distances(traj, self.distance_indexes, periodic=self.periodic)

    # does not need own hash impl, since we take prefix label into account

class ResidueMinDistanceFeature(DistanceFeature):

    def __init__(self, top, contacts, scheme, ignore_nonprotein, threshold):
        self.top = top
        self.contacts = contacts
        self.scheme = scheme
        self.threshold = threshold
        self.prefix_label = "RES_DIST (%s)"%scheme

        # mdtraj.compute_contacts might ignore part of the user input (if it is contradictory) and
        # produce a warning. I think it is more robust to let it run once on a dummy trajectory to
        # see what the actual size of the output is:
        dummy_traj = mdtraj.Trajectory(np.zeros((top.n_atoms, 3)), top)
        dummy_dist, dummy_pairs = mdtraj.compute_contacts(dummy_traj, contacts=contacts,
                                                          scheme=scheme,
                                                          ignore_nonprotein=ignore_nonprotein)
        self._dimension = dummy_dist.shape[1]
        self.distance_indexes = dummy_pairs

    def describe(self):
        labels = ["%s %s - %s" % (self.prefix_label,
                                  self.top.residue(pair[0]),
                                  self.top.residue(pair[1]))
                  for pair in self.distance_indexes]
        return labels

    @deprecated
    def map(self, traj):
        r"""Deprecated: use transform(traj)

        """
        return self.transform(traj)

    def transform(self, traj):
        # We let mdtraj compute the contacts with the input scheme
        D = mdtraj.compute_contacts(traj, contacts=self.contacts, scheme=self.scheme)[0]
        res = np.zeros_like(D)
        # Do we want binary?
        if self.threshold is not None:
            I = np.argwhere(D <= self.threshold)
            res[I[:, 0], I[:, 1]] = 1.0
        else:
            res = D
        return res

class GroupMinDistanceFeature(DistanceFeature):

    def __init__(self, top, group_pairs, distance_list, group_identifiers, threshold):
        self.top = top
        self.group_identifiers = group_identifiers
        self.distance_list = distance_list
        self.prefix_label = "GROUP_MINDIST"
        self.threshold = threshold
        self.distance_indexes = group_pairs

    def describe(self):
        labels = ["%s %s - %s" % (self.prefix_label,
                                  pair[0],
                                  pair[1])
                  for pair in self.distance_indexes]
        return labels

    @deprecated
    def map(self, traj):
        r"""Deprecated: use transform(traj)

        """
        return self.transform(traj)

    def transform(self, traj):
        # All needed distances
        Dall = mdtraj.compute_distances(traj, self.distance_list)
        # Just the minimas
        Dmin = np.zeros((traj.n_frames,self.dimension))
        res = np.zeros_like(Dmin)
        # Compute the min groupwise
        for ii, (gi, gf) in enumerate(self.group_identifiers):
            Dmin[:, ii] =  Dall[:,gi:gf].min(1)
        # Do we want binary?
        if self.threshold is not None:
            I = np.argwhere(Dmin <= self.threshold)
            res[I[:, 0], I[:, 1]] = 1.0
        else:
            res = Dmin

        return res

class ContactFeature(DistanceFeature):

    def __init__(self, top, distance_indexes, threshold=5.0, periodic=True):
        DistanceFeature.__init__(self, top, distance_indexes)
        self.prefix_label = "CONTACT:"
        self.threshold = threshold
        self.periodic = periodic

    @deprecated
    def map(self, traj):
        r"""Deprecated: use transform(traj)

        """
        return self.transform(traj)

    def transform(self, traj):
        dists = mdtraj.compute_distances(
            traj, self.distance_indexes, periodic=self.periodic)
        res = np.zeros(
            (len(traj), self.distance_indexes.shape[0]), dtype=np.float32)
        I = np.argwhere(dists <= self.threshold)
        res[I[:, 0], I[:, 1]] = 1.0
        return res

    def __hash__(self):
        hash_value = DistanceFeature.__hash__(self)
        hash_value ^= hash(self.threshold)
        return hash_value


class AngleFeature(object):

    def __init__(self, top, angle_indexes, deg=False, cossin=False):
        self.top = top
        self.angle_indexes = np.array(angle_indexes)
        self.deg = deg
        self.cossin = cossin

    def describe(self):
        if self.cossin:
            sin_cos = ("ANGLE: COS(%s - %s - %s)",
                       "ANGLE: SIN(%s - %s - %s)")
            labels = [s % (_describe_atom(self.top, triple[0]),
                           _describe_atom(self.top, triple[1]),
                           _describe_atom(self.top, triple[2]))
                      for triple in self.angle_indexes
                      for s in sin_cos]
        else:
            labels = ["ANGLE: %s - %s - %s " %
                      (_describe_atom(self.top, triple[0]),
                       _describe_atom(self.top, triple[1]),
                       _describe_atom(self.top, triple[2]))
                      for triple in self.angle_indexes]
        return labels

    @property
    def dimension(self):
        dim = self.angle_indexes.shape[0]
        if self.cossin:
            dim *= 2
        return dim

    @deprecated
    def map(self, traj):
        r"""Deprecated: use transform(traj)

        """
        return self.transform(traj)

    def transform(self, traj):
        rad = mdtraj.compute_angles(traj, self.angle_indexes)
        if self.cossin:
            rad = np.dstack((np.cos(rad), np.sin(rad)))
            rad = rad.reshape(functools.reduce(lambda x, y: x * y, rad.shape),)
        if self.deg:
            return np.rad2deg(rad)
        else:
            return rad

    def __hash__(self):
        hash_value = _hash_numpy_array(self.angle_indexes)
        hash_value ^= hash_top(self.top)
        hash_value ^= hash(self.deg)

        return hash_value

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


class DihedralFeature(object):

    def __init__(self, top, dih_indexes, deg=False, cossin=False):
        self.top = top
        self.dih_indexes = np.array(dih_indexes)
        self.deg = deg
        self.cossin = cossin
        self._dim = self.dih_indexes.shape[0]
        if self.cossin:
            self._dim *= 2

    def describe(self):
        if self.cossin:
            sin_cos = (
                "DIH: COS(%s -  %s - %s - %s)", "DIH: SIN(%s -  %s - %s - %s)")
            labels = [s %
                      (_describe_atom(self.top, quad[0]),
                       _describe_atom(self.top, quad[1]),
                       _describe_atom(self.top, quad[2]),
                       _describe_atom(self.top, quad[3]))
                      for quad in self.dih_indexes
                      for s in sin_cos]
        else:
            labels = ["DIH: %s - %s - %s - %s " %
                      (_describe_atom(self.top, quad[0]),
                       _describe_atom(self.top, quad[1]),
                       _describe_atom(self.top, quad[2]),
                       _describe_atom(self.top, quad[3]))
                      for quad in self.dih_indexes]
        return labels

    @property
    def dimension(self):
        return self._dim

    @deprecated
    def map(self, traj):
        r"""Deprecated: use transform(traj)

        """
        return self.transform(traj)

    def transform(self, traj):
        rad = mdtraj.compute_dihedrals(traj, self.dih_indexes)
        if self.cossin:
            rad = np.dstack((np.cos(rad), np.sin(rad)))
            rad = rad.reshape(rad.shape[0], rad.shape[1]*rad.shape[2])
        # convert to degrees
        if self.deg:
            rad = np.rad2deg(rad)

        return rad

    def __hash__(self):
        hash_value = _hash_numpy_array(self.dih_indexes)
        hash_value ^= hash_top(self.top)
        hash_value ^= hash(self.deg)
        hash_value ^= hash(self.cossin)

        return hash_value

    def __eq__(self, other):
        return hash(self) == hash(other)


class BackboneTorsionFeature(DihedralFeature):

    def __init__(self, topology, selstr=None, deg=False, cossin=False):
        ft = fake_traj(topology)
        _, indices = _get_indices_phi(ft)

        if not selstr:
            self._phi_inds = indices
        else:
            self._phi_inds = indices[np.in1d(indices[:, 1],
                                             topology.select(selstr), assume_unique=True)]

        _, indices = _get_indices_psi(ft)
        if not selstr:
            self._psi_inds = indices
        else:
            self._psi_inds = indices[np.in1d(indices[:, 1],
                                             topology.select(selstr), assume_unique=True)]

        # alternate phi, psi pairs (phi_1, psi_1, ..., phi_n, psi_n)
        dih_indexes = np.array(list(phi_psi for phi_psi in
                                    zip(self._phi_inds, self._psi_inds))).reshape(-1, 4)

        super(BackboneTorsionFeature, self).__init__(topology, dih_indexes,
                                                     deg=deg, cossin=cossin)

    def describe(self):
        top = self.top
        getlbl = lambda at: "%i %s %i " % (
            at.residue.chain.index, at.residue.name, at.residue.resSeq)

        if self.cossin:
            sin_cos = ("COS(PHI %s)", "SIN(PHI %s)")
            labels_phi = [s % getlbl(top.atom(ires[1])) for ires in self._phi_inds
                          for s in sin_cos]
            sin_cos = ("COS(PSI %s)", "SIN(PSI %s)")
            labels_psi = [s % getlbl(top.atom(ires[1])) for ires in self._psi_inds
                          for s in sin_cos]
        else:
            labels_phi = [
                "PHI %s" % getlbl(top.atom(ires[1])) for ires in self._phi_inds]
            labels_psi = [
                "PSI %s" % getlbl(top.atom(ires[1])) for ires in self._psi_inds]

        return labels_phi + labels_psi


class Chi1TorsionFeature(DihedralFeature):

    def __init__(self, topology, selstr=None, deg=False, cossin=False):
        ft = fake_traj(topology)
        _, indices = _get_indices_chi1(ft)
        if not selstr:
            dih_indexes = indices
        else:
            dih_indexes = indices[np.in1d(indices[:, 1],
                                          topology.select(selstr),
                                          assume_unique=True)]
        super(Chi1TorsionFeature, self).__init__(topology, dih_indexes,
                                                 deg=deg, cossin=cossin)

    def describe(self):
        top = self.top
        getlbl = lambda at: "%i %s %i " \
            % (at.residue.chain.index, at.residue.name, at.residue.resSeq)
        if self.cossin:
            cossin = ("COS(CHI1 %s)", "SIN(CHI1 %s)")
            labels_chi1 = [s % getlbl(top.atom(ires[1]))
                           for ires in self.dih_indexes
                           for s in cossin]
        else:
            labels_chi1 = ["CHI1" + getlbl(top.atom(ires[1]))
                           for ires in self.dih_indexes]

        return labels_chi1


class MinRmsdFeature(object):

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

    @deprecated
    def map(self, traj):
        r"""Deprecated: use transform(traj)

        """
        return self.transform(traj)

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

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


class MDFeaturizer(object):
    r"""Extracts features from MD trajectories."""

    # counting instances, incremented by name property.
    _ids = count(0)

    def __init__(self, topfile):
        """extracts features from MD trajectories.

       Parameters
       ----------

       topfile : str or mdtraj.Topology
           a path to a topology file (pdb etc.) or an mdtraj Topology() object
       """
        self.topologyfile = None
        if type(topfile) is str:
            self.topology = (mdtraj.load(topfile)).topology
            self.topologyfile = topfile
        else:
            self.topology = topfile
        self.active_features = []
        self._dim = 0

    @property
    def name(self):
        try:
            return self._name
        except AttributeError:
            self._name = instance_name(self, next(self._ids))
            return self._name

    @property
    def _logger(self):
        """ The logger for this Estimator """
        try:
            return self._logger_instance
        except AttributeError:
            create_logger(self)
            return self._logger_instance

    def __add_feature(self, f):
        # perform sanity checks
        if f.dimension == 0:
            self._logger.error("given an empty feature (eg. due to an empty/"
                               "ineffective selection). Skipping it."
                               " Feature desc: %s" % f.describe())
            return

        if f not in self.active_features:
            self.active_features.append(f)
        else:
            self._logger.warning("tried to re-add the same feature %s"
                                 % f.__class__.__name__)

    def describe(self):
        """
        Returns a list of strings, one for each feature selected,
        with human-readable descriptions of the features.

        Returns
        -------
        labels : list of str
            An ordered list of strings, one for each feature selected,
            with human-readable descriptions of the features.

        """
        all_labels = []
        for f in self.active_features:
            all_labels += f.describe()
        return all_labels

    def select(self, selstring):
        """
        Returns the indexes of atoms matching the given selection

        Parameters
        ----------
        selstring : str
            Selection string. See mdtraj documentation for details:
            http://mdtraj.org/latest/atom_selection.html

        Returns
        -------
        indexes : ndarray((n), dtype=int)
            array with selected atom indexes

        """
        return self.topology.select(selstring)

    def select_Ca(self):
        """
        Returns the indexes of all Ca-atoms

        Returns
        -------
        indexes : ndarray((n), dtype=int)
            array with selected atom indexes

        """
        return self.topology.select("name CA")

    def select_Backbone(self):
        """
        Returns the indexes of backbone C, CA and N atoms

        Returns
        -------
        indexes : ndarray((n), dtype=int)
            array with selected atom indexes

        """
        return self.topology.select("backbone and (name C or name CA or name N)")

    def select_Heavy(self):
        """
        Returns the indexes of all heavy atoms (Mass >= 2)

        Returns
        -------
        indexes : ndarray((n), dtype=int)
            array with selected atom indexes

        """
        return self.topology.select("mass >= 2")

    @staticmethod
    def pairs(sel, excluded_neighbors=0):
        """
        Creates all pairs between indexes. Will except closest neighbors up to :py:obj:`excluded_neighbors`
        The self-pair (i,i) is always excluded

        Parameters
        ----------
        sel : ndarray((n), dtype=int)
            array with selected atom indexes

        excluded_neighbors: int, default = 0
            number of neighbors that will be excluded when creating the pairs

        Returns
        -------
        sel : ndarray((m,2), dtype=int)
            m x 2 array with all pair indexes between different atoms that are at least :obj:`excluded_neighbors`
            indexes apart, i.e. if i is the index of an atom, the pairs [i,i-2], [i,i-1], [i,i], [i,i+1], [i,i+2], will
            not be in :py:obj:`sel` (n=excluded_neighbors) if :py:obj:`excluded_neighbors` = 2.
            Moreover, the list is non-redundant,i.e. if [i,j] is in sel, then [j,i] is not.

        """

        assert isinstance(excluded_neighbors,int)

        p = []
        for i in range(len(sel)):
            for j in range(i + 1, len(sel)):
                # get ordered pair
                I = sel[i]
                J = sel[j]
                if (I > J):
                    I = sel[j]
                    J = sel[i]
                # exclude 1 and 2 neighbors
                if (J > I + excluded_neighbors):
                    p.append([I, J])
        return np.array(p)

    def _check_indices(self, pair_inds, pair_n=2):
        """ensure pairs are valid (shapes, all atom indices available?, etc.) 
        """

        pair_inds = np.array(pair_inds).astype(dtype=np.int, casting='safe')

        if pair_inds.ndim != 2:
            raise ValueError("pair indices has to be a matrix.")

        if pair_inds.shape[1] != pair_n:
            raise ValueError("pair indices shape has to be (x, %i)." % pair_n)

        if pair_inds.max() > self.topology.n_atoms:
            raise ValueError("index out of bounds: %i."
                             " Maximum atom index available: %i"
                             % (pair_inds.max(), self.topology.n_atoms))

        return pair_inds

    def add_all(self):
        """
        Adds all atom coordinates to the feature list.
        The coordinates are flattened as follows: [x1, y1, z1, x2, y2, z2, ...]

        """
        # TODO: add possibility to align to a reference structure
        self.add_selection(list(range(self.topology.n_atoms)))

    def add_selection(self, indexes):
        """
        Adds the coordinates of the selected atom indexes to the feature list.
        The coordinates of the selection [1, 2, ...] are flattened as follows: [x1, y1, z1, x2, y2, z2, ...]

        Parameters
        ----------
        indexes : ndarray((n), dtype=int)
            array with selected atom indexes

        """
        # TODO: add possibility to align to a reference structure
        f = SelectionFeature(self.topology, indexes)
        self.__add_feature(f)

    def add_distances(self, indices, periodic=True, indices2=None):
        r"""
        Adds the distances between atoms to the feature list.

        Parameters
        ----------
        indices : can be of two types:

                ndarray((n, 2), dtype=int):
                    n x 2 array with the pairs of atoms between which the distances shall be computed

                iterable of integers (either list or ndarray(n, dtype=int)):
                    indices (not pairs of indices) of the atoms between which the distances shall be computed.

        indices2: iterable of integers (either list or ndarray(n, dtype=int)), optional:
                    Only has effect if :py:obj:`indices` is an iterable of integers. Instead of the above behaviour,
                    only the distances between the atoms in :py:obj:`indices` and :py:obj:`indices2` will be computed.


        .. note::
            When using the iterable of integers input, :py:obj:`indices` and :py:obj:`indices2`
            will be sorted numerically and made unique before converting them to a pairlist.
            Please look carefully at the output of :py:func:`describe()` to see what features exactly have been added.
        """

        atom_pairs = _parse_pairwise_input(
            indices, indices2, self._logger, fname='add_distances()')

        atom_pairs = self._check_indices(atom_pairs)
        f = DistanceFeature(self.topology, atom_pairs, periodic=periodic)
        self.__add_feature(f)

    def add_distances_ca(self, periodic=True):
        """
        Adds the distances between all Ca's to the feature list.

        """
        distance_indexes = self.pairs(self.select_Ca())
        self.add_distances(distance_indexes, periodic=periodic)

    def add_inverse_distances(self, indices, periodic=True, indices2=None):
        """
        Adds the inverse distances between atoms to the feature list.

        Parameters
        ----------
        indices : can be of two types:

                ndarray((n, 2), dtype=int):
                    n x 2 array with the pairs of atoms between which the inverse distances shall be computed

                iterable of integers (either list or ndarray(n, dtype=int)):
                    indices (not pairs of indices) of the atoms between which the inverse distances shall be computed.

        indices2: iterable of integers (either list or ndarray(n, dtype=int)), optional:
                    Only has effect if :py:obj:`indices` is an iterable of integers. Instead of the above behaviour,
                    only the inverse distances between the atoms in :py:obj:`indices` and :py:obj:`indices2` will be computed.


        .. note::
            When using the *iterable of integers* input, :py:obj:`indices` and :py:obj:`indices2`
            will be sorted numerically and made unique before converting them to a pairlist.
            Please look carefully at the output of :py:func:`describe()` to see what features exactly have been added.

        """

        atom_pairs = _parse_pairwise_input(
            indices, indices2, self._logger, fname='add_inverse_distances()')

        atom_pairs = self._check_indices(atom_pairs)
        f = InverseDistanceFeature(self.topology, atom_pairs, periodic=True)
        self.__add_feature(f)

    def add_contacts(self, indices, indices2=None, threshold=5.0, periodic=True):
        r"""
        Adds the contacts to the feature list.

        Parameters
        ----------
        indices : can be of two types:

                ndarray((n, 2), dtype=int):
                    n x 2 array with the pairs of atoms between which the contacts shall be computed

                iterable of integers (either list or ndarray(n, dtype=int)):
                    indices (not pairs of indices) of the atoms between which the contacts shall be computed.

        indices2: iterable of integers (either list or ndarray(n, dtype=int)), optional:
                    Only has effect if :py:obj:`indices` is an iterable of integers. Instead of the above behaviour,
                    only the contacts between the atoms in :py:obj:`indices` and :py:obj:`indices2` will be computed.

        threshold : float, optional, default = 5.0
            distances below this threshold will result in a feature 1.0, distances above will result in 0.0.
            The default is set with Angstrom distances in mind.
            Make sure that you know whether your coordinates are in Angstroms or nanometers when setting this threshold.


        .. note::
            When using the *iterable of integers* input, :py:obj:`indices` and :py:obj:`indices2`
            will be sorted numerically and made unique before converting them to a pairlist.
            Please look carefully at the output of :py:func:`describe()` to see what features exactly have been added.
        """

        atom_pairs = _parse_pairwise_input(
            indices, indices2, self._logger, fname='add_contacts()')

        atom_pairs = self._check_indices(atom_pairs)
        f = ContactFeature(self.topology, atom_pairs, threshold, periodic)
        self.__add_feature(f)

    def add_residue_mindist(self,
                            residue_pairs='all',
                            scheme='closest-heavy',
                            ignore_nonprotein=True,
                            threshold=None):
        r"""
        Adds the minimum distance between residues to the feature list. See below how
        the minimum distance can be defined.

        Parameters
        ----------
        residue_pairs : can be of two types:

            'all'
                Computes distances between all pairs of residues excluding first and second neighbors

            ndarray((n, 2), dtype=int):
                n x 2 array with the pairs residues for which distances will be computed

        scheme : 'ca', 'closest', 'closest-heavy', default is closest-heavy
                Within a residue, determines the sub-group atoms that will be considered when computing distances

        ignore_nonprotein : boolean, default True
                Ignore residues that are not of protein type (e.g. water molecules, post-traslational modifications etc)

        threshold : float, optional, default is None
            distances below this threshold (in nm) will result in a feature 1.0, distances above will result in 0.0. If
            left to None, the numerical value will be returned

        .. note::
            Using :py:obj:`scheme` = 'closest' or 'closest-heavy' with :py:obj:`residue pairs` = 'all'
            will compute nearly all interatomic distances, for every frame, before extracting the closest pairs.
            This can be very time consuming. Those schemes are intended to be used with a subset of residues chosen
            via :py:obj:`residue_pairs`.
        """

        if scheme != 'ca' and residue_pairs == 'all':
            self._logger.warning("Using all residue pairs with schemes like closest or closest-heavy is "
                                 "very time consuming. Consider reducing the residue pairs")

        f = ResidueMinDistanceFeature(self.topology, residue_pairs, scheme, ignore_nonprotein, threshold)
        self.__add_feature(f)

    def add_group_mindist(self,
                            group_definitions,
                            group_pairs='all',
                            threshold=None,
                            ):
        r"""
        Adds the minimum distance between groups of atoms to the feature list. If the groups of
        atoms are identical to residues, use :py:obj:`add_residue_mindist <pyemma.coordinates.data.featurizer.MDFeaturizer.add_residue_mindist>`.

        Parameters
        ----------

        group_definition : list of 1D-arrays/iterables containing the group definitions via atom indices.
            If there is only one group_definition, it is assumed the minimum distance within this group (excluding the
            self-distance) is wanted. In this case, :py:obj:`group_pairs` is ignored.

        group_pairs :  Can be of two types:
            'all'
                Computes minimum distances between all pairs of groups contained in the group definitions

            ndarray((n, 2), dtype=int):
                n x 2 array with the pairs of groups for which the minimum distances will be computed.

        threshold : float, optional, default is None
            distances below this threshold (in nm) will result in a feature 1.0, distances above will result in 0.0. If
            left to None, the numerical value will be returned

        """

        # Some thorough input checking and reformatting
        __, group_pairs, distance_list, group_identifiers = _parse_groupwise_input(group_definitions, group_pairs, self._logger, 'add_group_mindist')
        distance_list = self._check_indices(distance_list)

        f = GroupMinDistanceFeature(self.topology, group_pairs, distance_list, group_identifiers, threshold)
        self.__add_feature(f)

    def add_angles(self, indexes, deg=False, cossin=False):
        """
        Adds the list of angles to the feature list

        Parameters
        ----------
        indexes : np.ndarray, shape=(num_pairs, 3), dtype=int
            an array with triplets of atom indices
        deg : bool, optional, default = False
            If False (default), angles will be computed in radians.
            If True, angles will be computed in degrees.
        cossin : bool, optional, default = False
            If True, each angle will be returned as a pair of (sin(x), cos(x)).
            This is useful, if you calculate the mean (e.g TICA/PCA, clustering)
            in that space.

        """
        indexes = self._check_indices(indexes, pair_n=3)
        f = AngleFeature(self.topology, indexes, deg=deg, cossin=cossin)
        self.__add_feature(f)

    def add_dihedrals(self, indexes, deg=False, cossin=False):
        """
        Adds the list of dihedrals to the feature list

        Parameters
        ----------
        indexes : np.ndarray, shape=(num_pairs, 4), dtype=int
            an array with quadruplets of atom indices
        deg : bool, optional, default = False
            If False (default), angles will be computed in radians.
            If True, angles will be computed in degrees.
        cossin : bool, optional, default = False
            If True, each angle will be returned as a pair of (sin(x), cos(x)).
            This is useful, if you calculate the mean (e.g TICA/PCA, clustering)
            in that space.

        """
        indexes = self._check_indices(indexes, pair_n=4)
        f = DihedralFeature(self.topology, indexes, deg=deg, cossin=cossin)
        self.__add_feature(f)

    def add_backbone_torsions(self, selstr=None, deg=False, cossin=False):
        """
        Adds all backbone phi/psi angles or the ones specified in :obj:`selstr` to the feature list.

        Parameters
        ----------

        selstr : str, optional, default = ""
            selection string specifying the atom selection used to specify a specific set of backbone angles
            If "" (default), all phi/psi angles found in the topology will be computed
        deg : bool, optional, default = False
            If False (default), angles will be computed in radians.
            If True, angles will be computed in degrees.
        cossin : bool, optional, default = False
            If True, each angle will be returned as a pair of (sin(x), cos(x)).
            This is useful, if you calculate the mean (e.g TICA/PCA, clustering)
            in that space.
        """
        f = BackboneTorsionFeature(
            self.topology, selstr=selstr, deg=deg, cossin=cossin)
        self.__add_feature(f)

    def add_chi1_torsions(self, selstr="", deg=False, cossin=False):
        """
        Adds all chi1 angles or the ones specified in :obj:`selstr` to the feature list.

        Parameters
        ----------

        selstr : str, optional, default = ""
            selection string specifying the atom selection used to specify a specific set of backbone angles
            If "" (default), all chi1 angles found in the topology will be computed
        deg : bool, optional, default = False
            If False (default), angles will be computed in radians.
            If True, angles will be computed in degrees.
        cossin : bool, optional, default = False
            If True, each angle will be returned as a pair of (sin(x), cos(x)).
            This is useful, if you calculate the mean (e.g TICA/PCA, clustering)
            in that space.
        """
        f = Chi1TorsionFeature(
            self.topology, selstr=selstr, deg=deg, cossin=cossin)
        self.__add_feature(f)

    def add_custom_feature(self, feature):
        """
        Adds a custom feature to the feature list.

        Parameters
        ----------
        feature : object
            an object with interface like CustomFeature (map, describe methods)

        """
        if feature.dimension <= 0:
            raise ValueError("Dimension has to be positive. "
                             "Please override dimension attribute in feature!")

        if not hasattr(feature, 'map'):
            raise ValueError("no map method in given feature")
        else:
            if not callable(getattr(feature, 'map')):
                raise ValueError("map exists but is not a method")

        self.__add_feature(feature)

    def add_minrmsd_to_ref(self, ref, ref_frame=0, atom_indices=None, precentered=False):
        r"""
        Adds the minimum root-mean-square-deviation (minrmsd) with respect to a reference structure to the feature list.

        Parameters
        ----------
        ref:
            Reference structure for computing the minrmsd. Can be of two types:

                1. :py:obj:`mdtraj.Trajectory` object
                2. filename for mdtraj to load. In this case, only the :py:obj:`ref_frame` of that file will be used.

        ref_frame: integer, default=0
            Reference frame of the filename specified in :py:obj:`ref`.
            This parameter has no effect if :py:obj:`ref` is not a filename.

        atom_indices: array_like, default=None
            Atoms that will be used for:

                1. aligning the target and reference geometries.
                2. computing rmsd after the alignment.
            If left to None, all atoms of :py:obj:`ref` will be used.

        precentered: bool, default=False
            Use this boolean at your own risk to let mdtraj know that the target conformations are already
            centered at the origin, i.e., their (uniformly weighted) center of mass lies at the origin.
            This will speed up the computation of the rmsd.
        """

        f = MinRmsdFeature(ref, ref_frame=ref_frame, atom_indices=atom_indices, topology=self.topology,
                           precentered=precentered)
        self.__add_feature(f)

    def add_custom_func(self, func, dim, *args, **kwargs):
        """ adds a user defined function to extract features

        Parameters
        ----------
        func : function
            a user-defined function, which accepts mdtraj.Trajectory object as
            first parameter and as many optional and named arguments as desired.
            Has to return a numpy.ndarray
        dim : int
            output dimension of :py:obj:`function`
        args : any number of positional arguments
            these have to be in the same order as :py:obj:`func` is expecting them
        kwargs : dictionary
            named arguments passed to func

        """
        f = CustomFeature(func, dim=dim, *args, **kwargs)

        self.add_custom_feature(f)

    def dimension(self):
        """ current dimension due to selected features

        Returns
        -------
        dim : int
            total dimension due to all selection features

        """
        dim = sum(f.dimension for f in self.active_features)
        return dim

    @deprecated
    def map(self, traj):
        r"""Deprecated: use transform(traj)

        """
        return self.transform(traj)

    def transform(self, traj):
        """
        Maps an mdtraj Trajectory object to the selected output features

        Parameters
        ----------
        traj : mdtraj Trajectory
            Trajectory object used as an input

        Returns
        -------
        out : ndarray((T, n), dtype=float32)
            Output features: For each of T time steps in the given trajectory, 
            a vector with all n output features selected.

        """
        # if there are no features selected, return given trajectory
        if len(self.active_features) == 0:
            warnings.warn("You have no features selected."
                          " Returning plain coordinates.")
            s = traj.xyz.shape
            new_shape = (s[0], s[1] * s[2])
            return traj.xyz.reshape(new_shape)

        # handle empty chunks (which might occur due to time lagged access
        if traj.xyz.shape[0] == 0:
            return np.empty((0, self.dimension()))

        # TODO: define preprocessing step (RMSD etc.)

        # otherwise build feature vector.
        feature_vec = []

        # TODO: consider parallel evaluation computation here, this effort is
        # only worth it, if computation time dominates memory transfers
        for f in self.active_features:
            # perform sanity checks for custom feature input
            if isinstance(f, CustomFeature):
                # NOTE: casting=safe raises in numpy>=1.9
                vec = f.transform(traj).astype(np.float32, casting='safe')
                if vec.shape[0] == 0:
                    vec = np.empty((0, f.dimension))

                if not isinstance(vec, np.ndarray):
                    raise ValueError('Your custom feature %s did not return'
                                     ' a numpy.ndarray!' % str(f.describe()))
                if not vec.ndim == 2:
                    raise ValueError('Your custom feature %s did not return'
                                     ' a 2d array. Shape was %s'
                                     % (str(f.describe()),
                                        str(vec.shape)))
                if not vec.shape[0] == traj.xyz.shape[0]:
                    raise ValueError('Your custom feature %s did not return'
                                     ' as many frames as it received!'
                                     'Input was %i, output was %i'
                                     % (str(f.describe()),
                                        traj.xyz.shape[0],
                                        vec.shape[0]))
            else:
                vec = f.transform(traj).astype(np.float32)
            feature_vec.append(vec)

        if len(feature_vec) > 1:
            res = np.hstack(feature_vec)
        else:
            res = feature_vec[0]
        return res