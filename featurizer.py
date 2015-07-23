# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# -*- coding: uft-8 -*-
import mdtraj
from mdtraj.geometry.dihedral import _get_indices_phi, \
    _get_indices_psi, compute_dihedrals, _atom_sequence, CHI1_ATOMS

import numpy as np
import warnings
from itertools import combinations as _combinations, count
from itertools import product as _product
from pyemma.util.types import is_iterable_of_int as _is_iterable_of_int
import functools

from pyemma.util.log import getLogger
from pyemma.util.annotators import deprecated


__author__ = 'Frank Noe, Martin Scherer'
__all__ = ['MDFeaturizer',
           ]


def _get_indices_chi1(traj):
    rids, indices = zip(*(_atom_sequence(traj, atoms) for atoms in CHI1_ATOMS))
    id_sort = np.argsort(np.concatenate(rids))
    if not any(x.size for x in indices):
        return np.empty(shape=(0, 4), dtype=np.int)

    indices = np.vstack(x for x in indices if x.size)[id_sort]
    return id_sort, indices

# this is needed for get_indices functions, since they expect a Trajectory,
# not a Topology
class fake_traj():
    def __init__(self, top):
        self.top = top


def _describe_atom(topology, index):
    """
    Returns a string describing the given atom

    :param topology:
    :param index:
    :return:
    """
    assert isinstance(index, int)
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


def _hash_numpy_array(x):
    if x is None:
        return hash(None)
    x.flags.writeable = False
    hash_value = hash(x.shape)
    hash_value ^= hash(x.strides)
    hash_value ^= hash(x.data)
    x.flags.writeable = True

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

    Examples
    --------
    We define a feature that transforms all coordinates by :math:`1 / x^2`:

    >>> from pyemma.coordinates import source

    Define a function which transforms the coordinates of the trajectory object:
    >>> my_feature = CustomFeature(lambda x: 1.0 / x.xyz**2)
    >>> reader = pyemma.coordinates.load('traj.xtc', top='my_topology.pdb') # doctest: +SKIP
    # pass the feature to the featurizer and transform the data
    >>> reader.featurizer.add_custom_feature(my_feature) # doctest: +SKIP
    >>> data = reader.get_output() # doctest: +SKIP

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

    def map(self, traj):
        feature = self._func(traj, *self._args, **self._kwargs)
        if not isinstance(feature, np.ndarray):
            raise ValueError("your function should return a NumPy array!")
        return feature

    def __hash__(self):
        hash_value = hash(self._func)
        # if key contains numpy arrays, we hash their data arrays
        key = tuple(map(_catch_unhashable, self._args) +
                    map(_catch_unhashable, sorted(self._kwargs.items())))
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

    def map(self, traj):
        newshape = (traj.xyz.shape[0], 3 * self.indexes.shape[0])
        return np.reshape(traj.xyz[:, self.indexes, :], newshape)

    def __hash__(self):
        hash_value = hash(self.top)
        hash_value ^= _hash_numpy_array(self.indexes)
        hash_value ^= hash(self.prefix_label)

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

    def map(self, traj):
        return mdtraj.compute_distances(traj, self.distance_indexes, periodic=self.periodic)

    def __hash__(self):
        hash_value = _hash_numpy_array(self.distance_indexes)
        hash_value ^= hash(self.top)
        hash_value ^= hash(self.prefix_label)
        return hash_value

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


class InverseDistanceFeature(DistanceFeature):

    def __init__(self, top, distance_indexes, periodic=True):
        DistanceFeature.__init__(
            self, top, distance_indexes, periodic=periodic)
        self.prefix_label = "INVDIST:"

    def map(self, traj):
        return 1.0 / mdtraj.compute_distances(traj, self.distance_indexes, periodic=self.periodic)

    # does not need own hash impl, since we take prefix label into account


class ContactFeature(DistanceFeature):

    def __init__(self, top, distance_indexes, threshold=5.0, periodic=True):
        DistanceFeature.__init__(self, top, distance_indexes)
        self.prefix_label = "CONTACT:"
        self.threshold = threshold
        self.periodic = periodic

    def map(self, traj):
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
        self.top

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

    def map(self, traj):
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
        hash_value ^= hash(self.top)
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

    def map(self, traj):
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
        hash_value ^= hash(self.top)
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
        return self.ref.n_atoms

    def map(self, traj):
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

    _ids = count(0)

    def __init__(self, topfile):
        """extracts features from MD trajectories.

       Parameters
       ----------

       topfile : str
           a path to a topology file (pdb etc.)
       """
        self.topologyfile = topfile
        self.topology = (mdtraj.load(topfile)).topology
        self.active_features = []
        self._dim = 0
        self._create_logger()

    def _create_logger(self):
        count = self._ids.next()
        i = self.__module__.rfind(".")
        j = self.__module__.find(".") + 1
        package = self.__module__[j:i]
        name = "%s.%s[%i]" % (package, self.__class__.__name__, count)
        self._name = name
        self._logger = getLogger(name)

    def __add_feature(self, f):
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
    def pairs(sel):
        """
        Creates all pairs between indexes, except for 1 and 2-neighbors

        Parameters
        ----------
        sel : ndarray((n), dtype=int)
            array with selected atom indexes

        Return:
        -------
        sel : ndarray((m,2), dtype=int)
            m x 2 array with all pair indexes between different atoms that are at least 3 indexes apart,
            i.e. if i is the index of an atom, the pairs [i,i-2], [i,i-1], [i,i], [i,i+1], [i,i+2], will
            not be in sel. Moreover, the list is non-redundant, i.e. if [i,j] is in sel, then [j,i] is not.

        """
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
                if (J > I + 2):
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
        self.add_selection(range(self.topology.n_atoms))

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

    @deprecated
    def distances(self, atom_pairs):
        return self.add_distances(atom_pairs)

    def add_distances(self, indices, periodic=True, indices2=None):
        r"""
        Adds the distances between atoms to the feature list.

        Parameters
        ----------
        indices : can be of two types:

                ndarray((n, 2), dtype=int):
                    n x 2 array with the pairs of atoms between which the distances shall be computed

                iterable of integers (either list or ndarray(n, dtype=int)):
                    indices (**not pairs of indices**) of the atoms between which the distances shall be computed.
                    Note that this will produce a pairlist different from the pairlist produced by :py:func:`pairs` in that this **does not** exclude
                    1-2 neighbors.

        indices2: iterable of integers (either list or ndarray(n, dtype=int)), optional:
                    Only has effect if :py:obj:`indices` is an iterable of integers. Instead of the above behaviour,
                    only the distances between the atoms in :py:obj:`indices` and :py:obj:`indices2` will be computed.


        .. note::
            When using the *iterable of integers* input, :py:obj:`indices` and :py:obj:`indices2`
            will be sorted numerically and made unique before converting them to a pairlist.
            Please look carefully at the output of :py:func:`describe()` to see what features exactly have been added.
        """

        atom_pairs = _parse_pairwise_input(
            indices, indices2, self._logger, fname='add_distances()')

        atom_pairs = self._check_indices(atom_pairs)
        f = DistanceFeature(self.topology, atom_pairs, periodic=periodic)
        self.__add_feature(f)

    @deprecated
    def distancesCa(self):
        return self.add_distances_ca()

    def add_distances_ca(self, periodic=True):
        """
        Adds the distances between all Ca's (except for 1- and 2-neighbors) to the feature list.

        """
        distance_indexes = self.pairs(self.select_Ca())
        self.add_distances(distance_indexes, periodic=periodic)

    @deprecated
    def inverse_distances(self, atom_pairs):
        return self.add_inverse_distances(atom_pairs)

    def add_inverse_distances(self, indices, periodic=True, indices2=None):
        """
        Adds the inverse distances between atoms to the feature list.

        Parameters
        ----------
        indices : can be of two types:

                ndarray((n, 2), dtype=int):
                    n x 2 array with the pairs of atoms between which the inverse distances shall be computed

                iterable of integers (either list or ndarray(n, dtype=int)):
                    indices (**not pairs of indices**) of the atoms between which the inverse distances shall be computed.
                    Note that this will produce a pairlist different from the pairlist produced by :py:func:`pairs` in that this **does not** exclude
                    1-2 neighbors.

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

    @deprecated
    def contacts(self, *args):
        return self.add_contacts(args)

    def add_contacts(self, indices, indices2=None, threshold=5.0, periodic=True):
        r"""
        Adds the contacts to the feature list.

        Parameters
        ----------
        indices : can be of two types:

                ndarray((n, 2), dtype=int):
                    n x 2 array with the pairs of atoms between which the contacts shall be computed

                iterable of integers (either list or ndarray(n, dtype=int)):
                    indices (**not pairs of indices**) of the atoms between which the contacts shall be computed.
                    Note that this will produce a pairlist different from the pairlist produced by :py:func:`pairs` in that this **does not** exclude
                    1-2 neighbors.

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

    @deprecated
    def angles(self, *args):
        return self.add_angles(args)

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

    @deprecated
    def backbone_torsions(self, *args):
        return self.add_backbone_torsions(args)

    def add_backbone_torsions(self, selstr=None, deg=False, cossin=False):
        """
        Adds all backbone phi/psi angles or the ones specified in selstr to the feature list.
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
        f = BackboneTorsionFeature(
            self.topology, selstr=selstr, deg=deg, cossin=cossin)
        self.__add_feature(f)

    def add_chi1_torsions(self, selstr="", deg=False, cossin=False):
        """
        Adds all chi1 angles or the ones specified in selstr to the feature list.
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
            Use this boolean at your own risk to let mdtraj know that the target conformations are **already**
            centered at the origin, i.e., their (uniformly weighted) center of mass lies at the origin.
            This will speed up the computation of the rmsd.
        """

        f = MinRmsdFeature(ref, ref_frame=ref_frame, atom_indices=atom_indices, topology=self.topology,
                           precentered=precentered)
        self.__add_feature(f)

    def add_custom_func(self, func, dim, desc='', *args, **kwargs):
        """ adds a user defined function to extract features

        Parameters
        ----------
        func : function
            a user-defined function, which accepts mdtraj.Trajectory object as
            first parameter and as many optional and named arguments as desired.
            Has to return a numpy.ndarray
        dim : int
            output dimension of function
        desc : str
            description of your feature function
        args : list
            positional arguments passed to func
        kwargs : dictionary
            named arguments passed to func

        """
        f = CustomFeature(func, args, kwargs, dim=dim)

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

    def map(self, traj):
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
                vec = f.map(traj).astype(np.float32, casting='safe')
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
                vec = f.map(traj).astype(np.float32)
            feature_vec.append(vec)

        if len(feature_vec) > 1:
            res = np.hstack(feature_vec)
        else:
            res = feature_vec[0]
        return res
