
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

__author__ = 'Frank Noe, Martin Scherer'

import mdtraj
from mdtraj.geometry.dihedral import _get_indices_phi, \
    _get_indices_psi, compute_dihedrals

import numpy as np
import warnings

from pyemma.util.log import getLogger
from pyemma.util.annotators import deprecated

__all__ = ['MDFeaturizer',
           'CustomFeature',
           ]


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
    x.flags.writeable = False
    hash_value = hash(x.shape)
    hash_value ^= hash(x.strides)
    hash_value ^= hash(x.data)
    return hash_value


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
    # define a function which transforms the coordinates of the trajectory object
    >>> my_feature = CustomFeature(lambda x: 1.0 / x.xyz**2)
    >>> reader = load('traj.xtc', top='my_topology.pdb')
    # pass the feature to the featurizer and transform the data
    >>> reader.featurizer.add_custom_feature(my_feature)
    >>> data = reader.get_output()

    """

    def __init__(self, func=None, *args, **kwargs):
        self._func = func
        self._args = args
        self._kwargs = kwargs
        if 'dim' in kwargs:
            self.dimension = kwargs['dim']
        else:
            self.dimension = 0

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

    def __init__(self, top, angle_indexes, deg=False):
        self.top = top
        self.angle_indexes = np.array(angle_indexes)
        self.deg = deg

    def describe(self):
        labels = ["ANGLE: %s - %s - %s " %
                  (_describe_atom(self.top, triple[0]),
                   _describe_atom(self.top, triple[1]),
                   _describe_atom(self.top, triple[2]))
                  for triple in self.angle_indexes
                  ]

        return labels

    @property
    def dimension(self):
        return self.angle_indexes.shape[0]

    def map(self, traj):
        rad = mdtraj.compute_angles(traj, self.angle_indexes)
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

    def __init__(self, top, dih_indexes, deg=False):
        self.top = top
        self.dih_indexes = np.array(dih_indexes)
        self.deg = deg

    def describe(self):
        labels = ["DIH: %s - %s - %s - %s " %
                  (_describe_atom(self.top, quad[0]),
                   _describe_atom(self.top, quad[1]),
                   _describe_atom(self.top, quad[2]),
                   _describe_atom(self.top, quad[3]))
                  for quad in self.dih_indexes
                  ]
        return labels

    @property
    def dimension(self):
        return self.dih_indexes.shape[0]

    def map(self, traj):
        rad = mdtraj.compute_dihedrals(traj, self.dih_indexes)
        if self.deg:
            return np.rad2deg(rad)
        else:
            return rad

    def __hash__(self):
        hash_value = _hash_numpy_array(self.dih_indexes)
        hash_value ^= hash(self.top)
        hash_value ^= hash(self.deg)

        return hash_value

    def __eq__(self, other):
        return hash(self) == hash(other)


class BackboneTorsionFeature(object):
    # TODO: maybe consider this as a special case of DihedralFeature?

    def __init__(self, topology, deg=False):
        self.topology = topology
        self.deg = deg

        # this is needed for get_indices functions, since they expect a Trajectory,
        # not a Topology
        class fake_traj():

            def __init__(self, top):
                self.top = top

        ft = fake_traj(topology)
        _, indices = _get_indices_phi(ft)
        self._phi_inds = indices

        _, indices = _get_indices_psi(ft)
        self._psi_inds = indices

        self._dim = len(self._phi_inds) + len(self._psi_inds)

    def describe(self):
        top = self.topology
        labels_phi = ["PHI %s" % _describe_atom(top, i)
                      for ires in self._phi_inds
                      for i in ires]

        labels_psi = ["PSI %s" % _describe_atom(top, i)
                      for ires in self._psi_inds
                      for i in ires]

        return labels_phi + labels_psi

    @property
    def dimension(self):
        return self._dim

    def map(self, traj):
        # TODO: can we merge phi_inds and psi_inds to only call
        # compute_dihedrals once?
        y1 = compute_dihedrals(traj, self._phi_inds).astype(np.float32)
        y2 = compute_dihedrals(traj, self._psi_inds).astype(np.float32)
        rad = np.hstack((y1, y2))
        if self.deg:
            return np.rad2deg(rad)
        else:
            return rad

    def __hash__(self):
        hash_value = _hash_numpy_array(self._phi_inds)
        hash_value ^= _hash_numpy_array(self._psi_inds)
        hash_value ^= hash(self.topology)

        return hash_value

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


class MDFeaturizer(object):

    """extracts features from MD trajectories.

    Parameters
    ----------

    topfile : str
        a path to a topology file (pdb etc.)
    """

    def __init__(self, topfile):
        self.topologyfile = topfile
        self.topology = (mdtraj.load(topfile)).topology
        self.active_features = []
        self._dim = 0
        self._logger = getLogger("%s[%s]" %
                                 (self.__class__.__name__, hex(id(self))))

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

    def add_distances(self, atom_pairs, periodic=True):
        """
        Adds the distances between the given pairs of atoms to the feature list.

        Parameters
        ----------
        atom_pairs : ndarray((n, 2), dtype=int)
            n x 2 array with pairs of atoms between which the distances shall be computed

        """
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

    def add_inverse_distances(self, atom_pairs, periodic=True):
        """
        Adds the inverse distances between the given pairs of atoms to the feature list.

        Parameters
        ----------
        atom_pairs : ndarray((n,2), dtype=int)
            n x 2 array with pairs of atoms between which the inverse distances shall be computed

        """
        atom_pairs = self._check_indices(atom_pairs)
        f = InverseDistanceFeature(self.topology, atom_pairs, periodic=True)
        self.__add_feature(f)

    @deprecated
    def contacts(self, atom_pairs):
        return self.add_contacts(atom_pairs)

    def add_contacts(self, atom_pairs, threshold=5.0, periodic=True):
        """
        Adds the set of contacts to the feature list

        Parameters
        ----------
        atom_pairs : ndarray((n, 2), dtype=int)
            n x 2 array of pairs of atoms to compute contacts between
        threshold : float, optional, default = 5.0
            distances below this threshold will result in a feature 1.0, distances above will result in 0.0.
            The default is set with Angstrom distances in mind.
            Make sure that you know whether your coordinates are in Angstroms or nanometers when setting this threshold.
        """
        atom_pairs = self._check_indices(atom_pairs)
        f = ContactFeature(self.topology, atom_pairs, threshold, periodic)
        self.__add_feature(f)

    @deprecated
    def angles(self, indexes):
        return self.add_angles(indexes)

    def add_angles(self, indexes, deg=False):
        """
        Adds the list of angles to the feature list

        Parameters
        ----------
        indexes : np.ndarray, shape=(num_pairs, 3), dtype=int
            an array with triplets of atom indices
        deg : bool, optional, default = False
            If False (default), angles will be computed in radians.
            If True, angles will be computed in degrees.

        """
        indexes = self._check_indices(indexes, pair_n=3)
        f = AngleFeature(self.topology, indexes, deg=deg)
        self.__add_feature(f)

    def add_dihedrals(self, indexes, deg=False):
        """
        Adds the list of dihedrals to the feature list

        Parameters
        ----------
        indexes : np.ndarray, shape=(num_pairs, 4), dtype=int
            an array with quadruplets of atom indices
        deg : bool, optional, default = False
            If False (default), angles will be computed in radians.
            If True, angles will be computed in degrees.

        """
        indexes = self._check_indices(indexes, pair_n=4)
        f = DihedralFeature(self.topology, indexes, deg=deg)
        self.__add_feature(f)

    @deprecated
    def backbone_torsions(self):
        return self.add_backbone_torsions()

    def add_backbone_torsions(self, deg=False):
        """
        Adds all backbone phi/psi angles to the feature list.

        Parameters
        ----------
        deg : bool, optional, default = False
            If False (default), angles will be computed in radians.
            If True, angles will be computed in degrees.

        """
        f = BackboneTorsionFeature(self.topology, deg=deg)
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
        f = CustomFeature(func, args, kwargs)
        f.dimension = dim

        self.add_custom_feature(f)

    def dimension(self):
        """ current dimension due to selected features

        Returns
        -------
        dim : int
            total dimension due to all selection features

        """
        dim = 0
        for f in self.active_features:
            dim += f.dimension
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

                if not isinstance(vec, np.ndarray):
                    raise ValueError('Your custom feature %s did not return'
                                     ' a numpy.ndarray!' % str(f.describe()))
                if not np.ndim == 2:
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

        stack = np.hstack(feature_vec)
        return stack
