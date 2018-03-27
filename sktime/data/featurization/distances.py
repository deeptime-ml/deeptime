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
import mdtraj
import numpy as np

from pyemma.coordinates.data.featurization._base import Feature
from pyemma.coordinates.data.featurization.util import _describe_atom


class DistanceFeature(Feature):
    __serialize_version = 0
    __serialize_fields = ('distance_indexes', 'periodic')

    prefix_label = "DIST:"

    def __init__(self, top, distance_indexes, periodic=True):
        self.top = top
        self.distance_indexes = np.array(distance_indexes)
        if len(self.distance_indexes) == 0:
            raise ValueError("empty indices")
        self.periodic = periodic
        self._dim = len(distance_indexes)

    def describe(self):
        labels = ["%s %s - %s" % (self.prefix_label,
                                  _describe_atom(self.top, pair[0]),
                                  _describe_atom(self.top, pair[1]))
                  for pair in self.distance_indexes]
        return labels

    def transform(self, traj):
        return mdtraj.compute_distances(traj, self.distance_indexes, periodic=self.periodic)

    def __eq__(self, other):
        eq = super(DistanceFeature, self).__eq__(other)
        if not eq or not isinstance(other, DistanceFeature):
            return False
        return (self.periodic == other.periodic and np.all(self.distance_indexes == other.distance_indexes)
            and self.prefix_label == other.prefix_label)


class InverseDistanceFeature(DistanceFeature):
    __serialize_version = 0

    prefix_label = "INVDIST:"

    def __init__(self, top, distance_indexes, periodic=True):
        DistanceFeature.__init__(self, top, distance_indexes, periodic=periodic)

    def transform(self, traj):
        return 1.0 / mdtraj.compute_distances(traj, self.distance_indexes, periodic=self.periodic)

    # does not need own eq impl, since we take prefix label into account


class ResidueMinDistanceFeature(DistanceFeature):
    __serialize_version = 0
    __serialize_fields = ('contacts', 'scheme', 'ignore_nonprotein', 'threshold', 'prefix_label')

    def __init__(self, top, contacts, scheme, ignore_nonprotein, threshold, periodic):
        self.top = top
        self.contacts = contacts
        self.scheme = scheme
        self.threshold = threshold
        self.prefix_label = "RES_DIST (%s)" % scheme
        self.periodic = periodic
        self.ignore_nonprotein = ignore_nonprotein

        # mdtraj.compute_contacts might ignore part of the user input (if it is contradictory) and
        # produce a warning. I think it is more robust to let it run once on a dummy trajectory to
        # see what the actual size of the output is:
        dummy_traj = mdtraj.Trajectory(np.zeros((self.top.n_atoms, 3)), self.top)
        dummy_dist, dummy_pairs = mdtraj.compute_contacts(dummy_traj, contacts=contacts,
                                                          scheme=scheme, periodic=periodic,
                                                          ignore_nonprotein=ignore_nonprotein)
        self.dimension = dummy_dist.shape[1]
        self.distance_indexes = dummy_pairs

    def describe(self):
        labels = ["%s %s - %s" % (self.prefix_label,
                                  self.top.residue(pair[0]),
                                  self.top.residue(pair[1]))
                  for pair in self.distance_indexes]
        return labels

    def transform(self, traj):
        # We let mdtraj compute the contacts with the input scheme
        D = mdtraj.compute_contacts(traj, contacts=self.contacts, scheme=self.scheme, periodic=self.periodic)[0]
        res = np.zeros_like(D)
        # Do we want binary?
        if self.threshold is not None:
            I = np.argwhere(D <= self.threshold)
            res[I[:, 0], I[:, 1]] = 1.0
        else:
            res = D
        return res

    def __eq__(self, other):
        eq = super(ResidueMinDistanceFeature, self).__eq__(other)
        if not eq or not isinstance(other, ResidueMinDistanceFeature):
            return False
        return (np.all(self.contacts == other.contacts)
                and self.scheme == other.scheme
                and self.periodic == other.periodic
                and self.threshold == other.threshold
                and self.ignore_nonprotein == other.ignore_nonprotein)


class GroupMinDistanceFeature(DistanceFeature):
    __serialize_version = 0
    __serialize_fields = ('group_identifiers', 'group_definitions', 'threshold', 'group_pairs',
                          'distance_indexes')
    prefix_label = "GROUP_MINDIST"

    def __init__(self, top, group_definitions, group_pairs, distance_list, group_identifiers, threshold, periodic):
        super(GroupMinDistanceFeature, self).__init__(distance_indexes=distance_list, top=top)
        self.group_identifiers = group_identifiers
        self.group_definitions = np.array(group_definitions)
        self.threshold = threshold
        self.group_pairs = group_pairs

        self.periodic = periodic
        self.dimension = len(group_pairs)  # TODO: validate

    def describe(self):
        labels = ["%s %u--%u: [%s...%s]--[%s...%s]" % (self.prefix_label, pair[0], pair[1],
                                                       _describe_atom(self.top, self.group_definitions[pair[0]][0]),
                                                       _describe_atom(self.top, self.group_definitions[pair[0]][-1]),
                                                       _describe_atom(self.top, self.group_definitions[pair[1]][0]),
                                                       _describe_atom(self.top, self.group_definitions[pair[1]][-1])
                                                       ) for pair in self.group_pairs]
        return labels

    def transform(self, traj):
        # All needed distances
        Dall = mdtraj.compute_distances(traj, self.distance_indexes, periodic=self.periodic)
        # Just the minimas
        Dmin = np.zeros((traj.n_frames, self.dimension))
        res = np.zeros_like(Dmin)
        # Compute the min groupwise
        for ii, (gi, gf) in enumerate(self.group_identifiers):
            Dmin[:, ii] = Dall[:, gi:gf].min(1)
        # Do we want binary?
        if self.threshold is not None:
            I = np.argwhere(Dmin <= self.threshold)
            res[I[:, 0], I[:, 1]] = 1.0
        else:
            res = Dmin

        return res

    def __eq__(self, other):
        eq = super(GroupMinDistanceFeature, self).__eq__(other)
        if not eq or not isinstance(other, GroupMinDistanceFeature):
            return False
        return (np.all(self.group_identifiers == other.group_identifiers)
                and np.all(self.group_definitions == other.group_definitions)
                and np.all(self.group_pairs == other.group_pairs)
                and self.threshold == other.threshold
                and self.periodic == other.periodic)


class ContactFeature(DistanceFeature):
    prefix_label = "CONTACT:"
    __serialize_version = 0
    __serialize_fields = ('distance_indexes', 'prefix_label', 'count_contacts', 'threshold')

    def __init__(self, top, distance_indexes, threshold=5.0, periodic=True, count_contacts=False):
        DistanceFeature.__init__(self, top, distance_indexes, periodic=periodic)
        if count_contacts:
            self.prefix_label = "counted " + self.prefix_label
        self.threshold = threshold
        self.count_contacts = count_contacts
        if count_contacts:
            self.dimension = 1
        else:
            self.dimension = len(self.distance_indexes)

    def transform(self, traj):
        dists = mdtraj.compute_distances(
            traj, self.distance_indexes, periodic=self.periodic)
        res = np.zeros(
            (len(traj), self.distance_indexes.shape[0]), dtype=np.float32)
        I = np.argwhere(dists <= self.threshold)
        res[I[:, 0], I[:, 1]] = 1.0
        if self.count_contacts:
            return res.sum(1, keepdims=True)
        else:
            return res

    def __eq__(self, other):
        eq = super(ContactFeature, self).__eq__(other)
        if not eq or not isinstance(other, ContactFeature):
            return False
        return (self.count_contacts == other.count_contacts
                and self.threshold == other.threshold and self.periodic == other.periodic)
