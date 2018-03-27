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
import itertools

import mdtraj
import numpy as np
from mdtraj.geometry.dihedral import (indices_phi,
                                      indices_psi,
                                      indices_chi1,
                                      indices_omega)

from pyemma.coordinates.data.featurization._base import Feature
from pyemma.coordinates.data.featurization.util import _describe_atom


class AngleFeature(Feature):
    __serialize_version = 0
    __serialize_fields = ('angle_indexes', 'deg', 'cossin', 'periodic')

    def __init__(self, top, angle_indexes, deg=False, cossin=False, periodic=True):
        self.top = top
        self.angle_indexes = np.array(angle_indexes)
        if len(self.angle_indexes) == 0:
            raise ValueError("empty indices")
        self.deg = deg
        self.cossin = cossin
        self.periodic = periodic
        self.dimension = len(self.angle_indexes)
        if cossin:
            self.dimension *= 2

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

    def transform(self, traj):
        rad = mdtraj.compute_angles(traj, self.angle_indexes, self.periodic)
        if self.cossin:
            rad = np.dstack((np.cos(rad), np.sin(rad)))
            rad = rad.reshape(rad.shape[0], rad.shape[1] * rad.shape[2])
        if self.deg and not self.cossin:
            return np.rad2deg(rad)
        else:
            return rad

    def __eq__(self, other):
        eq = super(AngleFeature, self).__eq__(other)
        if not eq or not isinstance(other, AngleFeature):
            return False
        return self.cossin == other.cossin and np.all(self.angle_indexes == other.angle_indexes)


class DihedralFeature(AngleFeature):
    __serialize_version = 0

    def __init__(self, top, dih_indexes, deg=False, cossin=False, periodic=True):
        super(DihedralFeature, self).__init__(top=top,
                                              angle_indexes=dih_indexes,
                                              deg=deg,
                                              cossin=cossin,
                                              periodic=periodic)

    def describe(self):
        if self.cossin:
            sin_cos = (
                "DIH: COS(%s -  %s - %s - %s)", "DIH: SIN(%s -  %s - %s - %s)")
            labels = [s %
                      (_describe_atom(self.top, quad[0]),
                       _describe_atom(self.top, quad[1]),
                       _describe_atom(self.top, quad[2]),
                       _describe_atom(self.top, quad[3]))
                      for quad in self.angle_indexes
                      for s in sin_cos]
        else:
            labels = ["DIH: %s - %s - %s - %s " %
                      (_describe_atom(self.top, quad[0]),
                       _describe_atom(self.top, quad[1]),
                       _describe_atom(self.top, quad[2]),
                       _describe_atom(self.top, quad[3]))
                      for quad in self.angle_indexes]
        return labels

    def transform(self, traj):
        rad = mdtraj.compute_dihedrals(traj, self.angle_indexes, self.periodic)
        if self.cossin:
            rad = np.dstack((np.cos(rad), np.sin(rad)))
            rad = rad.reshape(rad.shape[0], rad.shape[1] * rad.shape[2])
        # convert to degrees
        if self.deg and not self.cossin:
            rad = np.rad2deg(rad)

        return rad


class BackboneTorsionFeature(DihedralFeature):
    __serialize_version = 0
    __serialize_fields = ('selstr', '_phi_inds', '_psi_inds')

    def __init__(self, topology, selstr=None, deg=False, cossin=False, periodic=True):
        self.top = topology
        indices = indices_phi(self.top)
        self.selstr = selstr

        if not selstr:
            self._phi_inds = indices
        else:
            self._phi_inds = indices[np.in1d(indices[:, 1],
                                             self.top.select(selstr), assume_unique=True)]

        indices = indices_psi(self.top)
        if not selstr:
            self._psi_inds = indices
        else:
            self._psi_inds = indices[np.in1d(indices[:, 1],
                                             self.top.select(selstr), assume_unique=True)]

        # alternate phi, psi pairs (phi_1, psi_1, ..., phi_n, psi_n)
        dih_indexes = np.array(list(phi_psi for phi_psi in
                                    zip(self._phi_inds, self._psi_inds))).reshape(-1, 4)

        super(BackboneTorsionFeature, self).__init__(self.top, dih_indexes,
                                                     deg=deg, cossin=cossin,
                                                     periodic=periodic)

    def describe(self):
        top = self.top
        getlbl = lambda at: "%i %s %i" % (at.residue.chain.index, at.residue.name, at.residue.resSeq)

        if self.cossin:
            sin_cos = ("COS(PHI %s)", "SIN(PHI %s)")
            labels_phi = [(sin_cos[0] % getlbl(top.atom(ires[1])),
                           sin_cos[1] % getlbl(top.atom(ires[1]))
                           ) for ires in self._phi_inds]
            sin_cos = ("COS(PSI %s)", "SIN(PSI %s)")
            labels_psi = [(sin_cos[0] % getlbl(top.atom(ires[1])),
                           sin_cos[1] % getlbl(top.atom(ires[1]))) for ires in self._psi_inds
                          ]
            # produce the same ordering as the given indices (phi_1, psi_1, ..., phi_n, psi_n)
            # or (cos(phi_1), sin(phi_1), cos(psi_1), sin(psi_1), ..., cos(phi_n), sin(phi_n), cos(psi_n), sin(psi_n))
            res = list(itertools.chain.from_iterable(
                itertools.chain.from_iterable(zip(labels_phi, labels_psi))))
        else:
            labels_phi = [
                "PHI %s" % getlbl(top.atom(ires[1])) for ires in self._phi_inds]
            labels_psi = [
                "PSI %s" % getlbl(top.atom(ires[1])) for ires in self._psi_inds]
            res = list(itertools.chain.from_iterable(zip(labels_phi, labels_psi)))
        return res


class SideChainTorsions(DihedralFeature):
    __serialize_version = 0
    __serialize_fields = ('_prefix_label_lengths',)
    options = ('chi1', 'chi2', 'chi3', 'chi4', 'chi5')

    def __init__(self, top, selstr=None, deg=False, cossin=True, periodic=True, which='all'):
        if not isinstance(which, (tuple, list)):
            which = [which]
        if not set(which).issubset(set(self.options) | {'all'}):
            raise ValueError('Argument "which" should only contain one of {}, but was {}'
                             .format(['all'] + list(self.options), which))
        if 'all' in which:
            which = self.options
        # get all dihedral index pairs
        from mdtraj.geometry import dihedral
        indices_dict = {k: getattr(dihedral, 'indices_%s' % k)(top) for k in which}
        valid = {k: indices_dict[k] for k in indices_dict if indices_dict[k].size > 0}
        if not valid:
            raise ValueError('Could not determine any side chain dihedrals for your topology!')
        self._prefix_label_lengths = np.array([len(indices_dict[k]) if k in which else 0 for k in self.options])
        indices = np.vstack(valid.values())
        if selstr:
            selection = top.select(selstr)
            mask = np.in1d(indices[:, 1], selection, assume_unique=True)
            indices = indices[mask]
        super(SideChainTorsions, self).__init__(top=top, dih_indexes=indices, deg=deg, cossin=cossin, periodic=periodic)

    def describe(self):
        getlbl = lambda at: '%i %s %i' % (at.residue.chain.index, at.residue.name, at.residue.resSeq)
        prefixes = []
        for lengths, label in zip(self._prefix_label_lengths, self.options):
            if self.cossin:
                lengths *= 2
            prefixes.extend([label.upper()] * lengths)

        if self.cossin:
            cossin = ('COS({dih} {res})', 'SIN({dih} {res})')
            labels = [s.format(dih=prefixes[i], res=getlbl(self.top.atom(ires[1])))
                      for i, ires in enumerate(self.angle_indexes)
                      for s in cossin]
        else:
            labels = ['{dih} {res}'.format(dih=prefixes[i], res=getlbl(self.top.atom(ires[1])))
                      for i, ires in enumerate(self.angle_indexes)]

        return labels
