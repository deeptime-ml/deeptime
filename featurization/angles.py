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
import functools

from mdtraj.geometry.dihedral import (indices_phi,
                                      indices_psi,
                                      indices_chi1,
                                      )
import mdtraj

from pyemma.coordinates.data.featurization._base import Feature
from pyemma.coordinates.data.featurization.util import (_hash_numpy_array,
                                                        hash_top, _describe_atom)
import numpy as np


class AngleFeature(Feature):

    def __init__(self, top, angle_indexes, deg=False, cossin=False, periodic=True):
        self.top = top
        self.angle_indexes = np.array(angle_indexes)
        if len(self.angle_indexes) == 0:
            raise ValueError("empty indices")
        self.deg = deg
        self.cossin = cossin
        self.periodic = periodic
        self._dim = len(self.angle_indexes)
        if cossin:
            self._dim *= 2

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
            rad = rad.reshape(functools.reduce(lambda x, y: x * y, rad.shape),)
        if self.deg:
            return np.rad2deg(rad)
        else:
            return rad

    def __hash__(self):
        hash_value = _hash_numpy_array(self.angle_indexes)
        hash_value ^= hash_top(self.top)
        hash_value ^= hash(self.deg)
        hash_value ^= hash(self.cossin)

        return hash_value


class DihedralFeature(AngleFeature):

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
            rad = rad.reshape(rad.shape[0], rad.shape[1]*rad.shape[2])
        # convert to degrees
        if self.deg:
            rad = np.rad2deg(rad)

        return rad


class BackboneTorsionFeature(DihedralFeature):

    def __init__(self, topology, selstr=None, deg=False, cossin=False, periodic=True):
        indices = indices_phi(topology)

        if not selstr:
            self._phi_inds = indices
        else:
            self._phi_inds = indices[np.in1d(indices[:, 1],
                                             topology.select(selstr), assume_unique=True)]

        indices = indices_psi(topology)
        if not selstr:
            self._psi_inds = indices
        else:
            self._psi_inds = indices[np.in1d(indices[:, 1],
                                             topology.select(selstr), assume_unique=True)]

        # alternate phi, psi pairs (phi_1, psi_1, ..., phi_n, psi_n)
        dih_indexes = np.array(list(phi_psi for phi_psi in
                                    zip(self._phi_inds, self._psi_inds))).reshape(-1, 4)

        super(BackboneTorsionFeature, self).__init__(topology, dih_indexes,
                                                     deg=deg, cossin=cossin,
                                                     periodic=periodic)

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

    def __init__(self, topology, selstr=None, deg=False, cossin=False, periodic=True):
        indices = indices_chi1(topology)
        if not selstr:
            dih_indexes = indices
        else:
            dih_indexes = indices[np.in1d(indices[:, 1],
                                          topology.select(selstr),
                                          assume_unique=True)]
        super(Chi1TorsionFeature, self).__init__(topology, dih_indexes,
                                                 deg=deg, cossin=cossin,
                                                 periodic=periodic)

    def describe(self):
        top = self.top
        getlbl = lambda at: "%i %s %i " \
            % (at.residue.chain.index, at.residue.name, at.residue.resSeq)
        if self.cossin:
            cossin = ("COS(CHI1 %s)", "SIN(CHI1 %s)")
            labels_chi1 = [s % getlbl(top.atom(ires[1]))
                           for ires in self.angle_indexes
                           for s in cossin]
        else:
            labels_chi1 = ["CHI1" + getlbl(top.atom(ires[1]))
                           for ires in self.angle_indexes]

        return labels_chi1
