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
from pyemma.util.indices import (combinations,
                                 product)
from pyemma.util.types import is_iterable_of_int, is_string
from six import PY3

import numpy as np


def _describe_atom(topology, index):
    """
    Returns a string describing the given atom

    :param topology:
    :param index:
    :return:
    """
    at = topology.atom(index)
    return "%s %i %s %i" % (at.residue.name, at.residue.resSeq, at.name, at.index)


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

    if is_iterable_of_int(indices1):
        MDlogger.warning('The 1D arrays input for %s have been sorted, and '
                         'index duplicates have been eliminated.\n'
                         'Check the output of describe() to see the actual order of the features' % fname)

        # Eliminate duplicates and sort
        indices1 = np.unique(indices1)

        # Intra-group distances
        if indices2 is None:
            atom_pairs = combinations(indices1, 2)

        # Inter-group distances
        elif is_iterable_of_int(indices2):

            # Eliminate duplicates and sort
            indices2 = np.unique(indices2)

            # Eliminate duplicates between indices1 and indices1
            uniqs = np.in1d(indices2, indices1, invert=True)
            indices2 = indices2[uniqs]
            atom_pairs = product(indices1, indices2)

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
    if is_string(group_pairs):
        if group_pairs == 'all':
            parsed_group_pairs = combinations(np.arange(len(group_definitions)), 2)
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
            parsed_group_pairs[ii, :] = np.sort(ipair)

    # Create the large list of distances that will be computed, and an array containing group identfiers
    # of the distances that actually characterize a pair of groups
    distance_pairs = []
    group_membership = np.zeros_like(parsed_group_pairs)
    b = 0
    for ii, pair in enumerate(parsed_group_pairs):
        if pair[0] != pair[1]:
            distance_pairs.append(product(parsed_group_definitions[pair[0]],
                                          parsed_group_definitions[pair[1]]))
        else:
            parsed = parsed_group_definitions[pair[0]]
            distance_pairs.append(combinations(parsed, 2))

        group_membership[ii, :] = [b, b + len(distance_pairs[ii])]
        b += len(distance_pairs[ii])

    return parsed_group_definitions, parsed_group_pairs, np.vstack(distance_pairs), group_membership

