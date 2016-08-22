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

from abc import ABCMeta, abstractmethod
from math import ceil

import numpy as np
import six

from pyemma.coordinates.data._base.iterable import Iterable
from pyemma.coordinates.data._base.random_accessible import TrajectoryRandomAccessible
from pyemma.util import config
from six import string_types
import os


class DataSource(Iterable, TrajectoryRandomAccessible):
    """
    Superclass for all pipeline elements. It inherits "Iterable", therefore serves as an iterator factory over the
    data it holds. The difference to Iterable is that DataSource is specialized for trajectories, whereas the concept
    of trajectories is generally unknown for Iterable.
    """

    def __init__(self, chunksize=1000):
        super(DataSource, self).__init__(chunksize=chunksize)

        # following properties have to be set in subclass
        self._ntraj = 0
        self._lengths = []
        self._offsets = []
        self._filenames = None
        self._is_reader = False

    @property
    def ntraj(self):
        __doc__ = self.number_of_trajectories.__doc__
        return self._ntraj

    @property
    def filenames(self):
        """ Property which returns a list of filenames the data is originally from.
        Returns
        -------
        list of str : list of filenames if data is originating from a file based reader
        """
        if self._is_reader:
            assert self._filenames is not None
            return self._filenames
        else:
            return self.data_producer.filenames

    @filenames.setter
    def filenames(self, filename_list):

        if isinstance(filename_list, string_types):
            filename_list = [filename_list]

        uniq = set(filename_list)
        if len(uniq) != len(filename_list):
            self.logger.warning("duplicate files/arrays detected")
            filename_list = list(uniq)

        from pyemma.coordinates.data.data_in_memory import DataInMemory

        if self._is_reader:
            if isinstance(self, DataInMemory):
                import warnings
                warnings.warn('filenames are not being used for DataInMemory')
                return

            self._ntraj = len(filename_list)
            if self._ntraj == 0:
                raise ValueError("empty file list")

            # validate files
            for f in filename_list:
                try:
                    stat = os.stat(f)
                except EnvironmentError:
                    self.logger.exception('Error during access of file "%s"' % f)
                    raise ValueError('could not read file "%s"' % f)

                if not os.path.isfile(f): # can be true for symlinks to directories
                    raise ValueError('"%s" is not a valid file')

                if stat.st_size == 0:
                    raise ValueError('file "%s" is empty' % f)

            # number of trajectories/data sets
            self._filenames = filename_list
            # determine len and dim via cache lookup,
            lengths = []
            offsets = []
            ndims = []
            # avoid cyclic imports
            from pyemma.coordinates.data.util.traj_info_cache import TrajectoryInfoCache
            if len(filename_list) > 3:
                self._progress_register(len(filename_list), 'Obtaining file info')
            for filename in filename_list:
                if config.use_trajectory_lengths_cache:
                    info = TrajectoryInfoCache.instance()[filename, self]
                else:
                    info = self._get_traj_info(filename)
                lengths.append(info.length)
                offsets.append(info.offsets)
                ndims.append(info.ndim)
                if len(filename_list) > 3:
                    self._progress_update(1)

            # ensure all trajs have same dim
            if not np.unique(ndims).size == 1:
                # group files by their dimensions to give user indicator
                ndims = np.array(ndims)
                filename_list = np.asarray(filename_list)
                sort_inds = np.argsort(ndims)
                import itertools, operator
                res = {}
                for dim, files in itertools.groupby(zip(ndims[sort_inds], filename_list[sort_inds]),
                                                    operator.itemgetter(0)):
                    res[dim] = list(f[1] for f in files)

                raise ValueError("Input data has different dimensions ({dims})!"
                                 " Files grouped by dimensions: {groups}".format(dims=res.keys(),
                                                                                  groups=res))

            self._ndim = ndims[0]
            self._lengths = lengths
            self._offsets = offsets

        else:
            # propagate this until we finally have a a reader
            self.data_producer.filenames = filename_list

    @property
    def is_reader(self):
        """
        Property telling if this data source is a reader or not.
        Returns
        -------
        bool: True if this data source is a reader and False otherwise
        """
        return self._is_reader

    @property
    def data_producer(self):
        """
        The data producer for this data source object (can be another data source object).
        Returns
        -------
        This data source's data producer.
        """
        return self

    def number_of_trajectories(self):
        r""" Returns the number of trajectories.

        Returns
        -------
            int : number of trajectories
        """
        return self._ntraj

    def trajectory_length(self, itraj, stride=1, skip=None):
        r"""Returns the length of trajectory of the requested index.

        Parameters
        ----------
        itraj : int
            trajectory index
        stride : int
            return value is the number of frames in the trajectory when
            running through it with a step size of `stride`.

        Returns
        -------
        int : length of trajectory
        """
        if itraj >= self._ntraj:
            raise IndexError("given index (%s) exceeds number of data sets (%s)."
                             " Zero based indexing!" % (itraj, self._ntraj))
        if isinstance(stride, np.ndarray):
            selection = stride[stride[:, 0] == itraj][:, 0]
            return 0 if itraj not in selection else len(selection)
        else:
            return (self._lengths[itraj] - (0 if skip is None else skip) - 1) // int(stride) + 1

    def trajectory_lengths(self, stride=1, skip=0):
        r""" Returns the length of each trajectory.

        Parameters
        ----------
        stride : int
            return value is the number of frames of the trajectories when
            running through them with a step size of `stride`.
        skip : int
            skip parameter

        Returns
        -------
        array(dtype=int) : containing length of each trajectory
        """
        n = self.number_of_trajectories()
        if isinstance(stride, np.ndarray):
            return np.fromiter((self.trajectory_length(itraj, stride)
                                for itraj in range(n)),
                               dtype=int, count=n)
        else:
            return np.fromiter(((l - skip - 1) // stride + 1 for l in self._lengths),
                               dtype=int, count=n)

    def n_frames_total(self, stride=1, skip=0):
        r"""Returns total number of frames.

        Parameters
        ----------
        stride : int
            return value is the number of frames in trajectories when
            running through them with a step size of `stride`.
        skip : int, default=0
            skip the first initial n frames per trajectory.
        Returns
        -------
        n_frames_total : int
            total number of frames.
        """
        if isinstance(stride, np.ndarray):
            return stride.shape[0]

        return sum(self.trajectory_lengths(stride=stride, skip=skip))


class IteratorState(object):
    """
    State class holding all the relevant information of an iterator's state.
    """

    def __init__(self, skip=0, chunk=0, return_trajindex=False, ntraj=0, cols=None):
        self.skip = skip
        self.chunk = chunk
        self.return_trajindex = return_trajindex
        self.itraj = 0
        self.ntraj = ntraj
        self.t = 0
        self.pos = 0
        self.pos_adv = 0
        self.stride = None
        self.uniform_stride = False
        self.traj_keys = None
        self.trajectory_lengths = None
        self.ra_indices_for_traj_dict = {}
        self.cols = cols

    def ra_indices_for_traj(self, traj):
        """
        Gives the indices for a trajectory file index (without changing the order within the trajectory itself).
        :param traj: a trajectory file index
        :return: a Nx1 - np.array of the indices corresponding to the trajectory index
        """
        assert not self.uniform_stride, "requested random access indices, but is in uniform stride mode"
        if traj in self.traj_keys:
            return self.ra_indices_for_traj_dict[traj]
        else:
            return np.array([])

    def ra_trajectory_length(self, traj):
        assert not self.uniform_stride, "requested random access trajectory length, but is in uniform stride mode"
        return int(self.trajectory_lengths[np.where(self.traj_keys == traj)]) if traj in self.traj_keys else 0

    @staticmethod
    def is_uniform_stride(stride):
        return not isinstance(stride, np.ndarray)

    def is_stride_sorted(self):
        if not self.uniform_stride:
            stride_traj_keys = self.stride[:, 0]
            if not all(np.diff(stride_traj_keys) >= 0):
                # traj keys were not sorted
                return False
            for idx in self.traj_keys:
                if not all(np.diff(self.stride[stride_traj_keys == idx][:, 1]) >= 0):
                    # traj indices were not sorted
                    return False
        return True


class DataSourceIterator(six.with_metaclass(ABCMeta)):
    """
    Abstract class for any data source iterator.
    """
    def __init__(self, data_source, skip=0, chunk=0, stride=1, return_trajindex=False, cols=None):
        self._data_source = data_source
        self.state = IteratorState(skip=skip, chunk=chunk,
                                   return_trajindex=return_trajindex,
                                   ntraj=self.number_of_trajectories(),
                                   cols=cols)
        self.__init_stride(stride)
        self._pos = 0
        self._last_chunk_in_traj = False

    def __init_stride(self, stride):
        self.state.stride = stride
        if isinstance(stride, np.ndarray):
            # shift frame indices by skip
            self.state.stride[:, 1] += self.state.skip
            keys = stride[:, 0]
            if keys.max() >= self.number_of_trajectories():
                raise ValueError("provided too large trajectory index in stride argument (given max index: %s, "
                                 "allowed: %s)" % (keys.max(), self.number_of_trajectories() - 1))
            self.state.traj_keys, self.state.trajectory_lengths = np.unique(keys, return_counts=True)
            self.state.ra_indices_for_traj_dict = {}
            for traj in self.state.traj_keys:
                self.state.ra_indices_for_traj_dict[traj] = self.state.stride[self.state.stride[:, 0] == traj][:, 1]
        else:
            self.state.traj_keys = None
        self.state.uniform_stride = IteratorState.is_uniform_stride(stride)
        if not IteratorState.is_uniform_stride(stride):
            if not self.state.is_stride_sorted():
                raise ValueError("Only sorted arrays allowed for iterator pseudo random access")
            # skip trajs which are not included in stride
            while self.state.itraj not in self.state.traj_keys and self.state.itraj < self._data_source.ntraj:
                self.state.itraj += 1

    def ra_indices_for_traj(self, traj):
        """
        Gives the indices for a trajectory file index (without changing the order within the trajectory itself).
        :param traj: a trajectory file index
        :return: a Nx1 - np.array of the indices corresponding to the trajectory index
        """
        return self.state.ra_indices_for_traj(traj)

    def ra_trajectory_length(self, traj):
        return self.state.ra_trajectory_length(traj)

    def is_stride_sorted(self):
        return self.state.is_stride_sorted()

    @property
    def _n_chunks(self):
        """ rough estimate of how many chunks will be processed """
        if self.chunksize != 0:
            if not DataSourceIterator.is_uniform_stride(self.stride):
                chunks = ceil(len(self.stride[:, 0]) / float(self.chunksize))
            else:
                chunks = sum((ceil(l / float(self.chunksize))
                              for l in self.trajectory_lengths()))
        else:
            chunks = self.number_of_trajectories()
        return int(chunks)

    def number_of_trajectories(self):
        return self._data_source.number_of_trajectories()

    def trajectory_length(self):
        return self._data_source.trajectory_length(self._itraj, self.stride, self.skip)

    def trajectory_lengths(self):
        return self._data_source.trajectory_lengths(self.stride, self.skip)

    def n_frames_total(self):
        return self._data_source.n_frames_total(stride=self.stride, skip=self.skip)

    @abstractmethod
    def close(self):
        """ closes the reader"""
        pass

    def reset(self):
        """
        Method allowing to reset the iterator so that it can iteration from beginning on again.
        """
        self._t = 0
        self._itraj = 0

    @property
    def pos(self):
        """
        Gives the current position in the current trajectory.
        Returns
        -------
        int
            The current iterator's position in the current trajectory.
        """
        return self.state.pos

    @property
    def current_trajindex(self):
        """
        Gives the current iterator's trajectory index.
        Returns
        -------
        int
            The current iterator's trajectory index.
        """
        return self.state.current_itraj

    @property
    def use_cols(self):
        return self.state.cols

    @property
    def skip(self):
        """
        Returns the skip value, i.e., the number of frames that are being omitted at the beginning of each
        trajectory.
        Returns
        -------
        int
            The skip value.
        """
        return self.state.skip

    @property
    def _t(self):
        """
        Reader-internal property that tracks the upcoming iterator position. Should not be used within iterator loop.
        Returns
        -------
        int
            The upcoming iterator position.
        """
        return self.state.t

    @_t.setter
    def _t(self, value):
        """
        Reader-internal property that tracks the upcoming iterator position.
        Parameters
        ----------
        value : int
            The upcoming iterator position.
        """
        self.state.t = value

    @property
    def _itraj(self):
        """
        Reader-internal property that tracks the upcoming trajectory index. Should not be used within iterator loop.
        Returns
        -------
        int
            The upcoming trajectory index.
        """
        return self.state.itraj

    @_itraj.setter
    def _itraj(self, value):
        """
        Reader-internal property that tracks the upcoming trajectory index. Should not be used within iterator loop.
        Parameters
        ----------
        value : int
            The upcoming trajectory index.
        """
        if value > self.state.ntraj:  # we never want to increase this value larger than ntraj.
            raise StopIteration("out of files bound")
        self.state.itraj = value

    @skip.setter
    def skip(self, value):
        """
        Sets the skip parameter. This can be used to skip the first n frames of the next trajectory in the iterator.
        Parameters
        ----------
        value : int
            The new skip parameter.
        """
        self.state.skip = value

    @property
    def chunksize(self):
        """
        The current chunksize of the iterator. Can be changed dynamically during iteration.
        Returns
        -------
        int
            The current chunksize of the iterator.
        """
        return self.state.chunk

    @chunksize.setter
    def chunksize(self, value):
        """
        Sets the current chunksize of the iterator. Can be changed dynamically during iteration.
        Parameters
        ----------
        value : int
            The chunksize of the iterator. Required to be non-negative.
        """
        if not value >= 0:
            raise ValueError("chunksize has to be non-negative")
        self.state.chunk = value

    @property
    def stride(self):
        """
        Gives the current stride parameter.
        Returns
        -------
        int
            The current stride parameter.
        """
        return self.state.stride

    @stride.setter
    def stride(self, value):
        """
        Sets the stride parameter.
        Parameters
        ----------
        value : int
            The new stride parameter.
        """
        self.__init_stride(value)

    @property
    def return_traj_index(self):
        """
        Property that gives information whether the trajectory index gets returned during the iteration.
        Returns
        -------
        bool
            True if the trajectory index should be returned, otherwise False.
        """
        return self.state.return_trajindex

    @property
    def traj_keys(self):
        """
        Random access property returning the trajectory indices that were handed in.
        Returns
        -------
        list
            Trajectories that are used in random access.
        """
        return self.state.traj_keys

    @property
    def uniform_stride(self):
        """
        Boolean property that tells if the stride argument was integral (i.e., uniform stride) or a random access
        dictionary.
        Returns
        -------
        bool
            True if the stride argument was integral, otherwise False.
        """
        return self.state.uniform_stride

    @return_traj_index.setter
    def return_traj_index(self, value):
        """
        Setter for return_traj_index, determining if the trajectory index gets returned in the iteration loop.
        Parameters
        ----------
        value : bool
            True if it should be returned, otherwise False
        """
        self.state.return_trajindex = value

    @staticmethod
    def is_uniform_stride(stride):
        return IteratorState.is_uniform_stride(stride)

    @property
    def last_chunk(self):
        """
        Property returning if the current chunk is the last chunk before the iterator terminates.
        Returns
        -------
        bool
            True if the iterator terminates after the current chunk, otherwise False
        """
        return self.current_trajindex == self.number_of_trajectories() - 1 and self.last_chunk_in_traj

    @property
    def last_chunk_in_traj(self):
        """
        Property returning if the current chunk is the last chunk before the iterator terminates or the next trajectory.
        Returns
        -------
        bool
            True if the next chunk either belongs to a new trajectory or the iterator terminates.
        """
        if self.chunksize > 0:
            return self._last_chunk_in_traj
        else:
            return True

    @abstractmethod
    def _next_chunk(self):
        pass

    def __next__(self):
        return self.next()

    def _use_cols(self, X):
        if self.use_cols is not None:
            return X[:, self.use_cols]
        return X

    def _it_next(self):
        # first chunk at all, skip prepending trajectories that are not considered in random access
        if self._t == 0 and self._itraj == 0 and not self.uniform_stride:
            while (self._itraj not in self.traj_keys or self._t >= self.ra_trajectory_length(self._itraj)) \
                    and self._itraj < self.number_of_trajectories():
                self._itraj += 1
        # we have to obtain the current index before invoking next_chunk (which increments itraj)
        self.state.current_itraj = self._itraj
        self.state.pos = self.state.pos_adv
        try:
            X = self._use_cols(self._next_chunk())
        except StopIteration:
            self._last_chunk_in_traj = True
            raise
        if self.state.current_itraj != self._itraj:
            self.state.pos_adv = 0
            self._last_chunk_in_traj = True
        else:
            self.state.pos_adv += len(X)
            if self.uniform_stride:
                length = self._data_source.trajectory_length(itraj=self.state.current_itraj,
                                                             stride=self.stride, skip=self.skip)
            else:
                length = self.ra_trajectory_length(self.state.current_itraj)
            self._last_chunk_in_traj = self.state.pos_adv >= length
        if self.return_traj_index:
            return self.state.current_itraj, X
        return X

    def next(self):
        X = self._it_next()
        while X is not None and (
                (not self.return_traj_index and len(X) == 0) or (self.return_traj_index and len(X[1]) == 0)
        ):
            X = self._it_next()
        return X

    def __iter__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

