from abc import ABCMeta, abstractmethod

import numpy as np
import numbers

import six


class NotRandomAccessibleException(Exception):
    pass


class TrajectoryRandomAccessible(object):
    def __init__(self):
        self._ra_cuboid = NotImplementedRandomAccessStrategy(self)
        self._ra_linear_strategy = NotImplementedRandomAccessStrategy(self)
        self._ra_linear_itraj_strategy = NotImplementedRandomAccessStrategy(self)
        self._ra_jagged = NotImplementedRandomAccessStrategy(self)
        self._is_random_accessible = False

    @property
    def is_random_accessible(self):
        """
        Check if self._is_random_accessible is set to true and if all the random access strategies are implemented.
        Returns
        -------
        bool : Returns True if random accessible via strategies and False otherwise.
        """
        return self._is_random_accessible and \
               not isinstance(self.ra_itraj_cuboid, NotImplementedRandomAccessStrategy) and \
               not isinstance(self.ra_linear, NotImplementedRandomAccessStrategy) and \
               not isinstance(self.ra_itraj_jagged, NotImplementedRandomAccessStrategy) and \
               not isinstance(self.ra_itraj_linear, NotImplementedRandomAccessStrategy)

    @property
    def ra_itraj_cuboid(self):
        """
        Implementation of random access with slicing that can be up to 3-dimensional, where the first dimension
        corresponds to the trajectory index, the second dimension corresponds to the frames and the third dimension
        corresponds to the dimensions of the frames.

        The with the frame slice selected frames will be loaded from each in the trajectory-slice selected trajectories
        and then sliced with the dimension slice. For example: The data consists out of three trajectories with length
        10, 20, 10, respectively. The slice `data[:, :15, :3]` returns a 3D array of shape (3, 10, 3), where the first
        component corresponds to the three trajectories, the second component corresponds to 10 frames (note that
        the last 5 frames are being truncated as the other two trajectories only have 10 frames) and the third component
        corresponds to the selected first three dimensions.

        :return: Returns an object that allows access by slices in the described manner.
        """
        if not self._is_random_accessible:
            raise NotRandomAccessibleException()
        return self._ra_cuboid

    @property
    def ra_itraj_jagged(self):
        """
        Behaves like ra_itraj_cuboid just that the trajectories are not truncated and returned as a list.

        :return: Returns an object that allows access by slices in the described manner.
        """
        if not self._is_random_accessible:
            raise NotRandomAccessibleException()
        return self._ra_jagged

    @property
    def ra_linear(self):
        """
        Implementation of random access that takes a (maximal) two-dimensional slice where the first component
        corresponds to the frames and the second component corresponds to the dimensions. Here it is assumed that
        the frame indexing is contiguous, i.e., the first frame of the second trajectory has the index of the last frame
        of the first trajectory plus one.

        :return: Returns an object that allows access by slices in the described manner.
        """
        if not self._is_random_accessible:
            raise NotRandomAccessibleException()
        return self._ra_linear_strategy

    @property
    def ra_itraj_linear(self):
        """
        Implementation of random access that takes arguments as the default random access (i.e., up to three dimensions
        with trajs, frames and dims, respectively), but which considers the frame indexing to be contiguous. Therefore,
        it returns a simple 2D array.

        :return: A 2D array of the sliced data containing [frames, dims].
        """
        if not self._is_random_accessible:
            raise NotRandomAccessibleException()
        return self._ra_linear_itraj_strategy


class RandomAccessStrategy(six.with_metaclass(ABCMeta)):
    """
    Abstract parent class for all random access strategies. It holds its corresponding data source and
    implements `__getitem__` as well as `__getslice__`, which both get delegated to `_handle_slice`.
    """
    def __init__(self, source, max_slice_dimension=-1):
        self._source = source
        self._max_slice_dimension = max_slice_dimension

    @abstractmethod
    def _handle_slice(self, idx):
        pass

    @property
    def max_slice_dimension(self):
        """
        Property that returns how many dimensions the slice can have.
        Returns
        -------
        int : the maximal slice dimension
        """
        return self._max_slice_dimension

    def __getitem__(self, idx):
        return self._handle_slice(idx)

    def __getslice__(self, start, stop):
        """For slices of the form data[1:3]."""
        return self.__getitem__(slice(start, stop))

    def _get_indices(self, item, length):
        if isinstance(item, slice):
            item = np.array(range(*item.indices(length)))
        elif not isinstance(item, np.ndarray):
            if isinstance(item, list):
                item = np.array(item)
            else:
                item = np.arange(0, length)[item]
                if isinstance(item, numbers.Integral):
                    item = np.array([item])
        return item

    def _max(self, elems):
        if isinstance(elems, numbers.Integral):
            elems = [elems]
        return max(elems)


class NotImplementedRandomAccessStrategy(RandomAccessStrategy):
    def _handle_slice(self, idx):
        raise NotImplementedError("Requested random access strategy is not implemented for the current data source.")
