import abc

import numpy as np


class Reader(object):
    pass


class DataSet(object):
    __metaclass__ = abc.ABCMeta

    def __iter__(self):
        """ Creates an iterator for the elements of this data set.

        Returns
        -------
        The iterator.
        """
        #
        # return DataSetIterator(self)
        #
        pass

    @abc.abstractmethod
    def shapes(self):
        """
        Gives the outputted shapes.

        Returns
        -------
        List of shapes
        """
        pass

    @staticmethod
    def from_reader(reader: Reader):
        """
        This could probably be collapsed into `from_generator`

        Parameters
        ----------
        reader the reader

        Returns
        -------
        a data set for the data in the reader's file

        """
        pass

    @staticmethod
    def from_numpy(array: np.ndarray):
        pass

    @staticmethod
    def from_generator(generator_fn, output_shapes):
        pass

    def zip(self, data_sets):
        return ZipDataSet(data_sets)

    def lag(self, lag_time: int, drop_tail: bool = False):
        """

        Parameters
        ----------
        lag_time: the lag time
        drop_tail: whether to include the tail

        Returns
        -------
        DataSet with lagged data
        """
        pass


class ZipDataSet(DataSet):

    def __init__(self, data_sets):
        self._data_sets = data_sets

    def shapes(self):
        return (ds.shapes() for ds in self._data_sets)
