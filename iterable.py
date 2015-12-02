from abc import ABCMeta, abstractmethod
import six


class Iterable(six.with_metaclass(ABCMeta)):
    def __init__(self, chunksize=100):
        self._default_chunksize = chunksize
        if self.default_chunksize < 0:
            raise ValueError("Chunksize of %s was provided, but has to be >= 0" % self.default_chunksize)

    @property
    def default_chunksize(self):
        return self._default_chunksize

    def iterator(self, lag=0, chunk=None, stride=1, return_trajindex=False):
        chunk = chunk if chunk is not None else self.default_chunksize
        it = self._create_iterator(skip=0, chunk=chunk, stride=stride, return_trajindex=return_trajindex)
        if lag > 0:
            it_lagged = self._create_iterator(skip=lag, chunk=chunk, stride=stride, return_trajindex=False)
            return LaggedIterator(it, it_lagged, return_trajindex)
        return it

    @abstractmethod
    def _create_iterator(self, skip=0, chunk=0, stride=1, return_trajindex=False):
        """
        Should be implemented by non-abstract subclasses. Creates an instance-independent iterator.
        :param skip: How many frames to skip before streaming.
        :param chunk: The chunksize.
        :param stride: Take only every stride'th frame.
        :param return_trajindex: take the trajindex into account
        :return: a chunk of data if return_trajindex is False, otherwise a tuple of (trajindex, data).
        """
        pass


class LaggedIterator(object):
    def __init__(self, it, it_lagged, return_trajindex):
        self._it = it
        self._it_lagged = it_lagged
        self._return_trajindex = return_trajindex

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        data = self._it.next()
        itraj = None
        if self._return_trajindex:
            itraj, data = data
        data_lagged = self._it_lagged.next()
        if data.shape[0] > data_lagged.shape[0]:
            # data chunk is bigger, truncate it to match data_lagged's shape
            data = data[:data_lagged.shape[0]]
        elif data.shape[0] < data_lagged.shape[0]:
            raise RuntimeError("chunk was smaller than time-lagged chunk (%s < %s), that should not happen!"
                               % (data.shape[0], data_lagged.shape[0]))
        if self._return_trajindex:
            return itraj, data, data_lagged
        return data, data_lagged
