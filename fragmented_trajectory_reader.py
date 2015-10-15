from pyemma.coordinates.data.interface import ReaderInterface

from six import string_types
from pyemma.coordinates.data.util.reader_utils import create_file_reader

class FragmentedTrajectoryReader(ReaderInterface):

    def __init__(self, trajectories, topologyfile=None, chunksize=100, featurizer=None):
        # sanity checks
        assert isinstance(trajectories, (list, tuple)), "input trajectories should be of list or tuple type"
        for item in trajectories:
            assert isinstance(item, string_types), \
                "all items of the trajectory list should be of string type"
            super(FragmentedTrajectoryReader, self).__init__(chunksize=chunksize)
        readers = []
        for input_item in trajectories:
            reader = create_file_reader(input_item, topologyfile, featurizer, chunksize)
            readers.append(reader)
        # store readers
        self._readers = readers
        # store trajectory files
        self._trajectories = trajectories
        # one (composite) trajectory
        self._ntraj = 1
        # lengths array per reader
        self._reader_lengths = [reader.trajectory_length(0, 1) for reader in self._readers]
        # composite trajectory length
        self._lengths = [sum(self._reader_lengths)]

    def describe(self):
        return "[FragmentedTrajectoryReader files=%s]" % self._trajectories

    def dimension(self):
        pass

    def _close(self):
        for reader in self._readers:
            reader._close()

    def _reset(self, context=None):
        for reader in self._readers:
            reader._reset(context)

    def _next_chunk(self, ctx):
        pass

    def parametrize(self, stride=1):
        for reader in self._readers:
            reader.parameterize(stride)