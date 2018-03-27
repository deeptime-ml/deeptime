from pyemma._base.serialization.serialization import SerializableMixIn

from pyemma.coordinates.data._base.datasource import DataSource
from pyemma.coordinates.data.data_in_memory import DataInMemoryIterator
from pyemma.coordinates.data.util.traj_info_cache import TrajInfo

__author__ = 'marscher'


class H5Reader(DataSource, SerializableMixIn):
    r""" Reader for HDF5 files.

    The reader needs h5py and optionally pytables installed. The first package is used for the actual file handling,
    while the latter is only used to import compression filters (eg. BLOSC).

    Parameters
    ----------
    filenames: list of str

    selection: str or mapping, default='/*'
        HDF5-path selection (group0/group_x/dataset_name) given as a regular expression. Eg.
             /group1/.*/my_dataset
        will selection all 'my_dataset' in group1 no matter which kind of sub-groups these may belong to.

        If a mapping is provided, it shall contain one selection per input file name.

        Note, that selections may contain wildcards to select multiple groups (trees) and data sets.
        The default selection will match all groups and all data sets.
        All selections have to begin with the root node '/'.

    chunk_size: int
    """
    __serialize_version = 0

    def __init__(self, filenames, selection='/*', chunk_size=5000, **kw):
        super(H5Reader, self).__init__(chunksize=chunk_size)

        self._is_reader = True
        self._is_random_accessible = True

        from pyemma.coordinates.data.data_in_memory import (DataInMemoryCuboidRandomAccessStrategy,
                                                            DataInMemoryJaggedRandomAccessStrategy,
                                                            DataInMemoryLinearRandomAccessStrategy,
                                                            DataInMemoryLinearItrajRandomAccessStrategy)
        self._ra_cuboid = DataInMemoryCuboidRandomAccessStrategy(self, 3)
        self._ra_jagged = DataInMemoryJaggedRandomAccessStrategy(self, 3)
        self._ra_linear_strategy = DataInMemoryLinearRandomAccessStrategy(self, 2)
        self._ra_linear_itraj_strategy = DataInMemoryLinearItrajRandomAccessStrategy(self, 3)

        # set selection first, so we can use it the filename setter.
        self.selection = selection
        # we count data sets as itrajs, because a hdf5 file can contain multiple data sets.
        from collections import defaultdict
        self._itraj_dataset_mapping = defaultdict(int)

        # we explicitly do not want to cache anything for H5, because the user can provide different selections
        # and the interface of the cache does not allow for such a mapping (1:1 relation filename:(dimension, len)).
        from pyemma.util.contexts import settings
        with settings(use_trajectory_lengths_cache=False):
            self.filenames = filenames

        # we need to override the ntraj attribute to be equal with the itraj_counter to respect all data sets.
        self._ntraj = self._itraj_counter

        # sanity
        if self._itraj_counter == 0:
            raise ValueError('Your provided selection did not match anything in your provided files. '
                             'Check the log output')

    def __reduce__(self):
        return H5Reader, (self.filenames, self.selection, self.chunksize)

    @property
    def selection(self):
        return self._selection

    @selection.setter
    def selection(self, value):
        # 1. open all files, and check that selection matches for each file
        # 2. check that dimension are all the same for each file
        if isinstance(value, dict):
            # check if we have a wildcard
            if any(('*' in e for e in value)):
                # TODO: expand this to match the len of filenames
                pass
            else:
                pass
        elif isinstance(value, str):
            if not value.startswith('/'):
                raise ValueError('selection has to start with the root node "/". '
                                 'Followed by an explicit group name or wildcard specifier.')

        self._selection = value
        # since we have update the selection, reset the counter for itrajs.
        self._itraj_counter = 0
        # we re-assign the file names here to reflect the updated selection (if filenames was set before).
        if self._filenames is not None:
            self.filenames = self.filenames

    @property
    def n_datasets(self):
        return self._itraj_counter

    def _reshape(self, array, dry=False):
        """
        reshape given array to 2d. If dry is True, the actual reshaping is not performed.
        returns tuple (array, shape_2d)
        """
        import functools, numpy as np
        if array.ndim == 1:
            shape = (array.shape[0], 1)
        else:
            # hold first dimension, multiply the rest
            shape = (array.shape[0], functools.reduce(lambda x, y: x * y, array.shape[1:]))
        if not dry:
            array = np.reshape(array, shape)
        return array, shape

    def _get_traj_info(self, filename):
        # noinspection PyUnresolvedReferences
        import tables
        import h5py

        with h5py.File(filename, mode='r') as f:
            try:
                sel = self.selection[filename]
            except (KeyError, TypeError):
                sel = self.selection
            import re

            # unfortunately keys do not start with a root, so insert it now to simplify matching.
            keys = list(f.keys())
            for i, k in enumerate(keys):
                if not k.startswith('/'):
                    keys[i] = '/' + k

            def name_matches_selection(_, obj, matches):
                if not isinstance(obj, h5py.Dataset):
                    return

                m = re.match(sel, obj.name)
                if m is not None:
                    matches.append(m)
            from functools import partial
            matches = []
            f.visititems(partial(name_matches_selection, matches=matches))

            if not matches:
                self.logger.warning('selection "%s" did not match any group/dataset in file "%s"', sel, filename)

            children = []
            for m in matches:
                path = m.string
                h5_item = f[path]
                _, shape_2d = self._reshape(h5_item, dry=True)
                lengths, ndim = shape_2d
                self._itraj_dataset_mapping[self._itraj_counter] = (filename, path)
                self._itraj_counter += 1
                children.append(TrajInfo(ndim, lengths))

        if children:
            t = children[0]
            t.children = children[1:]
        else:
            t = TrajInfo(-1, 0)
        return t

    def _load_file(self, itraj):
        # noinspection PyUnresolvedReferences
        import tables
        import h5py
        filename, path = self._itraj_dataset_mapping[itraj]
        self.logger.debug('load file %s with path %s', filename, path)
        file = h5py.File(filename, 'r')
        ds = file[path]
        return ds

    def _create_iterator(self, skip=0, chunk=0, stride=1, return_trajindex=True, cols=None):
        return H5Iterator(self, skip=skip, chunk=chunk, stride=stride, return_trajindex=return_trajindex, cols=cols)


class H5Iterator(DataInMemoryIterator):
    def __init__(self, data_source, skip=0, chunk=0, stride=1, return_trajindex=False, cols=False):
        super(H5Iterator, self).__init__(data_source=data_source, skip=skip,
                                         chunk=chunk, stride=stride,
                                         return_trajindex=return_trajindex,
                                         cols=cols)

    def close(self):
        if hasattr(self, '_fh'):
            self._fh.close()
            del self._fh

    def _select_file(self, itraj):
        if self._selected_itraj != itraj:
            self.close()
            self._t = 0
            self._itraj = itraj
            self._selected_itraj = self._itraj
            if itraj < self.number_of_trajectories():
                self.data = self._data_source._load_file(itraj)
                self._fh = self.data.file

    def _next_chunk(self):
        X = self._next_chunk_impl(self.data)
        X, _ = self._data_source._reshape(X, dry=False)
        return X
