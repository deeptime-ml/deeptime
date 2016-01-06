import numbers

from pyemma.coordinates.data.datasource import DataSource


class RandomAccessibleDataSource(DataSource, tuple):

    def __getitem__(self, idx):
        # todo
        if isinstance(idx, (numbers.Integral, slice)):
            idx = (idx, slice(None, None, None))
        elif len(idx) == 1:
            idx = (idx[0], slice(None, None, None))
        row_idx, col_idx = idx
        row_slice, col_slice = True, True
        if isinstance(row_idx, numbers.Integral):
            row_idx, row_slice = slice(row_idx, row_idx+1), False
        if isinstance(col_idx, numbers.Integral):
            col_idx, col_slice = slice(col_idx, col_idx+1), False
        ret = self.m[row_idx][col_idx]
        if not col_slice:
            ret = [row[0] for row in ret]
        if not row_slice:
            ret = ret[0]
        return ret