
# This file is part of MSMTools.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# MSMTools is free software: you can redistribute it and/or modify
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

r"""Unit tests for matrix io implementations

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
import os
import unittest

import numpy as np
import scipy.sparse

from msmtools.util import matrix

################################################################################
# util
################################################################################

from os.path import abspath, join
from os import pardir

testpath = abspath(join(abspath(__file__), pardir)) + '/testfiles/'

# this a workaround for a numpy bug in file io (http://github.com/numpy/numpy#5655)
def my_loadtxt(*args, **kw):
    if 'dtype' in kw and (kw['dtype'] == 'complex' or kw['dtype'] is np.complex):
        return np.genfromtxt(*args, **kw)
    else:
        return unpatched_loadtxt(*args, **kw)

unpatched_loadtxt = np.loadtxt
np.loadtxt = my_loadtxt


class TestIsSparseFile(unittest.TestCase):
    def setUp(self):
        self.dense_name1 = 'matrix.dat'
        self.dense_name2 = 'tmp/matrix.npy'
        self.sparse_name1 = 'matrix.coo.dat'
        self.sparse_name2 = 'tmp/matrix.coo.npy'

    def tearDown(self):
        pass

    def test_is_sparse_file(self):
        self.assertTrue(not matrix.is_sparse_file(self.dense_name1))
        self.assertTrue(not matrix.is_sparse_file(self.dense_name2))
        self.assertTrue(matrix.is_sparse_file(self.sparse_name1))
        self.assertTrue(matrix.is_sparse_file(self.sparse_name2))


class TestToDense(unittest.TestCase):
    def setUp(self):
        self.A = scipy.sparse.rand(100, 100)
        self.B = np.random.rand(10, 10)

    def tearDown(self):
        pass

    def test_todense(self):
        self.assertTrue(scipy.sparse.sputils.isdense(matrix.todense(self.A)))
        self.assertTrue(scipy.sparse.sputils.isdense(matrix.todense(self.B)))


class TestToSparse(unittest.TestCase):
    def setUp(self):
        self.A = scipy.sparse.rand(100, 100)
        self.B = np.random.rand(10, 10)

    def tearDown(self):
        pass

    def test_tosparse(self):
        self.assertTrue(scipy.sparse.issparse(matrix.tosparse(self.A)))
        self.assertTrue(scipy.sparse.issparse(matrix.tosparse(self.B)))

    ################################################################################


# ascii
################################################################################

################################################################################
# dense
################################################################################

class TestReadMatrixDense(unittest.TestCase):
    def setUp(self):
        self.filename_int = testpath + 'matrix_int.dat'
        self.filename_float = testpath + 'matrix_float.dat'
        self.filename_complex = testpath + 'matrix_complex.dat'

        self.A_int = np.loadtxt(self.filename_int, dtype=np.int)
        self.A_float = np.loadtxt(self.filename_float, dtype=np.float)
        self.A_complex = np.genfromtxt(self.filename_complex, dtype=np.complex)

    def tearDown(self):
        pass

    def test_read_matrix_dense(self):
        A = matrix.read_matrix_dense(self.filename_int, dtype=np.int)
        self.assertTrue(np.all(A == self.A_int))

        A = matrix.read_matrix_dense(self.filename_float)
        self.assertTrue(np.all(A == self.A_float))

        A = matrix.read_matrix_dense(self.filename_complex, dtype=np.complex)
        self.assertTrue(np.all(A == self.A_complex))


class TestWriteMatrixDense(unittest.TestCase):
    def setUp(self):
        self.filename_int = testpath + 'matrix_int_out.dat'
        self.filename_float = testpath + 'matrix_float_out.dat'
        self.filename_complex = testpath + 'matrix_complex_out.dat'

        self.A_int = np.arange(3 * 3).reshape(3, 3)
        self.A_float = 1.0 * self.A_int
        self.A_complex = np.arange(3 * 3).reshape(3, 3) + 1j * np.arange(9, 3 * 3 + 9).reshape(3, 3)

    def tearDown(self):
        os.remove(self.filename_int)
        os.remove(self.filename_float)
        os.remove(self.filename_complex)

    def test_write_matrix_dense(self):
        matrix.write_matrix_dense(self.filename_int, self.A_int, fmt='%d')
        An = np.loadtxt(self.filename_int, dtype=np.int)
        self.assertTrue(np.all(An == self.A_int))

        matrix.write_matrix_dense(self.filename_float, self.A_float)
        An = np.loadtxt(self.filename_int)
        self.assertTrue(np.all(An == self.A_float))

        matrix.write_matrix_dense(self.filename_complex, self.A_complex)
        An = np.loadtxt(self.filename_complex, dtype=np.complex)
        self.assertTrue(np.all(An == self.A_complex))


################################################################################
# sparse
################################################################################

class TestReadMatrixSparse(unittest.TestCase):
    def setUp(self):
        self.filename_int = testpath + 'spmatrix_int.coo.dat'
        self.filename_float = testpath + 'spmatrix_float.coo.dat'
        self.filename_complex = testpath + 'spmatrix_complex.coo.dat'

        """Reference matrices in dense storage"""
        self.reference_int = testpath + 'spmatrix_int_reference.dat'
        self.reference_float = testpath + 'spmatrix_float_reference.dat'
        self.reference_complex = testpath + 'spmatrix_complex_reference.dat'

    def tearDown(self):
        pass

    def test_read_matrix_sparse(self):
        A = np.loadtxt(self.reference_int, dtype=np.int)
        A_n = matrix.read_matrix_sparse(self.filename_int, dtype=np.int).toarray()
        self.assertTrue(np.all(A == A_n))

        A = np.loadtxt(self.reference_float)
        A_n = matrix.read_matrix_sparse(self.filename_float).toarray()
        self.assertTrue(np.all(A == A_n))

        A = np.loadtxt(self.reference_complex, dtype=np.complex)
        A_n = matrix.read_matrix_sparse(self.filename_complex, dtype=np.complex).toarray()
        self.assertTrue(np.all(A == A_n))


class TestWriteMatrixSparse(unittest.TestCase):
    def is_integer(self, x):
        """Check if elements of an array can be represented by integers.

        Parameters
        ----------
        x : ndarray
            Array to check.

        Returns
        -------
        is_int : ndarray of bool
            is_int[i] is True if x[i] can be represented
            as int otherwise is_int[i] is False.

        """
        is_int = np.equal(np.mod(x, 1), 0)
        return is_int

    def sparse_matrix_from_coo(self, coo):
        row = coo[:, 0]
        col = coo[:, 1]
        values = coo[:, 2]

        """Check if imaginary part of row and col is zero"""
        if np.all(np.isreal(row)) and np.all(np.isreal(col)):
            row = row.real
            col = col.real

            """Check if first and second column contain only integer entries"""
            if np.all(self.is_integer(row)) and np.all(self.is_integer(col)):

                """Convert row and col to int"""
                row = row.astype(int)
                col = col.astype(int)

                """Create coo-matrix"""
                A = scipy.sparse.coo_matrix((values, (row, col)))
                return A
            else:
                raise ValueError('coo contains non-integer entries for row and col.')
        else:
            raise ValueError('coo contains complex entries for row and col.')

    def setUp(self):
        self.filename_int = testpath + 'spmatrix_int_out.coo.dat'
        self.filename_float = testpath + 'spmatrix_float_out.coo.dat'
        self.filename_complex = testpath + 'spmatrix_complex_out.coo.dat'

        """Tri-diagonal test matrices"""
        dim = 10
        d0 = np.arange(0, dim)
        d1 = np.arange(dim, 2 * dim - 1)
        d_1 = np.arange(2 * dim, 3 * dim - 1)

        self.A_int = scipy.sparse.diags((d0, d1, d_1), (0, 1, -1), dtype=np.int).tocoo()
        self.A_float = scipy.sparse.diags((d0, d1, d_1), (0, 1, -1)).tocoo()
        self.A_complex = self.A_float + 1j * self.A_float

    def tearDown(self):
        os.remove(self.filename_int)
        os.remove(self.filename_float)
        os.remove(self.filename_complex)

    def test_write_matrix_sparse(self):
        matrix.write_matrix_sparse(self.filename_int, self.A_int, fmt='%d')
        coo_n = np.loadtxt(self.filename_int, dtype=np.int)
        """Create sparse matrix from coo data"""
        A_n = self.sparse_matrix_from_coo(coo_n)
        diff = (self.A_int - A_n).tocsr()
        """Check for empty array of non-zero entries"""
        self.assertTrue(np.all(diff.data == 0.0))

        matrix.write_matrix_sparse(self.filename_float, self.A_float)
        coo_n = np.loadtxt(self.filename_float, dtype=np.float)
        """Create sparse matrix from coo data"""
        A_n = self.sparse_matrix_from_coo(coo_n)
        diff = (self.A_float - A_n).tocsr()
        """Check for empty array of non-zero entries"""
        self.assertTrue(np.all(diff.data == 0.0))

        matrix.write_matrix_sparse(self.filename_complex, self.A_complex)
        coo_n = np.loadtxt(self.filename_complex, dtype=np.complex)
        """Create sparse matrix from coo data"""
        A_n = self.sparse_matrix_from_coo(coo_n)
        diff = (self.A_complex - A_n).tocsr()
        """Check for empty array of non-zero entries"""
        self.assertTrue(np.all(diff.data == 0.0))


################################################################################
# binary
################################################################################

################################################################################
# dense
################################################################################

class TestLoadMatrixDense(unittest.TestCase):
    def setUp(self):
        self.filename_int = testpath + 'matrix_int.npy'
        self.filename_float = testpath + 'matrix_float.npy'
        self.filename_complex = testpath + 'matrix_complex.npy'

        self.A_int = np.load(self.filename_int)
        self.A_float = np.load(self.filename_float)
        self.A_complex = np.load(self.filename_complex)

    def tearDown(self):
        pass

    def test_load_matrix_dense(self):
        A = matrix.load_matrix_dense(self.filename_int)
        self.assertTrue(np.all(A == self.A_int))

        A = matrix.load_matrix_dense(self.filename_float)
        self.assertTrue(np.all(A == self.A_float))

        A = matrix.load_matrix_dense(self.filename_complex)
        self.assertTrue(np.all(A == self.A_complex))


class TestSaveMatrixDense(unittest.TestCase):
    def setUp(self):
        self.filename_int = testpath + 'matrix_int_out.npy'
        self.filename_float = testpath + 'matrix_float_out.npy'
        self.filename_complex = testpath + 'matrix_complex_out.npy'

        self.A_int = np.arange(3 * 3).reshape(3, 3)
        self.A_float = 1.0 * self.A_int
        self.A_complex = np.arange(3 * 3).reshape(3, 3) + 1j * np.arange(9, 3 * 3 + 9).reshape(3, 3)

    def tearDown(self):
        os.remove(self.filename_int)
        os.remove(self.filename_float)
        os.remove(self.filename_complex)

    def test_write_matrix_dense(self):
        matrix.save_matrix_dense(self.filename_int, self.A_int)
        An = np.load(self.filename_int)
        self.assertTrue(np.all(An == self.A_int))

        matrix.save_matrix_dense(self.filename_float, self.A_float)
        An = np.load(self.filename_int)
        self.assertTrue(np.all(An == self.A_float))

        matrix.save_matrix_dense(self.filename_complex, self.A_complex)
        An = np.load(self.filename_complex)
        self.assertTrue(np.all(An == self.A_complex))


################################################################################
# sparse
################################################################################

class TestLoadMatrixSparse(unittest.TestCase):
    def setUp(self):
        self.filename_int = testpath + 'spmatrix_int.coo.npy'
        self.filename_float = testpath + 'spmatrix_float.coo.npy'
        self.filename_complex = testpath + 'spmatrix_complex.coo.npy'

        """Reference matrices in dense storage"""
        self.reference_int = testpath + 'spmatrix_int_reference.dat'
        self.reference_float = testpath + 'spmatrix_float_reference.dat'
        self.reference_complex = testpath + 'spmatrix_complex_reference.dat'

    def tearDown(self):
        pass

    def test_load_matrix_sparse(self):
        A = np.loadtxt(self.reference_int, dtype=np.int)
        A_n = matrix.load_matrix_sparse(self.filename_int).toarray()
        self.assertTrue(np.all(A == A_n))

        A = np.loadtxt(self.reference_float)
        A_n = matrix.load_matrix_sparse(self.filename_float).toarray()
        self.assertTrue(np.all(A == A_n))

        A = np.loadtxt(self.reference_complex, dtype=np.complex)
        A_n = matrix.load_matrix_sparse(self.filename_complex).toarray()
        self.assertTrue(np.all(A == A_n))


class TestSaveMatrixSparse(unittest.TestCase):
    def is_integer(self, x):
        """Check if elements of an array can be represented by integers.

        Parameters
        ----------
        x : ndarray
            Array to check.

        Returns
        -------
        is_int : ndarray of bool
            is_int[i] is True if x[i] can be represented
            as int otherwise is_int[i] is False.

        """
        is_int = np.equal(np.mod(x, 1), 0)
        return is_int

    def sparse_matrix_from_coo(self, coo):
        row = coo[:, 0]
        col = coo[:, 1]
        values = coo[:, 2]

        """Check if imaginary part of row and col is zero"""
        if np.all(np.isreal(row)) and np.all(np.isreal(col)):
            row = row.real
            col = col.real

            """Check if first and second column contain only integer entries"""
            if np.all(self.is_integer(row)) and np.all(self.is_integer(col)):

                """Convert row and col to int"""
                row = row.astype(int)
                col = col.astype(int)

                """Create coo-matrix"""
                A = scipy.sparse.coo_matrix((values, (row, col)))
                return A
            else:
                raise ValueError('coo contains non-integer entries for row and col.')
        else:
            raise ValueError('coo contains complex entries for row and col.')

    def setUp(self):
        self.filename_int = testpath + 'spmatrix_int_out.coo.npy'
        self.filename_float = testpath + 'spmatrix_float_out.coo.npy'
        self.filename_complex = testpath + 'spmatrix_complex_out.coo.npy'

        """Tri-diagonal test matrices"""
        dim = 10
        d0 = np.arange(0, dim)
        d1 = np.arange(dim, 2 * dim - 1)
        d_1 = np.arange(2 * dim, 3 * dim - 1)

        self.A_int = scipy.sparse.diags((d0, d1, d_1), (0, 1, -1), dtype=np.int).tocoo()
        self.A_float = scipy.sparse.diags((d0, d1, d_1), (0, 1, -1)).tocoo()
        self.A_complex = self.A_float + 1j * self.A_float

    def tearDown(self):
        os.remove(self.filename_int)
        os.remove(self.filename_float)
        os.remove(self.filename_complex)

    def test_save_matrix_sparse(self):
        matrix.save_matrix_sparse(self.filename_int, self.A_int)
        coo_n = np.load(self.filename_int)
        """Create sparse matrix from coo data"""
        A_n = self.sparse_matrix_from_coo(coo_n)
        diff = (self.A_int - A_n).tocsr()
        """Check for empty array of non-zero entries"""
        self.assertTrue(np.all(diff.data == 0.0))

        matrix.save_matrix_sparse(self.filename_float, self.A_float)
        coo_n = np.load(self.filename_float)
        """Create sparse matrix from coo data"""
        A_n = self.sparse_matrix_from_coo(coo_n)
        diff = (self.A_float - A_n).tocsr()
        """Check for empty array of non-zero entries"""
        self.assertTrue(np.all(diff.data == 0.0))

        matrix.save_matrix_sparse(self.filename_complex, self.A_complex)
        coo_n = np.load(self.filename_complex)
        """Create sparse matrix from coo data"""
        A_n = self.sparse_matrix_from_coo(coo_n)
        diff = (self.A_complex - A_n).tocsr()
        """Check for empty array of non-zero entries"""
        self.assertTrue(np.all(diff.data == 0.0))


if __name__ == "__main__":
    unittest.main()
