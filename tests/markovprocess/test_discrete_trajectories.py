import unittest
import numpy as np

import sktime.markovprocess._markovprocess_bindings as bindings

class TestDiscreteTrajectoriesUtils(unittest.TestCase):

    def test_foo(self):
        dtrajs = [np.arange(10), np.arange(15), np.arange(5)]
        out = bindings.sample.index_states(dtrajs, None)
        print(out)


if __name__ == '__main__':
    unittest.main()
