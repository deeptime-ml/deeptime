import unittest
from unittest import TestCase
import numpy as np

from deeptime.util.parallel import handle_n_jobs


class TestParallel(TestCase):
    def test_value_none(self):
        # use 1 cpu if using None as parameter
        value = handle_n_jobs(value=None)
        self.assertEqual(value, 1)

    def test_value_minus1(self):
        # use all cpus if using -1 as parameter
        value = handle_n_jobs(value=-1)
        try:
            from os import sched_getaffinity
            count = len(sched_getaffinity(0))
        except ImportError:
            from os import cpu_count
            count = cpu_count()
        self.assertEqual(value, count)

    def test_value_positive(self):
        # use exact numbers of cpus if using positive number as parameter
        value = handle_n_jobs(value=6)
        self.assertEqual(value, 6)
    
    def test_value_negative(self):
        # raise error if using other negative number as parameter
        self.assertRaisesRegex(ValueError, 
                               f"n_jobs can only be -1 \(in which case it will be determined from hardware\), None \(in which case one will be used\) or a positive number, but was -2.",
                               handle_n_jobs, -2)
    
    def test_value_non_integer(self):
        # raise error if using other positive non-integer as parameter
        self.assertRaisesRegex(ValueError, 
                               f"n_jobs can only be -1 \(in which case it will be determined from hardware\), None \(in which case one will be used\) or a positive number, but was 3.5.",
                               handle_n_jobs, 3.5)

if __name__ == '__main__':
    unittest.main()
