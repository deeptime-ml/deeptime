from numpy.testing import assert_raises

from deeptime.base import Dataset, Transformer


def test_dataset_interface():
    with assert_raises(TypeError):
        _ = Dataset()


def test_transformer_interface():
    with assert_raises(TypeError):
        _ = Transformer()
