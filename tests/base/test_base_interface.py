from numpy.testing import assert_raises, assert_, assert_equal

from deeptime.base import Dataset, Transformer, Model


def test_dataset_interface():
    with assert_raises(TypeError):
        _ = Dataset()


def test_transformer_interface():
    with assert_raises(TypeError):
        _ = Transformer()


class MockModelVarargs(Model):

    def __init__(self, *args):
        ...


class A(Model):

    def __init__(self, a):
        self.a = a


class MockModel(Model):

    def __init__(self, p1, p2, p3, p4=55, a=A(33), **kw):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.a = a
        self.kw = kw


def test_mock_model():
    with assert_raises(RuntimeError):
        MockModelVarargs().get_params()
    m = MockModel(1., 2., 3.)
    params = m.get_params()
    for i, val in zip([1, 2, 3, 4], [1., 2., 3., 55]):
        assert_(f'p{i}' in params)
        assert_equal(params[f'p{i}'], val)
    assert_equal(params['a'].a, 33)

    m.set_params()  # no-op
    m.set_params(**params)
    for i, val in zip([1, 2, 3, 4], [1., 2., 3., 55]):
        assert_(f'p{i}' in params)
        assert_equal(params[f'p{i}'], val)

    m.set_params(a__a=55)
    assert_equal(m.a.a, 55)

    with assert_raises(ValueError):
        m.set_params(nope=33)
