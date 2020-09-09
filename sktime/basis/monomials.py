from ._basis_bindings import evaluate_monomials as _eval
from .base import Observable


class Monomials(Observable):

    def __init__(self, p):
        self.p = p

    def _evaluate(self, x):
        return _eval(self.p, x)
