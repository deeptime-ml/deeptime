"""
Created on May 26, 2014

@author: marscher
"""


class SpectralWarning(RuntimeWarning):
    pass


class ImaginaryEigenValueWarning(SpectralWarning):
    pass


class PrecisionWarning(RuntimeWarning):
    r"""
    This warning indicates that some operation in your code leads
    to a conversion of data-types, which involves a loss/gain in
    precision.
    """
    pass


class NotConvergedWarning(RuntimeWarning):
    r"""
    This warning indicates that some iterative procedure has not
    converged or reached the maximum number of iterations implemented
    as a safe guard to prevent arbitrary many iterations in loops with
    a conditional termination criterion.
    """


class NotConvergedError(RuntimeError):
    pass


