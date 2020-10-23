import os
from typing import Optional


def handle_n_jobs(value: Optional[int]) -> int:
    r"""Handles the n_jobs parameter consistently so that a non-negative number is returned.
    In particular, if

      * value is None, use 1 job
      * value is negative, use number cores available * 2
      * value is positive, use value

    Parameters
    ----------
    value : int or None
        The provided n_jobs argument

    Returns
    -------
    n_jobs : int
        A non-negative integer value describing how many threads can be started simultaneously.
    """
    if value is None:
        return 1
    elif value < 0:
        count = os.cpu_count()
        if count is None:
            raise ValueError("Could not determine number of cpus in system, please provide n_jobs manually.")
        return os.cpu_count() * 2
    else:
        return value
