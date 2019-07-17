import numpy as np


def ensure_ndarray(arr, shape: tuple = None, ndim: int = None, dtype=None, size=None) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr, dtype=dtype)
    if shape is not None and arr.shape != shape:
        raise ValueError(f"Shape of provided array was {arr.shape} != {shape}")
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"ndim of provided array was {arr.ndim} != {ndim}")
    if size is not None and np.size(arr) != size:
        raise ValueError(f"size of provided array was {np.size(arr)} != {size}")
    if dtype is not None and arr.dtype != dtype:
        arr = arr.astype(dtype)
    return arr
