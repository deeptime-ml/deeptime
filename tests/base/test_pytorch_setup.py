import pytest
from numpy.testing import assert_array_almost_equal

pytest.importorskip("torch")

import numpy as np
import torch

eigh = torch.linalg.eigh if hasattr(torch, 'linalg') else lambda x: torch.symeig(x, eigenvectors=True)
multi_dot = torch.linalg.multi_dot if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'multi_dot') else \
    lambda args: torch.chain_matmul(*args)


def symeig_reg(mat, epsilon: float = 1e-6, mode='regularize', eigenvectors=True):
    if mode == 'regularize':
        identity = torch.eye(mat.shape[0], dtype=mat.dtype, device=mat.device)
        mat = mat + epsilon * identity

    # Calculate eigvalues and potentially eigvectors
    eigval, eigvec = eigh(mat)

    if eigenvectors:
        eigvec = eigvec.transpose(0, 1)

    if mode == 'trunc':
        # Filter out Eigenvalues below threshold and corresponding Eigenvectors
        mask = eigval > epsilon
        eigval = eigval[mask]
        if eigenvectors:
            eigvec = eigvec[mask]
    elif mode == 'regularize':
        # Calculate eigvalues and eigvectors
        eigval = torch.abs(eigval)
    elif mode == 'clamp':
        eigval = torch.clamp_min(eigval, min=epsilon)

    return eigval, eigvec


def sym_inverse(mat, epsilon: float = 1e-6, return_sqrt=False, mode='regularize'):
    eigval, eigvec = symeig_reg(mat, epsilon, mode)

    # Build the diagonal matrix with the filtered eigenvalues or square
    # root of the filtered eigenvalues according to the parameter
    if return_sqrt:
        diag = torch.diag(torch.sqrt(1. / eigval))
    else:
        diag = torch.diag(1. / eigval)

    return multi_dot([eigvec.t(), diag, eigvec])


def whiten(data, epsilon=1e-6, mode='regularize'):
    data_meanfree = data - data.mean(dim=0, keepdim=True)
    cov = 1 / (data.shape[0] - 1) * (data_meanfree.t() @ data_meanfree)
    cov_sqrt_inv = sym_inverse(cov, epsilon=epsilon, mode=mode, return_sqrt=True)
    return data_meanfree @ cov_sqrt_inv


def test_pytorch_installation():
    r""" Fast-failing test that can be used to predetermine faulty pytorch installations. Can save some CI time. """
    X = torch.from_numpy(np.random.normal(size=(20, 10)))
    Y = torch.from_numpy(np.random.normal(size=(10, 20)))
    M = X @ Y
    with torch.no_grad():
        assert_array_almost_equal(X.numpy() @ Y.numpy(), M.numpy())


def test_tmp():
    chi_X = np.random.uniform(-1, 1, size=(1000, 50))
    with torch.no_grad():
        xw = whiten(torch.from_numpy(chi_X), mode='clamp', epsilon=1e-10).numpy()
    np.testing.assert_array_almost_equal(xw.mean(axis=0), np.zeros((xw.shape[1],)))
    cov = 1 / (xw.shape[0] - 1) * xw.T @ xw
    np.testing.assert_array_almost_equal(cov, np.eye(50))
