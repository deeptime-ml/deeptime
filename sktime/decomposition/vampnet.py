import warnings

try:
    import torch
    import torch.nn as nn
except (ModuleNotFoundError, ImportError):
    warnings.warn("Tried importing VampNets; this only works with a PyTorch installation!")


def sym_inverse(mat, epsilon: float = 1e-6, ret_sqrt=False):
    """ Utility function that returns the inverse of a matrix, with the
    option to return the square root of the inverse matrix.

    Parameters
    ----------
    mat: numpy array with shape [m,m]
        Matrix to be inverted.
    epsilon : float
        Cutoff for eigenvalues.
    ret_sqrt: bool, optional, default = False
        if True, the square root of the inverse matrix is returned instead

    Returns
    -------
    x_inv: numpy array with shape [m,m]
        inverse of the original matrix
    """
    # Calculate eigvalues and eigvectors
    eigval_all, eigvec_all_t = torch.symeig(mat, eigenvectors=True)

    # Filter out eigvalues below threshold and corresponding eigvectors
    mask = eigval_all > epsilon

    eigval = eigval_all[mask]
    eigvec = eigvec_all_t.transpose(0, 1)[mask]
    eigvec_t = eigvec.transpose(0, 1)

    # Build the diagonal matrix with the filtered eigenvalues or square
    # root of the filtered eigenvalues according to the parameter

    if ret_sqrt:
        diag = torch.diag(torch.sqrt(1. / eigval))
    else:
        diag = torch.diag(1. / eigval)

    return torch.chain_matmul(eigvec_t, diag, eigvec)


def covariances(x, y, remove_mean: bool = True):
    """Utility function that returns the matrices used to compute the VAMP
    scores and their gradients for non-reversible problems.
    Parameters
    ----------
    x: tensorflow tensor with shape [batch_size, output_dim]
        output of the left lobe of the network

    y: tensorflow tensor with shape [batch_size, output_dim]
        output of the right lobe of the network

    remove_mean: bool, default=True
        Whether to first remove the mean of x and y

    Returns
    -------
    cov_00_inv_root: numpy array with shape [output_size, output_size]
        square root of the inverse of the auto-covariance matrix of x

    cov_11_inv_root: numpy array with shape [output_size, output_size]
        square root of the inverse of the auto-covariance matrix of y

    cov_01: numpy array with shape [output_size, output_size]
        cross-covariance matrix of x and y

    """

    assert x.shape == y.shape, "x and y must be of same shape"
    batch_size = x.shape[0]

    if remove_mean:
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)

    # Calculate the cross-covariance
    y_t = y.transpose(0, 1)
    x_t = x.transpose(0, 1)
    cov_01 = 1 / (batch_size - 1) * torch.matmul(x_t, y)
    # Calculate the auto-correations
    cov_00 = 1 / (batch_size - 1) * torch.matmul(x_t, x)
    cov_11 = 1 / (batch_size - 1) * torch.matmul(y_t, y)

    return cov_00, cov_01, cov_11


def score_vamp2(data_instantaneous: torch.Tensor, data_shifted: torch.Tensor):
    assert data_instantaneous.shape == data_shifted.shape

    cov_00, cov_0t, cov_tt = covariances(data_instantaneous, data_shifted, remove_mean=True)

    c00_inv_sqrt = sym_inverse(cov_00, ret_sqrt=True)
    ctt_inv_sqrt = sym_inverse(cov_tt, ret_sqrt=True)

    # Calculate the inverse of the self-covariance matrices
    vamp_matrix = torch.chain_matmul(c00_inv_sqrt, cov_0t, ctt_inv_sqrt)
    vamp_score = torch.norm(vamp_matrix, p='fro')
    return 1 + torch.square(vamp_score)


def loss_vamp2(data_instantaneous: torch.Tensor, data_shifted: torch.Tensor):
    return -1. * score_vamp2(data_instantaneous, data_shifted)
