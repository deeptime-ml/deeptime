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


def koopman_matrix(x, y):
    c00, c0t, ctt = covariances(x, y, remove_mean=True)
    c00_sqrt_inv = sym_inverse(c00, ret_sqrt=True)
    ctt_sqrt_inv = sym_inverse(ctt, ret_sqrt=True)
    return torch.chain_matmul(c00_sqrt_inv, c0t, ctt_sqrt_inv).t()


def covariances(x: torch.Tensor, y: torch.Tensor, remove_mean: bool = True):
    """Computes instantaneous and time-lagged covariances matrices.

    Parameters
    ----------
    x : (T, n) torch.Tensor
        Instantaneous data.
    y : (T, n) torch.Tensor
        Time-lagged data.
    remove_mean: bool, default=True
        Whether to remove the mean of x and y.

    Returns
    -------
    cov_00 : (n, n) torch.Tensor
        Auto-covariance matrix of x.
    cov_0t : (n, n) torch.Tensor
        Cross-covariance matrix of x and y.
    cov_tt : (n, n) torch.Tensor
        Auto-covariance matrix of y.

    See Also
    --------
    sktime.covariance.Covariance : Estimator yielding these kind of covariance matrices based on raw data.
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
    # Calculate the auto-correlations
    cov_00 = 1 / (batch_size - 1) * torch.matmul(x_t, x)
    cov_11 = 1 / (batch_size - 1) * torch.matmul(y_t, y)

    return cov_00, cov_01, cov_11


def score(data_instantaneous: torch.Tensor, data_shifted: torch.Tensor, method='VAMP2'):
    if method not in score.valid_methods:
        raise ValueError(f"Invalid method '{method}', supported are {score.valid_methods}")
    assert data_instantaneous.shape == data_shifted.shape

    koopman = koopman_matrix(data_instantaneous, data_shifted)
    vamp_score = torch.norm(koopman, p='fro')
    return 1 + torch.square(vamp_score)


score.valid_methods = ('VAMP2',)


def loss_vamp2(data_instantaneous: torch.Tensor, data_shifted: torch.Tensor):
    return -1. * score(data_instantaneous, data_shifted)
