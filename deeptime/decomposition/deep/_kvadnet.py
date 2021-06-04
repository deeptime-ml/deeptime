import torch
from . import _vampnet as vnet
from ...kernels import Kernel, GaussianKernel, is_torch_kernel


def whiten(data, epsilon=1e-6, mode='clamp'):
    data_meanfree = data - data.mean(dim=0, keepdim=True)
    cov = 1 / (data.shape[0] - 1) * (data_meanfree.t() @ data_meanfree)
    cov_sqrt_inv = vnet.sym_inverse(cov, epsilon=epsilon, mode=mode, return_sqrt=True)
    return data_meanfree @ cov_sqrt_inv


def gramian(y, kernel):
    with torch.no_grad():
        if is_torch_kernel(kernel):
            gramian = kernel.gram(y)
        else:
            gramian_np = kernel.gram(y.cpu().numpy())
            gramian = torch.as_tensor(gramian_np, dtype=y.dtype, device=y.device)
    return gramian


def kvad_score(chi_x, y, kernel: Kernel = GaussianKernel(1.), epsilon=1e-6, mode='regularize'):
    G = gramian(y, kernel)

    N = y.shape[0]
    chi_x_whitened = whiten(chi_x, epsilon=epsilon, mode=mode)
    x_g_x = torch.chain_matmul(chi_x_whitened.t(), G, chi_x_whitened)

    evals_sum = torch.trace(x_g_x / (N * N))

    return evals_sum + torch.mean(G)
