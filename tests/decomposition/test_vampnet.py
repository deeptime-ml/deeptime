import numpy as np
import pytest

import sktime
from sktime.decomposition.vampnet import sym_inverse, covariances, score_vamp2, loss_vamp2

try:
    import torch
except (ImportError, ModuleNotFoundError):
    pytest.skip("Skipping tests which depend on PyTorch because it is not installed in the environment.",
                allow_module_level=True)


def test_inverse_spd():
    X = np.random.normal(size=(15, 5))
    spd = X @ X.T  # rank at most 5
    spd_inv_qr = sktime.numeric.spd_inv(spd)
    with torch.no_grad():
        spd_tensor = torch.from_numpy(spd)
        spd_inv = sym_inverse(spd_tensor, epsilon=1e-6, ret_sqrt=False)
        np.testing.assert_array_almost_equal(spd_inv.numpy(), spd_inv_qr)


@pytest.mark.parametrize("remove_mean", [True, False], ids=lambda x: f"remove_mean={x}")
def test_covariances(remove_mean):
    data = sktime.data.ellipsoids().observations(1000, n_dim=5)
    tau = 10
    data_instantaneous = data[:-tau].astype(np.float64)
    data_shifted = data[tau:].astype(np.float64)
    cov_est = sktime.covariance.Covariance(lagtime=tau, compute_c0t=True, compute_ctt=True,
                                           remove_data_mean=remove_mean)
    reference_covs = cov_est.fit(data).fetch_model()
    with torch.no_grad():
        c00, c0t, ctt = covariances(torch.from_numpy(data_instantaneous), torch.from_numpy(data_shifted),
                                    remove_mean=remove_mean)
        np.testing.assert_array_almost_equal(reference_covs.cov_00, c00.numpy())
        np.testing.assert_array_almost_equal(reference_covs.cov_0t, c0t.numpy())
        np.testing.assert_array_almost_equal(reference_covs.cov_tt, ctt.numpy())


def test_vamp2():
    data = sktime.data.ellipsoids().observations(1000, n_dim=5)
    tau = 10

    vamp_model = sktime.decomposition.VAMP(lagtime=tau, dim=None).fit(data).fetch_model()
    with torch.no_grad():
        data_instantaneous = torch.from_numpy(data[:-tau].astype(np.float64))
        data_shifted = torch.from_numpy(data[tau:].astype(np.float64))
        score = score_vamp2(data_instantaneous, data_shifted)
        np.testing.assert_array_almost_equal(score.numpy(), vamp_model.score(score_method='VAMP2'))
