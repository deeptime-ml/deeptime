import numpy as np

from sktime.covariance.online_covariance import OnlineCovariance


if __name__ == '__main__':
    data = np.random.normal(size=(500000, 10))

    ################################################################################################
    # compute covariance matrix C00
    ################################################################################################

    # configure estimator with estimator-global parameters
    estimator = OnlineCovariance(compute_c00=True, remove_data_mean=True)

    for batch in np.array_split(data, 100):
        # during fit or partial fit parameters can be entered that are relevant for that batch only
        estimator.partial_fit(batch, weights=None, column_selection=None)

    # this finalizes the partial estimation (ie extracts means & covariance matrices from running covar)
    # and returns the current model
    model = estimator.model
    print(model.mean_0)

    # retrieves copy of current model
    model_copy = estimator.copy_current_model()
    assert np.all(model_copy.mean_0 == model.mean_0) and model_copy is not model

    ################################################################################################
    # compute covariance matrix C0t
    ################################################################################################

    tau = 10

    # configure estimator with estimator-global parameters
    estimator = OnlineCovariance(compute_c00=False, compute_c0t=True, remove_data_mean=True)

    # do one-shot estimation by giving tuple of instantaneous and time-shifted data (also possible with partial fit)
    estimator.fit((data[:-tau], data[tau:]))

    # finalize and retrieve model
    model = estimator.model

    print(model.cov_0t)

    ################################################################################################
    # outlook for transformers
    ################################################################################################

    # calls transform method on model

    # estimator.transform(data)

    # ... and is equivalent to

    # estimator.model.transform(data)

    # meaning that, e.g., different stages of estimation can be saved and used for data transformation
