from typing import Union, List

from deeptime.plots.util import default_colors
from deeptime.util import confidence_interval
from deeptime.util.validation import LaggedModelValidation


def plot_ck_test(data: Union[LaggedModelValidation, List[LaggedModelValidation]], conf: float = 0.95, axes=None):
    assert axes is not None  # todo remove

    colors = default_colors()

    if not isinstance(data, list):
        data = [data]

    for lagged_model_validation in data:
        if lagged_model_validation.has_errors:
            l_pred, r_pred = confidence_interval(lagged_model_validation.predictions_samples, conf=conf)
            l_est, r_est = confidence_interval(lagged_model_validation.estimates_samples, conf=conf)
    for ax in axes.flatten():
        pass
