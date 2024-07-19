"""Analytical surrogate models."""

from typing import List, Optional

import gpytorch
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import MixedSingleTaskGP, ModelListGP, SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from torch import Tensor


def get_single_task_model(
    X: Tensor, Y: Tensor, noise_level: float = 1e-6
) -> ModelListGP:
    """Return a trained SingleTaskGP model.

    Args:
        X (Tensor): Observed features.
        Y (Tensor): Observed targets.
        noise_level (float): Noise level.

    Returns : A trained SingleTaskGP model.
    """
    assert (
        X.shape[0] == Y.shape[0]
    ), "Number of observations must match for features and targets."
    if Y.ndim == 1:
        Y = Y.unsqueeze(-1)
    Yvar = torch.full_like(Y, noise_level)
    models = []
    for i in range(Y.shape[-1]):
        models.append(
            SingleTaskGP(
                X,
                Y[:, i].unsqueeze(-1),
                Yvar[:, i].unsqueeze(-1),
                input_transform=Normalize(d=X.shape[-1]),
                outcome_transform=Standardize(m=1),
                covar_module=gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.MaternKernel()
                )
                + gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel()),
                mean_module=gpytorch.means.ConstantMean(),
                likelihood=gpytorch.likelihoods.GaussianLikelihood(),
            )
        )
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model
