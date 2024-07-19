"""Monte-Carlo surrogate models."""

import torch
from botorch.fit import fit_fully_bayesian_model_nuts
from botorch.models import ModelListGP, SaasFullyBayesianSingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from torch import Tensor


def get_single_task_model(
    X: Tensor, Y: Tensor, noise_level: float = 1e-6
) -> ModelListGP:
    """Return a trained SaasFullyBayesianSingleTaskGP model.

    Args:
        X (Tensor): Observed features.
        Y (Tensor): Observed targets.
        noise_level (float): Noise level.

    Returns : A trained SaasFullyBayesianSingleTaskGP model.
    """
    assert (
        X.shape[0] == Y.shape[0]
    ), "Number of observations must match for features and targets."
    if Y.ndim == 1:
        Y = Y.unsqueeze(-1)
    Yvar = torch.full_like(Y, noise_level)
    models = []
    for i in range(Y.shape[-1]):
        gp = SaasFullyBayesianSingleTaskGP(
            X,
            Y[:, i].unsqueeze(-1),
            Yvar[:, i].unsqueeze(-1),
            input_transform=Normalize(d=X.shape[-1]),
            outcome_transform=Standardize(m=1),
        )
        fit_fully_bayesian_model_nuts(
            gp, warmup_steps=512, num_samples=256, thinning=16, disable_progbar=True
        )
        models.append(gp)
    model = ModelListGP(*models)
    return model