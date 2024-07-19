"""sklearn GP surrogate model."""

from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from botorch.models.gpytorch import Model
from botorch.posteriors import Posterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from numpy.typing import NDArray
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import DotProduct, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
from torch import Tensor


class SklearnGPModel(Model):  # type: ignore
    """Sklearn GP model."""

    def __init__(self, X: Tensor, Y: Tensor) -> None:
        self.Y = Y
        kernel = (
            C(1.0) * Matern(length_scale=1.0)
            + WhiteKernel(noise_level=1.0)
            + C(1.0) * DotProduct(sigma_0=1.0)
        )
        self.gp = [
            GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=8, normalize_y=True
            )
            for _ in range(Y.shape[-1])
        ]
        self.x_scaler = StandardScaler()
        X_transformed = self.x_scaler.fit_transform(X.detach().numpy())
        for i, gp in enumerate(self.gp):
            gp.fit(X_transformed, Y[:, i])

    @property
    def num_outputs(self) -> int:
        """Return the number of outputs."""
        return self.Y.shape[-1]

    def predict(
        self, X: Tensor
    ) -> Tuple[List[NDArray[np.float64]], List[NDArray[np.float64]]]:
        """Get the predictions.

        Args:
            X (Tensor): Inputs.

        Returns:
            Tensor: Predictions.
        """
        X_transformed = self.x_scaler.transform(X.detach().numpy())
        mu_out, cov_out = [], []
        for gp in self.gp:
            mu, cov = gp.predict(X_transformed, return_cov=True)
            mu_out.append(mu)
            cov_out.append(cov)
        return mu_out, cov_out

    def posterior(
        self,
        X: Tensor,
        posterior_transform: Optional[Callable[[Posterior], Posterior]] = None,
    ) -> GPyTorchPosterior:
        """Get the posterior.

        Args:
            X (Tensor): Inputs.
            posterior_transform (Callable[[Posterior], Posterior], optional):
                Posterior transform.

        Returns:
            GPyTorchPosterior: Posterior.
        """
        if len(X.shape) == 3:
            batch_size = X.shape[0]
            n_size = X.shape[1]
            mu_batch, cov_batch = [], []
            for i in range(batch_size):
                temp_mu, temp_cov = self.predict(X[i])
                mu_batch.append(temp_mu)
                cov_batch.append(temp_cov)
            mu = [
                torch.tensor(
                    np.concatenate([m[i] for m in mu_batch]), dtype=X.dtype
                ).reshape(batch_size, n_size)
                for i in range(self.Y.shape[-1])
            ]
            cov = [
                torch.tensor(
                    np.concatenate([c[i] for c in cov_batch]), dtype=X.dtype
                ).reshape(batch_size, n_size, n_size)
                for i in range(self.Y.shape[-1])
            ]
        else:
            temp_mu, temp_cov = self.predict(X)
            mu = [torch.tensor(mu, dtype=X.dtype) for mu in temp_mu]
            cov = [torch.tensor(cov, dtype=X.dtype) for cov in temp_cov]

        if self.Y.shape[-1] == 1:
            mvn = MultivariateNormal(mu[0], cov[0])
        else:
            mvns = [MultivariateNormal(mu[i], cov[i]) for i in range(self.Y.shape[-1])]
            mvn = MultitaskMultivariateNormal.from_independent_mvns(mvns=mvns)

        return GPyTorchPosterior(mvn)


def get_sklearn_gp_model(X: Tensor, Y: Tensor) -> Model:
    """Get the sklearn GP model.

    Args:
        X (Tensor): Inputs.
        Y (Tensor): Targets.

    Returns:
        SklearnGPModel: Model.
    """
    model = SklearnGPModel(X, Y)
    return model
