"""qEI Acquisition Function."""
import torch
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
)
from botorch.optim.optimize import optimize_acqf_discrete
from botorch.sampling.normal import SobolQMCNormalSampler
from torch import Tensor

from ..input_parameters import AcquisitionFunctionParameters

SOBOL_NUM_SAMPLES = 128

def discrete_qEI(params: AcquisitionFunctionParameters) -> Tensor:
    """Return the candidate points using qEI acquisition function.

    Args:
        params (AcquisitionFunction): Input Parameters.

    Returns:
        Tensor: selected candidate points.
    """
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([SOBOL_NUM_SAMPLES]))
    acq_func = qExpectedImprovement(
        params.model, sampler=sampler, best_f=params.Y_train.max()
    )
    candidates, _ = optimize_acqf_discrete(
        acq_function=acq_func, choices=params.X_test, q=params.cand_size, unique=True
    )
    return torch.tensor(candidates)


