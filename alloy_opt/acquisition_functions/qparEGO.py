"""qparEGO Acquisition Function."""

import torch
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
)
from botorch.acquisition.objective import GenericMCObjective
from botorch.optim.optimize import (
    optimize_acqf_discrete,
)
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import sample_simplex
from torch import Tensor

from ..input_parameters import AcquisitionFunctionParameters

SOBOL_NUM_SAMPLES = 128

def discrete_qparEGO(params: AcquisitionFunctionParameters) -> Tensor:
    """Return the candidate points using qparEGO acquisition function.

    Args:
        params (AcquisitionFunctionParameters): Input Parameters.

    Returns:
        Tensor: selected candidate points.
    """
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([SOBOL_NUM_SAMPLES]))
    dtype = params.Y_train.dtype
    device = params.Y_train.device
    num_objective = params.Y_train.shape[1]

    weights = sample_simplex(num_objective, dtype=dtype, device=device).squeeze()
    objective = GenericMCObjective(
        get_chebyshev_scalarization(weights=weights, Y=params.Y_train)
    )
    acq_func = qExpectedImprovement(
        model=params.model,
        objective=objective,
        best_f=params.Y_train.max(),
        sampler=sampler,
    )

    candidates, _ = optimize_acqf_discrete(
        acq_function=acq_func, choices=params.X_test, q=1
    )
    return torch.tensor(candidates)
