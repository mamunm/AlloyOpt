"""qNEHVI Acquisition Function."""

import torch
from botorch.acquisition.multi_objective import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.optim.optimize import (
    optimize_acqf_discrete,
)
from botorch.sampling.normal import SobolQMCNormalSampler
from torch import Tensor

from ..input_parameters import AcquisitionFunctionParameters

SOBOL_NUM_SAMPLES = 128

def discrete_qNEHVI(params: AcquisitionFunctionParameters) -> Tensor:
    """Return the candidate points using qNEHVI acquisition function.

    Args:
        params (AcquisitionFunctionParameters): Input Parameters.

    Returns:
        Tensor: selected candidate points.
    """
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([SOBOL_NUM_SAMPLES]))
    acq_func = qNoisyExpectedHypervolumeImprovement(
        model=params.model,
        ref_point=params.ref_points,
        X_baseline=params.X_train,
        prune_baseline=True,
        sampler=sampler,
    )
    candidates, _ = optimize_acqf_discrete(
        acq_function=acq_func, choices=params.X_test, q=params.cand_size, unique=True
    )
    return torch.tensor(candidates)