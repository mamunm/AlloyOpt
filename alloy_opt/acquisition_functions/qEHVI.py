"""qEHVI Acquisition Function."""

import torch
from botorch.acquisition.multi_objective import (
    qExpectedHypervolumeImprovement,
)
from botorch.optim.optimize import (
    optimize_acqf_discrete,
)
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from torch import Tensor

from ..input_parameters import AcquisitionFunctionParameters

SOBOL_NUM_SAMPLES = 128

def discrete_qEHVI(params: AcquisitionFunctionParameters) -> Tensor:
    """Return the candidate points using qEHVI acquisition function.

    Args:
        params (AcquisitionFunctionParameters): Input Parameters.

    Returns:
        Tensor: selected candidate points.
    """
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([SOBOL_NUM_SAMPLES]))
    partitioning = NondominatedPartitioning(
        ref_point=params.ref_points, Y=params.Y_train
    )
    acq_func = qExpectedHypervolumeImprovement(
        model=params.model,
        sampler=sampler,
        ref_point=params.ref_points,
        partitioning=partitioning,
    )
    candidates, _ = optimize_acqf_discrete(
        acq_function=acq_func, choices=params.X_test, q=params.cand_size, unique=True
    )
    return torch.tensor(candidates)
