"""Input parameters for alloy_opt package."""

import os
from pathlib import Path
from typing import List, Literal, NamedTuple, Optional

from botorch.models import ModelListGP
from torch import Tensor


class AcquisitionFunctionParameters(NamedTuple):
    """Inpute parameters for the multi-objective acquisition function.

    Args:
        model (ModelListGP): A trained model.
        Y_train (Tensor): Observed targets.
        cand_size (int): Number of candidate points.
        ref_points (Optional[Tensor]): Reference points.
        X_test (Optional[Tensor]): Test points.
        bounds (Optional[Tensor]): Bounds of the cont. parameter space.
        X_train (Optional[Tensor]): Training points.
    """

    model: ModelListGP
    Y_train: Tensor
    cand_size: int
    ref_points: Optional[Tensor] = None
    X_test: Optional[Tensor] = None
    bounds: Optional[Tensor] = None
    X_train: Optional[Tensor] = None

class BayesianOptimizationParameters(NamedTuple):
    """Input parameters for Bayesian optimization.

    Args:
        model (Literal["saasgp", "torchgp", "skgp"]): Name of the model.
        acq_func (Literal["qPI", "qEI", "qUCB", "qEHVI", "qNEHVI", "qparEGO",
            "qNparEGO"]): Name of the acquisition function.
        csv_file_loc (List[str]): Path to the CSV file
        features (List[str]): Names of the features.
        targets (List[str]): Names of the targets.
        target_masks (List[bool]): Mask of the targets.
        seed_points (int): Number of initial random data.
        cwd (Union[str, Path]): Path to the working directory.
        cur_iter (int): Current iteration.
        n_iterations (int): Number of iterations.
        n_candidates (int): Number of candidates to optimize in each iteration.
        experiment_name (str): Name of the experiment.
        device (Literal["cpu", "cuda"]): Device on which to run the optimization.
    """

    model: Literal["saasgp", "torchgp", "skgp"]
    acq_func: Literal["qPI", "qEI", "qUCB", "qEHVI", "qNEHVI", "qparEGO", "qNparEGO"]
    csv_file_loc: str
    features: List[str]
    targets: List[str]
    target_masks: List[bool]
    seed_points: int
    cwd: Path = Path(os.getcwd()).resolve()
    cur_iter: int = 0
    n_iterations: int = 10
    n_candidates: int = 1
    experiment_name: str = "Experiment_1"
    device: Literal["cpu", "cuda"] = "cpu"
    
    def to_dict(self):
        """Convert parameters to a dictionary."""
        return {
            "model": self.model,
            "acq_func": self.acq_func,
            "csv_file_loc": self.csv_file_loc,
            "features": self.features,
            "targets": self.targets,
            "target_masks": self.target_masks,
            "seed_points": self.seed_points,
            "cwd": str(self.cwd),
            "cur_iter": self.cur_iter,
            "n_iterations": self.n_iterations,
            "n_candidates": self.n_candidates,
            "experiment_name": self.experiment_name,
            "device": self.device,
        }