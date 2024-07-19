"""Bayesian optimization."""

import json
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import torch
from botorch.models import ModelListGP
from botorch.utils.multi_objective.box_decompositions import DominatedPartitioning
from loguru import logger
from numpy.typing import NDArray
from torch import Tensor

from .acquisition_functions.qEHVI import discrete_qEHVI
from.acquisition_functions.qNEHVI import discrete_qNEHVI
from.acquisition_functions.qparEGO import discrete_qparEGO
from.acquisition_functions.qNparEGO import discrete_qNparEGO
from .acquisition_functions.qEI import discrete_qEI
from.acquisition_functions.qPI import discrete_qPI

from .input_parameters import (
    AcquisitionFunctionParameters,
    BayesianOptimizationParameters,
)
from .models.saasgp import get_single_task_model as get_saasgp_model
from .models.skgp import get_sklearn_gp_model
from .models.torchgp import get_single_task_model as get_analytical_model

ACQ_FUNC_MAP = {
    "qEI": discrete_qEI,
    "qPI": discrete_qPI,
    "qEHVI": discrete_qEHVI,
    "qNEHVI": discrete_qNEHVI,
    "qparEGO": discrete_qparEGO,
    "qNparEGO": discrete_qNparEGO,
}


class BayesianOptimization:
    """Multi-Objective Bayesian optimization for Alloy Development.

    Args:
        params (BayesianOptimziationParameters): input parameters.
    """

    def __init__(self, params: BayesianOptimizationParameters) -> None:
        self.params = params
        self.df = pd.read_csv(self.params.csv_file_loc)

        for target in params.targets:
            self.df[f"BO_{target}"] = np.nan
        self.df["iteration"] = np.nan
        self.populate_seed_points()

    def candidate_handler_project(self, n_iter: int, candidates: Tensor) -> None:
        """Project candidate handler.

        Args:
            n_iter (int): Iteration.
            candidates (Tensor): Candidates.
        """
        for cand in candidates:
            temp_dict = dict(zip(self.params.features, cand.tolist()))
            idx = self.get_iloc(temp_dict)
            self.df.at[idx, "iteration"] = n_iter
            for i, target in enumerate(self.params.targets):
                self.df.loc[idx, f"BO_{target}"] = self.df.loc[idx, target]

    def compute_hv(self, is_global: bool = False) -> float:
        """Compute the hypervolume.

        Args:
            is_global (bool, optional): If True, compute the global hypervolume.
                Defaults to False.

        Returns:
            float: Hypervolume.
        """
        global_Y = torch.tensor(
            self.correct_sign(self.df[self.params.targets].values), dtype=torch.double
        )
        ref_points = global_Y.min(0)[0]
        if not is_global:
            cur_Y = torch.tensor(
                self.correct_sign(
                    self.df[~np.isnan(self.df["iteration"])][self.params.targets].values
                ),
                dtype=torch.double,
            )
        bd = DominatedPartitioning(
            ref_point=ref_points,
            Y=global_Y if is_global else cur_Y,
        )
        volume: float = bd.compute_hypervolume().item()
        return volume

    def correct_sign(
        self, x: Union[Tensor, NDArray[np.float64]]
    ) -> Union[Tensor, NDArray[np.float64]]:
        """Correct the sign of the targets.

        Args:
            x (Tensor): Targets.

        Returns:
            Tensor: Corrected targets.
        """
        for i, mask in enumerate(self.params.target_masks):
            x[:, i] = x[:, i] if mask else -x[:, i]  # type: ignore
        return x

    def get_candidates(self, model: ModelListGP, X: Tensor, Y: Tensor) -> Tensor:
        """Get the candidates.

        Args:
            model (ModelListGP): Trained model.
            X (Tensor): Inputs.
            Y (Tensor): Targets.

        Returns:
            Tensor: Candidates.
        """
        acq_key = f"{self.params.acq_func}"
        X_test = self.prepare_test_X()
        acq_params = AcquisitionFunctionParameters(
            model=model,
            Y_train=Y,
            cand_size=self.params.n_candidates,
            ref_points=Y.min(0)[0],
            X_test=X_test,
            X_train=X,
        )
        candidates = ACQ_FUNC_MAP[acq_key](acq_params)
        return candidates

    def get_iloc(self, data: Dict[str, float]) -> int:
        """Get row index from property data."""
        for idx in range(len(self.df)):
            row_dict = self.df.iloc[idx].to_dict()
            if all(np.isclose(data[feat], row_dict[feat])
                   for feat in self.params.features):
                return idx
        return -1

    def get_model(self, X: Tensor, Y: Tensor) -> ModelListGP:
        """Get the trained model.

        Args:
            X (Tensor): Inputs.
            Y (Tensor): Targets.

        Returns:
            ModelListGP: Model.
        """
        if self.params.model == "saasgp":
            return get_saasgp_model(X, Y)
        elif self.params.model == "torchgp":
            return get_analytical_model(X, Y)
        else:
            return get_sklearn_gp_model(X, Y)

    def populate_seed_points(self) -> None:
        """Populate the seed points."""
        seeds = np.random.choice(
            np.arange(len(self.df)), self.params.seed_points, replace=False
        )
        self.df.loc[seeds, "iteration"] = 0
        for i, target in enumerate(self.params.targets):
            self.df.loc[seeds, f"BO_{target}"] = self.df.loc[seeds, target]
        self.save_df()

    def prepare_train_X_Y(self) -> Tuple[Tensor, Tensor]:
        """Prepare training X and Y.

        Returns:
            Tuple[Tensor, Tensor]: X and Y.
        """
        train_df = self.df[~np.isnan(self.df["iteration"])]
        X = torch.tensor(
            train_df[self.params.features].values,
            dtype=torch.double,
            device=self.params.device,
        )
        Y = torch.tensor(
            self.correct_sign(train_df[self.params.targets].values),
            dtype=torch.double,
            device=self.params.device,
        )
        return X, Y

    def prepare_test_X(self) -> Tensor:
        """Prepare testing X.

        Returns:
            Tensor: X.
        """
        test_df = self.df[np.isnan(self.df["iteration"])]
        X = torch.tensor(
            test_df[self.params.features].values, dtype=torch.double,
            device=self.params.device
        )
        return X

    def run_optimization(self) -> None:
        """Run the optimization."""
        logger.info("Starting Bayesian Optimization:\n")
        self.save_features()
        for n_iter in range(self.params.cur_iter, self.params.n_iterations):
            logger.info(
                f"Running iteration {n_iter + 1} of {self.params.n_iterations}."
            )
            candidates = self.single_run()
            self.candidate_handler_project(n_iter + 1, candidates)
            self.save_df()
            self.save_params(n_iter + 1)
            if self.stop_optimization():
                logger.info(f"Reached optimum after {n_iter + 1} iterations")
                break
        logger.info("Optimization completed!\n")

    def save_df(self) -> None:
        """Save the df as a csv file."""
        self.df.to_csv(
            self.params.cwd / f"{self.params.experiment_name}.csv", index=False
        )

    def save_features(self) -> None:
        """Save the features."""
        with open(
            self.params.cwd / f"{self.params.experiment_name}_features.json", "w"
        ) as f:
            json.dump(self.params.features, f, indent=4)

    def save_params(self, n_iter: int) -> None:
        """Save the parameters.

        Args:
            n_iter (int): Iteration.
        """
        d = self.params.to_dict()
        d["cur_iter"] = n_iter
        with open(
            self.params.cwd / f"{self.params.experiment_name}_params.json", "w"
        ) as f:
            json.dump(d, f, indent=4)

    def single_run(self) -> Tensor:
        """Get the best candidate."""
        X, Y = self.prepare_train_X_Y()
        model = self.get_model(X, Y)
        candidates = self.get_candidates(model, X, Y)
        return candidates

    def stop_optimization(self) -> bool:
        """Stop the optimization."""
        if len(self.params.target_masks) == 1:
            if self.params.target_masks[0]:
                cur_max = self.df[f"BO_{self.params.targets[0]}"].max()
                global_max = self.df[self.params.targets[0]].max()
                return bool(np.isclose(cur_max, global_max))
            else:
                cur_min = self.df[f"BO_{self.params.targets[0]}"].min()
                global_min = self.df[self.params.targets[0]].min()
                return bool(np.isclose(cur_min, global_min))
        else:
            max_hv = self.compute_hv(True)
            hv = self.compute_hv()
            return bool(np.isclose(max_hv, hv))
