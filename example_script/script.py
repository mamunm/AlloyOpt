# An example script to run the alloy_opt

from pathlib import Path
import sys
import warnings
from loguru import logger

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).parents[1]))
from alloy_opt.input_parameters import (  # noqa: E402
    BayesianOptimizationParameters,
)
from alloy_opt.optimization import BayesianOptimization  # noqa: E402

if Path("example.log").exists():
    Path("example.log").unlink()

log_file = logger.add("example.log")
param = BayesianOptimizationParameters(
    model="skgp",
    acq_func="qEHVI",
    csv_file_loc="example.csv",
    features=["feat_1", "feat_2", "feat_3", "feat_4"],
    targets=["target_1", "target_2"],
    target_masks=[True, False],
    seed_points=20,
    n_iterations=5,
    device="cpu",
    n_candidates=1,
    experiment_name="example_run"
)

bo = BayesianOptimization(param)
bo.run_optimization()
logger.remove(log_file)
