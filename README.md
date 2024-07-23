# AlloyOpt — Multi-Objective Bayesian Optimization for Alloy Development

A general purpose toolbox for Bayesian Optimization in both single and multi-objective settings for both computational and experimental campaigns in discreet design space.

## Installation

To create a virtual environment with AlloyOpt using conda or mamba, run the following command:

```bash
mamba env create -n alloy_opt -f environment.yml
```

If one wants to run the code without installing it, you need to add the following to your run script:

```python
import sys
sys.path.append("/path/to/alloy_opt")
```

Additionally, one can install the package in the current environment with the following command:

```bash
pip install -e .
```

## Running an example project

To run a project, one needs to specify the location of the csv file and some additional
simulation details:

```python
# An example script to run the alloy_opt

# import sys
import warnings
from pathlib import Path

from loguru import logger

warnings.filterwarnings("ignore")

# sys.path.append(str(Path(__file__).parents[1]))
from alloy_opt.input_parameters import BayesianOptimizationParameters  # noqa: E402
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
    experiment_name="example_run",
)

bo = BayesianOptimization(param)
bo.run_optimization()
logger.remove(log_file)
```

Other than the csv file location, one needs to specify the `model`, `acq_func`,
`features`, `targets` etc. `target_masks` defines the maximimization or minimization
problem. For maximization, the corresponding target_mask should be `True` and
for minimization, the mask should be `False`.

## Author

- Osman Mamun, [Contact](mailto:mamun.che06@gmail.com), [Github](https://github.com/mamunm)
