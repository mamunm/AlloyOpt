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

## Author

- Osman Mamun, [Contact](mailto:mamun.che06@gmail.com), [Github](https://github.com/mamunm)
