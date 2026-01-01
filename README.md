# hybrid-tools

A Python package for state estimation in hybrid dynamical systems using the Salted Kalman Filter (SKF).

## Overview

This package provides tools for working with hybrid dynamical systems - systems that exhibit both continuous dynamics and discrete mode transitions. The main feature is the Salted Kalman Filter, which extends traditional Kalman filtering to handle these hybrid systems.

## Features

- **Salted Kalman Filter (SKF)**: State estimation for hybrid systems with mode transitions
- **Hybrid System Simulator**: Simulate hybrid dynamical systems with guards and resets
- **Saltation Matrix Computation**: Linearization tools for hybrid systems
- **Example Scripts**: Ready-to-run examples demonstrating the package capabilities

## Installation

### Quick Install with uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/robomechanics/hybrid-tools.git
cd hybrid-tools
uv sync
```

### Alternative: Install with pip

```bash
pip install git+https://github.com/robomechanics/hybrid-tools.git
```

For detailed installation instructions and development setup, see [INSTALL.md](INSTALL.md).

## Quick Start

```python
from hybrid_tools import SKF, HybridSimulator
import numpy as np

# Define your hybrid system dynamics, guards, and resets
# (see examples in scripts/ directory)

# Initialize the Salted Kalman Filter
skf = SKF(
    init_state=initial_state,
    init_mode="mode_name",
    init_cov=initial_covariance,
    dt=timestep,
    noise_matrices=noise_dict,
    dynamics=dynamics_dict,
    resets=resets_dict,
    guards=guards_dict,
    parameters=params
)

# Run prediction and update steps
predicted_state, predicted_cov = skf.predict(current_time, inputs)
filtered_state, filtered_cov = skf.update(current_time, inputs, measurement)
```

## Examples

The `scripts/` directory contains example implementations:

- `simple_hybrid_system.py`: Basic hybrid system demonstration
- `bouncing_ball_hybrid_system.py`: Bouncing ball with state estimation

Run examples after installation:

```bash
# With uv
uv run python scripts/bouncing_ball_hybrid_system.py

# Or with pip
python scripts/bouncing_ball_hybrid_system.py
```

### Bouncing Ball Example Output

The bouncing ball example demonstrates the Salted Kalman Filter tracking a ball bouncing under gravity with mode transitions at impact.

**Phase Plot:**

![Bouncing Ball Phase Plot](docs/bouncing_ball.png)

**Time Series:**

![Bouncing Ball Time Series](docs/bouncing_ball_timeseries.png)

The plots show:
- **Black line**: Actual system states
- **Red dots**: Noisy measurements
- **Blue dashed line**: SKF filtered estimates
- **Colored regions** (time series): Different hybrid system modes

## Citation

If you use this code for your academic research, please cite
```
@article{paper:kong-skf-2021,
  author       = {Nathan J. Kong and J. Joe Payne and George Council and Aaron M. Johnson},
  title        = {The {Salted} {Kalman} {Filter}:  {Kalman} Filtering on Hybrid Dynamical Systems},
  journal      = {Automatica},
  year         = {2021},
  volume       = {131},
  pages        = {109752},
  keywords     = {Hybrid Systems, State Estimation, Uncertainty, Contact},
  url_Publisher     = {https://www.sciencedirect.com/science/article/pii/S0005109821002727},
  url_DOI      = {https://doi.org/10.1016/j.automatica.2021.109752},
  url_arxiv    = {https://arxiv.org/abs/2007.12233},
  url_PDF      = {https://www.sciencedirect.com/science/article/pii/S0005109821002727/pdfft?md5=4bbaaa437df2d1f651affbded8b1115a&pid=1-s2.0-S0005109821002727-main.pdf}
}
```
or 
```
@article{paper:kong-saltation-2023,
  title         = {Saltation Matrices: The Essential Tool for Linearizing Hybrid Dynamical Systems},
  author        = {Kong, Nathan J. and Payne, J. Joe and Zhu, James and Johnson, Aaron M.},
  journal       = {arXiv:2306.06862 [cs.RO]},
  year          = {2023},
  note          = {Under review},
  keywords      = {Hybrid Systems, Control, Contact, Modeling, State Estimation},
  url_Info      = {https://arxiv.org/abs/2306.06862},
  url_PDF       = {https://arxiv.org/pdf/2306.06862}
}
```
