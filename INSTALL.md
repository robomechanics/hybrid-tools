# Installation Guide

## Installing hybrid-tools

### Option 1: Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver. This is the recommended method for development.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/robomechanics/hybrid-tools.git
cd hybrid-tools

# Sync dependencies and install the package
uv sync

# Or sync with development dependencies
uv sync --all-extras
```

### Option 2: Install from source (Development Mode with pip)

```bash
# Clone the repository
git clone https://github.com/robomechanics/hybrid-tools.git
cd hybrid-tools

# Install in editable mode with dependencies
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Option 3: Install from source (Standard with pip)

```bash
# Clone the repository
git clone https://github.com/robomechanics/hybrid-tools.git
cd hybrid-tools

# Install the package
pip install .
```

### Option 4: Install directly from GitHub

```bash
pip install git+https://github.com/robomechanics/hybrid-tools.git
```

## Verifying Installation

After installation, verify that the package is installed correctly:

```python
import hybrid_tools
print(hybrid_tools.__version__)

# Import main classes
from hybrid_tools import SKF, HybridSimulator
```

## Dependencies

The package requires the following dependencies (automatically installed):
- numpy >= 1.20.0
- scipy >= 1.7.0
- sympy >= 1.9
- matplotlib >= 3.3.0

## Running Examples

After installation, you can run the example scripts:

### With uv:
```bash
# Run the simple hybrid system example
uv run python scripts/simple_hybrid_system.py

# Run the bouncing ball example (note: this is a template and needs completion)
uv run python scripts/bouncing_ball_hybrid_system.py
```

### With pip:
```bash
# Run the simple hybrid system example
python scripts/simple_hybrid_system.py

# Run the bouncing ball example (note: this is a template and needs completion)
python scripts/bouncing_ball_hybrid_system.py
```

## Development Setup

### With uv (Recommended):
```bash
# Sync all dependencies including dev extras
uv sync --all-extras

# Run tests
uv run pytest

# Format code
uv run black src/ scripts/

# Lint code
uv run flake8 src/ scripts/

# Type check
uv run mypy src/
```

### With pip:
```bash
pip install -e ".[dev]"
```

Development dependencies include:
- pytest (testing)
- pytest-cov (coverage)
- black (code formatting)
- flake8 (linting)
- mypy (type checking)

## Troubleshooting

### Import Errors

If you encounter import errors after installation, ensure:
1. You're using the correct Python environment
2. The package was installed successfully: `pip list | grep hybrid-tools`
3. You're not in the source directory when trying to import (this can cause conflicts)

### Dependency Issues

If you have dependency conflicts, try creating a fresh virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

## Uninstalling

To uninstall the package:

```bash
pip uninstall hybrid-tools
