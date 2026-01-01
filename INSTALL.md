# Installation Guide

## Installing hybrid-tools

### Option 1: Install from PyPI (Recommended)

The package is now available on PyPI and can be installed directly:

```bash
pip install hybrid-tools
```

This is the simplest and recommended method for most users.

### Option 2: Install from Wheel (Pre-built Distribution)

If you have a pre-built wheel file (`.whl`), you can install it directly:

```bash
# Install from a local wheel file
pip install dist/hybrid_tools-0.1.0-py3-none-any.whl

# Or if the wheel is in a different location
pip install /path/to/hybrid_tools-0.1.0-py3-none-any.whl
```

**Building your own wheel:**

```bash
# Install build tools
pip install build

# Build the wheel (creates both .whl and .tar.gz in dist/)
python -m build

# Install the newly built wheel
pip install dist/hybrid_tools-0.1.0-py3-none-any.whl
```

### Option 3: Using uv (Recommended for Development)

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

### Option 4: Install from source (Development Mode with pip)

```bash
# Clone the repository
git clone https://github.com/robomechanics/hybrid-tools.git
cd hybrid-tools

# Install in editable mode with dependencies
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Option 5: Install from source (Standard with pip)

```bash
# Clone the repository
git clone https://github.com/robomechanics/hybrid-tools.git
cd hybrid-tools

# Install the package
pip install .
```

### Option 6: Install directly from GitHub

```bash
pip install git+https://github.com/robomechanics/hybrid-tools.git
```

## Verifying Installation

After installation, verify that the package is installed correctly:

```python
import hybrid_tools
print(hybrid_tools.__version__)  # Should print: 0.1.0

# Import main classes
from hybrid_tools import SKF, HybridSimulator
```

## Distribution

### Building Distribution Files

To create distribution files for sharing or publishing:

```bash
# Install build tools if not already installed
pip install build

# Build both wheel (.whl) and source distribution (.tar.gz)
python -m build

# Output files will be in the dist/ directory:
# - hybrid_tools-0.1.0-py3-none-any.whl (wheel)
# - hybrid_tools-0.1.0.tar.gz (source distribution)
```

### Managing Dependencies

**Important:** Keep runtime dependencies minimal! Only include packages that are **required** for users to run your code.

**Runtime dependencies** (in `dependencies` list):
- Only packages needed to use the library
- Current runtime dependencies: `numpy`, `scipy`, `sympy`, `matplotlib`

**Development dependencies** (in `[project.optional-dependencies]`):
- Testing tools: `pytest`, `pytest-cov`
- Linting/formatting: `ruff`, `mypy`
- Development tools: `pre-commit`

**Build/publish tools** (install separately, NOT in pyproject.toml):
- `build` - for building wheels
- `twine` - for uploading to PyPI

These should be installed separately when needed:
```bash
pip install build twine
```

### Publishing to PyPI

**Note:** The package is already published on PyPI at https://pypi.org/project/hybrid-tools/

To update the package on PyPI with a new version:

1. **Update the version number** in `pyproject.toml`

2. **Review dependencies** in `pyproject.toml`:
   - Ensure only necessary runtime dependencies are in the `dependencies` list
   - Move development tools to `[project.optional-dependencies]`
   - Never include `build` or `twine` in dependencies

3. **Build the distribution files:**
   ```bash
   # Install build tools (if not already installed)
   pip install build twine
   
   # Clean previous builds
   rm -rf dist/
   
   # Build new distribution files
   python -m build
   ```

4. **Upload to PyPI:**
   ```bash
   # Upload to PyPI (you'll be prompted for your API token)
   python -m twine upload dist/*
   ```

**Important Notes:**
- You need a PyPI account and API token to upload
- The version number in `pyproject.toml` must be higher than the current published version
- It's recommended to test on Test PyPI first for major updates:
  ```bash
  python -m twine upload --repository testpypi dist/*
  ```

After publishing, users can install the new version with:
```bash
pip install --upgrade hybrid-tools
```

## Dependencies

The package requires the following runtime dependencies (automatically installed):
- numpy >= 1.20.0
- scipy >= 1.7.0
- sympy >= 1.9
- matplotlib >= 3.3.0

**Note:** If you previously had `graphviz` or `twine` listed as dependencies, these have been removed as they are not required for normal package usage. `graphviz` is only needed if you want to visualize graphs, and `twine` is only needed for publishing packages to PyPI.

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
