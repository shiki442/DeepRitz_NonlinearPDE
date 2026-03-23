# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepRitz implementation for solving nonlinear elliptic PDEs using neural networks and the Deep Ritz Method. The project uses PyTorch and PyTorch Lightning for training, with Optuna for hyperparameter tuning.

## Commands

### Training

```bash
# Run training with PyTorch Lightning (main_pl.py uses config/eq1.py by default)
python main_pl.py

# Run training with legacy implementation
python main.py

# SLURM job submission
sbatch run.sh
```

### Hyperparameter Tuning

```bash
# Run Optuna tuning (tune_pl.py uses config/eq4.py by default)
python tune_pl.py
```

### Configuration

Each equation has its own config file in `config/`:
- `config/eq1.py` - Default equation (twopeak solution, exp nonlinearity)
- `config/eq2.py` - Nondiff solution, Sin nonlinearity (Poly in config)
- `config/eq3.py` - Liouville equation, Exp nonlinearity
- `config/eq4.py` - Yamabe equation, Poly nonlinearity

To switch equations, modify the import in `main_pl.py` or `tune_pl.py`:
```python
from config.eq1 import get_config  # Change eq1 to eq2/eq3/eq4
```

## Architecture

### Core Modules (`DeepRitz/`)

- **`nn.py`** - Neural network architectures
  - `SolutionNet`: Residual network with configurable depth/width/activation
  - `Block`: Residual block used by SolutionNet
  - `get_act()`: Factory for activation functions (ReLU6p, Tanh, SiLU, etc.)

- **`problem.py`** - PDE problem definitions
  - `EllipticPDE`: Base class defining exact solutions, boundary conditions, nonlinearities
  - Supported solutions: onepeak, twopeak, nondiff, liouville, yamabe
  - Supported nonlinearities: Sigmoid, Sin, Exp, Poly

- **`loss.py`** - Variational loss formulation
  - `VarLoss`: Deep Ritz variational loss with interior and boundary terms

- **`data_pl.py`** - Data loading for PyTorch Lightning
  - `DeepRitzDataModule`: Lightning data module with offline dataset generation
  - `DRMDataset`: Dataset using Sobol sequences for quasi-random sampling
  - `generate_offline_dataset()`: Pre-generates interior and boundary points

- **`model_pl.py`** - PyTorch Lightning system
  - `DeepRitzSystem`: LightningModule wrapping the network, loss, and optimizer
  - Uses ReduceLROnPlateau scheduler with val_error monitoring

- **`utils.py`** - Utilities for checkpointing, visualization, and grid generation

### Configuration System

Uses `ml_collections.ConfigDict` for structured configuration:
- `cfg.model`: PDE parameters (dim, batch sizes, nonlinearity, solution type, lambda_1)
- `cfg.training`: Training hyperparameters (epochs, lr, patience, betas)
- `cfg.net`: Network architecture (depth, width, activation)
- `cfg.data`: Data paths and parameters
- `cfg.verbose`: Logging intervals

### Key Design Patterns

1. **Offline dataset generation**: Interior and boundary points are pre-generated using Sobol sequences and saved to `.pt` files to avoid regeneration

2. **Automatic checkpoint resumption**: `utils.get_latest_checkpoint()` finds the latest checkpoint for auto-resume

3. **Fixed batch handling**: Dataset pre-splits into `n_steps` batches; changing `batch_in`/`batch_bd` requires regenerating the dataset file

4. **TensorBoard logging**: All metrics (loss, error, residual) and solution plots are logged to TensorBoard
