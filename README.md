# DeepRitz for Nonlinear Elliptic PDEs

Deep learning-based implementation of the Deep Ritz Method for solving nonlinear elliptic partial differential equations. This project is the official code release accompanying the paper "Analysis of Deep Ritz Methods for Semilinear Elliptic Equations".

**Paper Link**: [https://global-sci.com/index.php/nmtma/article/view/14474](https://global-sci.com/index.php/nmtma/article/view/14474)

## Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Equations and Boundary Value Problems](#equations-and-boundary-value-problems)
- [Network Architecture](#network-architecture)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Visualization and Logging](#visualization-and-logging)
- [Project Structure](#project-structure)
- [Citation](#citation)

## Introduction

The Deep Ritz Method is a variational principle-based deep learning numerical method for solving boundary value problems of partial differential equations. This method transforms the variational form of PDEs into a neural network optimization problem, training the network to approximate the PDE solution by minimizing the energy functional.

This implementation supports:
- Multiple nonlinear terms (Sigmoid, Sin, Exp, Poly, Inverse)
- Multiple exact solution types (onepeak, twopeak, nondiff, liouville, yamabe)
- Arbitrary dimensional problems (2D, 3D, 5D, etc.)
- Residual network architecture (ResNet-style)
- Automatic checkpoint resumption
- TensorBoard visualization
- Optuna hyperparameter auto-tuning

## Key Features

- **PyTorch Lightning Implementation**: Modular and extensible training system
- **Offline Dataset Generation**: Sobol sequence-based quasi-random sampling, pre-generated and saved for faster training
- **Automatic Learning Rate Scheduling**: ReduceLROnPlateau strategy monitoring validation error
- **Automatic Checkpoint Resumption**: Auto-find and load latest checkpoint to continue training
- **Hyperparameter Optimization**: Optuna integration with Bayesian optimization and automatic pruning
- **Visualization**: TensorBoard logging for loss curves, error analysis, and solution visualization

## Installation

### Dependencies

```bash
# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install torch pytorch-lightning ml_collections optuna scipy matplotlib tensorboard
```

### GPU Support

For GPU training, install CUDA-enabled PyTorch first:

```bash
# Visit https://pytorch.org for installation commands suitable for your CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### Training

```bash
# Train with default configuration (eq1: twopeak solution, Exp nonlinearity)
python main_pl.py

# SLURM cluster submission
sbatch run.sh
```

### Switch Equations

Modify the config import in `main_pl.py` or `tune_pl.py`:

```python
from config.eq1 import get_config  # Default: twopeak, Exp nonlinearity
from config.eq2 import get_config  # nondiff solution, Sin nonlinearity
from config.eq3 import get_config  # Liouville equation
from config.eq4 import get_config  # Yamabe equation, Poly nonlinearity
```

### Hyperparameter Tuning

```bash
# Run Optuna search (uses eq4 config by default)
python tune_pl.py
```

## Configuration

Configuration files are located in the `config/` directory, with one config file per equation. Configuration uses `ml_collections.ConfigDict`:

### Configuration Structure

```python
cfg.model      # PDE problem parameters
cfg.training   # Training hyperparameters
cfg.net        # Network architecture parameters
cfg.verbose    # Logging and visualization settings
cfg.data       # Data path configuration
cfg.device     # Computing device ('cuda' or 'cpu')
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model.dim` | Spatial dimension | 2 |
| `model.batch_in` | Interior points batch size | 500000 |
| `model.batch_bd` | Boundary points batch size | 200000 |
| `model.lambda_1` | Boundary penalty coefficient | 5000.0 |
| `model.sol_func` | Exact solution type | 'twopeak' |
| `model.nonli_func` | Nonlinearity type | 'Exp' |
| `model.beta` | Nonlinearity parameters | [1.0, 1.0] |
| `training.n_epochs` | Number of training epochs | 100000 |
| `training.lr` | Initial learning rate | 0.001 |
| `training.patience` | LR reduction patience | 10000 |
| `training.gamma` | LR reduction factor | 0.5 |
| `net.depth` | Number of residual blocks | 6 |
| `net.width` | Network width | 100 |
| `net.act` | Activation function | 'ReLU6p' |
| `verbose.plot_interval` | Visualization interval (epochs) | 1000 |
| `verbose.ckpt_interval` | Checkpoint save interval | 100 |

## Equations and Boundary Value Problems

### Supported Exact Solution Types

| Type | Description | Dimension |
|------|-------------|-----------|
| `onepeak` | Single peak: u(x) = ∏ 4xᵢ(1-xᵢ) | Any |
| `twopeak` | Two peaks: u(x) = sin(πx₁) × ∏ 4xᵢ(1-xᵢ) | Any |
| `nondiff` | Non-differentiable: u(x) = max(0, 0.5 - \|x\|²) | Any |
| `liouville` | Liouville equation: u(r) = C·log(1+r²) | 2D |
| `yamabe` | Yamabe problem: u(r) = k/(1+r²)^(d/2-1) | ≥3 |

### Supported Nonlinearity Types

| Type | Mathematical Form | Parameters beta |
|------|-------------------|-----------------|
| `Sigmoid` | V(u) = σ(u) | - |
| `Sin` | V(u) = β₁·sin(β₂·u) | [β₁, β₂] |
| `Exp` | V(u) = β·exp(u) | [β] |
| `Poly` | V(u) = β₁·u^β₂ | [β₁, β₂] |
| `Inverse` | V(u) = 1/(1+u²) | - |

## Network Architecture

### SolutionNet

Residual network architecture:

```
Input → Linear → [Block × N] → Linear → Output
                 ↑
            Skip Connection
```

### Block Structure

```
x → Dense1 → Act → Dense2 → Act → (+x) → Output
```

### Supported Activation Functions

| Name | Function | Description |
|------|----------|-------------|
| `ReLU6p` | ReLU6(x)^1.5 | Default activation |
| `ReLUsq` | ReLU(x)² | Squared ReLU |
| `ReLU6sq` | ReLU6(x)² | Squared ReLU6 |
| `ReLU` | ReLU(x) | Standard ReLU |
| `SiLU` | x·sigmoid(x) | Swish linear unit |
| `Tanh` | tanh(x) | Hyperbolic tangent |

## Hyperparameter Tuning

The project uses Optuna for automated hyperparameter search:

```python
# Search space in tune_pl.py
lr          # Learning rate: [2e-4, 2e-3] log scale
width       # Network width: [40, 150]
depth       # Network depth: [2, 6]
activation  # Activation: ['Tanh', 'ReLU6p']
lambda_1    # Boundary penalty: [2000, 5000]
```

### Pruning Strategy

Uses `MedianPruner` to automatically prune underperforming trials, reducing computational waste.

### Resumed Search

Optuna uses SQLite to store search results, supporting interruption and resumption:

```bash
# Automatically loads previous search results to continue optimization
python tune_pl.py
```

## Visualization and Logging

### TensorBoard

Start TensorBoard to view training progress:

```bash
tensorboard --logdir ./logs/
```

### Visualization Content

- **Loss Curves**: Total loss, interior loss, boundary loss
- **Error Analysis**: Relative L2 error
- **Residual**: Mean absolute PDE residual
- **Solution Visualization**: Exact solution, numerical solution, absolute error, residual distribution
- **Diagonal Comparison**: Exact vs numerical solution along the diagonal

### Checkpoints

Model checkpoints are saved in `./checkpoints/`:
- `last.ckpt`: Latest checkpoint (for auto-resume)
- `model-{epoch:02d}-{val_error:.2e}.ckpt`: Top-K checkpoints by validation error

## Project Structure

```
DeepRitz_NonlinearPDE/
├── main_pl.py              # PyTorch Lightning training entry
├── main.py                 # Legacy training entry
├── tune_pl.py              # Optuna hyperparameter tuning
├── plot_loss_curves.py     # Loss curve visualization script
├── plot_checkpoint.py      # Checkpoint visualization script
├── read_tb_logs.py         # TensorBoard log reader utility
├── visualize_n.py          # Solution visualization script
├── visualize_w.py          # Weight visualization script
│
├── config/
│   ├── eq1.py              # Default config (twopeak, Exp)
│   ├── eq2.py              # nondiff solution, Sin nonlinearity
│   ├── eq3.py              # Liouville equation
│   ├── eq4.py              # Yamabe equation, Poly nonlinearity
│   └── eq5.py              # Other equation configs
│
├── DeepRitz/
│   ├── nn.py               # Neural network architectures (SolutionNet, Block)
│   ├── problem.py          # PDE problem definitions (EllipticPDE)
│   ├── loss.py             # Variational loss (VarLoss)
│   ├── data_pl.py          # PyTorch Lightning data module
│   ├── model_pl.py         # PyTorch Lightning model
│   └── utils.py            # Utility functions (checkpoint, visualization, grid generation)
│
├── data/                   # Offline datasets (.pt files)
├── checkpoints/            # Model checkpoints
├── logs/                   # TensorBoard logs
├── dbeq4.sqlite            # Optuna search results database
│
├── CLAUDE.md               # Claude Code project guide
└── README.md               # This file (Chinese version)
└── README_EN.md            # This file (English version)
```

## Data Generation

Training data uses Sobol sequences for quasi-random sampling:

- **Interior Points**: Uniformly sampled within the computational domain
- **Boundary Points**: Sampled on boundary faces

Datasets are pre-generated and saved as `.pt` files to avoid regeneration on each training run.

```python
# Data generation config
cfg.data.file_path = './data/eq1_{batch_in}_{batch_bd}.pt'
```

**Note**: After modifying `batch_in` or `batch_bd`, delete old data files to regenerate.

## Algorithm Overview

The Deep Ritz Method is based on the following variational principle:

For nonlinear elliptic equations:

```
-Δu + V(u) = f,  in Ω
u = g,           on ∂Ω
```

The corresponding energy functional is:

```
E(u) = ∫Ω [½|∇u|² + F(u) - fu] dx + λ∫∂Ω |u-g|² ds
```

where F(u) = ∫V(u)du, and λ is the boundary penalty coefficient.

The neural network approximates the true solution by minimizing the discretized energy functional.

## Frequently Asked Questions

### Q: How to modify the problem dimension?

Change `cfg.model.dim` in the configuration file, and ensure you select a compatible exact solution type (e.g., liouville only supports 2D).

### Q: Training not converging?

1. Increase boundary penalty coefficient `lambda_1`
2. Adjust learning rate or use smaller initial learning rate
3. Increase network depth or width
4. Try different activation functions (Tanh is often smoother)

### Q: How to resume training?

The program automatically finds the latest checkpoint in `checkpoints/` and resumes training. Ensure `last.ckpt` file exists.

### Q: How to modify boundary conditions?

Currently implements Dirichlet boundary conditions. To change boundary condition types, modify the `EllipticPDE` class in `DeepRitz/problem.py`.

## Citation

If you use this code, please cite:

```bibtex
@article{chen2024analysis,
  title={Analysis of Deep Ritz methods for semilinear elliptic equations},
  author={Chen, Mo and Jiao, Yuling and Lu, Xiliang and Song, Pengcheng and Wang, Fengru and Yang, Jerry Zhijian},
  journal={Numerical Mathematics: Theory, Methods and Applications},
  volume={17},
  number={1},
  pages={181--209},
  year={2024}
}
```

## License

This project code is provided for academic research purposes only.
