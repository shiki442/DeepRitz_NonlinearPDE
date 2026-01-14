import os
from pyexpat import model
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from DeepRitz.problem import EllipticPDE
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

def precompute_grid(dim, grid_type='diagonal', nonli_func='func1'):
    # calculate the exact solution, create grid
    n = 100
    if nonli_func == "func1":
        grid = torch.linspace(0.0, 1.0, n + 1)
    else:
        grid = torch.linspace(-1.0, 1.0, n + 1)
    if grid_type == 'grid':
        d_grid = torch.meshgrid([grid for i in range(dim)])
        xyz_grid = torch.empty([(n + 1) ** dim, dim])
        for i in range(dim):
            a = d_grid[i].reshape(-1, 1)
            for j in range((n + 1) ** dim):
                xyz_grid[j, i] = a[j, 0]
    elif grid_type == 'x-axis':
        xyz_grid = 0.5 * torch.ones(n + 1, dim)
        for j in range(n + 1):
            xyz_grid[j, 0] = grid[j]
    elif grid_type == 'diagonal':
        xyz_grid = torch.zeros(n + 1, dim)
        for j in range(n + 1):
            for i in range(dim):
                xyz_grid[j, i] = grid[j]
    elif grid_type == 'random':
        xyz_grid = torch.rand(100000, dim)
    else:
        print('Error: invalid grid type')
        sys.exit(-1)
    return xyz_grid


def plot_pde_results(pde:EllipticPDE, net:torch.nn.Module, writer=None, epoch=None):
    # ---------------------------------------------------------
    # 定义网格范围
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)

    xyz_plot = torch.tensor(np.vstack([X.ravel(), Y.ravel()]).T, dtype=torch.float32).to(next(net.parameters()).device)
    if pde.dim >= 2:
        xyz_plot = F.pad(xyz_plot, (0, pde.dim - 2), "constant", 0.5)

    net.eval() 
    with torch.no_grad():
        u_nn = net(xyz_plot)
    u_nn = u_nn.reshape(X.shape).detach().cpu().numpy()

    u_exact = pde.u_exact(xyz_plot)
    u_exact = u_exact.reshape(X.shape).cpu().numpy()

    # 计算绝对误差 (Absolute Error)
    Error = np.abs(u_exact - u_nn)

    # ---------------------------------------------------------
    # 创建一个包含3个子图的画布
    # --- 图1: 精确解 ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    im1 = axes[0].pcolormesh(X, Y, u_exact, cmap='viridis', shading='auto')
    axes[0].set_title('Exact Solution ($u_{exact}$)', fontsize=15)
    axes[0].set_xlabel('$x$')
    axes[0].set_ylabel('$y$')
    axes[0].set_aspect('equal') # 保持x和y轴比例一致
    cbar1 = fig.colorbar(im1, ax=axes[0])
    cbar1.ax.set_ylabel('u value')

    # --- 图2: 数值解 ---
    im2 = axes[1].pcolormesh(X, Y, u_nn, cmap='viridis', shading='auto')
    axes[1].set_title('Numerical Solution ($u_{num}$)', fontsize=15)
    axes[1].set_xlabel('$x$')
    axes[1].set_ylabel('$y$')
    axes[1].set_aspect('equal')
    cbar2 = fig.colorbar(im2, ax=axes[1])
    cbar2.ax.set_ylabel('u_nn value')

    # --- 图3: 误差 ---
    im3 = axes[2].pcolormesh(X, Y, Error, cmap='inferno', shading='auto')
    axes[2].set_title('Absolute Error ($|u_{exact} - u_{num}|$)', fontsize=15)
    axes[2].set_xlabel('$x$')
    axes[2].set_ylabel('$y$')
    axes[2].set_aspect('equal')
    cbar3 = fig.colorbar(im3, ax=axes[2])
    cbar3.ax.set_ylabel('Error magnitude')

    if writer is not None and epoch is not None:
        writer.add_figure('Solution/NN Solution', fig, epoch)
    plt.close(fig)
