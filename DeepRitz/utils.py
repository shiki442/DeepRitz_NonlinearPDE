import os
import glob
from pyexpat import model
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from DeepRitz.problem import EllipticPDE
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


def get_latest_checkpoint(ckpt_dir):
    """
    自动查找最新的 checkpoint 文件。
    策略：
    1. 优先找 'last.ckpt' (由 save_last=True 生成)
    2. 如果没有，找 epoch 数最大的 model-*.ckpt
    """
    if not os.path.exists(ckpt_dir):
        return None

    # 1. 检查是否存在 last.ckpt
    last_ckpt_path = os.path.join(ckpt_dir, "last.ckpt")
    if os.path.exists(last_ckpt_path):
        print(f"[Auto Resume] Found last.ckpt, resuming from: {last_ckpt_path}")
        return last_ckpt_path

    # 2. 如果没有 last.ckpt，尝试根据文件名中的 epoch 排序
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpts:
        return None

    latest_ckpt = max(ckpts, key=os.path.getmtime)
    print(f"[Auto Resume] Found latest checkpoint by time: {latest_ckpt}")
    return latest_ckpt


def precompute_grid(dim, grid_type='diagonal', bound=(0.0, 1.0), padding=0.5):
    # calculate the exact solution, create grid
    n = 100
    grid = torch.linspace(bound[0], bound[1], n + 1)

    if grid_type == 'grid':
        active_dims = min(dim, 2)
        d_grid = torch.meshgrid([grid for _ in range(active_dims)], indexing='ij')
        xyz_grid = torch.stack([g.reshape(-1) for g in d_grid], dim=1)
        if dim>2:
            xyz_grid = F.pad(xyz_grid, (0, dim-2), "constant", padding)
    elif grid_type == 'diagonal':
        xyz_grid = grid.view(-1, 1).expand(-1, dim).clone()
    elif grid_type == 'random':
        xyz_grid = torch.rand(100000, dim)
    else:
        print('Error: invalid grid type')
        sys.exit(-1)
    return xyz_grid


def plot_pde_results(pde: EllipticPDE, net: torch.nn.Module, writer=None, epoch=None, padding=0.5):
    # ---------------------------------------------------------
    # 定义网格范围
    x = np.linspace(pde.bound[0], pde.bound[1], 100)
    y = np.linspace(pde.bound[0], pde.bound[1], 100)
    X, Y = np.meshgrid(x, y)

    xyz_plot = torch.tensor(np.vstack([X.ravel(), Y.ravel()]).T, dtype=torch.float32).to(next(net.parameters()).device)
    if pde.dim >= 2:
        xyz_plot = F.pad(xyz_plot, (0, pde.dim - 2), "constant", padding)

    net.eval()
    with torch.no_grad():
        u_nn = net(xyz_plot)
    u_nn = u_nn.reshape(X.shape).detach().cpu().numpy()

    u_exact = pde.u_exact(xyz_plot)
    u_exact = u_exact.reshape(X.shape).cpu().numpy()

    # 计算绝对误差 (Absolute Error)
    Error = np.abs(u_exact - u_nn)

    Res = pde.res(net, xyz_plot).cpu().numpy()
    # ---------------------------------------------------------
    # 创建一个包含4个子图的画布
    # --- 图1: 精确解 ---
    fig, axes = plt.subplots(1, 4, figsize=(22, 5), constrained_layout=True)
    im1 = axes[0].pcolormesh(X, Y, u_exact, cmap='viridis', shading='auto')
    axes[0].set_title('Exact Solution ($u_{exact}$)', fontsize=15)
    axes[0].set_xlabel('$x$')
    axes[0].set_ylabel('$y$')
    axes[0].set_aspect('equal')  # 保持x和y轴比例一致
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

    # --- 图4: 残差 ---
    im4 = axes[3].pcolormesh(X, Y, Res.reshape(X.shape), cmap='plasma', shading='auto')
    axes[3].set_title('Residual ($Res$)', fontsize=15)
    axes[3].set_xlabel('$x$')
    axes[3].set_ylabel('$y$')
    axes[3].set_aspect('equal')
    cbar4 = fig.colorbar(im4, ax=axes[3])
    cbar4.ax.set_ylabel('Residual magnitude')

    if writer is not None and epoch is not None:
        writer.add_figure('Solution/NN Solution', fig, epoch)
    plt.close(fig)

    # ---------------------------------------------------------
    # 对角线对比图
    k = np.sqrt(pde.dim)
    xyz_diag = torch.linspace(pde.bound[0], pde.bound[1], 100).view(-1, 1).expand(-1, pde.dim).clone()
    with torch.no_grad():
        xyz_diag = xyz_diag.to(next(net.parameters()).device)
        u_nn = net(xyz_diag).detach().cpu().numpy()
        u_exact = pde.u_exact(xyz_diag).cpu().numpy()

    x_axis = k * xyz_diag[:, 0].cpu().numpy()

    # 2. 绘制图像
    fig = plt.figure(figsize=(8, 5))
    plt.plot(x_axis, u_exact, 'k--', label='Exact')
    plt.plot(x_axis, u_nn, 'r-', alpha=0.7, label='NN')
    plt.title(f'Epoch {epoch}: Diagonal Comparison')
    plt.xlabel('Coordinate')
    plt.ylabel('u')
    plt.legend()
    plt.grid(True)
    if writer is not None and epoch is not None:
        writer.add_figure('Solution/Diagonal_Plot', fig, epoch)
    plt.close(fig)