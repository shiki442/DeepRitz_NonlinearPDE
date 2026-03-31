"""
从 checkpoint 加载训练好的模型，绘制 Solution 热力图和对角线上的 1D 图像
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.functional as F

from DeepRitz.nn import SolutionNet
from DeepRitz.problem import EllipticPDE


def load_checkpoint(ckpt_path, cfg):
    """
    从 checkpoint 加载模型
    """
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # 获取网络结构参数
    dim = cfg.model.dim
    block_width = cfg.net.width
    n_blocks = cfg.net.depth
    act = cfg.net.act

    # 创建网络
    net = SolutionNet(
        in_features=dim,
        out_features=1,
        block_width=block_width,
        n_blocks=n_blocks,
        act=act
    )

    # 加载权重 - 处理 PyTorch Lightning 的 state_dict 格式
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # 移除 'solution_net.' 前缀
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('solution_net.'):
                new_state_dict[key.replace('solution_net.', '')] = value
            elif not key.startswith('xyz_grid'):  # 跳过 buffer
                new_state_dict[key] = value

        net.load_state_dict(new_state_dict, strict=False)
    else:
        net.load_state_dict(checkpoint)

    print(f"Loaded checkpoint from: {ckpt_path}")
    return net


def plot_solution_heatmap(pde, net, save_path=None):
    """
    绘制 Solution 热力图（2D）
    """
    # 定义网格范围
    bound = pde.bound
    x = np.linspace(bound[0], bound[1], 200)
    y = np.linspace(bound[0], bound[1], 200)
    X, Y = np.meshgrid(x, y)

    # 创建网格点
    xyz_plot = torch.tensor(np.vstack([X.ravel(), Y.ravel()]).T, dtype=torch.float32)
    if pde.dim >= 2:
        xyz_plot = F.pad(xyz_plot, (0, pde.dim - 2), "constant", pde.padding if hasattr(pde, 'padding') else 0.5)

    net.eval()
    with torch.no_grad():
        xyz_plot = xyz_plot.to(next(net.parameters()).device)
        u_nn = net(xyz_plot)
    u_nn = u_nn.reshape(X.shape).detach().cpu().numpy()

    # 计算精确解
    u_exact = pde.u_exact(xyz_plot)
    u_exact = u_exact.reshape(X.shape).cpu().numpy()

    # 计算绝对误差
    Error = np.abs(u_exact - u_nn)

    # 计算统一的 colorbar 范围（取数值解和精确解的全局最值）
    vmin = u_exact.min()
    vmax = u_exact.max()

    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    # 图 1: 数值解
    im1 = axes[0].pcolormesh(X, Y, u_nn, cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)
    axes[0].set_title('Network Solution ($u_{pred}$)', fontsize=15)
    axes[0].set_xlabel('$x$')
    axes[0].set_ylabel('$y$')
    axes[0].set_aspect('equal')
    cbar1 = fig.colorbar(im1, ax=axes[0])
    cbar1.ax.set_ylabel('u value')

    # 图 2: 精确解
    im2 = axes[1].pcolormesh(X, Y, u_exact, cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)
    axes[1].set_title('Exact Solution ($u_{exact}$)', fontsize=15)
    axes[1].set_xlabel('$x$')
    axes[1].set_ylabel('$y$')
    axes[1].set_aspect('equal')
    cbar2 = fig.colorbar(im2, ax=axes[1])
    cbar2.ax.set_ylabel('u value')

    # 图 3: 误差
    im3 = axes[2].pcolormesh(X, Y, Error, cmap='inferno', shading='auto')
    axes[2].set_title('Absolute Error ($|u_{exact} - u_{pred}|$)', fontsize=15)
    axes[2].set_xlabel('$x$')
    axes[2].set_ylabel('$y$')
    axes[2].set_aspect('equal')
    cbar3 = fig.colorbar(im3, ax=axes[2])
    cbar3.ax.set_ylabel('Error magnitude')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved heatmap to: {save_path}")

    plt.close(fig)
    return fig


def plot_diagonal_1d(pde, net, save_path=None):
    """
    绘制对角线上的 1D 图像
    """
    net.eval()

    # 对角线上的点 (x, x, x, ...)
    bound = pde.bound
    xyz_diag = torch.linspace(bound[0], bound[1], 200).view(-1, 1)
    xyz_diag = xyz_diag.expand(-1, pde.dim).clone()
    
    with torch.no_grad():
        xyz_diag = xyz_diag.to(next(net.parameters()).device)
        u_nn = net(xyz_diag).detach().cpu().numpy()
        u_exact = pde.u_exact(xyz_diag).cpu().numpy()

    # 计算误差
    error = u_nn - u_exact

    # x 轴坐标 (缩放后)
    k = np.sqrt(pde.dim)
    x_axis = k * xyz_diag[:, 0].cpu().numpy()

    # 创建图形，左右两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：函数值对比
    ax1.plot(x_axis, u_exact, 'k--', linewidth=2, label='$u_{exact}$')
    ax1.plot(x_axis, u_nn, 'r-', linewidth=2, alpha=0.7, label='$u_{pred}$')
    ax1.set_title(f'Diagonal Comparison ($x_1=x_2$={x_axis[0]:.2f} to {x_axis[-1]:.2f})', fontsize=15)
    ax1.set_xlabel(f'$\\sqrt{{{pde.dim}}} \\cdot x$', fontsize=12)
    ax1.set_ylabel('u', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.3, color='gray')

    # 右图：误差曲线
    max_abs_error = np.max(np.abs(error))
    ax2.plot(x_axis, error, '--', linewidth=2, color='#3b82f6', alpha=0.7, label='Error ($u_{pred}-u_{exact}$)')
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_title('Absolute Error on Diagonal', fontsize=15)
    ax2.set_xlabel(f'$\\sqrt{{{pde.dim}}} \\cdot x$', fontsize=12)
    ax2.set_ylabel('Error', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.3, color='gray')
    # 设置 y 轴范围
    ax2.set_ylim(-max_abs_error * 5.0, max_abs_error * 5.0)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved diagonal plot to: {save_path}")

    plt.close(fig)
    return fig


def plot_3d_surface(pde, net, save_path=None):
    """
    绘制 3D 表面图
    """
    bound = pde.bound
    x = np.linspace(bound[0], bound[1], 100)
    y = np.linspace(bound[0], bound[1], 100)
    X, Y = np.meshgrid(x, y)

    xyz_plot = torch.tensor(np.vstack([X.ravel(), Y.ravel()]).T, dtype=torch.float32)
    if pde.dim >= 2:
        xyz_plot = F.pad(xyz_plot, (0, pde.dim - 2), "constant", pde.padding if hasattr(pde, 'padding') else 0.5)

    net.eval()
    with torch.no_grad():
        xyz_plot = xyz_plot.to(next(net.parameters()).device)
        u_nn = net(xyz_plot)
    Z = u_nn.reshape(X.shape).detach().cpu().numpy()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
    ax.set_title('Network Solution (3D Surface)', fontsize=15)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('u')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved 3D surface to: {save_path}")

    plt.close(fig)
    return fig


def main():
    # 选择要绘制的 checkpoint
    print("Available checkpoints:")
    print("=" * 60)

    checkpoint_dirs = ['checkpoints/eq1','checkpoints/eq2', 'checkpoints/eq3', 'checkpoints/eq4']
    all_checkpoints = {}

    for ckpt_dir in checkpoint_dirs:
        if os.path.exists(ckpt_dir):
            ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
            if ckpts:
                all_checkpoints[ckpt_dir] = ckpts
                print(f"\n{ckpt_dir}/:")
                for ckpt in sorted(ckpts):
                    print(f"  - {ckpt}")

    print("\n" + "=" * 60)

    # 配置：选择要绘制的 checkpoint 和对应的配置文件
    # 默认使用每个方程的 last.ckpt 或最好的 checkpoint

    checkpoint_configs = [
        # ('checkpoints/eq1/model-epoch=2399-val_error=5.44e-03.ckpt', 'config.eq1', 'eq1_twopeak'),
        # ('checkpoints/eq2/model-epoch=399-val_error=5.96e-03.ckpt', 'config.eq2', 'eq2_nondiff'),
        # ('checkpoints/eq3/model-epoch=11699-val_error=3.05e-04.ckpt', 'config.eq3', 'eq3_liouville'),
        ('checkpoints/eq4/model-epoch=1899-val_error=8.02e-03.ckpt', 'config.eq4', 'eq4_yamabe'),
    ]

    # 创建输出目录
    os.makedirs('figures', exist_ok=True)

    for ckpt_path, config_module_name, output_prefix in checkpoint_configs:
        if not os.path.exists(ckpt_path):
            print(f"\nSkipping {ckpt_path} (not found)")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {ckpt_path}")
        print(f"{'='*60}")

        # 导入配置
        config_module = __import__(config_module_name, fromlist=['get_config'])
        cfg = config_module.get_config()

        # 设置 padding
        # cfg.data.padding = 0.5

        # 加载模型
        net = load_checkpoint(ckpt_path, cfg)
        device = cfg.device
        net = net.to(device)
        net.eval()

        # 创建 PDE 问题
        pde = EllipticPDE(
            sol=cfg.model.sol_func,
            nonli=cfg.model.nonli_func,
            dim=cfg.model.dim,
            beta=cfg.model.beta,
            bound=tuple(cfg.model.bound)
        )
        pde.padding = cfg.data.padding

        # 绘制热力图
        heatmap_path = f'figures/{output_prefix}_heatmap.png'
        plot_solution_heatmap(pde, net, save_path=heatmap_path)

        # 绘制对角线 1D 图
        diagonal_path = f'figures/{output_prefix}_diagonal.png'
        plot_diagonal_1d(pde, net, save_path=diagonal_path)

        # 绘制 3D 表面图
        surface_path = f'figures/{output_prefix}_3d.png'
        plot_3d_surface(pde, net, save_path=surface_path)

        print(f"\nCompleted: {output_prefix}")

    print("\n" + "=" * 60)
    print("All done! Check the 'figures' folder for the plots.")


if __name__ == '__main__':
    main()
