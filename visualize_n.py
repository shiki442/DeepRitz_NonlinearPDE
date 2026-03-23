"""
Eq5 可视化脚本：不同样本量实验的结果分析
完成以下任务：
1. 读取 tensorboard 日志提取相关数据
2. 绘制 5 个不同样本量下的 error-epoch 曲线（同一个折线图）
3. 绘制五个参数下的解的热力图以及误差的热力图（subfigure）
4. 绘制样本数 - 误差关系折线图
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tensorboard.backend.event_processing import event_accumulator
import torch.nn.functional as F
from importlib import import_module
from scipy.ndimage import uniform_filter1d

from DeepRitz.nn import SolutionNet
from DeepRitz.problem import EllipticPDE

# ============================================================
# 配置：5 个不同样本量的实验
# ============================================================
EXPERIMENTS = {
    '1000':   {'version': 'version_6', 'log_dir': './logs/eq5/lightning_logs/version_6', 'ckpt_dir': './checkpoints/eq5_1000'},
    '5000':   {'version': 'version_3', 'log_dir': './logs/eq5/lightning_logs/version_3', 'ckpt_dir': './checkpoints/eq5_5000'},
    '10000':  {'version': 'version_5', 'log_dir': './logs/eq5/lightning_logs/version_5', 'ckpt_dir': './checkpoints/eq5_10000'},
    '50000':  {'version': 'version_4', 'log_dir': './logs/eq5/lightning_logs/version_4', 'ckpt_dir': './checkpoints/eq5_50000'},
    '500000': {'version': 'version_0', 'log_dir': './logs/eq5/lightning_logs/version_0', 'ckpt_dir': './checkpoints/eq5_500000'},
}

# 样本数（用于排序和绘图）
SAMPLE_SIZES = [1000, 5000, 10000, 50000, 500000]
COLORS = plt.cm.viridis(np.linspace(0, 0.9, len(SAMPLE_SIZES)))

# 输出目录
OUTPUT_DIR = './figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 工具函数
# ============================================================

def extract_scalar_data(ea, tag):
    """提取标量数据并返回 epoch 和 value 列表"""
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def load_tensorboard_data(log_dir):
    """从 tensorboard 日志中加载数据"""
    ea = event_accumulator.EventAccumulator(
        log_dir,
        size_guidance={event_accumulator.SCALARS: 0}
    )
    ea.Reload()
    return ea


def get_best_checkpoint(ckpt_dir):
    """获取最佳 checkpoint（最低 val_error）"""
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt') and 'val_error' in f]
    if not ckpts:
        return None

    best_ckpt = None
    best_error = float('inf')
    for ckpt in ckpts:
        try:
            error_str = ckpt.split('val_error=')[1].split('.ckpt')[0]
            error = float(error_str.replace('e-0', 'e-').replace('e-', 'e-'))
            if error < best_error:
                best_error = error
                best_ckpt = ckpt
        except:
            continue
    return os.path.join(ckpt_dir, best_ckpt) if best_ckpt else None


def load_checkpoint(ckpt_path, cfg):
    """从 checkpoint 加载模型"""
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    dim = cfg.model.dim
    block_width = cfg.net.width
    n_blocks = cfg.net.depth
    act = cfg.net.act

    net = SolutionNet(
        in_features=dim,
        out_features=1,
        block_width=block_width,
        n_blocks=n_blocks,
        act=act
    )

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('solution_net.'):
                new_state_dict[key.replace('solution_net.', '')] = value
            elif not key.startswith('xyz_grid'):
                new_state_dict[key] = value
        net.load_state_dict(new_state_dict, strict=False)
    else:
        net.load_state_dict(checkpoint)

    print(f"Loaded checkpoint from: {ckpt_path}")
    return net


def create_pde(cfg):
    """创建 PDE 问题"""
    pde = EllipticPDE(
        sol=cfg.model.sol_func,
        nonli=cfg.model.nonli_func,
        dim=cfg.model.dim,
        beta=cfg.model.beta,
        bound=tuple(cfg.model.bound)
    )
    pde.padding = cfg.data.padding
    return pde


def compute_solution_and_error(pde, net, n_points=200):
    """计算数值解、精确解和误差"""
    bound = pde.bound
    x = np.linspace(bound[0], bound[1], n_points)
    y = np.linspace(bound[0], bound[1], n_points)
    X, Y = np.meshgrid(x, y)

    xyz_plot = torch.tensor(np.vstack([X.ravel(), Y.ravel()]).T, dtype=torch.float32)
    if pde.dim >= 2:
        xyz_plot = F.pad(xyz_plot, (0, pde.dim - 2), "constant", pde.padding)

    net.eval()
    device = next(net.parameters()).device
    with torch.no_grad():
        xyz_plot = xyz_plot.to(device)
        u_nn = net(xyz_plot)
    u_nn = u_nn.reshape(X.shape).detach().cpu().numpy()

    u_exact = pde.u_exact(xyz_plot)
    u_exact = u_exact.reshape(X.shape).cpu().numpy()

    error = np.abs(u_exact - u_nn)
    return X, Y, u_nn, u_exact, error


# ============================================================
# 任务 1 & 2: 读取 TensorBoard 日志并绘制 error-epoch 曲线
# ============================================================
def plot_error_epoch_curves():
    """绘制 5 个不同样本量下的 error-epoch 曲线（光滑化）"""
    print("\n" + "="*60)
    print("任务 1 & 2: 读取 TensorBoard 日志并绘制 error-epoch 曲线")
    print("="*60)

    fig, ax = plt.subplots(figsize=(12, 7))

    all_data = {}

    for sample_size, color in zip(SAMPLE_SIZES, COLORS):
        exp_key = str(sample_size)
        exp_info = EXPERIMENTS[exp_key]
        log_dir = exp_info['log_dir']

        if not os.path.exists(log_dir):
            print(f"Warning: Log directory {log_dir} not found")
            continue

        try:
            ea = load_tensorboard_data(log_dir)
            steps, values = extract_scalar_data(ea, 'val_error')

            # step 转换为 epoch (每 10 步为一个 epoch)
            epochs = np.array(steps) / 10
            values = np.array(values)

            # 光滑化处理
            values_smooth = uniform_filter1d(values, size=15, mode='nearest')

            # 保存数据
            all_data[sample_size] = {'epochs': epochs, 'errors': values, 'errors_smooth': values_smooth}

            # 绘制光滑后的曲线
            ax.plot(epochs, values_smooth, label=f'n={sample_size:,}', linewidth=1.5, color=color, linestyle='-', alpha=0.7)

            # 打印统计信息
            if len(values) > 0:
                print(f"\nn={sample_size:,}:")
                print(f"  初始误差：{values[0]:.6e}")
                print(f"  最终误差：{values[-1]:.6e}")
                print(f"  最小误差：{np.min(values):.6e} (Epoch {np.argmin(values)})")

        except Exception as e:
            print(f"Error loading {log_dir}: {e}")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation Error", fontsize=12)
    ax.set_title("Validation Error vs Epoch", fontsize=14)
    ax.legend(title="Sample Size", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_yscale('log')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'eq5_n_error_epoch_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n已保存：{save_path}")
    plt.close(fig)

    return all_data


# ============================================================
# 任务 3: 绘制解和误差热力图
# ============================================================

def plot_heatmaps():
    """绘制五个参数下的解的热力图以及误差的热力图"""
    print("\n" + "="*60)
    print("任务 3: 绘制解和误差热力图")
    print("="*60)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.subplots_adjust(hspace=0.16, wspace=0.05, left=0.05, right=0.90)

    # 存储每个实验的最佳误差和数值解范围（用于统一 colorbar）
    best_errors = {}
    u_nn_all = {}
    error_all = {}

    for idx, sample_size in enumerate(SAMPLE_SIZES):
        exp_key = str(sample_size)
        exp_info = EXPERIMENTS[exp_key]
        ckpt_dir = exp_info['ckpt_dir']

        # 获取最佳 checkpoint
        ckpt_path = get_best_checkpoint(ckpt_dir)
        if not ckpt_path:
            print(f"Warning: No checkpoint found for {ckpt_dir}")
            continue

        # 从文件名中提取 val_error
        try:
            error_str = ckpt_path.split('val_error=')[1].split('.ckpt')[0]
            best_error = float(error_str.replace('e-0', 'e-').replace('e-', 'e-'))
            best_errors[sample_size] = best_error
        except:
            best_errors[sample_size] = None

        # 创建临时配置（用于加载模型）
        from ml_collections import ConfigDict
        cfg = ConfigDict()
        cfg.model = ConfigDict()
        cfg.model.dim = 2
        cfg.model.sol_func = 'onepeak'
        cfg.model.nonli_func = 'inverse'
        cfg.model.beta = []
        cfg.model.bound = [-0.0, 1.0]
        cfg.net = ConfigDict()
        cfg.net.depth = 3
        cfg.net.width = 80
        cfg.net.act = 'ReLU6p'
        cfg.data = ConfigDict()
        cfg.data.padding = 0.5

        # 加载模型
        try:
            net = load_checkpoint(ckpt_path, cfg)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            net = net.to(device)
            net.eval()

            # 创建 PDE
            pde = create_pde(cfg)

            # 计算解和误差
            X, Y, u_nn, u_exact, error = compute_solution_and_error(pde, net)

            # 存储用于统一 colorbar 范围
            u_nn_all[sample_size] = u_nn
            error_all[sample_size] = error

        except Exception as e:
            print(f"Error processing {ckpt_path}: {e}")

    # 计算统一的 colorbar 范围
    if u_nn_all:
        vmin_u = min(np.min(u) for u in u_nn_all.values())
        vmax_u = max(np.max(u) for u in u_nn_all.values())
    if error_all:
        vmin_e = 0
        vmax_e = 0.015

    # 绘制热力图
    for idx, sample_size in enumerate(SAMPLE_SIZES):
        if sample_size not in u_nn_all:
            continue

        u_nn = u_nn_all[sample_size]
        error = error_all[sample_size]
        best_error = best_errors.get(sample_size)

        # 第一行：数值解
        im0 = axes[0, idx].pcolormesh(X, Y, u_nn, cmap='coolwarm', shading='auto',
                                       vmin=vmin_u, vmax=vmax_u)
        axes[0, idx].set_title(f'n={sample_size:,}', fontsize=10)
        axes[0, idx].set_xlabel('x')
        axes[0, idx].set_ylabel('y')
        axes[0, idx].set_aspect('equal')
        axes[0, idx].tick_params(labelsize=8)

        # 第二行：误差
        im2 = axes[1, idx].pcolormesh(X, Y, error, cmap='inferno', shading='auto',
                                       vmin=vmin_e, vmax=vmax_e)
        # error_title = f'Error: {best_error:.2e}' if best_error else 'Error'
        # axes[1, idx].set_title(error_title, fontsize=10)
        axes[1, idx].set_xlabel('x')
        axes[1, idx].set_ylabel('y')
        axes[1, idx].set_aspect('equal')
        axes[1, idx].tick_params(labelsize=8)

    # 添加行标签（纵向排列）
    fig.text(0.02, 0.7, 'Network\nSolution', fontsize=12, va='center', ha='center',
             fontweight='bold', rotation='vertical')
    fig.text(0.02, 0.3, 'Absolute\nError', fontsize=12, va='center', ha='center',
             fontweight='bold', rotation='vertical')

    # 统一的 colorbar（只在最右边显示）
    cbar_ax_u = fig.add_axes([0.92, 0.53, 0.015, 0.35])
    cbar_ax_e = fig.add_axes([0.92, 0.11, 0.015, 0.35])

    fig.colorbar(plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin_u, vmax_u)),
                 cax=cbar_ax_u, label='u')
    fig.colorbar(plt.cm.ScalarMappable(cmap='inferno', norm=plt.Normalize(vmin_e, vmax_e)),
                 cax=cbar_ax_e, label='|error|')

    save_path = os.path.join(OUTPUT_DIR, 'eq5_n_heatmaps_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n已保存：{save_path}")
    plt.close(fig)

    return best_errors


# ============================================================
# 任务 4: 绘制样本数 - 误差关系图
# ============================================================

def plot_sample_size_vs_error(best_errors):
    """绘制样本数 - 误差关系折线图（基于 checkpoint 数据）"""
    print("\n" + "="*60)
    print("任务 4: 绘制样本数 - 误差关系图")
    print("="*60)

    fig, ax = plt.subplots(figsize=(8, 5))

    # 准备数据：从 checkpoint 文件名中提取的最佳误差
    sample_sizes = []
    errors = []

    for size in SAMPLE_SIZES:
        if size in best_errors and best_errors[size] is not None:
            sample_sizes.append(size)
            errors.append(best_errors[size])

    print("\nCheckpoint 最佳误差:")
    for size, err in zip(sample_sizes, errors):
        print(f"  n={size:,}: val_error={err:.6e}")

    # 绘制样本数 - 误差关系图
    ax.plot(sample_sizes, errors, 'o-', linestyle='--', linewidth=2, markersize=8, color='#f59e0b', alpha=0.7)

    # 标注数据点
    for size, err in zip(sample_sizes, errors):
        ax.annotate(f'{err:.2e}', xy=(size, err),
                     xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax.set_xlabel("Sample Size (n)", fontsize=12)
    ax.set_ylabel("Validation Error", fontsize=12)
    ax.set_title("Validation Error vs Sample Size", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.3, color='gray', which='minor')
    ax.set_xscale('log')
    ax.set_yscale('log')

    save_path = os.path.join(OUTPUT_DIR, 'eq5_n_sample_size_vs_error.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n已保存：{save_path}")
    plt.close(fig)


# ============================================================
# 主函数
# ============================================================

def main():
    print("="*60)
    print("Eq5 可视化分析 - 不同样本量实验")
    print("="*60)
    print(f"输出目录：{OUTPUT_DIR}")

    # 任务 1: 读取日志并绘制 error-epoch 曲线
    all_data = plot_error_epoch_curves()

    # 任务 2: 绘制热力图
    best_errors = plot_heatmaps()

    # 任务 3: 绘制样本数 - 误差关系图
    plot_sample_size_vs_error(best_errors)

    print("\n" + "="*60)
    print(f"完成！所有图表已保存到 {OUTPUT_DIR}/")
    print("="*60)


if __name__ == '__main__':
    main()
