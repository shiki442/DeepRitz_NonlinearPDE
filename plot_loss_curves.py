import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

# 读取 CSV 数据
steps = []
loss_total = []
loss_in = []
loss_bd = []
val_error = []
residual = []

with open("./logs/eq4/loss_data.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        steps.append(int(row["step"]))
        loss_total.append(float(row["loss_total"]) if row["loss_total"] else np.nan)
        loss_in.append(float(row["loss_in"]) if row["loss_in"] else np.nan)
        loss_bd.append(float(row["loss_bd"]) if row["loss_bd"] else np.nan)
        val_error.append(float(row["val_error"]) if row["val_error"] else np.nan)
        residual.append(float(row["residual"]) if row["residual"] else np.nan)

MAX_STEP = 20000

# 转换为 numpy 数组
steps = np.array(steps[:MAX_STEP])
loss_total = np.array(loss_total[:MAX_STEP])
loss_in = np.array(loss_in[:MAX_STEP])
loss_bd = np.array(loss_bd[:MAX_STEP])
val_error = np.array(val_error[:MAX_STEP])
residual = np.array(residual[:MAX_STEP])

# 参数设置
N_STEPS_PER_EPOCH = 10  # 每个 epoch 有 10 个 step
N_TOTAL_STEPS = len(steps)  # 1000000
N_EPOCHS = N_TOTAL_STEPS // N_STEPS_PER_EPOCH  # 100000 个 epoch

# val_error 每 10 步记录一次（在每个 epoch 结束时），所以有 100000 个值
# 提取 val_error 的有效数据（非 nan）
valid_mask = ~np.isnan(val_error)
val_error_valid = val_error[valid_mask]
val_epochs = np.arange(1, len(val_error_valid) + 1)  # 1, 2, 3, ..., N_EPOCHS

# 将 step 转换为 epoch (0, 1, 2, ..., 99999)
epochs_full = steps / N_STEPS_PER_EPOCH

# 平滑窗口大小（以数据点为单位）
LOSS_SMOOTH_WINDOW = 60  # loss 平滑窗口（500 个 step = 50 个 epoch）
VAL_SMOOTH_WINDOW = 10   # val_error 平滑窗口（50 个 epoch）

# 对 loss 进行平滑处理
loss_total_smooth = uniform_filter1d(loss_total, size=LOSS_SMOOTH_WINDOW, mode='nearest')
loss_in_smooth = uniform_filter1d(loss_in, size=LOSS_SMOOTH_WINDOW, mode='nearest')
loss_bd_smooth = uniform_filter1d(loss_bd, size=LOSS_SMOOTH_WINDOW, mode='nearest')

# 对 val_error 进行平滑处理
val_error_smooth = uniform_filter1d(val_error_valid, size=VAL_SMOOTH_WINDOW, mode='nearest')

# 重新采样 loss 到每个 epoch 一个数据点（取每个 epoch 的第一个 step）
epoch_indices = np.arange(0, N_TOTAL_STEPS, N_STEPS_PER_EPOCH)
epochs_sampled = epochs_full[epoch_indices]
loss_total_sampled = loss_total_smooth[epoch_indices]
loss_in_sampled = loss_in_smooth[epoch_indices]
loss_bd_sampled = loss_bd_smooth[epoch_indices]

# 绘图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ----- 左图: Loss 曲线 -----
ax1 = axes[0]
eps = 2.e-4
ax1.plot(epochs_sampled, loss_total_sampled - np.nanmin(loss_total_sampled) + eps, label=r"$|\mathcal{J}_{total} - \mathcal{J}_{min}| +\varepsilon$", linewidth=1.5, linestyle='-', color='#3b82f6', alpha=0.7)
ax1.plot(epochs_sampled, np.abs(loss_in_sampled - np.nanmin(loss_in_sampled[100:])) + eps, label=r"$|\mathcal{J}_{in} - \mathcal{J}_{min}| + \varepsilon$", linewidth=1.5, linestyle='-', color='#10b981', alpha=0.7)
ax1.plot(epochs_sampled, loss_bd_sampled, label=r"$\mathcal{J}_{bdy}$", linewidth=1.5, linestyle='-', color='#f59e0b', alpha=0.7)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training Loss vs Epoch")
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.3, color='gray')
ax1.set_yscale("log")

# ----- 右图: Validation Error 曲线 -----
ax2 = axes[1]
ax2.plot(val_epochs, val_error_smooth, label="Validation Error", linewidth=1.5, color='#ef4444', alpha=0.7)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Relative Error")
ax2.set_title("Validation Error vs Epoch")
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.3, color='gray')
ax2.set_yscale("log")  # 对数坐标更好地展示下降趋势

# 标注最小值点
min_idx = np.argmin(val_error_smooth)
ax2.annotate(
    f"Min: {val_error_smooth[min_idx]:.2e}\nEpoch: {min_idx + 1}",
    xy=(min_idx + 1, val_error_smooth[min_idx]),
    xytext=(min_idx + 1, val_error_smooth[min_idx] * 3),
    fontsize=9,
    arrowprops=dict(arrowstyle="->", color="black", alpha=0.6)
)

# plt.tight_layout()
plt.savefig("./logs/eq4/loss_val_error_curves.png", dpi=150, bbox_inches="tight")
print(f"已保存图表：./logs/eq4/loss_val_error_curves.png")

# 打印统计信息
print("\n" + "=" * 60)
print("Loss 数据统计")
print("=" * 60)
print(f"Total Loss:")
print(f"  初始：{loss_total_sampled[0]:.6e}")
print(f"  最终：{loss_total_sampled[-1]:.6e}")
print(f"  最小：{np.nanmin(loss_total_sampled):.6e}")

print(f"\nValidation Error 统计")
print("=" * 60)
print(f"  初始：{val_error_smooth[0]:.6e}")
print(f"  最终：{val_error_smooth[-1]:.6e}")
print(f"  最小：{val_error_smooth.min():.6e} (Epoch {np.argmin(val_error_smooth) + 1})")
print(f"  数据点数：{len(val_error_smooth)}")
