import os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np

# TensorBoard 日志路径
log_dir = "./logs/eq3/lightning_logs/version_0"

# 创建事件累加器
ea = event_accumulator.EventAccumulator(
    log_dir,
    size_guidance={
        event_accumulator.SCALARS: 0,  # 0 表示读取所有数据
        event_accumulator.IMAGES: 0,
    }
)
ea.Reload()

print("=" * 60)
print("TensorBoard 日志读取")
print("=" * 60)

# -----------------------------------------------------------
# 1. 读取标量数据 (Scalars)
# -----------------------------------------------------------
print("\n可用的标量 tags:")
scalar_tags = ea.Tags()["scalars"]
for tag in scalar_tags:
    print(f"  - {tag}")

# 提取 loss 和 val_error 数据
def extract_scalar_data(ea, tag):
    """提取标量数据并返回 epoch 和 value 列表"""
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values

# 绘制 loss-epoch 折线图
plt.figure(figsize=(14, 5))

# Loss 曲线
plt.subplot(1, 2, 1)
loss_tags = [t for t in scalar_tags if "loss" in t.lower()]
for tag in loss_tags:
    steps, values = extract_scalar_data(ea, tag)
    plt.plot(steps, values, label=tag, linewidth=1)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.grid(True, alpha=0.3)

# Validation Error 曲线
plt.subplot(1, 2, 2)
error_tags = [t for t in scalar_tags if "error" in t.lower() or "err" in t.lower()]
for tag in error_tags:
    steps, values = extract_scalar_data(ea, tag)
    plt.plot(steps, values, label=tag, linewidth=1, color="red")
plt.xlabel("Step")
plt.ylabel("Error")
plt.title("Validation Error")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("./logs/eq3/loss_error_plot.png", dpi=150)
print("\n已保存: loss_error_plot.png")

# -----------------------------------------------------------
# 2. 导出数据到 CSV
# -----------------------------------------------------------
import csv

# 导出 loss 数据
with open("./logs/eq3/loss_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "loss_total", "loss_in", "loss_bd", "val_error", "residual"])

    # 获取所有相关数据
    data_dict = {}
    for tag in scalar_tags:
        steps, values = extract_scalar_data(ea, tag)
        data_dict[tag] = dict(zip(steps, values))

    # 获取所有 step
    all_steps = set()
    for tag_data in data_dict.values():
        all_steps.update(tag_data.keys())
    all_steps = sorted(all_steps)

    for step in all_steps:
        row = [step]
        for col_tag in ["loss/total", "loss/in", "loss/bd", "val_error", "loss/residual"]:
            row.append(data_dict.get(col_tag, {}).get(step, ""))
        writer.writerow(row)

print("已保存: loss_data.csv")

# -----------------------------------------------------------
# 3. 读取并保存图像 (Solution 热力图)
# -----------------------------------------------------------
print("\n可用的图像 tags:")
image_tags = ea.Tags().get("images", [])
for tag in image_tags:
    print(f"  - {tag}")

if image_tags:
    # 获取最后一个 epoch 的图像
    for tag in image_tags:
        events = ea.Images(tag)
        if events:
            # 保存最后一张图像
            last_event = events[-1]
            img_data = last_event.encoded_image_string

            # 解码并保存
            from PIL import Image
            import io

            img = Image.open(io.BytesIO(img_data))
            img.save(f"./logs/eq3/{tag.replace('/', '_')}_last.png")
            print(f"已保存图像：{tag}_last.png")

            # 也保存第一张图像用于对比
            if len(events) > 1:
                first_event = events[0]
                img_first = Image.open(io.BytesIO(first_event.encoded_image_string))
                img_first.save(f"./logs/eq3/{tag.replace('/', '_')}_first.png")
                print(f"已保存初始图像：{tag}_first.png")

# -----------------------------------------------------------
# 4. 打印关键统计信息
# -----------------------------------------------------------
print("\n" + "=" * 60)
print("关键统计信息")
print("=" * 60)

for tag in ["loss/total", "val_error", "loss/residual"]:
    if tag in scalar_tags:
        steps, values = extract_scalar_data(ea, tag)
        if values:
            print(f"\n{tag}:")
            print(f"  初始值：{values[0]:.6e}")
            print(f"  最终值：{values[-1]:.6e}")
            print(f"  最小值：{np.min(values):.6e} (step {steps[np.argmin(values)]})")
            print(f"  最大值：{np.max(values):.6e}")
            print(f"  数据点数量：{len(values)}")

print("\n完成！")
