# DeepRitz for Nonlinear Elliptic PDEs

基于深度学习的 Deep Ritz 方法实现，用于求解非线性椭圆型偏分方程。本项目是论文 "Analysis of Deep Ritz Methods for Semilinear Elliptic Equations" 的官方代码实现。

**论文链接**: [https://global-sci.com/index.php/nmtma/article/view/14474](https://global-sci.com/index.php/nmtma/article/view/14474)

## 目录

- [项目简介](#项目简介)
- [主要特性](#主要特性)
- [环境安装](#环境安装)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [方程与边值问题](#方程与边值问题)
- [网络架构](#网络架构)
- [超参数调优](#超参数调优)
- [可视化与日志](#可视化与日志)
- [项目结构](#项目结构)
- [引用](#引用)

## 项目简介

Deep Ritz 方法是一种基于变分原理的深度学习数值方法，用于求解偏微分方程的边值问题。该方法将 PDE 的变分形式转化为神经网络的优化问题，通过最小化能量泛函来训练神经网络逼近 PDE 的解。

本实现支持：
- 多种非线性项（Sigmoid, Sin, Exp, Poly, Inverse）
- 多种精确解类型（onepeak, twopeak, nondiff, liouville, yamabe）
- 任意维数问题（2D, 3D, 5D 等）
- 残差网络架构（ResNet-style）
- 自动断点恢复训练
- TensorBoard 可视化
- Optuna 超参数自动调优

## 主要特性

- **PyTorch Lightning 实现**: 模块化、可扩展的训练系统
- **离线数据集生成**: 使用 Sobol 序列生成拟随机采样点，预先生成并保存以加速训练
- **自动学习率调整**: 基于 ReduceLROnPlateau 策略，监控验证误差
- **自动断点恢复**: 自动查找并加载最新检查点继续训练
- **超参数优化**: 集成 Optuna，支持自动剪枝的贝叶斯优化
- **可视化**: TensorBoard 记录损失曲线、误差分析和解的可视化

## 环境安装

### 依赖

```bash
# 创建虚拟环境（可选）
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 安装依赖
pip install torch pytorch-lightning ml_collections optuna scipy matplotlib tensorboard
```

### GPU 支持

如需使用 GPU 训练，请先安装 CUDA 版本的 PyTorch：

```bash
# 访问 https://pytorch.org 获取适合你 CUDA 版本的安装命令
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## 快速开始

### 训练

```bash
# 使用默认配置（eq1: twopeak 解，Exp 非线性）训练
python main_pl.py

# SLURM 集群提交
sbatch run.sh
```

### 切换方程

修改 `main_pl.py` 或 `tune_pl.py` 中的配置导入：

```python
from config.eq1 import get_config  # 默认：twopeak, Exp 非线性
from config.eq2 import get_config  # nondiff 解，Sin 非线性
from config.eq3 import get_config  # Liouville 方程
from config.eq4 import get_config  # Yamabe 方程，Poly 非线性
```

### 超参数调优

```bash
# 运行 Optuna 搜索（默认使用 eq4 配置）
python tune_pl.py
```

## 配置说明

配置文件位于 `config/` 目录，每个方程对应一个配置文件。配置使用 `ml_collections.ConfigDict` 组织：

### 配置结构

```python
cfg.model      # PDE 问题参数
cfg.training   # 训练超参数
cfg.net        # 网络架构参数
cfg.verbose    # 日志与可视化设置
cfg.data       # 数据路径配置
cfg.device     # 计算设备 ('cuda' 或 'cpu')
```

### 关键参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `model.dim` | 空间维数 | 2 |
| `model.batch_in` | 内部点批次大小 | 500000 |
| `model.batch_bd` | 边界点批次大小 | 200000 |
| `model.lambda_1` | 边界惩罚系数 | 5000.0 |
| `model.sol_func` | 精确解类型 | 'twopeak' |
| `model.nonli_func` | 非线性类型 | 'Exp' |
| `model.beta` | 非线性参数 | [1.0, 1.0] |
| `training.n_epochs` | 训练轮数 | 100000 |
| `training.lr` | 初始学习率 | 0.001 |
| `training.patience` | 学习率衰减耐心值 | 10000 |
| `training.gamma` | 学习率衰减因子 | 0.5 |
| `net.depth` | 残差块数量 | 6 |
| `net.width` | 网络宽度 | 100 |
| `net.act` | 激活函数 | 'ReLU6p' |
| `verbose.plot_interval` | 可视化间隔（epoch）| 1000 |
| `verbose.ckpt_interval` | 检查点保存间隔 | 100 |

## 方程与边值问题

### 支持的精确解类型

| 类型 | 描述 | 维数 |
|------|------|------|
| `onepeak` | 单峰解：u(x) = ∏ 4xᵢ(1-xᵢ) | 任意 |
| `twopeak` | 双峰解：u(x) = sin(πx₁) × ∏ 4xᵢ(1-xᵢ) | 任意 |
| `nondiff` | 不可微解：u(x) = max(0, 0.5 - |x|²) | 任意 |
| `liouville` | Liouville 方程解：u(r) = C·log(1+r²) | 2D |
| `yamabe` | Yamabe 问题解：u(r) = k/(1+r²)^(d/2-1) | ≥3 |

### 支持的非线性类型

| 类型 | 数学形式 | 参数 beta |
|------|----------|-----------|
| `Sigmoid` | V(u) = σ(u) | - |
| `Sin` | V(u) = β₁·sin(β₂·u) | [β₁, β₂] |
| `Exp` | V(u) = β·exp(u) | [β] |
| `Poly` | V(u) = β₁·u^β₂ | [β₁, β₂] |
| `Inverse` | V(u) = 1/(1+u²) | - |

## 网络架构

### SolutionNet

采用残差网络架构：

```
输入 → Linear → [Block × N] → Linear → 输出
              ↑
         残差连接
```

### Block 结构

```
x → Dense1 → Act → Dense2 → Act → (+x) → 输出
```

### 支持的激活函数

| 名称 | 函数 | 说明 |
|------|------|------|
| `ReLU6p` | ReLU6(x)^1.5 | 默认激活函数 |
| `ReLUsq` | ReLU(x)² | 平方 ReLU |
| `ReLU6sq` | ReLU6(x)² | 平方 ReLU6 |
| `ReLU` | ReLU(x) | 标准 ReLU |
| `SiLU` | x·sigmoid(x) | 挤压线性单元 |
| `Tanh` | tanh(x) | 双曲正切 |

## 超参数调优

项目使用 Optuna 进行自动化超参数搜索：

```python
# tune_pl.py 中的搜索空间
lr          # 学习率：[2e-4, 2e-3] 对数刻度
width       # 网络宽度：[40, 150]
depth       # 网络深度：[2, 6]
activation  # 激活函数：['Tanh', 'ReLU6p']
lambda_1    # 边界惩罚系数：[2000, 5000]
```

### 剪枝策略

使用 `MedianPruner` 自动剪枝表现不佳的试验，减少计算资源浪费。

### 断点续传

Optuna 使用 SQLite 存储搜索结果，支持中断后继续：

```bash
# 自动加载之前的搜索结果继续优化
python tune_pl.py
```

## 可视化与日志

### TensorBoard

启动 TensorBoard 查看训练过程：

```bash
tensorboard --logdir ./logs/
```

### 可视化内容

- **损失曲线**: 总损失、内部损失、边界损失
- **误差分析**: 相对 L2 误差
- **残差**: PDE 残差的平均绝对值
- **解的可视化**: 精确解、数值解、绝对误差、残差分布
- **对角线对比**: 沿对角线的精确解与数值解对比

### 检查点

模型检查点保存在 `./checkpoints/` 目录：
- `last.ckpt`: 最新检查点（自动恢复用）
- `model-{epoch:02d}-{val_error:.2e}.ckpt`: 按误差保存的前 K 个检查点

## 项目结构

```
DeepRitz_NonlinearPDE/
├── main_pl.py              # PyTorch Lightning 训练入口
├── main.py                 # 传统训练入口（遗留）
├── tune_pl.py              # Optuna 超参数调优
├── plot_loss_curves.py     # 损失曲线可视化脚本
├── plot_checkpoint.py      # 检查点可视化脚本
├── read_tb_logs.py         # TensorBoard 日志读取工具
├── visualize_n.py          # 解可视化脚本
├── visualize_w.py          # 权重可视化脚本
│
├── config/
│   ├── eq1.py              # 默认配置（twopeak, Exp）
│   ├── eq2.py              # nondiff 解，Sin 非线性
│   ├── eq3.py              # Liouville 方程
│   ├── eq4.py              # Yamabe 方程，Poly 非线性
│   └── eq5.py              # 其他方程配置
│
├── DeepRitz/
│   ├── nn.py               # 神经网络架构（SolutionNet, Block）
│   ├── problem.py          # PDE 问题定义（EllipticPDE）
│   ├── loss.py             # 变分损失（VarLoss）
│   ├── data_pl.py          # PyTorch Lightning 数据模块
│   ├── model_pl.py         # PyTorch Lightning 模型
│   └── utils.py            # 工具函数（检查点、可视化、网格生成）
│
├── data/                   # 离线数据集（.pt 文件）
├── checkpoints/            # 模型检查点
├── logs/                   # TensorBoard 日志
├── dbeq4.sqlite            # Optuna 搜索结果数据库
│
├── CLAUDE.md               # Claude Code 项目指南
└── README.md               # 本文件
```

## 数据生成

训练数据使用 Sobol 序列生成准随机采样点：

- **内部点**: 在计算域内均匀采样
- **边界点**: 在各个边界面上采样

数据集预先生成并保存为 `.pt` 文件，避免每次训练重新生成。

```python
# 数据生成配置
cfg.data.file_path = './data/eq1_{batch_in}_{batch_bd}.pt'
```

**注意**: 修改 `batch_in` 或 `batch_bd` 后需要删除旧的数据文件以重新生成。

## 算法原理

Deep Ritz 方法基于以下变分原理：

对于非线性椭圆方程：

```
-Δu + V(u) = f,  in Ω
u = g,           on ∂Ω
```

对应的能量泛函为：

```
E(u) = ∫Ω [½|∇u|² + F(u) - fu] dx + λ∫∂Ω |u-g|² ds
```

其中 F(u) = ∫V(u)du，λ 是边界惩罚系数。

神经网络通过最小化离散化的能量泛函来逼近真解。

## 常见问题

### Q: 如何修改问题维度？

修改配置文件中的 `cfg.model.dim`，并确保选择适合的精确解类型（如 liouville 仅支持 2D）。

### Q: 训练不收敛怎么办？

1. 增大边界惩罚系数 `lambda_1`
2. 调整学习率或使用更小的初始学习率
3. 增加网络深度或宽度
4. 尝试不同的激活函数（Tanh 通常更平滑）

### Q: 如何继续训练？

程序会自动查找 `checkpoints/` 目录下的最新检查点并恢复训练。确保 `last.ckpt` 文件存在。

### Q: 如何修改边界条件？

目前实现的是 Dirichlet 边界条件。如需修改边界条件类型，需要修改 `DeepRitz/problem.py` 中的 `EllipticPDE` 类。

## 引用

如果使用本代码，请引用：

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

本项目代码仅供学术研究使用。
