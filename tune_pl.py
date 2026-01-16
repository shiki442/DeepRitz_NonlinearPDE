import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl
import os
import torch

# 引入你现有的模块
from config.eq3 import get_config 
from DeepRitz.data_pl import DeepRitzDataModule
from DeepRitz.model_pl import DeepRitzSystem

def objective(trial: optuna.trial.Trial):
    # 1. 获取基础配置
    cfg = get_config()

    # -----------------------------------------------------------
    # 2. 定义搜索空间 (在这里修改你想调的参数)
    # -----------------------------------------------------------
    
    # 学习率: 建议使用对数刻度
    cfg.training.lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    
    # 网络架构
    cfg.net.width = trial.suggest_int("width", 10, 50, step=5)
    cfg.net.depth = trial.suggest_int("depth", 2, 6)
    cfg.net.act = trial.suggest_categorical("activation", ["Tanh", "ReLU6p"])

    # 惩罚项系数 (lambda_1): DeepRitz 中边界惩罚非常关键
    if hasattr(cfg.model, 'lambda_1'):
        cfg.model.lambda_1 = trial.suggest_float("lambda_1", 100, 5000, log=True)

    # 注意：不要在调参中修改 batch_size 或 data 相关参数，
    # 因为你的 data.py 会生成固定文件名的 .pt 文件，修改这些参数需要重新生成数据。

    # -----------------------------------------------------------
    # 3. 实例化模型与数据
    # -----------------------------------------------------------
    pl.seed_everything(134) # 保持种子一致以确保比较公平，或者去掉以引入随机性

    print(f"\n[Trial {trial.number}] 开始训练 | 参数: {trial.params}")
    data_module = DeepRitzDataModule(cfg)
    model = DeepRitzSystem(cfg)

    # -----------------------------------------------------------
    # 4. 配置 Trainer 与 剪枝 (Pruning)
    # -----------------------------------------------------------
    
    # 定义 logger，为了避免产生海量日志文件夹，可以设为 False 或者指定临时目录
    # 这里我们设为 False，只通过 Optuna 追踪
    logger = False 

    # 使用 Optuna 的回调，监控 'val_error'
    # 如果训练初期的 val_error 下降太慢，Optuna 会抛出 TrialPruned 异常中止训练
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_error")

    trainer = pl.Trainer(
        max_epochs=cfg.training.n_epochs, # 或者为了搜索速度，设一个小一点的值
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        enable_checkpointing=False, # 搜索时不保存大量权重文件
        callbacks=[pruning_callback],
        enable_progress_bar=True,  # 关闭进度条以减少控制台输出
        log_every_n_steps=10
    )

    # -----------------------------------------------------------
    # 5. 开始训练
    # -----------------------------------------------------------
    try:
        trainer.fit(model, datamodule=data_module)
    except Exception as e:
        # 捕获可能的 NaN 错误或其他训练崩溃
        print(f"Trial failed with error: {e}")
        return float('inf')

    # -----------------------------------------------------------
    # 6. 返回目标指标
    # -----------------------------------------------------------
    # 从 callback_metrics 中获取最后一次 logged 的 val_error
    val_error = trainer.callback_metrics.get("val_error")
    
    if val_error is None:
        return float('inf')
        
    return val_error.item()

if __name__ == "__main__":
    # 创建 Study 对象
    study = optuna.create_study(
        study_name="deepritz_study",         # 给任务起个名
        storage="sqlite:///db.sqlite3",      # 必须指定 storage
        load_if_exists=True,                 # 支持断点续传
        direction="minimize",                # 最小化 error
        pruner=optuna.pruners.MedianPruner() # 自动剪枝策略
    )
    
    print("开始 Optuna 超参数搜索...")
    # n_trials 设置尝试的次数
    study.optimize(objective, n_trials=50, timeout=None)

    print("\n-----------------------------------------------------------")
    print("搜索结束!")
    print(f"最佳 Trial 编号: {study.best_trial.number}")
    print(f"最佳 Error: {study.best_value:.4e}")
    print("最佳参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("-----------------------------------------------------------")

    # 可视化 (可选，需要安装 plotly)
    # optuna.visualization.plot_optimization_history(study).show()
    # optuna.visualization.plot_param_importances(study).show()