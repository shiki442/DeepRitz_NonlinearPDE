import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# 引入你的配置加载器 (假设你有 hydra 或类似配置)
from config.eq5 import get_config
from DeepRitz.data_pl import DeepRitzDataModule
from DeepRitz.model_pl import DeepRitzSystem
from DeepRitz.utils import get_latest_checkpoint

# import torch

# # 建议在 Lightning 训练开始前设置
# torch.set_float32_matmul_precision('high')


def main(cfg):
    # 1. 设置随机种子
    pl.seed_everything(134)

    # 2. 实例化 DataModule
    data_module = DeepRitzDataModule(cfg)

    # 3. 实例化 LightningModule
    model = DeepRitzSystem(cfg)

    # 4. 设置回调 (Checkpoint)
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.data.ckpt_dir,
        filename='model-{epoch:02d}-{val_error:.2e}',
        save_top_k=5,
        monitor='val_error',
        mode='min',
        save_last=True,
        every_n_epochs=cfg.verbose.ckpt_interval,
    )

    # 5. 设置 Logger
    logger = TensorBoardLogger(save_dir=cfg.data.log_dir)

    # 6. 初始化 Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.n_epochs,
        accelerator="gpu" if cfg.device == 'cuda' else "cpu",
        devices=1,  # 或 [0]
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,  # 因为 data steps 很少，建议设为 1
        enable_progress_bar=True,  # 关闭进度条以减少控制台输出
        # accumulate_grad_batches=5,  # 如果显存不足，可以开启梯度累积
    )

    # 7. 自动查找断点路径
    ckpt_path = get_latest_checkpoint(cfg.data.ckpt_dir)

    # 8. 开始训练 (如果 ckpt_path 不为 None，就会自动加载权重、优化器状态和 Epoch 计数)
    if ckpt_path:
        print(f"Resuming training from checkpoint: {ckpt_path}")
    else:
        print("No checkpoint found. Starting training from scratch.")

    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    # 模拟一个 cfg 对象，或者使用 Hydra/Argparse 加载
    cfg = get_config()
    main(cfg)
