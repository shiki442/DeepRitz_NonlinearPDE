import ml_collections
import torch
import pytorch_lightning as pl
from torch import optim
import DeepRitz.utils as utils
from DeepRitz.nn import SolutionNet
from DeepRitz.problem import EllipticPDE
from DeepRitz.loss import VarLoss

# torch.serialization.add_safe_globals([ml_collections.config_dict.config_dict.ConfigDict])

class DeepRitzSystem(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        # self.save_hyperparameters() # 自动保存 cfg 到 hparams.yaml
        self.cfg = cfg
        
        # 1. 初始化网络
        dim = cfg.model.dim
        # 假设这些参数在 cfg 中，或者你可以硬编码/从 utils 获取
        block_width = cfg.net.width if hasattr(cfg.net, 'width') else 20
        n_blocks = cfg.net.depth if hasattr(cfg.net, 'depth') else 4
        
        self.solution_net = SolutionNet(
            in_features=dim, 
            out_features=1, 
            block_width=block_width, 
            n_blocks=n_blocks
        )
        
        # 2. 初始化 PDE 问题和损失函数
        self.problem = EllipticPDE(cfg.model.sol_func, cfg.model.nonli_func, cfg.model.dim, cfg.model.beta, cfg.model.bound)
        self.loss_func = VarLoss(self.problem)
        
        # 预计算用于评估的网格 (注册为 buffer 以便自动移动到 device)
        xyz_grid = utils.precompute_grid(dim, 'grid', bound=cfg.model.bound, padding=cfg.data.padding)
        self.register_buffer('xyz_grid', xyz_grid)

    def forward(self, x):
        return self.solution_net(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.cfg.training.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.cfg.training.patience, factor=self.cfg.training.gamma, min_lr=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_error",
            },
    }

    def training_step(self, batch, batch_idx):
        # batch 是从 DataLoader 出来的，维度通常是 [1, N, dim]
        # 因为原代码 DataLoader batch_size=1, 所以需要 squeeze
        xyz_in, xyz_bd = batch
        xyz_in = xyz_in.squeeze(0) # [N_in, dim]
        xyz_bd = xyz_bd.squeeze(0) # [N_bd, dim]

        # 这里的 requires_grad 设置通常由 SolutionNet 内部处理，
        # 但如果是 DeepRitz，通常需要输入坐标的梯度。
        # 确保 xyz_in 开启了梯度追踪（如果 Dataset 里没开的话）
        if not xyz_in.requires_grad:
            xyz_in.requires_grad_(True)
            
        u_in = self.solution_net(xyz_in)
        u_bd = self.solution_net(xyz_bd)
        
        # 计算 u 对 x 的梯度
        u_grad = self.solution_net.grad_u(xyz_in)
        
        loss, loss_in, loss_bd = self.loss_func(xyz_in, xyz_bd, u_in, u_grad, u_bd, self.cfg)

        # 记录日志 (显示在进度条和 TensorBoard)
        self.log('loss/total', loss, prog_bar=True)
        self.log('loss/in', loss_in)
        self.log('loss/bd', loss_bd)

        return loss

    def on_train_epoch_end(self):
        """
        每个 Epoch 结束时计算精确解误差并绘图
        对应原代码中的 _train_info
        """
        # 获取当前 Epoch
        epoch = self.current_epoch
        
        # 1. 计算误差
        # self.xyz_grid 已经在正确的 device 上
        u_nn = self.solution_net(self.xyz_grid)
        solution_exact = self.problem.u_exact(self.xyz_grid)
        
        err = self._relative_err(u_nn, solution_exact)
        
        # 计算残差
        res_nn = self.problem.res(self.solution_net, self.xyz_grid)
        res = torch.mean(torch.abs(res_nn))

        # 记录 Metric
        self.log('val_error', err, prog_bar=True)
        self.log('loss/residual', res)

        # 2. 打印信息 (Lightning 的进度条已经包含 Loss 和 Error，这里可选择性打印)
        # if (epoch + 1) % self.cfg.verbose.print_interval == 0:
        #     print(f"\nEpoch {epoch}: Error: {err:.4e} | Res: {res:.4e}")

        # 3. 绘图 (使用 Lightning 的 logger)
        if (epoch + 1) % self.cfg.verbose.plot_interval == 0 or epoch == 0:
            # 只有当 Logger 是 TensorBoardLogger 时才绘图
            if hasattr(self.logger, 'experiment'):
                utils.plot_pde_results(self.problem, self.solution_net, self.logger.experiment, epoch, padding=self.cfg.data.padding)

    def _relative_err(self, u_nn, u_exact, ord=2):
        diff_norm = torch.linalg.norm(u_nn - u_exact, ord=ord)
        target_norm = torch.linalg.norm(u_exact, ord=ord)
        return diff_norm / target_norm if target_norm != 0 else 0