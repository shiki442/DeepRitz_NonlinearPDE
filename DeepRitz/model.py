import os
import sys

import numpy as np
import scipy.io as sio
import torch
from torch import nn, optim

from DeepRitz.matplot import Result
from DeepRitz.loss import VarLoss
from DeepRitz.nn import SolutionNet
from DeepRitz.problem import EllipticPDE
import DeepRitz.utils as utils

from torch.utils.tensorboard import SummaryWriter


class Model:
    def __init__(self, cfg, net: nn.Module = None):

        self.cfg = cfg
        self.device = cfg.device
        if net is None:
            self.solution_net = SolutionNet(in_features=dim, out_features=1, block_width=block_width, n_blocks=n_blocks).to(self.device)
        else:
            self.solution_net = net
        self.problem = EllipticPDE(cfg.model.sol_func, cfg.model.nonli_func, cfg.model.dim)
        self.loss_func = VarLoss(self.problem).to(self.device)
        self.optimizer = optim.Adam(params=self.solution_net.parameters(), lr=cfg.training.lr)

        self.writer = SummaryWriter(log_dir="./logs/" + cfg.model.sol_func)
        self.ckpt_dir = "./checkpoints/" + cfg.model.sol_func
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.verbose = cfg.verbose.train_info
        self.xyz_plot = None  # 用于绘图的网格点

    def _train_step(self, xyz_in, xyz_bd, cfg, epoch):

        def closure():
            self.optimizer.zero_grad()
            u_in = self.solution_net(xyz_in)
            u_bd = self.solution_net(xyz_bd)
            u_grad = self.solution_net.grad_u(xyz_in)
            loss, loss_in, loss_bd = self.loss_func(xyz_in, xyz_bd, u_in, u_grad, u_bd, cfg)
            loss.backward()
            nn.utils.clip_grad_norm_(self.solution_net.parameters(), max_norm=1.0e5, norm_type=1)

            self.writer.add_scalar('Loss/Total Loss', loss.item(), epoch)
            self.writer.add_scalar('Loss/Interior Loss', loss_in.item(), epoch)
            self.writer.add_scalar('Loss/Boundary Loss', loss_bd.item(), epoch)
            return loss

        return self.optimizer.step(closure)

    def train(self):
        torch.manual_seed(1234)
        dim = self.cfg.model.dim
        xyz_grid = utils.precompute_grid(dim, 'random', self.cfg.model.sol_func).to(self.device)
        solution_exact = self.problem.u_exact(xyz_grid).to(self.device)

        for epoch in range(self.cfg.training.n_epochs):
            # torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)
            # torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.cfg.training.n_epochs, eta_min=1e-7)
            n = self.cfg.model.batch_in
            xyz_in = torch.rand(n, dim)
            if self.cfg.model.sol_func == "func2":
                xyz_in = xyz_in * 2 - 1
            xyz_in = xyz_in.to(self.device)

            m = self.cfg.model.batch_bd // (2 * dim)
            xyz_bd = torch.rand(2 * dim * m, dim)
            for i in range(dim):
                # 加入边界点
                xyz_bd[i * m : (i + 1) * m, i] = 1.0
                xyz_bd[(i + dim) * m : (i + dim + 1) * m, i] = -0.0
            xyz_bd = xyz_bd.to(self.device)

            self.solution_net.train(mode=True)
            loss = self._train_step(xyz_in, xyz_bd, self.cfg, epoch)  # 训练
            self.solution_net.train(mode=False)
            self._train_info(loss, xyz_grid, solution_exact, epoch)
        self.writer.close()
        return loss.item()

    def _relative_err(self, u_nn, u_exact, ord=2):
        diff_norm = torch.linalg.norm(u_nn.view(-1) - u_exact, ord=ord)
        target_norm = torch.linalg.norm(u_exact, ord=ord)
        return diff_norm / target_norm if target_norm != 0 else 0

    def _train_info(self, loss, xyz_grid, solution_exact, epoch):
        u_nn = self.solution_net(xyz_grid)
        err = self._relative_err(u_nn, solution_exact)
        res_nn = self.problem.res(self.solution_net, xyz_grid)
        res = torch.linalg.norm(res_nn, ord=2)
        self.writer.add_scalar('Error/Solution Error', err, epoch)
        self.writer.add_scalar('Error/Residual', res, epoch)

        if self.verbose:
            if (epoch + 1) % self.cfg.verbose.print_interval == 0 or epoch == 0:
                # :>5d 表示右对齐，占5个字符宽
                # :10.4e 表示科学计数法，占10个字符宽
                print(
                    f"Epoch: [{epoch + 1:>5d}/{self.cfg.training.n_epochs}] | "
                    f"Loss: {loss:10.4e} | "
                    f"Sol_Err: {err:10.4e} | "
                    f"Res_Norm: {res:10.4e}"
                )

        if (epoch + 1) % self.cfg.verbose.plot_interval == 0 or epoch == 0:
            utils.plot_pde_results(self.problem, self.solution_net, self.writer, epoch)

        if (epoch + 1) % self.cfg.verbose.ckpt_interval == 0 or epoch == 0:
            self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch):
        ckpt_path = os.path.join(self.ckpt_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(self.solution_net.state_dict(), ckpt_path)

    def load_checkpoint(self, epoch):
        ckpt_path = os.path.join(self.ckpt_dir, f"model_epoch_{epoch+1}.pth")
        self.solution_net.load_state_dict(torch.load(ckpt_path, map_location=self.device))
