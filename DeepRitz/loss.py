import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from DeepRitz.problem import EllipticPDE


class VarLoss(nn.Module):
    def __init__(self, problem: EllipticPDE):
        super().__init__()
        self.g = problem.g
        self.h = problem.h
        self.Fu = problem.Fu
        self.boundary_indicator = problem.boundary_indicator
        self.u_exact = problem.u_exact

    def _loss_var(self, xyz_in, xyz_bd, u_in, u_grad, u_bd, cfg) -> torch.Tensor:
        u_grad_squ = 0.5 * torch.mean(torch.sum(torch.square(u_grad), dim=1))
        g = self.g(xyz_in).to(cfg.device)
        gu = torch.mean(torch.mul(g, u_in))
        Fu = torch.mean(self.Fu(u_in))
        loss_in = u_grad_squ + Fu - gu

        h = self.h(xyz_bd).to(cfg.device)
        loss_bd = torch.mean(torch.square(u_bd - h))
        loss = loss_in + cfg.model.lambda_1 * loss_bd

        return loss, loss_in, loss_bd

    def forward(self, xyz_in, xyz_bd, u_in, u_grad, u_bd, cfg):
        return self._loss_var(xyz_in, xyz_bd, u_in, u_grad, u_bd, cfg)
