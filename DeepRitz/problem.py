import math
import sys
import torch
import torch.nn.functional
from scipy import interpolate
import numpy as np
from torch.func import functional_call, vmap, jacrev, hessian

class EllipticPDE():
    def __init__(self, sol='func1', nonli='Sigmoid', dim=2):
        self.sol = sol
        self.nonli = nonli
        self.bound_cond = 'Dirichlet'
        self.dim = dim

    def u_exact(self, xyz: torch.Tensor) -> torch.Tensor:
        dim = self.dim
        if self.sol == "func1":
            # u(x) = Π4xi(1-xi)
            ui = 4 * torch.mul(xyz, 1 - xyz)
            u_exact = torch.prod(ui, dim=1)
        elif self.sol == "func2":
            # u_raw = \bar{x}^2
            u_raw = torch.square(torch.mean(xyz, dim=1))
            mask_flat = u_raw < 0.3
            mask_quad = u_raw >= 0.3
            u = torch.zeros_like(u_raw)
            u[mask_quad] = u_raw[mask_quad]
            u[mask_flat] = 0.3
        return u_exact

    def h(self, xyz: torch.Tensor) -> torch.Tensor:
        if self.bound_cond == 'Dirichlet':
            return self.u_exact(xyz)

    def V_ex(self, xyz: torch.Tensor) -> torch.Tensor:
        if self.nonli == "Sigmoid":
            u = self.u_exact(xyz)
            V = torch.sigmoid(u)
            return V

    def V(self, u: torch.Tensor) -> torch.Tensor:
        if self.nonli == "Sigmoid":
            Vu = torch.sigmoid(u)
            return Vu

    def D2u_ex(self, xyz: torch.Tensor) -> torch.Tensor:
        dim = self.dim
        if self.sol == "func1":
            # u设为\prod 4xi(1-xi)
            # D2u[:, i]储存u对xi的二阶导
            D2u = torch.ones_like(xyz)
            ui = 4 * torch.mul(xyz, 1 - xyz)
            # -Δu = Σ (8*Πui(i≠j))  D2u
            for i in range(dim):
                for j in range(dim):
                    if j != i:
                        D2u[:, i] *= ui[:, j]
            return -8 * torch.sum(D2u, dim=1)           
        elif self.sol == "func2":
            u_raw = torch.square(torch.mean(xyz, dim=1))
            laplace_term = torch.zeros_like(u_raw)
            laplace_term[u_raw >= 0.3] = 2.0 / dim  # 仅在二次区域有值
        return D2u
        
    def g(self, xyz: torch.Tensor) -> torch.Tensor:
        g = -self.D2u_ex(xyz) + self.V_ex(xyz)
        return g.view(-1, 1)

    def res(self, u_nn, xyz: torch.Tensor) -> torch.Tensor:
        dim = self.dim
        if not xyz.requires_grad:
            xyz.requires_grad_(True)
        u = u_nn(xyz)
        u_grad = torch.autograd.grad(u, xyz, grad_outputs=torch.ones_like(u),
                                     create_graph=True, retain_graph=True)[0]
        u_laplace = torch.zeros_like(u)
        for i in range(dim):
            u_i = u_grad[:, i]
            u_ii = torch.autograd.grad(u_i, xyz, grad_outputs=torch.ones_like(u_i),
                                       create_graph=True, retain_graph=True)[0][:, i]
            u_laplace += u_ii.view(-1, 1)
        res = -u_laplace + self.V(u) - self.g(xyz)
        return res.detach()

    @staticmethod
    def boundary_indicator(xy: torch.Tensor) -> torch.Tensor:
        a = torch.add(torch.Tensor(
            [-1.0]), torch.sin(math.pi * xy[0:, 0]).min(torch.sin(math.pi * xy[0:, 1])))
        a = torch.pow(a.reshape(-1, 1), 100)
        return a
