import math
import sys
import torch
import torch.nn.functional
from scipy import interpolate
import numpy as np
from torch.func import functional_call, vmap, jacrev, hessian


class EllipticPDE:
    def __init__(self, sol='onepeak', nonli='Sigmoid', dim=2, beta=None, bound=(0.0, 1.0)):
        self.sol = sol
        self.nonli = nonli
        self.bound = bound
        self.bound_cond = 'Dirichlet'
        self.dim = dim
        self.beta = beta

    def u_exact(self, xyz: torch.Tensor) -> torch.Tensor:
        dim = self.dim
        if self.sol.lower() == "onepeak":
            # u(x) = Π4xi(1-xi)
            ui = 4 * torch.mul(xyz, 1 - xyz)
            u_exact = torch.prod(ui, dim=1, keepdim=True)
        elif self.sol.lower() == "twopeak":
            # u(x) = sin(pi*x1)*Π4xi(1-xi) %(i=2-d)
            u1 = torch.sin(2 * math.pi * xyz[:, 0:1])
            u_2_d = 4 * torch.mul(xyz[:, 1:], 1 - xyz[:, 1:])
            u_exact = u1 * torch.prod(u_2_d, dim=1, keepdim=True)
        elif self.sol.lower() == "nondiff":
            u_exact = 0.5 - torch.sum(torch.square(xyz), dim=1, keepdim=True)
            mask_flat = u_exact > 1.0 / 3
            u_exact[mask_flat] = 1.0 / 3
        elif self.sol.lower() == "liouville":
            # u(x,y) = -2log(1+r^2), r = sqrt(x^2 + y^2)
            r2 = torch.sum(torch.square(xyz), dim=1, keepdim=True)
            u_exact = 0.25 * self.beta[0] * torch.log(1.0 + r2)
        elif self.sol.lower() == "yamabe":
            # lambda=1, x_0=0.3
            x0 = 0.3
            r2 = torch.sum(torch.square(xyz - x0), dim=1, keepdim=True)
            k = (dim * (dim - 2)) ** 0.5
            u_exact = (k / (1 + r2)) ** (dim / 2.0 - 1.0)
        return u_exact

    def h(self, xyz: torch.Tensor) -> torch.Tensor:
        if self.bound_cond.lower() == 'dirichlet':
            return self.u_exact(xyz)

    def V_ex(self, xyz: torch.Tensor) -> torch.Tensor:
        if self.nonli.lower() == "sigmoid":
            u = self.u_exact(xyz)
            V = torch.sigmoid(u)
        elif self.nonli.lower() == "sin":
            u = self.u_exact(xyz)
            V = self.beta[0] * torch.sin(self.beta[1] * u)
        elif self.nonli.lower() == "exp":
            u = self.u_exact(xyz)
            V = self.beta[0] * torch.exp(u)
        elif self.nonli.lower() == "poly":
            u = self.u_exact(xyz)
            V = self.beta[0] * torch.pow(u, self.beta[1])
        elif self.nonli.lower() == "inverse":
            u = self.u_exact(xyz)
            V = 1.0 / (1.0 + torch.square(u))
        else:
            V = torch.zeros_like(u)
        return V

    def V(self, u: torch.Tensor) -> torch.Tensor:
        if self.nonli.lower() == "sigmoid":
            Vu = torch.sigmoid(u)
        elif self.nonli.lower() == "sin":
            Vu = self.beta[0] * torch.sin(self.beta[1] * u)
        elif self.nonli.lower() == "exp":
            Vu = self.beta[0] * torch.exp(u)
        elif self.nonli.lower() == "poly":
            Vu = self.beta[0] * torch.pow(u, self.beta[1])
        elif self.nonli.lower() == "inverse":
            Vu = 1.0 / (1.0 + torch.square(u))
        else:
            Vu = torch.zeros_like(u)
        return Vu

    def Fu(self, u: torch.Tensor) -> torch.Tensor:
        if self.nonli.lower() == "sigmoid":
            SigmoidU = torch.sigmoid(u)
            Fu = u - torch.log(SigmoidU)
        elif self.nonli.lower() == "sin":
            Fu = -self.beta[0] / self.beta[1] * torch.cos(u)
        elif self.nonli.lower() == "exp":
            Fu = self.beta[0] * torch.exp(u)
        elif self.nonli.lower() == "poly":
            u = torch.abs(u)
            # s = torch.sgn(u)
            Fu = (self.beta[0] / (self.beta[1] + 1)) * torch.pow(u, self.beta[1] + 1)
        elif self.nonli.lower() == "inverse":
            # F(u) = ∫ 1/(1+s^2) ds = arctan(u)
            Fu = torch.atan(u)
        else:
            Fu = torch.zeros_like(u)
        return Fu

    def D2u_ex(self, xyz: torch.Tensor) -> torch.Tensor:
        dim = self.dim
        if self.sol.lower() == "onepeak":
            # u设为\prod 4xi(1-xi)
            # D2u[:, i]储存u对xi的二阶导
            D2u = torch.ones_like(xyz)
            ui = 4 * torch.mul(xyz, 1 - xyz)
            # -Δu = Σ (8*Πui(i≠j))  D2u
            for i in range(dim):
                for j in range(dim):
                    if j != i:
                        D2u[:, i] *= ui[:, j]
            return -8 * torch.sum(D2u, dim=1, keepdim=True)
        elif self.sol.lower() == "twopeak":
            u_exact = self.u_exact(xyz)
            laplace_1 = -((2 * math.pi) ** 2) * u_exact
            xi = xyz[:, 1:]
            second_deriv_multipliers = -8 / (4 * xi * (1 - xi) + 1e-12)
            laplace_2_d = torch.sum(u_exact * second_deriv_multipliers, dim=1, keepdim=True)
            D2u = laplace_1 + laplace_2_d
            return D2u
        elif self.sol.lower() == "nondiff":
            u_raw = 0.5 - torch.square(torch.mean(xyz, dim=1, keepdim=True))
            # D2u = torch.zeros_like(u_raw)
            # D2u[u_raw >= 1.0 / 6.0] = 2.0 / dim  # 仅在二次区域有值
            D2u = -2.0 * torch.ones_like(u_raw) * dim
            D2u[u_raw > 1.0 / 3] = 0.0
            return D2u
        elif self.sol.lower() == "liouville":
            r2 = torch.sum(torch.square(xyz), dim=1, keepdim=True)
            D2u = self.beta[0] / ((r2 + 1.0) ** 2)
            return D2u
        elif self.sol.lower() == "yamabe":
            r2 = torch.sum(torch.square(xyz), dim=1, keepdim=True)
            k = (dim * (dim - 2)) ** 0.5
            D2u = -((k / (1 + r2)) ** (dim / 2.0 - 1.0))
            return D2u

    def g(self, xyz: torch.Tensor) -> torch.Tensor:
        g = -self.D2u_ex(xyz) + self.V_ex(xyz)
        return g

    def res(self, u_nn, xyz: torch.Tensor) -> torch.Tensor:
        dim = self.dim
        if not xyz.requires_grad:
            xyz.requires_grad_(True)
        u = u_nn(xyz)
        u_grad = torch.autograd.grad(u, xyz, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_laplace = torch.zeros_like(u)
        for i in range(dim):
            u_i = u_grad[:, i]
            u_ii = torch.autograd.grad(u_i, xyz, grad_outputs=torch.ones_like(u_i), create_graph=True, retain_graph=True)[0][:, i]
            u_laplace += u_ii.view(-1, 1)
        # res = -u_laplace + self.V(u) - self.g(xyz)
        res = -u_laplace + self.V_ex(xyz) - self.g(xyz)
        return res.detach()

    @staticmethod
    def boundary_indicator(xy: torch.Tensor) -> torch.Tensor:
        a = torch.add(torch.Tensor([-1.0]), torch.sin(math.pi * xy[0:, 0]).min(torch.sin(math.pi * xy[0:, 1])))
        a = torch.pow(a.reshape(-1, 1), 100)
        return a
