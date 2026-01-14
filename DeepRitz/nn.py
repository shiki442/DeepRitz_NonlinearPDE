import torch
import torch.nn as nn
from torch import autograd


def gradient(outputs, inputs):
    grad = autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True, only_inputs=True)
    return grad[0]


def relu_pow(inputs, pow_deg=1.5):
    # return torch.pow(torch.relu(inputs), pow_deg)
    f = nn.ReLU6()
    return torch.pow(f(inputs), pow_deg)


@torch.no_grad()
def _init_params(m):
    if isinstance(m, nn.Linear):
        torch.manual_seed(1234)
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


class Block(nn.Module):
    def __init__(self, n_features, width):
        super(Block, self).__init__()
        self.dense1 = nn.Linear(in_features=n_features, out_features=width)
        self.dense2 = nn.Linear(in_features=width, out_features=n_features)

    def forward(self, x) -> torch.Tensor:
        residual = self.dense1(x)
        residual = relu_pow(residual)
        residual = self.dense2(residual)
        residual = relu_pow(residual)
        x = torch.add(x, residual)
        return x


class SolutionNet(nn.Module):
    def __init__(self, in_features, out_features, block_width, n_blocks):
        super(SolutionNet, self).__init__()
        self.module_list = nn.ModuleList()
        # append input layer
        self.module_list.append(nn.Linear(in_features=in_features, out_features=block_width))
        # append blocks
        for _ in range(n_blocks):
            self.module_list.append(Block(n_features=block_width, width=block_width))
        # append output layer
        self.module_list.append(nn.Linear(in_features=block_width, out_features=out_features))
        # initialize parameters
        self.apply(_init_params)

    def forward(self, x) -> torch.Tensor:
        n_modules = len(self.module_list)
        x = relu_pow(self.module_list[0](x))
        for idx in range(1, n_modules - 1):
            x = self.module_list[idx](x)
        x = self.module_list[-1](x)
        return x

    def grad_u(self, xy) -> torch.Tensor:
        xy_clone = xy.clone().detach().requires_grad_(True)
        u = self.forward(xy_clone)
        return gradient(u, xy_clone)
