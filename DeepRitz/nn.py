import torch
import torch.nn as nn
from torch import autograd
from torch.func import vmap, grad, jacrev


def gradient(outputs, inputs):
    grad = autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True, only_inputs=True)
    return grad[0]


def get_act(act_name):
    if act_name == 'ReLU6p':
        def activation(inputs):
            return torch.pow(nn.ReLU6()(inputs), 1.5)
        return activation
    elif act_name == 'ReLUsq':
        def activation(inputs):
            return torch.square(nn.ReLU()(inputs))
        return activation
    elif act_name == 'ReLU6sq':
        def activation(inputs):
            return torch.square(nn.ReLU6()(inputs))
        return activation
    elif act_name == 'ReLU':
        return nn.ReLU()
    elif act_name == 'SiLU':
        return nn.SiLU()
    elif act_name == 'Tanh':
        return nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation function: {act_name}")


@torch.no_grad()
def _init_params(m):
    if isinstance(m, nn.Linear):
        torch.manual_seed(1234)
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


class Block(nn.Module):
    def __init__(self, n_features, width, act):
        super(Block, self).__init__()
        self.dense1 = nn.Linear(in_features=n_features, out_features=width)
        self.dense2 = nn.Linear(in_features=width, out_features=n_features)
        self.act = get_act(act)

    def forward(self, x) -> torch.Tensor:
        residual = self.dense1(x)
        residual = self.act(residual)
        residual = self.dense2(residual)
        residual = self.act(residual)
        x = torch.add(x, residual)
        return x


class SolutionNet(nn.Module):
    def __init__(self, in_features, out_features, block_width, n_blocks, act='ReLU6p'):
        super(SolutionNet, self).__init__()
        self.module_list = nn.ModuleList()
        # append input layer
        self.module_list.append(nn.Linear(in_features=in_features, out_features=block_width))
        # append blocks
        for _ in range(n_blocks):
            self.module_list.append(Block(n_features=block_width, width=block_width, act=act))
        # append output layer
        self.module_list.append(nn.Linear(in_features=block_width, out_features=out_features))
        # initialize parameters
        self.apply(_init_params)
        self.act = get_act(act)

    def forward(self, x) -> torch.Tensor:
        n_modules = len(self.module_list)
        x = self.act(self.module_list[0](x))
        for idx in range(1, n_modules - 1):
            x = self.module_list[idx](x)
        x = self.module_list[-1](x)
        return x

    def grad_u(self, xy) -> torch.Tensor:
        xy_clone = xy.clone().detach().requires_grad_(True)
        u = self.forward(xy_clone)
        return gradient(u, xy_clone)

    def grad_u_single(self, xy):
        # 使用 functional_call 调用模型
        params = dict(self.named_parameters())
        # u = torch.func.functional_call(self, params, (xy,))
        # 计算导数
        du_dx = jacrev(lambda x: torch.func.functional_call(self, params, (x,)))(xy)
        return du_dx

    def grad_u_func(self, xy) -> torch.Tensor:
        return vmap(self.grad_u_single)(xy)
