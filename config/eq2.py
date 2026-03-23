from ml_collections import ConfigDict
import torch
import math


def get_config():
    cfg = ConfigDict()

    cfg.model = ConfigDict()
    cfg.model.dim = 5
    cfg.model.batch_in = 1500000
    cfg.model.batch_bd = 500000
    cfg.model.nonli_func = 'Sin'
    cfg.model.sol_func = 'nondiff'
    cfg.model.lambda_1 = 2500.0
    cfg.model.bound = [0.0, 1.0]
    cfg.model.beta = [2.0, 0.5 * math.pi]

    cfg.training = ConfigDict()
    cfg.training.n_epochs = 100000
    cfg.training.lr = 0.001
    cfg.training.patience = 10000
    cfg.training.gamma = 0.5
    cfg.training.betas = (0.9, 0.999)

    cfg.net = ConfigDict()
    cfg.net.depth = 5
    cfg.net.width = 70
    cfg.net.act = "ReLU6p"

    cfg.verbose = ConfigDict()
    cfg.verbose.train_info = True
    cfg.verbose.print_interval = 1
    cfg.verbose.plot_interval = 1000
    cfg.verbose.ckpt_interval = 100

    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg.data = ConfigDict()
    cfg.data.file_path = f'./data/eq2_{cfg.model.batch_in}_{cfg.model.batch_bd}.pt'
    cfg.data.ckpt_dir = './checkpoints/eq2/'
    cfg.data.log_dir = './logs/eq2/'
    cfg.data.n_steps = 10
    cfg.data.padding = 0.0
    return cfg
