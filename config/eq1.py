from ml_collections import ConfigDict
import torch


def get_config():
    cfg = ConfigDict()

    cfg.model = ConfigDict()
    cfg.model.dim = 2
    cfg.model.batch_in = 500000
    cfg.model.batch_bd = 200000
    cfg.model.nonli_func = 'exp'
    cfg.model.sol_func = 'onepeak'
    cfg.model.lambda_1 = 2000
    cfg.model.bound = [0.0, 1.0]
    cfg.model.beta = [1.0, 1.0]

    cfg.training = ConfigDict()
    cfg.training.n_epochs = 100000
    cfg.training.lr = 0.002
    cfg.training.patience = 50
    cfg.training.gamma = 0.5

    cfg.net = ConfigDict()
    cfg.net.depth = 2
    cfg.net.width = 120
    cfg.net.act = 'ReLU6p'  # 'ReLU6p' or 'SiLU' or 'ReLU' etc.

    cfg.verbose = ConfigDict()
    cfg.verbose.train_info = True
    cfg.verbose.print_interval = 1
    cfg.verbose.plot_interval = 100
    cfg.verbose.ckpt_interval = 100

    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg.data = ConfigDict()
    cfg.data.file_path = f'./data/eq1_{cfg.model.batch_in}_{cfg.model.batch_bd}.pt'
    cfg.data.ckpt_dir = './checkpoints/eq1/'
    cfg.data.log_dir = './logs/eq1/'
    cfg.data.n_steps = 10
    cfg.data.padding = 0.5
    return cfg
