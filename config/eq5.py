from ml_collections import ConfigDict
import torch


def get_config():
    cfg = ConfigDict()

    cfg.model = ConfigDict()
    cfg.model.dim = 2
    cfg.model.batch_in = 50000
    cfg.model.batch_bd = 20000
    cfg.model.nonli_func = 'inverse'
    cfg.model.sol_func = 'onepeak'
    cfg.model.lambda_1 = 4000.0
    cfg.model.bound = [-0.0, 1.0]
    cfg.model.beta = []

    cfg.training = ConfigDict()
    cfg.training.n_epochs = 5000
    cfg.training.lr = 0.0005153322505956383
    cfg.training.patience = 1000
    cfg.training.gamma = 0.5
    cfg.training.betas = (0.9, 0.99)

    cfg.net = ConfigDict()
    cfg.net.depth = 3
    cfg.net.width = 80
    cfg.net.act = "ReLU6p"

    cfg.verbose = ConfigDict()
    cfg.verbose.train_info = True
    cfg.verbose.print_interval = 1
    cfg.verbose.plot_interval = 500
    cfg.verbose.ckpt_interval = 100

    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg.data = ConfigDict()
    cfg.data.file_path = f'./data/eq5_{cfg.model.batch_in}_{cfg.model.batch_bd}.pt'
    cfg.data.ckpt_dir = './checkpoints/eq5/'
    cfg.data.log_dir = './logs/eq5/'
    cfg.data.n_steps = 10
    cfg.data.padding = 0.5
    return cfg
