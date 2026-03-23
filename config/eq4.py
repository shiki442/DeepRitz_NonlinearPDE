from ml_collections import ConfigDict
import torch


def get_config():
    cfg = ConfigDict()

    cfg.model = ConfigDict()
    cfg.model.dim = 3
    cfg.model.batch_in = 1500000
    cfg.model.batch_bd = 500000
    cfg.model.nonli_func = 'Poly'
    cfg.model.sol_func = 'Yamabe'
    cfg.model.lambda_1 = 5000.0
    cfg.model.bound = [-0.0, 1.0]
    cfg.model.beta = [-1.0, (cfg.model.dim+2.0)  / (cfg.model.dim-2.0)]  # beta[1]=(d+2)/(d-2)

    cfg.training = ConfigDict()
    cfg.training.n_epochs = 5000
    cfg.training.lr = 0.0005
    cfg.training.patience = 1000
    cfg.training.gamma = 0.5
    cfg.training.betas = (0.9, 0.99)

    cfg.net = ConfigDict()
    cfg.net.depth = 5
    cfg.net.width = 50
    cfg.net.act = "Tanh"

    cfg.verbose = ConfigDict()
    cfg.verbose.train_info = True
    cfg.verbose.print_interval = 1
    cfg.verbose.plot_interval = 100
    cfg.verbose.ckpt_interval = 100

    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg.data = ConfigDict()
    cfg.data.file_path = f'./data/eq4_{cfg.model.batch_in}_{cfg.model.batch_bd}.pt'
    cfg.data.ckpt_dir = './checkpoints/eq4/'
    cfg.data.log_dir = './logs/eq4/'
    cfg.data.n_steps = 10
    cfg.data.padding = 0.5
    return cfg
