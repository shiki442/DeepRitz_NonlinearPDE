from matplotlib.pyplot import step
from DeepRitz.model import Model
from DeepRitz.matplot import Result
from DeepRitz.nn import SolutionNet
from config.eq1 import get_config
import torch
import argparse
import os

cfg = get_config()

# ----------------------------------------loading trained model----------------------------------------
net = SolutionNet(in_features=cfg.model.dim, out_features=1, block_width=cfg.net.width, n_blocks=cfg.net.depth)
net.to(cfg.device)

# ----------------------------------------trianing----------------------------------------
model = Model(cfg, net)
model.train()
