import torch
import os
from torch.utils.data import Dataset, DataLoader


def generate_offline_dataset(cfg, filename='pde_dataset.pt'):
    print(f"正在生成数据集到 {filename} ...")

    dim = cfg.model.dim
    bound = cfg.model.bound

    # 1. 生成所有 Interior Points
    total_in = cfg.model.batch_in
    print(f"生成内部点: {total_in} 个...")
    all_xyz_in = torch.rand(total_in, dim)
    all_xyz_in = bound[0] + (bound[1] - bound[0]) * all_xyz_in  # 适应边界条件

    # 2. 生成所有 Boundary Points
    m = cfg.model.batch_bd // (2 * dim)

    print(f"生成边界点: {m} 个...")
    all_xyz_bd = torch.rand(2 * dim * m, dim)

    # 每一个 step 有 2*dim 个面，每个面有 m_per_step 个点
    # 我们可以把 all_xyz_bd 看作 [total_steps, 2*dim, m_per_step, dim]
    reshaped_bd = all_xyz_bd.view(2 * dim, m, dim)

    for i in range(dim):
        # 第 i 维的上界 (1.0) -> 对应第 i 个面
        reshaped_bd[i, :, i] = bound[1]
        # 第 i 维的下界 (-0.0) -> 对应第 i+dim 个面
        reshaped_bd[i + dim, :, i] = bound[0]

    all_xyz_bd = reshaped_bd.view(-1, dim)

    # 3. 保存到硬盘 (二进制格式)
    torch.save(
        {'xyz_in': all_xyz_in, 'xyz_bd': all_xyz_bd, 'batch_in': cfg.model.batch_in, 'batch_bd': cfg.model.batch_bd, 'dim': dim}, filename
    )

    print("数据集生成完毕，已保存！")


class DRMDataset(Dataset):
    def __init__(self, cfg):
        """
        初始化：加载整个 .pt 文件到 CPU 内存中
        """
        # 加载数据 (map_location='cpu' 确保数据加载到内存而不是显存)
        data = torch.load(cfg.data.pt_file_path, map_location='cpu')

        self.xyz_in = data['xyz_in']
        self.xyz_bd = data['xyz_bd']

        # 从保存的字典中获取配置信息
        self.batch_in_size = data['batch_in']
        self.batch_bd_size = data['batch_bd']

        self.dim = data['dim']

        self.n_steps = cfg.data.n_steps

    def __len__(self):
        return self.n_steps

    def __getitem__(self, idx):
        """
        获取第 idx 步所需的训练数据
        """
        # 1. 切取内部点 (Interior Points)
        start_in = idx * self.batch_in_size
        end_in = (idx + 1) * self.batch_in_size
        batch_xyz_in = self.xyz_in[start_in:end_in]

        # 2. 切取边界点 (Boundary Points)
        start_bd = idx * self.batch_bd_size
        end_bd = (idx + 1) * self.batch_bd_size
        batch_xyz_bd = self.xyz_bd[start_bd:end_bd]

        return batch_xyz_in, batch_xyz_bd
