import matplotlib.pyplot as plt
import numpy as np
import torch
import math
import sys
import os
import scipy.io as sio
from DeepRitz.problem import EllipticPDE


class Result(object):
    def __init__(self, dim, type_c, sysinfo, grid_type='diagonal'):
        """
        :param dim: 维度
        :param type_c: w(x)的类型，可选{None, 'exp', 'expc', 'cos', 'square', 'log'}
        :param grid_type: 计算误差和画示意图的网格选择方式，可选{'diagonal', 'x-axis'}
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.path = "./output/" + sysinfo
        self.figure_path = "./figures/" + sysinfo
        self.dim = dim
        self.type_c = type_c
        self.grid_type = grid_type

    def plot_unn(self):
        dim = self.dim
        model_path = os.path.join(self.path, "model.pt")
        if not os.path.exists(self.figure_path):
            os.makedirs(self.figure_path)
        unn_path = os.path.join(self.figure_path, "unn.eps")
        unn_path_jpg = os.path.join(self.figure_path, "unn.jpg")
        SolutionNet = torch.load(model_path)

        xyz_grid = self.precompute_exact(dim, grid_type=self.grid_type).to(self.device)
        n = xyz_grid.shape[0]
        r_max = np.sqrt(dim)
        if self.type_c == "func1":
            x = np.linspace(0, r_max, n)
            midline = r_max / 2
        else:
            x = np.linspace(-r_max, r_max, n)
            midline = 0.0
        u_ex = EllipticPDE.u_exact(xyz_grid, dim, type_c=self.type_c).detach().cpu().numpy()
        u_nn = SolutionNet(xyz_grid).detach().cpu().numpy()
        u_nn = np.sign(u_nn[n // 3]) * u_nn

        fig1 = plt.figure()
        ax = fig1.add_subplot(111)
        ax.set(title='Solution approximation', xlabel='distance from x to origin', ylabel='u(x)')
        ax.plot(x, u_nn, label='approximation')
        ax.plot(x, u_ex, label='reference')
        ax.axvline(x=midline, ls="-.", c="green")  # 添加垂直直线
        ax.legend()
        plt.savefig(unn_path, format='eps')
        plt.savefig(unn_path_jpg, format='jpg')
        plt.show()

    def plot_unn_heatmap(self):
        dim = 2
        model_path = os.path.join(self.path, "model.pt")
        if not os.path.exists(self.figure_path):
            os.makedirs(self.figure_path)
        unn_path = os.path.join(self.figure_path, "unn_heat.eps")
        unn_path_jpg = os.path.join(self.figure_path, "unn_heat.jpg")
        SolutionNet = torch.load(model_path)

        dpi = 200
        dpi += 1
        if self.type_c == "func1":
            x, y = torch.meshgrid(torch.linspace(0, 1, dpi), torch.linspace(0, 1, dpi), indexing='ij')
        else:
            x, y = torch.meshgrid(torch.linspace(-1, 1, dpi), torch.linspace(-1, 1, dpi), indexing='ij')
        xyz_grid = torch.empty(dpi**2, dim).to(self.device)
        for i in range(dpi):
            for j in range(dpi):
                xyz_grid[i + dpi * j, 0] = x[i, j]
                xyz_grid[i + dpi * j, 1] = y[i, j]

        u_ex = EllipticPDE.u_exact(xyz_grid, dim, type_c=self.type_c).detach().cpu().numpy()
        u_nn = SolutionNet(xyz_grid).detach().cpu().numpy()
        u_nn = np.sign(u_nn[dpi // 3]) * u_nn

        z_ex = np.empty([dpi, dpi])
        z_nn = np.empty([dpi, dpi])
        for i in range(dpi):
            for j in range(dpi):
                z_ex[i, j] = u_ex[i + dpi * j]
                z_nn[i, j] = u_nn[i + dpi * j]

        dpi -= 1
        fig1 = plt.figure("imshow", figsize=(8, 6), facecolor="lightgray")
        ax = fig1.add_subplot(111)
        ax.set(title='Solution approximation')
        plt.xticks(np.arange(0, dpi + dpi / 4, dpi / 4), labels=np.linspace(-1, 1, 5))
        plt.yticks(np.arange(0, dpi + dpi / 4, dpi / 4), labels=np.linspace(-1, 1, 5))

        plt.imshow(z_nn - z_ex, cmap="jet", origin="lower")
        plt.colorbar()

        plt.show()

    def plot_result(self):
        loss_path = os.path.join(self.path, "loss.mat")
        data = sio.loadmat(loss_path)
        loss_data = data['loss'].reshape(-1)
        solution_err_data = data['solution_err'].reshape(-1)
        boundary_norm_data = data['boundary_norm'].reshape(-1)
        epoch = solution_err_data.size
        loss = np.empty(epoch)
        solution_err = np.ones(epoch)
        boundary_norm = np.ones(epoch)

        r = 100
        for i in range(r, epoch):
            loss[i] = np.average(loss_data[i - r : i])
            solution_err[i] = np.average(solution_err_data[i - r : i])
            boundary_norm[i] = np.average(boundary_norm_data[i - r : i])

        loss_path = os.path.join(self.figure_path, "loss.eps")
        solution_err_path = os.path.join(self.figure_path, "solution_err.eps")
        boundary_norm_path = os.path.join(self.figure_path, "boundary_norm.eps")
        loss_path_jpg = os.path.join(self.figure_path, "loss.jpg")
        solution_err_path_jpg = os.path.join(self.figure_path, "solution_err.jpg")
        boundary_norm_path_jpg = os.path.join(self.figure_path, "boundary_norm.jpg")
        print(np.average(loss[-r:-1]))

        solution_err_smooth = np.ones(epoch // r)
        for i in range(len(solution_err_smooth)):
            solution_err_smooth[i] = solution_err[r * i]

        x = np.linspace(0, epoch, epoch)
        # ----------------------------------------Figures 1 loss----------------------------------------
        fig1 = plt.figure()
        ax = fig1.add_subplot(111)
        ax.set(xlim=[-1000, epoch], ylim=[0.56, 1.0], title='loss', xlabel='epoch')
        ax.plot(x, loss)
        ax.legend()
        plt.savefig(loss_path, format='eps')
        plt.savefig(loss_path_jpg, format='jpg')
        # ----------------------------------------Figures 2 error----------------------------------------
        fig2 = plt.figure()
        ax = fig2.add_subplot(111)
        ax.set(xlim=[-10, epoch // r], title='Logarithmic loss of solution error', ylabel='solution error', xlabel='epoch', yscale='log')
        x_smooth = np.linspace(0, epoch // r, epoch // r)
        aa = np.linspace(0, epoch, 5 + 1, dtype=int)
        ax.set_xticks(np.arange(0, epoch // r + 1, epoch // (5 * r)), labels=np.linspace(0, epoch, 5 + 1, dtype=int))
        ax.plot(x_smooth, solution_err_smooth)
        plt.savefig(solution_err_path, format='eps')
        plt.savefig(solution_err_path_jpg, format='jpg')
        # ----------------------------------------Figures 3 boundary loss----------------------------------------
        fig3 = plt.figure()
        ax = fig3.add_subplot(111)
        ax.set(xlim=[100, epoch], title='Logarithmic loss of boundary norm', xlabel='epoch', yscale='log')
        ax.plot(x, boundary_norm)
        plt.savefig(boundary_norm_path, format='eps')
        plt.savefig(boundary_norm_path_jpg, format='jpg')
        plt.show()

    def precompute_exact(self, dim, grid_type='diagonal'):
        # calculate the exact solution, create grid
        n = 100
        if self.type_c == "func1":
            grid = torch.linspace(0.0, 1.0, n + 1)
        else:
            grid = torch.linspace(-1.0, 1.0, n + 1)
        if grid_type == 'grid':
            d_grid = torch.meshgrid([grid for i in range(dim)])
            xyz_grid = torch.empty([(n + 1) ** dim, dim])
            for i in range(dim):
                a = d_grid[i].reshape(-1, 1)
                for j in range((n + 1) ** dim):
                    xyz_grid[j, i] = a[j, 0]
        elif grid_type == 'x-axis':
            xyz_grid = 0.5 * torch.ones(n + 1, dim)
            for j in range(n + 1):
                xyz_grid[j, 0] = grid[j]
        elif grid_type == 'diagonal':
            xyz_grid = torch.zeros(n + 1, dim)
            for j in range(n + 1):
                for i in range(dim):
                    xyz_grid[j, i] = grid[j]
        elif grid_type == 'random':
            xyz_grid = torch.rand(100000, dim)
        else:
            print('Error: invalid grid type')
            sys.exit(-1)
        return xyz_grid

    @staticmethod
    def plot_eigenvector():
        u_FEM = np.loadtxt('./loss_FEM/eigenvector1.txt')
        eigenvector = u_FEM[:, 0]
        x = np.linspace(0, 1, 9999)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set(xlim=[0, 1.0], ylim=[-0.03, 0.03], title='An Example Axes', ylabel='Y-Axis', xlabel='X-Axis')
        plt.plot(x, eigenvector)
        plt.show()
