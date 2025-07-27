import os
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from time import time
from datetime import datetime  # 用于时间戳
from pyDOE import lhs

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", device)


# PINN 网络结构
class PINN3DSpherical(nn.Module):
    def __init__(self, layers):
        super().__init__()
        seq = []
        for i in range(len(layers) - 2):
            seq.append(nn.Linear(layers[i], layers[i+1]))
            seq.append(nn.Tanh())
        seq.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*seq)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
                nn.init.zeros_(m.bias)

    def forward(self, x, y, z, t):
        input = torch.cat([x, y, z, t], dim=1)
        out = self.net(input)
        u = out[:, 0:1]
        v = out[:, 1:2]
        return u, v

# 求解器
class Solver3DSpherical:
    def __init__(self, model, B, t, Nf,
                 X_f, U2, V2,
                 x_flat, y_flat, z_flat, n_fft, time_points,
                 k, R0, delta,
                 lr_ic, lr_pde, lr_norm, lr_ana, lr_fft, lr_mom, lr_ene):

        self.B = B
        self.t = t
        self.N_r = Nf

        self.lr_ic = lr_ic
        self.lr_pde = lr_pde
        self.lr_norm = lr_norm
        self.lr_ana = lr_ana
        self.lr_fft = lr_fft
        self.lr_x_mom = lr_mom
        self.lr_y_mom = lr_mom
        self.lr_z_mom = lr_mom
        self.lr_ene = lr_ene

        # 构建一次性空间网格 (n_fft³)
        self.n_fft = n_fft
        self.time_points = time_points
        self.x_flat = x_flat
        self.y_flat = y_flat
        self.z_flat = z_flat

        # 模型
        self.model = model.to(device)

        # 初始参数
        self.k = k  # 初始动量
        self.R0 = R0  # 起始坐标（起始位置放置中心0）
        self.delta = delta  # 波包宽度参数

        # 解析解引导数据
        self.X2 = X_f
        self.u2 = U2
        self.v2 = V2
        # 原始大数量数组
        self.data = X_f
        # 初始时刻总采样点
        # self.X0 = X0

        # 门空参数
        self.gamma = -0.5

    # 其他时刻归一化数据集合
    def grid_generator(self, n_per_dim=8):
        t_values = np.linspace(0, self.t, 10)
        ub1 = np.array([self.B, self.B, self.B, 0.5])
        lb = np.array([-self.B, -self.B, -self.B, 0.0])
        for t in t_values:
            grids = [np.linspace(lb[j], ub1[j], n_per_dim) for j in range(4)]
            mesh = np.meshgrid(*grids)
            X = np.vstack([m.ravel() for m in mesh]).T
            X[:, 3] = t
            yield X  # 每次只返回一个时间步的数据

    def update_residual_points(self, X_f_new, U2_new, V2_new):
        """
        动态更新 PDE 残差点和解析引导数据
        """
        self.X2 = X_f_new
        self.data = X_f_new
        self.u2 = U2_new
        self.v2 = V2_new

    def schrodinger_residual(self, x, y, z, t):
        """
        计算直角坐标系下的薛定谔方程残差：
        i ∂ψ/∂t + ½ Δψ = 0
        """
        lap_r, lap_i, psi_i_t, psi_r_t = self.laplacian(x, y, z, t)

        # 实部残差： Re{iψ_t + ½Δψ} = ψ_i_t - ½Δψ_r
        # 虚部残差： Im{iψ_t + ½Δψ} = -ψ_r_t - ½Δψ_i
        res_r = psi_i_t - 0.5 * lap_r
        res_i = -psi_r_t - 0.5 * lap_i

        # 可选：限制发散
        threshold = 1e5
        res_r = torch.clamp(res_r, -threshold, threshold)
        res_i = torch.clamp(res_i, -threshold, threshold)

        return res_r, res_i

    def split_input_with_grad(self, data):
        """
        输入数据为 (N, 4): [x, y, z, t]
        分离出各变量并设置 requires_grad
        """
        data = torch.tensor(data, dtype=torch.float32, device=device)
        x = data[:, 0:1].clone().detach().to(device).requires_grad_(True)
        y = data[:, 1:2].clone().detach().to(device).requires_grad_(True)
        z = data[:, 2:3].clone().detach().to(device).requires_grad_(True)
        t = data[:, 3:4].clone().detach().to(device).requires_grad_(True)
        return x, y, z, t

    def laplacian(self, x, y, z, t):
        """计算三维直角坐标下的拉普拉斯项及时间导数"""

        # print("x.requires_grad =", x.requires_grad)  # 应该为 True

        # 模型预测 ψ_r, ψ_i
        psi_r, psi_i = self.model(x, y, z, t)

        # 一阶导数
        psi_r_t = torch.autograd.grad(psi_r, t, grad_outputs=torch.ones_like(psi_r), create_graph=True)[0]
        psi_i_t = torch.autograd.grad(psi_i, t, grad_outputs=torch.ones_like(psi_i), create_graph=True)[0]

        # 二阶导数用于 Laplacian
        psi_r_xx = torch.autograd.grad(psi_r, x, grad_outputs=torch.ones_like(psi_r), create_graph=True, allow_unused=True)[0]
        psi_r_xx = torch.autograd.grad(psi_r_xx, x, grad_outputs=torch.ones_like(psi_r_xx), create_graph=True)[0]

        psi_r_yy = torch.autograd.grad(psi_r, y, grad_outputs=torch.ones_like(psi_r), create_graph=True)[0]
        psi_r_yy = torch.autograd.grad(psi_r_yy, y, grad_outputs=torch.ones_like(psi_r_yy), create_graph=True)[0]

        psi_r_zz = torch.autograd.grad(psi_r, z, grad_outputs=torch.ones_like(psi_r), create_graph=True)[0]
        psi_r_zz = torch.autograd.grad(psi_r_zz, z, grad_outputs=torch.ones_like(psi_r_zz), create_graph=True)[0]

        psi_i_xx = torch.autograd.grad(psi_i, x, grad_outputs=torch.ones_like(psi_i), create_graph=True)[0]
        psi_i_xx = torch.autograd.grad(psi_i_xx, x, grad_outputs=torch.ones_like(psi_i_xx), create_graph=True)[0]

        psi_i_yy = torch.autograd.grad(psi_i, y, grad_outputs=torch.ones_like(psi_i), create_graph=True)[0]
        psi_i_yy = torch.autograd.grad(psi_i_yy, y, grad_outputs=torch.ones_like(psi_i_yy), create_graph=True)[0]

        psi_i_zz = torch.autograd.grad(psi_i, z, grad_outputs=torch.ones_like(psi_i), create_graph=True)[0]
        psi_i_zz = torch.autograd.grad(psi_i_zz, z, grad_outputs=torch.ones_like(psi_i_zz), create_graph=True)[0]

        # 直角坐标下拉普拉斯
        lap_r = psi_r_xx + psi_r_yy + psi_r_zz
        lap_i = psi_i_xx + psi_i_yy + psi_i_zz

        return lap_r, lap_i, psi_i_t, psi_r_t

    # 时间门控函数
    def causal_gate(self, t, alpha=5.0):
        """
        t: torch tensor of time values
        gamma: gate center shift (scalar)
        alpha: gate steepness
        """
        t_norm = t / self.t
        return 0.5 * (1.0 - torch.tanh(alpha * (t_norm - self.gamma)))

    def update_gamma(self, loss, eta=0.01, epsilon=10.0):
        """
        gamma: current gamma (float)
        loss: current PDE causal loss (float)
        """
        loss_np = loss.detach().cpu().numpy()
        return self.gamma + eta * np.exp(-epsilon * loss_np)

    def loss(self, batch_data):
        mse_energy = []
        mse_momentum_x = []
        mse_momentum_y = []
        mse_momentum_z = []
        mse_norm = []


        # 计算动量、能量损失
        for arrays in self.grid_generator():
            if arrays[0, 3] == 0:
                self.X0 = arrays

                psi_r0, psi_i0 = self.initial_wave_cartesian(arrays)
                # 初始时刻密度
                self.mean_density0 = self.calculate_norm(psi_r0, psi_i0)
                # 1. 初始能量
                lap_r0, lap_i0 = self.laplacian_ini_psi(arrays)
                self.energy_r0 = torch.mean(psi_r0 * lap_r0 + psi_i0 * lap_i0)

                # 2. 初始动量
                psi_r_x0, psi_i_x0, psi_r_y0, psi_i_y0, psi_r_z0, psi_i_z0 = self.grad_ini_psi(arrays)

                self.momentum_x0 = torch.mean(psi_r0 * psi_i_x0 -
                                         psi_i0 * psi_r_x0)
                self.momentum_y0 = torch.mean(psi_r0 * psi_i_y0 -
                                         psi_i0 * psi_r_y0)
                self.momentum_z0 = torch.mean(psi_r0 * psi_i_z0 -
                                         psi_i0 * psi_r_z0)
            else:
                # 3. 其他时刻
                x, y, z, t = self.split_input_with_grad(arrays)
                psi_r, psi_i = self.model(x, y, z, t)
                # energy
                lap_r, lap_i, _, _ = self.laplacian(x, y, z, t)
                energy_r = torch.mean(psi_r * lap_r + psi_i * lap_i)
                mse_energy.append((energy_r - self.energy_r0).pow(2) * self.causal_gate(t))

                # momentum
                grad_r = torch.autograd.grad(psi_r, [x, y, z, t], grad_outputs=torch.ones_like(psi_r), create_graph=True)
                grad_i = torch.autograd.grad(psi_i, [x, y, z, t], grad_outputs=torch.ones_like(psi_i), create_graph=True)
                psi_r_x, psi_r_y, psi_r_z, _ = grad_r
                psi_i_x, psi_i_y, psi_i_z, _ = grad_i
                momentum_x = torch.mean(psi_r * psi_i_x -
                                        psi_i * psi_r_x)
                momentum_y = torch.mean(psi_r * psi_i_y -
                                        psi_i * psi_r_y)
                momentum_z = torch.mean(psi_r * psi_i_z -
                                        psi_i * psi_r_z)
                mse_momentum_x.append((momentum_x - self.momentum_x0).pow(2) * self.causal_gate(t))
                mse_momentum_y.append((momentum_y - self.momentum_y0).pow(2) * self.causal_gate(t))
                mse_momentum_z.append((momentum_z - self.momentum_z0).pow(2) * self.causal_gate(t))

                # norm
                mean_density = (self.calculate_norm(psi_r, psi_i) - self.mean_density0) # 其他时刻密度

                mse_norm.append(mean_density.pow(2) * self.causal_gate(t))

        mse_energy_total = torch.stack(mse_energy).mean()
        mse_momentum_x_total = torch.stack(mse_momentum_x).mean()
        mse_momentum_y_total = torch.stack(mse_momentum_y).mean()
        mse_momentum_z_total = torch.stack(mse_momentum_z).mean()
        mse_norm_total = torch.stack(mse_norm).mean()

        # 初始条件损失
        x0, y0, z0, t0 = self.split_input_with_grad(self.X0)
        u0_pred, v0_pred = self.model(x0, y0, z0, t0)
        u0, v0 = self.initial_wave_cartesian(self.X0)
        mse_ic = torch.mean((u0_pred - u0) ** 2 * self.causal_gate(t0)) + torch.mean((v0_pred - v0) ** 2 * self.causal_gate(t0))

        # 解析引导损失
        x2, y2, z2, t2 = self.split_input_with_grad(self.X2)
        u2_pred, v2_pred = self.model(x2, y2, z2, t2)
        mse_ana = torch.mean((u2_pred - self.u2) ** 2 * self.causal_gate(t2)) + torch.mean((v2_pred - self.v2) ** 2 * self.causal_gate(t2))

        # PDE 残差
        x3, y3, z3, t3 = self.split_input_with_grad(batch_data)
        res_r, res_i = self.schrodinger_residual(x3, y3, z3, t3)
        mse_pde = torch.mean((res_r ** 2 + res_i ** 2) * self.causal_gate(t3))

        # FFT 损失
        mse_fft = 0.0
        for t_val in self.time_points:
            t_flat = t_val.repeat(self.x_flat.shape[0], 1)
            psi_r, psi_i = self.model(self.x_flat, self.y_flat, self.z_flat, t_flat)
            psi_complex = (psi_r.squeeze(-1) + 1j * psi_i.squeeze(-1)).reshape(self.n_fft, self.n_fft, self.n_fft)
            fft_vals = torch.fft.fftn(psi_complex, dim=(0, 1, 2), norm='forward')
            const_mod = torch.abs(fft_vals[0, 0, 0])
            mse_fft += const_mod.pow(2) * self.causal_gate(t_val)
        mse_fft = mse_fft / len(self.time_points)

        # 权重加权总损失
        mse_ic = mse_ic * self.lr_ic
        mse_pde = mse_pde * self.lr_pde
        mse_norm_total = mse_norm_total * self.lr_norm
        mse_ana = mse_ana * self.lr_ana
        mse_fft = mse_fft * self.lr_fft
        mse_momentum_x_total = mse_momentum_x_total * self.lr_x_mom
        mse_momentum_y_total = mse_momentum_y_total * self.lr_y_mom
        mse_momentum_z_total = mse_momentum_z_total * self.lr_z_mom
        mse_energy_total = mse_energy_total * self.lr_ene

        total_loss = (mse_ic + mse_pde + mse_norm_total + mse_ana +
                      mse_fft + mse_momentum_x_total + mse_momentum_y_total + mse_momentum_z_total + mse_energy_total)
        print("gamma:", self.gamma)
        self.gamma = self.update_gamma(total_loss)


        return (total_loss, mse_ic, mse_pde, mse_norm_total, mse_ana, mse_fft,
                mse_momentum_x_total, mse_momentum_y_total, mse_momentum_z_total, mse_energy_total)

    def train(self, epochs, initial_batch_size, optimizer, resample_every):
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=100)
        history = {'loss': [], 'ic': [], 'pde': [], 'norm': [], 'ana': [], 'fft': [], 'mom_x': [], 'mom_y': [], 'mom_z': [], 'ene': []}
        save_path = r'./save_model/training_loss.png'
        start = time()

        # ========== 实时绘图设置 ==========
        plt.ion()  # 打开交互模式
        fig, ax = plt.subplots(figsize=(10, 6))
        keys = ['loss', 'ic', 'pde', 'mom_x', 'mom_y', 'mom_z', 'ene']
        labels = ['Total Loss', 'IC Loss', 'PDE Loss', 'Mom_x Loss', 'Mom_y Loss', 'Mom_z Loss', 'Ene Loss', 'Norm Loss', 'Ana Loss', 'FFT Loss']
        colors = ['blue', 'green', 'red', 'orange', 'yellow', 'purple', 'brown', 'white', 'white', 'white']
        line_styles = ['-', '--', '-.', ':', '-.', '-.', '-.', '-.', '-.', '-.']
        lines = []
        for i, label in enumerate(labels):
            line, = ax.semilogy([], [], label=label, color=colors[i], linestyle=line_styles[i], linewidth=2)
            lines.append(line)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Real-Time Loss History')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # 定义 batch_size 调整计划
        batch_size_schedule = {
            0: initial_batch_size,
            5000000: initial_batch_size // 2,
            10000000: initial_batch_size // 4,
        }

        for ep in range(1, epochs + 1):

            # 每隔 resample_every 轮重新采样残差点
            if resample_every is not None and ep % resample_every == 0:
                print(f"\n[Resampling X_f at epoch {ep}]")
                X_f_new = self.r3_sampling(N_r=self.N_r, max_iterations=2)
                U2_new, V2_new = analytic_solution_cartesian(self.k, self.R0, X_f_new, self.delta)
                self.update_residual_points(X_f_new, U2_new, V2_new)

            # 动态 batch_size 调整
            current_batch_size = max([v for k, v in batch_size_schedule.items() if ep >= k])

            # 随机打乱数据
            num_batches = len(self.data) // current_batch_size
            batch_indices = np.arange(len(self.data))
            np.random.shuffle(batch_indices)

            for batch_idx in range(num_batches):
                start_idx = batch_idx * current_batch_size
                end_idx = (batch_idx + 1) * current_batch_size
                batch_data = self.data[batch_indices[start_idx:end_idx]]
                optimizer.zero_grad()
                loss, ic, pde, norm, ana, fft, mom_x, mom_y, mom_z, ene = self.loss(batch_data)
                loss.backward(retain_graph=True)
                optimizer.step()

            # 每个 epoch 结束后评估一次总损失
            (total_loss, total_ic, total_pde, total_norm, total_ana, total_fft,
             total_mom_x, total_mom_y, total_mom_z, total_ene) = self.loss(
                self.data)
            scheduler.step(total_loss)

            # 记录历史
            history['loss'].append(total_loss.item())
            history['ic'].append(total_ic.item())
            history['pde'].append(total_pde.item())
            history['norm'].append(total_norm.item())
            history['ana'].append(total_ana.item())
            history['fft'].append(total_fft.item())
            history['mom_x'].append(total_mom_x.item())
            history['mom_y'].append(total_mom_y.item())
            history['mom_z'].append(total_mom_z.item())
            history['ene'].append(total_ene.item())

            # ========== 实时更新图像 ==========
            for i, key in enumerate(keys):
                lines[i].set_data(range(len(history[key])), history[key])
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()

            if ep % 1 == 0:
                print(
                    f"Epoch {ep}/{epochs}, Batch Size: {current_batch_size}, Loss: {total_loss.item():.4e}, IC: {total_ic.item():.4e}, "
                    f"PDE: {total_pde.item():.4e},"
                    # f" norm: {total_norm.item():.4e}, ana: {total_ana.item():.4e}, fft: {total_fft.item():.4e}, "
                    f"mom_x: {total_mom_x.item():.4e}, mom_y: {total_mom_y.item():.4e}, mom_z: {total_mom_z.item():.4e}, ene: {total_ene.item():.4e}")

        plt.ioff()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Training completed in {time() - start:.1f}s")
        return history

    # 初始条件
    def initial_wave_cartesian(self, X0):
        """
        初始波函数在直角坐标系下的形式。
        X0: (N,3) -> [x, y, z]
        k:  (3,)  -> [kx, ky, kz]
        R0: (3,)  -> [x0, y0, z0]
        delta: 标准差（波包宽度）
        """
        delta = torch.tensor(self.delta, dtype=torch.float32, device=device)

        kx, ky, kz = [torch.tensor(v, dtype=torch.float32, device=device) for v in self.k]
        x0, y0, z0 = [torch.tensor(v, dtype=torch.float32, device=device) for v in self.R0]

        X0_tensor = torch.tensor(X0, dtype=torch.float32, device=device)
        x = X0_tensor[:, 0:1]
        y = X0_tensor[:, 1:2]
        z = X0_tensor[:, 2:3]

        # 常数项
        A = (2 * torch.pi * delta ** 2) ** (-3 / 4)

        # |r - r0|^2
        r_r0_sq = (x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2
        envelope = torch.exp(-r_r0_sq / (4 * delta ** 2))

        # 相位项：k·r
        phase_arg = kx * x + ky * y + kz * z

        # 波函数实部与虚部
        psi_r = A * envelope * torch.cos(phase_arg)
        psi_i = A * envelope * torch.sin(phase_arg)

        return psi_r, psi_i

    def laplacian_ini_psi(self, X0):
        """
        拉普拉斯算子作用下的复波函数：∇²ψ
        返回：实部、虚部
        """
        delta = torch.tensor(self.delta, dtype=torch.float32, device=device)

        kx, ky, kz = [torch.tensor(v, dtype=torch.float32, device=device) for v in self.k]
        x0, y0, z0 = [torch.tensor(v, dtype=torch.float32, device=device) for v in self.R0]

        X0_tensor = torch.tensor(X0, dtype=torch.float32, device=device)
        x = X0_tensor[:, 0:1]
        y = X0_tensor[:, 1:2]
        z = X0_tensor[:, 2:3]

        pi = torch.pi
        delta2 = delta ** 2
        delta4 = delta2 ** 2
        norm = 4 * (2 * pi) ** (3 / 4) * delta4 * delta2 ** (3 / 4)

        # 相对位移
        dx = x - x0
        dy = y - y0
        dz = z - z0

        # |r - r0|^2
        r_r0_sq = dx ** 2 + dy ** 2 + dz ** 2

        # 相位项和 envelope
        phase = kx * x + ky * y + kz * z
        envelope = torch.exp(-r_r0_sq / (4 * delta2))

        # 指数项
        exp_arg = torch.exp(1j * phase) * envelope  # complex

        # 复系数项
        real_part = r_r0_sq - 6 * delta2 - 4 * delta4 * (kx ** 2 + ky ** 2 + kz ** 2)
        imag_part = -4 * delta2 * (kx * dx + ky * dy + kz * dz)

        coeff = real_part + 1j * imag_part  # complex

        psi_lap = exp_arg * coeff / norm  # complex

        # 分离实部和虚部
        return torch.real(psi_lap), torch.imag(psi_lap)


    def grad_ini_psi(self, X0):
        """
        梯度 ∇ψ(x, y, z)：返回复梯度的三个方向分量（gx, gy, gz）
        """

        delta = torch.tensor(self.delta, dtype=torch.float32, device=device)

        kx, ky, kz = [torch.tensor(v, dtype=torch.float32, device=device) for v in self.k]
        x0, y0, z0 = [torch.tensor(v, dtype=torch.float32, device=device) for v in self.R0]

        X0_tensor = torch.tensor(X0, dtype=torch.float32, device=device)
        x = X0_tensor[:, 0:1]
        y = X0_tensor[:, 1:2]
        z = X0_tensor[:, 2:3]

        pi = torch.pi
        delta2 = delta ** 2
        norm = (2 * pi) ** (3 / 4) * delta2 ** (3 / 4)

        # 相对位置
        dx = x - x0
        dy = y - y0
        dz = z - z0

        # r · k 和 r - r0 的平方
        phase = kx * x + ky * y + kz * z
        r_r0_sq = dx ** 2 + dy ** 2 + dz ** 2

        # 共用指数项
        envelope = torch.exp(-r_r0_sq / (4 * delta2))
        exp_phase = torch.exp(1j * phase)
        exp_total = exp_phase * envelope  # shape: (N, 1) complex

        # 梯度各分量
        gx = exp_total * (1j * kx + (-dx) / (2 * delta2)) / norm
        gy = exp_total * (1j * ky + (-dy) / (2 * delta2)) / norm
        gz = exp_total * (1j * kz + (-dz) / (2 * delta2)) / norm

        # 返回每一维梯度的实部与虚部
        return (
            torch.real(gx), torch.imag(gx),
            torch.real(gy), torch.imag(gy),
            torch.real(gz), torch.imag(gz)
        )

    # 计算归一化数值
    def calculate_norm(self, u, v):
        density = u ** 2 + v ** 2
        mean_density = torch.sqrt(torch.sum(density))
        return mean_density

    def r3_sampling(self, N_r, max_iterations):
        """
        R3 sampling for residual points based on PDE residuals.

        Parameters:
            N_r: number of residual points
            max_iterations: number of R3 iterations
        Returns:
            X_r: updated residual points (N_r, 4)
        """
        B = self.B  # 边界
        t = self.t
        lb = np.array([-B, -B, -B, 0.0])
        # 定义所有训练点数据
        ub = np.array([B, B, B, t])
        X_r = lb + (ub - lb) * lhs(4, N_r)

        for _ in range(max_iterations):
            x, y, z, t = self.split_input_with_grad(X_r)
            res_r, res_i = self.schrodinger_residual(x, y, z, t)
            residual = torch.sqrt(res_r ** 2 + res_i ** 2).detach().cpu().numpy().flatten()

            tau = np.mean(residual)
            retained_mask = residual > tau
            X_retained = X_r[retained_mask]

            N_new = N_r - X_retained.shape[0]
            X_new = lb + (ub - lb) * lhs(4, N_new)

            X_r = np.vstack([X_retained, X_new])

        return X_r

def analytic_solution_cartesian(K, R, X, delta, m=1.0):
    """
    自由粒子在三维直角坐标下高斯波包的解析解（Hartree 单位制：ℏ=1, m=1）。
    X: (N,4) -> [x, y, z, t]
    K: (3,)  -> [k_x, k_y, k_z]
    R: (3,)  -> [x0, y0, z0]
    """
    delta = torch.tensor(delta, dtype=torch.float32, device=device)
    m = torch.tensor(m, dtype=torch.float32, device=device)

    kx, ky, kz = [torch.tensor(k, dtype=torch.float32, device=device) for k in K]
    x0, y0, z0 = [torch.tensor(r, dtype=torch.float32, device=device) for r in R]

    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    x = X_tensor[:, 0:1]
    y = X_tensor[:, 1:2]
    z = X_tensor[:, 2:3]
    t = X_tensor[:, 3:4]

    # 1. σ(t): 宽度随时间演化
    sigma = delta ** 2 + 1j * t / (2 * m)

    # 2. 归一化因子
    norm = (2 * torch.pi * (delta + 1j * t / (2 * m * delta)) ** 2) ** (-3 / 4)

    # 3. 波包中心位置随时间变化（漂移项）
    dx = (kx / m) * t
    dy = (ky / m) * t
    dz = (kz / m) * t
    xc = x0 + dx
    yc = y0 + dy
    zc = z0 + dz

    # 4. |r - r_c(t)|^2
    dx2 = (x - xc) ** 2 + (y - yc) ** 2 + (z - zc) ** 2

    # 5. 相位项：k·r - (k^2)/(2m)t
    phase_arg = (kx * x + ky * y + kz * z - (kx ** 2 + ky ** 2 + kz ** 2) / (2 * m) * t)

    # 6. 波函数表达式
    psi = norm * torch.exp(-dx2 / (4 * sigma) + 1j * phase_arg)
    psi_real = psi.real
    psi_imag = psi.imag

    return psi_real, psi_imag


# 测试点 y=0; z=0; t=0.5截面
def predict_and_plot(solver, B, k, R0, delta):
    N = 100
    z = 0
    y = 0
    # x = 0
    t = 0.0
    x = np.linspace(-10, 10, N)
    # x = np.full((1, N), x)
    y_test = np.full((1, N), y)
    # y_test = np.linspace(-10, 10, N)
    z_test = np.full((1, N), z)
    t_test = np.full((1, N), t)
    x_y_z_test = np.vstack([x, y_test, z_test, t_test]).T
    # 解析解
    psi_exact_real, psi_exact_imag = analytic_solution_cartesian(k, R0, x_y_z_test, delta)
    psi_exact_real =psi_exact_real.cpu().detach().numpy()
    psi_exact_imag = psi_exact_imag.cpu().detach().numpy()
    # PINN 预测
    xf = torch.tensor(x_y_z_test[:,0:1], dtype=torch.float32, device=device)
    yf = torch.tensor(x_y_z_test[:,1:2], dtype=torch.float32, device=device)
    zf = torch.tensor(x_y_z_test[:,2:3], dtype=torch.float32, device=device)
    tf = torch.tensor(x_y_z_test[:,3:4], dtype=torch.float32, device=device)
    psi_pinn_real, psi_pinn_imag = solver.model(xf, yf, zf, tf)
    psi_pinn_real =psi_pinn_real.cpu().detach().numpy()
    psi_pinn_imag = psi_pinn_imag.cpu().detach().numpy()
    # 绘图对比
    plt.figure(figsize=(8, 4))
    plt.plot(x, psi_exact_real, '--', label='Analytic Real', color='red', linewidth=2) # 注意负号
    plt.plot(x, psi_pinn_real, '-', label='PINN Real', color='red', linewidth=2)
    plt.plot(x, psi_exact_imag, '--', label='Analytic Imag', color='black', linewidth=2)
    plt.plot(x, psi_pinn_imag, '-', label='PINN Imag', color='black', linewidth=2)
    plt.xlabel('x')
    plt.ylabel(r'$\psi$')
    plt.title(f'Wavefunction slice at y={y}, z={z}, t={t}')
    plt.legend()
    plt.tight_layout()
    plt.show()

# 绘制 y=0; z=0 全时空实部、虚部和振幅偏差
def calculate_and_plot_diffs(solver, B, k, R0, delta):
    # 定义时间范围
    t_values = np.linspace(0, 1, 100)
    N = 100
    z = 0
    y = 0
    x = np.linspace(-B, B, N)
    y = np.full((1, N), y)
    z = np.full((1, N), z)

    real_diffs = []
    imag_diffs = []
    amp_diffs = []

    for t in t_values:
        t_test = np.full((1, N), t)
        x_y_z_test_all = np.vstack([x, y, z, t_test]).T
        xf = torch.tensor(x_y_z_test_all[:, 0:1], dtype=torch.float32, device=device)
        yf = torch.tensor(x_y_z_test_all[:, 1:2], dtype=torch.float32, device=device)
        zf = torch.tensor(x_y_z_test_all[:, 2:3], dtype=torch.float32, device=device)
        tf = torch.tensor(x_y_z_test_all[:, 3:4], dtype=torch.float32, device=device)
        psi_pinn_real, psi_pinn_imag = solver.model(xf, yf, zf, tf)
        psi_pinn_real = psi_pinn_real.cpu().detach().numpy()
        psi_pinn_imag = psi_pinn_imag.cpu().detach().numpy()

        psi_exact_real, psi_exact_imag = analytic_solution_cartesian(k, R0, x_y_z_test_all, delta)
        psi_exact_real = psi_exact_real.cpu().detach().numpy()
        psi_exact_imag = psi_exact_imag.cpu().detach().numpy()
        psi_pinn = psi_pinn_real + 1j * psi_pinn_imag
        psi_exact = psi_exact_real + 1j * psi_exact_imag

        real_diff = psi_pinn_real - psi_exact_real
        imag_diff = psi_pinn_imag - psi_exact_imag
        amp_diff = np.abs(psi_pinn) - np.abs(psi_exact)

        real_diffs.append(real_diff)
        imag_diffs.append(imag_diff)
        amp_diffs.append(amp_diff)

    real_diffs = np.stack(real_diffs, axis=1)
    imag_diffs = np.stack(imag_diffs, axis=1)
    amp_diffs = np.stack(amp_diffs, axis=1)

    # 绘制热力图
    fig = plt.figure(figsize=(18, 5))
    gs = GridSpec(1, 3, wspace=0.4, hspace=0.4)

    labels = ['Real Part Difference', 'Imaginary Part Difference', 'Amplitude Difference']
    diffs = [real_diffs, imag_diffs, amp_diffs]
    cmaps = ['coolwarm', 'coolwarm', 'plasma']

    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(diffs[i], extent=[t_values.min(), t_values.max(), x.min(), x.max()],
                       aspect='auto', origin='lower', cmap=cmaps[i])
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('x')
        ax.set_title(labels[i])
        plt.colorbar(im, ax=ax)

    plt.show()


def normalize_wavefunction(psi_r, psi_i, dV=1.0):
    rho = psi_r**2 + psi_i**2
    # rho = torch.tensor(rho, dtype=torch.float32, device=device)
    norm = torch.sqrt(torch.sum(rho) * dV)
    return psi_r / norm, psi_i / norm

def overlap_integral(psi_r1, psi_i1, psi_r2, psi_i2, dV=1.0):
    real_part = psi_r1 * psi_r2 + psi_i1 * psi_i2
    imag_part = psi_r1 * psi_i2 - psi_i1 * psi_r2
    overlap_real = torch.sum(real_part) * dV
    overlap_imag = torch.sum(imag_part) * dV
    return torch.sqrt(overlap_real**2 + overlap_imag**2)


def pro_p_ene_plot(solver, k, R0, delta, m):

    # k = np.array([1.0, 0, 0.0]) # 初始动量
    # R0 = np.array([0.0, 0.0, 0.0]) # 起始坐标（起始位置放置中心0）
    # delta = np.array(0.5) # 波包宽度参数
    rho_list, x_list, px_list, py_list, pz_list, energy_list, overlap_list = [], [], [], [], [], [], []
    time_points, kinetic_list = [], []

    # 计算动量、能量损失
    for arrays in solver.grid_generator():

        xf, yf, zf, tf = solver.split_input_with_grad(arrays)
        t = tf[0].item()  # 获取当前时间点
        time_points.append(t)

        # 模型输出
        psi_r, psi_i = solver.model(xf, yf, zf, tf)
        psi_r2, psi_i2 = analytic_solution_cartesian(k, R0, arrays, delta)

        # 归一化（可选）
        psi_r1, psi_i1 = normalize_wavefunction(psi_r, psi_i, dV=1.0)
        psi_r2, psi_i2 = normalize_wavefunction(psi_r2, psi_i2, dV=1.0)

        # 计算交叠
        overlap = overlap_integral(psi_r1, psi_i1, psi_r2, psi_i2, dV=1.0)

        # 几率密度
        rho = psi_r**2 + psi_i**2
        rho_total = rho.mean()  # 或用 .sum() * dx*dy*dz

        # 计算x期望 <x> = sum(x * rho) / sum(rho)
        x_mean = torch.sum(xf * rho) / torch.sum(rho)

        # 动量
        grad_r = torch.autograd.grad(psi_r, [xf, yf, zf, tf], grad_outputs=torch.ones_like(psi_r), create_graph=True)
        grad_i = torch.autograd.grad(psi_i, [xf, yf, zf, tf], grad_outputs=torch.ones_like(psi_i), create_graph=True)
        psi_r_x, psi_r_y, psi_r_z, _ = grad_r
        psi_i_x, psi_i_y, psi_i_z, _ = grad_i
        px = torch.mean(psi_r * psi_i_x -
                                psi_i * psi_r_x)
        py = torch.mean(psi_r * psi_i_y -
                                psi_i * psi_r_y)
        pz = torch.mean(psi_r * psi_i_z -
                                psi_i * psi_r_z)

        # 动能 = (px^2 + py^2 + pz^2) / 2m
        kinetic = (px ** 2 + py ** 2 + pz ** 2) / (2 * m)
        # T = torch.tensor(T, dtype=torch.float32, device=device)

        # 能量
        lap_r, lap_i, _, _ = solver.laplacian(xf, yf, zf, tf)
        energy = -1 /(2*m) * torch.mean(psi_r * lap_r + psi_i * lap_i)
        # energy = torch.tensor(energy, dtype=torch.float32, device=device)

        # 保存数据
        rho_list.append(rho_total.item())
        px_list.append(px.item())
        py_list.append(py.item())
        pz_list.append(pz.item())
        energy_list.append(energy.item())
        kinetic_list.append(kinetic.item())
        x_list.append(x_mean.item())
        overlap_list.append(overlap.item())

    # 转为 numpy 数组
    time_points = np.array(time_points)
    def percent_change(lst):
        lst = np.array(lst)
        return (lst - lst[0]) / abs(lst[0]) * 100

    # 绘图：百分比变化
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(time_points, percent_change(rho_list))
    plt.title("Probability (% Change) vs Time")
    plt.ylabel("% Change")

    plt.subplot(1, 2, 2)
    plt.plot(time_points, percent_change(energy_list))
    plt.title("Energy (% Change) vs Time")
    plt.ylabel("% Change")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(time_points, percent_change(kinetic_list))
    plt.title("T (% Change) vs Time")
    plt.ylabel("% Change")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(time_points, overlap_list)
    plt.title("overlap vs Time")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(time_points, x_list)
    plt.title("x_expectation vs Time")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 4))
    plt.subplot(2, 3, 1)
    plt.plot(time_points, percent_change(px_list))
    plt.title("Momentum Px (% Change)")

    plt.subplot(2, 3, 2)
    plt.plot(time_points, percent_change(py_list))
    plt.title("Momentum Py (% Change)")

    plt.subplot(2, 3, 3)
    plt.plot(time_points, percent_change(pz_list))
    plt.title("Momentum Pz (% Change)")
    plt.tight_layout()
    plt.show()


# 主函数示例
if __name__ == '__main__':
    B = 5.0 # 边界
    t = 1.0
    # 其他数据
    k = np.array([1.0, 0, 0.0]) # 初始动量
    R0 = np.array([0.0, 0.0, 0.0]) # 起始坐标（起始位置放置中心0）
    delta = np.array(0.5) # 波包宽度参数
    m = np.array(1.0) # 质量


    lb = np.array([-B, -B, -B, 0.0])
    # 定义所有训练点数据
    ub = np.array([B, B, B, t])
    Nf = 40000
    X_f = lb + (ub - lb) * lhs(4, Nf)
    # 定义初始时刻数据
    # ub0 = np.array([B, B, B, 0])
    # N0 = 10000
    # X0 = lb + (ub0 - lb) * lhs(4, N0)
    # U0_ana, V0_ana = analytic_solution_cartesian(k, R0, X0, delta)
    # 定义解析解引导数据
    U2, V2 = analytic_solution_cartesian(k, R0, X_f, delta)

    # 定义其他时刻 100 个数组归一化数据


    # 损失函数权重
    lr_ic = 0.1
    lr_pde = 1.0
    lr_norm = 1e-10
    lr_ana = 1e-10
    lr_fft = 1e-10
    lr_mom = 1e3
    lr_ene = 1e2

    n_times = 10  # <<< 新增：时刻个数
    n_fft = 16  # <<< 新增：FFT 网格边长
    # n 个均匀时刻
    time_points = torch.linspace(0, t,
                                      n_times, device=device)
    #构建一次性空间网格 (n_fft³)
    # 空间域边界 [-B, B]³
    x_lin = torch.linspace(-B, B, n_fft, device=device)
    y_lin = torch.linspace(-B, B, n_fft, device=device)
    z_lin = torch.linspace(-B, B, n_fft, device=device)

    # 构造三维笛卡尔网格
    xx, yy, zz = torch.meshgrid(x_lin, y_lin, z_lin, indexing='ij')

    # 展平成 (N_spatial, 1)
    x_flat = xx.reshape(-1, 1).float()
    y_flat = yy.reshape(-1, 1).float()
    z_flat = zz.reshape(-1, 1).float()
    # 学习率
    lr = 1e-10
    epochs = 1
    # 网络配置
    layers = [4, 64, 64, 64, 64, 2]
    model = PINN3DSpherical(layers)
    solver = Solver3DSpherical(model, B, t, Nf,
                               X_f, U2, V2,
                               x_flat, y_flat, z_flat,  n_fft, time_points,
                               k, R0, delta,
                               lr_ic, lr_pde, lr_norm, lr_ana, lr_fft, lr_mom, lr_ene)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 初始条件
    # ana_mean_density = solver.calculate_norm(U0_ana, V0_ana)
    # U0, V0 = solver.initial_wave_cartesian(X0)
    # mean_density = solver.calculate_norm(U0, V0)
    # print("ana_mean_density - mean_density =", ana_mean_density - mean_density) # 同一时刻下计算归一化值

    file_path = './save_model/model_epoch8000_9000_0711_1650.pkl' # 有这个文件则代表在此基础上训练
    start_epoch = 0
    if os.path.exists(file_path):
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 覆盖旧学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        mean_density = checkpoint['mean_density']
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Loaded model weights from {file_path}")

    # 训练模型
    history = solver.train(epochs=epochs, initial_batch_size=3000, optimizer=optimizer, resample_every=10)

    # 保存模型到一个带时间戳的文件中
    now = datetime.now().strftime('%m%d_%H%M')
    save_path = f'./save_model/model_epoch{start_epoch}_{start_epoch+epochs}_{now}.pkl'

    torch.save({
        'epoch': start_epoch + epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 'mean_density': mean_density,
        # 'X0': X0,
        'X_f': X_f,
        'final_loss': history,
    }, save_path)

    print(f"Model saved to: {save_path}")


    # 测试点 只看 z=0; t=0.5截面
    predict_and_plot(solver, B, k, R0, delta)
    # 绘制 z=0全时空实部、虚部和振幅偏差
    calculate_and_plot_diffs(solver, B, k, R0, delta)
    # 绘制几率、能量、动量随时间变化
    pro_p_ene_plot(solver, k, R0, delta, m)

    # === 保存参数信息到 txt 文件 ===
    param_txt_path = save_path.replace('.pkl', '_params.txt')

    with open(param_txt_path, 'w') as f:
        f.write("=== 模型训练参数记录 ===\n")
        f.write(f"模型保存时间：{now}\n")
        f.write(f"模型路径：{save_path}\n")
        f.write(f"初始 epoch：{start_epoch}\n")
        f.write(f"总训练 epoch：{epochs}\n")
        f.write(f"最终 epoch：{start_epoch + epochs}\n")
        f.write(f"学习率：{lr}\n")
        f.write(f"网络结构：{layers}\n")
        f.write("\n=== 物理参数 ===\n")
        f.write(f"质量 m：{m}\n")
        f.write(f"波包宽度 delta：{delta}\n")
        f.write(f"初始动量 k：{k}\n")
        f.write(f"初始位置 R0：{R0}\n")
        f.write(f"空间边界 B：{B}\n")
        f.write(f"时间终点 t：{t}\n")
        f.write("\n=== 损失函数权重 ===\n")
        f.write(f"lr_ic: {lr_ic}\n")
        f.write(f"lr_pde: {lr_pde}\n")
        f.write(f"lr_norm: {lr_norm}\n")
        f.write(f"lr_ana: {lr_ana}\n")
        f.write(f"lr_fft: {lr_fft}\n")
        f.write(f"lr_mom: {lr_mom}\n")
        f.write(f"lr_ene: {lr_ene}\n")
        f.write("\n=== FFT / 网格参数 ===\n")
        f.write(f"n_fft: {n_fft}\n")
        f.write(f"n_times: {n_times}\n")
        f.write(f"time_points: {time_points.cpu().numpy()}\n")
        f.write(f"x_flat.shape: {x_flat.shape}\n")
        f.write("\n=== 其它参数 ===\n")
        # f.write(f"n_per_dim: {n_per_dim}\n")  # 每维取几个点（4维空间中每维取10个点 -> 共 10^4 = 10000 个点）
        # f.write(f"X0.shape: {X0.shape}\n")
        f.write(f"X_f.shape: {X_f.shape}\n")
        # f.write(f"解析解数组数量: {len(arrays)}\n")

    print(f"参数信息保存到: {param_txt_path}")



