import os
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from time import time
from pyDOE import lhs

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    def forward(self, r, theta, phi, t):
        inp = torch.cat([r, theta, phi, t], dim=1)
        out = self.net(inp)
        u = out[:, 0:1]
        v = out[:, 1:2]
        return u, v

# 求解器
class Solver3DSpherical:
    def __init__(self, model, X0, U0, V0, X_f, X1, mean_density):
        self.model = model.to(device)
        # 初始时刻归一化数值
        self.mean_density = mean_density
        # 初始条件数据
        self.r0 = torch.tensor(X0[:, 0:1], dtype=torch.float32, device=device)
        self.theta0 = torch.tensor(X0[:, 1:2], dtype=torch.float32, device=device)
        self.phi0 = torch.tensor(X0[:, 2:3], dtype=torch.float32, device=device)
        self.t0 = torch.tensor(X0[:, 3:4], dtype=torch.float32, device=device)
        self.u0 = torch.tensor(U0, dtype=torch.float32, device=device)
        self.v0 = torch.tensor(V0, dtype=torch.float32, device=device)
        # 残差点
        self.rf = torch.tensor(X_f[:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
        self.thetaf = torch.tensor(X_f[:, 1:2], dtype=torch.float32, device=device, requires_grad=True)
        self.phif = torch.tensor(X_f[:, 2:3], dtype=torch.float32, device=device, requires_grad=True)
        self.tf = torch.tensor(X_f[:, 3:4], dtype=torch.float32, device=device, requires_grad=True)
        # 其他时刻归一化数据点
        self.r1 = torch.tensor(X1[:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
        self.theta1 = torch.tensor(X1[:, 1:2], dtype=torch.float32, device=device, requires_grad=True)
        self.phi1 = torch.tensor(X1[:, 2:3], dtype=torch.float32, device=device, requires_grad=True)
        self.t1 = torch.tensor(X1[:, 3:4], dtype=torch.float32, device=device, requires_grad=True)


    def schrodinger_residual(self):

        psi_r, psi_i = self.model(self.rf, self.thetaf, self.phif, self.tf)
        # 一阶导数
        grads = torch.autograd.grad(
            psi_r, [self.rf, self.thetaf, self.phif, self.tf],
            grad_outputs=torch.ones_like(psi_r), create_graph=True
        ) + torch.autograd.grad(
            psi_i, [self.rf, self.thetaf, self.phif, self.tf],
            grad_outputs=torch.ones_like(psi_i), create_graph=True
        )
        psi_r_r, psi_r_th, psi_r_p, psi_r_t, psi_i_r, psi_i_th, psi_i_p, psi_i_t = grads
        # 二阶导数
        psi_r_rr = torch.autograd.grad(psi_r_r, self.rf, grad_outputs=torch.ones_like(psi_r_r), create_graph=True)[0]
        psi_r_thth = torch.autograd.grad(psi_r_r, self.thetaf, grad_outputs=torch.ones_like(psi_r_r), create_graph=True)[0]
        psi_r_pp = torch.autograd.grad(psi_r_r, self.phif, grad_outputs=torch.ones_like(psi_r_r), create_graph=True)[0]

        psi_i_rr = torch.autograd.grad(psi_i_r, self.rf, grad_outputs=torch.ones_like(psi_i_r), create_graph=True)[0]
        psi_i_thth = torch.autograd.grad(psi_i_r, self.thetaf, grad_outputs=torch.ones_like(psi_i_r), create_graph=True)[0]
        psi_i_pp = torch.autograd.grad(psi_i_r, self.phif, grad_outputs=torch.ones_like(psi_i_r), create_graph=True)[0]

        # 坐标分量
        r = self.rf
        theta = self.thetaf
        sin_t = torch.sin(theta)
        sin2_t = sin_t ** 2

        # 构造 Laplacian (实/虚部共用结构)
        lap_r = (
                psi_r_rr
                + 2 / r * psi_r_r
                + (1 / (r ** 2)) * psi_r_thth
                + (torch.cos(theta) / (r ** 2 * sin_t)) * psi_r_th
                + (1 / (r ** 2 * sin2_t)) * psi_r_pp
        )

        lap_i = (
                psi_i_rr
                + 2 / r * psi_i_r
                + (1 / (r ** 2)) * psi_i_thth
                + (torch.cos(theta) / (r ** 2 * sin_t)) * psi_i_th
                + (1 / (r ** 2 * sin2_t)) * psi_i_pp
        )
        # 实/虚部残差： iψ_t + ½Δψ = 0
        res_r = psi_i_t - 0.5 * lap_r
        # res_r_num = res_r.detach().numpy()
        res_i = -psi_r_t - 0.5 * lap_i

        threshold = 1e2
        # 对res_r中大于threshold的值设为0
        res_r = torch.where(torch.abs(res_r) > threshold, torch.tensor(0.0, device=res_r.device), res_r)
        # res_r_num = res_r.detach().numpy()
        # 对res_i中大于threshold的值设为0
        res_i = torch.where(torch.abs(res_i) > threshold, torch.tensor(0.0, device=res_i.device), res_i)

        return res_r, res_i

    def loss(self):
        # 初始条件损失
        u0_pred, v0_pred = self.model(self.r0, self.theta0, self.phi0, self.t0)
        mse_ic = torch.mean((u0_pred - self.u0)**2) + torch.mean((v0_pred - self.v0)**2)
        # 归一化损失
        psi_r, psi_i = self.model(self.r1, self.theta1, self.phi1, self.t1)
        density1 = psi_r ** 2 + psi_i ** 2
        mean_density1 = torch.sqrt(torch.sum(density1))
        mse_norm = (mean_density1 - self.mean_density)**2
        # 残差
        res_r, res_i = self.schrodinger_residual()
        mse_pde = torch.mean(res_r**2) + torch.mean(res_i**2)
        loss = mse_ic + mse_pde + mse_norm
        return loss, mse_ic, mse_pde, mse_norm

    def train(self, epochs, lr=1e-3):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=100, verbose=True)
        history = {'loss': [], 'ic': [], 'pde': [], 'norm': []}
        start = time()
        for ep in range(1, epochs+1):
            optimizer.zero_grad()
            loss, ic, pde, norm = self.loss()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            history['loss'].append(loss.item())
            history['ic'].append(ic.item())
            history['pde'].append(pde.item())
            history['norm'].append(norm.item())
            if ep % 1 == 0:
                print(f"Epoch {ep}/{epochs}, Loss: {loss.item():.4e}, IC: {ic.item():.4e}, PDE: {pde.item():.4e}, norm: {norm.item():.4e}")
        print(f"Training completed in {time()-start:.1f}s")
        return history

def analytic_solution_polar(k, R0, X_f, delta=1, m=1):
    # 自由粒子在三维极坐标下高斯波包的解析解（Hartree 单位制：ℏ=1, m=1）。
    # x: (N,4) -> [r, θ, φ, t]
    # k: (N,3) -> [k0, θ0, φ0]
    # r0: 默认与r方向一致
    delta = torch.tensor(delta, dtype=torch.float32, device=device)
    k0 = torch.tensor(k[0], dtype=torch.float32, device=device)
    theta_k = torch.tensor(k[1], dtype=torch.float32, device=device)
    phi_k = torch.tensor(k[2], dtype=torch.float32, device=device)

    r = torch.tensor(X_f[:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
    theta_r = torch.tensor(X_f[:, 1:2], dtype=torch.float32, device=device, requires_grad=True)
    phi_r = torch.tensor(X_f[:, 2:3], dtype=torch.float32, device=device, requires_grad=True)
    t = torch.tensor(X_f[:, 3:4], dtype=torch.float32, device=device, requires_grad=True)

    r0 = torch.tensor(R0[0], dtype=torch.float32, device=device)
    theta_r0 = torch.tensor(R0[1], dtype=torch.float32, device=device)
    phi_r0 = torch.tensor(R0[2], dtype=torch.float32, device=device)

    # 1. 宽度演化 σ = Δ^2 + i t / (2m)
    sigma = delta ** 2 + 1j * t / (2 * m)
    # 2. 归一化因子
    norm = (2 * torch.pi * (delta + 1j * t / (2 * m * delta)) ** 2) ** (-3 / 4)
    # 3. 极坐标 → 笛卡尔转换
    x_r = r * torch.sin(theta_r) * torch.cos(phi_r)
    y_r = r * torch.sin(theta_r) * torch.sin(phi_r)
    z_r = r * torch.cos(theta_r)

    x0 = r0 * torch.sin(theta_r0) * torch.cos(phi_r0)
    y0 = r0 * torch.sin(theta_r0) * torch.sin(phi_r0)
    z0 = r0 * torch.cos(theta_r0)

    x_k = k0 * torch.sin(theta_k) * torch.cos(phi_k)
    y_k = k0 * torch.sin(theta_k) * torch.sin(phi_k)
    z_k = k0 * torch.cos(theta_k)
    # 4. 移动距离 = (k0/m) * t
    drx = (x_k / m) * t
    dry = (y_k / m) * t
    drz = (z_k / m) * t

    xs = x0 + drx
    ys = y0 + dry
    zs = z0 + drz
    # 5. 计算 [r-(r0+(k0/m)*t)]^2
    dx2 = (x_r - xs) ** 2 + (y_r - ys) ** 2 + (z_r - zs) ** 2
    # 6. 相位项： k·r - (k0^2/(2m))t
    # 极坐标系下：k·r = k0 * r * (sinφ_k * sinφ_r  * cos(θ_k-θ_r) + cosφ_k * cosφ_r)
    phase_arg = (k0 * r * (
            torch.sin(phi_k) * torch.sin(phi_r) * torch.cos(theta_k - theta_r)
            + torch.cos(phi_k) * torch.cos(phi_r)
        )
        - (k0 ** 2 / (2 * m)) * t
    )
    # 7. 构造解析解
    psi = norm * torch.exp(-dx2 / (4 * sigma) + 1j * phase_arg)
    psi_real = psi.real.detach().numpy()
    psi_imag = psi.imag.detach().numpy()

    return psi_real, psi_imag

# 初始条件
def initial_wave_spherical(k, R0, X0, delta):
    # x: (N,4) -> [r, θ, φ, t]
    # k: (N,3) -> [k0, θ0, φ0]
    delta = torch.tensor(delta, dtype=torch.float32, device=device)
    k0 = torch.tensor(k[0], dtype=torch.float32, device=device)
    theta_k = torch.tensor(k[1], dtype=torch.float32, device=device)
    phi_k = torch.tensor(k[2], dtype=torch.float32, device=device)

    r = torch.tensor(X0[:, 0:1], dtype=torch.float32, device=device)
    theta_r = torch.tensor(X0[:, 1:2], dtype=torch.float32, device=device)
    phi_r = torch.tensor(X0[:, 2:3], dtype=torch.float32, device=device)

    r0 = torch.tensor(R0[0], dtype=torch.float32, device=device)
    theta_r0 = torch.tensor(R0[1], dtype=torch.float32, device=device)
    phi_r0 = torch.tensor(R0[2], dtype=torch.float32, device=device)
    # 参数
    A = (2 * torch.pi * delta ** 2) ** (-3 / 4)
    # 极坐标 → 笛卡尔转换
    x_r = r * torch.sin(theta_r) * torch.cos(phi_r)
    y_r = r * torch.sin(theta_r) * torch.sin(phi_r)
    z_r = r * torch.cos(theta_r)

    x_r0 = r0 * torch.sin(theta_r0) * torch.cos(phi_r0)
    y_r0 = r0 * torch.sin(theta_r0) * torch.sin(phi_r0)
    z_r0 = r0 * torch.cos(theta_r0)
    # 计算矢量（r-r0）**2
    r_r0_2 = (x_r - x_r0) ** 2 + (y_r - y_r0) ** 2 + (z_r - z_r0) ** 2

    envelope = torch.exp(-r_r0_2 / (4 * delta ** 2))
    # 极坐标系下：k·r = k0 * r * (sinφ_k * sinφ_r  * cos(θ_k-θ_r) + cosφ_k * cosφ_r)
    phase_arg = (k0 * r * (
            torch.sin(phi_k) * torch.sin(phi_r) * torch.cos(theta_k - theta_r)
            + torch.cos(phi_k) * torch.cos(phi_r)
        )
    )
    psi_r = A * envelope * torch.cos(phase_arg)
    psi_i = A * envelope * torch.sin(phase_arg)
    return psi_r, psi_i

# 计算归一化数值
def calculate_norm(u, v):
    density = u ** 2 + v ** 2
    mean_density = torch.sqrt(torch.sum(density))
    return mean_density

# 可视化函数
def plot_history(history, save_path=None):
    # 创建一个新的图形
    plt.figure(figsize=(10, 6))  # 设置图形大小
    # 定义颜色和线条样式
    colors = ['blue', 'green', 'red', 'purple']
    line_styles = ['-', '--', '-.', ':']
    # 绘制每种损失曲线
    loss_labels = ['Total Loss', 'IC Loss', 'PDE Loss', 'Norm Loss']
    for i, (key, label) in enumerate(zip(['loss', 'ic', 'pde', 'norm'], loss_labels)):
        # 确保历史记录中有这个键
        if key in history:
            plt.semilogy(history[key], label=label, color=colors[i % len(colors)], linestyle=line_styles[i % len(line_styles)], linewidth=2)
    # 添加图表元素
    plt.xlabel('Epoch', fontsize=12)  # 设置x轴标签
    plt.ylabel('Loss', fontsize=12)   # 设置y轴标签
    plt.title('Loss History', fontsize=14)  # 设置图表标题
    plt.legend(fontsize=10)  # 设置图例
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # 添加网格
    plt.tight_layout()  # 自动调整布局
    # 保存图表（如果指定了保存路径）
    if save_path:
        plt.savefig(os.path.join(save_path, 'loss_history.png'), dpi=300, bbox_inches='tight')
    # 显示图表
    plt.show()

# 残差可视化示例（r-t 平面切片）
def plot_residual_slice(solver, r_vals, t_vals):
    R, T = np.meshgrid(r_vals, t_vals)
    X = np.hstack((R.flatten()[:,None], (np.pi/6)*np.ones((R.size,1)), np.zeros((R.size,1)), T.flatten()[:,None]))
    rf = torch.tensor(X[:,0:1], dtype=torch.float32, device=device, requires_grad=True)
    thf = torch.tensor(X[:,1:2], dtype=torch.float32, device=device, requires_grad=True)
    phf = torch.tensor(X[:,2:3], dtype=torch.float32, device=device, requires_grad=True)
    tf = torch.tensor(X[:,3:4], dtype=torch.float32, device=device, requires_grad=True)
    solver.rf, solver.thetaf, solver.phif, solver.tf = rf, thf, phf, tf
    res_r, res_i = solver.schrodinger_residual()
    res = (res_r.pow(2) + res_i.pow(2)).detach().cpu().numpy().reshape(R.shape)
    plt.figure()
    im = plt.imshow(res, origin='lower', extent=[r_vals.min(), r_vals.max(), t_vals.min(), t_vals.max()], aspect='auto')
    plt.xlabel('r')
    plt.ylabel('t')
    plt.title('Residual magnitude')
    plt.colorbar(im)
    plt.show()

# 测试点 只看 φ=pi/6; θ=0; t=0.5截面
def predict_and_plot(solver, R, k, R0):
    N = 100
    phi = np.pi/6
    theta = 0
    t = 0.5
    r = np.linspace(-R, R, N)
    phi_test = np.full((1, N), phi)
    theta_test = np.full((1, N), theta)
    t_test = np.full((1, N), t)
    r_phi_theta_test = np.vstack([r, theta_test, phi_test, t_test]).T
    # 解析解
    psi_exact_real, psi_exact_imag = analytic_solution_polar(k, R0, r_phi_theta_test)
    # PINN 预测
    rf = torch.tensor(r_phi_theta_test[:,0:1], dtype=torch.float32, device=device, requires_grad=True)
    thf = torch.tensor(r_phi_theta_test[:,1:2], dtype=torch.float32, device=device, requires_grad=True)
    phf = torch.tensor(r_phi_theta_test[:,2:3], dtype=torch.float32, device=device, requires_grad=True)
    tf = torch.tensor(r_phi_theta_test[:,3:4], dtype=torch.float32, device=device, requires_grad=True)
    psi_pinn_real, psi_pinn_imag = solver.model(rf, thf, phf, tf)
    psi_pinn_real =psi_pinn_real.detach().numpy()
    psi_pinn_imag = psi_pinn_imag.detach().numpy()
    # 绘图对比
    plt.figure(figsize=(8, 4))
    plt.plot(r, psi_exact_real, '--', label='Analytic Real', color='red', linewidth=2)
    plt.plot(r, psi_pinn_real, '-', label='PINN Real', color='red', linewidth=2)
    plt.plot(r, psi_exact_imag, '--', label='Analytic Imag', color='black', linewidth=2)
    plt.plot(r, psi_pinn_imag, '-', label='PINN Imag', color='black', linewidth=2)
    plt.xlabel('r')
    plt.ylabel(r'$\psi$')
    plt.title(f'Wavefunction slice at phi={phi:.2f}, theta={theta}, t={t}')
    plt.legend()
    plt.tight_layout()
    plt.show()

# 绘制 φ=pi/6; θ=0全时空实部、虚部和振幅偏差
def calculate_and_plot_diffs(solver, R, k, R0):
    # 定义时间范围
    t_values = np.linspace(0, 1, 100)
    N = 100
    phi = np.pi/6
    theta = 0
    r = np.linspace(-R, R, N)
    phi_test = np.full((1, N), phi)
    theta_test = np.full((1, N), theta)

    real_diffs = []
    imag_diffs = []
    amp_diffs = []

    for t in t_values:
        t_test = np.full((1, N), t)
        r_phi_theta_test_all = np.vstack([r, theta_test, phi_test, t_test]).T
        rf = torch.tensor(r_phi_theta_test_all[:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
        thf = torch.tensor(r_phi_theta_test_all[:, 1:2], dtype=torch.float32, device=device, requires_grad=True)
        phf = torch.tensor(r_phi_theta_test_all[:, 2:3], dtype=torch.float32, device=device, requires_grad=True)
        tf = torch.tensor(r_phi_theta_test_all[:, 3:4], dtype=torch.float32, device=device, requires_grad=True)
        psi_pinn_real, psi_pinn_imag = solver.model(rf, thf, phf, tf)
        psi_pinn_real = psi_pinn_real.detach().numpy()
        psi_pinn_imag = psi_pinn_imag.detach().numpy()

        psi_exact_real, psi_exact_imag = analytic_solution_polar(k, R0, r_phi_theta_test_all)
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
        im = ax.imshow(diffs[i], extent=[t_values.min(), t_values.max(), r.min(), r.max()],
                       aspect='auto', origin='lower', cmap=cmaps[i])
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('r')
        ax.set_title(labels[i])
        plt.colorbar(im, ax=ax)

    plt.show()

# 主函数示例
if __name__ == '__main__':
    R = 1.0
    t = 1.0
    # 其他数据
    k = np.array([2*R, 0.0, 0.0]) # 初始动量
    R0 = np.array([-R, 0.0, 0.0]) # 起始坐标
    delta = np.array(1.0) # 波包宽度参数
    m = np.array(1.0) # 质量
    lb = np.array([0.0, 0.0, 0.0, 0.0])       # r>=0, theta>=0, phi>=0, t>=0
    # 定义所有训练点数据
    ub = np.array([R, np.pi, 2*np.pi, t])
    Nf = 20000
    X_f = lb + (ub - lb) * lhs(4, Nf)
    # 定义初始时刻数据
    ub0 = np.array([R, np.pi, 2 * np.pi, 0])
    N0 = 1000
    X0 = lb + (ub0 - lb) * lhs(4, N0)
    # 定义其他时刻归一化数据
    ub1 = np.array([R, np.pi, 2 * np.pi, 0.5])
    N1 = 1000
    X1 = lb + (ub1 - lb) * lhs(4, N1)
    X1[:, 3] = 0.5
    # 初始条件
    U0, V0 = initial_wave_spherical(k, R0, X0, delta)
    mean_density = calculate_norm(U0, V0)

    # 网络配置
    layers = [4, 128, 128, 128, 2]
    model = PINN3DSpherical(layers)
    solver = Solver3DSpherical(model, X0, U0, V0, X_f, X1, mean_density)
    # 训练
    history = solver.train(epochs=10, lr=1e-3)
    # 可视化损失
    plot_history(history)
    # 残差平面展示
    r_vals = np.linspace(0.1, R,100)
    t_vals = np.linspace(0, t,100)
    plot_residual_slice(solver, r_vals, t_vals)
    # 测试点 只看 φ=pi/6; θ=0; t=0.5截面
    predict_and_plot(solver, R, k, R0)
    # 绘制 φ=pi/6; θ=0全时空实部、虚部和振幅偏差
    calculate_and_plot_diffs(solver, R, k, R0)


