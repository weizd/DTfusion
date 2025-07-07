import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
import torch
# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", device)


def analytic_solution_polar(k, R0, X_f, delta, m=1):
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
    psi_real = psi.real.cpu().detach().numpy()
    psi_imag = psi.imag.cpu().detach().numpy()

    return psi_real, psi_imag

# 初始条件
def initial_wave_spherical(k, R0, X0, delta=1):
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


def predict_and_plot(R, k, R0, delta):
    N = 100
    phi = np.pi/6
    theta = np.pi
    t = 0
    r = np.linspace(0, R, N)
    phi_test = np.full((1, N), phi)
    theta_test = np.full((1, N), theta)
    t_test = np.full((1, N), t)
    r_phi_theta_test = np.vstack([r, theta_test, phi_test, t_test]).T
    # 解析解
    psi_exact_real, psi_exact_imag = analytic_solution_polar(k, R0, r_phi_theta_test, delta)
    # 初始值解
    psi_ini_real, psi_ini_imag = initial_wave_spherical(k, R0, r_phi_theta_test)


    # 绘图对比
    plt.figure(figsize=(8, 4))
    plt.plot(r, psi_exact_real, '-.', label='Analytic Real', color='red', linewidth=2)
    plt.plot(r, psi_ini_real, '--', label='ini Real', color='red', linewidth=2)
    plt.plot(r, psi_exact_imag, '-.', label='Analytic Imag', color='black', linewidth=2)
    plt.plot(r, psi_ini_imag, '--', label='ini Imag', color='black', linewidth=2)
    plt.xlabel('r')
    plt.ylabel(r'$\psi$')
    plt.title(f'Wavefunction slice at phi={phi:.2f}, theta={theta}, t={t}')
    plt.legend()
    plt.tight_layout()
    plt.show()

R = 1.0
delta = 1.0
# 其他数据
k = np.array([4*R, np.pi, 0.0]) # 初始动量
R0 = np.array([R, 0.0, 0.0]) # 起始坐标
predict_and_plot(R, k, R0, delta)
