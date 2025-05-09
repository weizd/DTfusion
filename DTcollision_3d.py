import deepxde as dde
import torch as th
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# 最大半径
R = 5.0
# 球坐标域：r ∈ [0,R], θ ∈ [0,π], φ ∈ [0,2π]
geom = dde.geometry.Cuboid(xmin=[0.0, 0.0, 0.0], xmax=[R, np.pi, 2*np.pi])
# 时间域 t ∈ [0,T]
T = 1.0
timedomain = dde.geometry.TimeDomain(0.0, T)
# 合成时空域
space_time = dde.geometry.GeometryXTime(geom, timedomain)


def schrodinger_pde_spherical(x, y):
    """
    x: (N,4) -> [r, θ, φ, t]
    y: (N,2) -> [ψ_r, ψ_i]
    """
    # 网络输出
    psi_r = y[:, 0:1]
    psi_i = y[:, 1:2]

    # 时间导数
    psi_r_t = dde.grad.jacobian(y, x, i=0, j=3)
    psi_i_t = dde.grad.jacobian(y, x, i=1, j=3)

    # 对 r, θ, φ 求导
    psi_r_r   = dde.grad.jacobian(y, x, i=0, j=0)
    psi_r_rr  = dde.grad.hessian(y, x, i=0, j=0)
    psi_r_th  = dde.grad.jacobian(y, x, i=0, j=1)
    psi_r_thth = dde.grad.hessian(y, x, i=0, j=1)
    psi_r_pp  = dde.grad.hessian(y, x, i=0, j=2)

    psi_i_r   = dde.grad.jacobian(y, x, i=1, j=0)
    psi_i_rr  = dde.grad.hessian(y, x, i=1, j=0)
    psi_i_th  = dde.grad.jacobian(y, x, i=1, j=1)
    psi_i_thth = dde.grad.hessian(y, x, i=1, j=1)
    psi_i_pp  = dde.grad.hessian(y, x, i=1, j=2)

    # 坐标分量
    r     = x[:, 0]
    theta = x[:, 1]
    sin_t = th.sin(theta)
    sin2_t = sin_t**2

    # 构造 Laplacian (实/虚部共用结构)
    lap_r = (
        psi_r_rr
        + 2/r * psi_r_r
        + (1/(r**2)) * psi_r_thth
        + (th.cos(theta)/(r**2*sin_t)) * psi_r_th
        + (1/(r**2*sin2_t)) * psi_r_pp
    )
    lap_i = (
        psi_i_rr
        + 2/r * psi_i_r
        + (1/(r**2)) * psi_i_thth
        + (th.cos(theta)/(r**2*sin_t)) * psi_i_th
        + (1/(r**2*sin2_t)) * psi_i_pp
    )

    # 实/虚部残差： iψ_t + ½Δψ = 0
    res_r = psi_i_t - 0.5 * lap_r
    res_i = -psi_r_t - 0.5 * lap_i

    return [res_r, res_i]


def initial_wave_spherical(x):
    # x: (N,4) -> [r, θ, φ, t]
    # k: (N,4) -> [k0, θ0, φ0]
    k = th.tensor([5.0, 0.0, 0.0])
    r0 = th.tensor(0.0)
    delta = th.tensor(0.5)
    m = th.tensor(1.0)

    r = th.tensor(x[:, 0])
    theta_r = th.tensor(x[:, 1])
    phi_r = th.tensor(x[:, 2])

    k0 = k[0]
    theta_k = k[1]
    phi_k= k[2]
    # 参数
    A = (2*np.pi*delta**2)**(-3/4)
    envelope = np.exp(-((r - r0)**2)/(4*delta**2))
    # 极坐标系下：k·r = k0 * r * (sinφ_k * sinφ_r  * cos(θ_k-θ_r) + cosφ_k * cosφ_r)
    phase_arg = k0 * r * (th.sin(phi_k)* th.sin(phi_r) * th.cos(theta_k-theta_r) + th.cos(theta_k) * th.cos(theta_r))
    psi_r = A * envelope * th.cos(phase_arg)
    psi_i = A * envelope * th.sin(phase_arg)
    return psi_r, psi_i


# 在 t=0 时刻强制初始条件
ic_real = dde.icbc.IC(space_time,
                   lambda x: initial_wave_spherical(x)[0],
                   lambda _, on_i: on_i)
ic_imag = dde.icbc.IC(space_time,
                   lambda x: initial_wave_spherical(x)[1],
                   lambda _, on_i: on_i)


def analytic_solution_polar(x):
    # 自由粒子在三维极坐标下高斯波包的解析解（Hartree 单位制：ℏ=1, m=1）。
    # x: (N,4) -> [r, θ, φ, t]
    # k: (N,4) -> [k0, θ0, φ0]
    # r0: 默认与r方向一致

    k = th.tensor([5.0, 0.0, 0.0])
    r0 = th.tensor(0.0)
    delta = th.tensor(0.5)
    m = th.tensor(1.0)

    r = th.tensor(x[:, 0])
    theta_r = th.tensor(x[:, 1])
    phi_r = th.tensor(x[:, 2])
    t = th.tensor(x[:, 3])

    k0 = k[0]
    theta_k = k[1]
    phi_k= k[2]

    # 1. 宽度演化 σ = Δ^2 + i t / (2m)
    sigma = delta ** 2 + 1j * t / (2 * m)
    # 2. 归一化因子
    norm = (2 * th.pi * (delta + 1j * t / (2 * m * delta)) ** 2) ** (-3/4)
    # 3. 极坐标 → 笛卡尔转换
    x_r = r * th.sin(theta_r) * th.cos(phi_r)
    y_r = r * th.sin(theta_r) * th.sin(phi_r)
    z_r = r * th.cos(theta_r)

    x_k = k0 * th.sin(theta_k) * th.cos(phi_k)
    y_k = k0 * th.sin(theta_k) * th.sin(phi_k)
    z_k = k0 * th.cos(theta_k)
    # 4. 平移项：径向动量 k0 沿径向传播(z轴)
    #    移动距离 = (k0/m) * t
    drx = (x_k / m) * t
    dry = (y_k / m) * t
    drz = (z_k / m) * t
    # 由极坐标可推出 x_shift, y_shift
    x0 = r0 * th.sin(theta_r) * th.cos(phi_r)
    y0 = r0 * th.sin(theta_r) * th.sin(phi_r)
    z0 = r0 * th.cos(theta_r)

    xs = x0 + drx
    ys = y0 + dry
    zs = z0 + drz
    # 5. 计算 (x-xs)^2+(y-ys)^2
    dx2 = (x_r - xs) ** 2 + (y_r - ys) ** 2 + (z_r - zs) ** 2
    # 6. 相位项： k·r - (k0^2/(2m))t
    # 极坐标系下：k·r = k0 * r * (sinφ_k * sinφ_r  * cos(θ_k-θ_r) + cosφ_k * cosφ_r)
    phase_arg = (k0 * r * (th.sin(phi_k)* th.sin(phi_r) * th.cos(theta_k-theta_r) + th.cos(theta_k) * th.cos(theta_r))
                 - (k0 ** 2 / (2 * m)) * t)
    # 7. 构造解析解
    psi = norm * np.exp(-dx2 / (4 * sigma) + 1j * phase_arg)
    psi_real = psi.real.detach().numpy()
    psi_imag = psi.imag.detach().numpy()

    return psi_real, psi_imag


net = dde.maps.FNN([4] + [500]*3 + [2], "relu", "Glorot uniform")  # 输入 4 维
data = dde.data.TimePDE(
    space_time,
    schrodinger_pde_spherical,
    [ic_real, ic_imag],
    num_domain=2000,        # 域内残差点
    # num_boundary=10,       # 边界点
    num_initial=10,        # 初始时刻点
    num_test=1000,
    # solution=analytic_solution_polar()[2],
    train_distribution="pseudo")

model = dde.Model(data, net)
model.compile("adam", lr=1e-2, loss="MSE")
losshistory, train_state = model.train(epochs=1, iterations=2000)
dde.saveplot(losshistory, train_state, issave=False, isplot=True)

# model.compile("L-BFGS")
# model.train()


# 测试点 只看 φ=pi; θ=0; t=0.5截面
N = 100
phi = np.pi
theta = 0
t = 0.5
r = np.linspace(0, R, N) # 拉普拉斯算子存在导致r不能取太小
phi_test = np.full((1, N), phi)
theta_test = np.full((1, N), theta)
t_test = np.full((1, N), t)
r_phi_theta_test = np.vstack([r, theta_test, phi_test, t_test]).T

# 解析解
psi_exact_real, psi_exact_imag= analytic_solution_polar(r_phi_theta_test)

# PINN 预测
y_pred = model.predict(r_phi_theta_test)                          # 输出 (N,2)
psi_pinn_real = y_pred[:,0]
psi_pinn_imag = y_pred[:,1]


# 绘图对比
plt.figure(figsize=(8,4))
plt.plot(r, psi_exact_real, '--', label='Analytic Real', color='red', linewidth=2)
plt.plot(r, psi_pinn_real,  '-', label='PINN Real', color='red', linewidth=2)
plt.plot(r, psi_exact_imag, '--', label='Analytic Imag', color='black', linewidth=2)
plt.plot(r, psi_pinn_imag,  '-', label='PINN Imag', color='black', linewidth=2)
plt.xlabel('r') ; plt.ylabel(r'$\psi$') ; plt.title(f'Wavefunction slice at phi={phi:.2f}, theta={theta}, t={t}')
plt.legend(); plt.tight_layout(); plt.show()


real_diffs = []
imag_diffs = []
amp_diffs = []

# 定义时间范围
t_values = np.linspace(0, 1, 100)  # 所有时间点

for t in t_values:
    t = np.array([[t]])  # 当前时间点
    t_test = np.full((1, N), t)
    r_phi_theta_test_all = np.vstack([r, theta_test, phi_test, t_test]).T
    y_pred = model.predict(r_phi_theta_test_all)
    psi_pinn_real = y_pred[:, 0]
    psi_pinn_imag = y_pred[:, 1]

    psi_exact_real, psi_exact_imag= analytic_solution_polar(r_phi_theta_test_all)
    # psi_exact_real = psi_exact_real.detach().numpy()
    # psi_exact_imag = psi_exact_imag.detach().numpy()


    real_diff = psi_pinn_real - psi_exact_real
    imag_diff = psi_pinn_imag - psi_exact_imag
    psi_pinn = psi_pinn_real + 1j*psi_pinn_imag
    psi_exact = psi_exact_real + 1j*psi_exact_imag
    amp_diff = np.abs(psi_pinn) - np.abs(psi_exact)

    real_diffs.append(real_diff)
    imag_diffs.append(imag_diff)
    amp_diffs.append(amp_diff)

# 将列表转换为NumPy数组
real_diffs = np.stack(real_diffs, axis=1)  # 每一列代表某个时间点的x切片
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
