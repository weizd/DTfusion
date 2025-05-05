import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

R = 500.0
geom = dde.geometry.Rectangle(xmin=[0.0, 0.0], xmax=[R, 2*np.pi])  # [r, φ] 域
timedomain = dde.geometry.TimeDomain(0.0, 1.0)                    # 时间域 t∈[0,1]
space_time = dde.geometry.GeometryXTime(geom, timedomain)          # 组合几何与时间


def schrodinger_pde_polar(x, y):
    """
    x: (N,3) -> [r, φ, t]
    y: (N,2) -> [ψ_r, ψ_i]
    """
    psi_r, psi_i = y[:, 0:1], y[:, 1:2]

    # 时间导数
    psi_r_t = dde.grad.jacobian(y, x, i=0, j=2)
    psi_i_t = dde.grad.jacobian(y, x, i=1, j=2)

    # r 和 φ 的一阶、二阶导
    psi_r_r   = dde.grad.jacobian(y, x, i=0, j=0)
    psi_r_rr  = dde.grad.hessian(y, x, i=0, j=0)
    psi_r_pp  = dde.grad.hessian(y, x, i=0, j=1)

    psi_i_r   = dde.grad.jacobian(y, x, i=1, j=0)
    psi_i_rr  = dde.grad.hessian(y, x, i=1, j=0)
    psi_i_pp  = dde.grad.hessian(y, x, i=1, j=1)

    r = x[:, 0:1]

    # 极坐标拉普拉斯：ψ_rr + (1/r) ψ_r + (1/r^2) ψ_φφ
    lap_r = psi_r_rr + psi_r_r / r + psi_r_pp / (r**2)
    lap_i = psi_i_rr + psi_i_r / r + psi_i_pp / (r**2)

    # 构造 PDE 残差：iψ_t + ½ Δψ = 0(正号负号???)
    res_r = psi_i_t + 0.5 * lap_r
    res_i = -psi_r_t + 0.5 * lap_i

    return [res_r, res_i]  # PINN 的 PDE 残差

def initial_wave_polar(x, r0=0.0, delta=0.5,
                            k0=1.0):
    r, phi = x[:,0:1], x[:,1:2]
    A = (2*np.pi*delta**2)**(-3/4)
    envelope = np.exp(-((r-r0)**2)/(4*delta**2))
    phase_r = A * envelope * np.cos(k0 * r * np.cos(phi))
    phase_i = A * envelope * np.sin(k0 * r * np.cos(phi))
    return phase_r, phase_i

def analytic_solution_polar(x, t, r0=0.0, delta=0.5,
                            k0=1.0,  # 径向动量大小
                            m=1.0):
    """
    自由粒子在二维极坐标下高斯波包的解析解（Hartree 单位制：ℏ=1, m=1）。

    参数
    ----
    r    : array_like, shape (N,)
        径向坐标。
    phi  : array_like, shape (N,)
        角坐标，单位为弧度。
    t    : float or array_like, shape (N,)
        时间点。
    r0   : float
        初始包心在径向的偏移（通常取 0）。
    delta: float
        初始高斯包宽度。
    k0   : float
        初始动量大小，沿径向方向。
    m    : float
        质量，Hartree 单位制下取 1。
    返回
    ----
    psi  : ndarray, shape (N,)
        复数波函数值 ψ(r,φ,t)。
    """
    # x: (N,3)，是 [r,phi,t]

    # 确保输入形状一致
    r = x[:, 0].reshape(-1, 1)
    phi = x[:, 1].reshape(-1, 1)
    t = np.array(t)
    # 1. 宽度演化 σ = Δ^2 + i t / (2m)
    sigma = delta ** 2 + 1j * t / (2 * m)
    # 2. 归一化因子
    norm = (2 * np.pi * (delta + 1j * t / (2 * m * delta)) ** 2) ** (-3/4)
    # 3. 极坐标 → 笛卡尔转换
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    # 4. 平移项：径向动量 k0 沿径向传播
    #    移动距离 = (k0/m) * t
    dr = (k0 / m) * t
    # 由极坐标可推出 x_shift, y_shift
    x0 = r0 * np.cos(phi)
    y0 = r0 * np.sin(phi)
    xs = x0 + dr * np.cos(phi)
    ys = y0 + dr * np.sin(phi)
    # 5. 计算 (x-xs)^2+(y-ys)^2
    dx2 = (x - xs) ** 2 + (y - ys) ** 2
    # 6. 相位项： k0·r - (k0^2/(2m))t
    #    因为 k0 目前纯径向，k·r = k0 * r
    phase_arg = k0 * r - (k0 ** 2 / (2 * m)) * t
    # 7. 构造解析解
    psi = norm * np.exp(-dx2 / (4 * sigma) + 1j * phase_arg)
    return psi.flatten()

# 在 t=0 时刻强制初始条件
ic_real = dde.icbc.IC(space_time, lambda x: initial_wave_polar(x)[0], lambda _, on_initial: on_initial)
ic_imag = dde.icbc.IC(space_time, lambda x: initial_wave_polar(x)[1], lambda _, on_initial: on_initial)

# # 边界条件：r=R Dirichlet，φ 周期
# def boundary_sphere(x, on_boundary):
#     return on_boundary & np.isclose(x[0], R)
#
# bc_real = dde.DirichletBC(space_time, lambda x: 0, boundary_sphere, component=0)
# bc_imag = dde.DirichletBC(space_time, lambda x: 0, boundary_sphere, component=1)
# bc_phi = dde.PeriodicBC(space_time, [1], [0], component=0)


net = dde.maps.FNN([3] + [50]*3 + [2], "relu", "Glorot uniform")  # 输入 3 维
data = dde.data.TimePDE(
    space_time,
    schrodinger_pde_polar,
    [ic_real, ic_imag],
    num_domain=200,        # 域内残差点
    num_boundary=10,       # 边界点
    num_initial=10,        # 初始时刻点
    num_test=200,
    train_distribution="Hammersley")

model = dde.Model(data, net)
model.compile("adam", lr=1e-3, loss="MSE")
losshistory, train_state = model.train(epochs=2, iterations=2000)
dde.saveplot(losshistory, train_state, issave=False, isplot=True)

# model.compile("L-BFGS")
# model.train()


# 测试点
t_test = 0.5
N = 100
r_test = np.linspace(0, 5, N) #拉普拉斯算子存在导致r不能取太小
phi_test = 0.5*np.full((1, N), np.pi)  # 例如只看 φ=pi 截面
r_phi_test = np.vstack([r_test, phi_test]).T

# 解析解
psi_exact = analytic_solution_polar(r_phi_test, t_test)

# PINN 预测
X_test = np.hstack([r_phi_test, t_test*np.ones((N,1))])
y_pred = model.predict(X_test)                          # 输出 (N,2)
psi_pinn = y_pred[:,0] + 1j*y_pred[:,1]


# 绘图对比
plt.figure(figsize=(8,4))
plt.plot(r_test, psi_exact.real, '--', label='Analytic Real', color='red', linewidth=2)
plt.plot(r_test, psi_pinn.real,  '-', label='PINN Real', color='red', linewidth=2)
plt.plot(r_test, psi_exact.imag, '--', label='Analytic Imag', color='black', linewidth=2)
plt.plot(r_test, psi_pinn.imag,  '-', label='PINN Imag', color='black', linewidth=2)
plt.xlabel('x') ; plt.ylabel(r'$\psi$') ; plt.title(f'Wavefunction slice at phi=0.5*pi, t={t_test}')
plt.legend(); plt.tight_layout(); plt.show()


real_diffs = []
imag_diffs = []
amp_diffs = []

# 定义空间和时间范围
N_new = 100
r = np.linspace(0, 5, N_new) #拉普拉斯算子存在导致r不能取太小
phi_test = 0.5*np.full((1, N), np.pi)  # 例如只看 φ=pi 截面
t_values = np.linspace(0, 1, N_new)  # 所有时间点

for t in t_values:
    t = np.array([[t]])  # 当前时间点
    r_phi = np.vstack([r_test, phi_test]).T
    X_test = np.hstack([r_phi, t.repeat(N, axis=0)])
    y_pred = model.predict(X_test)
    psi_pinn = y_pred[:, 0] + 1j * y_pred[:, 1]
    # psi_pinn = psi_pinn.reshape(-1, 1)
    psi_exact = analytic_solution_polar(r_phi, t)


    real_diff = np.real(psi_pinn - psi_exact)
    imag_diff = np.imag(psi_pinn - psi_exact)
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
    ax.set_ylabel('x')
    ax.set_title(labels[i])
    plt.colorbar(im, ax=ax)

plt.show()
