import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 物理参数
m = 1.0

# 空间域：半径为 R 的三维球体
R = 5.0
geom = dde.geometry.Hypersphere([0.0, 0.0, 0.0], R)  # 三维球形域，中心在原点，半径 R
# 时间域：t ∈ [0, T]
T = 1.0
timedom = dde.geometry.TimeDomain(0.0, T)             # 时间区间

# 时空域：几何 × 时间
space_time = dde.geometry.GeometryXTime(geom, timedom)  # 组合几何与时间
def schrodinger_pde_3d(x, y):
    # x: (N,4) -> [x,y,z,t], y: (N,2) -> [ψ_r, ψ_i]
    psi_r = y[:, 0:1]
    psi_i = y[:, 1:2]
    # 时间导数
    psi_r_t = dde.grad.jacobian(y, x, i=0, j=3)
    psi_i_t = dde.grad.jacobian(y, x, i=1, j=3)
    # 空间二阶导数（拉普拉斯）
    psi_r_xx = (
        dde.grad.hessian(y, x, i=0, j=0)
      + dde.grad.hessian(y, x, i=0, j=1)
      + dde.grad.hessian(y, x, i=0, j=2)
    )
    psi_i_xx = (
        dde.grad.hessian(y, x, i=1, j=0)
      + dde.grad.hessian(y, x, i=1, j=1)
      + dde.grad.hessian(y, x, i=1, j=2)
    )
    # 构造残差
    res_r = psi_i_t - (1 / (2*m)) * psi_r_xx
    res_i = -psi_r_t - (1 / (2*m)) * psi_i_xx
    return [res_r, res_i]  # PINN 的 PDE 约束项


# 初始高斯波包参数
x0 = [0.0, 0.0, 0.0]   # 中心
delta = 0.5
k0 = [1.0, 0.0, 0.0]   # 平面波方向

def analytic_solution_3d(x, t, x0 = [0.0, 0.0, 0.0], delta = 0.5, k0 = [1.0, 0.0, 0.0]):
    """
    x: numpy array of shape (N,3), 每行是 [x,y,z]
    t: 标量或长度为 N 的数组
    返回：复数 numpy 数组 shape (N,t)
    """
    m = 1.0
    x = x[:, 0].reshape(-1, 1)
    t = np.array(t)
    k0 = np.array(k0)
    sigma2 = (delta + 1j * t / (2*m*delta))**2
    norm = (2*np.pi*sigma2)**(-3/4)

    # 平移项
    shift = (k0 * t)/m
    diff = x - np.array(x0) - shift
    r2 = np.sum(diff**2, axis=1, keepdims=True)
    # 相位项
    fir = -r2 / (4*(delta**2 + 1j*t/(2*m)))
    sec = 1j*np.sum(k0*(x - shift/2), axis=1)
    phase = np.exp(fir + sec.reshape(-1, 1))
    return (norm * phase).flatten()

def initial_wave_3d(x):
    # x: (N,4)，其中 x[:,0:3] 是 [x,y,z]，x[:,3] = t
    X = x[:, 0:3]
    # 计算空间偏移和高斯衰减
    diff = X - np.array(x0)
    r2 = np.sum(diff**2, axis=1, keepdims=True)
    envelope = np.exp(-r2 / (4 * delta**2))
    norm = (2 * np.pi * delta**2) ** (-3 / 4)
    phase = np.dot(X, np.array(k0)).reshape(-1, 1)  # k0·x
    psi_r = norm * envelope * np.cos(phase)
    psi_i = norm * envelope * np.sin(phase)
    return psi_r, psi_i  # 返回 (实部, 虚部) 两个数组

# 在 t=0 时刻强制初始条件
ic_real = dde.icbc.IC(space_time, lambda x: initial_wave_3d(x)[0], lambda _, on_initial: on_initial)
ic_imag = dde.icbc.IC(space_time, lambda x: initial_wave_3d(x)[1], lambda _, on_initial: on_initial)




net = dde.maps.FNN([4] + [50]*4 + [2], "tanh", "Glorot uniform")  # 输入 4 维
data = dde.data.TimePDE(
    space_time,
    schrodinger_pde_3d,
    [ic_real, ic_imag],
    num_domain=2000,        # 域内残差点
    num_boundary=50,       # 边界点
    num_initial=50,        # 初始时刻点
    num_test=100,
    train_distribution="Hammersley")

model = dde.Model(data, net)
model.compile("adam", lr=1e-3, loss="MSE")
losshistory, train_state = model.train(epochs=2, iterations=200)
dde.saveplot(losshistory, train_state, issave=False, isplot=True)

# model.compile("L-BFGS")
# model.train()



# 选取 t_test 时间和 y=z=0 截面
t_test = 0.5
N = 200
x_line = np.linspace(-5, 5, N)
xyz = np.vstack([x_line, np.zeros(N), np.zeros(N)]).T

# PINN 预测
X_test = np.hstack([xyz, t_test*np.ones((N,1))])
y_pred = model.predict(X_test)                          # 输出 (N,2)
psi_pinn = y_pred[:,0] + 1j*y_pred[:,1]

# 解析解
psi_exact = analytic_solution_3d(xyz, t_test)

# 绘图对比
plt.figure(figsize=(8,4))
plt.plot(x_line, psi_exact.real, '--', label='Analytic Real', color='red', linewidth=4)
plt.plot(x_line, psi_pinn.real,  '-', label='PINN Real', color='red', linewidth=4)
plt.plot(x_line, psi_exact.imag, '--', label='Analytic Imag', color='black', linewidth=4)
plt.plot(x_line, psi_pinn.imag,  '-', label='PINN Imag', color='black', linewidth=4)
plt.xlabel('x') ; plt.ylabel(r'$\psi$') ; plt.title(f'Wavefunction slice at y=z=0, t={t_test}')
plt.legend(); plt.tight_layout(); plt.show()



real_diffs = []
imag_diffs = []
amp_diffs = []

# 定义空间和时间范围
N = 200
x = np.linspace(-5, 5, N)
t_values = np.linspace(0, 1, N)  # 所有时间点

for t in t_values:
    t = np.array([[t]])  # 当前时间点
    xyz = np.stack([x, np.zeros(N), np.zeros(N)]).T
    X_test = np.hstack([xyz, t.repeat(N, axis=0)])
    y_pred = model.predict(X_test)
    psi_pinn = y_pred[:, 0] + 1j * y_pred[:, 1]
    # psi_pinn = psi_pinn.reshape(-1, 1)
    psi_exact = analytic_solution_3d(xyz, t)
    # psi_exact = psi_exact.reshape(-1, 1)  # 确保形状匹配

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
    im = ax.imshow(diffs[i], extent=[t_values.min(), t_values.max(), x.min(), x.max()],
                   aspect='auto', origin='lower', cmap=cmaps[i])
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('x')
    ax.set_title(labels[i])
    plt.colorbar(im, ax=ax)

plt.show()
