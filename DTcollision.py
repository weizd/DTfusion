import deepxde as dde
import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import griddata

geom = dde.geometry.TimeDomain(0, 10)
def ode_system(t, y):
    y1, y2 = y[:, 0:1], y[:, 1:]  # y1 与 y2
    dy1_dt = dde.gradients.jacobian(y, t, i=0) # 计算y1相对于t的偏导
    dy2_dt = dde.gradients.jacobian(y, t, i=1) # 计算y2相对于t的偏导
    return [dy1_dt - y2, dy2_dt + y1]  # 微分方程组的右端项

def boundary(t, on_initial):
    return np.isclose(t[0], 0)

ic1 = dde.icbc.IC(geom, lambda x: 0, boundary, component=0)
ic2 = dde.icbc.IC(geom, lambda x: 1, boundary, component=1)

def func(x):
    return np.hstack((np.sin(x), np.cos(x)))

data = dde.data.PDE(geom, ode_system, [ic1, ic2],
				    35, 2, solution=func, num_test=10)

layer_size = [1] + [3] * 1 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
model.compile(optimizer = "adam", lr=0.001, metrics=["l2 relative error"])

losshistory, train_state = model.train(iterations=20000)


dde.saveplot(losshistory, train_state, issave=False, isplot=True)

















# # 物理参数
# hbar = 1.0
# m = 1.0
#
# # 定义计算域和 IC/BC
# geom = dde.geometry.Interval(-5.0, 5.0)
# timedom = dde.geometry.TimeDomain(0.0, 1.0)
# space_time = dde.geometry.GeometryXTime(geom, timedom)
#
# # PDE 定义：实部和虚部残差
# def schrodinger_pde(x, y):
#     # 对于一维薛定谔方程，x 的形状为 (N, 2)，其中 N 是样本点的数量，第一列是空间坐标，第二列是时间坐标
#     # y形状为 (N, 2)，第一列是波函数的实部 \(\psi_r\)，第二列是虚部 \(\psi_i\)
#     # psi_r = y[:, 0:1]
#     # psi_i = y[:, 1:2]
#     psi_r_t = dde.grad.jacobian(y, x, i=0, j=1)
#     # dde.grad.jacobian(y, x, i, j) 计算输出 y 的第 i 列对输入 x 的第 j 列的雅可比矩阵。这里 j=1 表示对时间（x 的第二列）求导
#     psi_i_t = dde.grad.jacobian(y, x, i=1, j=1)
#     psi_r_xx = dde.grad.hessian(y, x, i=0, j=0)
#     # dde.grad.hessian(y, x, i, j) 计算输出 y 的第 i 列对输入 x 的第 j 列的二阶导数。这里 j=0 表示对空间（x 的第一列）求导
#     psi_i_xx = dde.grad.hessian(y, x, i=1, j=0)
#     res_r = psi_i_t - (hbar / (2 * m)) * psi_r_xx
#     res_i = -psi_r_t - (hbar / (2 * m)) * psi_i_xx
#     return [res_r, res_i]
#
# # 自由演化解析解（Gauss 波包）
# def analytic_solution(x, t, x0=0.0, k0=1.0, delta=0.5):
#     # x: array of shape (N,1), t: scalar or array
#     # 复波函数
#     sigma2 = delta**2 + 1j * hbar * t / (2 * m)
#     norm = (2 * np.pi * sigma2)**(-0.25)
#     shift = x0 + k0 * t / m
#     phase = np.exp(- (x - shift)**2 / (4 * sigma2) + 1j * (k0 * x - k0**2 * t / (2*m)))
#     return norm * phase
#
#
# # 初始条件：高斯波包
# def initial_wave_real(x):
#     print("x:", x)
#     print("ini_x_shape:", x.shape)
#     print("x[:, 0]:", x[:, 0])
#     real_part = np.exp(-((x[:, 0] - 0.0) ** 2) / (4 * 0.5 ** 2)) * np.cos(1.0 * x[:, 0:1])
#     return real_part[:, None]
#
# def initial_wave_image(x):
#     imag_part = np.exp(-((x[:, 0] - 0.0) ** 2) / (4 * 0.5 ** 2)) * np.sin(1.0 * x[:, 0:1])
#     return imag_part[:, None]
#
# ic_real = dde.icbc.IC(space_time, initial_wave_real, lambda _, on_initial: on_initial)
# ic_image = dde.icbc.IC(space_time, initial_wave_image, lambda _, on_initial: on_initial)
#
# # bc = dde.icbc.DirichletBC(space_time, lambda x: [0, 0], lambda _, on_boundary: on_boundary)
# # 边界条件：Dirichlet
# def boundary_left(x, on_boundary):
#     print("x:", x)
#     print("bou_x_shape:", x.shape)
#     print("x[0]:", x[0])
#     return on_boundary and np.isclose(x[0], -5.0)
#
# def boundary_right(x, on_boundary):
#     return on_boundary and np.isclose(x[0], 5.0)
#
# bc_left_real = dde.DirichletBC(space_time, lambda x: 0, boundary_left, component=0)
# bc_left_imag = dde.DirichletBC(space_time, lambda x: 0, boundary_left, component=1)
# bc_right_real = dde.DirichletBC(space_time, lambda x: 0, boundary_right, component=0)
# bc_right_imag = dde.DirichletBC(space_time, lambda x: 0, boundary_right, component=1)
#
# data = dde.data.TimePDE(space_time, schrodinger_pde,
#     [ic_real, ic_image, bc_left_real, bc_left_imag, bc_right_real, bc_right_imag],
#                         num_domain=100,
#                         num_boundary=10,
#                         num_initial=20,
#                         train_distribution="pseudo",)
# # 构建 PINN 模型
# net = dde.maps.FNN([2] + [50]*4 + [2], "tanh", "Glorot uniform")
# model = dde.Model(data, net)
# model.compile("adam", lr=1e-3, loss="MSE")
# model.train(epochs=5000)
# model.compile("L-BFGS")
# model.train()
#
# # 测试与对比
# # 采样测试点
# t_test = 0.5
# x_test = np.linspace(-5, 5, 200)[:, None]
# X_test = np.hstack([x_test, t_test * np.ones_like(x_test)])
# print("X_test:", X_test)
#
# # PINN 预测
# y_pred = model.predict(X_test)
# psi_pinn = y_pred[:, 0] + 1j * y_pred[:, 1]
#
# # 解析解
# psi_anal = analytic_solution(x_test, t_test, x0=0.0, k0=1.0, delta=0.5)
#
# # 误差评估
# error = np.linalg.norm(psi_pinn - psi_anal, 2) / np.linalg.norm(psi_anal, 2)
# print(f"Relative L2 error at t={t_test}: {error:.2e}")
#
# # 可视化对比
# plt.figure(figsize=(8,4))
# plt.plot(x_test, np.real(psi_anal), '--', label='Analytic Real')
# plt.plot(x_test, np.real(psi_pinn), '-', label='PINN Real')
# plt.plot(x_test, np.imag(psi_anal), '--', label='Analytic Imag')
# plt.plot(x_test, np.imag(psi_pinn), '-', label='PINN Imag')
# plt.legend()
# plt.xlabel('x')
# plt.title(f'Wavefunction at t={t_test}')
# plt.show()
