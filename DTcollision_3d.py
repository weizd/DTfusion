import deepxde as dde
import torch as th
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

os.environ["DDE_BACKEND"] = "pytorch"  # DeepXDE 将自动切换到 PyTorch 后端



class SchrodingerEquationSolver:
    def __init__(self, space_time, k, r0, delta, m, R, T, num_domain, num_initial, num_test, iterations,
                 norm_samples, fixed_times, num_spatial_points_per_time,
                 loss_PDE_real, loss_PDE_imag, loss_nomr, loss_IC_real, loss_IC_imag, lr):

        self.lr = lr
        self.loss_PDE_real = loss_PDE_real
        self.loss_PDE_imag = loss_PDE_imag
        self.loss_nomr = loss_nomr
        self.loss_IC_real = loss_IC_real
        self.loss_IC_imag = loss_IC_imag

        self.num_spatial_points_per_time = num_spatial_points_per_time
        self.fixed_times = fixed_times
        self.norm_samples = norm_samples

        self.space_time = space_time
        self.k = k
        self.r0 = r0
        self.delta = delta
        self.m = m
        self.R = R
        self.T =T
        self.num_domain = num_domain
        self.num_initial = num_initial
        self.num_test = num_test
        self.iterations = iterations
        # 在 t=0 时刻强制初始条件
        self.ic_real = dde.icbc.IC(self.space_time,
                                   lambda x: self.initial_wave_spherical(x)[0],
                                   lambda _, on_i: on_i)
        self.ic_imag = dde.icbc.IC(self.space_time,
                                   lambda x: self.initial_wave_spherical(x)[1],
                                   lambda _, on_i: on_i)
        # 创建网络
        self.net = dde.maps.FNN([4] + [500] * 3 + [2], "relu", "Glorot uniform")
        # 创建数据对象
        self.data = dde.data.TimePDE(
            self.space_time,
            self.schrodinger_pde_spherical,
            [self.ic_real, self.ic_imag],
            num_domain=self.num_domain,
            num_initial=self.num_initial,
            num_test=self.num_test,
            train_distribution="pseudo"
        )
        # 创建模型
        self.model = dde.Model(self.data, self.net)
        self.model.compile(
            optimizer="adam",
            lr=self.lr,
            loss=["MSE", # PDE 实部残差
                  "MSE", # PDE 虚部残差
                  "MSE", # 归一化损失
                  "MSE", # IC real
                  "MSE",], # IC imag
            loss_weights=[self.loss_PDE_real, self.loss_PDE_imag, self.loss_nomr, self.loss_IC_real, self.loss_IC_imag])

        # 训练模型，添加自定义回调
        self.callback = self.NormLossCallback(self.norm_samples, self.fixed_times, self.num_spatial_points_per_time)
        losshistory, train_state = self.model.train(epochs=1, display_every=1, iterations=self.iterations, callbacks=[self.callback])
        dde.saveplot(losshistory, train_state, issave=False, isplot=True)

    class NormLossCallback(dde.callbacks.Callback):
        def __init__(self, norm_samples, fixed_times, num_spatial_points_per_time):
            super().__init__()
            self.norm_samples = norm_samples
            self.fixed_times = fixed_times
            self.num_spatial_points_per_time = num_spatial_points_per_time

        def on_epoch_end(self):
            # 获取模型的预测结果
            y_pred = self.model.predict(self.norm_samples)

            # 筛选出固定时刻下的输出值
            y_values = []
            start_idx = 0
            for _ in self.fixed_times:
                end_idx = start_idx + self.num_spatial_points_per_time
                y_values.append(y_pred[start_idx:end_idx])
                start_idx = end_idx

            # 计算归一化损失
            norm_loss_val = self.calculate_norm_loss(y_values)
            # print(f"Normalization Loss = {norm_loss_val:.6f}")
            return norm_loss_val

        def calculate_norm_loss(self, y_values):
            total_loss = 0
            for y in y_values:
                u, v = y[:, 0], y[:, 1]
                density = u ** 2 + v ** 2
                mean_density = np.mean(density)
                total_loss += (mean_density - 1.0) ** 2
            return total_loss / len(y_values)


    def schrodinger_pde_spherical(self, x, y):
        """
        x: (N,4) -> [r, θ, φ, t]
        y: (N,2) -> [ψ_r, ψ_i]
        """

        # 时间导数
        psi_r_t = dde.grad.jacobian(y, x, i=0, j=3)
        # psi_r_t_num = psi_r_t.detach().numpy()
        psi_i_t = dde.grad.jacobian(y, x, i=1, j=3)
        # psi_i_t_num = psi_i_t.detach().numpy()
        # 对 r, θ, φ 求导
        psi_r_r = dde.grad.jacobian(y, x, i=0, j=0)
        # psi_r_r_num = psi_r_r.detach().numpy()
        psi_r_rr = dde.grad.hessian(y, x, i=0, j=0)
        psi_r_th = dde.grad.jacobian(y, x, i=0, j=1)
        # psi_r_th_num = psi_r_th.detach().numpy()
        psi_r_thth = dde.grad.hessian(y, x, i=0, j=1)
        psi_r_pp = dde.grad.hessian(y, x, i=0, j=2)

        psi_i_r = dde.grad.jacobian(y, x, i=1, j=0)
        psi_i_rr = dde.grad.hessian(y, x, i=1, j=0)
        psi_i_th = dde.grad.jacobian(y, x, i=1, j=1)
        psi_i_thth = dde.grad.hessian(y, x, i=1, j=1)
        psi_i_pp = dde.grad.hessian(y, x, i=1, j=2)

        # 坐标分量
        r = x[:, 0].reshape(-1, 1)
        theta = x[:, 1].reshape(-1, 1)
        sin_t = th.sin(theta)
        sin2_t = sin_t ** 2

        # 构造 Laplacian (实/虚部共用结构)
        lap_r = (
                psi_r_rr
                + 2 / r * psi_r_r
                + (1 / (r ** 2)) * psi_r_thth
                + (th.cos(theta) / (r ** 2 * sin_t)) * psi_r_th
                + (1 / (r ** 2 * sin2_t)) * psi_r_pp
        )
        # lap_r_num = lap_r.detach().numpy()
        lap_i = (
                psi_i_rr
                + 2 / r * psi_i_r
                + (1 / (r ** 2)) * psi_i_thth
                + (th.cos(theta) / (r ** 2 * sin_t)) * psi_i_th
                + (1 / (r ** 2 * sin2_t)) * psi_i_pp
        )

        # 实/虚部残差： iψ_t + ½Δψ = 0
        res_r = psi_i_t - 0.5 * lap_r
        res_i = -psi_r_t - 0.5 * lap_i

        threshold = 1e2

        # 对res_r中大于threshold的值设为0
        res_r = th.where(th.abs(res_r) > threshold, th.tensor(0.0, device=res_r.device), res_r)
        # res_r_num = res_r.detach().numpy()
        # 对res_i中大于threshold的值设为0
        res_i = th.where(th.abs(res_i) > threshold, th.tensor(0.0, device=res_i.device), res_i)

        norm_loss = self.callback.on_epoch_end()
        norm_loss = th.from_numpy(np.array([norm_loss], dtype=np.float32))
        norm_loss_expanded = norm_loss.expand_as(res_i)

        return [res_r, res_i, norm_loss_expanded]

    def initial_wave_spherical(self, x):
        # x: (N,4) -> [r, θ, φ, t]
        # k: (N,3) -> [k0, θ0, φ0]
        k0, theta_k, phi_k = self.k[0], self.k[1], self.k[2]

        r = th.tensor(x[:, 0]).reshape(-1, 1)
        theta_r = th.tensor(x[:, 1]).reshape(-1, 1)
        phi_r = th.tensor(x[:, 2]).reshape(-1, 1)

        r0 = self.r0[0]
        theta_r0 = self.r0[1]
        phi_r0 = self.r0[2]
        # 参数
        A = (2 * th.pi * self.delta ** 2) ** (-3 / 4)
        # 极坐标 → 笛卡尔转换
        x_r = r * th.sin(theta_r) * th.cos(phi_r)
        y_r = r * th.sin(theta_r) * th.sin(phi_r)
        z_r = r * th.cos(theta_r)

        x_r0 = r0 * th.sin(theta_r0) * th.cos(phi_r0)
        y_r0 = r0 * th.sin(theta_r0) * th.sin(phi_r0)
        z_r0 = r0 * th.cos(theta_r0)
        # 计算矢量（r-r0）**2
        r_r0_2 = (x_r - x_r0) ** 2 + (y_r - y_r0) ** 2 + (z_r - z_r0) ** 2

        envelope = th.exp(-r_r0_2 / (4 * self.delta ** 2))
        # 极坐标系下：k·r = k0 * r * (sinφ_k * sinφ_r  * cos(θ_k-θ_r) + cosφ_k * cosφ_r)
        phase_arg = (k0 * r * (
                th.sin(phi_k) * th.sin(phi_r) * th.cos(theta_k - theta_r)
                + th.cos(phi_k) * th.cos(phi_r)
            )
        )
        psi_r = A * envelope * th.cos(phase_arg)
        psi_i = A * envelope * th.sin(phase_arg)
        return psi_r, psi_i

    def analytic_solution_polar(self, x):
        # 自由粒子在三维极坐标下高斯波包的解析解（Hartree 单位制：ℏ=1, m=1）。
        # x: (N,4) -> [r, θ, φ, t]
        # k: (N,3) -> [k0, θ0, φ0]
        # r0: 默认与r方向一致

        k0, theta_k, phi_k = self.k[0], self.k[1], self.k[2]

        r = th.tensor(x[:, 0])
        theta_r = th.tensor(x[:, 1])
        phi_r = th.tensor(x[:, 2])
        t = th.tensor(x[:, 3])

        r0 = self.r0[0]
        theta_r0 = self.r0[1]
        phi_r0 = self.r0[2]
        # 1. 宽度演化 σ = Δ^2 + i t / (2m)
        sigma = self.delta ** 2 + 1j * t / (2 * self.m)
        # 2. 归一化因子
        norm = (2 * th.pi * (self.delta + 1j * t / (2 * self.m * self.delta)) ** 2) ** (-3 / 4)
        # 3. 极坐标 → 笛卡尔转换
        x_r = r * th.sin(theta_r) * th.cos(phi_r)
        y_r = r * th.sin(theta_r) * th.sin(phi_r)
        z_r = r * th.cos(theta_r)

        x0 = r0 * th.sin(theta_r0) * th.cos(phi_r0)
        y0 = r0 * th.sin(theta_r0) * th.sin(phi_r0)
        z0 = r0 * th.cos(theta_r0)

        x_k = k0 * th.sin(theta_k) * th.cos(phi_k)
        y_k = k0 * th.sin(theta_k) * th.sin(phi_k)
        z_k = k0 * th.cos(theta_k)
        # 4. 移动距离 = (k0/m) * t
        drx = (x_k / self.m) * t
        dry = (y_k / self.m) * t
        drz = (z_k / self.m) * t

        xs = x0 + drx
        ys = y0 + dry
        zs = z0 + drz
        # 5. 计算 [r-(r0+(k0/m)*t)]^2
        dx2 = (x_r - xs) ** 2 + (y_r - ys) ** 2 + (z_r - zs) ** 2
        # 6. 相位项： k·r - (k0^2/(2m))t
        # 极坐标系下：k·r = k0 * r * (sinφ_k * sinφ_r  * cos(θ_k-θ_r) + cosφ_k * cosφ_r)
        phase_arg = (k0 * r * (
                th.sin(phi_k) * th.sin(phi_r) * th.cos(theta_k - theta_r)
                + th.cos(phi_k) * th.cos(phi_r)
            )
            - (k0 ** 2 / (2 * self.m)) * t
        )
        # 7. 构造解析解
        psi = norm * th.exp(-dx2 / (4 * sigma) + 1j * phase_arg)
        psi_real = psi.real.detach().numpy()
        psi_imag = psi.imag.detach().numpy()

        return psi_real, psi_imag


    def predict_and_plot(self):
        # 测试点 只看 φ=pi; θ=0; t=0.5截面
        N = 100
        phi = np.pi
        theta = 0
        t = 0.5
        r = np.linspace(-self.R, self.R, N)
        phi_test = np.full((1, N), phi)
        theta_test = np.full((1, N), theta)
        t_test = np.full((1, N), t)
        r_phi_theta_test = np.vstack([r, theta_test, phi_test, t_test]).T

        # 解析解
        psi_exact_real, psi_exact_imag = self.analytic_solution_polar(r_phi_theta_test)

        # PINN 预测
        y_pred = self.model.predict(r_phi_theta_test)
        psi_pinn_real = y_pred[:, 0]
        psi_pinn_imag = y_pred[:, 1]

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

    def calculate_and_plot_diffs(self):
        # 定义时间范围
        t_values = np.linspace(0, self.T, 100)
        N = 100
        phi = np.pi
        theta = 0
        r = np.linspace(-self.R, self.R, N)
        phi_test = np.full((1, N), phi)
        theta_test = np.full((1, N), theta)

        real_diffs = []
        imag_diffs = []
        amp_diffs = []

        for t in t_values:
            t_test = np.full((1, N), t)
            r_phi_theta_test_all = np.vstack([r, theta_test, phi_test, t_test]).T
            y_pred = self.model.predict(r_phi_theta_test_all)
            psi_pinn_real = y_pred[:, 0]
            psi_pinn_imag = y_pred[:, 1]

            psi_exact_real, psi_exact_imag = self.analytic_solution_polar(r_phi_theta_test_all)
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



# 示例用法
if __name__ == "__main__":

    # 告诉PyTorch不使用GPU
    th.set_default_tensor_type(th.FloatTensor)  ###
    # device = th.device("cuda" if th.cuda.is_available() else "cpu")
    # print("device:", device)

    # 最大半径
    R = 1.0
    # 球坐标域：r ∈ [0,R], θ ∈ [0,π], φ ∈ [0,2π]
    geom = dde.geometry.Cuboid(xmin=[0.0, 0.0, 0.0], xmax=[R, np.pi, 2 * np.pi])
    # 时间域 t ∈ [0,T]
    T = 1.0
    timedomain = dde.geometry.TimeDomain(0.0, T)
    # 合成时空域
    space_time = dde.geometry.GeometryXTime(geom, timedomain)

    # 定义固定时刻（例如，在时间域 [0, T] 内均匀分布的 n 个时刻）
    fixed_times = np.linspace(0.2*T, 0.6*T, 3)  # 固定时刻
    # 定义每个固定时刻的空间采样点数量
    num_spatial_points_per_time = 1000
    # 生成空间采样点
    spatial_points = geom.uniform_points(num_spatial_points_per_time)
    # 将空间采样点与固定时刻组合
    norm_samples = []
    for t in fixed_times:
        t_points = np.full((spatial_points.shape[0], 1), t)
        norm_samples.append(np.hstack((spatial_points, t_points)))

    norm_samples = np.vstack(norm_samples)

    # 学习率
    lr = 1e-3
    # 损失权重
    loss_PDE_real = 1e-2
    loss_PDE_imag = 1e-2
    loss_nomr = 1.0
    loss_IC_real = 1.0
    loss_IC_imag = 1.0

    # 创建求解器实例
    k = th.tensor([2*R, 0.0, 0.0]) # 初始动量
    r0 = th.tensor([-R, 0.0, 0.0]) # 起始坐标
    delta = th.tensor(1.0) # 波包宽度参数
    m = th.tensor(1.0) # 质量
    num_domain = 1000 # 球内采样
    num_initial = 1000 # 起始采样
    num_test = 1000 # 测试点

    iterations = 5 # 训练轮数
    solver = SchrodingerEquationSolver(space_time, k, r0, delta, m, R, T, num_domain, num_initial, num_test, iterations,
                                       norm_samples, fixed_times, num_spatial_points_per_time,
                                       loss_PDE_real, loss_PDE_imag, loss_nomr, loss_IC_real, loss_IC_imag, lr)

    solver.predict_and_plot()
    solver.calculate_and_plot_diffs()

