import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pyDOE import lhs

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", device)


def schrodinger_residual(batch_data, psi_r, psi_i):
    lap_r, lap_i, psi_i_t, psi_r_t = laplacian(batch_data, psi_r, psi_i)
    res_r = psi_i_t - 0.5 * lap_r
    res_i = -psi_r_t - 0.5 * lap_i
    return res_r, res_i

def laplacian(data, psi_r, psi_i):

    grad_psi_r = torch.autograd.grad(psi_r, data, grad_outputs=torch.ones_like(psi_r), create_graph=True)[0]
    grad_psi_i = torch.autograd.grad(psi_i, data, grad_outputs=torch.ones_like(psi_i), create_graph=True)[0]

    psi_r_r   = grad_psi_r[:, 0:1]
    psi_r_th  = grad_psi_r[:, 1:2]
    psi_r_p   = grad_psi_r[:, 2:3]
    psi_r_t   = grad_psi_r[:, 3:4]

    psi_i_r   = grad_psi_i[:, 0:1]
    psi_i_th  = grad_psi_i[:, 1:2]
    psi_i_p   = grad_psi_i[:, 2:3]
    psi_i_t   = grad_psi_i[:, 3:4]

    # 二阶导数
    psi_r_rr = torch.autograd.grad(psi_r_r, data, grad_outputs=torch.ones_like(psi_r_r), create_graph=True)[0][:, 0:1]
    psi_r_thth = torch.autograd.grad(psi_r_th, data, grad_outputs=torch.ones_like(psi_r_th), create_graph=True)[0][:, 1:2]
    psi_r_pp = torch.autograd.grad(psi_r_p, data, grad_outputs=torch.ones_like(psi_r_p), create_graph=True)[0][:, 2:3]

    psi_i_rr = torch.autograd.grad(psi_i_r, data, grad_outputs=torch.ones_like(psi_i_r), create_graph=True)[0][:, 0:1]
    psi_i_thth = torch.autograd.grad(psi_i_th, data, grad_outputs=torch.ones_like(psi_i_th), create_graph=True)[0][:, 1:2]
    psi_i_pp = torch.autograd.grad(psi_i_p, data, grad_outputs=torch.ones_like(psi_i_p), create_graph=True)[0][:, 2:3]

    # 从 data 中再获取 r, theta, phi
    r = data[:, 0:1]
    theta = data[:, 1:2]

    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    sin2_theta = sin_theta ** 2
    r2 = r ** 2

    lap_r = psi_r_rr + 2 / r * psi_r_r + 1 / r2 * psi_r_thth + cos_theta / (r2 * sin_theta) * psi_r_th + 1 / (r2 * sin2_theta) * psi_r_pp
    lap_i = psi_i_rr + 2 / r * psi_i_r + 1 / r2 * psi_i_thth + cos_theta / (r2 * sin_theta) * psi_i_th + 1 / (r2 * sin2_theta) * psi_i_pp

    return lap_r, lap_i, psi_i_t, psi_r_t



def analytic_solution_polar(K, R, X, delta=1.0, m=1.0):
    # Constants
    k0, theta_k, phi_k = [torch.as_tensor(x, dtype=torch.float32, device=X.device) for x in K]
    r0, theta_r0, phi_r0 = [torch.as_tensor(x, dtype=torch.float32, device=X.device) for x in R]
    delta = torch.tensor(delta, dtype=torch.float32, device=X.device)
    m = torch.tensor(m, dtype=torch.float32, device=X.device)

    # Variables
    r, theta_r, phi_r, t = X[:, 0:1], X[:, 1:2], X[:, 2:3], X[:, 3:4]

    # Convert to Cartesian
    x_r = r * torch.sin(theta_r) * torch.cos(phi_r)
    y_r = r * torch.sin(theta_r) * torch.sin(phi_r)
    z_r = r * torch.cos(theta_r)

    x0 = r0 * torch.sin(theta_r0) * torch.cos(phi_r0)
    y0 = r0 * torch.sin(theta_r0) * torch.sin(phi_r0)
    z0 = r0 * torch.cos(theta_r0)

    x_k = k0 * torch.sin(theta_k) * torch.cos(phi_k)
    y_k = k0 * torch.sin(theta_k) * torch.sin(phi_k)
    z_k = k0 * torch.cos(theta_k)

    drx, dry, drz = (x_k / m) * t, (y_k / m) * t, (z_k / m) * t
    xs, ys, zs = x0 + drx, y0 + dry, z0 + drz
    dx2 = (x_r - xs)**2 + (y_r - ys)**2 + (z_r - zs)**2

    # σ(t)
    sigma_r = delta**2
    sigma_i = t / (2 * m)
    sigma_abs = torch.sqrt(sigma_r**2 + sigma_i**2)  # |σ(t)|

    denom = 4 * (sigma_r**2 + sigma_i**2)
    exp_re = -dx2 * sigma_r / denom
    exp_im =  dx2 * sigma_i / denom

    # Phase term
    cos_phase = torch.sin(phi_k) * torch.sin(phi_r) * torch.cos(theta_k - theta_r) + torch.cos(phi_k) * torch.cos(phi_r)
    phase = k0 * r * cos_phase - (k0**2 / (2 * m)) * t

    # Time-dependent normalization factor
    A = (2 * torch.pi * sigma_abs) ** (-0.75)

    # Final ψ
    real_part = torch.exp(exp_re) * torch.cos(exp_im + phase)
    imag_part = torch.exp(exp_re) * torch.sin(exp_im + phase)

    psi_r = A * real_part
    psi_i = A * imag_part

    return psi_r, psi_i


def cartesian_to_spherical(X_f):
    x, y, z, t = X_f[:, 0], X_f[:, 1], X_f[:, 2], X_f[:, 3]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    phi = np.mod(np.arctan2(y, x) + 2 * np.pi, 2 * np.pi)
    return np.stack((r, theta, phi, t), axis=1)


# 主函数
if __name__ == '__main__':
    R, t_end = 1.0, 1.0
    k = [4 * R, np.pi, 0.0]
    R0 = [R, 0.0, 0.0]

    lb = np.array([-1.0, -1.0, -1.0, 0.0])
    ub = np.array([1.0, 1.0, 1.0, t_end])

    # 采样并转换坐标
    X_test = cartesian_to_spherical(lb + (ub - lb) * lhs(4, 1000))
    X_test_torch = torch.tensor(X_test, dtype=torch.float32, device=device, requires_grad=True)


    psi_r, psi_i = analytic_solution_polar(k, R0, X_test_torch)

    grad_r = torch.autograd.grad(psi_r.sum(), X_test_torch, retain_graph=True)[0]
    print("grad shape:", grad_r.shape)
    print("Any NaN in grad?", torch.isnan(grad_r).any().item())

    res_r, res_i = schrodinger_residual(X_test_torch, psi_r, psi_i)

    print(f"Mean PDE Residual (real): {res_r.abs().mean().item():.2e}")
    print(f"Mean PDE Residual (imag): {res_i.abs().mean().item():.2e}")
    print(f"Max PDE Residual: {max(res_r.abs().max().item(), res_i.abs().max().item()):.2e}")

    # 可视化
    plt.figure(figsize=(8, 4))
    plt.hist(res_r.detach().cpu().numpy(), bins=100, alpha=0.5, label='Re(res)')
    plt.hist(res_i.detach().cpu().numpy(), bins=100, alpha=0.5, label='Im(res)')
    plt.yscale("log")
    plt.xlabel("Residual Value")
    plt.ylabel("Count (log scale)")
    plt.legend()
    plt.title("PDE Residual Histogram (Analytic ψ)")
    plt.tight_layout()
    plt.show()

