import torch

# 构造一个假的 u(x, t) 函数，例如：
# u(x, t) = exp(-π^2 t) * sin(π x)
def u(x, t):
    return torch.exp(-torch.pi**2 * t) * torch.sin(torch.pi * x)

# 构造点 (x, t)，开启自动求导
x = torch.tensor([0.5], requires_grad=True)
t = torch.tensor([0.1], requires_grad=True)

# 计算 u(x, t)
u_val = u(x, t)

# 一阶导 ∂u/∂t
u_t = torch.autograd.grad(u_val, t, create_graph=True)[0]

# 二阶导 ∂²u/∂x²
u_x = torch.autograd.grad(u_val, x, create_graph=True)[0]
u_xx = torch.autograd.grad(u_x, x, create_graph=True)[0]

# 打印结果
print(f"u(x,t) = {u_val.item():.6f}")
print(f"∂u/∂t = {u_t.item():.6f}")
print(f"∂²u/∂x² = {u_xx.item():.6f}")

# 检查残差 ∂u/∂t - ∂²u/∂x² ≈ 0？
residual = u_t - u_xx
print(f"PDE residual (∂u/∂t - ∂²u/∂x²): {residual.item():.6e}")
