import torch as th

# 创建一个示例张量
x = th.tensor([[1, 2, 3], [4, 5, 6]])

# 展平张量
x_flattened = x.flatten().detach().numpy()

print(x_flattened)