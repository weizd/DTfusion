import deepxde as dde
import numpy as np

# 定义几何区域（三维立方体）
geom = dde.geometry.Hypersphere([0.0, 0.0, 0.0], 1)

# 定义时间区域
timedom = dde.geometry.TimeDomain(0, 1)

# 组合几何与时间域
space_time = dde.geometry.GeometryXTime(geom, timedom)

# 定义目标采样点数量
n = 1000

# 计算空间点数量 nx
nx = int(
    np.ceil(
        (
            n
            * np.prod(space_time.geometry.bbox[1] - space_time.geometry.bbox[0])
            / space_time.timedomain.diam
        )
        ** 0.5
    )
)

print(f"Calculated number of spatial points: {nx}")