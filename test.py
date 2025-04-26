import numpy as np
a2 = [1, 1, 1]
a1 = np.array(a2)
a3 = a1.reshape(1, -1)
a = np.random.rand(1, 3)
b = np.random.rand(10, 3)

c = np.sum(a3*b, axis=1)
print(c)
d = np.dot(b, a1).reshape(-1, 1)
print(d)