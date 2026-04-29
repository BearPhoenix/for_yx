import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 随机生成数据
np.random.seed(0)
x1 = np.random.rand(50)
x2 = np.random.rand(50)
y = 2 * x1 + 3 * x2 + np.random.randn(50) * 0.2  # y 与 x1, x2 线性相关并加噪声

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, y, c='b', marker='o')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('3D Scatter Plot Example')

plt.show()
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # 假设 x1, x2, y 都是一维数组或 Series
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(df['x1'], df['x2'], df['y'], c='b', marker='o')

# ax.set_xlabel('x1')
# ax.set_ylabel('x2')
# ax.set_zlabel('y')
# ax.set_title('3D Scatter Plot')

# plt.show()