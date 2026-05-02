import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 随机生成数据
# np.random.seed(0)
# x1 = np.random.rand(50)
# x2 = np.random.rand(50)
# y = 2 * x1 + 3 * x2 + np.random.randn(50) * 0.2  # y 与 x1, x2 线性相关并加噪声

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x1, x2, y, c='b', marker='o')

# ax.set_xlabel('x1')
# ax.set_ylabel('x2')
# ax.set_zlabel('y')
# ax.set_title('3D Scatter Plot Example')

# plt.show()



def scatter_3d_with_surface(self, x_col, y_col, z_col, model, title=None, save=False, save_path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 绘制原始散点
    ax.scatter(self.df[x_col], self.df[y_col], self.df[z_col], c='b', marker='o', label='Data')
    # 构造网格
    x = self.df[x_col]
    y = self.df[y_col]
    x_surf, y_surf = np.meshgrid(
        np.linspace(x.min(), x.max(), 30),
        np.linspace(y.min(), y.max(), 30)
    )
    # 构造高次特征
    X_pred = np.column_stack([
        x_surf.ravel()**2, x_surf.ravel(), y_surf.ravel()**2, x_surf.ravel()*y_surf.ravel()**2, x_surf.ravel()*y_surf.ravel()
    ])
    z_surf = model.predict(X_pred).reshape(x_surf.shape)
    # 绘制拟合曲面
    ax.plot_surface(x_surf, y_surf, z_surf, color='orange', alpha=0.5, label='Fitted Surface')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.set_title(title or f'{z_col} vs {x_col} and {y_col}')
    plt.tight_layout()
    if save and save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()