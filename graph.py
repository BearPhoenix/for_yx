# -*- coding: utf-8 -*-
# 面板数据回归可视化工具类
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

class PanelPlotter:		
	def __init__(self, df):
		"""
		df: pandas.DataFrame，包含所有分析数据
		"""
		self.df = df

	def _set_axis_scale(self, ax, xscale=None, yscale=None):
		if xscale:
			ax.set_xscale(xscale)
		if yscale:
			ax.set_yscale(yscale)
		ax.autoscale(enable=True, axis='both')

	def coef_plot(self, coef, ci_low, ci_high, labels=None, title="regression factor", yscale=None, save=False, save_path=None):
		"""
		coef, ci_low, ci_high: 系数及置信区间（list/array）
		labels: 变量名列表
		"""
		fig, ax = plt.subplots()
		y_pos = np.arange(len(coef))
		ax.errorbar(coef, y_pos, xerr=[coef - ci_low, ci_high - coef], fmt='o', capsize=5)
		ax.set_yticks(y_pos)
		if labels:
			ax.set_yticklabels(labels)
		ax.axvline(0, color='grey', linestyle='--', lw=1)
		ax.set_title(title)
		self._set_axis_scale(ax, yscale=yscale)
		plt.tight_layout()
		if save and save_path:
			plt.savefig(save_path, dpi=300, bbox_inches='tight')
		plt.show()

	def scatter_with_custom_curve(self, x_col, y_col, curve_func, xscale=None, yscale=None, title=None, save=False, save_path=None, curve_label='Custom Curve'):
		"""
		绘制原始数据散点图，并叠加自定义回归曲线。
		x_col: DataFrame中x轴列名
		y_col: DataFrame中y轴列名
		curve_func: 可调用的回归函数（如np.poly1d对象或自定义函数）
		xscale/yscale: 支持'log'等非线性缩放
		curve_label: 曲线图例名称
		"""

		fig, ax = plt.subplots()
		# 绘制散点
		sns.scatterplot(data=self.df, x=x_col, y=y_col, ax=ax, color='tab:blue', label='Data')

		x = self.df[x_col].values
		mask = ~np.isnan(x)
		x_valid = x[mask]
		if len(x_valid) > 0:
			x_line = np.linspace(x_valid.min(), x_valid.max(), 200)
			y_line = curve_func(x_line)
			ax.plot(x_line, y_line, color='tab:orange', label=curve_label)

		self._set_axis_scale(ax, xscale, yscale)
		ax.set_title(title or f"{y_col} vs {x_col} with Custom Curve")
		ax.legend()
		plt.tight_layout()
		if save and save_path:
			plt.savefig(save_path, dpi=300, bbox_inches='tight')
		plt.show()

	def scatter_with_polyfit(self, x_col, y_col, degree=2, xscale=None, yscale=None, title=None, save=False, save_path=None):
		"""
		绘制原始数据散点图，并拟合高次多项式回归曲线。
		x_col: DataFrame中x轴列名
		y_col: DataFrame中y轴列名
		degree: 多项式阶数（默认2，即二次）
		xscale/yscale: 支持'log'等非线性缩放
		"""
		fig, ax = plt.subplots()
		# 绘制散点
		sns.scatterplot(data=self.df, x=x_col, y=y_col, ax=ax, color='tab:blue', label='Data')

		x = self.df[x_col].values
		y = self.df[y_col].values
		mask = ~np.isnan(x) & ~np.isnan(y)
		x_valid = x[mask]
		y_valid = y[mask]
		if len(x_valid) > degree:
			# 多项式拟合
			coeffs = np.polyfit(x_valid, y_valid, degree)
			poly = np.poly1d(coeffs)
			x_line = np.linspace(x_valid.min(), x_valid.max(), 200)
			y_line = poly(x_line)
			ax.plot(x_line, y_line, color='tab:orange', label=f'Poly{degree} Fit')

		self._set_axis_scale(ax, xscale, yscale)
		ax.set_title(title or f"{y_col} vs {x_col} with Poly{degree} Fit")
		ax.legend()
		plt.tight_layout()
		if save and save_path:
			plt.savefig(save_path, dpi=300, bbox_inches='tight')
		plt.show()

	def scatter_plot(self, x_col, y_col, hue=None, xscale=None, yscale=None, title=None, save=False, save_path=None):
		fig, ax = plt.subplots()
		sns.scatterplot(data=self.df, x=x_col, y=y_col, hue=hue, ax=ax)
		self._set_axis_scale(ax, xscale, yscale)
		ax.set_title(title or f"{y_col} vs {x_col}")
		plt.tight_layout()
		if save and save_path:
			plt.savefig(save_path, dpi=300, bbox_inches='tight')
		plt.show()

	def hist_plot(self, col, bins=30, xscale=None, title=None, save=False, save_path=None):
		fig, ax = plt.subplots()
		sns.histplot(self.df[col], bins=bins, ax=ax)
		self._set_axis_scale(ax, xscale)
		ax.set_title(title or f"{col} 分布直方图")
		plt.tight_layout()
		if save and save_path:
			plt.savefig(save_path, dpi=300, bbox_inches='tight')
		plt.show()

	def qq_plot(self, col, title=None, save=False, save_path=None):
		import scipy.stats as stats
		fig, ax = plt.subplots()
		stats.probplot(self.df[col], dist="norm", plot=ax)
		ax.set_title(title or f"{col} QQ图")
		plt.tight_layout()
		if save and save_path:
			plt.savefig(save_path, dpi=300, bbox_inches='tight')
		plt.show()

	def effect_dist_plot(self, col, by=None, kind="box", title=None, save=False, save_path=None):
		fig, ax = plt.subplots()
		if kind == "box":
			sns.boxplot(data=self.df, x=by, y=col, ax=ax)
		elif kind == "violin":
			sns.violinplot(data=self.df, x=by, y=col, ax=ax)
		elif kind == "hist":
			sns.histplot(self.df[col], ax=ax)
		ax.set_title(title or f"{col} 效应分布图")
		plt.tight_layout()
		if save and save_path:
			plt.savefig(save_path, dpi=300, bbox_inches='tight')
		plt.show()

	def line_plot(self, x_col, y_col, group_col=None, xscale=None, yscale=None, title=None, save=False, save_path=None):
		fig, ax = plt.subplots()
		if group_col:
			sns.lineplot(data=self.df, x=x_col, y=y_col, hue=group_col, ax=ax)
		else:
			sns.lineplot(data=self.df, x=x_col, y=y_col, ax=ax)
		self._set_axis_scale(ax, xscale, yscale)
		ax.set_title(title or f"{y_col} 随 {x_col} 变化")
		plt.tight_layout()
		if save and save_path:
			plt.savefig(save_path, dpi=300, bbox_inches='tight')
		plt.show()

	def bar_plot(self, x_col, y_col, xscale=None, yscale=None, title=None, save=False, save_path=None):
		fig, ax = plt.subplots()
		sns.barplot(data=self.df, x=x_col, y=y_col, ax=ax)
		self._set_axis_scale(ax, xscale, yscale)
		ax.set_title(title or f"{y_col} 按 {x_col} 分组")
		plt.tight_layout()
		if save and save_path:
			plt.savefig(save_path, dpi=300, bbox_inches='tight')
		plt.show()

	def scatter_with_reg_line(self, x_col, y_col, xscale=None, yscale=None, title=None, save=False, save_path=None):
		"""
		在同一张图上绘制指定两列的散点图和回归折线。
		x_col: DataFrame中x轴列名
		y_col: DataFrame中y轴列名
		xscale/yscale: 支持'log'等非线性缩放
		"""
		fig, ax = plt.subplots()
		# 绘制散点
		sns.scatterplot(data=self.df, x=x_col, y=y_col, ax=ax, color='tab:blue', label='Data')

		# 拟合回归线
		x = self.df[x_col].values.reshape(-1, 1)
		y = self.df[y_col].values
		mask = ~np.isnan(x).flatten() & ~np.isnan(y)
		x_valid = x[mask]
		y_valid = y[mask]
		if len(x_valid) > 1:
			model = LinearRegression()
			model.fit(x_valid, y_valid)
			x_line = np.linspace(x_valid.min(), x_valid.max(), 100).reshape(-1, 1)
			y_line = model.predict(x_line)
			ax.plot(x_line, y_line, color='tab:orange', label='Regression Line')

		self._set_axis_scale(ax, xscale, yscale)
		ax.set_title(title or f"{y_col} vs {x_col} with Regression Line")
		ax.legend()
		plt.tight_layout()
		if save and save_path:
			plt.savefig(save_path, dpi=300, bbox_inches='tight')
		plt.show()

	def scatter_3d_with_surface(self, dic, x_col, y_col, z_col, model, x_pred, title=None, save=False, save_path=None):

		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		# 绘制原始散点
		ax.scatter(self.df[x_col], self.df[y_col], self.df[z_col], c='b', marker='o', label='Data')

		if not isinstance(x_pred, pd.DataFrame):
			raise ValueError("x_pred 必须是 pandas.DataFrame")
		if x_col not in x_pred.columns or y_col not in x_pred.columns:
			raise ValueError(f"x_pred 必须包含列 {x_col} 和 {y_col}")

		x_vals = x_pred[x_col].values
		y_vals = x_pred[y_col].values
		unique_x = np.unique(x_vals)
		unique_y = np.unique(y_vals)
		if unique_x.size * unique_y.size != len(x_pred):
			raise ValueError("x_pred 必须包含完整的 x/y 网格数据")

		x_surf, y_surf = np.meshgrid(unique_x, unique_y)


		# 构造高次特征
		# X_pred = np.column_stack([
		# 	# y_surf.ravel()**2, x_surf.ravel(), y_surf.ravel()**2, x_surf.ravel()*y_surf.ravel()**2, x_surf.ravel()*y_surf.ravel()
		# 	y_surf.ravel(),np.zeros_like(x_surf.ravel()),np.zeros_like(x_surf.ravel()),
		# 	np.zeros_like(x_surf.ravel()),np.zeros_like(x_surf.ravel()),np.zeros_like(x_surf.ravel()),
		# 	np.zeros_like(x_surf.ravel()),np.zeros_like(x_surf.ravel()),np.zeros_like(x_surf.ravel()),
		# 	np.zeros_like(x_surf.ravel()),x_surf.ravel()**2,x_surf.ravel()*y_surf.ravel(),y_surf.ravel()*x_surf.ravel()**2,
		# 	x_surf.ravel()
		# ])
		z_pred = model.predict(x_pred)
		z_surf = np.asarray(z_pred).reshape(x_surf.shape)
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




	# 可扩展：自动高亮显著性、添加参考线、自动标注结论等
	# 可根据需要添加更多方法
	