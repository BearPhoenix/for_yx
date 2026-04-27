# -*- coding: utf-8 -*-
# 面板数据回归可视化工具类
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
		import numpy as np
		import matplotlib.pyplot as plt
		import seaborn as sns
		from sklearn.linear_model import LinearRegression

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

       
	# 可扩展：自动高亮显著性、添加参考线、自动标注结论等
	# 可根据需要添加更多方法
