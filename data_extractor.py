# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


class DataExtractor:
	"""
	数据提取与清洗类
	"""
	def __init__(self, df):
		"""
		初始化，传入DataFrame
		Args:
			df (pd.DataFrame): 需要处理的数据
		"""
		if not isinstance(df, pd.DataFrame):
			raise ValueError("请传入pandas.DataFrame类型的数据")
		self.df = df.copy()
		self.cross_table = None  # 初始化交叉表属性
		self.cnt = None  # 初始化计数表属性

	def drop_na_rows(self, columns=None):
		"""
		删除指定列含有NA/Nan的行。如果不指定列，则删除所有含NA/Nan的行。
		Args:
			columns (str or list, optional): 指定列名或列名列表
		Returns:
			dataframe
		"""
		if columns is None:
			self.df = self.df.dropna()
		else:
			if isinstance(columns, str):
				columns = [columns]
			for col in columns:
				if col not in self.df.columns:
					raise ValueError(f"列 '{col}' 不存在于数据框中")
			self.df = self.df.dropna(subset=columns)
		return self.df

	def fill_na_in_columns(self, columns, fill_value):
		"""
		对指定列的缺失值用指定的数值填充
		Args:
			columns (str or list): 列名或列名列表
			fill_value: 用于填充的数值
		Returns:
			dataframe
		"""
		if isinstance(columns, str):
			columns = [columns]
		for col in columns:
			if col not in self.df.columns:
				raise ValueError(f"列 '{col}' 不存在于数据框中")
		for col in columns:
			self.df[col] = self.df[col].fillna(fill_value)
		return self.df

	def cross_sum_dataframe(self, index_cols, value_col):
		"""
		根据两个索引列和值列生成交叉求和的新DataFrame。
		Args:
			index_cols (list): 长度为2的列表，指定两个索引列名
			value_col (str): 指定的值列名
		Returns:
			self: 便于链式调用
		"""
		if not (isinstance(index_cols, list) and len(index_cols) == 2):
			raise ValueError("index_cols必须为长度为2的list")
		if not all(col in self.df.columns for col in index_cols):
			raise ValueError("索引列不存在于DataFrame中")
		if value_col not in self.df.columns:
			raise ValueError("值列不存在于DataFrame中")

		# 生成透视表
		result = self.df.pivot_table(index=index_cols[1], columns=index_cols[0], values=value_col, aggfunc='sum', fill_value=0)
		cnt = self.df.pivot_table(index=index_cols[1], columns=index_cols[0], values=value_col, aggfunc='size', fill_value=0)
		self.cross_table = result
		self.cnt = cnt
		# return self

	def add_new_column_by_pivot(self, index_cols, value_col, new_col_name):
		"""
		基于已生成的self.cross_table（求和表）和self.cnt（计数表）新增列：
		对每一行，设该行在index_cols两列上的值分别为a(列索引值)、b(行索引值)，
		value_col该行值为c；交叉表对应位置求和为d、计数为e，
		则新值 = (d - c) / (e - 1)
		"""
		if not (isinstance(index_cols, list) and len(index_cols) == 2):
			raise ValueError("index_cols必须为长度为2的list")
		if not all(col in self.df.columns for col in index_cols):
			raise ValueError("索引列不存在于DataFrame中")
		if value_col not in self.df.columns:
			raise ValueError("值列不存在于DataFrame中")
		if not isinstance(new_col_name, str) or not new_col_name:
			raise ValueError("new_col_name必须为非空字符串")

		# 如果透视表还没准备好，先按既有规则生成
		if self.cross_table is None or self.cnt is None:
			self.cross_sum_dataframe(index_cols, value_col)

		col_key = index_cols[0]  # 列索引
		row_key = index_cols[1]  # 行索引

		new_values = []
		for _, row in self.df.iterrows():
			a = row[col_key]
			b = row[row_key]
			c = row[value_col]

			# d/e 来自交叉表对应单元格
			if (b in self.cross_table.index) and (a in self.cross_table.columns):
				d = self.cross_table.loc[b, a]
				e = self.cnt.loc[b, a]

				# e-1为0时无法计算，记为NA
				if pd.isna(e) or e <= 1:
					new_values.append(pd.NA)
				else:
					new_values.append((d - c) / (e - 1))
			else:
				new_values.append(pd.NA)

		self.df[new_col_name] = new_values
		return self.df
		
	
	def peer_digital_calculate(self, index_cols, value_col, new_col_name):
		self.cross_sum_dataframe(index_cols, value_col)
		# self.cross_table.to_excel("temp.xlsx")
		self.add_new_column_by_pivot(index_cols, value_col, new_col_name)
		# print(self.df.shape[0])
		return self.df
	

	def remove_outliers_by_quantiles(self, columns):
		"""
		按分位数范围移除指定列中的异常值行。
		对每个指定列，分别计算当前DataFrame该列的1%分位数和99%分位数，
		并仅保留落在该区间内的行。
		Returns:
			pd.DataFrame: 过滤异常值后的当前数据。

		"""
		if isinstance(columns, str):
			columns = [columns]
		for col in columns:
			if col not in self.df.columns:
				raise ValueError(f"列 '{col}' 不存在于数据框中")
			q_low = self.df[col].quantile(0.01)
			q_high = self.df[col].quantile(0.99)
			self.df = self.df[(self.df[col] >= q_low) & (self.df[col] <= q_high)]
		return self.df

	def regression_y_calculation(self, raw_y, vars, new_col_name):
		"""
		计算回归的因变量值。每行：raw_y - sum(vars)
		Args:
			raw_y (str): 原始因变量列名
			vars (list): 需要从原始因变量中减去的变量列名列表
			new_col_name (str): 新增回归因变量列名
		Returns:
			pd.DataFrame: 新增回归因变量列后的当前数据
		"""
		if raw_y not in self.df.columns:
			raise ValueError(f"原始因变量列 '{raw_y}' 不存在于数据框中")
		for var in vars:
			if var not in self.df.columns:
				raise ValueError(f"变量列 '{var}' 不存在于数据框中")
		self.df[new_col_name] = self.df[raw_y] - self.df[vars].sum(axis=1)
		return self.df

	def get_dataframe(self):
		"""
		获取当前DataFrame
		Returns:
			pd.DataFrame: 当前数据
		"""
		return self.df

	def calculate_year_and_fixed_effects(self, index_cols, effect_cols, y_col):
		"""
		使用linearmodels.PanelOLS估计年份效应和个体固定效应，
		并在self.df中新增对应列。

		Args:
			index_cols (list): 长度为2的列表，依次为[年份列名, 个体列名]
			effect_cols (list): 长度为2的列表，依次为[年份效应列名, 固定效应列名]
			y_col (str): 因变量列名

		Returns:
			pd.DataFrame: 新增效应列后的当前数据
		"""
		try:
			from linearmodels.panel import PanelOLS
		except ImportError:
			raise ImportError("请先安装linearmodels：pip install linearmodels")

		if not (isinstance(index_cols, list) and len(index_cols) == 2):
			raise ValueError("index_cols必须为长度为2的list，格式为[年份列名, 个体列名]")
		if not (isinstance(effect_cols, list) and len(effect_cols) == 2):
			raise ValueError("effect_cols必须为长度为2的list，格式为[年份效应列名, 固定效应列名]")

		year_col, entity_col = index_cols
		year_effect_col, fixed_effect_col = effect_cols

		if year_col not in self.df.columns:
			raise ValueError(f"年份列 '{year_col}' 不存在于数据框中")
		if entity_col not in self.df.columns:
			raise ValueError(f"个体列 '{entity_col}' 不存在于数据框中")
		if not isinstance(y_col, str) or not y_col:
			raise ValueError("y_col必须为非空字符串")
		if y_col not in self.df.columns:
			raise ValueError(f"因变量列 '{y_col}' 不存在于数据框中")

		# 保留原始行索引，便于后续合并回原表
		work_df = self.df[[year_col, entity_col, y_col]].dropna().copy()
		if work_df.empty:
			raise ValueError("用于估计效应的数据为空，请检查缺失值")
		work_df["__row_idx__"] = work_df.index

		# 过滤singleton，降低自由度为0导致的除零风险
		entity_n = work_df.groupby(entity_col)[y_col].transform("size")
		time_n = work_df.groupby(year_col)[y_col].transform("size")
		work_df = work_df[(entity_n > 1) & (time_n > 1)].copy()
		if work_df.empty:
			raise ValueError("过滤singleton后无可用样本，无法估计固定效应")

		# PanelOLS要求多重索引：(entity, time)
		panel_df = work_df.set_index([entity_col, year_col]).sort_index()
		exog = pd.DataFrame({"const": 1.0}, index=panel_df.index)

		try:
			# 个体固定效应（entity FE）
			entity_res = PanelOLS(panel_df[y_col], exog, entity_effects=True).fit(debiased=False)
			panel_df[fixed_effect_col] = entity_res.estimated_effects.iloc[:, 0]

			# 年份效应（time FE）
			time_res = PanelOLS(panel_df[y_col], exog, time_effects=True).fit(debiased=False)
			panel_df[year_effect_col] = time_res.estimated_effects.iloc[:, 0]
		except ZeroDivisionError:
			raise ValueError("固定效应估计失败：有效自由度为0，请检查样本量、分组数量或过度筛选")

		result_df = panel_df[["__row_idx__", year_effect_col, fixed_effect_col]].reset_index(drop=True)

		# 合并回原数据（未参与估计的行保留NA）
		self.df[year_effect_col] = np.nan
		self.df[fixed_effect_col] = np.nan
		self.df.loc[result_df["__row_idx__"], year_effect_col] = result_df[year_effect_col].values
		self.df.loc[result_df["__row_idx__"], fixed_effect_col] = result_df[fixed_effect_col].values

		return self.df
	
	# 新增方法对self.df进行缩尾巴处理
	def tail_process(self, columns):
		"""
		对指定列进行缩尾处理。每列：将小于1%分位数的值替换为1%分位数，将大于99%分位数的值替换为99%分位数。
		Args:
			columns (str or list): 列名或列名列表
		Returns:
			pd.DataFrame: 处理后的当前数据
		"""
		if isinstance(columns, str):
			columns = [columns]
		for col in columns:
			if col not in self.df.columns:
				raise ValueError(f"列 '{col}' 不存在于数据框中")
			q_low = self.df[col].quantile(0.01)
			q_high = self.df[col].quantile(0.99)
			self.df[col] = self.df[col].clip(lower=q_low, upper=q_high)
		return self.df

	def subtract_columns_to_new_column(self, raw_col, minus_cols, new_col_name):
		"""
		根据指定列生成新列：raw_col - sum(minus_cols)
		Args:
			raw_col (str): 被减数列名
			minus_cols (list): 需要相减的列名列表
			new_col_name (str): 新列名
		Returns:
			pd.DataFrame: 新增列后的当前数据
		"""
		if not isinstance(raw_col, str) or not raw_col:
			raise ValueError("raw_col必须为非空字符串")
		if raw_col not in self.df.columns:
			raise ValueError(f"原始列 '{raw_col}' 不存在于数据框中")

		if isinstance(minus_cols, str):
			minus_cols = [minus_cols]
		if not (isinstance(minus_cols, list) and len(minus_cols) > 0):
			raise ValueError("minus_cols必须为非空list")
		for col in minus_cols:
			if col not in self.df.columns:
				raise ValueError(f"相减列 '{col}' 不存在于数据框中")

		if not isinstance(new_col_name, str) or not new_col_name:
			raise ValueError("new_col_name必须为非空字符串")

		self.df[new_col_name] = self.df[raw_col] - self.df[minus_cols].sum(axis=1)
		return self.df
