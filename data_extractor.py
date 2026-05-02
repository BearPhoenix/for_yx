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

	def reload_dataframe(self, new_df):
		"""
		重新加载DataFrame，替换当前数据
		Args:
			new_df (pd.DataFrame): 新的数据框
		Returns:
			self: 便于链式调用
		"""
		if not isinstance(new_df, pd.DataFrame):
			raise ValueError("请传入pandas.DataFrame类型的数据")
		self.df = new_df.copy()
		self.cross_table = None  # 重置交叉表属性
		self.cnt = None  # 重置计数表属性
		return self
	
	# 删除指定列含有NA/Nan的行
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

	# 对指定列的缺失值用指定的数值填充
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

	# 根据两个索引列和值列生成交叉求和的新DataFrame
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

	# 根据交叉表求peer_digital
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
		
	# 计算peer digital列，基于index_cols指定的两列生成交叉求和表，并根据value_col指定的值列计算peer digital值，结果存入new_col_name指定的新列
	def peer_digital_calculate(self, index_cols, value_col, new_col_name):
		self.cross_sum_dataframe(index_cols, value_col)
		# self.cross_table.to_excel("temp.xlsx")
		self.add_new_column_by_pivot(index_cols, value_col, new_col_name)
		# print(self.df.shape[0])
		return self.df
	
	# 剔除1%和99%外的异常数据
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

    # 返回dataframe
	def get_dataframe(self):
		"""
		获取当前DataFrame
		Returns:
			pd.DataFrame: 当前数据
		"""
		return self.df
	
	# 对self.df进行缩尾处理
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
	
	def add_high_order_col(self, base_col, order):
		"""
		基于指定列生成高阶项新列：base_col的order次幂
		Args:
			base_col (str): 基础列名
			new_col_name (str): 新列名
			order (int): 幂次，默认为2（即平方）
		Returns:
			pd.DataFrame: 新增高阶项列后的当前数据
		"""
		if not isinstance(base_col, str) or not base_col:
			raise ValueError("base_col必须为非空字符串")
		if base_col not in self.df.columns:
			raise ValueError(f"基础列 '{base_col}' 不存在于数据框中")

		if not isinstance(order, int) or order < 1:
			raise ValueError("order必须为大于等于1的整数")

		for i in range(2, order + 1):
			intermediate_col_name = f"{base_col}_{i}"
			self.df[intermediate_col_name] = self.df[base_col] ** i
		# self.df[new_col_name] = self.df[base_col] ** order
		return self.df

	def add_new_col_by_2_origin(self, raw_col, minus_cols, new_col_name, op="subtract"):
		"""
		根据指定列生成新列：raw_col 与 minus_cols 的运算结果
		Args:
			raw_col (str): 原始列名
			minus_cols (list or str): 用于运算的列名列表或单个列名
			new_col_name (str): 新列名
			op (str): 运算符，支持 add/+/subtract/-/multiply/*/divide/
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
				raise ValueError(f"运算列 '{col}' 不存在于数据框中")

		if not isinstance(new_col_name, str) or not new_col_name:
			raise ValueError("new_col_name必须为非空字符串")

		op_map = {
			"add": "+",
			"+": "+",
			"subtract": "-",
			"-": "-",
			"multiply": "*",
			"*": "*",
			"divide": "/",
			"/": "/",
		}
		if op not in op_map:
			raise ValueError("op必须是 add/+/subtract/-/multiply/*/divide/\/ 中的一个")

		operator = op_map[op]
		if operator in {"+", "-"}:
			other = self.df[minus_cols].sum(axis=1)
		else:
			other = self.df[minus_cols].prod(axis=1)

		if operator == "+":
			self.df[new_col_name] = self.df[raw_col] + other
		elif operator == "-":
			self.df[new_col_name] = self.df[raw_col] - other
		elif operator == "*":
			self.df[new_col_name] = self.df[raw_col] * other
		else:
			if (other == 0).any():
				raise ZeroDivisionError("除法运算时，minus_cols 的乘积不能包含 0")
			self.df[new_col_name] = self.df[raw_col] / other

		return self.df

	def add_new_col_by_1_origin(self, raw_col, new_col_name, op):
		"""
		根据指定列生成新列：对指定列取对数、取指数、平方或立方运算
		Args:
			raw_col (str): 原始列名
			new_col_name (str): 新列名
			op (str): 操作类型，支持 log/exp/square/cube
		Returns:
			pd.DataFrame: 新增列后的当前数据
		"""
		if not isinstance(raw_col, str) or not raw_col:
			raise ValueError("raw_col必须为非空字符串")
		if raw_col not in self.df.columns:
			raise ValueError(f"原始列 '{raw_col}' 不存在于数据框中")

		if not isinstance(new_col_name, str) or not new_col_name:
			raise ValueError("new_col_name必须为非空字符串")

		if not np.issubdtype(self.df[raw_col].dtype, np.number):
			raise ValueError(f"列 '{raw_col}' 必须是数值类型才能进行运算")

		op = str(op).lower()
		op_map = {
			"log": "log",
			"ln": "log",
			"exp": "exp",
			"square": "square",
			"pow2": "square",
			"cube": "cube",
			"pow3": "cube",
		}
		if op not in op_map:
			raise ValueError("op必须是 log/exp/square/cube 之一")

		operation = op_map[op]
		if operation == "log":
			if (self.df[raw_col] <= 0).any():
				raise ValueError("取对数时，raw_col 中的值必须全部大于 0")
			self.df[new_col_name] = np.log(self.df[raw_col])
		elif operation == "exp":
			self.df[new_col_name] = np.exp(self.df[raw_col])
		elif operation == "square":
			self.df[new_col_name] = self.df[raw_col] ** 2
		else:
			self.df[new_col_name] = self.df[raw_col] ** 3

		return self.df

	def add_future_column(self, raw_col, new_col_name, periods=1):
		"""
		根据指定列生成新列：未来 n 期的值（类似于 Stata 的 F.ROA）
		Args:
			raw_col (str): 原始列名
			new_col_name (str): 新列名
			periods (int): 未来期数，默认 1
		Returns:
			pd.DataFrame: 新增列后的当前数据
		"""
		if not isinstance(raw_col, str) or not raw_col:
			raise ValueError("raw_col必须为非空字符串")
		if raw_col not in self.df.columns:
			raise ValueError(f"原始列 '{raw_col}' 不存在于数据框中")

		if not isinstance(new_col_name, str) or not new_col_name:
			raise ValueError("new_col_name必须为非空字符串")

		if not isinstance(periods, int) or periods <= 0:
			raise ValueError("periods必须为大于0的整数")

		self.df[new_col_name] = self.df[raw_col].shift(-periods)
		return self.df
	
	