def subtract_columns_to_new_column(self, raw_col, minus_cols, new_col_name, op="subtract"):
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