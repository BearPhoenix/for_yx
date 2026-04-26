import pandas as pd


class DataLoader:
    """
    数据加载器类，用于加载Excel文件并进行数据清洗
    """

    def __init__(self, file_path=None):
        """
        初始化数据加载器

        Args:
            file_path (str, optional): Excel文件路径
        """
        self.file_path = file_path
        self.df = None

    def load_excel(self, file_path=None):
        """
        加载Excel文件为DataFrame

        Args:
            file_path (str, optional): Excel文件路径。如果未提供，使用初始化时的路径

        Returns:
            pd.DataFrame: 加载的数据框
        """
        if file_path is not None:
            self.file_path = file_path

        if self.file_path is None:
            raise ValueError("请提供Excel文件路径")

        try:
            self.df = pd.read_excel(self.file_path)
            print(f"成功加载文件: {self.file_path}")
            print(f"数据形状: {self.df.shape}")
            return self.df
        except Exception as e:
            raise Exception(f"加载文件失败: {str(e)}")

    def drop_rows_with_na(self):
        """
        删除包含NA值的所有行

        Returns:
            pd.DataFrame: 清洗后的数据框
        """
        if self.df is None:
            raise ValueError("请先加载数据")

        original_rows = len(self.df)
        self.df = self.df.dropna()
        removed_rows = original_rows - len(self.df)

        print(f"删除了 {removed_rows} 行含有NA的数据")
        print(f"剩余数据形状: {self.df.shape}")

        return self.df

    def fill_na_in_columns(self, columns, fill_value):
        """
        在指定列中填补NA值

        Args:
            columns (str or list): 要填补NA值的列名或列名列表
            fill_value: 用于填补NA的值

        Returns:
            pd.DataFrame: 处理后的数据框
        """
        if self.df is None:
            raise ValueError("请先加载数据")

        if isinstance(columns, str):
            columns = [columns]

        for col in columns:
            if col not in self.df.columns:
                raise ValueError(f"列 '{col}' 不存在于数据框中")

        for col in columns:
            na_count = self.df[col].isna().sum()
            if na_count > 0:
                self.df[col] = self.df[col].fillna(fill_value)
                print(f"在列 '{col}' 中填补了 {na_count} 个NA值，使用值: {fill_value}")
            else:
                print(f"列 '{col}' 中没有NA值")

        return self.df

    def print_columns(self):
        """
        获取所有列名（list输出）

        Returns:
            list: 所有列名
        """
        if self.df is None:
            raise ValueError("请先加载数据")
        return self.df.columns.tolist()

    def print_column_value_counts(self, column):
        """
        获取指定列中所有值及其出现频率（list输出）

        Args:
            column (str): 列名

        Returns:
            list: 形如 [{"value": 值, "count": 次数, "frequency": 频率}] 的列表
        """
        if self.df is None:
            raise ValueError("请先加载数据")
        if column not in self.df.columns:
            raise ValueError(f"列 '{column}' 不存在于数据框中")

        counts = self.df[column].value_counts(dropna=False)
        total = len(self.df)
        return [
            {"value": idx, "count": int(cnt), "frequency": float(cnt / total)}
            for idx, cnt in counts.items()
        ]

    def get_dataframe(self, columns=None):
        """
        获取当前的DataFrame或指定列的数据

        Args:
            columns (str or list, optional): 指定要返回的列名或列名列表。如果为None，返回全部数据

        Returns:
            pd.DataFrame 或 pd.Series: 当前数据框或指定列数据
        """
        if self.df is None:
            raise ValueError("请先加载数据")
        if columns is None:
            return self.df
        if isinstance(columns, str):
            columns = [columns]
        for col in columns:
            if col not in self.df.columns:
                raise ValueError(f"列 '{col}' 不存在于数据框中")
        return self.df[columns]

    def get_info(self):
        """
        显示数据框的基本信息
        """
        if self.df is None:
            raise ValueError("请先加载数据")

        print("\n=== 数据框信息 ===")
        print(f"形状: {self.df.shape}")
        print(f"\n列名: {list(self.df.columns)}")
        print(f"\n数据类型:\n{self.df.dtypes}")
        print(f"\n缺失值统计:\n{self.df.isna().sum()}")
        print("\n前5行数据:")
        print(self.df.head())
