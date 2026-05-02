# 项目说明

本项目包含一套面向面板数据分析的工具：
- `data_loader.py`：Excel 数据加载与基础清洗
- `data_extractor.py`：数据处理、交叉表、变量构造与特征生成
- `regression.py`：回归分析包装器，支持线性、岭回归、Lasso、随机森林和面板固定效应回归
- `graph.py`：可视化工具，用于绘制回归系数、散点图、直方图、QQ 图、3D 曲面等
- `main.py`：示例主程序，演示数据加载、处理、回归与可视化流程

## 依赖

```bash
pip install pandas numpy scikit-learn statsmodels linearmodels matplotlib seaborn scipy openpyxl
```

## 模块说明

### 1. data_loader.py

#### DataLoader

用于加载 Excel 数据并对数据框执行常见检查与清洗。

主要方法：
- `load_excel(file_path=None)`：读取 Excel 文件
- `drop_rows_with_na()`：删除包含任意 NA 的行
- `fill_na_in_columns(columns, fill_value)`：指定列填补缺失值
- `print_columns()`：返回所有列名列表
- `print_column_value_counts(column)`：返回指定列值频率
- `get_dataframe(columns=None)`：获取全表或指定列数据
- `get_info()`：打印数据基本信息

示例：

```python
from data_loader import DataLoader

loader = DataLoader()
df = loader.load_excel('0421.xlsx')
loader.fill_na_in_columns(['Size', 'ROA'], 0)
print(loader.print_columns())
print(loader.get_info())
```

### 2. data_extractor.py

#### DataExtractor

用于进一步清洗、生成交叉表、计算同行变量、移除极端值、生成新列等。

主要方法：
- `drop_na_rows(columns=None)`：删除指定列含 NA 的行
- `fill_na_in_columns(columns, fill_value)`：指定列缺失值填充
- `cross_sum_dataframe(index_cols, value_col)`：生成交叉求和表
- `add_new_column_by_pivot(index_cols, value_col, new_col_name)`：基于交叉表生成新列
- `peer_digital_calculate(index_cols, value_col, new_col_name)`：生成同行数字化变量
- `remove_outliers_by_quantiles(columns)`：按 1%/99% 分位数移除异常值
- `tail_process(columns)`：对指定列进行缩尾处理
- `subtract_columns_to_new_column(raw_col, minus_cols, new_col_name)`：生成 `raw_col - sum(minus_cols)`
- `add_new_col_by_2_origin(raw_col, minus_cols, new_col_name, op='subtract')`：支持加/减/乘/除的组合变量生成
- `add_new_col_by_1_origin(raw_col, new_col_name, op)`：支持 `log`、`exp`、`square`、`cube`
- `add_future_column(raw_col, new_col_name, periods=1)`：生成未来 n 期值，类似 Stata 的 `F.` 语法
- `get_dataframe()`：返回当前 DataFrame

示例：

```python
from data_extractor import DataExtractor

extractor = DataExtractor(df)
extractor.drop_na_rows(columns=['Size', 'ROA'])
extractor.remove_outliers_by_quantiles(columns=['Size', 'ROA'])
extractor.peer_digital_calculate(index_cols=['accper', 'nnindcd'], value_col='digitaltransindex', new_col_name='peer_digital')
extractor.add_new_col_by_2_origin(raw_col='Size', minus_cols=['ROA'], new_col_name='Size_adjusted', op='subtract')
extractor.add_new_col_by_1_origin(raw_col='ROA', new_col_name='ROA_log', op='log')
extractor.add_future_column(raw_col='ROA', new_col_name='F_ROA', periods=1)
result_df = extractor.get_dataframe()
```

### 3. regression.py

#### RegressionModel

封装了多种回归方法，支持与 Stata `xtreg` 对齐的面板固定效应估计。

支持方法：
- `linear`：普通最小二乘线性回归
- `ridge`：岭回归
- `lasso`：Lasso 回归
- `rf`：随机森林回归
- `panel_fe` / `xtreg`：面板固定效应回归（`linearmodels`）

主要用法：
- `RegressionModel(dataframe, x_vars, y_var, method='linear')`
- `fit()`：拟合模型
- `xtfit(entity_col, time_col, time_effects=True, robust=True, cluster_entity=True, cluster_time=False)`：面板固定效应拟合
- `predict(X=None)`：预测
- `score(X=None, y=None)`：计算 R² 或 within R²
- `coef_and_residuals(x_var_list)`：返回系数和残差信息

示例：

```python
from regression import RegressionModel

reg = RegressionModel(df, x_vars=['peer_digital', 'Size', 'Lev'], y_var='digitaltransindex', method='linear')
reg.fit()
print('R2=', reg.score())

panel = RegressionModel(df, x_vars=['peer_digital', 'Size', 'Lev'], y_var='digitaltransindex', method='xtreg')
panel.xtfit(entity_col='Stkcd', time_col='accper', time_effects=True, robust=True)
print(panel.score())
```

### 4. graph.py

#### PanelPlotter

可视化模块用于绘制各类图形。

支持绘图类型：
- `coef_plot(...)`：回归系数及置信区间
- `scatter_with_custom_curve(...)`：散点 + 自定义曲线
- `scatter_with_polyfit(...)`：散点 + 多项式拟合曲线
- `scatter_plot(...)`：普通散点图
- `hist_plot(...)`：直方图
- `qq_plot(...)`：QQ 图
- `effect_dist_plot(...)`：箱线图/小提琴图/直方图
- `line_plot(...)`：折线图
- `bar_plot(...)`：柱状图
- `scatter_with_reg_line(...)`：散点 + 简单线性回归线
- `scatter_3d_with_surface(...)`：三维散点 + 拟合曲面

示例：

```python
from graph import PanelPlotter

plotter = PanelPlotter(df)
plotter.scatter_plot('peer_digital', 'digitaltransindex', save=True, save_path='scatter.svg')
plotter.hist_plot('ROA', bins=40, save=True, save_path='roa_hist.svg')
```

三维曲面示例：

```python
plotter.scatter_3d_with_surface(
    dic=var_dict,
    x_col='Subsidy',
    y_col='peer_digital',
    z_col='digitaltransindex',
    model=reg.model,
    x_pred=X_pred,
    save=True,
    save_path='surface.svg'
)
```

### 5. main.py

`main.py` 是项目示例入口，展示了数据加载、处理、回归和可视化的完整流程。

关键流程：
1. 通过 `DataLoader` 加载 Excel 文件
2. 使用 `DataExtractor` 进行缺失值处理、异常值剔除、`peer_digital` 生成等
3. 构造额外变量（如 `x_2`, `px`, `px_2`）
4. 使用 `RegressionModel` 进行 `xtreg` 固定效应回归
5. 使用 `PanelPlotter` 绘制图形

如果需要自定义回归变量：
- 修改 `control_var`、`main_var`、`must_var`
- 修改 `var_dict['select_var']`
- 修改 `var_dict['regression_x']`

示例运行：

```bash
python main.py
```

如果想在 `main.py` 中添加自定义可视化：
- 在 `main()` 中构造 `X_pred` 网格数据
- 调用 `plotter.scatter_3d_with_surface(...)`

## 常见注意事项

- 运行 `xtreg` 之前，`DataExtractor` 会自动生成 `peer_digital` 并删除缺失值
- `RegressionModel` 的面板回归依赖 `linearmodels` 包，请确保已安装
- `add_future_column` 生成的未来期列会使用 `shift(-periods)`，因此最后 `periods` 行会产生缺失值

## 推荐流程

1. `DataLoader.load_excel()`
2. `DataExtractor.drop_na_rows()`
3. `DataExtractor.remove_outliers_by_quantiles()` 或 `tail_process()`
4. `DataExtractor.peer_digital_calculate()`
5. `RegressionModel(...).xtfit(...)`
6. `PanelPlotter` 绘图
