# --- 封装主流程各步骤为独立函数 ---
def load_and_preprocess_data():
    loader = DataLoader()
    df = loader.load_excel(file_path)
    df = loader.get_dataframe(select_list)
    extractor = DataExtractor(df)
    data = extractor.drop_na_rows(columns=select_list)
    print(f"数据的形状: {df.shape}")
    data_new = extractor.remove_outliers_by_quantiles(columns=tail_process_list)
    data_new = extractor.tail_process(columns=tail_process_list)
    data_new = extractor.peer_digital_calculate(index_cols=["accper", "nnindcd"], value_col="digitaltransindex", new_col_name="peer_digital")
    data_new = data_new[data_new["nnindcd"] != "C43"]
    return data_new, extractor

def process_high_order_terms(data_new, extractor):
    if not switch_for_high_order:
        return data_new, [], []
    tmp = []
    for ele in high_order_terms:
        data_new = extractor.add_high_order_col(ele[0], ele[1])
    for ele in high_order_terms:
        for j in range(2, ele[1]+1):
            tmp.append(f"{ele[0]}_{j}")
    print(tmp)
    return data_new, tmp, tmp

def check_model_flags():
    model_flags = [int(bool(m1)), int(bool(m2)), int(bool(m3))]
    if sum(model_flags) != 1:
        raise ValueError("m1/m2/m3 必须且只能有一个为1")

def fit_regression(data_new, regression_x, regression_y):
    reg = RegressionModel(data_new, x_vars=regression_x, y_var=regression_y, method="linear")
    if m3:
        reg.xtfit(
            entity_col="Stkcd",
            time_col="accper",
            time_effects=True,
            robust=True,
            cluster_entity=True,
            cluster_time=False,
        )
    else:
        reg.fit()
    coef_result = reg.coef_and_residuals(regression_x)
    return reg, coef_result

def print_regression_results(reg, coef_result):
    print("回归结果：")
    print(f"截距: {coef_result['intercept']}")
    print(f"回归系数: {coef_result['coefficients']}")
    print(f"R²: {reg.score():.6f}")
    if m3:
        print(f"稳健标准误: {coef_result['std_errors']}")
        print(f"p值: {coef_result['p_values']}")

def plot_regression_results(data_new, coef_result):
    from graph import PanelPlotter
    plotter = PanelPlotter(data_new)
    coef = np.array(list(coef_result['coefficients'].values()))
    labels = list(coef_result['coefficients'].keys())
    if m3:
        std_err = np.array(list(coef_result['std_errors'].values()))
        ci_low = coef - 1.96 * std_err
        ci_high = coef + 1.96 * std_err
    else:
        ci_low = coef - 0.1
        ci_high = coef + 0.1
    plotter.coef_plot(coef, ci_low, ci_high, labels=labels, save=True, save_path=save_path_coef)

    resid = coef_result['residuals']
    fitted = coef_result['fitted']
    temp_df = data_new.copy()
    temp_df['fitted'] = np.array(fitted)
    temp_df['residuals'] = np.array(resid)
    plotter2 = PanelPlotter(temp_df)
    plotter2.scatter_plot('fitted', 'residuals', save=True, save_path=save_path_resid.replace('.png', '.svg'), title='Residuals vs Fitted')
    plotter2.hist_plot('residuals', save=True, save_path=save_path_hist.replace('.png', '.svg'), title='Histogram of Residuals')
    plotter2.qq_plot('residuals', save=True, save_path=save_path_qq.replace('.png', '.svg'), title='QQ Plot of Residuals')

    if switch_for_high_order:
        plotter.hist_plot(main_var[0], bins=30, save=True, save_path=f"{main_var[0]}_hist.svg", title=f"{main_var[0]} 分布直方图")
        coef_dict = coef_result['coefficients']
        intercept = coef_result['intercept']
        degree = high_order_terms[0][1] if high_order_terms else 2
        base = main_var[0]
        coeffs = [coef_dict.get(f"{base}_{d}", 0) if coef_dict.get(f"{base}_{d}") is not None else 0 for d in range(degree, 1, -1)]
        coeffs.append(coef_dict.get(base, 0) if coef_dict.get(base) is not None else 0)
        coeffs.append(intercept if intercept is not None else 0)
        poly_func = np.poly1d(coeffs)
        plotter.scatter_with_custom_curve(base, target_var, poly_func, title=f'{target_var} vs {base} 回归曲线', save=True, save_path=scatter_reg_save_path)
    else:
        plotter.scatter_with_reg_line(main_var[0], target_var, save=True, save_path=scatter_reg_save_path, title=f'{target_var} vs {main_var[0]} with Regression Line')

def main():
    global regression_x, select_list
    data_new, extractor = load_and_preprocess_data()
    check_model_flags()
    data_new, high_order_cols, high_order_cols2 = process_high_order_terms(data_new, extractor)
    regression_x += high_order_cols
    select_list += high_order_cols2
    reg, coef_result = fit_regression(data_new, regression_x, regression_y)
    print_regression_results(reg, coef_result)
    plot_regression_results(data_new, coef_result)

if __name__ == "__main__":
    main()
#引用data_extractor和data_loader模块
from data_extractor import DataExtractor
from data_loader import DataLoader
from regression import RegressionModel
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import numpy as np


#global var
# name_list = ["year_code", "Stkcd", "accper", "digitaltransindex", "MD_Digital", "PeerDigital", "Subsidy", "Analyst", "ExcessDigita", "Size", 
#              "Lev", "ROA", "Growth", "PPE", "Age", "Board", "Dual", "Competition", "Synch", "Big4", "Top10", "SA", "PROVINCECD", "city", "nnindcd", 
#              "F100901A", "F100902A", "Betavals", "RDSpendSumRatio", "RDPerson"]
#to yx : 这是你需要修改的东西了，具体的看注释就行
file_path = "./0421.xlsx"   #把你的数据文件放在这个文件夹下面，并把0421.xlsx改成你的文件名，注意要保持.xlsx格式

#下面三个变量同时只能有一个为1
m1 = 0      #把这里置为1仅使用核心变量回归
m2 = 0      #这里置1，使用核心变量+控制变量
m3 = 1      #这里置1，使用核心变量+控制变量+固定效应（注意：如果使用m3，m1和m2必须为0）

#如果是高次项回归，设置这个开关为1，并在下面的high_order_terms里定义你需要的高次项
switch_for_high_order = 1
high_order_terms = [("peer_digital", 2)]   #定义高次项，格式是 [(base_col1, order1), (base_col2, order2), ...]


control_var = ["Size", "Lev", "ROA", "Growth", "PPE", "Age", "Board", "Dual", "Competition"]    #控制变量
target_var = "digitaltransindex"                                                                #被解释变量，不加入效应
main_var = ["peer_digital"]                                                                     #核心变量
must_var = ["year_code", "Stkcd", "accper", "nnindcd"]                                          #其他必须的变量
tail_process_list = ["Size", "Lev", "ROA", "Growth", "PPE", "Age", "Board", "Competition"]      #需要缩尾处理的列名称，格式是 “your_col_name”，注意要放在list里，即使只有一个
effext_src = ["accper", "Stkcd"]                                                                #计算固定效应的来源列，先时间后个体
effect_des = ["year_effect", "firm_effect"]                                                     #生成效应的结果
select_list = must_var + control_var + [target_var]                                             #全部需要的列名称
dependent_var = target_var

# ----------- 图像保存路径变量声明区（35行后） -----------
save_path_coef = "coef_plot.png"
save_path_resid = "resid_scatter.png"
save_path_hist = "resid_hist.png"
save_path_qq = "resid_qq.png"
scatter_reg_save_path = 'scatter_reg_line.svg'


#控制是否加入效应和控制变量
if(m1):     #只有peer_digital
    regression_x = main_var
    regression_y = target_var
elif(m2):   #加入控制变量
    regression_x = main_var + control_var 
    regression_y = target_var
elif(m3):   #加入固定效应
    regression_x = main_var + control_var 
    regression_y = target_var
else:
    pass


# 内生性检验配置（可按需要调整）
endog_vars = ["peer_digital"]
instrument_vars = ["accper"]  # 使用年份列作为工具变量
exog_vars = control_var




if __name__ == "__main__":
    # 初始化数据加载器，读取当前目录下的Excel文件
    loader = DataLoader()
    df = loader.load_excel(file_path)
    # 从全表中获取指定列的数据
    df = loader.get_dataframe(select_list)
    # 注册数据清洗器
    extractor = DataExtractor(df)
    # 清洗掉缺失值
    data = extractor.drop_na_rows(columns=select_list)
    #打印df的shape
    print(f"数据的形状: {df.shape}")
    #截尾处理异常值
    data_new = extractor.remove_outliers_by_quantiles(columns=tail_process_list)
    #缩尾处理
    data_new = extractor.tail_process(columns=tail_process_list)
    #在表格中加入peer digital列，
    data_new = extractor.peer_digital_calculate(index_cols=["accper", "nnindcd"], value_col="digitaltransindex", new_col_name="peer_digital")

    # data_new.to_excel("processed_data.xlsx", index=False)
    #data_new中删除nnindcd列中值为c43的行
    data_new = data_new[data_new["nnindcd"] != "C43"]

    # 只允许一个模型开关为1，避免混淆
    model_flags = [int(bool(m1)), int(bool(m2)), int(bool(m3))]
    if sum(model_flags) != 1:
        raise ValueError("m1/m2/m3 必须且只能有一个为1")

    if(switch_for_high_order):
        for ele in high_order_terms:
            data_new = extractor.add_high_order_col(ele[0], ele[1])
        tmp = []
        for ele in high_order_terms:
            for j in range(2, ele[1]+1):
                # new_col_name = f"{ele[0]}_{j}"
                tmp.append(f"{ele[0]}_{j}")
        print(tmp)
        regression_x += tmp
        select_list += tmp
        # data_new = extractor.add_high_order_terms(data_new, high_order_terms)
    #回归分析
    reg = RegressionModel(data_new, x_vars=regression_x, y_var=regression_y, method="linear")

    if m3:
        # 对齐 Stata: xtreg Digital PeerDigital controls i.year, fe robust
        reg.xtfit(
            entity_col="Stkcd",
            time_col="accper",
            time_effects=True,
            robust=True,
            cluster_entity=True,
            cluster_time=False,
        )
    else:
        reg.fit()

    coef_result = reg.coef_and_residuals(regression_x)

    print("回归结果：")
    print(f"截距: {coef_result['intercept']}")
    print(f"回归系数: {coef_result['coefficients']}")
    print(f"R²: {reg.score():.6f}")
    if m3:
        print(f"稳健标准误: {coef_result['std_errors']}")
        print(f"p值: {coef_result['p_values']}")

    # ----------- 绘图并保存 -----------
    from graph import PanelPlotter
    plotter = PanelPlotter(data_new)
    # 1. 回归系数及置信区间图
    coef = np.array(list(coef_result['coefficients'].values()))
    labels = list(coef_result['coefficients'].keys())
    if m3:
        std_err = np.array(list(coef_result['std_errors'].values()))
        ci_low = coef - 1.96 * std_err
        ci_high = coef + 1.96 * std_err
    else:
        # 若无std_err，简单用0.1做示例
        ci_low = coef - 0.1
        ci_high = coef + 0.1
    plotter.coef_plot(coef, ci_low, ci_high, labels=labels, save=True, save_path=save_path_coef)

    # 2. 残差散点图（残差 vs 拟合值）
    resid = coef_result['residuals']
    fitted = coef_result['fitted']
    temp_df = data_new.copy()
    import numpy as np
    temp_df['fitted'] = np.array(fitted)
    temp_df['residuals'] = np.array(resid)
    plotter2 = PanelPlotter(temp_df)
    # 2. Residuals vs Fitted Scatter Plot
    plotter2.scatter_plot('fitted', 'residuals', save=True, save_path=save_path_resid.replace('.png', '.svg'), title='Residuals vs Fitted')

    # 3. Residuals Histogram
    plotter2.hist_plot('residuals', save=True, save_path=save_path_hist.replace('.png', '.svg'), title='Histogram of Residuals')

    # 4. Residuals QQ Plot
    plotter2.qq_plot('residuals', save=True, save_path=save_path_qq.replace('.png', '.svg'), title='QQ Plot of Residuals')

    if(switch_for_high_order):
        # 检查 main_var[0] 的分布
        plotter.hist_plot(main_var[0], bins=30, save=True, save_path=f"{main_var[0]}_hist.svg", title=f"{main_var[0]} 分布直方图")
        # 用回归结果绘制自定义曲线
        coef_dict = coef_result['coefficients']
        intercept = coef_result['intercept']
        # 构造多项式系数数组，假设高次项变量名格式为 base_2, base_3 ...
        degree = high_order_terms[0][1] if high_order_terms else 2
        base = main_var[0]
        coeffs = [coef_dict.get(f"{base}_{d}", 0) if coef_dict.get(f"{base}_{d}") is not None else 0 for d in range(degree, 1, -1)]
        coeffs.append(coef_dict.get(base, 0) if coef_dict.get(base) is not None else 0)
        coeffs.append(intercept if intercept is not None else 0)
        poly_func = np.poly1d(coeffs)
        plotter.scatter_with_custom_curve(base, target_var, poly_func, title=f'{target_var} vs {base} 回归曲线', save=True, save_path=scatter_reg_save_path)
    else:
        # 5. Scatter with regression line: main_var vs target_var        
        plotter.scatter_with_reg_line(main_var[0], target_var, save=True, save_path=scatter_reg_save_path, title=f'{target_var} vs {main_var[0]} with Regression Line')