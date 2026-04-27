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
# select_list = ["year_code", "Stkcd", "accper", "digitaltransindex", "nnindcd"]  #把你感兴趣的列名称写在这个变量里，格式是 “your_col_name”
# tail_process_list = ["peer_digital"]  #需要缩尾处理的列名称，格式是 “your_col_name”，注意要放在list里，即使只有一个
# regression_y = "digitaltransindex"  #回归分析的因变量名称，格式是 “your_col_name”
# regression_x = ["peer_digital"]  #回归分析的自变量名称，格式是 [“your_col_name”]，注意要放在list里，即使只有一个

#下面三个变量同时只能有一个为1
m1 = 0      #把这里置为1仅使用核心变量回归
m2 = 0      #这里置1，使用核心变量+控制变量
m3 = 1      #这里置1，使用核心变量+控制变量+固定效应（注意：如果使用m3，m1和m2必须为0）


control_var = ["Size", "Lev", "ROA", "Growth", "PPE", "Age", "Board", "Dual", "Competition"]    #控制变量
target_var = "digitaltransindex"                                                                #被解释变量，不加入效应
main_var = ["peer_digital"]                                                                     #核心变量
must_var = ["year_code", "Stkcd", "accper", "nnindcd"]                                          #其他必须的变量
tail_process_list = ["Size", "Lev", "ROA", "Growth", "PPE", "Age", "Board", "Competition"]      #需要缩尾处理的列名称，格式是 “your_col_name”，注意要放在list里，即使只有一个
effext_src = ["accper", "Stkcd"]                                                                #计算固定效应的来源列，先时间后个体
effect_des = ["year_effect", "firm_effect"]                                                     #生成效应的结果
select_list = must_var + control_var + [target_var]                                             #全部需要的列名称


# regression_x = ["peer_digital"] + control_var 
dependent_var = target_var
# regression_y = "digitaltransindex_new"

# ----------- 图像保存路径变量声明区（35行后） -----------
save_path_coef = "coef_plot.png"
save_path_resid = "resid_scatter.png"
save_path_hist = "resid_hist.png"
save_path_qq = "resid_qq.png"



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

    # 5. Scatter with regression line: main_var vs target_var
    scatter_reg_save_path = 'scatter_reg_line.svg'
    plotter.scatter_with_reg_line(main_var[0], target_var, save=True, save_path=scatter_reg_save_path, title=f'{target_var} vs {main_var[0]} with Regression Line')