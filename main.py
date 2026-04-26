#引用data_extractor和data_loader模块
from data_extractor import DataExtractor
from data_loader import DataLoader
from regression import RegressionModel
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso



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
instrument_vars = ["year_effect"]
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

    # test_result = reg.endogeneity_test(
    #     endog_vars=endog_vars,
    #     instrument_vars=instrument_vars,
    #     exog_vars=exog_vars,
    #     alpha=0.05
    # )
    # print("内生性检验结果：")
    # print(f"检验方法: {test_result['test_name']}")
    # print(f"样本量: {test_result['n_obs']}")
    # print(f"F统计量: {test_result['test_stat']:.6f}")
    # print(f"p值: {test_result['p_value']:.6f}")
    # print(f"结论: {test_result['conclusion']}")
    
    

    # loader.get_info()