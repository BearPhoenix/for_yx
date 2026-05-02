from data_extractor import DataExtractor
from data_loader import DataLoader
from regression import RegressionModel
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import numpy as np
from graph import PanelPlotter

debug_log = 0
customer = 1

file_path = "./0421.xlsx"
control_var = ["Size", "Lev", "ROA", "Growth", "PPE", "Age", "Board", "Dual", "Competition"]    #控制变量
target_var = "digitaltransindex"                                                                #被解释变量，不加入效应
main_var = ["peer_digital"]                                                                     #核心变量
must_var = ["year_code", "Stkcd", "accper", "nnindcd"]                                          #其他必须的变量
tail_process_list = ["Size", "Lev", "ROA", "Growth", "PPE", "Age", "Board", "Competition"]      #需要缩尾处理的列名称，格式是 “your_col_name”，注意要放在list里，即使只有一个
effect_src = ["accper", "Stkcd"]                                                                #计算固定效应的来源列，先时间后个体
effect_des = ["year_effect", "firm_effect"]                                                     #生成效应的结果
select_list = must_var + control_var + [target_var]                                             #全部需要的列名称
dependent_var = target_var
regression_x = []
regression_y = target_var

var_dict = {
    "control_var": control_var,
    "target_var": target_var,
    "main_var": main_var,
    "must_var": must_var,
    "tail_var": tail_process_list,
    "effect_src": effect_src,
    "effect_des": effect_des,
    "select_var": select_list,
    "regression_x": regression_x,
    "regression_y": regression_y
}

save_path_coef = "coef_plot.png"
save_path_resid = "resid_scatter.png"
save_path_hist = "resid_hist.png"
save_path_qq = "resid_qq.png"
scatter_reg_save_path = 'scatter_reg_line.svg'

graph_dic = {
    "save_path_coef": save_path_coef,
    "save_path_resid": save_path_resid,
    "save_path_hist": save_path_hist,
    "save_path_qq": save_path_qq,
}

#变量赋值
def init_variable_by_level(dic, file_path, level):
    """
    初始化变量，根据level设置回归自变量
    level=1: 只有核心变量
    level=2: 核心变量 + 控制变量
    level=3: 核心变量 + 控制变量 + 固定效应
    """
    loader = DataLoader()
    
    if(level == 1):
        dic["regression_x"] = dic["main_var"]
    elif(level == 2):
        dic["regression_x"] = dic["main_var"] + dic["control_var"]
    elif(level == 3):
        dic["regression_x"] = dic["main_var"] + dic["control_var"]
    else:
        raise ValueError("level取值需要小于4")
    
    df = loader.load_excel(file_path)    
    df = loader.get_dataframe(dic["select_var"])
    extractor = DataExtractor(df)
    data = extractor.drop_na_rows(columns=dic["select_var"])
    if(debug_log):
        print(f"数据的形状: {df.shape}")
    data_new = extractor.remove_outliers_by_quantiles(columns=dic["tail_var"])
    data_new = extractor.peer_digital_calculate(index_cols=["accper", "nnindcd"], value_col="digitaltransindex", new_col_name="peer_digital")
    return data_new


def regression_analysis(reg, type):
    if(type == "reg"):
        reg.fit()
    elif(type == "xtreg"):
        reg.xtfit(
            entity_col="Stkcd",
            time_col="accper",
            time_effects=True,
            robust=True,
            cluster_entity=True,
            cluster_time=False,
        )
    else:
        raise ValueError("type取值只能是'reg'或'xtreg'")
    coef_result = reg.coef_and_residuals(reg.x_vars)
    print("回归结果：")
    print(f"截距: {coef_result['intercept']}")
    print(f"回归系数: {coef_result['coefficients']}")
    print(f"R²: {reg.score():.6f}")
    print(f"稳健标准误: {coef_result['std_errors']}")
    print(f"p值: {coef_result['p_values']}")
    
def graph(graph_dic, data, reg, level):
    plotter = PanelPlotter(data)
    # 1. 回归系数及置信区间图
    coef_result = reg.coef_and_residuals(var_dict["regression_x"])
    coef = np.array(list(coef_result['coefficients'].values()))
    labels = list(coef_result['coefficients'].keys())
    if(level == 3):
        std_err = np.array(list(coef_result['std_errors'].values()))
        ci_low = coef - 1.96 * std_err
        ci_high = coef + 1.96 * std_err
    else:
        # 若无std_err，简单用0.1做示例
        ci_low = coef - 0.1
        ci_high = coef + 0.1
    plotter.coef_plot(coef, ci_low, ci_high, labels=labels, save=True, save_path=graph_dic["save_path_coef"])

    # 2. 残差散点图（残差 vs 拟合值）
    resid = coef_result['residuals']
    fitted = coef_result['fitted']
    temp_df = data.copy()
    temp_df['fitted'] = np.array(fitted)
    temp_df['residuals'] = np.array(resid)
    plotter2 = PanelPlotter(temp_df)
    # 2. Residuals vs Fitted Scatter Plot
    plotter2.scatter_plot('fitted', 'residuals', save=True, save_path=graph_dic["save_path_resid"].replace('.png', '.svg'), title='Residuals vs Fitted')

    # 3. Residuals Histogram
    plotter2.hist_plot('residuals', save=True, save_path=graph_dic["save_path_hist"].replace('.png', '.svg'), title='Histogram of Residuals')

    # 4. Residuals QQ Plot
    plotter2.qq_plot('residuals', save=True, save_path=graph_dic["save_path_qq"].replace('.png', '.svg'), title='QQ Plot of Residuals')

def graph_linear_regression(data, dic, save_path):
    plotter = PanelPlotter(data)
    plotter.scatter_with_reg_line(dic["main_var"][0], dic["regression_y"], save=True, save_path=save_path, title=f'{dic["regression_y"]} vs {dic["main_var"][0]} with Regression Line')

#纯个性化需求，
def graph_high_order_regression(data, dic, reg, save_path):
    pass

def main():
    if(customer):   #在这里新增选定列
        # ls_var = ["Analyst", "Subsidy"]
        new_src = "Subsidy"
        var_dict["select_var"].append(new_src)
        pass
    data = init_variable_by_level(var_dict, file_path, 3)
    if(customer):
        new_col = ["x_2", "px", "px_2"]
        data["x_2"] = data[new_src] ** 2
        data["px"] = data[new_src] * data["peer_digital"]
        data["px_2"] = data["peer_digital"] * data[new_src] ** 2
        var_dict["regression_x"] += new_col
        var_dict["regression_x"].append(new_src)
        new_col = []
        pass
    print("===============")
    print("回归自变量列表：", var_dict["regression_x"])
    reg = RegressionModel(data, x_vars=var_dict["regression_x"], y_var=var_dict["regression_y"], method="linear")
    regression_analysis(reg, "xtreg")
    # graph(graph_dic, data, reg, 3)
    #线性回归散点图
    # graph_linear_regression(data, var_dict, scatter_reg_save_path)

    #高次回归散点图，纯粹的个性化需求，暂时不做封装了
    if(customer):
        plotter = PanelPlotter(data)
        plotter.scatter_3d_with_surface(
            x_col=new_src,
            y_col=var_dict["main_var"][0],
            z_col=var_dict["regression_y"],
            model=reg.model,
            dic=var_dict,
            save=True,
            save_path="./test.svg"
        )

    # 自定义



if __name__ == "__main__":
    main()
    

