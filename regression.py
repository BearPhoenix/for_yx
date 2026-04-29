# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm

class RegressionModel:
    """
    通用回归器，支持多种回归方式，便于扩展。
    """
    @staticmethod
    def available_methods():
        return ['linear', 'ridge', 'lasso', 'rf', 'panel_fe', 'xtreg']
    
    
    def __init__(self, dataframe, x_vars, y_var, method='linear', **kwargs):
        """
        :param dataframe: 输入的pandas DataFrame
        :param x_vars: 自变量名列表
        :param y_var: 因变量名（字符串）
        :param method: 回归方法，默认'linear'，可选'ridge', 'lasso', 'rf'，也可传入自定义回归器对象（需有fit/predict/score方法）
        :param kwargs: 传递给回归模型的其他参数
        """
        self.df = dataframe
        self.x_vars = x_vars
        # 兼容传入单元素list/tuple的情况，统一为列名字符串
        if isinstance(y_var, (list, tuple)):
            if len(y_var) != 1:
                raise ValueError("y_var 需为字符串或单元素list/tuple")
            self.y_var = y_var[0]
        else:
            self.y_var = y_var
        self.method = method
        self.kwargs = kwargs
        self.model = self._init_model(method, **kwargs)
        self.panel_result = None
        self.fitted = False

    def _is_panel_method(self):
        return self.method in ('panel_fe', 'xtreg')

    def _init_model(self, method, **kwargs):
        # 支持自定义回归器对象（需有fit/predict/score方法）
        if hasattr(method, 'fit') and hasattr(method, 'predict'):
            return method
        if method == 'linear':
            return LinearRegression(**kwargs)
        elif method == 'ridge':
            return Ridge(**kwargs)
        elif method == 'lasso':
            return Lasso(**kwargs)
        elif method == 'rf':
            return RandomForestRegressor(**kwargs)
        elif method == 'panel_fe':
            # PanelOLS 在 fit 阶段依赖数据列构造，初始化时不实例化
            return None
        elif method == 'xtreg':
            # 通过 xtfit 执行，初始化时不实例化
            return None
        else:
            raise ValueError(f"未知回归方法: {method}")

    def xtfit(self, entity_col, time_col, time_effects=True, robust=True,
              cluster_entity=True, cluster_time=False):
        """
        与 Stata xtreg 对齐的拟合入口（固定效应面板回归）。

        对应关系：
        - xtreg ..., fe            -> entity_effects=True
        - xtreg ... i.year, fe     -> entity_effects=True, time_effects=True
        - xtreg ..., fe robust     -> cov_type='clustered', cluster_entity=True
        """
        self.method = 'xtreg'
        self.kwargs['entity_col'] = entity_col
        self.kwargs['time_col'] = time_col
        self.kwargs['time_effects'] = bool(time_effects)

        if robust:
            self.kwargs['cov_type'] = 'clustered'
            self.kwargs['cluster_entity'] = bool(cluster_entity)
            self.kwargs['cluster_time'] = bool(cluster_time)
        else:
            self.kwargs['cov_type'] = 'unadjusted'
            self.kwargs['cluster_entity'] = False
            self.kwargs['cluster_time'] = False

        return self._fit_panel_fe()

    def _fit_panel_fe(self):
        try:
            from linearmodels.panel import PanelOLS
        except ImportError:
            raise ImportError("请先安装linearmodels：pip install linearmodels")

        entity_col = self.kwargs.get('entity_col')
        time_col = self.kwargs.get('time_col')
        time_effects = self.kwargs.get('time_effects', True)
        cov_type = self.kwargs.get('cov_type', 'clustered')
        cluster_entity = self.kwargs.get('cluster_entity', True)
        cluster_time = self.kwargs.get('cluster_time', False)

        if not isinstance(entity_col, str) or not entity_col:
            raise ValueError("panel_fe 模式下必须提供 entity_col（个体列名）")
        if not isinstance(time_col, str) or not time_col:
            raise ValueError("panel_fe 模式下必须提供 time_col（时间列名）")

        required_cols = list(dict.fromkeys(self.x_vars + [self.y_var, entity_col, time_col]))
        missing_cols = [c for c in required_cols if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"以下列不存在: {missing_cols}")

        data = self.df[required_cols].copy().replace({pd.NA: np.nan})
        data[self.x_vars] = data[self.x_vars].apply(pd.to_numeric, errors='coerce')
        data[self.y_var] = pd.to_numeric(data[self.y_var], errors='coerce')

        # 与 xtreg 的可用样本逻辑一致：仅使用完整观测
        data = data.dropna(subset=self.x_vars + [self.y_var, entity_col, time_col])
        if data.shape[0] == 0:
            raise ValueError("无可用于面板回归的完整样本")

        panel_data = data.set_index([entity_col, time_col]).sort_index()

        model = PanelOLS(
            panel_data[self.y_var],
            panel_data[self.x_vars],
            entity_effects=True,
            time_effects=time_effects,
            drop_absorbed=True,
        )

        fit_kwargs = {'cov_type': cov_type}
        if cov_type == 'clustered':
            fit_kwargs['cluster_entity'] = cluster_entity
            fit_kwargs['cluster_time'] = cluster_time

        self.panel_result = model.fit(**fit_kwargs)
        self.model = self.panel_result
        self.fitted = True
        return self

    def fit(self):
        if self._is_panel_method():
            return self._fit_panel_fe()

        X = self.df[self.x_vars].copy()
        y = self.df[self.y_var].copy()

        # 将pd.NA等统一转为np.nan，并尽量转换为数值类型
        X = X.replace({pd.NA: np.nan}).apply(pd.to_numeric, errors='coerce')
        y = pd.to_numeric(y.replace({pd.NA: np.nan}), errors='coerce')

        # 仅保留可用于回归的完整样本
        valid_mask = X.notna().all(axis=1) & y.notna()
        if valid_mask.sum() == 0:
            raise ValueError("无可用于回归的完整样本：请检查自变量/因变量是否包含大量缺失值")

        X = X.loc[valid_mask]
        y = y.loc[valid_mask]
        self.model.fit(X, y)
        self.fitted = True
        return self

    def predict(self, X=None):
        if not self.fitted:
            raise RuntimeError("请先调用fit()进行拟合")
        if self._is_panel_method():
            if X is not None:
                raise ValueError("panel_fe 模式暂不支持传入新 X 进行预测")
            return self.panel_result.fitted_values
        if X is None:
            X = self.df[self.x_vars]
        return self.model.predict(X)

    def score(self, X=None, y=None):
        """
        计算模型在给定数据上的评分。

        在 sklearn 的大多数回归模型中，`score` 返回的是决定系数 R²：
        - 取值通常不大于 1，越接近 1 说明拟合效果越好；
        - 0 表示与使用 y 的均值作为预测基线相当；
        - 小于 0 表示模型效果比基线更差。

        注意：如果传入的是自定义回归器对象，其 `score` 的含义以该对象实现为准。
        """
        if not self.fitted:
            raise RuntimeError("请先调用fit()进行拟合")
        if self._is_panel_method():
            # 与固定效应回归常见报告口径一致
            return float(self.panel_result.rsquared_within)
        if X is None:
            X = self.df[self.x_vars]
        if y is None:
            y = self.df[self.y_var]
        return self.model.score(X, y)

    def coef_and_residuals(self, x_var_list):
        """
        基于已拟合模型，按传入的自变量列名列表输出回归系数和拟合残差。

        :param x_var_list: list，自变量列名列表
        :return: dict，包含：
                 - coefficients: 每个自变量对应的回归系数（dict）
                 - intercept: 截距
                 - residuals: 残差序列（pd.Series）
        """
        if not self.fitted:
            raise RuntimeError("请先调用fit()进行拟合")

        if not isinstance(x_var_list, list) or len(x_var_list) == 0:
            raise ValueError("x_var_list 必须是非空 list")

        missing_cols = [c for c in x_var_list if c not in self.x_vars]
        if missing_cols:
            raise ValueError(f"以下列不在已拟合自变量中: {missing_cols}")

        if self._is_panel_method():
            params = self.panel_result.params.to_dict()
            selected_coef = {col: params.get(col, np.nan) for col in x_var_list}
            residuals = self.panel_result.resids
            fitted = self.panel_result.fitted_values
            return {
                'coefficients': selected_coef,
                'intercept': None,
                'residuals': residuals,
                'fitted': fitted,
                'std_errors': self.panel_result.std_errors.to_dict(),
                'p_values': self.panel_result.pvalues.to_dict(),
            }

        # 残差基于已拟合时使用的自变量集合 self.x_vars 计算，避免再次 fit
        X = self.df[self.x_vars].copy()
        y = self.df[self.y_var].copy()

        # 将缺失值和非数值统一处理
        X = X.replace({pd.NA: np.nan}).apply(pd.to_numeric, errors='coerce')
        y = pd.to_numeric(y.replace({pd.NA: np.nan}), errors='coerce')

        valid_mask = X.notna().all(axis=1) & y.notna()
        if valid_mask.sum() == 0:
            raise ValueError("无可用于回归的完整样本：请检查自变量/因变量是否包含大量缺失值")

        X_valid = X.loc[valid_mask]
        y_valid = y.loc[valid_mask]

        if not hasattr(self.model, 'coef_'):
            raise ValueError(f"当前回归方法 {self.method} 不支持直接输出回归系数")

        y_pred = self.model.predict(X_valid)
        residuals = y_valid - y_pred

        all_coef = dict(zip(self.x_vars, self.model.coef_))
        selected_coef = {col: all_coef[col] for col in x_var_list}

        return {
            'coefficients': selected_coef,
            'intercept': getattr(self.model, 'intercept_', None),
            'residuals': residuals,
            'fitted': y_pred
        }

    def endogeneity_test(self, endog_vars, instrument_vars, exog_vars=None, alpha=0.05):
        """
        使用控制函数（Control Function）的 Durbin-Wu-Hausman 思路进行内生性检验。

        参数：
        :param endog_vars: list，怀疑存在内生性的变量名列表（必须是 self.x_vars 的子集）
        :param instrument_vars: list，工具变量列表
        :param exog_vars: list，可选，外生控制变量列表；若为 None，则自动取 self.x_vars 中除 endog_vars 外的变量
        :param alpha: float，显著性水平，默认 0.05

        返回：
        dict，包含：
        - test_stat: F 统计量（联合检验所有一阶段残差系数 = 0）
        - p_value: 对应 p 值
        - alpha: 显著性水平
        - reject_null: bool，是否拒绝原假设
        - has_endogeneity: bool，是否判定存在内生性
        - conclusion: 文字结论

        如何根据返回值判断内生性：
        1) 原假设 H0：endog_vars 是外生的（不存在内生性）。
        2) 若 p_value < alpha（且 reject_null=True），则拒绝 H0，判定"存在内生性"。
        3) 若 p_value >= alpha（且 reject_null=False），则"未发现显著内生性证据"。
        """
        if not isinstance(endog_vars, list) or len(endog_vars) == 0:
            raise ValueError("endog_vars 必须是非空 list")
        if not isinstance(instrument_vars, list) or len(instrument_vars) == 0:
            raise ValueError("instrument_vars 必须是非空 list")

        if exog_vars is None:
            exog_vars = [v for v in self.x_vars if v not in endog_vars]
        if not isinstance(exog_vars, list):
            raise ValueError("exog_vars 必须是 list 或 None")

        invalid_endog = [v for v in endog_vars if v not in self.x_vars]
        if invalid_endog:
            raise ValueError(f"以下 endog_vars 不在已设定自变量中: {invalid_endog}")

        overlap = set(endog_vars).intersection(set(instrument_vars))
        if overlap:
            raise ValueError(f"endog_vars 与 instrument_vars 不能重叠: {sorted(list(overlap))}")

        # 判断是否为面板数据模式
        is_panel = self._is_panel_method()
        
        if is_panel:
            # 面板数据模式：使用 linearmodels 进行内生性检验
            return self._endogeneity_test_panel(endog_vars, instrument_vars, exog_vars, alpha)
        else:
            # 普通 OLS 模式
            return self._endogeneity_test_ols(endog_vars, instrument_vars, exog_vars, alpha)

    def _endogeneity_test_panel(self, endog_vars, instrument_vars, exog_vars, alpha):
        """面板数据模式下的内生性检验（使用 linearmodels）"""
        try:
            from linearmodels.panel import PanelOLS
        except ImportError:
            raise ImportError("请先安装 linearmodels：pip install linearmodels")

        entity_col = self.kwargs.get('entity_col')
        time_col = self.kwargs.get('time_col')
        time_effects = self.kwargs.get('time_effects', True)

        # 准备数据：包含因变量、内生变量、外生变量、工具变量、个体和时间标识
        required_cols = list(dict.fromkeys(
            [self.y_var] + endog_vars + exog_vars + instrument_vars + [entity_col, time_col]
        ))
        missing_cols = [c for c in required_cols if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"以下列不存在: {missing_cols}")

        data = self.df[required_cols].copy()
        data = data.replace({pd.NA: np.nan})
        data[self.x_vars] = data[self.x_vars].apply(pd.to_numeric, errors='coerce')
        data[self.y_var] = pd.to_numeric(data[self.y_var], errors='coerce')
        data = data.dropna(subset=self.x_vars + [self.y_var, entity_col, time_col])
        if data.shape[0] == 0:
            raise ValueError("有效样本为空：请检查变量是否存在缺失值或非数值")

        # 设置面板索引
        panel_data = data.set_index([entity_col, time_col]).sort_index()

        # 一阶段：每个内生变量对 外生变量 + 工具变量 回归，提取残差
        first_stage_X_cols = exog_vars + instrument_vars
        residual_names = []
        
        for v in endog_vars:
            first_stage_model = PanelOLS(
                panel_data[v],
                panel_data[first_stage_X_cols],
                entity_effects=True,
                time_effects=time_effects,
                drop_absorbed=True
            ).fit(cov_type='unadjusted')
            
            r_name = f"__resid_{v}"
            panel_data[r_name] = first_stage_model.resids
            residual_names.append(r_name)

        # 二阶段：因变量对 外生变量 + 内生变量 + 残差 回归
        second_stage_vars = exog_vars + endog_vars + residual_names
        second_stage_model = PanelOLS(
            panel_data[self.y_var],
            panel_data[second_stage_vars],
            entity_effects=True,
            time_effects=time_effects,
            drop_absorbed=True
        ).fit(cov_type='unadjusted')

        # 联合检验残差系数是否全为 0
        hypothesis = " = 0, ".join(residual_names) + " = 0"
        f_test_res = second_stage_model.f_test(hypothesis)

        f_stat = float(np.asarray(f_test_res.fvalue).reshape(-1)[0])
        p_value = float(np.asarray(f_test_res.pvalue).reshape(-1)[0])
        reject_null = p_value < alpha
        has_endogeneity = reject_null

        return {
            'test_name': 'Durbin-Wu-Hausman (panel control function)',
            'n_obs': int(panel_data.shape[0]),
            'endog_vars': endog_vars,
            'instrument_vars': instrument_vars,
            'exog_vars': exog_vars,
            'test_stat': f_stat,
            'p_value': p_value,
            'alpha': alpha,
            'reject_null': reject_null,
            'has_endogeneity': has_endogeneity,
            'conclusion': '存在内生性' if has_endogeneity else '未发现显著内生性证据'
        }

    def _endogeneity_test_ols(self, endog_vars, instrument_vars, exog_vars, alpha):
        """普通 OLS 模式下的内生性检验（使用 statsmodels）"""
        required_cols = list(dict.fromkeys([self.y_var] + endog_vars + exog_vars + instrument_vars))
        missing_cols = [c for c in required_cols if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"以下列不存在: {missing_cols}")

        data = self.df[required_cols].copy()
        data = data.replace({pd.NA: np.nan}).apply(pd.to_numeric, errors='coerce').dropna()
        if data.shape[0] == 0:
            raise ValueError("有效样本为空：请检查变量是否存在缺失值或非数值")

        y = data[self.y_var]

        # 一阶段：每个可疑内生变量对 exog + instruments 回归，提取残差
        first_stage_X = sm.add_constant(data[exog_vars + instrument_vars], has_constant='add')
        residual_names = []
        for v in endog_vars:
            first_stage_model = sm.OLS(data[v], first_stage_X).fit()
            r_name = f"__resid_{v}"
            data[r_name] = first_stage_model.resid
            residual_names.append(r_name)

        # 二阶段：y 对 exog + endog + 一阶段残差回归，联合检验残差系数是否全为 0
        second_stage_vars = exog_vars + endog_vars + residual_names
        second_stage_X = sm.add_constant(data[second_stage_vars], has_constant='add')
        second_stage_model = sm.OLS(y, second_stage_X).fit()

        hypothesis = " = 0, ".join(residual_names) + " = 0"
        f_test_res = second_stage_model.f_test(hypothesis)

        f_stat = float(np.asarray(f_test_res.fvalue).reshape(-1)[0])
        p_value = float(np.asarray(f_test_res.pvalue).reshape(-1)[0])
        reject_null = p_value < alpha
        has_endogeneity = reject_null

        return {
            'test_name': 'Durbin-Wu-Hausman (control function)',
            'n_obs': int(data.shape[0]),
            'endog_vars': endog_vars,
            'instrument_vars': instrument_vars,
            'exog_vars': exog_vars,
            'test_stat': f_stat,
            'p_value': p_value,
            'alpha': alpha,
            'reject_null': reject_null,
            'has_endogeneity': has_endogeneity,
            'conclusion': '存在内生性' if has_endogeneity else '未发现显著内生性证据'
        }

    

# 示例用法：
# reg = RegressionModel(df, x_vars=['x1', 'x2'], y_var='y', method='ridge', alpha=1.0)
# reg.fit()
# preds = reg.predict()
# score = reg.score()
