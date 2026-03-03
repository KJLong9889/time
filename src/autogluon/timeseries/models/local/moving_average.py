import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional

# 针对 Local Model 使用 AbstractLocalModel 是正确的，它比教程里的 AbstractTimeSeriesModel 更高效
from autogluon.timeseries.models.local.abstract_local_model import AbstractLocalModel

logger = logging.getLogger(__name__)

class MovingAverageInterpolationModel(AbstractLocalModel):
    """
    升级版季节性移动平均差值法 - 修复版
    符合 AutoGluon 自定义模型接口规范
    """

    allowed_local_model_args = [
        "seasonal_period",  
        "window_size",      
        "diff_clip_q",      
        "max_growth",       
        "min_growth",       
        "base_periods",     
    ]

    def _predict_with_local_model(
        self,
        time_series: pd.Series,
        **kwargs  # 官方规范：超参数会直接作为解包后的关键字参数传入
    ) -> pd.DataFrame:
        
        # 1. 获取超参数（从 kwargs 获取，这是自定义模型的标准做法）
        seasonal_period = kwargs.get("seasonal_period", 12)
        window = kwargs.get("window_size", 36)
        diff_clip_q = kwargs.get("diff_clip_q", (0.1, 0.9)) 
        q_low_param, q_high_param = diff_clip_q[0], diff_clip_q[1]
        
        max_growth = kwargs.get("max_growth", 0.25)
        min_growth = kwargs.get("min_growth", -0.15)
        base_period_multipliers = kwargs.get("base_periods", [1, 2])

        prediction_length = self.prediction_length

        # 2. 严格的数据清洗（参考教程 preprocess 思想）
        # 将数据转为 float64，处理 Inf，并填充/删除 NaN
        clean_series = pd.to_numeric(time_series, errors='coerce').replace([np.inf, -np.inf], np.nan)
        # 内部前向填充，确保中间没有空洞
        clean_series = clean_series.ffill().bfill() 
        history_values = clean_series.dropna().values.tolist()
        
        # 兜底：如果完全没数据
        if not history_values:
            history_values = [0.0]
        
        forecast_values = []

        # 3. 递归预测循环
        for i in range(prediction_length):
            current_len = len(history_values)
            
            # 如果历史太短无法计算季节性，直接用最后一个值（Naive）
            if current_len <= seasonal_period:
                pred = history_values[-1]
            else:
                # 计算季节性差分
                needed_len = window + seasonal_period
                slice_start = -needed_len if current_len > needed_len else 0
                hist_tail = np.array(history_values[slice_start:])
                
                # 差分序列计算
                seasonal_diff = hist_tail[seasonal_period:] - hist_tail[:-seasonal_period]
                seasonal_diff = seasonal_diff[np.isfinite(seasonal_diff)] # 过滤无效值

                if len(seasonal_diff) > 0:
                    # 异常值截断
                    q_low = np.quantile(seasonal_diff, q_low_param)
                    q_high = np.quantile(seasonal_diff, q_high_param)
                    seasonal_diff_clipped = np.clip(seasonal_diff, q_low, q_high)

                    # 加权平均差值
                    weights = np.linspace(0.5, 1.0, len(seasonal_diff_clipped))
                    avg_diff = np.average(seasonal_diff_clipped, weights=weights)
                else:
                    avg_diff = 0.0

                # 计算季节基准（同比值）
                base_vals = []
                for k_mult in base_period_multipliers:
                    lag_idx = seasonal_period * k_mult
                    if current_len >= lag_idx:
                        val = history_values[-lag_idx]
                        if np.isfinite(val):
                            base_vals.append(val)
                
                if base_vals:
                    base_val = np.mean(base_vals)
                    raw_forecast = base_val + avg_diff
                    
                    # 增长率约束
                    y_last = history_values[-1]
                    pred = np.clip(
                        raw_forecast,
                        y_last * (1 + min_growth),
                        y_last * (1 + max_growth)
                    )
                else:
                    pred = history_values[-1]

            # 4. 最终安全性检查：防止 NaN 产生
            if not np.isfinite(pred):
                pred = history_values[-1] if np.isfinite(history_values[-1]) else 0.0

            forecast_values.append(pred)
            history_values.append(pred)

        # 5. 封装结果（!!! 必须包含 mean 列 !!!）
        forecast_array = np.array(forecast_values)
        
        # 教程要求：返回包含 mean 和所有分位数列的 DataFrame
        forecast_df = {}
        for q in self.quantile_levels:
            forecast_df[str(q)] = forecast_array
        
        # 强制添加 mean 列，通常点预测等于 0.5 分位数
        forecast_df["mean"] = forecast_array
        
        return pd.DataFrame(forecast_df)