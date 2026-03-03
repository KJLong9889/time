import os
import time
import warnings
import numpy as np
import pandas as pd  # <---【修复1】必须放在文件最顶部，解决 UnboundLocalError
import torch
from torch.utils.data import DataLoader, Dataset

# AutoGluon 基础类
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
from autogluon.timeseries.dataset.ts_dataframe import TimeSeriesDataFrame

# 假设你的 TimeXer 模型和工具类在这个路径下
# 请根据你实际的项目结构调整这些 import
try:
    from .timexer_lib.utils.timefeatures import time_features
    # 假设你的模型定义在这里，如果类名不同请修改
    from .timexer_lib.models import TimeXer as TimeXerModel 
except ImportError:
    # 仅作为示例，如果找不到库则忽略，实际运行时会报错
    pass

class TimeSeriesDataset(Dataset):
    def __init__(self, data, df_stamp, freq='h', flag='train', size=None,
                 features='S', target='value', scale=True, timeenc=0, freq_map=None):
        # 初始化参数
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        # -----------------------------------------------------------
        # 【修复2】健壮的时间特征处理
        # 确保传入 time_features 的是 DatetimeIndex 或 Series，而不是 DataFrame
        # -----------------------------------------------------------
        if isinstance(df_stamp, pd.DataFrame):
            if 'date' in df_stamp.columns:
                time_vals = pd.to_datetime(df_stamp['date'].values)
            elif isinstance(df_stamp.index, pd.DatetimeIndex):
                time_vals = df_stamp.index
            else:
                # 尝试取第一列
                time_vals = pd.to_datetime(df_stamp.iloc[:, 0].values)
        else:
            time_vals = pd.to_datetime(df_stamp)

        # 生成时间特征 (dayofweek, month, etc.)
        self.data_stamp = time_features(time_vals, freq=freq)
        
        # 处理实际数据 X 和 Y
        # 注意：这里假设 data 已经是 numpy array 或者处理好的 DataFrame
        if isinstance(data, pd.DataFrame):
            self.data_x = data.values
            self.data_y = data.values
        else:
            self.data_x = data
            self.data_y = data

    def __getitem__(self, index):
        # 标准的时间序列切片逻辑
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1


class TimeXerModel(AbstractTimeSeriesModel):
    def __init__(self, freq=None, prediction_length: int = 24, path: str = None, 
                 name: str = "TimeXer", eval_metric: str = None, 
                 hyperparameters: dict = None, **kwargs):
        """
        初始化模型
        """
        # 修正 hyperparameters 默认值
        hyperparameters = hyperparameters if hyperparameters is not None else {}
        
        super().__init__(
            path=path,
            freq=freq,
            prediction_length=prediction_length,
            name=name,
            eval_metric=eval_metric,
            hyperparameters=hyperparameters,
            **kwargs
        )
        
        # 从超参数中读取模型配置
        self.seq_len = self.params.get('seq_len', 96)
        self.label_len = self.params.get('label_len', 48)
        self.pred_len = prediction_length  # 预测长度通常由 AutoGluon 指定
        self.batch_size = self.params.get('batch_size', 32)
        self.features = self.params.get('features', 'S') # S:Univariate, M:Multivariate
        self.model = None # 实际的 torch 模型

    def _fit(self, train_data, val_data=None, time_limit=None, **kwargs):
        """
        训练入口函数
        """
        # -----------------------------------------------------------
        # 【修复3】数据预处理：将 AutoGluon 的 MultiIndex 数据转换为模型可用的格式
        # -----------------------------------------------------------
        
        # 1. 打印调试信息（可选）
        # print(f"DEBUG: Input train_data shape: {train_data.shape}")

        # 2. 重置索引，将 item_id 和 timestamp 变为普通列
        train_df = train_data.reset_index()

        # 3. 提取时间戳列
        if 'timestamp' in train_df.columns:
            timestamps = train_df['timestamp']
        else:
            # 兜底：寻找 datetime 类型的列
            timestamps = train_df.select_dtypes(include=[np.datetime64, 'datetime']).iloc[:, 0]

        # 4. 准备 df_stamp (用于生成时间特征)
        # 这里解决了 'UnboundLocalError: local variable 'pd' referenced before assignment'
        # 因为 pd 已经在文件顶部导入了
        df_stamp = pd.DataFrame({"date": timestamps})

        # 5. 准备目标数据 (Target)
        # 确保使用用户指定的 target 列 (通常是 'value')
        target_col = self.target if self.target is not None else 'value'
        if target_col not in train_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in train_data")
            
        data_values = train_df[[target_col]].values # 转为 numpy array
        
        # -----------------------------------------------------------
        # 实例化 Dataset 和 DataLoader
        # -----------------------------------------------------------
        train_dataset = TimeSeriesDataset(
            data=data_values,
            df_stamp=df_stamp,
            freq=self.freq,
            flag='train',
            size=[self.seq_len, self.label_len, self.pred_len],
            features=self.features,
            target=target_col
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0, # 避免多进程报错，生产环境可调大
            drop_last=True
        )

        # 初始化你的模型 (这里需要根据你的 TimeXerModel 实际参数调整)
        # self.model = TimeXerModel(
        #     self.seq_len, self.label_len, self.pred_len, ...
        # ).float()
        
        # if torch.cuda.is_available():
        #     self.model.cuda()

        # 模拟训练循环 (你需要填入实际的训练代码)
        print(f"Training TimeXer with {len(train_dataset)} samples...")
        # for epoch in range(self.params.get('epochs', 10)):
        #     for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        #         optimizer.zero_grad()
        #         outputs = self.model(batch_x, batch_x_mark)
        #         loss = criterion(outputs, batch_y)
        #         loss.backward()
        #         optimizer.step()
        
        # 训练完成后返回自身
        return self

    def predict(self, data, known_covariates=None, **kwargs):
        """
        预测入口函数
        """
        # 简化的预测逻辑框架
        # 1. 同样需要对 data (test_data) 进行 reset_index 和预处理
        # 2. 构造 Dataset (flag='test' 或 'pred')
        # 3. 使用 self.model 进行推理
        # 4. 将结果封装回 TimeSeriesDataFrame
        
        # 这里返回一个占位符，避免直接报错，你需要填入实际推理逻辑
        print("Predicting with TimeXer...")
        
        # 构造一个符合格式的空的预测结果
        # 预测结果必须包含 mean 和 quantiles
        prediction_length = self.pred_len
        item_ids = data.item_ids
        
        # 生成未来的时间戳
        future_timestamps = [] # 需要根据 data 的最后时间点往后推
        
        # 此处省略具体实现，通常你需要遍历每个 item_id 进行预测
        # 并返回一个 TimeSeriesDataFrame
        
        # 这是一个假的返回，仅用于跑通流程
        return super().predict(data, known_covariates, **kwargs)