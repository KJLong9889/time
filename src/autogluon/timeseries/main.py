import pandas as pd
import numpy as np
import time
import os
import shutil
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
logging.getLogger('autogluon').setLevel(logging.ERROR)   # 只打 ERROR
import os



os.environ["TABPFN_MODEL_CACHE_DIR"] = "checkpoints"
# ====================== 路径配置 ======================
INPUT_HISTORY_DIR = "data1"
OUTPUT_DIR = "results_v2/"
MODEL_PATH_TEMPLATE = "AutogluonModels/AutogluonModels_{filename}"
METRICS_FILE = "evaluation_metrics1.csv"  # 新增：评估指标文件

FORCE_DELETE_MODEL = True
TEST_DAYS = 7  # 测试集长度 = 预测长度
PREDICTION_LENGTH = TEST_DAYS

# ====================== 核心字段 ======================
ID_COLUMN = "item_id"
TIMESTAMP_COLUMN = "date"
TARGET_COLUMN = "value"

# ====================== 已知未来协变量 ======================
KNOWN_COVARIATES = [
    'precip', 'windmax', 'windmaxdir', 'rhhi',
    'temphi', 'templo', 'avgtemp',
    'day_of_week', 'is_weekend',
    'week_sin', 'week_cos',
    'month_sin', 'month_cos',
    'days_since_start','month'
]


# ====================== 时间特征函数 ======================
def add_time_features(df, timestamp_col='date'):
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col)

    df['month'] = df[timestamp_col].dt.month
    df['day_of_week'] = df[timestamp_col].dt.weekday
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    base_date = df[timestamp_col].min()
    df['days_since_start'] = (df[timestamp_col] - base_date).dt.days

    df['week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)


    return df


# ====================== 单文件处理 ======================
def process_file(history_file):
    base_name = os.path.splitext(history_file)[0]

    try:
        # ---------- 读取历史数据 ----------
        df = pd.read_csv(os.path.join(INPUT_HISTORY_DIR, history_file))
        df[ID_COLUMN] = base_name
        df = add_time_features(df, TIMESTAMP_COLUMN)

        # ---------- 字段检查 ----------
        required_cols = [ID_COLUMN, TIMESTAMP_COLUMN, TARGET_COLUMN] + KNOWN_COVARIATES
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return f"❌ {history_file} 缺少列: {missing}", None

        df = df.sort_values(TIMESTAMP_COLUMN)

        if len(df) <= TEST_DAYS:
            return f"⚠️ {history_file} 数据量不足", None

        # ======================================================
        # ✅ 核心改动：时间切分
        # 训练集：过去
        # 测试集：真实未来
        # ======================================================
        df_train = df.iloc[:-TEST_DAYS].copy()
        df_test = df.iloc[-TEST_DAYS:].copy()

        print(f"📊 {history_file} | Train: {len(df_train)} | Test(Future): {len(df_test)}")

        # ---------- 清理模型路径 ----------
        model_path = MODEL_PATH_TEMPLATE.format(filename=base_name)
        if FORCE_DELETE_MODEL and os.path.exists(model_path):
            shutil.rmtree(model_path)

        # ---------- 构造训练 TSDF ----------
        train_tsdf = TimeSeriesDataFrame.from_data_frame(
            df_train,
            id_column=ID_COLUMN,
            timestamp_column=TIMESTAMP_COLUMN
        )

        # ---------- 构造预测器 ----------
        predictor = TimeSeriesPredictor(
            prediction_length=PREDICTION_LENGTH,
            freq='D',
            target=TARGET_COLUMN,
            known_covariates_names=KNOWN_COVARIATES,
            eval_metric="WQL",
            eval_metric_seasonal_period=1,
            quantile_levels=[0.3, 0.5, 0.7],
            path=model_path
        )

        # ---------- 模型训练 ----------
        predictor.fit(
            train_data=train_tsdf,
            time_limit=1200,
            enable_ensemble=True,
            hyperparameters={
                # 1. 您原有的基于表格的模型 (将时序转化为回归问题)
                "DirectTabular": {
                    "tabular_hyperparameters": {
                        "TABPFNV2": {}  # 使用 TabPFN 作为底层回归器
                    }
                },

                # 2. 深度学习模型 (通常效果最好，适合捕捉复杂模式)
                "DeepAR": {},  # 经典的概率预测模型，稳健性好
                "PatchTST": {},  # 基于 Transformer 的最新 SOTA 模型，擅长长序列
                "TiDE": {},  # 谷歌推出的基于 MLP 的模型，速度快且效果好
            }
        )

        # ======================================================
        # ✅ 核心改动：未来协变量 = df_test
        # ======================================================
        future_cov = df_test[
            [ID_COLUMN, TIMESTAMP_COLUMN] + KNOWN_COVARIATES
            ].copy()

        future_tsdf = TimeSeriesDataFrame.from_data_frame(
            future_cov,
            id_column=ID_COLUMN,
            timestamp_column=TIMESTAMP_COLUMN
        )

        # ---------- 预测 ----------
        predictions = predictor.predict(
            data=train_tsdf,
            known_covariates=future_tsdf
        )

        # ---------- 结果处理 ----------
        pred_df = predictions.reset_index()

        if "timestamp" in pred_df.columns and TIMESTAMP_COLUMN != "timestamp":
            pred_df.rename(columns={"timestamp": TIMESTAMP_COLUMN}, inplace=True)

            # 同样，如果 ID 列名不一致，也可以安全地改回来（虽然 item_id 不影响 merge）
        if "item_id" in pred_df.columns and ID_COLUMN != "item_id":
            pred_df.rename(columns={"item_id": ID_COLUMN}, inplace=True)

        pred_df[TIMESTAMP_COLUMN] = pd.to_datetime(pred_df[TIMESTAMP_COLUMN])
        df_test[TIMESTAMP_COLUMN] = pd.to_datetime(df_test[TIMESTAMP_COLUMN])
        # 合并真实值
        pred_df = pred_df.merge(
            df_test[[TIMESTAMP_COLUMN, TARGET_COLUMN]],
            on=TIMESTAMP_COLUMN,
            how='left'
        )

        pred_df.rename(columns={TARGET_COLUMN: "actual"}, inplace=True)
        pred_df['error'] = pred_df['0.5'] - pred_df['actual']
        pred_df['abs_error'] = pred_df['error'].abs()

        # ---------- 评估 ----------
        mae = pred_df['abs_error'].mean()
        rmse = np.sqrt((pred_df['error'] ** 2).mean())
        mape = (pred_df['abs_error'] / pred_df['actual']).mean() * 100

        # ---------- 保存 ----------
        output_file = os.path.join(OUTPUT_DIR, f"{base_name}_forecast.csv")
        pred_df.to_csv(output_file, index=False, encoding='utf-8-sig')

        # 返回评估指标
        metrics = {
            'filename': history_file,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }

        return (
            f"✅ {history_file} 完成 | "
            f"MAE={mae:.2f} RMSE={rmse:.2f} MAPE={mape:.2f}%"
        ), metrics

    except Exception as e:
        import traceback
        error_msg = f"❌ {history_file} 失败\n{traceback.format_exc()[:300]}"
        return error_msg, None


# ====================== 并行处理 ======================
def batch_process_parallel(max_workers=4):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = [f for f in os.listdir(INPUT_HISTORY_DIR) if f.endswith(".csv")]

    all_metrics = []  # 存储所有文件的评估指标

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, f): f for f in files}
        for future in as_completed(futures):
            result, metrics = future.result()
            print(result)

            if metrics is not None:
                all_metrics.append(metrics)

    # 保存所有评估指标到CSV文件
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df = metrics_df[['filename', 'mae', 'rmse', 'mape']]  # 确保列顺序
        metrics_path = os.path.join(OUTPUT_DIR, METRICS_FILE)
        metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
        print(f"\n📊 评估指标已保存到: {metrics_path}")
        print(f"共处理 {len(metrics_df)} 个文件")
        print("\n评估指标摘要:")
        print(metrics_df)
    else:
        print("⚠️ 未生成任何评估指标")


if __name__ == "__main__":
    batch_process_parallel(max_workers=4)
    print("📢 全部处理完成")