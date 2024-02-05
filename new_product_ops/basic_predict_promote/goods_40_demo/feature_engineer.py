# encoding: utf-8
# @created: 2023/11/16
# @author: yuyang.lin

import sys
import os

sys.path.insert(0, '/home/dinghuo/alg_dh/')
sys.path.insert(0, '/home/dinghuo/luckystoreordering/')
from __init__ import project_path
from areas.table_info.dh_dw_table_info import dh_dw
from utils_offline.a00_imports import dfu, log20 as log, DayStr, read_api, shuttle, \
    bip3_save_df2, bip3, bip2

from datetime import timedelta
from datetime import datetime

import numpy as np
import pandas as pd

class FeatureEngineer:
    def __init__(self):
        pass

    def feature_generated(self, df):
        df = df.reset_index(drop=True)
        # 添加時間列
        df["year"] = df["ds"].dt.year
        df["month"] = df["ds"].dt.month
        df["day"] = df["ds"].dt.day
        df["day_of_week"] = df["ds"].dt.dayofweek
        df["week_of_year"] = df["ds"].dt.isocalendar().week
        # 回归7 days，14 days,21 days, 28 days, 30 days, 60 days, 120 days 的mean, std, min, max
        lag_list = [7, 14, 21, 28, 30, 60, 120]

        df_rolled_1d = df["y"].rolling(window=1, min_periods=0)
        df_mean_1d = df_rolled_1d.mean().shift(1).reset_index()

        df_rolled_7d = df["y"].rolling(window=7, min_periods=0)
        df_mean_7d = df_rolled_7d.mean().shift(1).reset_index()
        df_std_7d = df_rolled_7d.std().shift(1).reset_index()
        df_min_7d = df_rolled_7d.min().shift(1).reset_index()
        df_max_7d = df_rolled_7d.max().shift(1).reset_index()

        df_rolled_14d = df["y"].rolling(window=14, min_periods=0)
        df_mean_14d = df_rolled_14d.mean().shift(1).reset_index()
        df_std_14d = df_rolled_14d.std().shift(1).reset_index()
        df_min_14d = df_rolled_14d.min().shift(1).reset_index()
        df_max_14d = df_rolled_14d.max().shift(1).reset_index()

        df_rolled_21d = df["y"].rolling(window=21, min_periods=0)
        df_mean_21d = df_rolled_21d.mean().shift(1).reset_index()
        df_std_21d = df_rolled_21d.std().shift(1).reset_index()
        df_min_21d = df_rolled_21d.min().shift(1).reset_index()
        df_max_21d = df_rolled_21d.max().shift(1).reset_index()

        df_rolled_28d = df["y"].rolling(window=28, min_periods=0)
        df_mean_28d = df_rolled_28d.mean().shift(1).reset_index()
        df_std_28d = df_rolled_28d.std().shift(1).reset_index()
        df_min_28d = df_rolled_28d.min().shift(1).reset_index()
        df_max_28d = df_rolled_28d.max().shift(1).reset_index()

        df_rolled_30d = df["y"].rolling(window=30, min_periods=0)
        df_mean_30d = df_rolled_30d.mean().shift(1).reset_index()
        df_std_30d = df_rolled_30d.std().shift(1).reset_index()
        df_min_30d = df_rolled_30d.min().shift(1).reset_index()
        df_max_30d = df_rolled_30d.max().shift(1).reset_index()

        df_rolled_60d = df["y"].rolling(window=60, min_periods=0)
        df_mean_60d = df_rolled_60d.mean().shift(1).reset_index()
        df_std_60d = df_rolled_60d.std().shift(1).reset_index()
        df_min_60d = df_rolled_60d.min().shift(1).reset_index()
        df_max_60d = df_rolled_60d.max().shift(1).reset_index()

        df_rolled_120d = df["y"].rolling(window=120, min_periods=0)
        df_mean_120d = df_rolled_120d.mean().shift(1).reset_index()
        df_std_120d = df_rolled_120d.std().shift(1).reset_index()
        df_min_120d = df_rolled_120d.min().shift(1).reset_index()
        df_max_120d = df_rolled_120d.max().shift(1).reset_index()

        df["lag_1_mean"] = df_mean_1d["y"]

        df["lag_7_mean"] = df_mean_7d["y"]
        df["lag_7_std"] = df_std_7d["y"]
        df["lag_7_min"] = df_min_7d["y"]
        df["lag_7_max"] = df_max_7d["y"]

        df["lag_14_mean"] = df_mean_14d["y"]
        df["lag_14_std"] = df_std_14d["y"]
        df["lag_14_min"] = df_min_14d["y"]
        df["lag_14_max"] = df_max_14d["y"]

        df["lag_21_mean"] = df_mean_21d["y"]
        df["lag_21_std"] = df_std_21d["y"]
        df["lag_21_min"] = df_min_21d["y"]
        df["lag_21_max"] = df_max_21d["y"]

        df["lag_28_mean"] = df_mean_28d["y"]
        df["lag_28_std"] = df_std_28d["y"]
        df["lag_28_min"] = df_min_28d["y"]
        df["lag_28_max"] = df_max_28d["y"]

        df["lag_30_mean"] = df_mean_30d["y"]
        df["lag_30_std"] = df_std_30d["y"]
        df["lag_30_min"] = df_min_30d["y"]
        df["lag_30_max"] = df_max_30d["y"]

        df["lag_60_mean"] = df_mean_60d["y"]
        df["lag_60_std"] = df_std_60d["y"]
        df["lag_60_min"] = df_min_60d["y"]
        df["lag_60_max"] = df_max_60d["y"]

        df["lag_120_mean"] = df_mean_120d["y"]
        df["lag_120_std"] = df_std_120d["y"]
        df["lag_120_min"] = df_min_120d["y"]
        df["lag_120_max"] = df_max_120d["y"]
        return df

    # 遍历各仓各货物
    def feature_combined(self, df):
        df_feature = pd.DataFrame()
        for (wh_dept_id, goods_id), df_current in df.groupby(["wh_dept_id", "goods_id"]):
            df_current = df_current.reset_index(drop=True)
            df_feature = pd.concat([df_feature, self.feature_generated(df_current)])
        return df_feature


