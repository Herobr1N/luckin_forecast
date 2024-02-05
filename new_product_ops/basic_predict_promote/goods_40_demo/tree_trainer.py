# encoding: utf-8
# @created: 2023/11/16
# @author: yuyang.lin

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/dinghuo/alg_dh/')
sys.path.insert(0, '/home/dinghuo/luckystoreordering/')
from __init__ import project_path
from areas.table_info.dh_dw_table_info import dh_dw
from utils_offline.a00_imports import dfu, log20 as log, DayStr, read_api, shuttle, bip3_save_df2, bip3, bip2

from datetime import timedelta
from datetime import datetime
import catboost as cb
import lightgbm as lgb
import xgboost as xgb
from tqdm import tqdm

from feature_engineer import FeatureEngineer


class TreeModelTrainer:
    def __init__(self, features, true_label, train_ratio, model_label):
        self.features = features
        self.true_label = true_label
        self.train_ratio = train_ratio
        self.model_label = model_label

    # 训练集&测试集分割
    def split_train_and_test(self, df):
        train_size = int(len(df) * self.train_ratio)
        train_df = df[:train_size]
        test_df = df[train_size:]
        X_train = train_df[self.features]
        y_train = train_df[self.true_label]
        X_test = test_df[self.features]
        y_test = test_df[self.true_label]
        return train_df, test_df, X_train, y_train, X_test, y_test

    # xgb模型
    def train_xgboost(self, X_train, y_train):
        import xgboost as xgb
        model = xgb.XGBRegressor(objective='reg:gamma',
                                 n_estimators=1500,
                                 learning_rate=0.05,
                                 max_depth=7,
                                 eval_metric=["error", "logloss"])

        eval_set = [(X_train, y_train)]
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        return model

    # caboost模型
    def train_catboost(self, X_train, y_train):
        import catboost as cb
        model = cb.CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6)
        model.fit(X_train, y_train, verbose=False)
        return model

    # lightgbm模型
    def train_lightgbm(self, X_train, y_train):
        import lightgbm as lgb
        model = lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=1500)
        model.fit(X_train, y_train, eval_set=[(X_train, y_train)], eval_metric='l2')
        return model

    def train_model(self, X_train, y_train):
        if self.model_label == 'xgb':
            model = self.train_xgboost(X_train, y_train)
        elif self.model_label == 'catboost':
            model = self.train_catboost(X_train, y_train)
        elif self.model_label == 'lgbm':
            model = self.train_lightgbm(X_train, y_train)
        return model

    # 未来滚动预测单货物- 指定模型，货物，仓库
    def rolling_predict(self, start_date, window, goods_id, wh_dept_id, model, df):
        day_30 = pd.to_datetime(start_date) + pd.DateOffset(days=window - 1)
        feature_generator = FeatureEngineer()

        df_target = df[(df['goods_id'] == goods_id) & (df['wh_dept_id'] == wh_dept_id)]

        start_date = pd.to_datetime(start_date)

        df_target = df_target[df_target['ds'] <= start_date]
        for dt in pd.date_range(start_date, day_30):
            # 生成下一天特征
            next_day_feature = df_target.iloc[-1:].copy()
            # 下一天日期生成
            next_day_feature['ds'] = next_day_feature['ds'] + pd.DateOffset(days=1)
            # 下一天真实值设为na
            next_day_feature['y'] = np.nan
            # 把下一天接进来
            df_target = pd.concat([df_target, next_day_feature])
            # index重置
            df_target = df_target.reset_index(drop=True).copy()
            # 特征重生成
            df_target = feature_generator.feature_generated(df_target)
            # 将生成好特征的最新一行来预测最新的一天
            next_day_feature = df_target.iloc[-1:].copy()
            next_day_y_pred = model.predict(next_day_feature[self.features]).round(1)
            # 预测值替换
            df_target.loc[df_target['ds'] == next_day_feature['ds'].values[0], 'y'] = next_day_y_pred
        return df_target

    # 并行计算
    def parallel_run(self, start_date, end_date, window, item, model):
        df_current = self.rolling_predict(
            start_date=start_date,
            window=window,
            goods_id=item[0][1],
            wh_dept_id=item[0][0],
            model=model,
            df=item[1]
        )
        df_current = df_current[(df_current['ds'] >= start_date) & (df_current['ds'] <= end_date)].reset_index(
            drop=True)
        return df_current

    def whole_country_rolling_predict(self, start_date, window, model, df):
        end_date = DayStr.n_day_delta(start_date, n=+ window)

        content_list = list(df.groupby(["wh_dept_id", "goods_id"]))

        from datetime import datetime, timedelta
        from joblib import Parallel, delayed

        res_list = Parallel(n_jobs=25)(delayed(self.parallel_run)(start_date,
                                                                  end_date,
                                                                  window,
                                                                  item,
                                                                  model) for item in content_list)

        df_daily_pred = pd.concat(res_list, ignore_index=True)
        # 格式修改
        df_daily_pred["dt"] = pd.to_datetime(start_date)
        df_daily_pred["ds"] = df_daily_pred["ds"].dt.strftime("%Y-%m-%d")
        df_daily_pred.rename(columns={"ds": "predict_dt", "y": "demand"}, inplace=True)
        df_daily_pred = df_daily_pred[["predict_dt", "wh_dept_id", "goods_id", "demand", "dt"]]
        df_daily_pred["wh_dept_id"] = df_daily_pred["wh_dept_id"].astype(int)
        df_daily_pred["goods_id"] = df_daily_pred["goods_id"].astype(int)

        return df_daily_pred



