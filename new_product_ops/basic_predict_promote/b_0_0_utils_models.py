# encoding: utf-8
# @created: 2023/11/2
# @author: jieqin.lin
# @file: projects/basic_predict_promote/b_0_0_utils_models.py


"""
## 工具： 树模型公用的类
"""

from __init__ import project_path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from areas.table_info.dh_dw_table_info import dh_dw
from utils_offline.a00_imports import dfu, log20 as log, DayStr, read_api, shuttle, bip3_save_df2, bip3
from utils.a71_save_df import bip3_save_model
import concurrent.futures
import xgboost as xgb
import catboost as cb
import lightgbm as lgb

f"Import from {project_path}"


class BaseModelTrainer:
    def __init__(self, true_label, model_label, data_label, goods_label):
        self.true_label = true_label
        self.model_label = model_label
        self.data_label = data_label
        self.goods_label = goods_label

    # 日维度，构造特征
    @staticmethod
    def feature_generated(df):
        # 添加時間列
        df = df.copy().reset_index(drop=True)
        df['ds'] = pd.to_datetime(df['ds'])
        df["year"] = df["ds"].dt.year
        df["month"] = df["ds"].dt.month
        df["day"] = df["ds"].dt.day
        df["day_of_week"] = df["ds"].dt.dayofweek
        df["week_of_year"] = df["ds"].dt.isocalendar().week
        df["week_of_year"] = df["week_of_year"].astype(int)
        # 回归7 days，14 days,21 days, 28 days, 30 days, 60 days, 120 days 的mean, std, min, max
        lag_list = [7, 14, 21, 28, 30, 60, 120]

        df_rolled_1d = df["y"].rolling(window=1, min_periods=0)
        df_mean_1d = df_rolled_1d.mean().shift(1).reset_index()
        df["lag_1_mean"] = df_mean_1d["y"]

        for lag in lag_list:
            window = lag

            df_rolled = df["y"].rolling(window=window, min_periods=0)
            df_mean = df_rolled.mean().shift(1).reset_index()
            df_std = df_rolled.std().shift(1).reset_index()
            df_min = df_rolled.min().shift(1).reset_index()
            df_max = df_rolled.max().shift(1).reset_index()

            df[f"lag_{lag}_mean"] = df_mean["y"]
            df[f"lag_{lag}_std"] = df_std["y"]
            df[f"lag_{lag}_min"] = df_min["y"]
            df[f"lag_{lag}_max"] = df_max["y"]
        return df

    # 周维度，构造特征
    @staticmethod
    def feature_generated_week_of_year(df):
        # 添加時間列
        df = df.copy().reset_index(drop=True)
        df['ds'] = pd.to_datetime(df['ds'])
        df["year"] = df["ds"].dt.year
        df["month"] = df["ds"].dt.month
        df["week_of_year"] = df["ds"].dt.isocalendar().week
        df["week_of_year"] = df["week_of_year"].astype(int)
        # 回归2 weeks，4 weeks,6 weeks, 8 weeks, 12 weeks, 16 weeks, 24 weeks 的mean, std, min, max
        lag_list = [2, 4, 6, 8, 12, 16, 24]

        df_rolled_1d = df["y"].rolling(window=1, min_periods=0)
        df_mean_1d = df_rolled_1d.mean().shift(1).reset_index()
        df["lag_1_mean"] = df_mean_1d["y"]

        for lag in lag_list:
            window = lag

            df_rolled = df["y"].rolling(window=window, min_periods=0)
            df_mean = df_rolled.mean().shift(1).reset_index()
            df_std = df_rolled.std().shift(1).reset_index()
            df_min = df_rolled.min().shift(1).reset_index()
            df_max = df_rolled.max().shift(1).reset_index()

            df[f"lag_{lag}_mean"] = df_mean["y"]
            df[f"lag_{lag}_std"] = df_std["y"]
            df[f"lag_{lag}_min"] = df_min["y"]
            df[f"lag_{lag}_max"] = df_max["y"]

        return df

    # 月维度，构造特征
    @staticmethod
    def feature_generated_month_of_year(df):
        # 添加時間列
        df = df.copy().reset_index(drop=True)
        df['ds'] = pd.to_datetime(df['ds'])
        df["year"] = df["ds"].dt.year
        df["month"] = df["ds"].dt.month

        # 回归2 months，3 months,4 months, 6 months,8 months, 12 months 的mean, std, min, max
        lag_list = [2, 3, 4, 6, 8, 12]

        df_rolled_1d = df["y"].rolling(window=1, min_periods=0)
        df_mean_1d = df_rolled_1d.mean().shift(1).reset_index()
        df["lag_1_mean"] = df_mean_1d["y"]

        for lag in lag_list:
            window = lag

            df_rolled = df["y"].rolling(window=window, min_periods=0)
            df_mean = df_rolled.mean().shift(1).reset_index()
            df_std = df_rolled.std().shift(1).reset_index()
            df_min = df_rolled.min().shift(1).reset_index()
            df_max = df_rolled.max().shift(1).reset_index()

            df[f"lag_{lag}_mean"] = df_mean["y"]
            df[f"lag_{lag}_std"] = df_std["y"]
            df[f"lag_{lag}_min"] = df_min["y"]
            df[f"lag_{lag}_max"] = df_max["y"]

        return df

    # 评估模型 全国，货物，仓库货物维度
    def evaluate_model_wh_goods(self, df, dt):
        # 转成金额
        df = self.change_to_price(df=df, dt=dt)

        y_true = df['y_price'].values
        y_pred = df['pred_price'].values

        # 全国准确率
        df_all_metric = self.evaluate_model(y_true=y_true, y_pred=y_pred)
        df_all_metric['dt'] = dt
        df_all_metric['model'] = self.model_label

        # 货物准确率
        df_goods_metric = df.groupby('goods_id').apply(
            lambda x: self.evaluate_model(x['y_price'].values, x['pred_price'].values)).reset_index()
        df_goods_metric['dt'] = dt
        df_goods_metric['model'] = self.model_label

        # 仓库货物准确率
        df_wh_goods_metric = df.groupby(['wh_dept_id', 'goods_id']).apply(
            lambda x: self.evaluate_model(x['y_price'].values, x['pred_price'].values)).reset_index()
        df_wh_goods_metric['dt'] = dt
        df_wh_goods_metric['model'] = self.model_label

        return df_all_metric, df_goods_metric, df_wh_goods_metric

    # 预测用量转成金额
    def change_to_price(self, df, dt):
        # 转成价格
        df_price = read_api.read_dt_folder(
            bip3("model/basic_predict_promote_online", "wh_goods_price"), dt)

        df_price_mean = df_price.groupby('goods_id')['unit_price'].mean().round(4).rename(
            'unit_price_mean').reset_index()
        df_price_wh = df_price.groupby('wh_dept_id')['unit_price'].mean().round(4).rename(
            'wh_unit_price_mean').reset_index()
        df = df.merge(df_price, 'left').merge(df_price_mean, 'left').merge(df_price_wh, 'left')
        df.update(df['unit_price'].fillna(df['unit_price_mean']).fillna(df['wh_unit_price_mean']))
        df.drop(['unit_price_mean'], axis=1, inplace=True)

        df['y_price'] = (df[self.true_label] * df['unit_price']).round(1)
        df['pred_price'] = (df['pred'] * df['unit_price']).round(1)
        return df

    # 评估
    @staticmethod
    def evaluate_model(y_true, y_pred):
        MSE = mean_squared_error(y_true, y_pred).round(4)
        MAE = mean_absolute_error(y_true, y_pred).round(4)
        MAPE = mean_absolute_percentage_error(y_true, y_pred).round(4)
        ACC = (1 - MAPE).round(4)
        df_metric = pd.DataFrame({'MSE': [MSE], 'MAE': [MAE], 'MAPE': [MAPE], 'ACC': [ACC]})
        return df_metric

    # 保存评估
    def save_metric_df(self, m_all_df, m_goods_df, m_wh_goods_df, time_label, dt):
        # save 时间维度++逐日递归预测准确度
        cols_output = ['dt', 'model', 'MAE', 'MSE', 'MAPE', 'ACC']
        m_all_df = m_all_df[cols_output]
        bip3_save_df2(m_all_df,
                      table_folder_name=f'metric_{time_label}_{self.model_label}_wh_goods_pred_{self.data_label}_{self.goods_label}',
                      bip_folder='model/basic_predict_promote',
                      output_name=f'metric_{time_label}_{self.model_label}_wh_goods_pred_{self.data_label}_{self.goods_label}',
                      folder_dt=dt)

        # -------------------------
        # save 时间维度++货物++逐日递归预测准确度
        cols_output = ['dt', 'model', 'goods_id', 'MAE', 'MSE', 'MAPE', 'ACC']
        m_goods_df = m_goods_df[cols_output]
        bip3_save_df2(m_goods_df,
                      table_folder_name=f'metric_{time_label}_goods_{self.model_label}_wh_goods_pred_{self.data_label}_{self.goods_label}',
                      bip_folder='model/basic_predict_promote',
                      output_name=f'metric_{time_label}_goods_{self.model_label}_wh_goods_pred_{self.data_label}_{self.goods_label}',
                      folder_dt=dt)

        # -------------------------
        # save 时间维度++仓库++货物++逐日递归预测准确度
        cols_output = ['dt', 'model', 'wh_dept_id', 'goods_id', 'MAE', 'MSE', 'MAPE', 'ACC']
        m_wh_goods_df = m_wh_goods_df[cols_output]
        bip3_save_df2(m_wh_goods_df,
                      table_folder_name=f'metric_{time_label}_wh_goods_{self.model_label}_wh_goods_pred_{self.data_label}_{self.goods_label}',
                      bip_folder='model/basic_predict_promote',
                      output_name=f'metric_{time_label}_wh_goods_{self.model_label}_wh_goods_pred_{self.data_label}_{self.goods_label}',
                      folder_dt=dt)

    # 保存预测
    def save_pred_df(self, df_daily_pred, df_month_pred, dt):
        # -------------------------
        # save 数据格式
        df_daily_pred.rename(columns={'ds': 'predict_dt', 'pred': 'predict_demand'}, inplace=True)
        df_daily_pred['predict_dt'] = df_daily_pred['predict_dt'].dt.strftime('%Y-%m-%d')
        df_daily_pred['dt'] = dt
        df_daily_pred['model'] = self.model_label
        cols_int = ['wh_dept_id', 'goods_id']
        df_daily_pred = dfu.df_col_to_numeric(df_daily_pred, cols_int)
        df_daily_pred['predict_demand'] = df_daily_pred['predict_demand'].round(1)
        cols_output = ['predict_dt', 'wh_dept_id', 'goods_id', 'model', 'predict_demand', 'dt']
        df_daily_pred = df_daily_pred[cols_output].sort_values('predict_dt')

        bip3_save_df2(df_daily_pred,
                      table_folder_name=f'{self.model_label}_wh_goods_pred_{self.data_label}_{self.goods_label}',
                      bip_folder='model/basic_predict_promote_online',
                      output_name=f'{self.model_label}_wh_goods_pred_{self.data_label}_{self.goods_label}',
                      folder_dt=dt)

        # -------------------------
        # save 数据格式
        df_month_pred.rename(columns={'ds': 'predict_dt', 'pred': 'predict_demand'}, inplace=True)
        df_month_pred['dt'] = dt
        df_month_pred['model'] = self.model_label
        cols_int = ['wh_dept_id', 'goods_id']
        df_month_pred = dfu.df_col_to_numeric(df_month_pred, cols_int)
        df_month_pred['predict_demand'] = df_month_pred['predict_demand'].round(1)
        cols_output = ['wh_dept_id', 'goods_id', 'model', 'predict_demand', 'dt']
        df_month_pred = df_month_pred[cols_output]
        bip3_save_df2(df_month_pred,
                      table_folder_name=f'{self.model_label}_wh_goods_pred_{self.data_label}_{self.goods_label}_sum',
                      bip_folder='model/basic_predict_promote_online',
                      output_name=f'{self.model_label}_wh_goods_pred_{self.data_label}_{self.goods_label}_sum',
                      folder_dt=dt)

    # 保存训练集的评估
    def save_train_df(self, m_train_df, dt):
        cols_output = ['dt', 'model', 'MAE', 'MSE', 'MAPE', 'ACC']
        m_train_df = m_train_df[cols_output]
        bip3_save_df2(m_train_df,
                      table_folder_name=f'train_metric_{self.model_label}_wh_goods_pred_{self.data_label}_{self.goods_label}',
                      bip_folder='model/basic_predict_promote_online',
                      output_name=f'train_metric_{self.model_label}_wh_goods_pred_{self.data_label}_{self.goods_label}',
                      folder_dt=dt)

    # 保存训练集的评估
    def save_model_df(self, model, dt):
        bip3_save_model(model,
                        table_folder_name=f'model_{self.model_label}_wh_goods_pred_{self.data_label}_{self.goods_label}',
                        bip_folder='model/basic_predict_promote_online',
                        output_name=f'model_{self.model_label}_wh_goods_pred_{self.data_label}_{self.goods_label}',
                        folder_dt=dt)


class TreeModelTrainer(BaseModelTrainer):
    def __init__(self, features, train_ratio, true_label, model_label, data_label, goods_label):
        self.features = features
        self.train_ratio = train_ratio
        self.true_label = true_label
        self.model_label = model_label
        self.data_label = data_label
        self.goods_label = goods_label

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

    # =============================================================================
    # 功能：参数设置
    # 1）xgboost：objective='reg:gamma',n_estimators=1500, learning_rate=0.05, max_depth=7, eval_metric=["error", "logloss"]
    # 2）catboost: iterations=1000, learning_rate=0.1, depth=6
    # 3）lightgbm: objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=1500, eval_metric='l2'

    # 增量训练说明
    # 1）增量训练的过程中，模型会保留之前已经学到的知识，并尝试进一步优化模型的性能。
    # 2）通过增加n_estimators的值，可以增加训练的轮数（树的数量），从而进一步提升模型的性能
    # =============================================================================

    # xgb模型
    @staticmethod
    def train_xgboost(X_train, y_train, model=None):
        if model is None:
            model = xgb.XGBRegressor(objective='reg:gamma',
                                     n_estimators=1500,
                                     learning_rate=0.05,
                                     max_depth=7,
                                     eval_metric=["error", "logloss"])
        else:
            log.debug("xgb增量训练")
            model.set_params(n_estimators=model.get_params()['n_estimators'] + 1500)

        eval_set = [(X_train, y_train)]
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        return model

    # caboost模型
    @staticmethod
    def train_catboost(X_train, y_train, model=None):
        if model is None:
            model = cb.CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, verbose=False)
        else:
            log.debug("catboost增量训练")
            model.set_params(iterations=model.get_params()['iterations'] + 1000)
        model.fit(X_train, y_train, verbose=False)
        return model

    # lightgbm模型
    @staticmethod
    def train_lightgbm(X_train, y_train, model=None):
        if model is None:
            model = lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=1500,
                                      verbose=-1, verbose_eval=False)
        else:
            log.debug("lightgbm增量训练")
            model.set_params(n_estimators=model.get_params()['n_estimators'] + 1500)

        model.fit(X_train, y_train, eval_set=[(X_train, y_train)], eval_metric='l2')
        return model

    # 选择不同的树模型
    def select_tree_model(self, X_train, y_train, model=None):
        if self.model_label == 'xgb':
            model = self.train_xgboost(X_train, y_train, model=model)
        elif self.model_label == 'catboost':
            model = self.train_catboost(X_train, y_train, model=model)
        elif self.model_label == 'lgbm':
            model = self.train_lightgbm(X_train, y_train, model=model)
        return model

    # =============================================================================
    # 功能：逐日递归预测
    # 1）循环30次，预测未来30天
    # 2）构造新特征：7，14，21，28，30，60，120天的统计值（均值，标准差，最大值，最小值等）
    # =============================================================================

    # 未来滚动预测单货物- 指定模型，货物，仓库
    def rolling_predict(self, start_date, window, goods_id, wh_dept_id, model, df, label):
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(start_date) + pd.DateOffset(days=window - 1)
        df_target = (df.query(f"goods_id == {goods_id} and wh_dept_id == {wh_dept_id}")
                     .reset_index(drop=True).copy())
        df_target.query(f"ds <= '{start_date}'", inplace=True)
        log.debug(f"Processing:  wh_dept_id：{wh_dept_id}, goods_id：{goods_id}, model: {label}")

        if len(df_target) > 0:
            for dt in pd.date_range(start_date, end_date):
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
                df_target = super().feature_generated(df_target)
                # 将生成好特征的最新一行来预测最新的一天
                next_day_feature = df_target.iloc[-1:].copy()
                next_day_y_pred = model.predict(next_day_feature[self.features]).round(1)
                if len(next_day_feature['ds'].values) > 0:
                    df_target.loc[df_target['ds'] == next_day_feature['ds'].values[0], 'y'] = next_day_y_pred

        return df_target

    # 未来滚动预测单货物- 指定模型，货物，仓库  parallel
    def parallel_run(self, start_date, end_date, window, item, model, label):
        df_current = self.rolling_predict(
            start_date=start_date,
            window=window,
            goods_id=item[0][1],
            wh_dept_id=item[0][0],
            model=model,
            df=item[1],
            label=label)
        df_current = df_current.query(f"'{start_date}' <= ds <= '{end_date}'").reset_index(drop=True)
        return df_current

    # 各仓-各货物预测 parallel
    def whole_country_rolling_predict(self, start_date, window, model, df, label):
        end_date = DayStr.n_day_delta(start_date, n=+ window)

        content_list = list(df.groupby(["wh_dept_id", "goods_id"]))

        res_list = Parallel(n_jobs=15)(delayed(self.parallel_run)(start_date,
                                                                  end_date,
                                                                  window,
                                                                  item,
                                                                  model,
                                                                  label) for item in content_list)

        df_daily_pred = pd.concat(res_list, ignore_index=True)
        return df_daily_pred

    # 各仓-各货物预测  for循环
    def whole_country_rolling_predict_for(self, start_date, window, model, df, label):
        end_date = DayStr.n_day_delta(start_date, n=+ window)
        df_daily_pred = pd.DataFrame()
        for (wh_dept_id, goods_id), df_wh_goods in df.groupby(["wh_dept_id", "goods_id"]):
            df_current = self.rolling_predict(
                start_date=start_date,
                window=window,
                goods_id=goods_id,
                wh_dept_id=wh_dept_id,
                model=model,
                df=df_wh_goods,
                label=label)
            df_current = df_current.query(f"'{start_date}' <= ds <= '{end_date}'").reset_index(drop=True)
            df_daily_pred = pd.concat([df_daily_pred, df_current])
        return df_daily_pred

    # 各仓-各货物预测  递归
    def whole_country_rolling_predict_recursive(self, start_date, window, model, df, label, df_daily_pred):
        end_date = DayStr.n_day_delta(start_date, n=+window)
        group_data = list(df.groupby(["wh_dept_id", "goods_id"]))

        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futures = []

            for (wh_dept_id, goods_id), df_wh_goods in group_data:
                future = executor.submit(self.rolling_predict_recursive, start_date, window, goods_id, wh_dept_id,
                                         model,
                                         df_wh_goods, label)
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                df_current = future.result()
                df_current = df_current.query(f"'{start_date}' <= ds <= '{end_date}'").reset_index(drop=True)
                df_daily_pred = pd.concat([df_daily_pred, df_current])

        return df_daily_pred

    # 未来滚动预测单货物- 指定模型，货物，仓库  递归
    def rolling_predict_recursive(self, start_date, window, goods_id, wh_dept_id, model, df, label):
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(start_date) + pd.DateOffset(days=window - 1)
        df_target = df.query(f"goods_id == {goods_id} and wh_dept_id == {wh_dept_id}").reset_index(drop=True).copy()
        df_target.query(f"ds <= '{start_date}'", inplace=True)
        log.debug(f"Processing: wh_dept_id：{wh_dept_id}, goods_id：{goods_id}, model: {label}")

        return self.greedy_rolling_predict_recursive(start_date, end_date, df_target, model)

    # 未来滚动预测单货物- 指定模型，货物，仓库 贪心
    def greedy_rolling_predict_recursive(self, start_date, end_date, df_target, model):
        df_target = df_target.copy()

        while start_date <= end_date:
            while start_date <= end_date:
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
                df_target = super().feature_generated(df_target)
                # 将生成好特征的最新一行来预测最新的一天
                next_day_feature = df_target.iloc[-1:].copy()
                next_day_y_pred = model.predict(next_day_feature[self.features]).round(1)
                # 预测值替换
                if len(next_day_feature['ds'].values) > 0:
                    df_target.loc[df_target['ds'] == next_day_feature['ds'].values[0], 'y'] = next_day_y_pred
                start_date += pd.DateOffset(days=1)

        return df_target


class FeatureEngineer(BaseModelTrainer):

    def __init__(self, data_label):
        self.data_label = data_label
        # 遍历各仓各货物

    def feature_generated_wh_goods(self, df, time_label='daily'):
        df_feature = pd.DataFrame()
        for (wh_dept_id, goods_id), df_current in df.groupby(["wh_dept_id", "goods_id"]):
            df_current = df_current.reset_index(drop=True)
            if time_label == 'daily':
                df_feature = pd.concat([df_feature, super().feature_generated(df_current)])
            if time_label == 'week':
                df_feature = pd.concat([df_feature, super().feature_generated_week_of_year(df_current)])
            if time_label == 'month':
                df_feature = pd.concat([df_feature, super().feature_generated_month_of_year(df_current)])
        return df_feature

    # 获取特征数据
    def get_feature_wh_goods(self, start_date, end_date, sel_goods_ls):
        if self.data_label == 'normal':
            df = read_api.read_dt_folder(
                bip3("model/basic_predict_promote_online", "feature_engineering_wh_goods_normal")
                , start_date, end_date)

        if self.data_label == 'all':
            df = read_api.read_dt_folder(
                bip3("model/basic_predict_promote_online", "feature_engineering_wh_goods_all")
                , start_date, end_date)

        if self.data_label == 'new_shop_normal':
            df = read_api.read_dt_folder(
                bip3("model/basic_predict_promote_online", "feature_engineering_wh_goods_new_shop_normal")
                , start_date, end_date)

        df.query(f"goods_id == {sel_goods_ls}", inplace=True)
        df.sort_values('ds', inplace=True)
        df['y'] = df['y'].round(1)
        return df

    # 获取自然周汇总特征数据
    def get_feature_week_of_year_wh_goods(self, start_date, end_date, sel_goods_ls):
        if self.data_label == 'normal':
            df = read_api.read_dt_folder(
                bip3("model/basic_predict_promote_online", "feature_engineering_wh_goods_week_of_year_normal")
                , start_date, end_date)

        if self.data_label == 'all':
            df = read_api.read_dt_folder(
                bip3("model/basic_predict_promote_online", "feature_engineering_wh_goods_week_of_year_all")
                , start_date, end_date)

        if self.data_label == 'new_shop_normal':
            df = read_api.read_dt_folder(
                bip3("model/basic_predict_promote_online", "feature_engineering_wh_goods_week_of_year_new_shop_normal")
                , start_date, end_date)

        df.query(f"goods_id == {sel_goods_ls}", inplace=True)
        df.sort_values('ds', inplace=True)
        df['y'] = df['y'].round(1)
        return df

    # 获取自然月汇总特征数据
    def get_feature_month_of_year_wh_goods(self, start_date, end_date, sel_goods_ls):
        if self.data_label == 'normal':
            df = read_api.read_dt_folder(
                bip3("model/basic_predict_promote_online", "feature_engineering_wh_goods_month_of_year_normal")
                , start_date, end_date)

        if self.data_label == 'all':
            df = read_api.read_dt_folder(
                bip3("model/basic_predict_promote_online", "feature_engineering_wh_goods_month_of_year_all")
                , start_date, end_date)

        if self.data_label == 'new_shop_normal':
            df = read_api.read_dt_folder(
                bip3("model/basic_predict_promote_online", "feature_engineering_wh_goods_month_of_year_new_shop_normal")
                , start_date, end_date)

        df.query(f"goods_id == {sel_goods_ls}", inplace=True)
        df.sort_values('ds', inplace=True)
        df['y'] = df['y'].round(1)
        return df

    # 获取历史消耗数据
    def get_his_stock_wh_goods(self, start_date, end_date, sel_goods_ls):
        if self.data_label == 'all':
            df = read_api.read_dt_folder(
                bip3("model/basic_predict_promote_online", "stock_wh_goods_theory_sale_cnt")
                , start_date, end_date)

        if self.data_label == 'normal':
            df = read_api.read_dt_folder(
                bip3("model/basic_predict_promote_online", "stock_wh_goods_type_flg_theory_sale_cnt")
                , start_date, end_date)

            df.query("type_flg =='norm' and shop_type_flg =='norm_shop'", inplace=True)
            df.drop(['type_flg', 'shop_type_flg'], axis=1, inplace=True)

        if self.data_label == 'new_shop_normal':
            df = read_api.read_dt_folder(
                bip3("model/basic_predict_promote_online", "stock_wh_goods_type_flg_theory_sale_cnt")
                , start_date, end_date)

            df.query("type_flg =='norm' and shop_type_flg =='new_shop'", inplace=True)
            df.drop(['type_flg', 'shop_type_flg'], axis=1, inplace=True)

        df["wh_dept_id"] = df["wh_dept_id"].astype(int)
        df["goods_id"] = df["goods_id"].astype(int)
        df.query(f"goods_id  in {sel_goods_ls}", inplace=True)
        df["theory_sale_cnt"] = np.abs(df["theory_sale_cnt"])
        df["dt"] = pd.to_datetime(df["dt"])
        df.rename(columns={"theory_sale_cnt": "y", "dt": "ds"}, inplace=True)

        cols_output = ['ds', 'wh_dept_id', 'goods_id', 'y']
        df_stock_sell = df[cols_output]

        return df_stock_sell

    # 获取按（自然周）汇总的历史消耗数据
    def get_his_week_of_year_stock_wh_goods(self, start_date, end_date, sel_goods_ls):
        """
        start_date, end_date 为自然周的第一天
        """
        if self.data_label == 'all':
            df = read_api.read_dt_folder(
                bip3("model/basic_predict_promote_online", "week_of_year_stock_wh_goods_theory_sale_cnt")
                , start_date, end_date)

        if self.data_label == 'normal':
            df = read_api.read_dt_folder(
                bip3("model/basic_predict_promote_online", "week_of_year_stock_wh_goods_type_flg_theory_sale_cnt")
                , start_date, end_date)

            df.query("type_flg =='norm' and shop_type_flg =='norm_shop'", inplace=True)
            df.drop(['type_flg', 'shop_type_flg'], axis=1, inplace=True)

        if self.data_label == 'new_shop_normal':
            df = read_api.read_dt_folder(
                bip3("model/basic_predict_promote_online", "week_of_year_stock_wh_goods_type_flg_theory_sale_cnt")
                , start_date, end_date)

            df.query("type_flg =='norm' and shop_type_flg =='new_shop'", inplace=True)
            df.drop(['type_flg', 'shop_type_flg'], axis=1, inplace=True)

        df["wh_dept_id"] = df["wh_dept_id"].astype(int)
        df["goods_id"] = df["goods_id"].astype(int)
        df.query(f"goods_id  in {sel_goods_ls}", inplace=True)
        df["theory_sale_cnt"] = np.abs(df["theory_sale_cnt"])
        df["first_day_of_week"] = pd.to_datetime(df["first_day_of_week"])
        df.rename(columns={"theory_sale_cnt": "y", "first_day_of_week": "ds"}, inplace=True)

        cols_output = ['ds', 'wh_dept_id', 'goods_id', 'y']
        df_stock_sell = df[cols_output]

        return df_stock_sell

    # 获取按（自然月）汇总的历史消耗数据
    def get_his_month_of_year_stock_wh_goods(self, start_date, end_date, sel_goods_ls):
        """
        start_date, end_date 为自然月的第一天
        """
        if self.data_label == 'all':
            df = read_api.read_dt_folder(
                bip3("model/basic_predict_promote_online", "month_of_year_stock_wh_goods_theory_sale_cnt")
                , start_date, end_date)

        if self.data_label == 'normal':
            df = read_api.read_dt_folder(
                bip3("model/basic_predict_promote_online", "month_of_year_stock_wh_goods_type_flg_theory_sale_cnt")
                , start_date, end_date)

            df.query("type_flg =='norm' and shop_type_flg =='norm_shop'", inplace=True)
            df.drop(['type_flg', 'shop_type_flg'], axis=1, inplace=True)

        if self.data_label == 'new_shop_normal':
            df = read_api.read_dt_folder(
                bip3("model/basic_predict_promote_online", "month_of_year_stock_wh_goods_type_flg_theory_sale_cnt")
                , start_date, end_date)

            df.query("type_flg =='norm' and shop_type_flg =='new_shop'", inplace=True)
            df.drop(['type_flg', 'shop_type_flg'], axis=1, inplace=True)

        df["wh_dept_id"] = df["wh_dept_id"].astype(int)
        df["goods_id"] = df["goods_id"].astype(int)
        df.query(f"goods_id  in {sel_goods_ls}", inplace=True)
        df["theory_sale_cnt"] = np.abs(df["theory_sale_cnt"])
        df["first_day_of_month"] = pd.to_datetime(df["first_day_of_month"])
        df.rename(columns={"theory_sale_cnt": "y", "first_day_of_month": "ds"}, inplace=True)

        cols_output = ['ds', 'wh_dept_id', 'goods_id', 'y']
        df_stock_sell = df[cols_output]

        return df_stock_sell


class SelectGoodsList:
    def __init__(self):
        pass

    # 获取kmean是所有 21个货物ID list
    @staticmethod
    def get_group_a_goods_id():
        dtw_goods = [49, 52, 80, 408, 411, 414, 415, 456,
                     756, 4173, 4488, 6064, 7282, 20610, 20716, 20721,
                     20818, 25316, 25869]
        return dtw_goods

    # 获取历史预测准确率低的货物ID 10个货物ID list
    @staticmethod
    def get_group_b_goods_id():
        # 准确率低
        # ['16oz 冰杯拱盖', '淡奶油', '绿茶调味茶固体饮料', '纯牛奶', '抹茶拿铁（固体饮料）']
        bad_goods = [412, 354, 397, 755, 27926, 660, 27942]
        return bad_goods

    # 获取3个货物ID list
    @staticmethod
    def get_group_c_goods_id():
        # '冰凉感厚椰饮品', '冷萃厚牛乳', '北海道丝绒风味厚乳'
        sel_goods_ls = [19952, 23628, 27954]
        return sel_goods_ls

    # 获取other 货物ID list
    @staticmethod
    def get_group_d_goods_id():
        # sql_material_pool = f"""
        # SELECT
        # goods_id,long_period_cal_type
        # FROM
        # lucky_cooperation.t_material_pool
        # WHERE del=0
        # """
        # df_material_pool = shuttle.query_dataq(sql_material_pool)
        # goods_long_purchase = df_material_pool['goods_id'].drop_duplicates().tolist()
        not_in_goods = ['冷冻调制血橙', '柚子复合果汁饮料浓浆']
        # # 非配方包材
        # PACKAGE_MATERIALS = [72, 260, 290, 442, 450, 19343, 22837, 25274, 869]
        #
        # goods_long_purchase = [80, 408, 411, 412, 414, 415, 456, 4173, 20610,
        #                         20617, 20721, 20968, 24959, 25316, 52, 20725, 354,
        #                         49, 4488, 7282,  6064, 397, 19952, 23628, 25869,
        #                         20818, 27954, 660, 755, 756, 20716, 27926,
        #                         27942]
        # # 准确率低
        # bad_goods = [412, 354, 397, 755, 27926, 660, 27942]
        # demo_goods = [19952, 23628, 27954]
        #
        # dtw_goods = [49, 52, 80, 408, 411, 414, 415, 456,
        #              756, 4173, 4488, 6064, 7282, 20610, 20716, 20721,
        #              20818, 25316, 25869]
        #
        # df_goods_info = dh_dw.dim_stock.goods_info()
        # sel_goods_ls = df_goods_info.query(
        #     f"goods_id in {goods_long_purchase} "
        #     f"and goods_name !={not_in_goods}  "
        #     f"and goods_id!={PACKAGE_MATERIALS} "
        #     f"and goods_id!={bad_goods} "
        #     f"and goods_id!={demo_goods} "
        #     f"and goods_id!={dtw_goods} ")[
        #     'goods_id'].unique().tolist()
        sel_goods_ls = [20813, 732, 256,25456]
        return sel_goods_ls

    # 获取kmeans 聚类label_1 11个货物ID list
    @staticmethod
    def get_group_e_goods_id():
        # df_label = read_api.read_dt_folder(
        #     bip3("model/basic_predict_promote_online", f"predict_40_goods_all_label"))
        # label_1 = df_label.query("label==1")['goods_id'].unique().tolist()
        label_1 = [80, 411, 6064, 7282, 20716, 25316]
        return label_1

    # 获取kmeans 聚类label_0 10个货物ID list
    @staticmethod
    def get_group_f_goods_id():
        # df_label = read_api.read_dt_folder(
        #     bip3("model/basic_predict_promote_online", f"predict_40_goods_all_label"))
        # label_0 = df_label.query("label==0")['goods_id'].unique().tolist()
        label_0 = [49, 52, 408, 414, 415, 456, 756, 4173, 4488, 20610, 20721, 20818, 25869]
        return label_0

    # 获取SOE 货物ID list
    @staticmethod
    def get_group_g_goods_id():
        # 12oz SOE冰杯 单支吸管粗（PLA） D直饮杯盖  水洗耶加雪菲咖啡豆
        sel_goods_ls = [20617, 20968, 24959, 20725]
        return sel_goods_ls

    # 获取长周期货物ID list
    @staticmethod
    def get_long_period_goods_id():
        # sql_material_pool = f"""
        # SELECT
        # goods_id,long_period_cal_type
        # FROM
        # lucky_cooperation.t_material_pool
        # WHERE del=0
        # """
        # df_material_pool = shuttle.query_dataq(sql_material_pool)
        # goods_long_purchase = df_material_pool['goods_id'].drop_duplicates().tolist()
        # not_in_goods = ['冷冻调制血橙', '柚子复合果汁饮料浓浆']
        # # 非配方包材
        # PACKAGE_MATERIALS = [72, 260, 290, 442, 450, 19343, 22837, 25274, 869]
        # df_goods_info = dh_dw.dim_stock.goods_info()
        # sel_goods_ls = df_goods_info.query(
        #     f"goods_id in {goods_long_purchase} and goods_name !={not_in_goods}  and goods_id!={PACKAGE_MATERIALS}")[
        #     'goods_id'].unique().tolist()

        sel_goods_ls = [80, 408, 411, 412, 414, 415, 456, 4173, 20610,
                        20617, 20721, 20968, 24959, 25316, 52, 20725, 354,
                        49, 4488, 7282,  6064, 397, 19952, 23628, 25869,
                        20818, 27954, 660, 755, 756, 20716, 27926,
                        27942]
        return sel_goods_ls

    # 将货物分类
    def get_group_all_goods_ls(self):
        # 指标差的 7个货物
        sel_b_goods_ls = self.get_group_b_goods_id()
        # 3个货物
        sel_c_goods_ls = self.get_group_c_goods_id()
        # # 剩余货物 3个
        # sel_d_goods_ls = self.get_group_d_goods_id()
        # 聚类10个
        sel_e_goods_ls = self.get_group_e_goods_id()
        # 聚类10个
        sel_f_goods_ls = self.get_group_f_goods_id()
        # 4个
        sel_g_goods_ls = self.get_group_g_goods_id()
        # 合并
        sel_all_goods_ls = {'group_b': sel_b_goods_ls,
                            'group_c': sel_c_goods_ls,
                            'group_e': sel_e_goods_ls,
                            'group_f': sel_f_goods_ls,
                            'group_g': sel_g_goods_ls
                            }

        return sel_all_goods_ls


class ModelManager(BaseModelTrainer):
    def __init__(self, model_label, data_label, goods_label):
        self.model_label = model_label
        self.data_label = data_label
        self.goods_label = goods_label

    # 获取上个月准确率
    def get_last_acc(self, window):
        cols_output = ['wh_dept_id', 'goods_id', 'model', f'last_{window}_acc', 'dt']
        ld = read_api.read_dt_folder(
            bip3("model/basic_predict_promote_online",
                 f"{self.model_label}_last_{window}_acc_{self.data_label}_{self.goods_label}"))[
            cols_output]
        df = ld.sort_values('dt').groupby(['wh_dept_id', 'goods_id']).tail(1)

        return df

    # 获取预测
    def get_pred(self, dt):
        df_pred = read_api.read_dt_folder(
            bip3("model/basic_predict_promote_online",
                 f"{self.model_label}_wh_goods_pred_{self.data_label}_{self.goods_label}"), dt)

        return df_pred
