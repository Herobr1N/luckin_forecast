# encoding: utf-8
# @created: 2023/11/14
# @author: yuyang.lin
# @file: projects/basic_predict_promote/goods_40.py
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
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm


class GoodsData:
    """
    商品数据类

    使用示例:
    data = GoodsData(data_label, dt)
    sel_goods_ls = data.goods_consume_data()
    df_stock_sell = data.get_his_true_value()

    """
    def __init__(self, data_label, dt):
        """
        初始化函数

        参数:
        - data_label (str): 数据标签，可以是"normal"或"all"
        - dt (str): 日期，格式为"YYYY-MM-DD"
        """
        self.data_label = data_label
        self.dt = dt

    def goods_consume_data(self):
        """
        获取商品名称数据

        返回:
        - sel_goods_ls (list): 商品ID列表
        """
        df_goods_info = dh_dw.dim_stock.goods_info()
        sql_material_pool = """
        SELECT 
        goods_id,long_period_cal_type
        FROM  
        lucky_cooperation.t_material_pool
        WHERE del=0
        """
        df_material_pool = shuttle.query_dataq(sql_material_pool)
        goods_long_purchase = df_material_pool['goods_id'].drop_duplicates().tolist()
        not_in_goods = ['冷冻调制血橙', '柚子复合果汁饮料浓浆']
        # 非配方包材
        PACKAGE_MATERIALS = [72, 260, 290, 442, 450, 19343, 22837, 25274, 869]
        sel_goods_ls = df_goods_info.query(
            f"goods_id in {goods_long_purchase} and goods_name !={not_in_goods}  and goods_id!={PACKAGE_MATERIALS}")['goods_id'].unique().tolist()
        return sel_goods_ls

    def get_his_true_value(self, sel_goods_ls):
        """
        获取历史真实值数据

        返回:
        - df_stock_sell (DataFrame): 包含历史真实值的数据框
        """
        # 过去120天的仓库货物消耗数据

        day_minus120 = datetime.strptime(self.dt, "%Y-%m-%d") - pd.DateOffset(days=120)
        day_minus120 = day_minus120.strftime("%Y-%m-%d")

        if self.data_label == 'normal':
            df_stock_sell = read_api.read_dt_folder(
                bip3("model/basic_predict_promote", "stock_wh_goods_type_flg_theory_sale_cnt"),
                day_minus120, self.dt)
            df_stock_sell.query("type_flg =='norm' and shop_type_flg =='norm_shop'", inplace=True)
            df_stock_sell.drop(['type_flg', 'shop_type_flg'], axis=1, inplace=True)

        if self.data_label == 'all':
            df_stock_sell = read_api.read_dt_folder(
                bip3("model/basic_predict_promote", "stock_wh_goods_theory_sale_cnt"),
                day_minus120, self.dt)
        # 取绝对值

        df_stock_sell["theory_sale_cnt"] = np.abs(df_stock_sell["theory_sale_cnt"])
        df_stock_sell.query(f"goods_id in {sel_goods_ls}", inplace=True)
        df_stock_sell["dt"] = pd.to_datetime(df_stock_sell["dt"])
        df_stock_sell.rename(columns={"theory_sale_cnt": "y", "dt": "ds"}, inplace=True)

        return df_stock_sell


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


class TreeModelTrainer:
    def __init__(self, features, true_label, train_ratio, model_label):
        self.features = features
        self.true_label = true_label
        self.train_ratio = train_ratio
        self.model_label = model_label

    # 训练集&测试集分割
    def split_train_and_test(self, df):
        log.debug("split train and test set")
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
        log.debug("xgb model train")
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
        log.debug("catboost model train")
        import catboost as cb
        model = cb.CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6)
        model.fit(X_train, y_train, verbose=False)
        return model

    # lightgbm模型
    def train_lightgbm(self, X_train, y_train):
        log.debug("lightgbm model train")
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
    def model_prediction(self, start_date, window, goods_id, wh_dept_id, model, df):
        day_30 = pd.to_datetime(start_date) + pd.DateOffset(days = window )
        feature_generator = FeatureEngineer()

        df_target = df.query(f"goods_id == {goods_id} and wh_dept_id == {wh_dept_id}")
        start_date = pd.to_datetime(start_date)

        df_target = df_target.query(f"ds <= '{start_date}'")

        log.debug(f"country prediction/model prediction ---- current warehouse {wh_dept_id}, current_goods {goods_id}")

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

    # 全国训练一颗树的去预测各仓各货物
    def country_prediction(self, start_date, window, model, df):
        from datetime import datetime, timedelta
        df_country = pd.DataFrame()
        # 各仓-各货物预测
        for (wh_dept_id, goods_id), df_wh_goods in df.groupby(["wh_dept_id", "goods_id"]):
            df_current = self.model_prediction(
                start_date=start_date,
                window=window,
                goods_id=goods_id,
                wh_dept_id=wh_dept_id,
                model=model,
                df=df_wh_goods)

            end_date = pd.to_datetime(start_date) + pd.DateOffset(days=window)

            df_current = df_current.query(f"ds >= '{start_date}' and ds <= '{end_date}'").reset_index(drop=True)
            df_country = pd.concat([df_country, df_current])
        # 格式修改
        df_country["dt"] = (pd.to_datetime(start_date) + timedelta(days=1)).strftime("%Y-%m-%d")
        df_country["ds"] = df_country["ds"].dt.strftime("%Y-%m-%d")
        df_country.rename(columns={"ds": "predict_dt", "y": "demand"}, inplace=True)
        df_country = df_country[["predict_dt", "wh_dept_id", "goods_id", "demand", "dt"]]
        df_country["wh_dept_id"] = df_country["wh_dept_id"].astype(int)
        df_country["goods_id"] = df_country["goods_id"].astype(int)
        return df_country

    # 全国一个货物一棵树
    def country_goods_prediction(self, start_date, window, goods_id, model, df):
        from datetime import datetime, timedelta
        df_goods_country = pd.DataFrame()
        # 各仓-各货物预测
        for (wh_dept_id), df_wh_goods in df.groupby(["wh_dept_id"]):
            log.debug(f"goods_id {goods_id} wh_dept_id {wh_dept_id}")
            df_current = self.model_prediction(
                start_date=start_date,
                window=window,
                goods_id=goods_id,
                wh_dept_id=wh_dept_id,
                model=model,
                df=df_wh_goods)

            end_date = pd.to_datetime(start_date) + pd.DateOffset(days=window)
            df_current = df_current.query(f"ds >= '{start_date}' and ds <= '{end_date}'").reset_index(drop=True)
            df_goods_country = pd.concat([df_goods_country, df_current])
        # 格式修改
        df_goods_country["dt"] = (pd.to_datetime(start_date) + timedelta(days=1)).strftime("%Y-%m-%d")
        df_goods_country["ds"] = df_goods_country["ds"].dt.strftime("%Y-%m-%d")
        df_goods_country.rename(columns={"ds": "predict_dt", "y": "demand"}, inplace=True)
        df_goods_country = df_goods_country[["predict_dt", "wh_dept_id", "goods_id", "demand", "dt"]]
        df_goods_country["wh_dept_id"] = df_goods_country["wh_dept_id"].astype(int)
        df_goods_country["goods_id"] = df_goods_country["goods_id"].astype(int)
        return df_goods_country


def main(pred_calc_day=None):
    def predict_different_goods(df_feature, features, true_label, train_ratio, model_label, start_date):
        df_country = pd.DataFrame()  # 创建一个空的 DataFrame 用于存储结果
        model_dict = {}  # 创建一个空的字典用于存储模型
        for goods_id, df in tqdm(df_feature.groupby(["goods_id"]), total=len(df_feature["goods_id"].unique()),
                                 desc="Processing goods_id"):
            # 针对每个 goods_id 进行循环
            xgb_trainer = TreeModelTrainer(features, true_label, train_ratio, model_label)
            df = df.query(f"ds <= '{pd.to_datetime(start_date)}'")  # 根据 start_date 过滤数据
            train_df, test_df, X_train, y_train, X_test, y_test = xgb_trainer.split_train_and_test(df)
            if train_df.empty:
                continue  # 如果 train_df 为空，则跳过当前 goods_id
            xgb_model = xgb_trainer.train_model(X_train, y_train)  # 训练模型
            model_dict[goods_id] = xgb_model  # 将模型存储到字典中
            df_goods_country = xgb_trainer.country_goods_prediction(start_date = start_date, window=30, goods_id=goods_id,
                                                                    model=xgb_model, df=df)
            df_country = pd.concat([df_country, df_goods_country])  # 将预测结果添加到 df_country 中
        return df_country, model_dict

    pred_calc_day = DayStr.get_dt_str(pred_calc_day)
    pred_minus1_day = DayStr.n_day_delta(pred_calc_day, n=-1)
    # 获取商品名
    normal_goods = GoodsData("normal", pred_calc_day).goods_consume_data()
    all_goods = GoodsData("all", pred_calc_day).goods_consume_data()
    # 获取历史销售记录
    df_stock_sell_all = GoodsData("all", pred_calc_day).get_his_true_value(all_goods)
    df_stock_sell_normal = GoodsData("normal", pred_calc_day).get_his_true_value(normal_goods)
    # 特征生成
    feature_generator = FeatureEngineer()
    df_feature_all = feature_generator.feature_combined(df_stock_sell_all.reset_index(drop=True))
    df_feature_norm = feature_generator.feature_combined(df_stock_sell_normal.reset_index(drop=True))

    features = ['wh_dept_id', 'year', 'month', 'day', 'day_of_week',
                'lag_1_mean', 'lag_7_mean', 'lag_7_std', 'lag_7_min', 'lag_7_max', 'lag_14_mean',
                'lag_14_std', 'lag_14_min', 'lag_14_max', 'lag_21_mean', 'lag_21_std',
                'lag_21_min', 'lag_21_max', 'lag_28_mean', 'lag_28_std', 'lag_28_min',
                'lag_28_max', 'lag_30_mean', 'lag_30_std', 'lag_30_min', 'lag_30_max',
                'lag_60_mean', 'lag_60_std', 'lag_60_min', 'lag_60_max', 'lag_120_mean',
                'lag_120_std', 'lag_120_min', 'lag_120_max']

    true_label = "y"
    train_ratio = 1
    model_label = "xgb"

    #全量
    df_all_predict,all_model = predict_different_goods(df_feature_all,
                                                       features,
                                                       true_label = "y",
                                                       train_ratio = 1,
                                                       model_label = "xgb",
                                                       start_date = pred_calc_day)
    #常规品
    df_norm_predict,normal_model = predict_different_goods(df_feature_norm,
                                                           features,
                                                           true_label = "y",
                                                           train_ratio = 1,
                                                           model_label = "xgb",
                                                           start_date = pred_calc_day)
    #落库
    bip3_save_df2(
        df_all_predict,
        table_folder_name='predict_40_goods_all',
        bip_folder='model/basic_predict_promote',
        output_name=f"predict_40_goods_all",
        folder_dt = pred_calc_day
    )
    bip3_save_df2(
        df_norm_predict,
        table_folder_name='predict_40_goods_norm',
        bip_folder='model/basic_predict_promote',
        output_name=f"predict_40_goods_norm",
        folder_dt = pred_calc_day
    )

    return None