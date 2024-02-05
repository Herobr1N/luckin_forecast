# encoding: utf-8
# @created: 2023/11/9
# @author: yuyang.lin

import sys
import os

sys.path.insert(0, '/home/dinghuo/alg_dh/')
sys.path.insert(0, '/home/dinghuo/luckystoreordering/')
from __init__ import project_path
from areas.table_info.dh_dw_table_info import dh_dw
from utils_offline.a00_imports import dfu, log20 as log, DayStr, read_api, shuttle, bip3_save_df2, bip3
from datetime import timedelta
import pandas as pd
from prophet import Prophet


# This class is used for monitoring the country consumption of three goods using the Prophet forecasting model.
# It provides a convenient way to train the model and make predictions for each goods_id in the input dataframe.


class DataProcessor:
    def __init__(self, train_size):
        # 定义取多长训练集做为监控训练
        self.train_size = train_size

    def consume_data(self, date):
        log.debug(" -----------------历史消耗数据获取-------------------")
        """
        Read goods names from the database and filter based on criteria.

        Args:
            date (str): The date for which predictions are made.

        Returns:
            df (pd.DataFrame): The processed dataframe with filtered goods data.
        """
        sql_consume = f"""
                    SELECT DISTINCT goods_id,goods_name, is_formula, large_class_name
                    FROM dw_ads_scm_alg.dev_luckin_demand_forecast_category_info1
                    WHERE dt >= '2023-08-01' AND (large_class_id in (3,6,36) or (large_class_id = 4 and is_formula = 1))
                """
        goods_ls_consume = shuttle.query_dataq(sql_consume)
        initial_date = DayStr.n_day_delta(date, n=- self.train_size)

        goods_name_list = ["冷萃厚牛乳", "北海道丝绒风味厚乳", "冰凉感厚椰饮品"]

        goods_id_list = list(
            goods_ls_consume[goods_ls_consume["goods_name"].isin(goods_name_list)]["goods_id"].unique())
        # 数据读取
        df = read_api.read_dt_folder(
            bip3("model/basic_predict_promote", "stock_wh_goods_type_flg_theory_sale_cnt"),
            initial_date, date)

        # Convert to positive values
        df["theory_sale_cnt"] = abs(df["theory_sale_cnt"])
        # Filter goods
        df = df[df["goods_id"].isin(goods_id_list)].reset_index(drop=True)
        # Convert time column
        df["dt"] = pd.to_datetime(df["dt"])
        # Rename columns
        df.rename(columns={"theory_sale_cnt": "y", "dt": "ds"}, inplace=True)
        return df

    def data_extract(self, df):
        """
        Filter and aggregate data based on type_flag and shop_type_flag.

        Args:
            df (pd.DataFrame): The input dataframe containing the goods data.

        Returns:
            df_all (pd.DataFrame): The processed dataframe with aggregated data.
            df_norm (pd.DataFrame): The processed dataframe with filtered and aggregated data.
        """

        df_norm = df[
            (df["type_flg"] == "norm")
            & (df["shop_type_flg"] == "norm_shop")
            ].reset_index(drop=True).copy()
        df_norm = df_norm.groupby(["ds", "goods_id"]).agg({"y": "sum"}).reset_index()
        # 2. Aggregation of three type_flags
        df_all = df.reset_index(drop=True).copy()
        df_all = df_all.groupby(["ds", "goods_id"]).agg({"y": "sum"}).reset_index()
        return df_all, df_norm

    def feature_generated_prophet(self, df):
        """
        Add additional time-related features to the dataframe.

        Args:
            df (pd.DataFrame): The input dataframe containing the goods data.

        Returns:
            df (pd.DataFrame): The processed dataframe with added time-related features.
        """
        log.debug("-----------------时间特征生成-----------------")
        df["year"] = df["ds"].dt.year
        df["month"] = df["ds"].dt.month
        df["day"] = df["ds"].dt.day
        df["day_of_week"] = df["ds"].dt.dayofweek
        return df


class ProphetTrainer:
    def __init__(self,
                 today,
                 train_size,
                 test_size,
                 change_point_range,
                 change_point_scaler,
                 seasonal_m="additive",
                 growth_m="logistic"):

        """
        Initialize the ProphetTrainer class.

        Args:
            today (str): The current date in the format "YYYY-MM-DD".
            train_size (int): The number of days to include in the training set.
            test_size (int): The number of days to include in the test set.
            change_point_range (float): The proportion of the history to consider for potential changepoints.
            change_point_scaler (float): The flexibility of the automatic changepoint selection.
            seasonal_m (str, optional): The mode of the seasonality component. Defaults to "additive".
            growth_m (str, optional): The mode of the growth component. Defaults to "logistic".
        """
        self.today = today
        self.train_size = train_size
        self.test_size = test_size
        self.change_point_range = change_point_range
        self.change_point_scaler = change_point_scaler
        self.seasonal_m = seasonal_m
        self.growth_m = growth_m
        self.model = None

    def train(self, df):
        """
        Train the Prophet model and make predictions.

        Args:
            df (pd.DataFrame): The input dataframe containing the time series data.

        Returns:
            df_result (pd.DataFrame): The dataframe containing the predicted values.
            df_train (pd.DataFrame): The dataframe used for training the model.
        """

        # train test split

        train_start_date = pd.to_datetime(self.today) - timedelta(days=self.train_size)
        test_end_date = pd.to_datetime(self.today) + timedelta(days=self.test_size)

        log.debug(
            f" -----------------训练集开始日期 {train_start_date} 训练集结束日期 {pd.to_datetime(self.today) - timedelta(days=1)}")

        df_train = df[
            (df["ds"] < pd.to_datetime(self.today))
            & (df["ds"] >= train_start_date)
            ].copy().reset_index(drop=True)

        log.debug(
            f" -----------------测试集开始日期 {pd.to_datetime(self.today)} 测试集结束日期 {test_end_date - timedelta(days=1)}")

        df_test = df[
            (df["ds"] >= pd.to_datetime(self.today))
            & (df["ds"] < test_end_date)
            ].copy().reset_index(drop=True)
        # 测试集扩充
        if df_test.empty:
            first_row = df_train.iloc[-1].copy()
            df_test.loc[0] = first_row.values
            df_test["ds"] = pd.to_datetime(self.today)
            df_test["year"] = df_test["ds"].dt.year
            df_test["month"] = df_test["ds"].dt.month
            df_test["day"] = df_test["ds"].dt.day
            df_test["day_of_week"] = df_test["ds"].dt.day_of_week
            df_test["y"] = float("nan")

        if test_end_date > df_test["ds"].max():
            rows_needed = self.test_size - len(df_test)

            df_fill = pd.DataFrame({"ds": pd.date_range(
                df_test["ds"].max() + timedelta(days=1),
                periods=rows_needed, freq="D")}
            )
            df_test = pd.concat([df_test, df_fill]).reset_index(drop=True)

            df_test["goods_id"].fillna(method="ffill", inplace=True)
            df_test["year"].fillna(df_test["ds"].dt.year, inplace=True)
            df_test["month"].fillna(df_test["ds"].dt.month, inplace=True)
            df_test["day"].fillna(df_test["ds"].dt.day, inplace=True)
            df_test["day_of_week"].fillna(df_test["ds"].dt.dayofweek, inplace=True)

        cap = df_train["y"].max()
        floor = df_train["y"].min()

        df_train["cap"] = cap
        df_test["cap"] = cap
        df_train["floor"] = floor
        df_test["floor"] = floor

        self.model = Prophet(
            growth=self.growth_m,
            changepoint_range=self.change_point_range,
            changepoint_prior_scale=self.change_point_scaler,
            seasonality_mode=self.seasonal_m,
            interval_width=0.95
        )

        self.model.add_regressor("month")
        self.model.add_country_holidays(country_name="China")

        df_result = self.model.fit(df_train).predict(df_test)

        df_result = df_test.merge(
            df_result,
            left_on="ds",
            right_on="ds",
            how="left"
        )

        df_result["predict_dt"] = pd.to_datetime(self.today)

        return df_result, df_train

    def predict_all(self, df):
        """
        Train the Prophet model and make predictions for each goods_id in the input dataframe.

        Args:
            df (pd.DataFrame): The input dataframe containing the time series data.

        Returns:
            predict_goods (pd.DataFrame): The dataframe containing the predicted values for all goods_id.
            train_goods (pd.DataFrame): The dataframe containing the training data for all goods_id.
        """

        predict_goods = pd.DataFrame()
        train_goods = pd.DataFrame()
        for (goods_id), df_goods in df.groupby(["goods_id"]):
            log.debug(f" -----------------当前货物id {goods_id} -----------------")
            df_predict, df_train = self.train(df_goods)
            predict_goods = pd.concat([predict_goods, df_predict])
            train_goods = pd.concat([train_goods, df_train])
        # 格式修改
        predict_goods = predict_goods[["ds", "goods_id", "yhat", "predict_dt"]]
        predict_goods["ds"] = predict_goods["ds"].dt.strftime("%Y-%m-%d")
        predict_goods["predict_dt"] = predict_goods["predict_dt"].dt.strftime("%Y-%m-%d")
        predict_goods["goods_id"] = predict_goods["goods_id"].astype(int)
        predict_goods.rename(columns={"ds": "dt", "yhat": "demand"}, inplace=True)

        return predict_goods, train_goods


def main(pred_calculation_day=None):
    # 当前代码为执行t - 1，即输入 2023-10-01,执行预测日为 2023-09-30
    pred_calc_day = DayStr.get_dt_str(pred_calculation_day)
    pred_minus1_day = DayStr.n_day_delta(pred_calc_day, n=-1)
    log.debug(f"---------------执行日期 {pred_minus1_day}--------------")
    # 数据获取
    # all: 全量 norm：常规品
    processor = DataProcessor(train_size=180)

    df_sell = processor.consume_data(pred_minus1_day)
    df_all, df_norm = processor.data_extract(df_sell)

    df_all = processor.feature_generated_prophet(df_all)
    df_norm = processor.feature_generated_prophet(df_norm)

    # prophet 配置项，具体参考 ProphetTrainer代码
    """
    Prophet配置项
    today = pred_minus1_day  # 预测的日期
    train_size = 180  # 训练集的大小
    test_size = 30  # 测试集的大小
    change_point_range = 0.99  # 变化点检测的敏感度范围 (0-1)，越接近1表示更敏感
    change_point_scaler = 0.01  # 变化点检测的灵敏度调整因子，较小的值增加变化点数量，较大的值减少变化点数量
    seasonal_m = "additive"  # 季节性模式类型，可能的选择为 "additive"（加法模式）或 "multiplicative"（乘法模式）
    growth_m = "logistic"  # 趋势增长模型类型，可能的选择为 "logistic"（逻辑增长模型）或 "linear"（线性增长模型）
    """
    today = pred_minus1_day
    train_size = 180
    test_size = 30
    change_point_range = 0.99
    change_point_scaler = 0.01
    seasonal_m = "additive"
    growth_m = "logistic"
    # model
    trainer = ProphetTrainer(
        today=today,
        train_size=train_size,
        test_size=test_size,
        change_point_range=change_point_range,
        change_point_scaler=change_point_scaler,
        seasonal_m=seasonal_m,
        growth_m=growth_m
    )
    # 这里只输出predict_xxxx_goods这张表作为监控
    log.debug(' -----------------常规消耗 -----------------')
    predict_norm_goods, train_norm_goods = trainer.predict_all(df_norm)
    log.debug(' -----------------全量消耗 -----------------')
    predict_all_goods, train_all_goods = trainer.predict_all(df_all)
    # 常规品监控落库
    bip3_save_df2(
        predict_norm_goods,
        table_folder_name='prophet_monitor_norm',
        bip_folder='model/basic_predict_promote',
        output_name=f"prophet_monitor_norm",
        folder_dt=pred_minus1_day
    )
    # 全量监控落库
    bip3_save_df2(
        predict_norm_goods,
        table_folder_name='prophet_monitor_all',
        bip_folder='model/basic_predict_promote',
        output_name=f"prophet_monitor_all",
        folder_dt=pred_minus1_day
    )
    return

