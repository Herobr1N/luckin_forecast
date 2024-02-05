import sys
import pandas as pd
import numpy as np
sys.path.insert(0, "/home/dinghuo/alg_dh")

from utils_offline.a00_imports import (dfu, log, DayStr,
                                       argv_date, dop, bip2, bip1, bip3, c_path, read_api, shuttle, c_path_save_df, bip3_save_df2)

def price_data_extract(pred_minus1_day):
    # 读各仓货物单价
    sql_price = f"""-- 仓，货物，钱
    SELECT
        wh_dept_id
        , goods_id
        , SUM(end_wh_stock_money) / SUM(end_wh_stock_cnt) AS unit_price
    FROM dw_dws.dws_stock_warehouse_stock_adjust_d_inc_summary
    WHERE dt >= DATE_SUB('{pred_minus1_day}', 180)
      AND spec_status = 1
      AND end_wh_stock_cnt > 0
      AND goods_id in (19952, 23628, 27954)
    GROUP BY wh_dept_id, goods_id
    """
    df_price = shuttle.query_dataq(sql_price)

    df_price_mean = df_price.groupby('goods_id')['unit_price'].mean().rename('unit_price_mean').reset_index()
    return df_price_mean
"""
原始消耗数据读取
"""
def consume_data(pred_minus1_day):
    #读取货物名
    sql_consume = f"""
                SELECT DISTINCT goods_id,goods_name, is_formula, large_class_name
                FROM dw_ads_scm_alg.dev_luckin_demand_forecast_category_info1
                WHERE dt >= '2023-08-01' AND (large_class_id in (3,6,36) or (large_class_id = 4 and is_formula = 1))
            """
    goods_ls_consume = shuttle.query_dataq(sql_consume)

    goods_name_list = ["冷萃厚牛乳","北海道丝绒风味厚乳","冰凉感厚椰饮品"]
    goods_id_list = list(goods_ls_consume[goods_ls_consume["goods_name"].isin(goods_name_list)]["goods_id"].unique())
    df_stock_sell = read_api.read_dt_folder(
        bip3("model/basic_predict_promote", "stock_wh_goods_type_flg_theory_sale_cnt")
        , '2020-01-01', pred_minus1_day)

    # 转成正数
    df_stock_sell["theory_sale_cnt"] = abs(df_stock_sell["theory_sale_cnt"])

    # 货物筛选
    df_stock_sell = df_stock_sell[df_stock_sell["goods_id"].isin(goods_id_list)].reset_index(drop=True)
    # 转换时间列
    df_stock_sell["dt"] = pd.to_datetime(df_stock_sell["dt"])
    # 列重命名
    df_stock_sell.rename(columns={"theory_sale_cnt": "y"
        , "dt": "ds"}, inplace=True)
    return df_stock_sell
"""
读取货物消耗数据
"""
def data_extract(df_stock_sell, wh_dept_id = 4001,goods_id = 19952):
    #货物筛选
    #有两组数据需要预测：常规品 & 全量
    # 1.type_flag = True
    df_current_stock_norm =  df_stock_sell[
        (df_stock_sell["wh_dept_id"] == wh_dept_id)
        &(df_stock_sell["goods_id"] == goods_id)
        &(df_stock_sell["type_flg"] == "norm")
        &(df_stock_sell["shop_type_flg"] == "norm_shop")
    ].reset_index(drop = True).copy()
    # 2.三种type_flag聚合
    df_current_stock_all = df_stock_sell[
        (df_stock_sell["wh_dept_id"] == wh_dept_id)
        &(df_stock_sell["goods_id"] == goods_id)
    ].reset_index(drop = True).copy()
    df_current_stock_all = df_current_stock_all.groupby(["ds",
                                                         "wh_dept_id",
                                                         "goods_id"]).agg({"y":"sum"}).reset_index()
    return df_current_stock_all, df_current_stock_norm
"""
单仓库，货物 - 特征生成
"""
def feature_generated(df_current_stock_all):
    # 添加時間列
    df_current_stock_all["year"] = df_current_stock_all["ds"].dt.year
    df_current_stock_all["month"] = df_current_stock_all["ds"].dt.month
    df_current_stock_all["day"] = df_current_stock_all["ds"].dt.day
    df_current_stock_all["day_of_week"] = df_current_stock_all["ds"].dt.dayofweek

    # 回归7 days，14 days,21 days, 28 days, 30 days, 60 days, 120 days 的mean, std, min, max
    lag_list = [7, 14, 21, 28, 30, 60, 120]

    df_rolled_1d = df_current_stock_all["y"].rolling(window=1, min_periods=0)
    df_mean_1d = df_rolled_1d.mean().shift(1).reset_index()

    df_rolled_7d = df_current_stock_all["y"].rolling(window=7, min_periods=0)
    df_mean_7d = df_rolled_7d.mean().shift(1).reset_index()
    df_std_7d = df_rolled_7d.std().shift(1).reset_index()
    df_min_7d = df_rolled_7d.min().shift(1).reset_index()
    df_max_7d = df_rolled_7d.max().shift(1).reset_index()

    df_rolled_14d = df_current_stock_all["y"].rolling(window=14, min_periods=0)
    df_mean_14d = df_rolled_14d.mean().shift(1).reset_index()
    df_std_14d = df_rolled_14d.std().shift(1).reset_index()
    df_min_14d = df_rolled_14d.min().shift(1).reset_index()
    df_max_14d = df_rolled_14d.max().shift(1).reset_index()

    df_rolled_21d = df_current_stock_all["y"].rolling(window=21, min_periods=0)
    df_mean_21d = df_rolled_21d.mean().shift(1).reset_index()
    df_std_21d = df_rolled_21d.std().shift(1).reset_index()
    df_min_21d = df_rolled_21d.min().shift(1).reset_index()
    df_max_21d = df_rolled_21d.max().shift(1).reset_index()

    df_rolled_28d = df_current_stock_all["y"].rolling(window=28, min_periods=0)
    df_mean_28d = df_rolled_28d.mean().shift(1).reset_index()
    df_std_28d = df_rolled_28d.std().shift(1).reset_index()
    df_min_28d = df_rolled_28d.min().shift(1).reset_index()
    df_max_28d = df_rolled_28d.max().shift(1).reset_index()

    df_rolled_30d = df_current_stock_all["y"].rolling(window=30, min_periods=0)
    df_mean_30d = df_rolled_30d.mean().shift(1).reset_index()
    df_std_30d = df_rolled_30d.std().shift(1).reset_index()
    df_min_30d = df_rolled_30d.min().shift(1).reset_index()
    df_max_30d = df_rolled_30d.max().shift(1).reset_index()

    df_rolled_60d = df_current_stock_all["y"].rolling(window=60, min_periods=0)
    df_mean_60d = df_rolled_60d.mean().shift(1).reset_index()
    df_std_60d = df_rolled_60d.std().shift(1).reset_index()
    df_min_60d = df_rolled_60d.min().shift(1).reset_index()
    df_max_60d = df_rolled_60d.max().shift(1).reset_index()

    df_rolled_120d = df_current_stock_all["y"].rolling(window=120, min_periods=0)
    df_mean_120d = df_rolled_120d.mean().shift(1).reset_index()
    df_std_120d = df_rolled_120d.std().shift(1).reset_index()
    df_min_120d = df_rolled_120d.min().shift(1).reset_index()
    df_max_120d = df_rolled_120d.max().shift(1).reset_index()

    df_current_stock_all["lag_1_mean"] = df_mean_1d["y"]

    df_current_stock_all["lag_7_mean"] = df_mean_7d["y"]
    df_current_stock_all["lag_7_std"] = df_std_7d["y"]
    df_current_stock_all["lag_7_min"] = df_min_7d["y"]
    df_current_stock_all["lag_7_max"] = df_max_7d["y"]

    df_current_stock_all["lag_14_mean"] = df_mean_14d["y"]
    df_current_stock_all["lag_14_std"] = df_std_14d["y"]
    df_current_stock_all["lag_14_min"] = df_min_14d["y"]
    df_current_stock_all["lag_14_max"] = df_max_14d["y"]

    df_current_stock_all["lag_21_mean"] = df_mean_21d["y"]
    df_current_stock_all["lag_21_std"] = df_std_21d["y"]
    df_current_stock_all["lag_21_min"] = df_min_21d["y"]
    df_current_stock_all["lag_21_max"] = df_max_21d["y"]

    df_current_stock_all["lag_28_mean"] = df_mean_28d["y"]
    df_current_stock_all["lag_28_std"] = df_std_28d["y"]
    df_current_stock_all["lag_28_min"] = df_min_28d["y"]
    df_current_stock_all["lag_28_max"] = df_max_28d["y"]

    df_current_stock_all["lag_30_mean"] = df_mean_30d["y"]
    df_current_stock_all["lag_30_std"] = df_std_30d["y"]
    df_current_stock_all["lag_30_min"] = df_min_30d["y"]
    df_current_stock_all["lag_30_max"] = df_max_30d["y"]

    df_current_stock_all["lag_60_mean"] = df_mean_60d["y"]
    df_current_stock_all["lag_60_std"] = df_std_60d["y"]
    df_current_stock_all["lag_60_min"] = df_min_60d["y"]
    df_current_stock_all["lag_60_max"] = df_max_60d["y"]

    df_current_stock_all["lag_120_mean"] = df_mean_120d["y"]
    df_current_stock_all["lag_120_std"] = df_std_120d["y"]
    df_current_stock_all["lag_120_min"] = df_min_120d["y"]
    df_current_stock_all["lag_120_max"] = df_max_120d["y"]

    return df_current_stock_all


def feature_combined(df_stock_sell):
    df_current_stock_all_large = pd.DataFrame()
    df_current_stock_norm_large = pd.DataFrame()
    for wh_id in df_stock_sell["wh_dept_id"].unique():
        for g_id in goods_id_list:
            df_current_stock_all, df_current_stock_norm = data_extract(wh_dept_id=wh_id,
                                                                       goods_id=g_id)

            # 特征生成
            df_current_stock_all = feature_generated(df_current_stock_all)

            df_current_stock_norm = feature_generated(df_current_stock_norm)

            df_current_stock_all_large = pd.concat([df_current_stock_all_large,
                                                    df_current_stock_all])
            df_current_stock_norm_large = pd.concat([df_current_stock_norm_large,
                                                     df_current_stock_norm])

    return df_current_stock_all_large, df_current_stock_norm_large

def feaature_data_to_hdfs(df_stock_sell):
    #特征工程
    df_current_stock_all_large, df_current_stock_norm_large = feature_combined(df_stock_sell)
    #落库
    bip3_save_df2(df_current_stock_all_large,
          table_folder_name='feature_engineering_all',
          bip_folder='model/basic_predict_promote',
          output_name="feature_engineering_all",
          folder_dt=pred_minus1_day
    )
    bip3_save_df2(df_current_stock_norm_large,
          table_folder_name='feature_engineering_normal',
          bip_folder='model/basic_predict_promote',
          output_name=f"feature_engineering_norm",
          folder_dt=pred_minus1_day
    )


    #常规品+次新品+新品消耗
    tt = read_api.read_dt_folder(
    bip3("model/basic_predict_promote", "feature_engineering_all")
    , pred_minus1_day)



    #常规店常规消耗
    yy = read_api.read_dt_folder(
    bip3("model/basic_predict_promote", "feature_engineering_normal")
    , pred_minus1_day)
    return tt,yy

def data_feature_main():
    pred_calculation_day = None
    pred_calc_day = DayStr.get_dt_str(pred_calculation_day)
    pred_minus1_day = DayStr.n_day_delta(pred_calc_day, n=-1)
    days_to_include = 6
    start_dt = DayStr.n_day_delta(pred_minus1_day, n=-days_to_include)


    df_price_mean = price_data_extract(pred_minus1_day)
    df_stock_sell = consume_data(pred_minus1_day)
    # tt 是全量数据， yy是常规品数据
    tt,yy = feaature_data_to_hdfs(df_stock_sell)
    return None