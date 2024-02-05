# encoding: utf-8
# @created: 2023/11/23
# @author: yuyang.lin
import sys
import os

sys.path.insert(0, '/home/dinghuo/luckystoreordering/')
sys.path.insert(0, '/home/dinghuo/alg_dh/')

from utils_offline.a00_imports import log20 as log, DayStr, argv_date, c_path, read_api, shuttle, c_path_save_df, dop, \
    bip2, bip3, \
    hive_table_hdfs_path, list_info_hdfs_path, upload_file_to_hdfs_path, shuttle, bip3_save_df2, dfu, bip1
from areas.table_info.dh_dw_table_info import dh_dw
import numpy as np
from datetime import timedelta
from datetime import datetime
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

"""
读取前n个月
每个月第一天的预测
"""


def get_previous_months_first_days(today, n):
    result = []
    today = datetime.strptime(today, "%Y-%m-%d")

    for i in range(n):
        year = today.year
        month = today.month
        # 如果是第一个月，返回上一年
        if month == 1:
            year -= 1
            month = 12
        else:
            month -= 1

        first_day = datetime(year, month, 1)
        result.append(first_day.strftime("%Y-%m-%d"))

        today = first_day
    return sorted(result)


"""
读取真实消耗数据
按货物维度月汇总消耗
"""


def backtest(data_label, model_label, backtest_month_list):
    # 常规品
    if data_label == 'normal':
        # 真实值
        df_stock_sell = read_api.read_dt_folder(
            bip3("model/basic_predict_promote_online", "stock_wh_goods_type_flg_theory_sale_cnt"),
            min(backtest_month_list), DayStr.n_day_delta(max(backtest_month_list), n=30))
        df_stock_sell = df_stock_sell[
            (df_stock_sell['type_flg'] == 'norm') & (df_stock_sell['shop_type_flg'] == 'norm_shop')]

        df_stock_sell.drop(['type_flg', 'shop_type_flg'], axis=1, inplace=True)
        # 预测
        df_predict = read_api.read_dt_folder(
            bip3("model/basic_predict_promote", f"predict_40_goods_norm_{model_label}"), min(backtest_month_list),
            max(backtest_month_list))
    # 全量
    if data_label == 'all':
        # 真实值
        df_stock_sell = read_api.read_dt_folder(
            bip3("model/basic_predict_promote_online", "stock_wh_goods_theory_sale_cnt"),
            min(backtest_month_list), DayStr.n_day_delta(max(backtest_month_list), n=30))
        # 预测
        df_predict = read_api.read_dt_folder(
            bip3("model/basic_predict_promote", f"predict_40_goods_all_{model_label}"), min(backtest_month_list),
            max(backtest_month_list))
    # 新品
    if data_label == 'new_shop_normal':
        # 真实值
        df_stock_sell = read_api.read_dt_folder(
            bip3("model/basic_predict_promote_online", "stock_wh_goods_type_flg_theory_sale_cnt"),
            min(backtest_month_list), DayStr.n_day_delta(max(backtest_month_list), n=30))

        df_stock_sell = df_stock_sell[
            (df_stock_sell['type_flg'] == 'norm') & (df_stock_sell['shop_type_flg'] == 'new_shop')]
        df_stock_sell.drop(['type_flg', 'shop_type_flg'], axis=1, inplace=True)
        # 预测
        df_predict = read_api.read_dt_folder(
            bip3("model/basic_predict_promote", f"predict_40_goods_new_shop_normal_{model_label}"),
            min(backtest_month_list), max(backtest_month_list))

        # 消耗数据转正数
    df_stock_sell["theory_sale_cnt"] = np.abs(df_stock_sell["theory_sale_cnt"])
    # 两表合并
    df_merge = df_predict.merge(
        df_stock_sell,
        left_on=["predict_dt", "wh_dept_id", "goods_id"],
        right_on=["dt", "wh_dept_id", "goods_id"],
        how="left")

    # 按仓货汇总
    df_wh_goods = df_merge.groupby(
        ["goods_id", "wh_dept_id", "dt_x"]
    ).agg(
        {"theory_sale_cnt": "sum", "demand": "sum"}
    ).reset_index()
    df_wh_goods["model"] = model_label
    # 按货汇总
    df_goods = df_merge.groupby(
        ["goods_id", "dt_x"]
    ).agg(
        {"theory_sale_cnt": "sum", "demand": "sum"}
    ).reset_index()
    df_goods["model"] = model_label
    # 按全国维度汇总
    df_country = df_merge.groupby(
        ["dt_x"]
    ).agg(
        {"theory_sale_cnt": "sum", "demand": "sum"}
    ).reset_index()
    df_country["model"] = model_label

    return df_wh_goods, df_goods, df_country


def metric(df_month):
    # 计算 MAPE
    df_month['MAPE'] = np.abs((df_month['demand'] - df_month['theory_sale_cnt']) / df_month['theory_sale_cnt'])

    # 计算 MAE
    df_month['MAE'] = np.abs(df_month['demand'] - df_month['theory_sale_cnt'])

    # 计算 MSE
    df_month['MSE'] = (df_month['demand'] - df_month['theory_sale_cnt']) ** 2

    # 计算 Accuracy
    df_month['ACC'] = 1 - df_month['MAPE']
    df_month.loc[df_month['ACC'] < 0, 'ACC'] = 0

    return df_month


"""
三种树模型准确率汇总
"""


def metric_summary(data_label, backtest_month_list):
    # 仓货
    df_wh_goods_acc = pd.DataFrame()
    # 货
    df_goods_acc = pd.DataFrame()
    # 全国
    df_country_acc = pd.DataFrame()
    for model_label in ["xgb", "catboost", "lgbm"]:
        # 数据汇总
        df_wh_goods_model, df_goods_model, df_country_model = backtest(data_label=data_label,
                                                                       model_label=model_label,
                                                                       backtest_month_list=backtest_month_list)

        df_wh_goods_acc_model = metric(df_wh_goods_model)
        df_goods_acc_model = metric(df_goods_model)
        df_country_acc_model = metric(df_country_model)

        df_wh_goods_acc = pd.concat([df_wh_goods_acc, df_wh_goods_acc_model])
        df_goods_acc = pd.concat([df_goods_acc, df_goods_acc_model])
        df_country_acc = pd.concat([df_country_acc, df_country_acc_model])
    # 格式修改
    df_wh_goods_acc.rename(columns={"dt_x": "dt"}, inplace=True)
    df_wh_goods_acc["dt"] = df_wh_goods_acc["dt"].dt.strftime("%Y-%m-%d").astype(str)
    df_wh_goods_acc = df_wh_goods_acc[['goods_id', 'wh_dept_id', 'dt', 'model', 'MAPE', 'MAE', 'MSE', 'ACC']]

    df_goods_acc.rename(columns={"dt_x": "dt"}, inplace=True)
    df_goods_acc["dt"] = df_goods_acc["dt"].dt.strftime("%Y-%m-%d").astype(str)
    df_goods_acc = df_goods_acc[['goods_id', 'dt', 'model', 'MAPE', 'MAE', 'MSE', 'ACC']]

    df_country_acc.rename(columns={"dt_x": "dt"}, inplace=True)
    df_country_acc["dt"] = df_country_acc["dt"].dt.strftime("%Y-%m-%d").astype(str)
    df_country_acc = df_country_acc[['dt', 'model', 'MAPE', 'MAE', 'MSE', 'ACC']]

    return df_wh_goods_acc, df_goods_acc, df_country_acc


def main_acc_backtest(pred_calc_day=None):
    pred_calc_day = DayStr.get_dt_str(pred_calc_day)
    # 读前三个月第一天日期
    date_list = get_previous_months_first_days(pred_calc_day, 1)
    for data_label in ["all", "normal", "new_shop_normal"]:
        # 计算树模型过去三个月准确率
        df_wh_goods_summary, df_goods_summary, df_country_summary = metric_summary(
            data_label=data_label,
            backtest_month_list=date_list
        )
        # 落库
        bip3_save_df2(
            df_wh_goods_summary,
            table_folder_name=f'predict_40_goods_{data_label}_past_3_months_acc_wh_goods_summary',
            bip_folder='model/basic_predict_promote',
            output_name=f"predict_40_goods_{data_label}_past_3_months_acc_wh_goods_summary",
            folder_dt=pred_calc_day
        )
        bip3_save_df2(
            df_goods_summary,
            table_folder_name=f'predict_40_goods_{data_label}_past_3_months_acc_goods_summary',
            bip_folder='model/basic_predict_promote',
            output_name=f"predict_40_goods_{data_label}_past_3_months_acc_goods_summary",
            folder_dt=pred_calc_day
        )
        bip3_save_df2(
            df_country_summary,
            table_folder_name=f'predict_40_goods_{data_label}_past_3_months_acc_country_summary',
            bip_folder='model/basic_predict_promote',
            output_name=f"predict_40_goods_{data_label}_past_3_months_acc_country_summary",
            folder_dt=pred_calc_day
        )
    return None