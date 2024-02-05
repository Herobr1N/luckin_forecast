# encoding: utf-8
# @created: 2023/11/2
# @author: jieqin.lin
# @file: projects/basic_predict_promote/bpp02_feature_engineering_week.py


"""
## 特征工程：7，14，21，28，30，60，120天的统计值（均值，标准差，最大值，最小值等）

生成数据

# 自然周维度
bip3("model/basic_predict_promote_online", "feature_engineering_wh_goods_week_of_year_all")
bip3("model/basic_predict_promote_online", "feature_engineering_wh_goods_week_of_year_normal")
bip3("model/basic_predict_promote_online", "feature_engineering_wh_goods_week_of_year_new_shop_normal")

# 自然月维度
bip3("model/basic_predict_promote_online", "feature_engineering_wh_goods_month_of_year_all")
bip3("model/basic_predict_promote_online", "feature_engineering_wh_goods_month_of_year_normal")
bip3("model/basic_predict_promote_online", "feature_engineering_wh_goods_month_of_year_new_shop_normal")

依赖数据
bip3("model/basic_predict_promote_online", "stock_wh_goods_type_flg_theory_sale_cnt")
bip3("model/basic_predict_promote_online", "stock_wh_goods_theory_sale_cnt")

bip3("model/basic_predict_promote_online", "week_of_year_stock_wh_goods_type_flg_theory_sale_cnt")        自然周++仓库++货物++类型++消耗
bip3("model/basic_predict_promote_online", "week_of_year_stock_wh_goods_theory_sale_cnt")                 自然周++仓库++货物++消耗

bip3("model/basic_predict_promote_online", "month_of_year_stock_wh_goods_type_flg_theory_sale_cnt")        自然月++仓库++货物++类型++消耗
bip3("model/basic_predict_promote_online", "month_of_year_stock_wh_goods_theory_sale_cnt")                 自然月++仓库++货物++消耗

"""

import pandas as pd
import numpy as np
from __init__ import project_path
from projects.basic_predict_promote.b_0_0_utils_models import SelectGoodsList, FeatureEngineer
from utils_offline.a00_imports import log20 as log, DayStr, argv_date, bip3_save_df2

f"Import from {project_path}"


def main_feature_engineering_wh_goods(pred_calculation_day=None):
    """
    ## 历史回测 方法一
    from projects.basic_predict_promote.bpp02_feature_engineering import *
    data_back_fill_feature_engineering_wh_goods()
    """
    data_label_ls = ['normal', 'all', 'new_shop_normal']

    for data_label in data_label_ls:
        feature02_engineering_wh_goods_week_of_year(pred_calculation_day, days_to_include=180, data_label=data_label)
        feature03_engineering_wh_goods_month_of_year(pred_calculation_day, days_to_include=380, data_label=data_label)


def feature02_engineering_wh_goods_week_of_year(pred_calculation_day=None, days_to_include=180, data_label='all'):
    """
    构造特征：2, 4, 6, 8, 12, 16, 24周的统计值（均值，标准差，最大值，最小值等）
    """

    pred_calc_day = DayStr.get_dt_str(pred_calculation_day)
    pred_minus1_day = DayStr.n_day_delta(pred_calc_day, n=-1)
    start_date = DayStr.n_day_delta(pred_minus1_day, n=-days_to_include)

    # 货物范围 46个长周期货物
    selector = SelectGoodsList()
    sel_goods_ls = selector.get_long_period_goods_id()

    # 历史售卖
    feature_generator = FeatureEngineer(data_label=data_label)
    df_stock_sell = feature_generator.get_his_week_of_year_stock_wh_goods(start_date=start_date,
                                                                          end_date=pred_minus1_day,
                                                                          sel_goods_ls=sel_goods_ls)
    # 特征
    df_feature_wh_goods = feature_generator.feature_generated_wh_goods(df=df_stock_sell, time_label='week')
    max_dt = df_feature_wh_goods['ds'].max().strftime('%Y-%m-%d')
    df_feature = df_feature_wh_goods.query(f"ds =='{max_dt}'").copy()

    # save
    bip3_save_df2(df_feature,
                  table_folder_name=f'feature_engineering_wh_goods_week_of_year_{data_label}',
                  bip_folder='model/basic_predict_promote_online',
                  output_name=f'feature_engineering_wh_goods_week_of_year_{data_label}',
                  folder_dt=max_dt)


def feature03_engineering_wh_goods_month_of_year(pred_calculation_day=None, days_to_include=380, data_label='all'):
    """
    构造特征：2，3，4，6，12月的统计值（均值，标准差，最大值，最小值等）
    """

    pred_calc_day = DayStr.get_dt_str(pred_calculation_day)
    pred_minus1_day = DayStr.n_day_delta(pred_calc_day, n=-1)
    start_date = DayStr.n_day_delta(pred_minus1_day, n=-days_to_include)

    # 货物范围 46个长周期货物
    selector = SelectGoodsList()
    sel_goods_ls = selector.get_long_period_goods_id()

    # 历史售卖
    feature_generator = FeatureEngineer(data_label=data_label)
    df_stock_sell = feature_generator.get_his_month_of_year_stock_wh_goods(start_date=start_date,
                                                                           end_date=pred_minus1_day,
                                                                           sel_goods_ls=sel_goods_ls)
    # 特征
    df_feature_wh_goods = feature_generator.feature_generated_wh_goods(df=df_stock_sell, time_label='month')
    max_dt = df_feature_wh_goods['ds'].max().strftime('%Y-%m-%d')
    df_feature = df_feature_wh_goods.query(f"ds =='{max_dt}'").copy()

    # save
    bip3_save_df2(df_feature,
                  table_folder_name=f'feature_engineering_wh_goods_month_of_year_{data_label}',
                  bip_folder='model/basic_predict_promote_online',
                  output_name=f'feature_engineering_wh_goods_month_of_year_{data_label}',
                  folder_dt=max_dt)


def data_back_fill_feature_engineering_wh_goods():
    import concurrent.futures
    """
    回刷数据 并行构造特征
    """

    def process_data(time_label, data_label):
        log.debug(f"{time_label}--{data_label}")
        pred_calc_day = DayStr.get_dt_str(None)
        pred_minus1_day = DayStr.n_day_delta(pred_calc_day, n=-1)
        start_date = '2019-12-01'

        # 货物范围 46个长周期货物
        selector = SelectGoodsList()
        sel_goods_ls = selector.get_long_period_goods_id()

        # 历史售卖
        feature_generator = FeatureEngineer(data_label=data_label)

        if time_label == 'daily':
            df_stock_sell = feature_generator.get_his_stock_wh_goods(start_date=start_date,
                                                                     end_date=pred_minus1_day,
                                                                     sel_goods_ls=sel_goods_ls)

        if time_label == 'week':
            df_stock_sell = feature_generator.get_his_week_of_year_stock_wh_goods(start_date=start_date,
                                                                                  end_date=pred_minus1_day,
                                                                                  sel_goods_ls=sel_goods_ls)

        if time_label == 'month':
            df_stock_sell = feature_generator.get_his_month_of_year_stock_wh_goods(start_date=start_date,
                                                                                   end_date=pred_minus1_day,
                                                                                   sel_goods_ls=sel_goods_ls)

        # 特征
        df_feature_wh_goods = feature_generator.feature_generated_wh_goods(df=df_stock_sell, time_label=time_label)

        for dt in df_feature_wh_goods['ds'].unique():
            dt_str = np.datetime_as_string(dt, unit='D')
            if dt_str >= '2020-01-01':
                print(dt_str)
                df_feature = df_feature_wh_goods.query(f"ds =='{dt}'").copy()
                if len(df_feature) > 0:
                    if time_label == 'daily':
                        # save
                        bip3_save_df2(df_feature,
                                      table_folder_name=f'feature_engineering_wh_goods_{data_label}',
                                      bip_folder='model/basic_predict_promote_online',
                                      output_name=f'feature_engineering_wh_goods_{data_label}',
                                      folder_dt=dt_str)
                    if time_label == 'week':
                        # save
                        bip3_save_df2(df_feature,
                                      table_folder_name=f'feature_engineering_wh_goods_week_of_year_{data_label}',
                                      bip_folder='model/basic_predict_promote_online',
                                      output_name=f'feature_engineering_wh_goods_week_of_year_{data_label}',
                                      folder_dt=dt_str)
                    if time_label == 'month':
                        # save
                        bip3_save_df2(df_feature,
                                      table_folder_name=f'feature_engineering_wh_goods_month_of_year_{data_label}',
                                      bip_folder='model/basic_predict_promote_online',
                                      output_name=f'feature_engineering_wh_goods_month_of_year_{data_label}',
                                      folder_dt=dt_str)

    data_label_ls = ['normal', 'all', 'new_shop_normal']
    time_label_ls = ['daily', 'week', 'month']

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []

        for time_label in time_label_ls:
            for data_label in data_label_ls:
                futures.append(executor.submit(process_data, time_label, data_label))

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                print(f"An error occurred: {str(e)}")


if __name__ == '__main__':
    argv_date(main_feature_engineering_wh_goods)
