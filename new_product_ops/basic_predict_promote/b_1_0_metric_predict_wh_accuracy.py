# -*- utf-8 -*-
# @Time: 2023-09-15
# @Author: jieqin.lin
# @File: projects/basic_predict_promote/b_1_0_metric_predict_wh_accuracy.py
"""
统计仓库准确率
原代码：与晗       mape准确率
改编1：冯凯       增加一段时间的准确度，仓库记分
改编2：结琴       增加mse,mae,rmse 等指标，改造spark 语法，能适配alg_dh 的python代码，能物理机运行，增加bad case分析报告

回刷历史 仓库准确率
from projects.basic_predict_promote.predict_wh_accuracy import *
for dt in pd.date_range('2023-01-01', pred_minus1_day):
    dt = dt.strftime('%Y-%m-%d')
    run_cal_acc(dt)

生成数据
bip3("metric","thirty_days_month_acc_by_all")                                                    月份++三十天++准确率等级
bip3("metric","thirty_days_month_acc_by_wh")                                                     月份++三十天++仓库++准确率等级
bip3("metric","thirty_days_month_acc_by_lc")                                                     月份++三十天++货物大类++准确率等级
bip3("metric","thirty_days_month_acc_by_goods")                                                  月份++三十天++货物++准确率等级
bip3("metric","thirty_days_month_acc_by_wh_goods")                                               月份++三十天++仓库++货物++准确率等级

bip3("process","two_week_acc_by_wh_goods")                                                       十四天++仓库+货物++准确率
bip3("process","two_week_acc_by_goods")                                                          十四天+货物++准确率
bip3("process","two_week_acc_by_lc")                                                             十四天+大类++准确率
bip3("process","two_week_acc_by_all")                                                            十四天+天维度++准确率
bip3("process","two_week_wh_com_type_goods_cost_rate")                                           十四天++仓库++类型++货物++消耗占比
bip3("process","two_week_wh_com_goods_cost_rate")                                                十四天++仓库++商品++货物++消耗占比


bip3("process","thirty_days_acc_by_wh_goods")                                                    三十天++仓库+货物++准确率
bip3("process","thirty_days_acc_by_goods")                                                       三十天+货物++准确率
bip3("process","thirty_days_acc_by_lc")                                                          三十天+大类++准确率
bip3("process","thirty_days_acc_by_all")                                                         三十天+天维度++准确率
bip3("process","thirty_days_wh_com_type_goods_cost_rate")                                        三十天++仓库++类型++货物++消耗占比
bip3("process","thirty_days_wh_com_goods_cost_rate")                                             三十天++仓库++商品++货物++消耗占比


bip3("process","sixty_days_acc_by_wh_goods")                                                    六十天++仓库+货物++准确率
bip3("process","sixty_days_acc_by_goods")                                                       六十天+货物++准确率
bip3("process","sixty_days_acc_by_lc")                                                          六十天+大类++准确率
bip3("process","sixty_days_acc_by_all")                                                         六十天+天维度++准确率
bip3("process","sixty_days_wh_com_type_goods_cost_rate")                                        六十天++仓库++类型++货物++消耗占比
bip3("process","sixty_days_wh_com_goods_cost_rate")                                             六十天++仓库++商品++货物++消耗占比

依赖数据

dw_ads_scm_alg.dev_luckin_demand_forecast_category_info1                                        货物范围
/projects/luckyml/purchase/demand_predict/{date_end}/demand_forecast_daily/consume_model/base/  实际消耗
/user/haoxuan.zou/demand_predict/{date_end}/delivery_model/base/                                实际出库

dw_ads_scm_alg.`dm_purchase_goods_demand`                                                       常规品预测值
dw_ads_scm_alg.`dm_wh_goods_demand_forecast_daily`                                              所有品预测值（常规+次新品+新品）
dw_dws.dws_stock_warehouse_stock_adjust_d_inc_summary                                           货物成本金额单价： 期末金额/期末货量
dw_ads_scm_alg.dim_warehouse_city_shop_d_his                                                    仓库-城市-门店关联信息
/user/yanxin.lu/sc/cmdty_plan/time/dt={dt}                                                      上市计划
dw_ads_scm_alg.da_wh_goods_cmdty_consume_d                                                      商品货物消耗占比
"""
from __init__ import project_path
import re
import os
import numpy as np
import pandas as pd
import seaborn as sns
import plotly as py
import plotly.express as px
import plotly.graph_objects as go
from areas.table_info.dh_dw_table_info import dh_dw
from utils.a91_wx_work_send import qiyeweixin_image
from utils_offline.a00_imports import dfu, log20 as log, DayStr, argv_date, read_api, shuttle, \
    bip3_save_df2, bip3
from z_config.cfg import DATA_OUTPUT_PATH

f"Import from: {project_path}"


def main_run_cal_acc(pred_calculation_day=None):
    # 每天跑
    run_cal_acc(pred_calculation_day)
    # 每月跑 指标
    # run_cal_acc_monthly(pred_calculation_day)
    # 每月跑 bad case html分析报告
    # monthly_bad_case_report(pred_calculation_day)
    # 每月跑 指标 发送企业微信图片
    # monthly_metric_send_img(pred_calculation_day)


def get_real_consume(dt, date_start, date_end):
    """
    真实消耗
    实际消耗部分：原料、轻食、零食
    """
    log.debug("ETL 货物范围")

    # 实际消耗部分：原料、轻食、零食
    sql_consume = f"""
                SELECT DISTINCT goods_id, is_formula, large_class_name
                FROM dw_ads_scm_alg.dev_luckin_demand_forecast_category_info1
                WHERE dt = '{date_start}' AND (large_class_id in (3,6,36) or (large_class_id = 4 and is_formula = 1))
            """
    goods_ls_consume = shuttle.query_dataq(sql_consume)

    # -----------------------
    # 实际消耗部分：办公用品, 器具类, 营销物料, 零件, 工服类, 日耗品
    sql_delivery = f"""
                SELECT DISTINCT goods_id, is_formula, large_class_name
                FROM dw_ads_scm_alg.dev_luckin_demand_forecast_category_info1
                WHERE dt = '{date_start}' AND (large_class_id  in (1,2,22,15,7,29) or (large_class_id = 4 and is_formula = 0))
            """
    goods_ls_delivery = shuttle.query_dataq(sql_delivery)

    # -----------------------
    log.debug("ETL 实际消耗")

    WH_CONSUME_MODEL_BASE = f'/projects/luckyml/purchase/demand_predict/{date_end}/demand_forecast_daily/consume_model/base/'
    actual_consume = read_api.read_one_folder(WH_CONSUME_MODEL_BASE)
    # sum
    actual_consume = (actual_consume
                      .query(f"'{date_start}'<= dt <='{date_end}'")
                      .groupby(['wh_dept_id', 'goods_id'], as_index=False)
                      .agg({'consume_origin': 'sum', 'consume': 'sum'})
                      .rename(columns={'consume': 'real_consume'})
                      .query("~goods_id.isnull()")
                      )
    # 数据格式
    actual_consume = dfu.df_col_to_numeric(actual_consume, ['goods_id', 'wh_dept_id'])
    actual_consume['real_consume'] = actual_consume['real_consume'].astype('float')
    # merge
    cols_actual = ['wh_dept_id', 'large_class_name', 'goods_id', 'real_consume']
    actual_consume = actual_consume.merge(goods_ls_consume, on=['goods_id'], how='inner')[cols_actual]

    # -----------------------
    log.debug("ETL 实际出库")

    WH_DELIVERY_MODEL_BASE = f"/user/haoxuan.zou/demand_predict/{date_end}/delivery_model/base/"
    wh_delivery = read_api.read_one_folder(WH_DELIVERY_MODEL_BASE)
    wh_delivery['dt'] = pd.to_datetime(wh_delivery['dt'])
    # sum
    wh_delivery = (wh_delivery
                   .query(f"'{date_start}'<= dt <='{date_end}'")
                   .groupby(['wh_dept_id', 'goods_id'], as_index=False).agg({'consume': 'sum'})
                   .rename(columns={'consume': 'real_consume'})
                   )
    # 数据格式
    wh_delivery = dfu.df_col_to_numeric(wh_delivery, ['goods_id', 'wh_dept_id'])
    wh_delivery['real_consume'] = wh_delivery['real_consume'].astype('float')
    # merge
    cols_delivery = ['wh_dept_id', 'large_class_name', 'goods_id', 'real_consume']
    wh_delivery = wh_delivery.merge(goods_ls_delivery, on=['goods_id'], how='inner')[cols_delivery]

    # concat
    df_real_consume_purchase = pd.concat([actual_consume, wh_delivery])

    return df_real_consume_purchase


def get_common_predict_dmd(dt, date_start, date_end):
    """
    常规品预测值
    """
    sql_common_pred = f"""
    select
        dt
        , wh_dept_id
        , goods_id
        , sum(demand) as monthly_common_pred
    from dw_ads_scm_alg.`dm_purchase_goods_demand`
    where dt = '{dt}'
      and predict_dt >= '{date_start}'
      and predict_dt <= '{date_end}'

    group by dt, wh_dept_id, goods_id
            """
    df_common_pred = shuttle.query_dataq(sql_common_pred)
    df_common_pred = dfu.df_col_to_numeric(df_common_pred, ['goods_id', 'wh_dept_id'])

    return df_common_pred


def get_all_predict_dmd(dt, date_start, date_end):
    """
    所有品预测值（常规+次新品+新品）
    """
    sql_all_pred = f"""
    select
        dt
      , wh_dept_id
      , goods_id
      , sum(demand) as monthly_all_pred
    from dw_ads_scm_alg.`dm_wh_goods_demand_forecast_daily`
    where
          dt = '{dt}'
      and predict_dt >= '{date_start}'
      and predict_dt <= '{date_end}'

    group by
        dt, wh_dept_id, goods_id
            """
    df_all_pred = shuttle.query_dataq(sql_all_pred)
    df_all_pred = dfu.df_col_to_numeric(df_all_pred, ['goods_id', 'wh_dept_id'])

    return df_all_pred


def get_goods_price(dt):
    """
    货物成本金额单价： 期末金额/期末货量
    """
    sql_price = f"""-- 仓，货物，钱
    SELECT
        wh_dept_id
        , goods_id
        , SUM(end_wh_stock_money) / SUM(end_wh_stock_cnt) AS unit_price
    FROM dw_dws.dws_stock_warehouse_stock_adjust_d_inc_summary
    WHERE dt >= DATE_SUB('{dt}', 180)
      AND spec_status = 1
      AND end_wh_stock_cnt > 0
    GROUP BY wh_dept_id, goods_id
        """
    price = shuttle.query_dataq(sql_price)
    return price


def dim_shop_city_warehouse_relation():
    """
    仓库-城市-门店关联信息
    """
    sql_wh_relation = '''
    SELECT DISTINCT 
           wh_dept_id
           , wh_name 
           , city_id
           , city_name
           , shop_dept_id
           , shop_name
    FROM dw_ads_scm_alg.dim_warehouse_city_shop_d_his
    WHERE dt = (SELECT MAX(dt) FROM dw_ads_scm_alg.dim_warehouse_city_shop_d_his) 
    
    '''
    wh_relation = shuttle.query_dataq(sql_wh_relation)
    return wh_relation


def get_predict_dt_plan(dt):
    """
    商品上市计划
    """
    three_week_ago = DayStr.n_day_delta(dt, n=-21)

    # -----------------------
    # 仓库-城市-门店关联信息
    wh_relation = dim_shop_city_warehouse_relation()
    wh_shop = wh_relation[['wh_dept_id', 'shop_dept_id']].rename(columns={'shop_dept_id': 'dept_id'})
    wh_shop = wh_relation[['wh_dept_id', 'shop_dept_id']].rename(columns={'shop_dept_id': 'dept_id'})

    # -----------------------
    # 商品上市计划
    df_plan = read_api.read_one_folder(f'/user/yanxin.lu/sc/cmdty_plan/time/dt={dt}')
    df_plan = df_plan.merge(wh_shop, on=['dept_id'], how='left')
    df_plan.launch_date = pd.to_datetime(df_plan.launch_date)
    df_plan.sale_date = pd.to_datetime(df_plan.sale_date)
    df_plan.actual_launch_date = pd.to_datetime(df_plan.actual_launch_date)

    # -----------------------
    # 常规品
    wh_plan_time = (df_plan
                    .groupby(['wh_dept_id', 'cmdty_id'], as_index=False)
                    .agg({'launch_date': 'median', 'sale_date': 'median', 'actual_launch_date': 'median'})
                    )
    wh_plan_time['cmdty_type'] = 'norm'

    # 新品
    sel_idx_new = ((wh_plan_time.launch_date > pd.to_datetime(dt)) & (wh_plan_time.sale_date.isnull()))
    wh_plan_time.loc[sel_idx_new, 'cmdty_type'] = 'new'

    # 次新品
    sel_idx_sub_new = ((wh_plan_time.sale_date > pd.to_datetime(three_week_ago)) & (
            wh_plan_time.sale_date <= pd.to_datetime(dt))) | (
                              (wh_plan_time.actual_launch_date > pd.to_datetime(three_week_ago)) & (
                              wh_plan_time.actual_launch_date <= pd.to_datetime(dt)))
    wh_plan_time.loc[sel_idx_sub_new, 'cmdty_type'] = 'subnew'

    return wh_plan_time


def get_common_ratio(dt, date_start, date_end, key):
    """
    商品货物消耗占比
    """
    # -----------------------
    # 仓库货物商品日售卖消耗数据
    sql_com_cost = f"""
        select
            goods_id
          , cmdty_id
          , cmdty_name
          , wh_dept_id
          , SUM(cmdty_cost_cnt)     AS cmdty_cost
        from dw_ads_scm_alg.da_wh_goods_cmdty_consume_d
        where dt >= '{date_start}'
          and dt <= '{date_end}'
        group by cmdty_id, cmdty_name,wh_dept_id, goods_id
        """
    com_cost = shuttle.query_dataq(sql_com_cost)

    # 商品上市计划
    wh_plan_time = get_predict_dt_plan(dt)

    # merge
    com_cost_type_mid = com_cost.merge(wh_plan_time, on=['wh_dept_id', 'cmdty_id'], how='left')
    # 补空
    com_cost_type_mid['cmdty_type'] = com_cost_type_mid['cmdty_type'].fillna('norm')
    com_cost_type_mid.cmdty_cost = com_cost_type_mid.cmdty_cost.fillna(0)

    # -----------------------
    # 仓库++商品++货物 sum
    com_cost_mid = (com_cost_type_mid
                    .groupby(['goods_id', 'wh_dept_id', 'cmdty_id'], as_index=False)
                    .agg({'cmdty_cost': 'sum'})
                    )

    # -----------------------
    # 仓库++类型++货物 sum
    com_type_cost_mid = (com_cost_type_mid
                         .groupby(['goods_id', 'wh_dept_id', 'cmdty_type'], as_index=False)
                         .agg({'cmdty_cost': 'sum'})
                         .rename(columns={'cmdty_cost': 'cmdty_type_cost'})
                         )

    # -----------------------
    # 仓库++货物 sum
    goods_cost_mid = (com_cost_type_mid
                      .groupby(['goods_id', 'wh_dept_id'], as_index=False)
                      .agg({'cmdty_cost': 'sum'})
                      .rename(columns={'cmdty_cost': 'cmdty_all_cost'})
                      )

    # merge
    com_ratio = com_cost_mid.merge(goods_cost_mid)
    com_ratio['cmdty_goods_ratio'] = round(com_ratio.cmdty_cost / com_ratio.cmdty_all_cost, 2)
    com_ratio['date_start'] = date_start
    com_ratio['date_end'] = date_end
    # save
    com_ratio.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    bip3_save_df2(com_ratio,
                  table_folder_name=f'{key}_wh_com_goods_cost_rate',
                  bip_folder='process',
                  output_name=f'{key}_wh_com_goods_cost_rate',
                  folder_dt=dt)
    # merge
    com_type_ratio = com_type_cost_mid.merge(goods_cost_mid)
    com_type_ratio['ratio'] = round(com_type_ratio.cmdty_type_cost / com_type_ratio.cmdty_all_cost, 2)
    # save
    bip3_save_df2(com_type_ratio,
                  table_folder_name=f'{key}_wh_com_type_goods_cost_rate',
                  bip_folder='process',
                  output_name=f'{key}_wh_com_type_goods_cost_rate',
                  folder_dt=dt)

    com_type_ratio = com_type_ratio.query("cmdty_type=='norm'").rename(columns={'ratio': 'common_ratio'})
    return com_type_ratio


def get_common_acc(dt, date_start, date_end, key):
    """
    常规品算法预测准确率
    """

    log.title(f"{key}常规品算法预测准确率 date_start={date_start} \n "
              f"date_start={date_end}")

    # -----------------------
    # 成本金额
    goods_price = get_goods_price(dt)
    # 货物消耗占比
    df_common_ratio = get_common_ratio(dt, date_start, date_end, key)
    # 实际消耗
    df_real_consume = get_real_consume(dt, date_start, date_end)
    # 常规品预测值
    df_common_pred = get_common_predict_dmd(dt, date_start, date_end)
    # 所有品预测值（常规+次新品+新品）
    df_all_pred = get_all_predict_dmd(dt, date_start, date_end)
    # 货物
    goods_info = dh_dw.dim_stock.goods_info()[['goods_id', 'goods_name']]

    # -----------------------
    # merge
    get_common_acc = (df_all_pred
                      .merge(df_common_pred, on=['dt', 'wh_dept_id', 'goods_id'], how='inner')
                      .merge(df_real_consume, on=['wh_dept_id', 'goods_id'], how='inner')
                      .fillna(0)
                      .merge(goods_price, on=['wh_dept_id', 'goods_id'], how='left')
                      .merge(goods_info, on=['goods_id'], how='left')
                      .merge(df_common_ratio[['wh_dept_id', 'goods_id', 'common_ratio']], on=['wh_dept_id', 'goods_id'],
                             how="left")
                      )

    # -----------------------
    # 计算
    get_common_acc['common_ratio'] = get_common_acc.apply(
        lambda x: 0 if ((x.monthly_all_pred > 0) & (x.monthly_common_pred == 0)) else x.common_ratio, axis=1)
    get_common_acc['common_ratio'] = get_common_acc['common_ratio'].fillna(1)
    get_common_acc['common_consume'] = round(get_common_acc['real_consume'] * get_common_acc['common_ratio'], 2)
    get_common_acc['all_consume_amount'] = round(get_common_acc['real_consume'] * get_common_acc['unit_price'], 2)
    get_common_acc['common_consume_amount'] = round(get_common_acc['common_consume'] * get_common_acc['unit_price'], 2)
    get_common_acc['common_predict_amount'] = round(
        get_common_acc['monthly_common_pred'] * get_common_acc['unit_price'], 2)
    get_common_acc['all_predict_amount'] = round(get_common_acc['monthly_all_pred'] * get_common_acc['unit_price'], 2)
    get_common_acc['diff_common'] = get_common_acc['common_predict_amount'] - get_common_acc['common_consume_amount']
    get_common_acc['diff_all'] = get_common_acc['all_predict_amount'] - get_common_acc['all_consume_amount']
    get_common_acc['abs_diff_common'] = np.abs(get_common_acc['diff_common'])
    get_common_acc['abs_diff_all'] = np.abs(get_common_acc['diff_all'])
    get_common_acc['DIFF_common'] = get_common_acc.diff_common ** 2
    get_common_acc['DIFF_all'] = get_common_acc.diff_all ** 2
    get_common_acc['MPE_common'] = round(
        (get_common_acc['monthly_common_pred'] - get_common_acc['common_consume']) / get_common_acc['common_consume'],
        2)
    get_common_acc['MAPE_common'] = np.abs(get_common_acc['MPE_common'])
    get_common_acc['MPE_all'] = round(
        (get_common_acc['all_predict_amount'] - get_common_acc['all_consume_amount']) / get_common_acc[
            'all_consume_amount'], 2)
    get_common_acc['MAPE_all'] = np.abs(get_common_acc['MPE_all'])

    # -----------------------
    # 长尾标识
    classified = read_api.read_one_folder(f"/user/haoxuan.zou/demand_predict/{dt}/delivery_model/classify")
    mask = classified.loc[classified.dly_model_type.isin(['f90', 'f180', 'f365'])] \
        .groupby(['goods_id'], as_index=False).agg({'wh_dept_id': 'count'}) \
        .query("wh_dept_id > 20")

    mask['is_longtail'] = 1
    mask = mask[['goods_id', 'is_longtail']]
    # 数据格式
    mask = dfu.df_col_to_numeric(mask, ['goods_id', 'is_longtail'])
    get_common_acc = (get_common_acc
                      .merge(mask, on=['goods_id'], how='left')
                      .fillna({'is_longtail': 0})
                      )
    # save
    get_common_acc.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

    # save
    bip3_save_df2(get_common_acc.query("large_class_name!=['零件']"),
                  table_folder_name=f'{key}_acc_by_wh_goods',
                  bip_folder='process',
                  output_name=f'{key}_acc_by_wh_goods',
                  folder_dt=dt)
    return get_common_acc


def get_acc_by_level(df):
    """
    汇总 算法预测准确率
    """
    # 筛选
    df_clean = df.query("large_class_name!= '零件' and (~unit_price.isnull())")
    df_clean['MAPE_all'] = np.abs(df_clean.MPE_all)
    common_mape_func = lambda x: x.abs_diff_common / x.common_consume_amount if x.common_consume_amount > 0 else np.nan
    all_mape_func = lambda x: x.abs_diff_all / x.all_consume_amount if x.all_consume_amount > 0 else np.nan

    # -----------------------
    # 货物维度
    cols_agg = {'all_consume_amount': 'sum', 'common_consume_amount': 'sum',
                'common_predict_amount': 'sum', 'all_predict_amount': 'sum', 'abs_diff_common': 'sum',
                'abs_diff_all': 'sum', 'is_longtail': 'median',
                'DIFF_common': 'sum', 'DIFF_all': 'sum', 'wh_dept_id': 'count'}
    df_goods = (df_clean.groupby(['dt', 'large_class_name', 'goods_name', 'goods_id'], as_index=False)
                .agg(cols_agg)
                )
    df_goods['mape_common'] = df_goods.apply(common_mape_func, axis=1)
    df_goods['mape_all'] = df_goods.apply(all_mape_func, axis=1)
    df_goods['is_longtail'] = df_goods.is_longtail.apply(lambda x: 1 if x > 0.5 else 0)

    df_goods['mse_common'] = df_goods.DIFF_common / df_goods.wh_dept_id
    df_goods['mse_all'] = df_goods.DIFF_all / df_goods.wh_dept_id

    df_goods['rmse_common'] = np.sqrt(df_goods.mse_common)
    df_goods['rmse_all'] = np.sqrt(df_goods.mse_all)

    df_goods['mae_common'] = df_goods.abs_diff_common / df_goods.wh_dept_id
    df_goods['mae_all'] = df_goods.abs_diff_all / df_goods.wh_dept_id

    cols_goods = ['dt', 'large_class_name', 'goods_name', 'goods_id', 'is_longtail',
                  'mape_common', 'mape_all', 'mse_common', 'mse_all', 'rmse_common', 'rmse_all', 'mae_common',
                  'mae_all']
    df_goods = df_goods[cols_goods]
    df_goods.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

    # -----------------------
    # 大类维度
    # 未去除长尾
    cols_agg = {'all_consume_amount': 'sum', 'common_consume_amount': 'sum',
                'common_predict_amount': 'sum', 'all_predict_amount': 'sum', 'abs_diff_common': 'sum',
                'abs_diff_all': 'sum', 'DIFF_common': 'sum', 'DIFF_all': 'sum', 'wh_dept_id': 'count'
                }
    df_large_class_long = (df_clean.groupby(['large_class_name', 'dt'], as_index=False)
                           .agg(cols_agg)
                           )
    df_large_class_long['mape_common'] = df_large_class_long.apply(common_mape_func, axis=1)
    df_large_class_long['mape_all'] = df_large_class_long.apply(all_mape_func, axis=1)

    df_large_class_long['mse_common'] = df_large_class_long.DIFF_common / df_large_class_long.wh_dept_id
    df_large_class_long['mse_all'] = df_large_class_long.DIFF_all / df_large_class_long.wh_dept_id

    df_large_class_long['rmse_common'] = np.sqrt(df_large_class_long.mse_common)
    df_large_class_long['rmse_all'] = np.sqrt(df_large_class_long.mse_all)

    df_large_class_long['mae_common'] = df_large_class_long.abs_diff_common / df_large_class_long.wh_dept_id
    df_large_class_long['mae_all'] = df_large_class_long.abs_diff_all / df_large_class_long.wh_dept_id

    # -----------------------
    # 去除长尾
    cols_agg = {'all_consume_amount': 'sum', 'common_consume_amount': 'sum',
                'common_predict_amount': 'sum', 'all_predict_amount': 'sum',
                'abs_diff_common': 'sum', 'abs_diff_all': 'sum',
                'DIFF_common': 'sum', 'DIFF_all': 'sum', 'wh_dept_id': 'count'
                }
    df_large_class_nolong = (df_clean.query("is_longtail==0")
                             .groupby(['large_class_name', 'dt'], as_index=False)
                             .agg(cols_agg)
                             )
    df_large_class_nolong['mape_common'] = df_large_class_nolong.apply(common_mape_func, axis=1)
    df_large_class_nolong['mape_all'] = df_large_class_nolong.apply(all_mape_func, axis=1)

    df_large_class_nolong['mse_common'] = df_large_class_nolong.DIFF_common / df_large_class_nolong.wh_dept_id
    df_large_class_nolong['mse_all'] = df_large_class_nolong.DIFF_all / df_large_class_nolong.wh_dept_id

    df_large_class_nolong['rmse_common'] = np.sqrt(df_large_class_nolong.mse_common)
    df_large_class_nolong['rmse_all'] = np.sqrt(df_large_class_nolong.mse_all)

    df_large_class_nolong['mae_common'] = df_large_class_nolong.abs_diff_common / df_large_class_nolong.wh_dept_id
    df_large_class_nolong['mae_all'] = df_large_class_nolong.abs_diff_all / df_large_class_nolong.wh_dept_id

    df_large_class = df_large_class_long.merge(df_large_class_nolong, on=['large_class_name', 'dt'],
                                               suffixes=['', '_nolongtail'])
    cols_large_class = ['large_class_name', 'dt',
                        'mape_common', 'mape_all',
                        'mse_common', 'mse_all',
                        'rmse_common', 'rmse_all',
                        'mae_common', 'mae_all',
                        'mape_common_nolongtail', 'mape_all_nolongtail',
                        'mse_common_nolongtail', 'mse_all_nolongtail',
                        'rmse_common_nolongtail', 'rmse_all_nolongtail',
                        'mae_common_nolongtail', 'mae_all_nolongtail'
                        ]
    df_large_class = df_large_class[cols_large_class]
    df_large_class.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

    # -----------------------
    # 月份维度
    cols_agg = {'all_consume_amount': 'sum', 'common_consume_amount': 'sum',
                'common_predict_amount': 'sum', 'all_predict_amount': 'sum', 'abs_diff_common': 'sum',
                'abs_diff_all': 'sum', 'DIFF_common': 'sum', 'DIFF_all': 'sum', 'wh_dept_id': 'count'
                }
    df_month_long = df_clean.groupby(['dt'], as_index=False).agg(cols_agg)
    df_month_long['mape_common'] = df_month_long.apply(common_mape_func, axis=1)
    df_month_long['mape_all'] = df_month_long.apply(all_mape_func, axis=1)

    df_month_long['mse_common'] = df_month_long.DIFF_common / df_month_long.wh_dept_id
    df_month_long['mse_all'] = df_month_long.DIFF_all / df_month_long.wh_dept_id

    df_month_long['rmse_common'] = np.sqrt(df_month_long.mse_common)
    df_month_long['rmse_all'] = np.sqrt(df_month_long.mse_all)

    df_month_long['mae_common'] = df_month_long.abs_diff_common / df_month_long.wh_dept_id
    df_month_long['mae_all'] = df_month_long.abs_diff_all / df_month_long.wh_dept_id

    cols_agg = {'all_consume_amount': 'sum', 'common_consume_amount': 'sum',
                'common_predict_amount': 'sum', 'all_predict_amount': 'sum', 'abs_diff_common': 'sum',
                'abs_diff_all': 'sum', 'DIFF_common': 'sum', 'DIFF_all': 'sum', 'wh_dept_id': 'count'
                }
    df_month_nolong = df_clean.query("is_longtail==0").groupby(['dt'], as_index=False).agg(cols_agg)
    df_month_nolong['mape_common'] = df_month_nolong.apply(common_mape_func, axis=1)
    df_month_nolong['mape_all'] = df_month_nolong.apply(all_mape_func, axis=1)

    df_month_nolong['mse_common'] = df_month_nolong.DIFF_common / df_month_nolong.wh_dept_id
    df_month_nolong['mse_all'] = df_month_nolong.DIFF_all / df_month_nolong.wh_dept_id

    df_month_nolong['rmse_common'] = np.sqrt(df_month_nolong.mse_common)
    df_month_nolong['rmse_all'] = np.sqrt(df_month_nolong.mse_all)

    df_month_nolong['mae_common'] = df_month_nolong.abs_diff_common / df_month_nolong.wh_dept_id
    df_month_nolong['mae_all'] = df_month_nolong.abs_diff_all / df_month_nolong.wh_dept_id

    df_month = df_month_long.merge(df_month_nolong, on=['dt'], suffixes=['', '_nolongtail'])
    cols_month = ['dt',
                  'mape_common', 'mape_all',
                  'mse_common', 'mse_all',
                  'rmse_common', 'rmse_all',
                  'mae_common', 'mae_all',
                  'mape_common_nolongtail', 'mape_all_nolongtail',
                  'mse_common_nolongtail', 'mse_all_nolongtail',
                  'rmse_common_nolongtail', 'rmse_all_nolongtail',
                  'mae_common_nolongtail', 'mae_all_nolongtail'
                  ]
    df_month = df_month[cols_month]
    df_month.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

    return df_goods, df_large_class, df_month


def run_cal_acc(pred_calculation_day=None):
    """
    回刷历史 仓库预测准确率
    from projects.basic_predict_promote.predict_wh_accuracy import *
    for dt in pd.date_range('2023-01-01', pred_minus1_day):
        dt = dt.strftime('%Y-%m-%d')
        run_cal_acc(dt)

    14天 仓库预测准确率
    30天 仓库预测准确率
    60天 仓库预测准确率
    """
    yesterday = DayStr.n_day_delta(pred_calculation_day, n=-1)
    two_week_bef = DayStr.n_day_delta(pred_calculation_day, n=-14)
    four_week_bef = DayStr.n_day_delta(pred_calculation_day, n=-30)
    sixty_days_bef = DayStr.n_day_delta(pred_calculation_day, n=-60)

    log.title(
        f"仓库预测准确率 yesterday={yesterday}  two_week_bef={two_week_bef}  four_week_bef={four_week_bef}  sixty_days_bef={sixty_days_bef} ")

    # -----------------------
    # 14天准确度
    acc_dt = DayStr.n_day_delta(two_week_bef, n=-1)
    two_week_acc = get_common_acc(acc_dt, two_week_bef, yesterday, 'two_week')
    # two_week_acc = read_api.read_one_folder(bip3("process","two_week_acc_by_wh_goods",acc_dt))
    two_week_goods, two_week_lc, two_week_month = get_acc_by_level(two_week_acc)

    save_dt = DayStr.n_day_delta(two_week_bef, n=-1)
    # save
    bip3_save_df2(two_week_goods,
                  table_folder_name=f'two_week_acc_by_goods',
                  bip_folder='process',
                  output_name=f'two_week_acc_by_goods',
                  folder_dt=save_dt)

    bip3_save_df2(two_week_lc,
                  table_folder_name=f'two_week_acc_by_lc',
                  bip_folder='process',
                  output_name=f'two_week_acc_by_lc',
                  folder_dt=save_dt)

    bip3_save_df2(two_week_month,
                  table_folder_name=f'two_week_acc_by_all',
                  bip_folder='process',
                  output_name=f'two_week_acc_by_all',
                  folder_dt=save_dt)

    # -----------------------
    # 30天准确度
    acc_dt = DayStr.n_day_delta(four_week_bef, n=-1)
    thirty_days_acc = get_common_acc(acc_dt, four_week_bef, yesterday, 'thirty_days')
    # thirty_days_acc = read_api.read_one_folder(bip3("process","thirty_days_acc_by_wh_goods",acc_dt))
    thirty_goods, thirty_lc, thirty_month = get_acc_by_level(thirty_days_acc)

    save_dt = DayStr.n_day_delta(four_week_bef, n=-1)
    # save
    bip3_save_df2(thirty_goods,
                  table_folder_name=f'thirty_days_acc_by_goods',
                  bip_folder='process',
                  output_name=f'thirty_days_acc_by_goods',
                  folder_dt=save_dt)

    bip3_save_df2(thirty_lc,
                  table_folder_name=f'thirty_days_acc_by_lc',
                  bip_folder='process',
                  output_name=f'thirty_days_acc_by_lc',
                  folder_dt=save_dt)

    bip3_save_df2(thirty_month,
                  table_folder_name=f'thirty_days_acc_by_all',
                  bip_folder='process',
                  output_name=f'thirty_days_acc_by_all',
                  folder_dt=save_dt)

    # -----------------------
    # 60天准确度
    acc_dt = DayStr.n_day_delta(sixty_days_bef, n=-1)
    sixty_days_acc = get_common_acc(acc_dt, sixty_days_bef, yesterday, 'sixty_days')
    # sixty_days_acc = read_api.read_one_folder(bip3("process","sixty_days_acc_by_wh_goods",acc_dt))
    sixty_days_goods, sixty_days_lc, sixty_days_month = get_acc_by_level(sixty_days_acc)

    save_dt = DayStr.n_day_delta(sixty_days_bef, n=-1)
    # save
    bip3_save_df2(sixty_days_goods,
                  table_folder_name=f'sixty_days_acc_by_goods',
                  bip_folder='process',
                  output_name=f'sixty_days_acc_by_goods',
                  folder_dt=save_dt)

    bip3_save_df2(sixty_days_lc,
                  table_folder_name=f'sixty_days_acc_by_lc',
                  bip_folder='process',
                  output_name=f'sixty_days_acc_by_lc',
                  folder_dt=save_dt)

    bip3_save_df2(sixty_days_month,
                  table_folder_name=f'sixty_days_acc_by_all',
                  bip_folder='process',
                  output_name=f'sixty_days_acc_by_all',
                  folder_dt=save_dt)


def run_cal_acc_monthly(pred_calculation_day=None):
    """
    取每月的最后一天的预测 ： 未来30天++仓库++货物++预测准确率
    开始：2023年1月 至今

    统计维度包括：
    1 每月准确率
    2 每月仓库准确率
    3 每月货物大类准确率
    4 每月货物准确率
    """
    pred_calc_day = DayStr.get_dt_str(pred_calculation_day)
    pred_minus1_day = DayStr.n_day_delta(pred_calc_day, n=-1)
    df_goods_info = dh_dw.dim_stock.goods_info()
    df_wh_info = dh_dw.dim_stock.wh_city_info()

    # -----------------------
    # 取每月的最后一天的月度预测
    import datetime
    from datetime import timedelta
    start_date = datetime.date(2023, 1, 1)
    today = datetime.date.today()
    previous_date = today.replace(day=1) - timedelta(days=1)
    date_ls = []
    month_ls = []
    current_date = start_date
    while current_date <= previous_date:
        next_month = current_date.replace(day=28) + datetime.timedelta(days=4)
        last_day = next_month - datetime.timedelta(days=next_month.day)
        date_ls.append(last_day.strftime('%Y-%m-%d'))
        month_ls.append((last_day.month % 12) + 1)
        current_date = next_month
    # 因2023-03-31缺数据，取最近的数据
    date_ls[2] = '2023-03-29'
    date_ls[7] = '2023-08-26'
    # 输出
    # date_ls = ['2023-01-31','2023-02-28','2023-03-29','2023-04-30','2023-05-31','2023-06-30','2023-07-30','2023-08-31']
    # month_ls = [2,3,4,5,6,7,8,9]

    # -----------------------
    # 统计 长周期货物 来源 admin物料管理池（运珠的线下excel）每月更新
    sql_material_pool = f"""
    SELECT 
    goods_id,long_period_cal_type
    FROM  
    lucky_cooperation.t_material_pool
    WHERE del=0
    """
    df_material_pool = shuttle.query_dataq(sql_material_pool)
    goods_long_purchase = df_material_pool['goods_id'].drop_duplicates().tolist()
    not_in_goods = ['冷冻调制血橙', '柚子复合果汁饮料浓浆']
    # -----------------------
    # 从今年开始，筛选长周期物料  仓库++货物++准确率
    df_predict_sum = pd.DataFrame()
    for date, month in zip(date_ls, month_ls):
        # df_predict = read_api.read_dt_folder(bip3("process", "thirty_days_acc_by_wh_goods"), date)
        df_predict = read_api.read_one_file(
            f"/user/yuhan.cui/demand_predict/acc_by_level/thirty_days/{date}/acc_by_wh_goods.csv")
        df_predict['month'] = month
        df_predict_sum = pd.concat([df_predict_sum, df_predict])
    df_predict_sum = df_predict_sum.query(f"goods_id in {goods_long_purchase} and goods_name !={not_in_goods}")
    cols = ['wh_dept_id', 'goods_id', 'MAPE_all', 'month']
    df_predict_sum = df_predict_sum[cols]

    # -----------------------
    # mape准确度分级： [0，0.2, 0.4, 0.6, 0.8, 1]
    wh_name = df_wh_info.drop_duplicates(['wh_dept_id', 'wh_name'])
    df_predict_sum['pe20'] = df_predict_sum['MAPE_all'].apply(lambda x: 1 if x <= 0.2 else 0)
    df_predict_sum['pe40'] = df_predict_sum['MAPE_all'].apply(lambda x: 1 if (x > 0.2 and x <= 0.4) else 0)
    df_predict_sum['pe60'] = df_predict_sum['MAPE_all'].apply(lambda x: 1 if (x > 0.4 and x <= 0.6) else 0)
    df_predict_sum['pe80'] = df_predict_sum['MAPE_all'].apply(lambda x: 1 if (x > 0.6 and x <= 0.8) else 0)
    df_predict_sum['pe100'] = df_predict_sum['MAPE_all'].apply(lambda x: 1 if (x > 0.8 and x <= 1) else 0)
    df_predict_sum['pe_more'] = df_predict_sum['MAPE_all'].apply(lambda x: 1 if x > 1 else 0)
    df_predict_sum = (df_predict_sum
                      .merge(df_goods_info[['goods_id', 'goods_name', 'large_class_id', 'large_class_name']])
                      .merge(wh_name)
                      )

    # -----------------------
    # 统计 月度  mape在 20% 的次数
    cols_agg = {'pe20': 'sum', 'pe40': 'sum', 'pe60': 'sum', 'pe80': 'sum', 'pe100': 'sum', 'pe_more': 'sum'}
    accuracy_levels = ['pe20', 'pe40', 'pe60', 'pe80', 'pe100', 'pe_more']

    df_result_all = (df_predict_sum.groupby('month')
                     .agg(cols_agg).reset_index())
    # 总次数
    df_result_sum = (df_predict_sum.groupby('month')
                     .size().rename("total").reset_index())
    df_result_all = df_result_all.merge(df_result_sum)

    # 占比 =  mape 在 20% 的次数/ 总次数
    for level in accuracy_levels:
        df_result_all[f'{level}_ratio'] = (df_result_all[level] / df_result_all['total']).round(4)

    # 输出
    cols_result = ['month', 'total',
                   'pe20', 'pe40', 'pe60', 'pe80', 'pe100', 'pe_more',
                   'pe20_ratio', 'pe40_ratio', 'pe60_ratio', 'pe80_ratio', 'pe100_ratio', 'pe_more_ratio']
    df_result_all = df_result_all[cols_result]
    df_result_all = dfu.df_col_to_numeric(df_result_all, df_result_all.columns)

    # -----------------------
    # 统计 月度 仓库  mape在 20% 的次数
    df_result_wh = (df_predict_sum.groupby(['month', 'wh_dept_id', 'wh_name'])
                    .agg(cols_agg).reset_index())
    # 总次数
    df_result_wh_sum = (df_predict_sum.groupby(['month', 'wh_dept_id', 'wh_name'])
                        .size().rename("total").reset_index())
    df_result_wh = df_result_wh.merge(df_result_wh_sum)

    # 占比 =  mape 在 20% 的次数/ 总次数

    for level in accuracy_levels:
        df_result_wh[f'{level}_ratio'] = (df_result_wh[level] / df_result_wh['total']).round(4)

    # 输出
    cols_result = ['month', 'wh_dept_id', 'wh_name', 'total',
                   'pe20', 'pe40', 'pe60', 'pe80', 'pe100', 'pe_more',
                   'pe20_ratio', 'pe40_ratio', 'pe60_ratio', 'pe80_ratio', 'pe100_ratio', 'pe_more_ratio']
    df_result_wh = df_result_wh[cols_result]
    cols_tmp_str = ["wh_name"]
    df_result_wh = dfu.df_col_to_numeric(df_result_wh, [cc for cc in df_result_wh.columns if cc not in cols_tmp_str])

    # -----------------------
    # 统计 月度 货物大类  mape在 20% 的次数
    df_result_lc = (df_predict_sum.groupby(['month', 'large_class_id', 'large_class_name'])
                    .agg(cols_agg).reset_index())
    # 总次数
    df_result_lc_sum = (df_predict_sum.groupby(['month', 'large_class_id', 'large_class_name'])
                        .size().rename("total").reset_index())
    df_result_lc = df_result_lc.merge(df_result_lc_sum)

    # 占比 =  mape 在 20% 的次数/ 总次数
    for level in accuracy_levels:
        df_result_lc[f'{level}_ratio'] = (df_result_lc[level] / df_result_lc['total']).round(4)

    # 输出
    cols_result = ['month', 'large_class_id', 'large_class_name', 'total',
                   'pe20', 'pe40', 'pe60', 'pe80', 'pe100', 'pe_more',
                   'pe20_ratio', 'pe40_ratio', 'pe60_ratio', 'pe80_ratio', 'pe100_ratio', 'pe_more_ratio']
    df_result_lc = df_result_lc[cols_result]
    cols_tmp_str = ["large_class_name"]
    df_result_lc = dfu.df_col_to_numeric(df_result_lc, [cc for cc in df_result_lc.columns if cc not in cols_tmp_str])

    # -----------------------
    # 统计 月度 仓库  mape在 20% 的次数
    df_result_goods = (df_predict_sum.groupby(['month', 'goods_id', 'goods_name'])
                       .agg(cols_agg).reset_index())
    # 总次数
    df_result_goods_sum = (df_predict_sum.groupby(['month', 'goods_id', 'goods_name'])
                        .size().rename("total").reset_index())
    df_result_goods = df_result_goods.merge(df_result_goods_sum)

    # 占比 =  mape 在 20% 的次数/ 总次数

    for level in accuracy_levels:
        df_result_goods[f'{level}_ratio'] = (df_result_goods[level] / df_result_goods['total']).round(4)

    # 输出
    cols_result = ['month',  'goods_id', 'goods_name', 'total',
                   'pe20', 'pe40', 'pe60', 'pe80', 'pe100', 'pe_more',
                   'pe20_ratio', 'pe40_ratio', 'pe60_ratio', 'pe80_ratio', 'pe100_ratio', 'pe_more_ratio']
    df_result_goods = df_result_goods[cols_result]
    cols_tmp_str = ["goods_name"]
    df_result_goods = dfu.df_col_to_numeric(df_result_goods, [cc for cc in df_result_goods.columns if cc not in cols_tmp_str])

    # -----------------------
    # 中文
    change_names_result = {'total': '总预测数',
                           'pe20': '绝对误差百分比(0~0.2)预测数',
                           'pe40': '绝对误差百分比(0.2~0.4)预测数',
                           'pe60': '绝对误差百分比小于(0.4~0.6)预测数',
                           'pe80': '绝对误差百分比小于(0.6~0.8)预测数',
                           'pe100': '绝对误差百分比(0.8~1)预测数',
                           'pe_more': '绝对误差百分比(1~)预测数',

                           'pe20_ratio': '(0~0.2)',
                           'pe40_ratio': '(0.2~0.4)',
                           'pe60_ratio': '(0.4~0.6)',
                           'pe80_ratio': '(0.6~0.8)',
                           'pe100_ratio': '(0.8~1)',
                           'pe_more_ratio': '(1~)',
                           }
    df_result_all_cn = df_result_all.rename(columns=change_names_result)
    df_result_lc_cn = df_result_lc.rename(columns=change_names_result)
    df_result_wh_cn = df_result_wh.rename(columns=change_names_result)
    df_result_goods_cn = df_result_goods.rename(columns=change_names_result)

    # save
    bip3_save_df2(df_predict_sum,
                  table_folder_name=f'thirty_days_month_acc_by_wh_goods',
                  bip_folder='metric',
                  output_name=f'thirty_days_month_acc_by_wh_goods',
                  folder_dt=pred_minus1_day)

    bip3_save_df2(df_result_all_cn,
                  table_folder_name=f'thirty_days_month_acc_by_all',
                  bip_folder='metric',
                  output_name=f'thirty_days_month_acc_by_all',
                  folder_dt=pred_minus1_day)

    # save
    bip3_save_df2(df_result_lc_cn,
                  table_folder_name=f'thirty_days_month_acc_by_lc',
                  bip_folder='metric',
                  output_name=f'thirty_days_month_acc_by_lc',
                  folder_dt=pred_minus1_day)

    # save
    bip3_save_df2(df_result_wh_cn,
                  table_folder_name=f'thirty_days_month_acc_by_wh',
                  bip_folder='metric',
                  output_name=f'thirty_days_month_acc_by_wh',
                  folder_dt=pred_minus1_day)
    # save
    bip3_save_df2(df_result_goods_cn,
                  table_folder_name=f'thirty_days_month_acc_by_goods',
                  bip_folder='metric',
                  output_name=f'thirty_days_month_acc_by_goods',
                  folder_dt=pred_minus1_day)


def monthly_bad_case_report(pred_calculation_day=None):
    """
    仓库预测准确率bad case分析报告
    输出HTML分析报告路径： /home/dinghuo/data_output/htmls/wh_bad_case
    """
    pred_calc_day = DayStr.get_dt_str(pred_calculation_day)
    pred_minus1_day = DayStr.n_day_delta(pred_calc_day, n=-1)

    # ------------------------------
    from datetime import date, timedelta
    today = date.today()
    previous_month = today.replace(day=1) - timedelta(days=1)
    previous_previous_month = previous_month.replace(day=1) - timedelta(days=1)
    # 计算上一个月的月份
    target_month = previous_month.month
    # target_month = today.month
    # 计算上上一个月的最后一天
    acc_dt = previous_previous_month.replace(day=previous_previous_month.day).strftime('%Y-%m-%d')
    # acc_dt = '2023-08-26'
    log.title(f"仓库预测准确率bad case分析报告 target_month={target_month}  target_dt={acc_dt}")

    # ------------------------------
    # 月份++仓库++货物++指标
    df_result_error_cn = read_api.read_dt_folder(
        bip3("metric", "thirty_days_month_acc_by_wh_goods"), pred_minus1_day)
    cols_error = ['wh_dept_id', 'wh_name', 'large_class_id', 'large_class_name', 'goods_id', 'goods_name', 'MAPE_all']
    #  筛选 MAPE大于0.4 作为bad case
    dfe1 = df_result_error_cn.query(f" MAPE_all >0.4 and month=={target_month}")[cols_error]
    accuracy_levels = ['pe20', 'pe40', 'pe60', 'pe80', 'pe100', 'pe_more']
    dfe1['accuracy_label'] = pd.cut(dfe1['MAPE_all'],
                                    bins=[-np.inf, 0.2, 0.4, 0.6, 0.8, 1, np.inf],
                                    labels=accuracy_levels)
    dfe1 = dfu.df_col_to_numeric(dfe1, ['wh_dept_id', 'large_class_id', 'goods_id'])
    dfe1['accuracy_label'] = dfe1['accuracy_label'].astype('str')

    # ------------------------------
    # 取上上一个月的最后一天的预测
    thirty_days_acc = read_api.read_one_file(
        f"/user/yuhan.cui/demand_predict/acc_by_level/thirty_days/{acc_dt}/acc_by_wh_goods.csv")
    # thirty_days_acc = read_api.read_one_folder(bip3("process", "thirty_days_acc_by_wh_goods", acc_dt))
    cols_acc = ['dt', 'wh_dept_id', 'goods_id', 'monthly_all_pred', 'real_consume']
    df_acc = thirty_days_acc[cols_acc]

    # ------------------------------
    # merge 月指标++日预测
    cols_round = {'MAPE_all': 2, 'monthly_all_pred': 2, 'real_consume': 2}
    dfe1 = (dfe1.merge(df_acc)
            .round(cols_round)
            .sort_values(['accuracy_label', 'MAPE_all'])
            )
    dfe1['wh_goods'] = dfe1['wh_name'] + '_' + dfe1['goods_name']

    # ------------------------------
    # 绝对误差百分比 大于1
    cols_e1 = ['wh_goods', 'MAPE_all', 'monthly_all_pred', 'real_consume']
    change_name_e1 = {'monthly_all_pred': '预测消耗', 'real_consume': '实际消耗'}
    df_pe_more = (dfe1.query("accuracy_label =='pe_more'")
                  [cols_e1].rename(columns=change_name_e1)
                  .reset_index(drop=True)
                  )
    df_pe_more['差异'] = df_pe_more.eval("预测消耗-实际消耗 ").round(1)

    # ------------------------------
    # 绝对误差百分比 大于0.8 小于1
    df_pe100 = (dfe1.query("accuracy_label =='pe100'")
                [cols_e1].rename(columns=change_name_e1)
                .reset_index(drop=True)
                )
    df_pe100['差异'] = df_pe100.eval("预测消耗-实际消耗 ").round(1)

    # ------------------------------
    # 绝对误差百分比 大于0.6  小于0.8
    df_pe80 = (dfe1.query("accuracy_label =='pe80'")
               [cols_e1].rename(columns=change_name_e1)
               .reset_index(drop=True)
               )
    df_pe80['差异'] = df_pe80.eval("预测消耗-实际消耗 ").round(1)

    # ------------------------------
    # 绝对误差百分比 大于0.4  小于0.6
    df_pe60 = (dfe1.query("accuracy_label =='pe60'")
               [cols_e1].rename(columns=change_name_e1)
               .reset_index(drop=True)
               )
    df_pe60['差异'] = df_pe60.eval("预测消耗-实际消耗 ").round(1)

    # ------------------------------
    # 货物大类 准确率
    df_result_lc_cn = read_api.read_dt_folder(
        bip3("metric", "thirty_days_month_acc_by_lc"))
    df_result_lc_cn = df_result_lc_cn[
        ['month', 'large_class_name', '(0~0.2)', '(0.2~0.4)', '(0.4~0.6)', '(0.6~0.8)', '(0.8~1)', '(1~)']]
    df_result_lc_cn_month = df_result_lc_cn.query(f"month =={target_month}").reset_index(drop=True)

    #  仓库 准确率
    df_result_wh_cn = read_api.read_dt_folder(
        bip3("metric", "thirty_days_month_acc_by_wh"))
    df_result_wh_cn = df_result_wh_cn[
        ['month', 'wh_name', '(0~0.2)', '(0.2~0.4)', '(0.4~0.6)', '(0.6~0.8)', '(0.8~1)', '(1~)']]
    df_result_wh_cn_month = df_result_wh_cn.query(f"month =={target_month}").reset_index(drop=True)

    #  货物 准确率 (选收入金额前20的商品)
    ld_price = read_api.read_dt_folder(
        bip3('sell', 'nation_goods__sell_income'))
    ld_price = ld_price.head(20)[['goods_id', 'cmdty_income']]
    df_result_goods = read_api.read_dt_folder(
        bip3("metric", "thirty_days_month_acc_by_goods"))
    df_result_goods = df_result_goods.merge(ld_price).sort_values(by=['month', 'cmdty_income'],ascending=[True, False])
    df_result_goods.rename(columns={'绝对误差百分比(0~0.2)预测数':'(0~0.2)仓库数'},inplace=True)
    df_result_goods_cn = df_result_goods[
        ['month', 'goods_name', '(0~0.2)', '(0.2~0.4)', '(0.4~0.6)', '(0.6~0.8)', '(0.8~1)', '(1~)']]
    df_result_goods_cn_month = df_result_goods_cn.query(f"month =={target_month}").reset_index(drop=True)

    # 月份全国 准确率
    df_result_all_cn = read_api.read_dt_folder(
        bip3("metric", "thirty_days_month_acc_by_all"))
    df_result_all_cn = df_result_all_cn[['month', '(0~0.2)', '(0.2~0.4)', '(0.4~0.6)', '(0.6~0.8)', '(0.8~1)', '(1~)']]

    # ------------------------------
    # 临时分析报告模版
    import jinja2
    html_path = os.path.join(DATA_OUTPUT_PATH, 'htmls/wh_bad_case/')
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath=html_path))
    template = env.get_template('Template.html')
    # 标题
    summary1 = u"月度"
    summary2 = u"货物大类"
    summary3 = u"仓库"
    summary8 = u"Top20收入金额货物"

    # 输出数据表格
    table1 = u"后面来制作数据表格"
    table2 = u"后面来制作数据表格"
    table3 = u"后面来制作数据表格"
    table8 = u"后面来制作数据表格"

    # 输出图表
    chart1 = u"后面来制作图片"
    chart2 = u"后面来制作图片"
    chart3 = u"后面来制作图片"
    chart8 = u"后面来制作图片"
    chart9 = u"后面来制作图片"
    html = template.render(summary1=summary1, table1=table1,
                           summary2=summary2, table2=table2,
                           summary3=summary3, table3=table3,
                           summary8=summary8, table8=table8,
                           chart1=chart1, chart2=chart2, chart3=chart3, chart8=chart8,chart9=chart9)
    html_path = os.path.join(DATA_OUTPUT_PATH, f"htmls/wh_bad_case/Report_{acc_dt}.html")
    with open(html_path, 'w') as f:
        f.write(html)

    # ------------------------------
    # 最终分析报告模版

    # 准确率
    cols = ['(0~0.2)', '(0.2~0.4)', '(0.4~0.6)', '(0.6~0.8)', '(0.8~1)', '(1~)']
    cm = sns.light_palette("green", as_cmap=True)
    # 月度
    styled_df1 = df_result_all_cn.style.background_gradient(cmap=cm, subset=cols).format("{:.2%}", subset=cols)

    # 货物大类
    styled_df2 = df_result_lc_cn_month.style.background_gradient(cmap=cm, subset=cols).format("{:.2%}", subset=cols)
    # 仓库
    styled_df3 = df_result_wh_cn_month.style.background_gradient(cmap=cm, subset=cols).format("{:.2%}", subset=cols)

    # 货物
    styled_df8 = df_result_goods_cn_month.style.background_gradient(cmap=cm, subset=cols).format("{:.2%}", subset=cols)

    # ------------
    # 异常案例
    cols_e = ['MAPE_all', '预测消耗', '实际消耗', '差异']
    styled_df4 = df_pe_more.style.bar('差异', vmin=0).background_gradient(cmap=cm, subset='MAPE_all').format("{:.2f}",
                                                                                                           subset=cols_e)
    styled_df5 = df_pe100.style.bar('差异', vmin=0).background_gradient(cmap=cm, subset='MAPE_all').format("{:.2f}",
                                                                                                         subset=cols_e)
    styled_df6 = df_pe80.style.bar('差异', vmin=0).background_gradient(cmap=cm, subset='MAPE_all').format("{:.2f}",
                                                                                                        subset=cols_e)
    styled_df7 = df_pe60.tail(10).style.bar('差异', vmin=0).background_gradient(cmap=cm, subset='MAPE_all').format(
        "{:.2f}", subset=cols_e)

    table1 = styled_df1.to_html()
    table2 = styled_df2.to_html()
    table3 = styled_df3.to_html()

    table4 = styled_df4.to_html()
    table5 = styled_df5.to_html()
    table6 = styled_df6.to_html()
    table7 = styled_df7.to_html()
    table8 = styled_df8.to_html()
    # ---------------
    # 准确率图
    fig1 = px.bar(df_result_all_cn, x='month', y='(0~0.2)', text='(0~0.2)',
                  color_discrete_sequence=['rgb(128, 192, 128)'])
    fig1.update_traces(texttemplate='%{text:.2%}', textposition='inside')
    chart1 = py.offline.plot(fig1, include_plotlyjs=True, output_type='div')

    fig2 = px.bar(df_result_lc_cn_month, x='large_class_name', y='(0~0.2)',
                  text='(0~0.2)', color_discrete_sequence=['rgb(128, 192, 128)'])
    fig2.update_traces(texttemplate='%{text:.2%}', textposition='inside')
    chart2 = py.offline.plot(fig2, include_plotlyjs=True, output_type='div')

    fig3 = px.bar(df_result_wh_cn_month, x='wh_name', y='(0~0.2)',
                  text='(0~0.2)', color_discrete_sequence=['rgb(128, 192, 128)'])
    fig3.update_traces(texttemplate='%{text:.2%}', textposition='inside')
    chart3 = py.offline.plot(fig3, include_plotlyjs=True, output_type='div')

    fig8 = px.bar(df_result_goods_cn_month, x='goods_name', y='(0~0.2)',
                  text='(0~0.2)', color_discrete_sequence=['rgb(128, 192, 128)'])
    fig8.update_traces(texttemplate='%{text:.2%}', textposition='inside')
    chart8 = py.offline.plot(fig8, include_plotlyjs=True, output_type='div')

    fig9 = px.bar(df_result_goods, facet_row='goods_name', x='month', y='(0~0.2)仓库数', text='(0~0.2)仓库数',
                  color_discrete_sequence=['rgb(128, 192, 128)'])
    fig9.update_traces(textposition='inside')
    fig9.update_layout(height=3000)
    chart9 = py.offline.plot(fig9, include_plotlyjs=True, output_type='div')

    # ---------------
    # 异常图
    fig4 = px.bar(df_pe_more,
                  x='wh_goods', y=['预测消耗', '实际消耗'],
                  barmode='group',
                  color_discrete_sequence=['rgb(128, 192, 128)', 'rgb(0,139,69)']
                  )
    fig4.update_xaxes(title='')
    fig4.update_yaxes(title='')
    chart4 = py.offline.plot(fig4, include_plotlyjs=True, output_type='div')
    fig5 = px.bar(df_pe100,
                  x='wh_goods', y=['预测消耗', '实际消耗'],
                  barmode='group',
                  color_discrete_sequence=['rgb(128, 192, 128)', 'rgb(0,139,69)']
                  )
    fig5.update_xaxes(title='')
    fig5.update_yaxes(title='')
    chart5 = py.offline.plot(fig5, include_plotlyjs=True, output_type='div')
    fig6 = px.bar(df_pe80,
                  x='wh_goods', y=['预测消耗', '实际消耗'],
                  barmode='group',
                  color_discrete_sequence=['rgb(128, 192, 128)', 'rgb(0,139,69)']
                  )
    fig6.update_xaxes(title='')
    fig6.update_yaxes(title='')
    chart6 = py.offline.plot(fig6, include_plotlyjs=True, output_type='div')

    fig7 = px.bar(df_pe60,
                  x='wh_goods', y=['预测消耗', '实际消耗'],
                  barmode='group',
                  color_discrete_sequence=['rgb(128, 192, 128)', 'rgb(0,139,69)']
                  )
    fig7.update_xaxes(title='')
    fig7.update_yaxes(title='')
    chart7 = py.offline.plot(fig7, include_plotlyjs=True, output_type='div')

    html = template.render(summary1=summary1, table1=table1,
                           summary2=summary2, table2=table2,
                           summary3=summary3, table3=table3,
                           summary8=summary8, table8=table8,
                           chart1=chart1, chart2=chart2, chart3=chart3, chart8=chart8,chart9=chart9,
                           table4=table4, table5=table5,
                           table6=table6, table7=table7,
                           chart4=chart4, chart5=chart5, chart6=chart6, chart7=chart7
                           )
    with open(html_path, 'w') as f:
        f.write(html)


def monthly_metric_send_img(pred_calculation_day=None):
    """
    仓库预测准确率 指标 发送企业微信图片
    输出图片路径： /home/dinghuo/data_output/images/base_pred
    """
    pred_calc_day = DayStr.get_dt_str(pred_calculation_day)
    pred_minus1_day = DayStr.n_day_delta(pred_calc_day, n=-1)

    # ------------------------------
    from datetime import date, timedelta
    today = date.today()
    previous_month = today.replace(day=1) - timedelta(days=1)
    previous_previous_month = previous_month.replace(day=1) - timedelta(days=1)
    # 计算上一个月的月份
    target_month = previous_month.month
    # target_month = today.month
    # 计算上上一个月的最后一天
    acc_dt = previous_previous_month.replace(day=previous_previous_month.day).strftime('%Y-%m-%d')
    # acc_dt='2023-08-26'
    log.title(f"仓库预测准确率bad case分析报告 target_month={target_month}  target_dt={acc_dt}")
    # ------------------------------
    # 货物大类 准确率
    df_result_lc_cn = read_api.read_dt_folder(
        bip3("metric", "thirty_days_month_acc_by_lc"))
    df_result_lc_cn = df_result_lc_cn[
        ['month', 'large_class_name', '(0~0.2)', '(0.2~0.4)', '(0.4~0.6)', '(0.6~0.8)', '(0.8~1)', '(1~)']]
    df_result_lc_cn_month = df_result_lc_cn.query(f"month =={target_month}").reset_index(drop=True)

    #  仓库 准确率
    df_result_wh_cn = read_api.read_dt_folder(
        bip3("metric", "thirty_days_month_acc_by_wh"))
    df_result_wh_cn = df_result_wh_cn[
        ['month', 'wh_name', '(0~0.2)', '(0.2~0.4)', '(0.4~0.6)', '(0.6~0.8)', '(0.8~1)', '(1~)']]
    df_result_wh_cn_month = df_result_wh_cn.query(f"month =={target_month}").reset_index(drop=True)

    # 月份全国 准确率
    df_result_all_cn = read_api.read_dt_folder(
        bip3("metric", "thirty_days_month_acc_by_all"))
    df_result_all_cn = df_result_all_cn[['month', '(0~0.2)', '(0.2~0.4)', '(0.4~0.6)', '(0.6~0.8)', '(0.8~1)', '(1~)']]

    metric_img(df_plot=df_result_all_cn, title_text='月度预测准确率', is_send=True, send_to=3)
    metric_img(df_plot=df_result_wh_cn_month, title_text='月度仓库预测准确率', is_send=True, send_to=3)
    metric_img(df_plot=df_result_lc_cn_month, title_text='月度货物大类预测准确率', is_send=True, send_to=3)


def metric_img(df_plot=None, title_text=None, is_send=True, send_to=3):
    s = len(df_plot)
    n_rows = len(re.findall("<Br>", title_text))
    top_height = 35 + 20 * n_rows
    h = 100 + top_height + 35 * (s - 1)
    plot_column_names = [f'<b>{x}</b>' for x in df_plot.columns]
    plot_column_width = [25] * len(df_plot.columns)

    column_color_dict = {
        '(0~0.2)': df_plot['(0~0.2)'],
        '(0.2~0.4)': df_plot['(0.2~0.4)'],
        '(0.4~0.6)': df_plot['(0.4~0.6)'],
        '(0.6~0.8)': df_plot['(0.6~0.8)'],
        '(0.8~1)': df_plot['(0.8~1)'],
        '(1~)': df_plot['(1~)']
    }

    column_color_ls = []
    for col in df_plot.columns:
        if col in column_color_dict:
            column_color_ls.append([apply_color_img(x) for x in column_color_dict[col]])
        else:
            column_color_ls.append(['#ffffff'] * len(df_plot))

    # Convert values to percentage format
    table_values = []
    for col in df_plot.columns:
        if col == 'month':
            table_values.append(df_plot[col].tolist())
        elif col == 'wh_name':
            table_values.append(df_plot[col].tolist())
        elif col == 'large_class_name':
            table_values.append(df_plot[col].tolist())
        else:
            table_values.append([f'{x:.2%}' for x in df_plot[col].tolist()])

    fig = go.Figure(data=[go.Table(
        columnwidth=plot_column_width,
        header=dict(
            values=plot_column_names,
            line_color='white',
            fill_color='white',
            font=dict(color='black', size=12),
            height=40
        ),
        cells=dict(
            values=table_values,
            line_color='white',
            fill=dict(color=column_color_ls),
            font_size=12,
            height=35
        )
    )], layout=go.Layout(
        autosize=True,
        margin={'l': 5, 'r': 5, 't': top_height, 'b': 0},
        title=f"<b>{title_text}</b>",
    ))

    # SAVE
    print("save")
    save_path = os.path.join(DATA_OUTPUT_PATH, f"images/base_pred/{title_text}.png")
    fig.write_image(save_path, width=sum(plot_column_width) * 6 + 50, height=h, scale=2, engine='kaleido')

    if is_send:
        qiyeweixin_image(save_path, send_to=send_to)


def apply_color_img(value):
    """
    列的渐变填充，从深绿到浅绿
    """
    if value == 0:
        color = '#ffffff'  # 白色
    elif 0 < value <= 0.1:
        color_range = [
            'rgb(104, 205, 104)',  # 深绿色
            'rgb(114, 210, 114)',
            'rgb(124, 215, 124)',
            'rgb(134, 220, 134)',
            'rgb(144, 225, 144)',
            'rgb(154, 230, 154)',
            'rgb(164, 235, 164)',
            'rgb(174, 240, 174)',
            'rgb(184, 245, 184)',
            'rgb(194, 250, 194)',
            'rgb(204, 255, 204)'  # 浅绿色
        ]

        index = int(value * 100)  # 根据值计算索引
        index = max(0, min(index, len(color_range) - 1))
        color = color_range[index]
    else:
        color_range = [
            'rgb(204, 255, 204)',  # 浅绿色
            'rgb(194, 250, 194)',
            'rgb(184, 245, 184)',
            'rgb(174, 240, 174)',
            'rgb(164, 235, 164)',
            'rgb(154, 230, 154)',
            'rgb(144, 225, 144)',
            'rgb(134, 220, 134)',
            'rgb(124, 215, 124)',
            'rgb(114, 210, 114)',
            'rgb(104, 205, 104)'  # 深绿色
        ]
        index = int((value - 0.1) * 10)  # 根据值计算索引
        index = max(0, min(index, len(color_range) - 1))
        color = color_range[index]

    return color


if __name__ == '__main__':
    argv_date(main_run_cal_acc)
