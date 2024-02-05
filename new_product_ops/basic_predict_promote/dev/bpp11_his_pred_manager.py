# encoding: utf-8
# @created: 2023/10/24
# @author: jieqin.lin
# @file: projects/basic_predict_promote/dev/bpp11_his_pred_manager.py


"""

生成数据

# 评选最佳模型
bip3("model/basic_predict_promote", "best_model_mix_1_wh_goods_pred_normal")                         过去1个月（仓库货物）最佳模型（acc均值最大）
bip3("model/basic_predict_promote", "best_model_mix_2_wh_goods_pred_normal")                         过去2个月（仓库货物）最佳模型（acc均值最大）
bip3("model/basic_predict_promote", "best_model_mix_3_wh_goods_pred_normal")                         过去3个月（仓库货物）最佳模型（acc均值最大）

bip3("model/basic_predict_promote", "best_model_mix_1_wh_goods_pred_all")                         过去1个月（仓库货物）最佳模型（acc均值最大）
bip3("model/basic_predict_promote", "best_model_mix_2_wh_goods_pred_all")                         过去2个月（仓库货物）最佳模型（acc均值最大）
bip3("model/basic_predict_promote", "best_model_mix_3_wh_goods_pred_all")                         过去3个月（仓库货物）最佳模型（acc均值最大）

# 预测
bip3("model/basic_predict_promote", "mix_1_wh_goods_pred_normal")                         常规用过去1个月最佳模型预测的结果
bip3("model/basic_predict_promote", "mix_2_wh_goods_pred_normal")                         常规用过去2个月最佳模型预测的结果
bip3("model/basic_predict_promote", "mix_3_wh_goods_pred_normal")                         常规用过去3个月最佳模型预测的结果

bip3("model/basic_predict_promote", "mix_1_wh_goods_pred_all")                            总用过去1个月最佳模型预测的结果
bip3("model/basic_predict_promote", "mix_2_wh_goods_pred_all")                            总用过去1个月最佳模型预测的结果
bip3("model/basic_predict_promote", "mix_3_wh_goods_pred_all")                            总用过去1个月最佳模型预测的结果


# 日维度 准确度
bip3("model/basic_predict_promote", "metric_daily_mix_1_wh_goods_pred_normal")            常规品天维度准确度
bip3("model/basic_predict_promote", "metric_daily_goods_mix_1_wh_goods_pred_normal")      常规品天维度货物级别准确度
bip3("model/basic_predict_promote", "metric_daily_wh_goods_mix_1_wh_goods_pred_normal")   常规品天维度仓库货物级别准确度

bip3("model/basic_predict_promote", "metric_daily_mix_1_wh_goods_pred_all)                总天维度准确度
bip3("model/basic_predict_promote", "metric_daily_goods_mix_1_wh_goods_pred_all)          总天维度货物级别准确度
bip3("model/basic_predict_promote", "metric_daily_wh_goods_mix_1_wh_goods_pred_all)       总天维度仓库货物级别准确度

# 月维度 准确度
bip3("model/basic_predict_promote", "metric_month_mix_1_wh_goods_pred_normal")            常规品月维度准确度
bip3("model/basic_predict_promote", "metric_month_goods_mix_1_wh_goods_pred_normal")      常规品月维度货物级别准确度
bip3("model/basic_predict_promote", "metric_month_wh_goods_mix_1_wh_goods_pred_normal")   常规品月维度仓库货物级别准确度

bip3("model/basic_predict_promote", "metric_month_mix_1_wh_goods_pred_all")               总月维度准确度
bip3("model/basic_predict_promote", "metric_month_goods_mix_1_wh_goods_pred_all")         总月维度货物级别准确度
bip3("model/basic_predict_promote", "metric_month_wh_goods_mix_1_wh_goods_pred_all")      总月维度仓库货物级别准确度

依赖数据
bip3("model/basic_predict_promote", "xgb_goods_last_1_acc_normal")                          过去1个月（仓库货物）acc均值最大
bip3("model/basic_predict_promote", "lgbm_goods_last_1_acc_normal")                         过去1个月（仓库货物）acc均值最大
bip3("model/basic_predict_promote", "catboost_goods_last_1_acc_normal")                     过去1个月（仓库货物）acc均值最大
bip3("model/basic_predict_promote", "holt_winters_goods_last_1_acc_normal")                 过去1个月（仓库货物）acc均值最大
bip3("model/basic_predict_promote", "online_goods_last_1_acc_normal")                       过去1个月（仓库货物）acc均值最大
bip3("model/basic_predict_promote", "model_goods_last_1_acc_norm")                         过去1个月（仓库货物）acc均值最大


"""

import pandas as pd

from __init__ import project_path
from areas.table_info.dh_dw_table_info import dh_dw
from projects.basic_predict_promote.dev.bpp12_his_pred_tree import (get_price,
                                                                    change_and_evaluate, save_pred_df, save_metric_df,
                                                                    get_his_true_value)
from utils_offline.a00_imports import DayStr, read_api, bip3_save_df2, bip3

f"Import from {project_path}"


def main_mix_mode():
    """
    回刷历史预测
    """
    df_goods_info = dh_dw.dim_stock.goods_info()
    # ----------------------------------
    target_goods_name = ['冰凉感厚椰饮品', '冷萃厚牛乳', '北海道丝绒风味厚乳']
    date_ls = ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01', '2023-06-01', '2023-07-01',
               '2023-08-01', '2023-09-01']
    sel_goods_ls = df_goods_info.query(f"goods_name == {target_goods_name}")['goods_id'].unique().tolist()
    model_ls = ['mix_1', 'mix_2', 'mix_3','mix_boost_1', 'mix_boost_2', 'mix_boost_3']
    data_label_ls = ['normal', 'all']
    for data_label in data_label_ls:
        for model_label in model_ls:
            for target_date in date_ls:
                main_train_mix_wh_goods_model(target_date=target_date,
                                              model_label=model_label,
                                              data_label=data_label,
                                              sel_goods_ls=sel_goods_ls,
                                              )


def thread_pool_mix_wh_goods_mode():
    """
    多线程回刷历史预测
    """
    import concurrent.futures
    import itertools
    date_ls = ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01', '2023-06-01', '2023-07-01',
               '2023-08-01', '2023-09-01']
    df_goods_info = dh_dw.dim_stock.goods_info()
    target_goods_name = ['冰凉感厚椰饮品', '冷萃厚牛乳', '北海道丝绒风味厚乳']
    sel_goods_ls = df_goods_info.query(f"goods_name in {target_goods_name}")['goods_id'].unique().tolist()
    model_ls = ['mix_1', 'mix_2', 'mix_3','mix_boost_1', 'mix_boost_2', 'mix_boost_3']
    data_label_ls = ['normal', 'all']

    def process_combination(combination):
        target_date, model_label, data_label = combination
        print(f"Processing: {target_date}, {model_label}, {data_label}")

        try:
            main_train_mix_wh_goods_model(target_date=target_date,
                                          model_label=model_label,
                                          data_label=data_label,
                                          sel_goods_ls=sel_goods_ls
                                          )
        except Exception as e:
            print(f"Failed: {str(e)}")

    combinations = itertools.product(date_ls, model_ls, data_label_ls)

    with concurrent.futures.ThreadPoolExecutor(max_workers=9) as executor:
        executor.map(process_combination, combinations)


def thread_pool_mix_goods_mode():
    """
    多线程回刷历史预测
    """
    import concurrent.futures
    import itertools
    date_ls = ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01', '2023-06-01', '2023-07-01',
               '2023-08-01', '2023-09-01']
    df_goods_info = dh_dw.dim_stock.goods_info()
    target_goods_name = ['冰凉感厚椰饮品', '冷萃厚牛乳', '北海道丝绒风味厚乳']
    sel_goods_ls = df_goods_info.query(f"goods_name in {target_goods_name}")['goods_id'].unique().tolist()
    model_ls = ['mix_goods_1', 'mix_goods_2', 'mix_goods_3','mix_boost_goods_1', 'mix_boost_goods_2', 'mix_boost_goods_3']

    data_label_ls = ['normal', 'all']

    def process_combination(combination):
        target_date, model_label, data_label = combination
        print(f"Processing: {target_date}, {model_label}, {data_label}")

        try:
            main_train_mix_goods_model(target_date=target_date,
                                       model_label=model_label,
                                       data_label=data_label,
                                       sel_goods_ls=sel_goods_ls
                                       )
        except Exception as e:
            print(f"Failed: {str(e)}")

    combinations = itertools.product(date_ls, model_ls, data_label_ls)

    with concurrent.futures.ThreadPoolExecutor(max_workers=9) as executor:
        executor.map(process_combination, combinations)


def main_train_mix_wh_goods_model(target_date, model_label, data_label, sel_goods_ls):
    day_minus1 = DayStr.n_day_delta(target_date, n=-1)
    day_30 = DayStr.n_day_delta(target_date, n=29)
    pred_calc_day = DayStr.get_dt_str(None)
    pred_minus1_day = DayStr.n_day_delta(pred_calc_day, n=-1)
    df_price = get_price(pred_minus1_day)
    # ------------------
    # 过去三个月准确率
    if model_label == 'mix_1':
        window = 1
    if model_label == 'mix_2':
        window = 2
    if model_label == 'mix_3':
        window = 3
    if model_label == 'mix_boost_1':
        window = 1
    if model_label == 'mix_boost_2':
        window = 2
    if model_label == 'mix_boost_3':
        window = 3
    df1, df2, df3, df4, df5, df6 = get_last_acc(data_label=data_label, window=window)
    # concat
    df = pd.concat([df1, df2, df3, df4, df5, df6])

    if model_label in ['mix_boost_1','mix_boost_2','mix_boost_3']:
        df.query("model in ['xgb','lgbm','catboost']", inplace=True)

    # ------------------
    # 最佳模型
    df_max = df.groupby(['dt', 'wh_dept_id', 'goods_id'])[f'last_{window}_acc'].max().reset_index()
    df_max_model = df_max.merge(df)[['dt', 'wh_dept_id', 'goods_id', 'model', f'last_{window}_acc']]
    df_max_model.sort_values(['dt', 'wh_dept_id', 'goods_id'], inplace=True)
    df_max_model['dt'] = pd.to_datetime(df_max_model['dt'])

    df_best_model = pd.DataFrame()

    for wh_id in df_max_model["wh_dept_id"].unique():
        for g_id in df_max_model.query(f"wh_dept_id == {wh_id}")["goods_id"].unique():
            df_wh_goods = (df_max_model.query(f"wh_dept_id == {wh_id} and  goods_id == {g_id}")
                           .reset_index(drop=True).copy()).sort_values('dt')
            df_wh_goods['best_model'] = df_wh_goods['model'].shift(1)
            df_best_model = pd.concat([df_best_model, df_wh_goods])
    df_best_model = df_best_model.drop(['model'], axis=1).rename(columns={'best_model': 'model'})

    bip3_save_df2(df_best_model,
                  table_folder_name=f'best_model_{model_label}_wh_goods_pred_{data_label}',
                  bip_folder='model/basic_predict_promote',
                  output_name=f'best_model_{model_label}_wh_goods_pred_{data_label}',
                  folder_dt=day_minus1)

    # ------------------
    # 预测
    df1_pred, df2_pred, df3_pred, df4_pred, df5_pred, df6_pred = get_pred(data_label=data_label, dt=day_minus1)
    # concat
    df_daily_pred = pd.concat([df1_pred, df2_pred, df3_pred, df4_pred, df5_pred, df6_pred])
    df_daily_pred.rename(columns={'predict_demand': 'pred', 'predict_dt': 'ds'}, inplace=True)
    df_daily_pred["ds"] = pd.to_datetime(df_daily_pred["ds"])
    df_daily_pred["dt"] = pd.to_datetime(df_daily_pred["dt"])

    # ------------------
    # 混合预测
    df_daily_pred = df_best_model.merge(df_daily_pred)

    df_true = get_his_true_value(data_label, sel_goods_ls, dt=day_30)
    dfc_metric = df_daily_pred.merge(df_true)
    df_month_pred = dfc_metric.groupby(['wh_dept_id', 'goods_id']).agg({'y': 'sum', 'pred': 'sum'}).round(
        1).reset_index()

    daily_all_metric, daily_goods_metric, daily_wh_goods_metric = change_and_evaluate(df=dfc_metric, df_price=df_price,
                                                                                      model_label=model_label,
                                                                                      dt=day_minus1)
    month_all_metric, month_goods_metric, month_wh_goods_metric = change_and_evaluate(df=df_month_pred,
                                                                                      df_price=df_price,
                                                                                      model_label=model_label,
                                                                                      dt=day_minus1)
    save_pred_df(df_daily_pred=df_daily_pred, df_month_pred=df_month_pred, model_label=model_label,
                 data_label=data_label, dt=day_minus1)
    save_metric_df(m_all_df=daily_all_metric, m_goods_df=daily_goods_metric, m_wh_goods_df=daily_wh_goods_metric,
                   model_label=model_label, data_label=data_label, time_label='daily', dt=day_minus1)
    save_metric_df(m_all_df=month_all_metric, m_goods_df=month_goods_metric, m_wh_goods_df=month_wh_goods_metric,
                   model_label=model_label, data_label=data_label, time_label='month', dt=day_minus1)


def main_train_mix_goods_model(target_date, model_label, data_label, sel_goods_ls):
    day_minus1 = DayStr.n_day_delta(target_date, n=-1)
    day_30 = DayStr.n_day_delta(target_date, n=29)
    pred_calc_day = DayStr.get_dt_str(None)
    pred_minus1_day = DayStr.n_day_delta(pred_calc_day, n=-1)
    df_price = get_price(pred_minus1_day)
    # ------------------
    # 过去三个月准确率
    if model_label == 'mix_goods_1':
        window = 1
    if model_label == 'mix_goods_2':
        window = 2
    if model_label == 'mix_goods_3':
        window = 3
    if model_label == 'mix_boost_goods_1':
        window = 1
    if model_label == 'mix_boost_goods_2':
        window = 2
    if model_label == 'mix_boost_goods_3':
        window = 3
    df1, df2, df3, df4, df5, df6 = get_goods_last_acc(data_label=data_label, window=window)
    # concat
    df = pd.concat([df1, df2, df3, df4, df5, df6])

    if model_label in ['mix_boost_goods_1','mix_boost_goods_2','mix_boost_goods_3']:
        df.query("model in ['xgb','lgbm','catboost']", inplace=True)
    # ------------------
    # 最佳模型
    df_max = df.groupby(['dt', 'goods_id'])[f'last_{window}_acc'].max().reset_index()
    df_max_model = df_max.merge(df)[['dt', 'goods_id', 'model', f'last_{window}_acc']]
    df_max_model.sort_values(['dt', 'goods_id'], inplace=True)
    df_max_model['dt'] = pd.to_datetime(df_max_model['dt'])

    df_best_model = pd.DataFrame()

    for g_id in df_max_model["goods_id"].unique():
        df_goods = (df_max_model.query(f"goods_id == {g_id}")
                    .reset_index(drop=True).copy()).sort_values('dt')
        df_goods['best_model'] = df_goods['model'].shift(1)
        df_best_model = pd.concat([df_best_model, df_goods])
    df_best_model = df_best_model.drop(['model'], axis=1).rename(columns={'best_model': 'model'})

    bip3_save_df2(df_best_model,
                  table_folder_name=f'best_model_goods_{model_label}_wh_goods_pred_{data_label}',
                  bip_folder='model/basic_predict_promote',
                  output_name=f'best_model_goods_{model_label}_wh_goods_pred_{data_label}',
                  folder_dt=day_minus1)

    # ------------------
    # 预测
    df1_pred, df2_pred, df3_pred, df4_pred, df5_pred, df6_pred = get_pred(data_label=data_label, dt=day_minus1)
    # concat
    df_daily_pred = pd.concat([df1_pred, df2_pred, df3_pred, df4_pred, df5_pred, df6_pred])
    df_daily_pred.rename(columns={'predict_demand': 'pred', 'predict_dt': 'ds'}, inplace=True)
    df_daily_pred["ds"] = pd.to_datetime(df_daily_pred["ds"])
    df_daily_pred["dt"] = pd.to_datetime(df_daily_pred["dt"])

    # ------------------
    # 混合预测
    df_true = get_his_true_value(data_label, sel_goods_ls, dt=day_30)
    dfc_metric = df_best_model.merge(df_daily_pred).merge(df_true)
    df_month_pred = dfc_metric.groupby(['wh_dept_id', 'goods_id']).agg({'y': 'sum', 'pred': 'sum'}).round(
        1).reset_index()

    daily_all_metric, daily_goods_metric, daily_wh_goods_metric = change_and_evaluate(df=dfc_metric, df_price=df_price,
                                                                                      model_label=model_label,
                                                                                      dt=day_minus1)
    month_all_metric, month_goods_metric, month_wh_goods_metric = change_and_evaluate(df=df_month_pred,
                                                                                      df_price=df_price,
                                                                                      model_label=model_label,
                                                                                      dt=day_minus1)
    save_pred_df(df_daily_pred=df_daily_pred, df_month_pred=df_month_pred, model_label=model_label,
                 data_label=data_label, dt=day_minus1)
    save_metric_df(m_all_df=daily_all_metric, m_goods_df=daily_goods_metric, m_wh_goods_df=daily_wh_goods_metric,
                   model_label=model_label, data_label=data_label, time_label='daily', dt=day_minus1)
    save_metric_df(m_all_df=month_all_metric, m_goods_df=month_goods_metric, m_wh_goods_df=month_wh_goods_metric,
                   model_label=model_label, data_label=data_label, time_label='month', dt=day_minus1)


def get_last_acc(data_label, window):

    cols_output = ['wh_dept_id', 'goods_id', 'model', f'last_{window}_acc', 'dt']
    df1 = read_api.read_dt_folder(
        bip3("model/basic_predict_promote", f"xgb_last_{window}_acc_{data_label}"))[cols_output]

    df2 = read_api.read_dt_folder(
        bip3("model/basic_predict_promote", f"lgbm_last_{window}_acc_{data_label}"))[cols_output]

    df3 = read_api.read_dt_folder(
        bip3("model/basic_predict_promote", f"catboost_last_{window}_acc_{data_label}"))[cols_output]

    df4 = read_api.read_dt_folder(
        bip3("model/basic_predict_promote", f"holt_winters_last_{window}_acc_{data_label}"))[cols_output]

    if data_label == 'all':
        df5 = read_api.read_dt_folder(
            bip3("model/basic_predict_promote", f"model_last_{window}_acc_{data_label}"))[cols_output]
    if data_label == 'normal':
        df5 = read_api.read_dt_folder(
            bip3("model/basic_predict_promote", f"model_last_{window}_acc_norm"))[cols_output]

    df6 = read_api.read_dt_folder(
        bip3("model/basic_predict_promote", f"online_last_{window}_acc_{data_label}"))[cols_output]

    return df1, df2, df3, df4, df5, df6


def get_goods_last_acc(data_label, window):

    cols_output = ['goods_id', 'model', f'last_{window}_acc', 'dt']
    df1 = read_api.read_dt_folder(
        bip3("model/basic_predict_promote", f"xgb_goods_last_{window}_acc_{data_label}"))[cols_output]

    df2 = read_api.read_dt_folder(
        bip3("model/basic_predict_promote", f"lgbm_goods_last_{window}_acc_{data_label}"))[cols_output]

    df3 = read_api.read_dt_folder(
        bip3("model/basic_predict_promote", f"catboost_goods_last_{window}_acc_{data_label}"))[cols_output]

    df4 = read_api.read_dt_folder(
        bip3("model/basic_predict_promote", f"holt_winters_goods_last_{window}_acc_{data_label}"))[cols_output]


    if data_label == 'all':
        df5 = read_api.read_dt_folder(
            bip3("model/basic_predict_promote", f"model_goods_last_{window}_acc_{data_label}"))[cols_output]
    if data_label == 'normal':
        df5 = read_api.read_dt_folder(
            bip3("model/basic_predict_promote", f"model_goods_last_{window}_acc_norm"))[cols_output]

    df6 = read_api.read_dt_folder(
        bip3("model/basic_predict_promote", f"online_goods_last_{window}_acc_{data_label}"))[cols_output]

    return df1, df2, df3, df4, df5, df6


def get_pred(data_label, dt):
    df1_pred = read_api.read_dt_folder(
        bip3("model/basic_predict_promote", f"xgb_wh_goods_pred_{data_label}"), dt)

    df2_pred = read_api.read_dt_folder(
        bip3("model/basic_predict_promote", f"lgbm_wh_goods_pred_{data_label}"), dt)

    df3_pred = read_api.read_dt_folder(
        bip3("model/basic_predict_promote", f"catboost_wh_goods_pred_{data_label}"), dt)

    df4_pred = read_api.read_dt_folder(
        bip3("model/basic_predict_promote", f"holt_winters_wh_goods_pred_{data_label}"), dt)

    df5_pred = read_api.read_dt_folder(
        bip3("model/basic_predict_promote", f"model_summary_{data_label}"), dt)

    df6_pred = read_api.read_dt_folder(
        bip3("model/basic_predict_promote", f"online_wh_goods_pred_{data_label}"), dt)

    return df1_pred, df2_pred, df3_pred, df4_pred, df5_pred, df6_pred
