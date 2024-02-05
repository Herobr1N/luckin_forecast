# encoding: utf-8
# @created: 2023/10/19
# @author: jieqin.lin
# @file: projects/basic_predict_promote/dev/bpp14_pred_smoothing.py


"""

生成数据
############################################
总预测过去3个月准确率均值
bip3("model/basic_predict_promote", "holt_winters_last_3_acc_all")

常规预测过去3个月准确率均值
bip3("model/basic_predict_promote", "holt_winters_last_3_acc_normal")

############################################

三次指数平滑 holt_winters

# 预测
bip3("model/basic_predict_promote", "holt_winters_wh_goods_pred_normal")                         预测常规消耗量
bip3("model/basic_predict_promote", "holt_winters_wh_goods_pred_all")                            预测总消耗量

# 参数
bip3("model/basic_predict_promote", "train_metric_holt_winters_wh_goods_pred_normal")            常规品训练集准确度
bip3("model/basic_predict_promote", "train_metric_holt_winters_wh_goods_pred_all")               总训练集准确度

# 训练
bip3("model/basic_predict_promote", "best_params_holt_winters_wh_goods_pred_normal")            常规品训练集准确度
bip3("model/basic_predict_promote", "best_params_holt_winters_wh_goods_pred_all")               总训练集准确度

# 天维度
bip3("model/basic_predict_promote", "metric_daily_holt_winters_wh_goods_pred_normal")            常规品天维度准确度
bip3("model/basic_predict_promote", "metric_daily_goods_holt_winters_wh_goods_pred_normal")      常规品天维度货物级别准确度
bip3("model/basic_predict_promote", "metric_daily_wh_goods_holt_winters_wh_goods_pred_normal")   常规品天维度仓库货物级别准确度

bip3("model/basic_predict_promote", "metric_daily_holt_winters_wh_goods_pred_all)                总天维度准确度
bip3("model/basic_predict_promote", "metric_daily_goods_holt_winters_wh_goods_pred_all)          总天维度货物级别准确度
bip3("model/basic_predict_promote", "metric_daily_wh_goods_holt_winters_wh_goods_pred_all)       总天维度仓库货物级别准确度

# 月度
bip3("model/basic_predict_promote", "metric_month_holt_winters_wh_goods_pred_normal")            常规品月维度准确度
bip3("model/basic_predict_promote", "metric_month_goods_holt_winters_wh_goods_pred_normal")      常规品月维度货物级别准确度
bip3("model/basic_predict_promote", "metric_month_wh_goods_holt_winters_wh_goods_pred_normal")   常规品月维度仓库货物级别准确度

bip3("model/basic_predict_promote", "metric_month_holt_winters_wh_goods_pred_all")               总月维度准确度
bip3("model/basic_predict_promote", "metric_month_goods_holt_winters_wh_goods_pred_all")         总月维度货物级别准确度
bip3("model/basic_predict_promote", "metric_month_wh_goods_holt_winters_wh_goods_pred_all")      总月维度仓库货物级别准确度

依赖数据
bip3("model/basic_predict_promote", "stock_wh_goods_type_flg_theory_sale_cnt")
bip3("model/basic_predict_promote", "stock_wh_goods_theory_sale_cnt")
bip3("model/basic_predict_promote", "feature_engineering_normal")
bip3("model/basic_predict_promote", "feature_engineering_all")
dw_dws.dws_stock_warehouse_stock_adjust_d_inc_summary
"""

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from sklearn.model_selection import ParameterGrid
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from __init__ import project_path
from areas.table_info.dh_dw_table_info import dh_dw
from projects.basic_predict_promote.dev.bpp12_his_pred_tree import (get_feature, get_price,
                                                                    change_and_evaluate, save_metric_df,
                                                                    save_pred_df, save_train_df, last_wh_goods_acc,
                                                                    last_goods_acc)
from utils_offline.a00_imports import DayStr, bip3_save_df2, read_api, bip3

f"Import from {project_path}"


def main():
    model_ls = ['holt_winters']

    # thread_pool_train_and_evaluate_models()
    main_train_and_evaluate_models()

    last_wh_goods_acc(model_ls=model_ls, window=6)
    last_wh_goods_acc(model_ls=model_ls, window=3)
    last_wh_goods_acc(model_ls=model_ls, window=2)
    last_wh_goods_acc(model_ls=model_ls, window=1)

    last_goods_acc(model_ls=model_ls, window=6)
    last_goods_acc(model_ls=model_ls, window=3)
    last_goods_acc(model_ls=model_ls, window=2)
    last_goods_acc(model_ls=model_ls, window=1)


def main_train_and_evaluate_models():
    """
    回刷历史预测
    """
    df_goods_info = dh_dw.dim_stock.goods_info()
    # ----------------------------------

    date_ls = ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01', '2023-06-01', '2023-07-01',
               '2023-08-01', '2023-09-01']
    target_goods_name = ['冰凉感厚椰饮品', '冷萃厚牛乳', '北海道丝绒风味厚乳']
    sel_goods_ls = df_goods_info.query(f"goods_name == {target_goods_name}")['goods_id'].unique().tolist()
    model_ls = ['holt_winters']
    data_label_ls = ['normal', 'all']
    window = 30
    for data_label in data_label_ls:
        for model_label in model_ls:
            for target_date in date_ls:
                print(target_date)
                # 找最佳参数
                # best_params_holt_winters(target_date, model_label, data_label, sel_goods_ls, window)
                # 训练模型
                train_and_evaluate_holt_winters(target_date, model_label, data_label, sel_goods_ls, window)

def thread_pool_train_and_evaluate_models():
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
    data_label_ls = ['normal', 'all']
    window = 30
    model_ls = ['holt_winters']

    def process_combination(combination):
        target_date, model_label, data_label = combination
        print(f"Processing: {target_date}, {model_label}, {data_label}")

        try:
            # best_params_holt_winters(target_date, model_label, data_label, sel_goods_ls, window)
            train_and_evaluate_holt_winters(target_date, model_label, data_label, sel_goods_ls, window)
        except Exception as e:
            print(f"Failed: {str(e)}")

    combinations = itertools.product(date_ls, model_ls, data_label_ls)

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        executor.map(process_combination, combinations)


def best_params_holt_winters(target_date, model_label, data_label, sel_goods_ls, window):
    """
    三次指数平滑
    """
    start_dt = '2020-01-01'
    day_minus1 = DayStr.n_day_delta(target_date, n=-1)
    pred_calc_day = DayStr.get_dt_str(None)
    pred_minus1_day = DayStr.n_day_delta(pred_calc_day, n=-1)
    df_price = get_price(pred_minus1_day)

    ld_feature = get_feature(data_label=data_label, sel_goods_ls=sel_goods_ls)
    df_feature = ld_feature.query(f"'{start_dt}' <= ds < '{target_date}'")

    best_params_dfs = []

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [
            executor.submit(process_best_params_combination, df_feature, df_price, data_label, model_label, wh_id, g_id,
                            window, day_minus1)
            for wh_id in df_feature["wh_dept_id"].unique()
            for g_id in df_feature.query(f"wh_dept_id == {wh_id}")["goods_id"].unique()
        ]

        for future in futures:
            result = future.result()
            best_params_dfs.append(result[0])

    best_params_df = pd.concat(best_params_dfs)

    save_best_params_df(best_params_df=best_params_df, model_label=model_label, data_label=data_label, dt=day_minus1)


def process_best_params_combination(df_feature, df_price, data_label, model_label,
                                    wh_id, g_id, window, day_minus1
                                    ):
    df_wh_goods = (df_feature.query(f"wh_dept_id == {wh_id} and  goods_id == {g_id}")
                   .reset_index(drop=True).copy())
    print(f"Processing: {len(df_wh_goods)}，{wh_id}，{g_id}，{data_label}")
    if len(df_wh_goods) > 2:
        # df转成序列
        data = pd.Series(df_wh_goods['y'].values, index=pd.DatetimeIndex(df_wh_goods['ds']))
        # 找最佳参数
        best_params = find_best_parameters_parallel(data=data, df_price=df_price, model_label=model_label,
                                                    dt=day_minus1,
                                                    wh_id=wh_id, g_id=g_id, window=window)
        return pd.DataFrame([best_params])
    else:
        return pd.DataFrame()


def train_and_evaluate_holt_winters(target_date, model_label, data_label, sel_goods_ls, window):
    """
    三次指数平滑
    """
    start_dt = '2020-01-01'
    day_minus1 = DayStr.n_day_delta(target_date, n=-1)
    pred_calc_day = DayStr.get_dt_str(None)
    pred_minus1_day = DayStr.n_day_delta(pred_calc_day, n=-1)
    df_price = get_price(pred_minus1_day)

    ld_feature = get_feature(data_label=data_label, sel_goods_ls=sel_goods_ls)
    df_feature = ld_feature.query(f"'{start_dt}' <= ds < '{target_date}'")

    train_dfs = []
    df_daily_preds = []

    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [
            executor.submit(process_wh_goods_combination, df_feature, data_label, model_label, wh_id, g_id,
                            window, target_date, day_minus1)
            for wh_id in df_feature["wh_dept_id"].unique()
            for g_id in df_feature.query(f"wh_dept_id == {wh_id}")["goods_id"].unique()
        ]

        for future in futures:
            result = future.result()
            train_dfs.append(result[0])
            df_daily_preds.append(result[1])

    train_df = pd.concat(train_dfs)
    df_daily_pred = pd.concat(df_daily_preds).dropna().reset_index(drop=True)
    df_daily_pred['ds'] = pd.to_datetime(df_daily_pred['ds'])

    cols_feature = ['ds', 'wh_dept_id', 'goods_id', 'y']
    dfc_metric = ld_feature[cols_feature].merge(df_daily_pred).dropna()
    train_df = ld_feature[cols_feature].merge(train_df).dropna()

    train_all_metric, train_goods_metric, train_wh_goods_metric = change_and_evaluate(df=train_df, df_price=df_price,
                                                                                      model_label=model_label,
                                                                                      dt=day_minus1)

    daily_all_metric, daily_goods_metric, daily_wh_goods_metric = change_and_evaluate(df=dfc_metric, df_price=df_price,
                                                                                      model_label=model_label,
                                                                                      dt=day_minus1)
    df_month_pred = dfc_metric.groupby(['wh_dept_id', 'goods_id']).agg({'y': 'sum', 'pred': 'sum'}).round(
        1).reset_index()

    month_all_metric, month_goods_metric, month_wh_goods_metric = change_and_evaluate(df=df_month_pred,
                                                                                      df_price=df_price,
                                                                                      model_label=model_label,
                                                                                      dt=day_minus1)

    save_pred_df(df_daily_pred=df_daily_pred, df_month_pred=df_month_pred, model_label=model_label,
                 data_label=data_label, dt=day_minus1)
    save_train_df(m_train_df=train_all_metric, model_label=model_label, data_label=data_label, dt=day_minus1)
    save_metric_df(m_all_df=daily_all_metric, m_goods_df=daily_goods_metric, m_wh_goods_df=daily_wh_goods_metric,
                   model_label=model_label, data_label=data_label, time_label='daily', dt=day_minus1)
    save_metric_df(m_all_df=month_all_metric, m_goods_df=month_goods_metric, m_wh_goods_df=month_wh_goods_metric,
                   model_label=model_label, data_label=data_label, time_label='month', dt=day_minus1)


def process_wh_goods_combination(df_feature, data_label, model_label,
                                 wh_id, g_id, window, target_date, day_minus1
                                 ):
    df_wh_goods = (df_feature.query(f"wh_dept_id == {wh_id} and  goods_id == {g_id}")
                   .reset_index(drop=True).copy())
    print(f"Processing: {len(df_wh_goods)}，{wh_id}，{g_id}，{data_label}")

    best_params = read_api.read_dt_folder(
        bip3("model/basic_predict_promote", f"best_params_{model_label}_wh_goods_pred_{data_label}"), day_minus1)
    best_params.query(f"wh_dept_id == {wh_id} and  goods_id == {g_id}", inplace=True)

    if len(df_wh_goods) > 2 and len(best_params) > 0:
        # df转成序列
        data = pd.Series(df_wh_goods['y'].values, index=pd.DatetimeIndex(df_wh_goods['ds']))

        model = train_holt_winters(data, alpha=best_params['alpha'].values[0], beta=best_params['beta'].values[0],
                                   gamma=best_params['gamma'].values[0], periods=best_params['periods'].values[0])

        # 训练
        df_fit = pd.DataFrame(pd.Series(model.fittedvalues)).reset_index()
        df_fit.columns = ['ds', 'pred']
        df_fit['pred'] = df_fit['pred'].round(1)
        df_fit['wh_dept_id'] = wh_id
        df_fit['goods_id'] = g_id
        df_fit = df_fit[['ds', 'wh_dept_id', 'goods_id', 'pred']].sort_values('ds')

        # 预测
        forecast = model.forecast(steps=window).round(1)
        df_forecast = pd.DataFrame(
            {'ds': pd.date_range(start=target_date, periods=window, freq='D'),
             'wh_dept_id': wh_id,
             'goods_id': g_id,
             'pred': forecast})

        df_forecast['pred'] = df_forecast['pred'].clip(lower=0)

        return df_fit, df_forecast
    else:
        return pd.DataFrame(), pd.DataFrame()


def train_holt_winters(data, alpha, beta, gamma, periods):
    model = ExponentialSmoothing(data, seasonal_periods=periods, trend="add", seasonal="add")
    model = model.fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma)
    return model


def save_best_params_df(best_params_df, model_label, data_label, dt):
    # -------------------------
    # save 训练集准确度
    best_params_df['dt'] = dt
    best_params_df['model'] = model_label
    cols_output = ['dt', 'model', 'wh_dept_id', 'goods_id', 'alpha', 'beta', 'gamma', 'periods', 'length', 'ACC']
    best_params_df = best_params_df[cols_output]
    bip3_save_df2(best_params_df,
                  table_folder_name=f'best_params_{model_label}_wh_goods_pred_{data_label}',
                  bip_folder='model/basic_predict_promote',
                  output_name=f'best_params_{model_label}_wh_goods_pred_{data_label}',
                  folder_dt=dt)


def find_best_parameters_parallel(data, df_price, model_label, dt, wh_id, g_id, window):
    best_score = float('-inf')
    best_params = {}

    # Define the parameter grid
    if len(data) >= window * 2:
        param_grid = {
            'alpha': [None, 0.3, 0.5, 0.7],
            'beta': [None, 0.3, 0.5, 0.7],
            'gamma': [None, 0.1, 0.3],
            'periods': [30]
        }
    else:
        param_grid = {
            'alpha': [None, 0.3, 0.5, 0.7],
            'beta': [None, 0.3, 0.5, 0.7],
            'gamma': [None, 0.1, 0.3],
            'periods': [2]
        }

    def evaluate_params(params):
        model = train_holt_winters(data, alpha=params['alpha'], beta=params['beta'], gamma=params['gamma'],
                                   periods=params['periods'])
        df_fit = pd.DataFrame(pd.Series(model.fittedvalues)).reset_index()
        df_fit.columns = ['ds', 'pred']
        df_fit['pred'] = df_fit['pred'].round(1)
        df_fit['wh_dept_id'] = wh_id
        df_fit['goods_id'] = g_id
        cols_fit = ['ds', 'wh_dept_id', 'goods_id', 'pred']
        df_fit = df_fit[cols_fit].sort_values('ds')
        df_true = pd.DataFrame(data).reset_index()
        df_true.columns = ['ds', 'y']
        df_fit = df_fit.merge(df_true)

        train_all_metric, train_goods_metric, train_wh_goods_metric = change_and_evaluate(df=df_fit, df_price=df_price,
                                                                                          model_label=model_label,
                                                                                          dt=dt)
        score = train_wh_goods_metric['ACC'].values[0]
        return params, score

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        results = executor.map(evaluate_params, ParameterGrid(param_grid))

    for params, score in results:
        if score > best_score:
            best_score = score
            best_params = params
            best_params['length'] = len(data)
            best_params['wh_dept_id'] = wh_id
            best_params['goods_id'] = g_id
            best_params['ACC'] = score
    return best_params
