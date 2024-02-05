# encoding: utf-8
# @created: 2023/11/22
# @author: jieqin.lin
# @file: projects/basic_predict_promote/dev/bpp15_pred_tree_group_b.py

from __init__ import project_path
from areas.table_info.dh_dw_table_info import dh_dw
from utils_offline.a00_imports import dfu, log20 as log, DayStr, read_api, shuttle, \
    bip3_save_df2, bip3
from utils.a71_save_df import bip3_save_model
from datetime import timedelta
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from projects.basic_predict_promote.b_0_0_utils_models import TreeModelTrainer, SelectGoodsList, FeatureEngineer

f"Import from {project_path}"


# =============================================================================
# 功能：基于清洗后仓库货物的历史数据，分别计算Xgboost模型，LightGBM模型和Catboost模型的预测值
# 1）因跟随，想要取最近的数据，不划分测试训练
# 2）测试货物: 冰凉感厚椰饮品，冷萃厚牛乳, 北海道丝绒风味厚乳
# 3）逐日递归预测: 每次预测未来1天，根据预测值，构造新的统计特征，输入模型预测下一天
# 4）多线程回刷历史预测
# =============================================================================


def main_train_and_evaluate_models():
    """
    回刷历史预测
    """
    # ----------------------------------

    date_ls = ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01', '2023-06-01', '2023-07-01',
               '2023-08-01', '2023-09-01','2023-10-01', '2023-11-01']
    model_ls = ['xgb', 'lgbm', 'catboost']
    data_label_ls = ['normal', 'all', 'new_shop_normal']
    window = 29
    goods_label = 'group_b'
    for data_label in data_label_ls:
        for model_label in model_ls:
            for target_date in date_ls:
                print(f"Processing: {target_date}, {model_label}, {data_label}")
                train_and_evaluate_models(target_date=target_date, model_label=model_label,
                                          data_label=data_label,
                                          goods_label=goods_label,
                                          window=window)


def main():
    model_ls = ['xgb', 'lgbm', 'catboost']
    thread_pool_train_and_evaluate_models()

    last_wh_goods_acc(model_ls=model_ls, window=6)
    last_wh_goods_acc(model_ls=model_ls, window=3)
    last_wh_goods_acc(model_ls=model_ls, window=2)
    last_wh_goods_acc(model_ls=model_ls, window=1)

    last_goods_acc(model_ls=model_ls, window=6)
    last_goods_acc(model_ls=model_ls, window=3)
    last_goods_acc(model_ls=model_ls, window=2)
    last_goods_acc(model_ls=model_ls, window=1)


def thread_pool_train_and_evaluate_models():
    """
    多线程回刷历史预测
    """
    print("准确率差")
    import concurrent.futures
    import itertools

    # ---------------
    date_ls = ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01', '2023-06-01', '2023-07-01',
               '2023-08-01', '2023-09-01','2023-10-01', '2023-11-01']
    model_ls = ['xgb', 'lgbm', 'catboost']
    data_label_ls = ['normal', 'all', 'new_shop_normal']
    window = 29
    goods_label = 'group_b'

    def process_combination(combination):
        target_date, model_label, data_label = combination
        print(f"Processing: {target_date}, {model_label}, {data_label}")

        try:
            train_and_evaluate_models(
                target_date, model_label, data_label,goods_label, window)
        except Exception as e:
            print(f"Failed: {str(e)}")

    combinations = itertools.product(date_ls, model_ls, data_label_ls)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.map(process_combination, combinations)


def train_and_evaluate_models(target_date, model_label, data_label, goods_label,window):
    """
    单日模型训练
    逐日递归
    """
    # 时间范围： 1天前 未来30天 120天前
    # start_dt = '2020-01-01'
    start_dt = DayStr.n_day_delta(target_date, n=-183)
    day_minus1 = DayStr.n_day_delta(target_date, n=-1)
    # 转成价格
    pred_calc_day = DayStr.get_dt_str(None)
    pred_minus1_day = DayStr.n_day_delta(pred_calc_day, n=-1)
    df_price = get_price(pred_minus1_day)

    # ----------------------
    # 特征
    features = ['wh_dept_id', 'goods_id', 'year', 'month', 'day', 'day_of_week',
                'lag_1_mean', 'lag_7_mean', 'lag_7_std', 'lag_7_min', 'lag_7_max', 'lag_14_mean',
                'lag_14_std', 'lag_14_min', 'lag_14_max', 'lag_21_mean', 'lag_21_std',
                'lag_21_min', 'lag_21_max', 'lag_28_mean', 'lag_28_std', 'lag_28_min',
                'lag_28_max', 'lag_30_mean', 'lag_30_std', 'lag_30_min', 'lag_30_max',
                'lag_60_mean', 'lag_60_std', 'lag_60_min', 'lag_60_max', 'lag_120_mean',
                'lag_120_std', 'lag_120_min', 'lag_120_max']
    # 真实值
    true_label = 'y'

    # ---------------
    # 货物范围
    # 初始化训练器
    trainer = TreeModelTrainer(features,
                               true_label="y",
                               train_ratio=1,
                               model_label=model_label,
                               data_label=data_label,
                               goods_label=goods_label)

    # ----------------------
    selector = SelectGoodsList()
    sel_goods_ls = selector.get_group_b_goods_id()
    # ----------------------
    log.debug("历史单日模型训练")


    # 特征，筛选数据从2020-01-01开始至今
    feature_generator = FeatureEngineer(data_label=data_label)
    ld_feature = feature_generator.get_feature_wh_goods(start_date=start_dt,
                                                        end_date=pred_minus1_day,
                                                        sel_goods_ls=sel_goods_ls)
    df_feature = ld_feature.query(f"'{start_dt}'<= ds <='{pred_minus1_day}'")

    # 不划分训练集/测试集
    train_ratio = 1
    train_df, test_df, X_train, y_train, X_test, y_test = split_train_and_test(
        df=df_feature, features=features, true_label=true_label, train_ratio=train_ratio)

    # 找最佳参数
    # model = grid_search_model(
    #     X_train=X_train, y_train=y_train, model_label=model_label)

    # 训练
    model = train_model(
        X_train=X_train, y_train=y_train, model_label=model_label)

    # -------------------------
    log.debug("未来逐日递归模型训练")

    # 逐日
    df_daily_pred = trainer.whole_country_rolling_predict(start_date=target_date,
                                                                    window=window,
                                                                    model=model,
                                                                    df=df_feature,
                                                                    label=f"{model_label}_{data_label}"
                                                                    )

    #  预测不为负
    df_daily_pred["y"] = np.clip(df_daily_pred["y"], 0, np.inf)

    # merge真实-预测
    cols_daily = ['ds', 'wh_dept_id', 'goods_id', 'y']
    df_daily_pred = df_daily_pred[cols_daily].rename(columns={'y': 'pred'})
    cols_feature = ['ds', 'wh_dept_id', 'goods_id', 'y']
    dfc_metric = ld_feature[cols_feature].merge(df_daily_pred)

    # 单日准确率
    daily_all_metric, daily_goods_metric, daily_wh_goods_metric = change_and_evaluate(df=dfc_metric, df_price=df_price,
                                                                                      model_label=model_label,
                                                                                      dt=day_minus1)

    # save 单日准确率评估
    save_metric_df(
        m_all_df=daily_all_metric,
        m_goods_df=daily_goods_metric,
        m_wh_goods_df=daily_wh_goods_metric,
        model_label=model_label,
        data_label=data_label,
        time_label='daily',
        dt=day_minus1)

    # 30天之和准确率
    df_month_pred = (dfc_metric.groupby(['wh_dept_id', 'goods_id'])
                     .agg({'y': 'sum', 'pred': 'sum'})
                     .round(1).reset_index()
                     )
    month_all_metric, month_goods_metric, month_wh_goods_metric = change_and_evaluate(df=df_month_pred,
                                                                                      df_price=df_price,
                                                                                      model_label=model_label,
                                                                                      dt=day_minus1)
    # save 预测
    save_pred_df(df_daily_pred=df_daily_pred, df_month_pred=df_month_pred, model_label=model_label,
                 data_label=data_label, dt=day_minus1)




    # save 月度准确率评估
    save_metric_df(
        m_all_df=month_all_metric,
        m_goods_df=month_goods_metric,
        m_wh_goods_df=month_wh_goods_metric,
        model_label=model_label,
        data_label=data_label,
        time_label='month',
        dt=day_minus1)

    bip3_save_model(model,
                    table_folder_name=f'model_{model_label}_wh_goods_pred_{data_label}',
                    bip_folder='model/basic_predict_promote_group_b',
                    output_name=f'model_{model_label}_wh_goods_pred_{data_label}',
                    folder_dt=day_minus1)


def split_train_and_test(df, features, true_label, train_ratio=0.8):
    """
    划分训练集 测试集
    """
    train_size = int(len(df) * train_ratio)
    train_df = df[:train_size]
    test_df = df[train_size:]
    X_train = train_df[features]
    y_train = train_df[true_label]
    X_test = test_df[features]
    y_test = test_df[true_label]
    return train_df, test_df, X_train, y_train, X_test, y_test


# =============================================================================
# 功能：参数设置
# 1）xgboost：objective='reg:gamma',n_estimators=1500, learning_rate=0.05, max_depth=7, eval_metric=["error", "logloss"]
# 2）catboost: iterations=1000, learning_rate=0.1, depth=6
# 3）lightgbm: objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=1500, eval_metric='l2'
# =============================================================================

def train_xgboost(X_train, y_train):
    log.debug('xgb')
    model = xgb.XGBRegressor(objective='reg:gamma',
                             n_estimators=1500,
                             learning_rate=0.05,
                             max_depth=7,
                             eval_metric=["error", "logloss"])
    eval_set = [(X_train, y_train)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    return model


def train_catboost(X_train, y_train):
    log.debug('catboost')
    model = cb.CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6)
    model.fit(X_train, y_train, verbose=False)
    return model



def train_lightgbm(X_train, y_train):
    log.debug('lgbm')
    model = lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=1500,
                              verbose=-1, verbose_eval=False)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train)], eval_metric='l2')
    return model



def train_model(X_train, y_train, model_label):
    if model_label == 'xgb':
        model = train_xgboost(X_train, y_train)
    elif model_label == 'catboost':
        model = train_catboost(X_train, y_train)
    elif model_label == 'lgbm':
        model = train_lightgbm(X_train, y_train)
    return model



def get_price(pred_minus1_day):
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
    price = shuttle.query_dataq(sql_price)
    return price




def change_to_price(df, df_price):
    df_price_mean = df_price.groupby('goods_id')['unit_price'].mean().round(4).rename('unit_price_mean').reset_index()
    df_price_wh = df_price.groupby('wh_dept_id')['unit_price'].mean().round(4).rename('wh_unit_price_mean').reset_index()
    df = df.merge(df_price, 'left').merge(df_price_mean, 'left').merge(df_price_wh, 'left')
    df.update(df['unit_price'].fillna(df['unit_price_mean']).fillna(df['wh_unit_price_mean']))
    df.drop(['unit_price_mean'], axis=1, inplace=True)
    df['y_price'] = df.eval("y * unit_price").round(1)
    df['pred_price'] = df.eval("pred * unit_price").round(1)
    return df


def evaluate_model(y_true, y_pred):
    MSE = mean_squared_error(y_true, y_pred).round(4)
    MAE = mean_absolute_error(y_true, y_pred).round(4)
    MAPE = mean_absolute_percentage_error(y_true, y_pred).round(4)
    ACC = (1 - MAPE).round(4)
    df_metric = pd.DataFrame({'MSE': [MSE], 'MAE': [MAE], 'MAPE': [MAPE], 'ACC': [ACC]})
    return df_metric


def change_and_evaluate(df, df_price, model_label, dt):
    df = change_to_price(df=df, df_price=df_price)

    # 准确率
    y_true = df['y_price'].values
    y_pred = df['pred_price'].values
    df_all_metric = evaluate_model(y_true=y_true, y_pred=y_pred)
    df_all_metric['dt'] = dt
    df_all_metric['model'] = model_label

    # 货物准确率
    df_goods_metric = df.groupby('goods_id').apply(
        lambda x: evaluate_model(x['y_price'].values, x['pred_price'].values)).reset_index()
    df_goods_metric['dt'] = dt
    df_goods_metric['model'] = model_label

    # 仓库货物准确率
    df_wh_goods_metric = df.groupby(['wh_dept_id', 'goods_id']).apply(
        lambda x: evaluate_model(x['y_price'].values, x['pred_price'].values)).reset_index()
    df_wh_goods_metric['dt'] = dt
    df_wh_goods_metric['model'] = model_label

    return df_all_metric, df_goods_metric, df_wh_goods_metric


# =============================================================================
# 功能：保存各个模型的预测数据
# 1）单日预测
# 2）预测的30天之和
# 3）评估（仓库货物，货物，全国维度）
# =============================================================================


def save_pred_df(df_daily_pred, df_month_pred, model_label, data_label, dt):
    # -------------------------
    # save 数据格式
    df_daily_pred.rename(columns={'ds': 'predict_dt', 'pred': 'predict_demand'}, inplace=True)
    df_daily_pred['predict_dt'] = df_daily_pred['predict_dt'].dt.strftime('%Y-%m-%d')
    df_daily_pred['dt'] = dt
    df_daily_pred['model'] = model_label
    cols_int = ['wh_dept_id', 'goods_id']
    df_daily_pred = dfu.df_col_to_numeric(df_daily_pred, cols_int)
    df_daily_pred['predict_demand'] = df_daily_pred['predict_demand'].round(1)
    cols_output = ['predict_dt', 'wh_dept_id', 'goods_id', 'model', 'predict_demand', 'dt']
    df_daily_pred = df_daily_pred[cols_output].sort_values('predict_dt')

    bip3_save_df2(df_daily_pred,
                  table_folder_name=f'{model_label}_wh_goods_pred_{data_label}',
                  bip_folder='model/basic_predict_promote_group_b',
                  output_name=f'{model_label}_wh_goods_pred_{data_label}',
                  folder_dt=dt)

    # -------------------------
    # save 数据格式
    df_month_pred.rename(columns={'ds': 'predict_dt', 'pred': 'predict_demand'}, inplace=True)
    df_month_pred['dt'] = dt
    df_month_pred['model'] = model_label
    cols_int = ['wh_dept_id', 'goods_id']
    df_month_pred = dfu.df_col_to_numeric(df_month_pred, cols_int)
    df_month_pred['predict_demand'] = df_month_pred['predict_demand'].round(1)
    cols_output = ['wh_dept_id', 'goods_id', 'model', 'predict_demand', 'dt']
    df_month_pred = df_month_pred[cols_output]
    bip3_save_df2(df_month_pred,
                  table_folder_name=f'{model_label}_wh_goods_pred_{data_label}_sum',
                  bip_folder='model/basic_predict_promote_group_b',
                  output_name=f'{model_label}_wh_goods_pred_{data_label}_sum',
                  folder_dt=dt)


def save_train_df(m_train_df, model_label, data_label, dt):
    # -------------------------
    # save 训练集准确度
    cols_output = ['dt', 'model', 'MAE', 'MSE', 'MAPE', 'ACC']
    m_train_df = m_train_df[cols_output]
    bip3_save_df2(m_train_df,
                  table_folder_name=f'train_metric_{model_label}_wh_goods_pred_{data_label}',
                  bip_folder='model/basic_predict_promote_group_b',
                  output_name=f'train_metric_{model_label}_wh_goods_pred_{data_label}',
                  folder_dt=dt)


def save_metric_df(m_all_df, m_goods_df, m_wh_goods_df, model_label, data_label, time_label, dt):
    # -------------------------
    # save 时间维度++逐日递归预测准确度
    cols_output = ['dt', 'model', 'MAE', 'MSE', 'MAPE', 'ACC']
    m_all_df = m_all_df[cols_output]
    bip3_save_df2(m_all_df,
                  table_folder_name=f'metric_{time_label}_{model_label}_wh_goods_pred_{data_label}',
                  bip_folder='model/basic_predict_promote_group_b',
                  output_name=f'metric_{time_label}_{model_label}_wh_goods_pred_{data_label}',
                  folder_dt=dt)

    # -------------------------
    # save 时间维度++货物++逐日递归预测准确度
    cols_output = ['dt', 'model', 'goods_id', 'MAE', 'MSE', 'MAPE', 'ACC']
    m_goods_df = m_goods_df[cols_output]
    bip3_save_df2(m_goods_df,
                  table_folder_name=f'metric_{time_label}_goods_{model_label}_wh_goods_pred_{data_label}',
                  bip_folder='model/basic_predict_promote_group_b',
                  output_name=f'metric_{time_label}_goods_{model_label}_wh_goods_pred_{data_label}',
                  folder_dt=dt)

    # -------------------------
    # save 时间维度++仓库++货物++逐日递归预测准确度
    cols_output = ['dt', 'model', 'wh_dept_id', 'goods_id', 'MAE', 'MSE', 'MAPE', 'ACC']
    m_wh_goods_df = m_wh_goods_df[cols_output]
    bip3_save_df2(m_wh_goods_df,
                  table_folder_name=f'metric_{time_label}_wh_goods_{model_label}_wh_goods_pred_{data_label}',
                  bip_folder='model/basic_predict_promote_group_b',
                  output_name=f'metric_{time_label}_wh_goods_{model_label}_wh_goods_pred_{data_label}',
                  folder_dt=dt)


def last_wh_goods_acc(model_ls, window=3):
    import concurrent.futures

    data_label_ls = ['normal', 'all', 'new_shop_normal']

    def process_date(data_label, model_label, window):
        log.debug(f"{data_label}--{model_label}")
        try:
            recent_month_wh_goods_metric_mean(data_label, model_label, window)
        except Exception as e:
            print(f"Failed: {str(e)}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for data_label in data_label_ls:
            for model_label in model_ls:
                future = executor.submit(process_date, data_label, model_label, window)
                futures.append(future)

        # Wait for all threads to complete
        concurrent.futures.wait(futures)


def last_goods_acc(model_ls, window=3):
    import concurrent.futures

    data_label_ls = ['normal', 'all', 'new_shop_normal']

    def process_date(data_label, model_label, window):
        log.debug(f"{data_label}--{model_label}")
        try:
            recent_month_goods_metric_mean(data_label, model_label, window)
        except Exception as e:
            print(f"Failed: {str(e)}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for data_label in data_label_ls:
            for model_label in model_ls:
                future = executor.submit(process_date, data_label, model_label, window)
                futures.append(future)

        # Wait for all threads to complete
        concurrent.futures.wait(futures)


def recent_month_wh_goods_metric_mean(data_label, model_label, window):
    start_dt = '2023-01-01'
    pred_calc_day = DayStr.get_dt_str(None)
    pred_minus1_day = DayStr.n_day_delta(pred_calc_day, n=-1)
    # 最近3个月准确率均值
    df_metric = read_api.read_dt_folder(
        bip3("model/basic_predict_promote_group_b", f"metric_month_wh_goods_{model_label}_wh_goods_pred_{data_label}"),
        start_dt, pred_minus1_day
    )
    df_metric['dt'] = pd.to_datetime(df_metric['dt'])

    df_mean = pd.DataFrame()
    for wh_id, group in df_metric.groupby(['wh_dept_id', 'goods_id']):
        df_wh_goods = group.reset_index(drop=True).copy()
        # 计算平均值和标准差
        df_wh_goods[f'last_{window}_mse'] = df_wh_goods['MSE'].rolling(window=window, min_periods=0).mean().round(4)
        df_wh_goods[f'last_{window}_mae'] = df_wh_goods['MAE'].rolling(window=window, min_periods=0).mean().round(4)
        df_wh_goods[f'last_{window}_mape'] = df_wh_goods['MAPE'].rolling(window=window, min_periods=0).mean().round(4)
        df_wh_goods[f'last_{window}_acc'] = df_wh_goods['ACC'].rolling(window=window, min_periods=0).mean().round(4)
        df_wh_goods[f'last_{window}_std'] = df_wh_goods['ACC'].rolling(window=window, min_periods=0).std().round(4)

        df_mean = pd.concat([df_mean, df_wh_goods])

    df_mean['model'] = model_label

    # -------------------------
    # save
    cols_output = ['wh_dept_id', 'goods_id', 'model', f'last_{window}_mse', f'last_{window}_mae', f'last_{window}_mape',
                   f'last_{window}_acc', f'last_{window}_std', 'dt']
    df_mean['dt'] = df_mean['dt'].dt.strftime('%Y-%m-%d')
    df_mean = df_mean[cols_output]
    bip3_save_df2(df_mean,
                  table_folder_name=f'{model_label}_last_{window}_acc_{data_label}',
                  bip_folder='model/basic_predict_promote_group_b',
                  output_name=f'{model_label}_last_{window}_acc_{data_label}',
                  folder_dt=pred_minus1_day)
    return df_mean


def recent_month_goods_metric_mean(data_label, model_label, window):
    start_dt = '2023-01-01'
    pred_calc_day = DayStr.get_dt_str(None)
    pred_minus1_day = DayStr.n_day_delta(pred_calc_day, n=-1)
    # 最近3个月准确率均值
    df_metric = read_api.read_dt_folder(
        bip3("model/basic_predict_promote_group_b", f"metric_month_goods_{model_label}_wh_goods_pred_{data_label}"),
        start_dt, pred_minus1_day
    )
    df_metric['dt'] = pd.to_datetime(df_metric['dt'])

    df_mean = pd.DataFrame()
    for g_id, group in df_metric.groupby(['goods_id']):
        df_goods = group.reset_index(drop=True).copy()
        df_goods[f'last_{window}_mse'] = df_goods['MSE'].rolling(window=window, min_periods=0).mean().round(4)
        df_goods[f'last_{window}_mae'] = df_goods['MAE'].rolling(window=window, min_periods=0).mean().round(4)
        df_goods[f'last_{window}_mape'] = df_goods['MAPE'].rolling(window=window, min_periods=0).mean().round(4)
        df_goods[f'last_{window}_acc'] = df_goods['ACC'].rolling(window=window, min_periods=0).mean().round(4)
        df_goods[f'last_{window}_std'] = df_goods['ACC'].rolling(window=window, min_periods=0).std().round(4)
        df_mean = pd.concat([df_mean, df_goods])
    df_mean['model'] = model_label

    # -------------------------
    # save
    cols_output = ['goods_id', 'model', f'last_{window}_mse', f'last_{window}_mae', f'last_{window}_mape',
                   f'last_{window}_acc', f'last_{window}_std', 'dt']
    df_mean['dt'] = df_mean['dt'].dt.strftime('%Y-%m-%d')
    df_mean = df_mean[cols_output]
    bip3_save_df2(df_mean,
                  table_folder_name=f'{model_label}_goods_last_{window}_acc_{data_label}',
                  bip_folder='model/basic_predict_promote_group_b',
                  output_name=f'{model_label}_goods_last_{window}_acc_{data_label}',
                  folder_dt=pred_minus1_day)
    return df_mean
