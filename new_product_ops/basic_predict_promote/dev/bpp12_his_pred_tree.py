# encoding: utf-8
# @created: 2023/10/9
# @author: jieqin.lin
# @file: projects/basic_predict_promote/dev/bpp12_his_pred_tree.py

"""
树模型 Xgboost LightGBM Catboost  预测仓库货物未来30天消耗

生成数据
############################################
(货物)总预测过去3个月准确率均值
bip3("model/basic_predict_promote", "xgb_goods_last_3_acc_all")
bip3("model/basic_predict_promote", "lgbm_goods_last_3_acc_all")
bip3("model/basic_predict_promote", "catboost_goods_last_3_acc_all")

(货物)总预测过去2个月准确率均值
bip3("model/basic_predict_promote", "xgb_goods_last_2_acc_all")
bip3("model/basic_predict_promote", "lgbm_goods_last_2_acc_all")
bip3("model/basic_predict_promote", "catboost_goods_last_2_acc_all")

(货物)总预测过去1个月准确率均值
bip3("model/basic_predict_promote", "xgb_goods_last_1_acc_all")
bip3("model/basic_predict_promote", "lgbm_goods_last_1_acc_all")
bip3("model/basic_predict_promote", "catboost_goods_last_1_acc_all")

(货物)常规预测过去3个月准确率均值
bip3("model/basic_predict_promote", "xgb_goods_last_2_acc_normal")
bip3("model/basic_predict_promote", "lgbm_goods_last_2_acc_normal")
bip3("model/basic_predict_promote", "catboost_goods_last_2_acc_normal")

(货物)常规预测过去2个月准确率均值
bip3("model/basic_predict_promote", "xgb_goods_last_3_acc_normal")
bip3("model/basic_predict_promote", "lgbm_goods_last_3_acc_normal")
bip3("model/basic_predict_promote", "catboost_goods_last_3_acc_normal")

(货物)常规预测过去1个月准确率均值
bip3("model/basic_predict_promote", "xgb_goods_last_1_acc_normal")
bip3("model/basic_predict_promote", "lgbm_goods_last_1_acc_normal")
bip3("model/basic_predict_promote", "catboost_goods_last_1_acc_normal")

############################################
(仓库货物)总预测过去3个月准确率均值
bip3("model/basic_predict_promote", "xgb_last_3_acc_all")
bip3("model/basic_predict_promote", "lgbm_last_3_acc_all")
bip3("model/basic_predict_promote", "catboost_last_3_acc_all")

(仓库货物)总预测过去2个月准确率均值
bip3("model/basic_predict_promote", "xgb_last_2_acc_all")
bip3("model/basic_predict_promote", "lgbm_last_2_acc_all")
bip3("model/basic_predict_promote", "catboost_last_2_acc_all")

总预测过去1个月准确率均值
bip3("model/basic_predict_promote", "xgb_last_1_acc_all")
bip3("model/basic_predict_promote", "lgbm_last_1_acc_all")
bip3("model/basic_predict_promote", "catboost_last_1_acc_all")

(仓库货物)常规预测过去3个月准确率均值
bip3("model/basic_predict_promote", "xgb_last_2_acc_normal")
bip3("model/basic_predict_promote", "lgbm_last_2_acc_normal")
bip3("model/basic_predict_promote", "catboost_last_2_acc_normal")

(仓库货物)常规预测过去2个月准确率均值
bip3("model/basic_predict_promote", "xgb_last_3_acc_normal")
bip3("model/basic_predict_promote", "lgbm_last_3_acc_normal")
bip3("model/basic_predict_promote", "catboost_last_3_acc_normal")

(仓库货物)常规预测过去1个月准确率均值
bip3("model/basic_predict_promote", "xgb_last_1_acc_normal")
bip3("model/basic_predict_promote", "lgbm_last_1_acc_normal")
bip3("model/basic_predict_promote", "catboost_last_1_acc_normal")

############################################
Xgboost
# 预测
bip3("model/basic_predict_promote", "xgb_wh_goods_pred_normal")                         预测常规消耗量
bip3("model/basic_predict_promote", "xgb_wh_goods_pred_all")                            预测总消耗量

# 训练
bip3("model/basic_predict_promote", "train_metric_xgb_wh_goods_pred_normal")            常规品训练集准确度
bip3("model/basic_predict_promote", "train_metric_xgb_wh_goods_pred_all")               总训练集准确度

# 逐日
bip3("model/basic_predict_promote", "metric_daily_xgb_wh_goods_pred_normal")            常规品天维度准确度
bip3("model/basic_predict_promote", "metric_daily_goods_xgb_wh_goods_pred_normal")      常规品天维度货物级别准确度
bip3("model/basic_predict_promote", "metric_daily_wh_goods_xgb_wh_goods_pred_normal")   常规品天维度仓库货物级别准确度

bip3("model/basic_predict_promote", "metric_daily_xgb_wh_goods_pred_all)                总天维度准确度
bip3("model/basic_predict_promote", "metric_daily_goods_xgb_wh_goods_pred_all)          总天维度货物级别准确度
bip3("model/basic_predict_promote", "metric_daily_wh_goods_xgb_wh_goods_pred_all)       总天维度仓库货物级别准确度

# 逐日汇总到月度
bip3("model/basic_predict_promote", "metric_month_xgb_wh_goods_pred_normal")            常规品月维度准确度
bip3("model/basic_predict_promote", "metric_month_goods_xgb_wh_goods_pred_normal")      常规品月维度货物级别准确度
bip3("model/basic_predict_promote", "metric_month_wh_goods_xgb_wh_goods_pred_normal")   常规品月维度仓库货物级别准确度

bip3("model/basic_predict_promote", "metric_month_xgb_wh_goods_pred_all")               总月维度准确度
bip3("model/basic_predict_promote", "metric_month_goods_xgb_wh_goods_pred_all")         总月维度货物级别准确度
bip3("model/basic_predict_promote", "metric_month_wh_goods_xgb_wh_goods_pred_all")      总月维度仓库货物级别准确度

############################################
LightGBM
# 预测
bip3("model/basic_predict_promote", "lgbm_wh_goods_pred_normal")                         预测常规消耗量
bip3("model/basic_predict_promote", "lgbm_wh_goods_pred_all")                            预测总消耗量

# 训练
bip3("model/basic_predict_promote", "train_metric_lgbm_wh_goods_pred_normal")            常规品训练集准确度
bip3("model/basic_predict_promote", "train_metric_lgbm_wh_goods_pred_all")               总训练集准确度

# 逐日
bip3("model/basic_predict_promote", "metric_daily_lgbm_wh_goods_pred_normal")            常规品天维度准确度
bip3("model/basic_predict_promote", "metric_daily_goods_lgbm_wh_goods_pred_normal")      常规品天维度货物级别准确度
bip3("model/basic_predict_promote", "metric_daily_wh_goods_lgbm_wh_goods_pred_normal")   常规品天维度仓库货物级别准确度

bip3("model/basic_predict_promote", "metric_daily_lgbm_wh_goods_pred_all)                总天维度准确度
bip3("model/basic_predict_promote", "metric_daily_goods_lgbm_wh_goods_pred_all)          总天维度货物级别准确度
bip3("model/basic_predict_promote", "metric_daily_wh_goods_lgbm_wh_goods_pred_all)       总天维度仓库货物级别准确度

# 逐日汇总到月度
bip3("model/basic_predict_promote", "metric_month_lgbm_wh_goods_pred_normal")            常规品月维度准确度
bip3("model/basic_predict_promote", "metric_month_goods_lgbm_wh_goods_pred_normal")      常规品月维度货物级别准确度
bip3("model/basic_predict_promote", "metric_month_wh_goods_lgbm_wh_goods_pred_normal")   常规品月维度仓库货物级别准确度

bip3("model/basic_predict_promote", "metric_month_lgbm_wh_goods_pred_all")               总月维度准确度
bip3("model/basic_predict_promote", "metric_month_goods_lgbm_wh_goods_pred_all")         总月维度货物级别准确度
bip3("model/basic_predict_promote", "metric_month_wh_goods_lgbm_wh_goods_pred_all")      总月维度仓库货物级别准确度


############################################
catboost
# 预测
bip3("model/basic_predict_promote", "catboost_wh_goods_pred_normal")                         预测常规消耗量
bip3("model/basic_predict_promote", "catboost_wh_goods_pred_all")                            预测总消耗量

# 训练
bip3("model/basic_predict_promote", "train_metric_catboost_wh_goods_pred_normal")            常规品训练集准确度
bip3("model/basic_predict_promote", "train_metric_catboost_wh_goods_pred_all")               总训练集准确度

# 逐日
bip3("model/basic_predict_promote", "metric_daily_catboost_wh_goods_pred_normal")            常规品天维度准确度
bip3("model/basic_predict_promote", "metric_daily_goods_catboost_wh_goods_pred_normal")      常规品天维度货物级别准确度
bip3("model/basic_predict_promote", "metric_daily_wh_goods_catboost_wh_goods_pred_normal")   常规品天维度仓库货物级别准确度

bip3("model/basic_predict_promote", "metric_daily_catboost_wh_goods_pred_all)                总天维度准确度
bip3("model/basic_predict_promote", "metric_daily_goods_catboost_wh_goods_pred_all)          总天维度货物级别准确度
bip3("model/basic_predict_promote", "metric_daily_wh_goods_catboost_wh_goods_pred_all)       总天维度仓库货物级别准确度

# 逐日汇总到月度
bip3("model/basic_predict_promote", "metric_month_catboost_wh_goods_pred_normal")            常规品月维度准确度
bip3("model/basic_predict_promote", "metric_month_goods_catboost_wh_goods_pred_normal")      常规品月维度货物级别准确度
bip3("model/basic_predict_promote", "metric_month_wh_goods_catboost_wh_goods_pred_normal")   常规品月维度仓库货物级别准确度

bip3("model/basic_predict_promote", "metric_month_catboost_wh_goods_pred_all")               总月维度准确度
bip3("model/basic_predict_promote", "metric_month_goods_catboost_wh_goods_pred_all")         总月维度货物级别准确度
bip3("model/basic_predict_promote", "metric_month_wh_goods_catboost_wh_goods_pred_all")      总月维度仓库货物级别准确度


############################################
依赖数据
bip3("model/basic_predict_promote", "stock_wh_goods_type_flg_theory_sale_cnt")
bip3("model/basic_predict_promote", "stock_wh_goods_theory_sale_cnt")
bip3("model/basic_predict_promote", "feature_engineering_normal")
bip3("model/basic_predict_promote", "feature_engineering_all")
dw_dws.dws_stock_warehouse_stock_adjust_d_inc_summary
"""
from __init__ import project_path
from areas.table_info.dh_dw_table_info import dh_dw

from utils_offline.a00_imports import dfu, log20 as log, DayStr, read_api, shuttle, \
    bip3_save_df2, bip3
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
    df_goods_info = dh_dw.dim_stock.goods_info()
    # ----------------------------------
    target_goods_name = ['冰凉感厚椰饮品', '冷萃厚牛乳', '北海道丝绒风味厚乳']
    date_ls = ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01', '2023-06-01', '2023-07-01',
               '2023-08-01', '2023-09-01', '2023-10-01']
    sel_goods_ls = df_goods_info.query(f"goods_name == {target_goods_name}")['goods_id'].unique().tolist()
    model_ls = ['xgb', 'lgbm', 'catboost']
    # data_label_ls = ['normal', 'all','new_shop_normal']
    data_label_ls = ['new_shop_normal']
    window = 29
    for data_label in data_label_ls:
        for model_label in model_ls:
            for target_date in date_ls:
                train_and_evaluate_models(target_date=target_date, model_label=model_label,
                                          data_label=data_label,
                                          sel_goods_ls=sel_goods_ls,
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
    import concurrent.futures
    import itertools
    # date_ls = ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01', '2023-06-01', '2023-07-01',
    #            '2023-08-01', '2023-09-01', '2023-10-01']
    date_ls = ['2023-10-01']
    df_goods_info = dh_dw.dim_stock.goods_info()
    target_goods_name = ['冰凉感厚椰饮品', '冷萃厚牛乳', '北海道丝绒风味厚乳']
    sel_goods_ls = df_goods_info.query(f"goods_name in {target_goods_name}")['goods_id'].unique().tolist()
    # data_label_ls = ['normal', 'all','new_shop_normal']
    # data_label_ls = ['new_shop_normal']
    data_label_ls = ['normal', 'all']
    window = 29
    model_ls = ['xgb', 'lgbm', 'catboost']

    def process_combination(combination):
        target_date, model_label, data_label = combination
        print(f"Processing: {target_date}, {model_label}, {data_label}")

        try:
            train_and_evaluate_models(
                target_date, model_label, data_label, sel_goods_ls, window)
        except Exception as e:
            print(f"Failed: {str(e)}")

    combinations = itertools.product(date_ls, model_ls, data_label_ls)

    with concurrent.futures.ThreadPoolExecutor(max_workers=9) as executor:
        executor.map(process_combination, combinations)


def train_and_evaluate_models(target_date, model_label, data_label, sel_goods_ls, window):
    """
    单日模型训练
    逐日递归
    """
    # 时间范围： 1天前 未来30天 120天前
    start_dt = '2020-01-01'
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

    # ----------------------
    log.debug("历史单日模型训练")

    # 消耗特征，筛选数据从2020-01-01到每月初
    ld_feature = get_feature(data_label=data_label, sel_goods_ls=sel_goods_ls, start_dt=start_dt,
                             end_dt=pred_minus1_day)
    df_feature = ld_feature.query(f"'{start_dt}'<=ds <'{target_date}'")

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
    train_df['pred'] = model.predict(X_train).round(1)
    train_all_metric, train_goods_metric, train_wh_goods_metric = change_and_evaluate(df=train_df, df_price=df_price,
                                                                                      model_label=model_label,
                                                                                      dt=day_minus1)
    # -------------------------
    log.debug("未来逐日递归模型训练")
    # 过去120天仓库货物真实消耗
    # df_true = get_his_true_value(
    #     data_label=data_label, sel_goods_ls=sel_goods_ls, dt=day_minus1)
    # 逐日
    # df_daily_pred = predict_daily(
    #     ld_feature=ld_feature, df_true=df_true, sel_goods_ls=sel_goods_ls,
    #     features=features, model=model, start_dt=target_date, window=window)

    # 逐日
    df_daily_pred = predict_daily(
        df_feature=df_feature,
        features=features, model=model, start_dt=target_date, window=window)

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

    # save 训练集准确率评估
    save_train_df(m_train_df=train_all_metric,
                  model_label=model_label,
                  data_label=data_label,
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

    # save 月度准确率评估
    save_metric_df(
        m_all_df=month_all_metric,
        m_goods_df=month_goods_metric,
        m_wh_goods_df=month_wh_goods_metric,
        model_label=model_label,
        data_label=data_label,
        time_label='month',
        dt=day_minus1)


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
    model = xgb.XGBRegressor(objective='reg:gamma', n_estimators=1500, learning_rate=0.05, max_depth=7,
                             eval_metric=["error", "logloss"])
    eval_set = [(X_train, y_train)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    return model


def grid_search_xgboost(X_train, y_train):
    parameters = {
        'n_estimators': [100, 500, 1000, 1500],
        'learning_rate': [0.01, 0.05, 0.06, 0.1],
        'max_depth': [3, 5, 6, 7],
        'gamma': [0, 0.1, 0.15, 0.2]
    }

    xgb_model = xgb.XGBRegressor(objective='reg:gamma', eval_metric=["error", "logloss"])

    grid_search = GridSearchCV(estimator=xgb_model, param_grid=parameters, cv=3, scoring="neg_mean_absolute_error")
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_model = grid_search.best_estimator_

    return best_params, best_score, best_model


def train_catboost(X_train, y_train):
    log.debug('catboost')
    model = cb.CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6)
    model.fit(X_train, y_train, verbose=False)
    return model


def grid_search_catboost(X_train, y_train):
    parameters = {
        'iterations': [100, 500, 1000],
        'learning_rate': [0.01, 0.05, 0.1],
        'depth': [3, 5, 6],
        'l2_leaf_reg': [1, 3, 5]
    }
    catboost_model = cb.CatBoostRegressor()
    grid_search = GridSearchCV(estimator=catboost_model, param_grid=parameters, cv=3, scoring="neg_mean_absolute_error")
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_model = grid_search.best_estimator_

    return best_params, best_score, best_model


def train_lightgbm(X_train, y_train):
    log.debug('lgbm')
    model = lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=1500,
                              verbose=-1, verbose_eval=False)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train)], eval_metric='l2')
    return model


def grid_search_lightgbm(X_train, y_train):
    parameters = {
        'num_leaves': [31, 50, 100],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 500, 1000],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0, 0.1, 0.5]
    }

    lgb_model = lgb.LGBMRegressor(objective='regression')
    grid_search = GridSearchCV(estimator=lgb_model, param_grid=parameters, cv=3, scoring="neg_mean_absolute_error")
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_model = grid_search.best_estimator_

    return best_params, best_score, best_model


def train_model(X_train, y_train, model_label):
    if model_label == 'xgb':
        model = train_xgboost(X_train, y_train)
    elif model_label == 'catboost':
        model = train_catboost(X_train, y_train)
    elif model_label == 'lgbm':
        model = train_lightgbm(X_train, y_train)
    return model


def grid_search_model(X_train, y_train, model_label):
    if model_label == 'xgb':
        best_params, best_score, best_model = grid_search_xgboost(X_train, y_train)
    elif model_label == 'catboost':
        best_params, best_score, best_model = grid_search_catboost(X_train, y_train)
    elif model_label == 'lgbm':
        best_params, best_score, best_model = grid_search_lightgbm(X_train, y_train)
    return best_model


# =============================================================================
# 功能：逐日递归预测
# 1）循环30次，预测未来30天
# 2）构造新特征：7，14，21，28，30，60，120天的统计值（均值，标准差，最大值，最小值等）
# =============================================================================
def predict_daily_1(ld_feature, df_true, sel_goods_ls, features, model, start_dt, window=29):
    day_30 = DayStr.n_day_delta(start_dt, n=+ window)
    df_daily = pd.DataFrame()
    log.debug(f"{start_dt},{day_30}")

    # 下一天的特征
    df_first = ld_feature.query(f"ds =='{start_dt}'").copy()
    df_first['y'] = model.predict(df_first[features]).round(1)

    df_X = df_first.copy()
    for dt in pd.date_range(start_dt, day_30):
        log.debug(f"{dt}")

        # 下下一天的特征
        df_X_2 = df_X.copy()
        df_X_2['ds'] = df_X_2['ds'] + timedelta(1)
        cols_feature = ['ds', 'wh_dept_id', 'goods_id', 'y']
        df_X_3 = pd.concat([df_X, df_X_2])[cols_feature]

        # 历史真实值 + 逐日日预测值
        df_true_new = pd.concat([df_true, df_X_3]).sort_values('ds')
        # 构造新特征
        df_feature_wh_goods = feature_generated_wh_goods(df=df_true_new, sel_goods_ls=sel_goods_ls)
        df_feature_wh_goods.query(f"ds >'{dt}'", inplace=True)

        # 预测
        df_X = df_feature_wh_goods.copy()
        df_X['y'] = model.predict(df_X[features]).round(1)
        # concat
        df_daily = pd.concat([df_daily, df_X])

    df_daily = pd.concat([df_first, df_daily])
    df_daily.query(f"'{start_dt}'<=ds<='{day_30}'", inplace=True)
    return df_daily


def predict_daily(df_feature, features, model, start_dt, window=29):
    day_30 = DayStr.n_day_delta(start_dt, n=+ window)
    log.debug(f"{start_dt},{day_30}")

    df_daily = pd.DataFrame()
    for wh_id in df_feature["wh_dept_id"].unique():
        for g_id in df_feature.query(f"wh_dept_id == {wh_id}")["goods_id"].unique():
            ld_feature_com = (df_feature.query(f"wh_dept_id == {wh_id} and  goods_id == {g_id}")
                              .reset_index(drop=True).copy())

            for dt in pd.date_range(start_dt, day_30):
                # 生成下一天的特征
                next_day_feature = ld_feature_com.iloc[-1:].copy()
                next_day_feature['ds'] = next_day_feature['ds'] + pd.DateOffset(days=1)
                next_day_feature['y'] = np.nan

                ld_feature_com = pd.concat([ld_feature_com, next_day_feature]).reset_index(drop=True)
                ld_feature_com = feature_generated(df=ld_feature_com)
                next_day_feature = ld_feature_com.iloc[-1:].copy()

                # 预测下一天的y
                next_day_y_pred = model.predict(next_day_feature[features]).round(1)
                ld_feature_com.loc[ld_feature_com['ds'] == next_day_feature['ds'].values[0], 'y'] = next_day_y_pred

            df_daily = pd.concat([df_daily, ld_feature_com])
    df_daily.query(f"'{start_dt}'<=ds<='{day_30}'", inplace=True)
    return df_daily


def get_his_true_value(data_label, sel_goods_ls, dt):
    # 过去120天的仓库货物消耗数据
    day_minus120 = DayStr.n_day_delta(dt, n=-123)

    if data_label == 'normal':
        df_stock_sell = read_api.read_dt_folder(
            bip3("model/basic_predict_promote", "stock_wh_goods_type_flg_theory_sale_cnt"),
            day_minus120, dt)
        df_stock_sell.query("type_flg =='norm' and shop_type_flg =='norm_shop'", inplace=True)
        df_stock_sell.drop(['type_flg', 'shop_type_flg'], axis=1, inplace=True)

    if data_label == 'all':
        df_stock_sell = read_api.read_dt_folder(
            bip3("model/basic_predict_promote", "stock_wh_goods_theory_sale_cnt"),
            day_minus120, dt)

    # 取绝对值
    df_stock_sell["theory_sale_cnt"] = np.abs(df_stock_sell["theory_sale_cnt"])
    df_stock_sell.query(f"goods_id in {sel_goods_ls}", inplace=True)
    df_stock_sell["dt"] = pd.to_datetime(df_stock_sell["dt"])
    df_stock_sell.rename(columns={"theory_sale_cnt": "y", "dt": "ds"}, inplace=True)

    return df_stock_sell


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


def get_feature(data_label, sel_goods_ls, start_dt, end_dt):
    if data_label == 'normal':
        ld_base = read_api.read_dt_folder(
            bip3("model/basic_predict_promote_online", "feature_engineering_wh_goods_normal")
            , start_dt, end_dt)
    if data_label == 'all':
        ld_base = read_api.read_dt_folder(
            bip3("model/basic_predict_promote_online", "feature_engineering_wh_goods_all")
            , start_dt, end_dt)

    if data_label == 'new_shop_normal':
        ld_base = read_api.read_dt_folder(
            bip3("model/basic_predict_promote_online", "feature_engineering_wh_goods_new_shop_normal")
            , start_dt, end_dt)

    ld_base.query(f"goods_id == {sel_goods_ls}", inplace=True)
    ld_base.sort_values('ds', inplace=True)
    ld_base['y'] = ld_base['y'].round(1)
    return ld_base


def change_to_price(df, df_price):
    df_price_mean = df_price.groupby('goods_id')['unit_price'].mean().round(4).rename('unit_price_mean').reset_index()
    df_price_wh = df_price.groupby('wh_dept_id')['unit_price'].mean().round(4).rename(
        'wh_unit_price_mean').reset_index()
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
                  bip_folder='model/basic_predict_promote',
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
                  bip_folder='model/basic_predict_promote',
                  output_name=f'{model_label}_wh_goods_pred_{data_label}_sum',
                  folder_dt=dt)


def save_train_df(m_train_df, model_label, data_label, dt):
    # -------------------------
    # save 训练集准确度
    cols_output = ['dt', 'model', 'MAE', 'MSE', 'MAPE', 'ACC']
    m_train_df = m_train_df[cols_output]
    bip3_save_df2(m_train_df,
                  table_folder_name=f'train_metric_{model_label}_wh_goods_pred_{data_label}',
                  bip_folder='model/basic_predict_promote',
                  output_name=f'train_metric_{model_label}_wh_goods_pred_{data_label}',
                  folder_dt=dt)


def save_metric_df(m_all_df, m_goods_df, m_wh_goods_df, model_label, data_label, time_label, dt):
    # -------------------------
    # save 时间维度++逐日递归预测准确度
    cols_output = ['dt', 'model', 'MAE', 'MSE', 'MAPE', 'ACC']
    m_all_df = m_all_df[cols_output]
    bip3_save_df2(m_all_df,
                  table_folder_name=f'metric_{time_label}_{model_label}_wh_goods_pred_{data_label}',
                  bip_folder='model/basic_predict_promote',
                  output_name=f'metric_{time_label}_{model_label}_wh_goods_pred_{data_label}',
                  folder_dt=dt)

    # -------------------------
    # save 时间维度++货物++逐日递归预测准确度
    cols_output = ['dt', 'model', 'goods_id', 'MAE', 'MSE', 'MAPE', 'ACC']
    m_goods_df = m_goods_df[cols_output]
    bip3_save_df2(m_goods_df,
                  table_folder_name=f'metric_{time_label}_goods_{model_label}_wh_goods_pred_{data_label}',
                  bip_folder='model/basic_predict_promote',
                  output_name=f'metric_{time_label}_goods_{model_label}_wh_goods_pred_{data_label}',
                  folder_dt=dt)

    # -------------------------
    # save 时间维度++仓库++货物++逐日递归预测准确度
    cols_output = ['dt', 'model', 'wh_dept_id', 'goods_id', 'MAE', 'MSE', 'MAPE', 'ACC']
    m_wh_goods_df = m_wh_goods_df[cols_output]
    bip3_save_df2(m_wh_goods_df,
                  table_folder_name=f'metric_{time_label}_wh_goods_{model_label}_wh_goods_pred_{data_label}',
                  bip_folder='model/basic_predict_promote',
                  output_name=f'metric_{time_label}_wh_goods_{model_label}_wh_goods_pred_{data_label}',
                  folder_dt=dt)


def last_wh_goods_acc(model_ls, window=3):
    import concurrent.futures

    data_label_ls = ['normal', 'all']

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

    data_label_ls = ['normal', 'all']

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
        bip3("model/basic_predict_promote", f"metric_month_wh_goods_{model_label}_wh_goods_pred_{data_label}"),
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
                  bip_folder='model/basic_predict_promote',
                  output_name=f'{model_label}_last_{window}_acc_{data_label}',
                  folder_dt=pred_minus1_day)
    return df_mean


def recent_month_goods_metric_mean(data_label, model_label, window):
    start_dt = '2023-01-01'
    pred_calc_day = DayStr.get_dt_str(None)
    pred_minus1_day = DayStr.n_day_delta(pred_calc_day, n=-1)
    # 最近3个月准确率均值
    df_metric = read_api.read_dt_folder(
        bip3("model/basic_predict_promote", f"metric_month_goods_{model_label}_wh_goods_pred_{data_label}"),
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
                  bip_folder='model/basic_predict_promote',
                  output_name=f'{model_label}_goods_last_{window}_acc_{data_label}',
                  folder_dt=pred_minus1_day)
    return df_mean


# 遍历各仓各货物
def feature_generated_wh_goods(df, sel_goods_ls):
    df_feature = pd.DataFrame()
    for (wh_dept_id, goods_id), df_current in df.groupby(["wh_dept_id", "goods_id"]):
        df_current = df_current.reset_index(drop=True)
        df_feature = pd.concat([df_feature, feature_generated(df_current)])
    return df_feature


def feature_generated(df):
        # 添加時間列
        df = df.copy().reset_index(drop=True)
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
