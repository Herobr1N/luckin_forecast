# encoding: utf-8
# @created: 2023/11/8
# @author: jieqin.lin
# @file: projects/basic_predict_promote/b_0_5_metric_pred.py


"""
## 评估模型

生成数据
############################################

(仓库货物)总预测过去1个月准确率均值

bip3("model/basic_predict_promote_online", "xgb_last_1_acc_all")
bip3("model/basic_predict_promote_online", "lgbm_last_1_acc_all")
bip3("model/basic_predict_promote_online", "catboost_last_1_acc_all")

(仓库货物)常规预测过去1个月准确率均值
bip3("model/basic_predict_promote_online", "xgb_last_1_acc_normal")
bip3("model/basic_predict_promote_online", "lgbm_last_1_acc_normal")
bip3("model/basic_predict_promote_online", "catboost_last_1_acc_normal")

############################################

Xgboost

# 逐日
bip3("model/basic_predict_promote_online", "metric_daily_xgb_wh_goods_pred_new_shop_normal")            新店天维度准确度
bip3("model/basic_predict_promote_online", "metric_daily_goods_xgb_wh_goods_pred_new_shop_normal")      新店天维度货物级别准确度
bip3("model/basic_predict_promote_online", "metric_daily_wh_goods_xgb_wh_goods_pred_new_shop_normal")   新店天维度仓库货物级别准确度

bip3("model/basic_predict_promote_online", "metric_daily_xgb_wh_goods_pred_normal")            常规品天维度准确度
bip3("model/basic_predict_promote_online", "metric_daily_goods_xgb_wh_goods_pred_normal")      常规品天维度货物级别准确度
bip3("model/basic_predict_promote_online", "metric_daily_wh_goods_xgb_wh_goods_pred_normal")   常规品天维度仓库货物级别准确度

bip3("model/basic_predict_promote_online", "metric_daily_xgb_wh_goods_pred_all)                总天维度准确度
bip3("model/basic_predict_promote_online", "metric_daily_goods_xgb_wh_goods_pred_all)          总天维度货物级别准确度
bip3("model/basic_predict_promote_online", "metric_daily_wh_goods_xgb_wh_goods_pred_all)       总天维度仓库货物级别准确度

# 逐日汇总到月度
bip3("model/basic_predict_promote_online", "metric_month_xgb_wh_goods_pred_new_shop_normal")            新店月维度准确度
bip3("model/basic_predict_promote_online", "metric_month_goods_xgb_wh_goods_pred_new_shop_normal")      新店月维度货物级别准确度
bip3("model/basic_predict_promote_online", "metric_month_wh_goods_xgb_wh_goods_pred_new_shop_normal")   新店月维度仓库货物级别准确度

bip3("model/basic_predict_promote_online", "metric_month_xgb_wh_goods_pred_normal")            常规品月维度准确度
bip3("model/basic_predict_promote_online", "metric_month_goods_xgb_wh_goods_pred_normal")      常规品月维度货物级别准确度
bip3("model/basic_predict_promote_online", "metric_month_wh_goods_xgb_wh_goods_pred_normal")   常规品月维度仓库货物级别准确度

bip3("model/basic_predict_promote_online", "metric_month_xgb_wh_goods_pred_all")               总月维度准确度
bip3("model/basic_predict_promote_online", "metric_month_goods_xgb_wh_goods_pred_all")         总月维度货物级别准确度
bip3("model/basic_predict_promote_online", "metric_month_wh_goods_xgb_wh_goods_pred_all")      总月维度仓库货物级别准确度

############################################
LightGBM
# 逐日

bip3("model/basic_predict_promote_online", "metric_daily_lgbm_wh_goods_pred_normal")            常规品天维度准确度
bip3("model/basic_predict_promote_online", "metric_daily_goods_lgbm_wh_goods_pred_normal")      常规品天维度货物级别准确度
bip3("model/basic_predict_promote_online", "metric_daily_wh_goods_lgbm_wh_goods_pred_normal")   常规品天维度仓库货物级别准确度

bip3("model/basic_predict_promote_online", "metric_daily_lgbm_wh_goods_pred_all)                总天维度准确度
bip3("model/basic_predict_promote_online", "metric_daily_goods_lgbm_wh_goods_pred_all)          总天维度货物级别准确度
bip3("model/basic_predict_promote_online", "metric_daily_wh_goods_lgbm_wh_goods_pred_all)       总天维度仓库货物级别准确度

bip3("model/basic_predict_promote_online", "metric_daily_lgbm_wh_goods_pred_new_shop_normal")           新店常规品天维度准确度
bip3("model/basic_predict_promote_online", "metric_daily_goods_lgbm_wh_goods_pred_new_shop_normal")      新店常规品天维度货物级别准确度
bip3("model/basic_predict_promote_online", "metric_daily_wh_goods_lgbm_wh_goods_pred_new_shop_normal")   新店常规品天维度仓库货物级别准确度

# 逐日汇总到月度
bip3("model/basic_predict_promote_online", "metric_month_lgbm_wh_goods_pred_normal")            常规品月维度准确度
bip3("model/basic_predict_promote_online", "metric_month_goods_lgbm_wh_goods_pred_normal")      常规品月维度货物级别准确度
bip3("model/basic_predict_promote_online", "metric_month_wh_goods_lgbm_wh_goods_pred_normal")   常规品月维度仓库货物级别准确度

bip3("model/basic_predict_promote_online", "metric_month_lgbm_wh_goods_pred_all")               总月维度准确度
bip3("model/basic_predict_promote_online", "metric_month_goods_lgbm_wh_goods_pred_all")         总月维度货物级别准确度
bip3("model/basic_predict_promote_online", "metric_month_wh_goods_lgbm_wh_goods_pred_all")      总月维度仓库货物级别准确度

bip3("model/basic_predict_promote_online", "metric_month_lgbm_wh_goods_pred_new_shop_normal")            新店常规品月维度准确度
bip3("model/basic_predict_promote_online", "metric_month_goods_lgbm_wh_goods_pred_new_shop_normal")      新店常规品月维度货物级别准确度
bip3("model/basic_predict_promote_online", "metric_month_wh_goods_lgbm_wh_goods_pred_new_shop_normal")   新店常规品月维度仓库货物级别准确度

############################################
catboost

# 逐日
bip3("model/basic_predict_promote_online", "metric_daily_catboost_wh_goods_pred_normal")            常规品天维度准确度
bip3("model/basic_predict_promote_online", "metric_daily_goods_catboost_wh_goods_pred_normal")      常规品天维度货物级别准确度
bip3("model/basic_predict_promote_online", "metric_daily_wh_goods_catboost_wh_goods_pred_normal")   常规品天维度仓库货物级别准确度

bip3("model/basic_predict_promote_online", "metric_daily_catboost_wh_goods_pred_all"))                总天维度准确度
bip3("model/basic_predict_promote_online", "metric_daily_goods_catboost_wh_goods_pred_all"))          总天维度货物级别准确度
bip3("model/basic_predict_promote_online", "metric_daily_wh_goods_catboost_wh_goods_pred_all"))       总天维度仓库货物级别准确度


bip3("model/basic_predict_promote_online", "metric_daily_catboost_wh_goods_pred_new_shop_normal"))                新店常规天维度准确度
bip3("model/basic_predict_promote_online", "metric_daily_goods_catboost_wh_goods_pred_new_shop_normal"))          新店常规天维度货物级别准确度
bip3("model/basic_predict_promote_online", "metric_daily_wh_goods_catboost_wh_goods_pred_new_shop_normal"))       新店常规天维度仓库货物级别准确度


# 逐日汇总到月度
bip3("model/basic_predict_promote_online", "metric_month_catboost_wh_goods_pred_normal")            常规品月维度准确度
bip3("model/basic_predict_promote_online", "metric_month_goods_catboost_wh_goods_pred_normal")      常规品月维度货物级别准确度
bip3("model/basic_predict_promote_online", "metric_month_wh_goods_catboost_wh_goods_pred_normal")   常规品月维度仓库货物级别准确度

bip3("model/basic_predict_promote_online", "metric_month_catboost_wh_goods_pred_all")               总月维度准确度
bip3("model/basic_predict_promote_online", "metric_month_goods_catboost_wh_goods_pred_all")         总月维度货物级别准确度
bip3("model/basic_predict_promote_online", "metric_month_wh_goods_catboost_wh_goods_pred_all")      总月维度仓库货物级别准确度

bip3("model/basic_predict_promote_online", "metric_month_catboost_wh_goods_pred_new_shop_normal")               新店常规月维度准确度
bip3("model/basic_predict_promote_online", "metric_month_goods_catboost_wh_goods_pred_new_shop_normal")         新店常规月维度货物级别准确度
bip3("model/basic_predict_promote_online", "metric_month_wh_goods_catboost_wh_goods_pred_new_shop_normal")      新店常规月维度仓库货物级别准确度

############################################

依赖数据

# 预测
bip3("model/basic_predict_promote_online", "xgb_wh_goods_pred_all")
bip3("model/basic_predict_promote_online", "lgbm_wh_goods_pred_all")
bip3("model/basic_predict_promote_online", "catboost_wh_goods_pred_all")

bip3("model/basic_predict_promote_online", "xgb_wh_goods_pred_normal")
bip3("model/basic_predict_promote_online", "lgbm_wh_goods_pred_normal")
bip3("model/basic_predict_promote_online", "catboost_wh_goods_pred_normal")

bip3("model/basic_predict_promote_online", "xgb_wh_goods_pred_new_shop_normal")
bip3("model/basic_predict_promote_online", "lgbm_wh_goods_pred_new_shop_normal")
bip3("model/basic_predict_promote_online", "catboost_wh_goods_pred_new_shop_normal")

"""
from __init__ import project_path
from datetime import datetime, timedelta
import pandas as pd
from projects.basic_predict_promote.b_0_0_utils_models import FeatureEngineer, SelectGoodsList, BaseModelTrainer
from utils_offline.a00_imports import log20 as log, read_api, bip3_save_df2, bip3, argv_date, DayStr

f"Import from {project_path}"


def main_thread_pool_metric_month_wh_goods_acc(pred_calculation_day=None):
    import concurrent.futures
    import itertools

    pred_calc_day = DayStr.get_dt_str(pred_calculation_day)
    pred_minus30_day = DayStr.n_day_delta(pred_calc_day, n=-30)
    # 评估
    model_ls = ['xgb', 'lgbm', 'catboost']
    data_label_ls = ['normal', 'all', 'new_shop_normal']
    window = 1
    # ----------------------
    # 货物范围 :{group_b,group_c,group_d,group_e,group_f}
    selector = SelectGoodsList()
    sel_all_goods_ls = selector.get_group_all_goods_ls()

    def process_combination(combination):
        model_label, data_label, (goods_label, sel_goods_ls) = combination
        log.debug(
            f"Processing: {model_label}, {data_label}, {goods_label}, 货物数{len(sel_goods_ls)}, {pred_calc_day}, {window}")
        try:
            # 评估每天生成的30天预测
            # metric_month_wh_goods_pred(target_date=pred_minus30_day, model_label=model_label, data_label=data_label,
            #                            goods_label=goods_label, sel_goods_ls=sel_goods_ls)
            # 最近的acc均值
            last_month_acc_mean(target_date=pred_calc_day, model_label=model_label,
                                data_label=data_label, goods_label=goods_label,
                                window=window)
        except Exception as e:
            print(f"失败: {str(e)}")

    combinations = itertools.product(model_ls, data_label_ls, zip(sel_all_goods_ls.keys(), sel_all_goods_ls.values()))
    with concurrent.futures.ThreadPoolExecutor(max_workers=9) as executor:
        executor.map(process_combination, combinations)


def metric_month_wh_goods_pred(target_date, model_label, data_label,goods_label, sel_goods_ls):
    # 初始化类
    base_trainer = BaseModelTrainer(true_label='y', model_label=model_label, data_label=data_label,goods_label=goods_label)
    feature_generator = FeatureEngineer(data_label=data_label)

    # 预测值
    df_daily_pred = read_api.read_one_folder(
        bip3("model/basic_predict_promote_online", f"{model_label}_wh_goods_pred_{data_label}_{goods_label}", target_date))

    cols_daily = ['predict_dt', 'wh_dept_id', 'goods_id', 'predict_demand']
    df_daily_pred = df_daily_pred[cols_daily].rename(columns={'predict_demand': 'pred', 'predict_dt': 'ds'})
    df_daily_pred['ds'] = pd.to_datetime(df_daily_pred['ds'])
    # 历史售卖
    start_date = target_date
    end_date = DayStr.n_day_delta(target_date, n=30)

    # 真实值
    ld_true = feature_generator.get_his_stock_wh_goods(start_date=start_date,
                                                       end_date=end_date,
                                                       sel_goods_ls=sel_goods_ls)
    # merge
    dfc_metric = ld_true.merge(df_daily_pred)

    # 单日准确率
    daily_all, daily_goods, daily_wh_goods = base_trainer.evaluate_model_wh_goods(df=dfc_metric, dt=target_date)

    # save 单日准确率评估
    base_trainer.save_metric_df(
        m_all_df=daily_all,
        m_goods_df=daily_goods,
        m_wh_goods_df=daily_wh_goods,
        time_label='daily',
        dt=target_date)

    # 30天之和准确率
    df_month_pred = (dfc_metric.groupby(['wh_dept_id', 'goods_id'])
                     .agg({'y': 'sum', 'pred': 'sum'})
                     .round(1).reset_index()
                     )
    month_all, month_goods, month_wh_goods = base_trainer.evaluate_model_wh_goods(df=df_month_pred, dt=target_date)

    # save 月度准确率评估
    base_trainer.save_metric_df(
        m_all_df=month_all,
        m_goods_df=month_goods,
        m_wh_goods_df=month_wh_goods,
        time_label='month',
        dt=target_date)


def last_month_acc_mean(target_date, model_label, data_label, goods_label, window):
    # 每月第一天
    date_ls = [datetime(year, month, 1).strftime('%Y-%m-%d') for year in range(2023, 2024) for month in range(1, 13)]

    # -------------------
    # 获取评估
    df_metric = pd.DataFrame()
    for dt in date_ls:
        previous_month = datetime.strptime(target_date, '%Y-%m-%d').replace(day=1) - timedelta(days=1)
        previous_month = previous_month.strftime("%Y-%m-%d")
        if dt < previous_month:
            # 最近N个月准确率均值
            ld_metric = read_api.read_dt_folder(
                bip3(f"model/basic_predict_promote_{goods_label}",
                     f"metric_month_wh_goods_{model_label}_wh_goods_pred_{data_label}"),
                dt)
            df_metric = pd.concat([df_metric, ld_metric])
    df_metric['dt'] = pd.to_datetime(df_metric['dt'])

    # -------------------
    # 求均
    df_mean = pd.DataFrame()
    for wh_id, group in df_metric.groupby(['wh_dept_id', 'goods_id']):
        df_wh_goods = group.reset_index(drop=True).copy()
        df_wh_goods[f'last_{window}_mse'] = df_wh_goods['MSE'].rolling(window=window, min_periods=0).mean().round(4)
        df_wh_goods[f'last_{window}_mae'] = df_wh_goods['MAE'].rolling(window=window, min_periods=0).mean().round(4)
        df_wh_goods[f'last_{window}_mape'] = df_wh_goods['MAPE'].rolling(window=window, min_periods=0).mean().round(4)
        df_wh_goods[f'last_{window}_acc'] = df_wh_goods['ACC'].rolling(window=window, min_periods=0).mean().round(4)
        df_wh_goods[f'last_{window}_std'] = df_wh_goods['ACC'].rolling(window=window, min_periods=0).std().round(4)
        df_mean = pd.concat([df_mean, df_wh_goods])
    df_mean['model'] = model_label

    # -------------------
    # save
    cols_output = ['wh_dept_id', 'goods_id', 'model', f'last_{window}_mse', f'last_{window}_mae', f'last_{window}_mape',
                   f'last_{window}_acc', f'last_{window}_std', 'dt']
    df_mean['dt'] = df_mean['dt'].dt.strftime('%Y-%m-%d')
    df_mean = df_mean[cols_output]
    bip3_save_df2(df_mean,
                  table_folder_name=f'{model_label}_last_{window}_acc_{data_label}_{goods_label}',
                  bip_folder='model/basic_predict_promote_online',
                  output_name=f'{model_label}_last_{window}_acc_{data_label}_{goods_label}',
                  folder_dt=target_date)
    return df_mean


if __name__ == '__main__':
    argv_date(main_thread_pool_metric_month_wh_goods_acc)
