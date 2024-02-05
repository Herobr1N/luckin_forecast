# encoding: utf-8
# @created: 2023/10/24
# @author: jieqin.lin
# @file: projects/basic_predict_promote/b_0_6_manager_pred.py


"""

生成数据

# 评选最佳模型
bip3("model/basic_predict_promote_online", "best_model_mix_boost_1_wh_goods_pred_normal_group_b")                      常规品过去1个月（仓库货物）最佳模型（acc均值最大）
bip3("model/basic_predict_promote_online", "best_model_mix_boost_1_wh_goods_pred_all_group_b")                         全量过去1个月（仓库货物）最佳模型（acc均值最大）
bip3("model/basic_predict_promote_online", "best_model_mix_boost_1_wh_goods_pred_new_shop_normal_group_b")             新店常规过去1个月（仓库货物）最佳模型（acc均值最大）


# 预测
bip3("model/basic_predict_promote_online", "mix_boost_1_wh_goods_pred_normal_group_b"")                         常规用过去1个月最佳模型预测的结果
bip3("model/basic_predict_promote_online", "mix_boost_1_wh_goods_pred_all_group_b"")                            全量用过去1个月最佳模型预测的结果
bip3("model/basic_predict_promote_online", "mix_boost_1_wh_goods_pred_new_shop_normal_group_b"")                新店常规用过去1个月最佳模型预测的结果


# 日维度 准确度
bip3("model/basic_predict_promote_online", "metric_daily_mix_boost_1_wh_goods_pred_normal")            常规品天维度准确度
bip3("model/basic_predict_promote_online", "metric_daily_goods_mix_boost_1_wh_goods_pred_normal")      常规品天维度货物级别准确度
bip3("model/basic_predict_promote_online", "metric_daily_wh_goods_mix_boost_1_wh_goods_pred_normal")   常规品天维度仓库货物级别准确度

bip3("model/basic_predict_promote_online", "metric_daily_mix_boost_1_wh_goods_pred_all)                总天维度准确度
bip3("model/basic_predict_promote_online", "metric_daily_goods_mix_boost_1_wh_goods_pred_all)          总天维度货物级别准确度
bip3("model/basic_predict_promote_online", "metric_daily_wh_goods_mix_boost_1_wh_goods_pred_all)       总天维度仓库货物级别准确度

# 月维度 准确度
bip3("model/basic_predict_promote_online", "metric_month_mix_boost_1_wh_goods_pred_normal")            常规品月维度准确度
bip3("model/basic_predict_promote_online", "metric_month_goods_mix_boost_1_wh_goods_pred_normal")      常规品月维度货物级别准确度
bip3("model/basic_predict_promote_online", "metric_month_wh_goods_mix_boost_1_wh_goods_pred_normal")   常规品月维度仓库货物级别准确度

bip3("model/basic_predict_promote_online", "metric_month_mix_boost_1_wh_goods_pred_all")               总月维度准确度
bip3("model/basic_predict_promote_online", "metric_month_goods_mix_boost_1_wh_goods_pred_all")         总月维度货物级别准确度
bip3("model/basic_predict_promote_online", "metric_month_wh_goods_mix_boost_1_wh_goods_pred_all")      总月维度仓库货物级别准确度

依赖数据
bip3("model/basic_predict_promote_online", "xgb_goods_last_1_acc_normal")                          过去1个月（仓库货物）acc均值最大
bip3("model/basic_predict_promote_online", "lgbm_goods_last_1_acc_normal")                         过去1个月（仓库货物）acc均值最大
bip3("model/basic_predict_promote_online", "catboost_goods_last_1_acc_normal")                     过去1个月（仓库货物）acc均值最大

"""
from __init__ import project_path
import pandas as pd
from projects.basic_predict_promote.b_0_0_utils_models import ModelManager,SelectGoodsList
from utils_offline.a00_imports import DayStr, bip3_save_df2, argv_date

f"Import from {project_path}"


def main_thread_pool_mix_wh_goods_mode(pred_calculation_day=None):
    """
    多线程
    """
    import concurrent.futures
    import itertools

    pred_calc_day = DayStr.get_dt_str(pred_calculation_day)

    # 参数
    model_ls = ['mix_boost_1']
    data_label_ls = ['normal', 'all', 'new_shop_normal']

    selector = SelectGoodsList()
    sel_all_goods_ls = selector.get_group_all_goods_ls()

    def process_combination(combination):
        model_label, data_label, (goods_label, sel_goods_ls) = combination
        print(f"Processing: {model_label}, {data_label}, {goods_label}, 货物数{len(sel_goods_ls)}")

        try:
            main_train_mix_wh_goods_model(target_date=pred_calc_day,
                                          model_label=model_label,
                                          data_label=data_label,
                                          goods_label=goods_label)
        except Exception as e:
            print(f"Failed: {str(e)}")

    combinations = itertools.product(model_ls, data_label_ls, zip(sel_all_goods_ls.keys(), sel_all_goods_ls.values()))

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        executor.map(process_combination, combinations)


def main_train_mix_wh_goods_model(target_date, model_label, data_label,goods_label):
    """
    模型管理器
    """
    # ------------------
    if model_label == 'mix_boost_1':
        window = 1

    # ------------------
    df_acc = pd.DataFrame()
    df_daily_pred = pd.DataFrame()
    boost_model_label = ['xgb', 'lgbm', 'catboost']

    for boost_model in boost_model_label:
        manager = ModelManager(model_label=boost_model, data_label=data_label,goods_label=goods_label)
        # 获取上个月月初的准确率
        ld_acc = manager.get_last_acc(window=window)
        df_acc = pd.concat([df_acc, ld_acc])
        # 获取预测
        ld_pred = manager.get_pred(dt=target_date)
        df_daily_pred = pd.concat([df_daily_pred, ld_pred])

    df_daily_pred.rename(columns={'predict_demand': 'pred', 'predict_dt': 'ds'}, inplace=True)
    df_daily_pred["ds"] = pd.to_datetime(df_daily_pred["ds"])
    df_daily_pred["dt"] = pd.to_datetime(df_daily_pred["dt"])

    # ------------------
    # 初始化
    manager_best = ModelManager(model_label=model_label, data_label=data_label,goods_label=goods_label)

    # 最佳模型 和 对应的预测
    df_max = df_acc.groupby(['wh_dept_id', 'goods_id'])[f'last_{window}_acc'].max().reset_index()
    cols_max_model = ['wh_dept_id', 'goods_id', 'model', f'last_{window}_acc']
    df_max_model = df_max.merge(df_acc)[cols_max_model]
    df_max_model.sort_values(['wh_dept_id', 'goods_id'], inplace=True)

    df_best_model = pd.DataFrame()
    for wh_id in df_max_model["wh_dept_id"].unique():
        for g_id in df_max_model.query(f"wh_dept_id == {wh_id}")["goods_id"].unique():
            df_wh_goods = (df_max_model.query(f"wh_dept_id == {wh_id} and  goods_id == {g_id}")
                           .reset_index(drop=True).copy())
            df_wh_goods['best_model'] = df_wh_goods['model']
            df_best_model = pd.concat([df_best_model, df_wh_goods])
    df_best_model = df_best_model.drop(['model'], axis=1).rename(columns={'best_model': 'model'})

    bip3_save_df2(df_best_model,
                  table_folder_name=f'best_model_{model_label}_wh_goods_pred_{data_label}_{goods_label}',
                  bip_folder='model/basic_predict_promote_online',
                  output_name=f'best_model_{model_label}_wh_goods_pred_{data_label}_{goods_label}',
                  folder_dt=target_date)

    # ------------------
    # 混合预测
    df_daily_pred = df_best_model.merge(df_daily_pred)
    df_month_pred = df_daily_pred.groupby(['wh_dept_id', 'goods_id']).agg({'pred': 'sum'}).round(
        1).reset_index()

    manager_best.save_pred_df(df_daily_pred=df_daily_pred, df_month_pred=df_month_pred, dt=target_date)


if __name__ == '__main__':
    argv_date(main_thread_pool_mix_wh_goods_mode)