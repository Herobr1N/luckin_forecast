# encoding: utf-8
# @created: 2023/10/9
# @author: jieqin.lin
# @file: projects/basic_predict_promote/b_0_4_inference_tree_models.py

"""
树模型 Xgboost LightGBM Catboost  预测仓库货物未来30天消耗

训练 + 推理：  训练 bpp03_pred_tree_models， 推理bpp04_inference_tree_models

生成数据

############################################
Xgboost
# 预测
bip3("model/basic_predict_promote_online", "xgb_wh_goods_pred_new_shop_normal")                预测新店消耗量
bip3("model/basic_predict_promote_online", "xgb_wh_goods_pred_normal")                         预测常规消耗量
bip3("model/basic_predict_promote_online", "xgb_wh_goods_pred_all")                            预测总消耗量

# 训练
bip3("model/basic_predict_promote_online", "train_metric_xgb_wh_goods_pred_new_shop_normal")   新店训练集准确度
bip3("model/basic_predict_promote_online", "train_metric_xgb_wh_goods_pred_normal")            常规品训练集准确度
bip3("model/basic_predict_promote_online", "train_metric_xgb_wh_goods_pred_all")               总训练集准确度

############################################
LightGBM
# 预测
bip3("model/basic_predict_promote_online", "lgbm_wh_goods_pred_new_shop_normal")                预测新店常规消耗量
bip3("model/basic_predict_promote_online", "lgbm_wh_goods_pred_normal")                         预测常规消耗量
bip3("model/basic_predict_promote_online", "lgbm_wh_goods_pred_all")                            预测总消耗量

# 训练
bip3("model/basic_predict_promote_online", "train_metric_lgbm_wh_goods_pred_new_shop_normal")   新店常规品训练集准确度
bip3("model/basic_predict_promote_online", "train_metric_lgbm_wh_goods_pred_normal")            常规品训练集准确度
bip3("model/basic_predict_promote_online", "train_metric_lgbm_wh_goods_pred_all")               总训练集准确度


############################################
catboost
# 预测
bip3("model/basic_predict_promote_online", "catboost_wh_goods_pred_new_shop_normal")        预测新店常规消耗量
bip3("model/basic_predict_promote_online", "catboost_wh_goods_pred__normal)                 预测常规消耗量
bip3("model/basic_predict_promote_online", "catboost_wh_goods_pred_all")                    预测总消耗量

# 训练
bip3("model/basic_predict_promote_online", "train_metric_catboost_wh_goods_pred_new_shop_normal")   新店常规品训练集准确度
bip3("model/basic_predict_promote_online", "train_metric_catboost_wh_goods_pred_normal")            常规品训练集准确度
bip3("model/basic_predict_promote_online", "train_metric_catboost_wh_goods_pred_all")               总训练集准确度

############################################
依赖数据
bip3("model/basic_predict_promote_online", "stock_wh_goods_type_flg_theory_sale_cnt")
bip3("model/basic_predict_promote_online", "stock_wh_goods_theory_sale_cnt")
bip3("model/basic_predict_promote_online", "feature_engineering_normal")
bip3("model/basic_predict_promote_online", "feature_engineering_all")
dw_dws.dws_stock_warehouse_stock_adjust_d_inc_summary
"""

from __init__ import project_path
import numpy as np
import pandas as pd
from projects.basic_predict_promote.b_0_0_utils_models import TreeModelTrainer, SelectGoodsList, FeatureEngineer
from utils_offline.a00_imports import log20 as log, DayStr, read_api, argv_date, bip3
import concurrent.futures
import itertools

f"Import from {project_path}"


# =============================================================================
# 功能：基于清洗后仓库货物的历史数据，分别计算Xgboost模型，LightGBM模型和Catboost模型的预测值
# 1）因跟随，想要取最近的数据，不划分测试训练
# 2）测试货物: 冰凉感厚椰饮品，冷萃厚牛乳, 北海道丝绒风味厚乳
# 3）逐日递归预测: 每次预测未来1天，根据预测值，构造新的统计特征，输入模型预测下一天
# 4）多线程预测
# 5) 如果有GPU可以做增量训练
# =============================================================================


def main_inference_tree_models_wh_goods(pred_calculation_day=None):
    """
    ## 历史回测 方法一
    from projects.basic_predict_promote.bpp03_pred_tree_models import *
    for dt in pd.date_range("2023-10-07","2023-11-07"):
        main_inference_tree_models_wh_goods(dt.strftime("%Y-%m-%d"))

    多线程跑：3类数据(全量，常规品,新店) 和 3个模型（'xgb', 'lgbm', 'catboost')
    """
    pred_calc_day = DayStr.get_dt_str(pred_calculation_day)

    # -------------------
    # 参数
    model_ls = ['xgb', 'lgbm', 'catboost']
    data_label_ls = ['normal', 'all', 'new_shop_normal']
    window = 120
    max_workers = 2

    # ----------------------
    # 货物范围:{group_b,group_c,group_d,group_e,group_f,group_g}
    selector = SelectGoodsList()
    sel_all_goods_ls = selector.get_group_all_goods_ls()

    # -------------------
    # 并行
    def process_tree_combination(combination):
        model_label, data_label, (goods_label, sel_goods_ls) = combination
        print(f"Processing: {model_label}, {data_label}, {goods_label}, 货物数{len(sel_goods_ls)}")
        try:
            inference_tree_models(
                target_date=pred_calc_day,
                model_label=model_label,
                data_label=data_label,
                window=window,
                goods_label=goods_label,
                sel_goods_ls=sel_goods_ls)
        except Exception as e:
            log.debug(f"失败: {str(e)}")

    combinations = itertools.product(model_ls, data_label_ls, zip(sel_all_goods_ls.keys(), sel_all_goods_ls.values()))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_tree_combination, combinations)


def inference_tree_models(target_date, model_label, data_label, window, goods_label, sel_goods_ls):
    """
    逐日递归
    # =============================================================================
    # 功能：保存各个模型的预测数据
    # 1）单日预测
    # 2）预测的30天之和
    # 3）评估（仓库货物，货物，全国维度）
    # =============================================================================

    """
    # start_dt = '2020-01-01'
    start_dt = DayStr.n_day_delta(target_date, n=-183)
    pred_minus30_day = DayStr.n_day_delta(target_date, n=-30)
    pred_minus1_day = DayStr.n_day_delta(target_date, n=-1)

    # ----------------------
    print(f"Processing: {model_label}, {data_label}, {goods_label}, 货物数{len(sel_goods_ls)}")

    # 特征
    features = ['wh_dept_id', 'goods_id', 'year', 'month', 'day', 'day_of_week', 'week_of_year',
                'lag_1_mean', 'lag_7_mean', 'lag_7_std', 'lag_7_min', 'lag_7_max', 'lag_14_mean',
                'lag_14_std', 'lag_14_min', 'lag_14_max', 'lag_21_mean', 'lag_21_std',
                'lag_21_min', 'lag_21_max', 'lag_28_mean', 'lag_28_std', 'lag_28_min',
                'lag_28_max', 'lag_30_mean', 'lag_30_std', 'lag_30_min', 'lag_30_max',
                'lag_60_mean', 'lag_60_std', 'lag_60_min', 'lag_60_max', 'lag_120_mean',
                'lag_120_std', 'lag_120_min', 'lag_120_max']

    # 初始化训练器
    trainer = TreeModelTrainer(features,
                               true_label="y",
                               train_ratio=1,
                               model_label=model_label,
                               data_label=data_label,
                               goods_label=goods_label)

    # 特征，筛选数据 取昨天
    feature_generator = FeatureEngineer(data_label=data_label)
    ld_feature = feature_generator.get_feature_wh_goods(start_date=pred_minus30_day,
                                                        end_date=pred_minus1_day,
                                                        sel_goods_ls=sel_goods_ls)

    df_feature = ld_feature.query(f"'{start_dt}'<= ds <='{pred_minus1_day}'")

    # 训练
    model = read_api.read_dt_pickle(
        bip3("model/basic_predict_promote_online", f"model_{model_label}_wh_goods_pred_{data_label}_{goods_label}"),
        target_date)

    # -------------------------
    log.debug(f"{model_label}, {data_label}, {goods_label}, 货物数{len(sel_goods_ls)}未来逐日递归模型训练")
    df_daily_pred = trainer.whole_country_rolling_predict(start_date=target_date,
                                                          window=window,
                                                          model=model,
                                                          df=df_feature,
                                                          label=f"{model_label}_{data_label}"
                                                          )

    cols_daily = ['ds', 'wh_dept_id', 'goods_id', 'y']
    df_daily_pred = df_daily_pred[cols_daily].rename(columns={'y': 'pred'})

    #  预测不为负
    df_daily_pred["pred"] = np.clip(df_daily_pred["pred"], 0, np.inf)

    # 30天之和
    df_month_pred = (df_daily_pred.groupby(['wh_dept_id', 'goods_id'])
                     .agg({'pred': 'sum'})
                     .round(1).reset_index()
                     )
    # save 预测
    trainer.save_pred_df(df_daily_pred=df_daily_pred,
                         df_month_pred=df_month_pred,
                         dt=target_date)


if __name__ == '__main__':
    argv_date(main_inference_tree_models_wh_goods)
