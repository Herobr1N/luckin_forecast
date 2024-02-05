# encoding: utf-8
# @created: 2023/11/9
# @author: yuyang.lin
import sys
import os
sys.path.insert(0, '/home/dinghuo/luckystoreordering/')
sys.path.insert(0, '/home/dinghuo/alg_dh/')


from utils_offline.a00_imports import log20 as log, DayStr, argv_date, c_path, read_api, shuttle, c_path_save_df, dop,  bip2, bip3,\
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
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.clustering import TimeSeriesKMeans


"""调用
    数据读取 GoodsData，
    特征工程 FeatureEngineer，
    树模型类 TreeModelTrainer
    回测模块（过去三个月） backtest
    
"""
from feature_engineer import FeatureEngineer
from goods_data import GoodsData
from tree_trainer import TreeModelTrainer
import backtest as backtest
import kmeans_module as kmeans_module



# 预测用函数
def predict_goods_by_labels(df_feature, features, true_label, train_ratio, model_label, goods_label, start_date):
    # 创建一个空的 DataFrame 用于存储结果
    df_country = pd.DataFrame()
    # 创建一个空的字典用于存储模型
    model_dict = {}
    # 记录带label的货物
    goods_list = goods_label["goods_id"].to_list()
    # 只保留带label的货物
    df_feature = df_feature.query(f"goods_id in {goods_list}").reset_index()
    df_feature = df_feature.merge(goods_label, on="goods_id", how="left")
    # 针对每个 label 进行训练
    for label_id, df in tqdm(df_feature.groupby(["label"]), total=len(df_feature["label"].unique()),
                             desc="Processing label"):

        xgb_trainer = TreeModelTrainer(features, true_label, train_ratio, model_label)
        df[df['ds'] <= pd.to_datetime(start_date)]  # 根据 start_date 过滤数据
        train_df, test_df, X_train, y_train, X_test, y_test = xgb_trainer.split_train_and_test(df)
        if train_df.empty:
            continue  # 如果 train_df 为空，则跳过当前 goods_id
        xgb_model = xgb_trainer.train_model(X_train, y_train)  # 训练模型
        model_dict[label_id] = xgb_model  # 将模型存储到字典中

        df_goods_country = xgb_trainer.whole_country_rolling_predict(start_date, window=30, model=xgb_model, df=df)
        df_country = pd.concat([df_country, df_goods_country])

    return df_country, model_dict


def main_labeled_goods(pred_calc_day=None):
    # 上游依赖-货物消耗数据

    ##df_stock_sell = read_api.read_dt_folder(
    ##bip3("model/basic_predict_promote_online", "stock_wh_goods_type_flg_theory_sale_cnt"),day_minus120, self.dt)

    ##df_stock_sell = read_api.read_dt_folder(
    ##bip3("model/basic_predict_promote_online", "stock_wh_goods_type_flg_theory_sale_cnt"),day_minus120, self.dt)

    ## 货物标签表读取
    ##goods_label = read_api.read_dt_folder(bip3("model/basic_predict_promote", f"predict_40_goods_{label}_label"),pred_calc_day)

    features = ['wh_dept_id', 'year', 'month', 'day', 'day_of_week', "week_of_year",
                'lag_1_mean', 'lag_7_mean', 'lag_7_std', 'lag_7_min', 'lag_7_max', 'lag_14_mean',
                'lag_14_std', 'lag_14_min', 'lag_14_max', 'lag_21_mean', 'lag_21_std',
                'lag_21_min', 'lag_21_max', 'lag_28_mean', 'lag_28_std', 'lag_28_min',
                'lag_28_max', 'lag_30_mean', 'lag_30_std', 'lag_30_min', 'lag_30_max',
                'lag_60_mean', 'lag_60_std', 'lag_60_min', 'lag_60_max', 'lag_120_mean',
                'lag_120_std', 'lag_120_min', 'lag_120_max']

    pred_calc_day = DayStr.get_dt_str(pred_calc_day)
    pred_minus1_day = DayStr.n_day_delta(pred_calc_day, n=-1)
    # 创建特征生成类
    feature_generator = FeatureEngineer()
    """从kmeans_Module.py 调用货物标签表"""
    # kmeans每日任务执行,每日落库
    kmeans_module.kmeans_main(pred_calc_day)

    # 遍历3种消耗类型 ["all","normal","new_shop_normal"]
    for label in ["all", "normal", "new_shop_normal"]:
        # 获取商品名
        goods = GoodsData(label, pred_calc_day, duration=120).goods_consume_data()
        # 货物标签表提取
        goods_label = read_api.read_dt_folder(g
            bip3("model/basic_predict_promote_online", f"predict_40_goods_{label}_label"), pred_calc_day)
        # 获取历史销售记录
        df_stock_sell = GoodsData(label, pred_calc_day, duration=120).get_his_true_value(goods)
        # 特征生成
        df_feature = feature_generator.feature_combined(df_stock_sell.reset_index(drop=True))
        # 遍历3个模型
        model_list = ["xgb", "catboost", "lgbm"]
        for model_name in model_list:
            # 预测
            df_predict, model = predict_goods_by_labels(df_feature,
                                                        features,
                                                        true_label="y",
                                                        train_ratio=1,
                                                        model_label=model_name,
                                                        goods_label=goods_label,
                                                        start_date=pred_calc_day)
            # 预测落库
            bip3_save_df2(
                df_predict,
                table_folder_name=f'predict_40_goods_{label}_{model_name}',
                bip_folder='model/basic_predict_promote',
                output_name=f"predict_40_goods_{label}_{model_name}",
                folder_dt=pred_calc_day
            )
    """调用回测模块"""
    backtest.main_acc_backtest(pred_calc_day=None)
    return None