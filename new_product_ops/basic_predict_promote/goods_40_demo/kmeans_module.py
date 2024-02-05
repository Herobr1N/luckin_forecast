# encoding: utf-8
# @created: 2023/11/9
# @author: yuyang.lin
import sys
import os

sys.path.insert(0, '/home/dinghuo/luckystoreordering/')
sys.path.insert(0, '/home/dinghuo/alg_dh/')

from utils_offline.a00_imports import log20 as log, DayStr, argv_date, c_path, read_api, shuttle, c_path_save_df, dop, \
    bip2, bip3, \
    hive_table_hdfs_path, list_info_hdfs_path, upload_file_to_hdfs_path, shuttle, bip3_save_df2, dfu, bip1
from areas.table_info.dh_dw_table_info import dh_dw
import numpy as np
from datetime import timedelta
from datetime import datetime
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.clustering import TimeSeriesKMeans

"""
依赖goodsData类获取
"""
from goods_data import GoodsData

"""聚类"""


def kmeans(df_stock_sell):
    df_country = df_stock_sell.groupby(["ds", "goods_id"]).agg({"y": "sum"}).reset_index()
    df_country = df_country.pivot(index='goods_id', columns='ds', values='y')
    # 剔除过去120天的新品和下市品(带缺失数据)
    df_country = df_country.dropna().reset_index()

    # 剔除一期货物 [19952, 27954, 23628] & 单独分类货物[660, 27942]
    goods_to_remove = [660, 19952, 27954, 23628, 27942]
    df_country = df_country.query(f"goods_id not in {goods_to_remove}").reset_index(drop=True)
    # 列名转str
    df_country.columns = df_country.columns.astype(str)
    columns_to_scale = [col for col in df_country.columns if col != 'goods_id']
    # 只用消耗做训练
    data_to_scale = df_country[columns_to_scale]
    scaler = TimeSeriesScalerMinMax()

    # Scale the data using the scaler
    scaled_data = scaler.fit_transform(data_to_scale)
    # Create an instance of TimeSeriesKMeans for each iteration
    kmeans = TimeSeriesKMeans(n_clusters=2, random_state=42)

    # Fit the model to the scaled data
    kmeans.fit(scaled_data)
    #
    labels = kmeans.labels_
    # Reshape the data to 2D
    n_samples, n_timesteps, n_features = scaled_data.shape
    reshaped_data = np.reshape(scaled_data, (n_samples, n_timesteps * n_features))
    df_country["label"] = labels
    return df_country[["goods_id", "label"]]


"""
kmeans任务落库
"""
def kmeans_main(pred_calc_day=None):
    pred_calc_day = DayStr.get_dt_str(pred_calc_day)
    pred_minus1_day = DayStr.n_day_delta(pred_calc_day, n=-1)

    for label in ["all", "normal", "new_shop_normal"]:
        try:
            # 获取商品名
            goods = GoodsData(label, "2023-11-01", duration=180).goods_consume_data()
            # 获取历史销售记录用于kmeans训练
            df_stock_sell_kmeans = GoodsData(label, "2023-11-01", 180).get_his_true_value(goods)
            # kmeans 获取标签
            goods_label = kmeans(df_stock_sell_kmeans)

            bip3_save_df2(
                goods_label,
                table_folder_name=f'predict_40_goods_{label}_label',
                bip_folder='model/basic_predict_promote_online',
                output_name=f"predict_40_goods_{label}_label",
                folder_dt=pred_calc_day
            )
        # t-1 降级
        except:
            df_yesterday = read_api.read_dt_folder(
                bip3("model/basic_predict_promote", f"predict_40_goods_{label}_label"),
                pred_minus1_day)

            bip3_save_df2(
                df_yesterday,
                table_folder_name=f'predict_40_goods_{label}_label',
                bip_folder='model/basic_predict_promote_online',
                output_name=f"predict_40_goods_{label}_label",
                folder_dt=pred_calc_day

    return None