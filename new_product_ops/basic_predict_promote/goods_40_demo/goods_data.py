# encoding: utf-8
# @created: 2023/11/9
# @author: yuyang.lin

import sys
import os

sys.path.insert(0, '/home/dinghuo/alg_dh/')
sys.path.insert(0, '/home/dinghuo/luckystoreordering/')
from __init__ import project_path
from areas.table_info.dh_dw_table_info import dh_dw
from utils_offline.a00_imports import dfu, log20 as log, DayStr, read_api, shuttle, bip3_save_df2, bip3, bip2

from datetime import timedelta
from datetime import datetime

import numpy as np
import pandas as pd

# encoding: utf-8
# @created: 2023/11/9
# @author: yuyang.lin

import sys
import os

sys.path.insert(0, '/home/dinghuo/alg_dh/')
sys.path.insert(0, '/home/dinghuo/luckystoreordering/')
from __init__ import project_path
from areas.table_info.dh_dw_table_info import dh_dw
from utils_offline.a00_imports import dfu, log20 as log, DayStr, read_api, shuttle, bip3_save_df2, bip3, bip2

from datetime import timedelta
from datetime import datetime

import numpy as np
import pandas as pd


class GoodsData:
    """
    商品数据类

    使用示例:
    data = GoodsData(data_label, dt)
    sel_goods_ls = data.goods_consume_data()
    df_stock_sell = data.get_his_true_value()

    """

    def __init__(self, data_label, dt, duration):
        """
        初始化函数

        参数:
        - data_label (str): 数据标签，可以是"normal"或"all"
        - dt (str): 日期，格式为"YYYY-MM-DD"
        - duraition (int): 时间跨度
        """
        self.data_label = data_label
        self.dt = dt
        self.duration = duration

    def goods_consume_data(self):
        """
        获取商品名称数据

        返回:
        - sel_goods_ls (list): 商品ID列表
        """
        df_goods_info = dh_dw.dim_stock.goods_info()
        sql_material_pool = """
        SELECT 
        goods_id,long_period_cal_type
        FROM  
        lucky_cooperation.t_material_pool
        WHERE del=0
        """
        df_material_pool = shuttle.query_dataq(sql_material_pool)
        goods_long_purchase = df_material_pool['goods_id'].drop_duplicates().tolist()
        # not_in_goods = ['冷冻调制血橙', '柚子复合果汁饮料浓浆',
        #                 '16oz 冰杯拱盖', '淡奶油', '绿茶调味茶固体饮料',
        #                 '12oz SOE冰杯', '纯牛奶', '单支吸管粗（PLA）',
        #                 '抹茶拿铁（固体饮料）', 'D直饮杯盖']
        not_in_goods = ['冷冻调制血橙', '柚子复合果汁饮料浓浆']

        dtw_goods = [49, 52, 80, 408, 411, 414, 415, 456,
                     756, 4173, 4488, 6064, 7282, 20610, 20716, 20721, 20725,
                     20818, 25316, 25456, 25869]
        # 非配方包材
        PACKAGE_MATERIALS = [72, 260, 290, 442, 450, 19343, 22837, 25274, 869]
        sel_goods_ls = df_goods_info[
            df_goods_info['goods_id'].isin(dtw_goods)
        ]['goods_id'].unique().tolist()

        return sel_goods_ls

    def get_his_true_value(self, sel_goods_ls):
        """
        获取历史真实值数据

        返回:
        - df_stock_sell (DataFrame): 包含历史真实值的数据框
        """
        # 过去120天的仓库货物消耗数据

        day_minus120 = datetime.strptime(self.dt, "%Y-%m-%d") - pd.DateOffset(days=self.duration)
        day_minus120 = day_minus120.strftime("%Y-%m-%d")
        # 数据依赖
        if self.data_label == 'normal':
            df_stock_sell = read_api.read_dt_folder(
                bip3("model/basic_predict_promote_online", "stock_wh_goods_type_flg_theory_sale_cnt"),
                day_minus120, self.dt)
            df_stock_sell = df_stock_sell[
                (df_stock_sell['type_flg'] == 'norm') & (df_stock_sell['shop_type_flg'] == 'norm_shop')]

            df_stock_sell.drop(['type_flg', 'shop_type_flg'], axis=1, inplace=True)

        if self.data_label == 'all':
            df_stock_sell = read_api.read_dt_folder(
                bip3("model/basic_predict_promote_online", "stock_wh_goods_theory_sale_cnt"),
                day_minus120, self.dt)

        if self.data_label == 'new_shop_normal':
            df_stock_sell = read_api.read_dt_folder(
                bip3("model/basic_predict_promote_online", "stock_wh_goods_type_flg_theory_sale_cnt"),
                day_minus120, self.dt)

            df_stock_sell = df_stock_sell[
                (df_stock_sell['type_flg'] == 'norm') & (df_stock_sell['shop_type_flg'] == 'new_shop')]
            df_stock_sell.drop(['type_flg', 'shop_type_flg'], axis=1, inplace=True)

        # 取绝对值
        df_stock_sell["wh_dept_id"] = df_stock_sell["wh_dept_id"].astype(int)
        df_stock_sell["goods_id"] = df_stock_sell["goods_id"].astype(int)
        df_stock_sell["theory_sale_cnt"] = np.abs(df_stock_sell["theory_sale_cnt"])
        df_stock_sell = df_stock_sell[df_stock_sell['goods_id'].isin(sel_goods_ls)]
        df_stock_sell["dt"] = pd.to_datetime(df_stock_sell["dt"])
        df_stock_sell.rename(columns={"theory_sale_cnt": "y", "dt": "ds"}, inplace=True)

        return df_stock_sell

