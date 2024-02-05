# encoding: utf-8 
""" 
@Project:workspace 
@Created: 2023/9/19 
@Author: cuiyuhan 
"""
import numpy as np
import pandas as pd

from utils.dtype_monitor_transform import *
from utils_offline.a00_imports import dfu, log, DayStr, argv_date, c_path, read_api, c_path_save_df, bip2, bip3, bip1
from datetime import datetime, timedelta
import logging
from datetime import timedelta

logger = logging.getLogger("alg_dh")


def check_input_quality(col_name=None, cond=[], error=True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 调用原函数，获取返回值
            df = func(*args, **kwargs)

            # 检查 DataFrame 是否为空

            if df.empty:
                raise ValueError(f"Result of {func.__name__} is empty.")

            # 检查特定列是否含不符合规范的条目
            if col_name is not None:
                if callable(cond):
                    mask = df[col_name].apply(cond)

                else:
                    mask = df[col_name].isin(cond)

                num_invalid = mask.sum()
                if isinstance(col_name, list):
                    cond_pop = any(num_invalid > 0)
                else:
                    cond_pop = num_invalid > 0

                if cond_pop:
                    if error:
                        raise ValueError(
                            f"Result of {func.__name__}, {col_name} contains {num_invalid.sum()} invalid entries in total.")
                    else:
                        print(
                            f"Warning: Result of {func.__name__}, {col_name} contains {num_invalid.sum()} invalid entries in total.")
            else:
                pass
            # 返回 DataFrame
            return df

        return wrapper

    return decorator


class NewEtl:
    def __init__(self, pred_calculation_day):
        # 日期相关
        self.ld_dt_label = None
        self.df_wh_cup_goods = None
        self.pred_calc_day = DayStr.get_dt_str(pred_calculation_day)
        self.pred_minus1_day = DayStr.n_day_delta(self.pred_calc_day, n=-1)
        self.folder_minutes = DayStr.now4()

        self.version_pool = []
        self.material_type = None
        self.select_goods_id = None
        self.scene = None
        self.stock_up_version = None
        self.consumer_version = None
        self.strategy_version_id = None
        self.sim_version = None
        self.commodity_list = []

        self.df_com_launch_plan = None
        self.max_launch_date = None
        self.min_launch_date = None
        self.min_pur_end_date = None
        self.max_pur_end_date = None
        self.min_cs_start = None
        self.launch_diff = None
        self.new_type = None

        self.cold_start_dur = 11
        self.max_sim_length = 75
        # 预定义

        self.ld_receive_order_date = None
        self.valid_shop_ratio = None
        self.valid_plan_scope = None
        self.valid_formula = None

