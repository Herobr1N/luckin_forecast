# encoding: utf-8
# @created: 2023/11/9
# @author: jieqin.lin
# @file: projects/basic_predict_promote/b_0_7_pred_add.py


"""
# 常规品 =  常规店常规品 + 新店常规品

生成数据
bip3("model/basic_predict_promote_online", "purchase_wh_goods_pred_all")
bip3("model/basic_predict_promote_online", "purchase_wh_goods_pred_normal")


依赖数据

bip3("model/basic_predict_promote_online", "mix_boost_1_wh_goods_pred_normal_group_b")
bip3("model/basic_predict_promote_online", "mix_boost_1_wh_goods_pred_new_shop_normal_group_b")
bip3("model/basic_predict_promote_online", "mix_boost_1_wh_goods_pred_all_group_b")

"""
from __init__ import project_path
import pandas as pd
from utils_offline.a00_imports import DayStr, read_api, argv_date, bip3, bip3_save_df2
from utils_offline.a31_pd_dataframe_util import dfu
from projects.basic_predict_promote.b_0_0_utils_models import ModelManager, SelectGoodsList

f"Import from {project_path}"


def main_purchase_wh_goods_pred_add(pred_calculation_day=None):
    """
    # 常规品 =  常规店常规品 + 新店常规品
    """

    pred_calc_day = DayStr.get_dt_str(pred_calculation_day)

    model_label = 'mix_boost_1'
    selector = SelectGoodsList()
    sel_all_goods_ls = selector.get_group_all_goods_ls()
    data_label_ls = ['normal', 'new_shop_normal']

    # --------------
    # 获取
    df = pd.DataFrame()
    for data_label in data_label_ls:
        for goods_label in sel_all_goods_ls.keys():
            manager = ModelManager(model_label=model_label, data_label=data_label, goods_label=goods_label)
            # 获取预测
            ld1 = manager.get_pred(dt=pred_calc_day)

            df = pd.concat([df, ld1])

    # --------------
    # sum 常规店常规品 + 新店常规品
    df_normal = df.groupby(['predict_dt', 'wh_dept_id', 'goods_id', 'model'])['predict_demand'].sum().reset_index()

    # --------------
    # 获取
    df_all = pd.DataFrame()
    data_label = 'all'
    for goods_label in sel_all_goods_ls.keys():
        manager = ModelManager(model_label=model_label, data_label=data_label, goods_label=goods_label)
        # 获取预测
        ld_all = manager.get_pred(dt=pred_calc_day)

        df_all = pd.concat([df_all, ld_all])

    # 去重复
    df_all = df_all.groupby(['predict_dt', 'wh_dept_id', 'goods_id', 'model'])['predict_demand'].max().reset_index()

    # --------------
    # save
    save_pred_df(df_normal, 'normal', pred_calc_day)

    save_pred_df(df_all, 'all', pred_calc_day)


def save_pred_df(df, data_label, dt):
    # save
    df.rename(columns={'predict_demand': 'demand'}, inplace=True)
    df['demand'] = df['demand'].round(1)
    df['dt'] = dt
    cols_int = ['wh_dept_id', 'goods_id', 'demand']
    df = dfu.df_col_to_numeric(df, cols_int)
    cols_output = ['predict_dt', 'wh_dept_id', 'goods_id', 'demand', 'dt']
    df_output = df[cols_output].sort_values('predict_dt').reset_index(drop=True)

    bip3_save_df2(df_output,
                  table_folder_name=f'purchase_wh_goods_pred_{data_label}',
                  bip_folder='model/basic_predict_promote_online',
                  output_name=f'purchase_wh_goods_pred_{data_label}',
                  folder_dt=dt)


if __name__ == '__main__':
    argv_date(main_purchase_wh_goods_pred_add)
