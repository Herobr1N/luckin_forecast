# encoding: utf-8
# @created: 2023/11/8
# @author: jieqin.lin
# @file: projects/basic_predict_promote/b_0_8_process_wh_goods_price.py


"""
计算仓库货物的成本金额
生成数据
bip3("model/basic_predict_promote_online", "wh_goods_price")     成本金额

依赖数据
dw_dws.dws_stock_warehouse_stock_adjust_d_inc_summary

"""

from __init__ import project_path
from utils_offline.a00_imports import DayStr, shuttle, bip3_save_df2
from utils_offline.a00_imports import argv_date

f"Import from {project_path}"


def get_wh_goods_price(pred_calculation_day=None):
    pred_calc_day = DayStr.get_dt_str(pred_calculation_day)
    pred_minus1_day = DayStr.n_day_delta(pred_calc_day, n=-1)
    sql_price = f"""-- 仓，货物，钱
    SELECT
        wh_dept_id
        , goods_id
        , SUM(end_wh_stock_money) / SUM(end_wh_stock_cnt) AS unit_price
    FROM dw_dws.dws_stock_warehouse_stock_adjust_d_inc_summary
    WHERE dt >= DATE_SUB('{pred_minus1_day}', 180)
      AND spec_status = 1
      AND end_wh_stock_cnt > 0
    GROUP BY wh_dept_id, goods_id
        """
    ld_price = shuttle.query_dataq(sql_price)

    ld_price['dt'] = pred_minus1_day
    cols_output = ['dt', 'wh_dept_id', 'goods_id', 'unit_price']
    df_price = ld_price[cols_output]
    bip3_save_df2(df_price,
                  table_folder_name=f'wh_goods_price',
                  bip_folder='model/basic_predict_promote_online',
                  output_name=f'wh_goods_price',
                  folder_dt=pred_minus1_day)


if __name__ == '__main__':
    argv_date(get_wh_goods_price)
