# -*- utf-8 -*-
# @Time: 2023/5/29 09:50
# @Author: kai.feng
# @File: projects/purchase/alg_v0_tower_simulation_hive_to_hdfs.py



"""
将控制塔V0版本 仓库模拟结果存hdfs，提升后续读写效率
"""
from utils_offline.a00_imports import log, DayStr, argv_date, bip2, read_api,shuttle,c_path,c_path_save_df
from utils.dtype_monitor_transform import convert_df_columns

def tower_simulation_v0_hive_to_hdfs(pred_calculation_day=None):
    pred_calc_day = DayStr.get_dt_str(pred_calculation_day)
    pred_minus1_day = DayStr.n_day_delta(pred_calc_day, n=-1)

    sql_simulation = f"""
            select * 
            from dw_lucky_dataplatform.dm_control_tower_wh_goods_stock_simulation_result_v2 
            where dt = '{pred_minus1_day}'
            """

    wh_stock_simulation = shuttle.query_dataq(sql_simulation)


    wh_stock_simulation = wh_stock_simulation.fillna({"use_day":0})

    cols = ['wh_dept_id','goods_id','use_day']
    for i in cols:
        wh_stock_simulation[i] = wh_stock_simulation[i].astype('int')

    wh_stock_simulation = wh_stock_simulation[['predict_dt', 'wh_dept_id', 'goods_id', 'begin_stock_amount',
                            'cg_zt_amount', 'end_stock_amount', 'alloc_zt_amount', 'loss_amount','use_day', 'com_demand', 'dt']]

    c_path_save_df(wh_stock_simulation,c_path.control_tower.wh_goods_stock_stimulation_v0, pred_minus1_day)


def tower_simulation_v0_without_trs_hive_to_hdfs(pred_calculation_day=None):
    log.debug('监控预警v0不含模拟cc 调拨')
    pred_calc_day = DayStr.get_dt_str(pred_calculation_day)
    pred_minus1_day = DayStr.n_day_delta(pred_calc_day, n=-1)
    sql_simulation = f"""
            select * 
            from dw_lucky_dataplatform.dm_control_tower_wh_goods_stock_simulation_without_cg_trs 
            where dt = '{pred_minus1_day}'
            """

    wh_stock_simulation = shuttle.query_dataq(sql_simulation)


    wh_stock_simulation = wh_stock_simulation.fillna({"use_day":0})

    cols = ['wh_dept_id','goods_id','use_day']
    for i in cols:
        wh_stock_simulation[i] = wh_stock_simulation[i].astype('int')

    wh_stock_simulation = wh_stock_simulation[['predict_dt', 'wh_dept_id', 'goods_id', 'begin_stock_amount',
                            'cg_zt_amount', 'end_stock_amount', 'alloc_zt_amount', 'loss_amount', 'use_day', 'com_demand', 'dt']]


    c_path_save_df(wh_stock_simulation,c_path.control_tower.wh_stimulation_no_trs_v0, pred_minus1_day)



def get_wh_sim_v0(pred_calculation_day=None):
    pred_calc_day = DayStr.get_dt_str(pred_calculation_day)
    pred_minus1_day = DayStr.n_day_delta(pred_calc_day, n=-1)
    sql_simulation = f"""
    select * 
    from dw_lucky_dataplatform.dm_control_tower_wh_goods_stock_simulation_result_v2 
    where dt = '{pred_minus1_day}'
    """
    wh_stock_simulation = shuttle.query_dataq(sql_simulation).pipe(convert_df_columns)
    c_path_save_df(wh_stock_simulation, c_path.control_tower.wh_sim_v0, pred_calc_day)

def etl_get_simulation_v0_shop_dim_query(pred_calculation_day=None):
    log.title("offline_v0_dm控制塔门店底表")

    pred_calc_day = DayStr.get_dt_str(pred_calculation_day)
    pred_minus1_day = DayStr.n_day_delta(pred_calc_day, n=-1)
    query = f'''SELECT DISTINCT
                    wh_dept_id
                  , wh_name
                  , stock_cell_id
                  , shop_dept_id
                  , city_id
                  , city_name
                  , company_no
                  , company_name
                  , coop_pattern_code
                  , coop_pattern_name
                  , shop_name
                  , shop_type
                  , shop_type_name
                FROM dw_lucky_dataplatform.dm_control_tower_shop_goods_stock_simulation_result_v2
                WHERE dt = '{pred_minus1_day}'
    '''

    cols = ['wh_dept_id', 'city_id', 'shop_dept_id']

    shop_stock_simulation_p1 = shuttle.query_dataq(query)
    for i in cols:
        shop_stock_simulation_p1[i] = shop_stock_simulation_p1[i].astype('int')

    c_path_save_df(shop_stock_simulation_p1, c_path.control_tower.simulation_v0_shop_dim, pred_calc_day)


# simulation_v0_dt_dim
def etl_get_simulation_v0_dt_dim_query(pred_calculation_day=None):
    log.title("offline_v0_dm控制塔门店底表")

    pred_calc_day = DayStr.get_dt_str(pred_calculation_day)
    pred_minus1_day = DayStr.n_day_delta(pred_calc_day, n=-1)
    query = f'''SELECT distinct
                    shop_dept_id 
                  , predict_dt 
                  , week
                  , is_recive_date
                  , next_rec_date
                  , order_date
                  , dly_cnt
                  , spent_days
                  , bp
                from dw_lucky_dataplatform.dm_control_tower_shop_goods_stock_simulation_result_v2
                WHERE dt = '{pred_minus1_day}'
    '''

    cols = ['shop_dept_id']

    shop_stock_simulation_p2 = shuttle.query_dataq(query)
    for i in cols:
        shop_stock_simulation_p2[i] = shop_stock_simulation_p2[i].astype('int')

    c_path_save_df(shop_stock_simulation_p2, c_path.control_tower.simulation_v0_dt_dim, pred_calc_day)
