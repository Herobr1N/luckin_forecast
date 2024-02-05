import unittest
from utils_offline.a00_imports import dfu, log, DayStr, argv_date, c_path, read_api, c_path_save_df, shuttle, bip2, bip3,dft
import requests
import json
import pandas as pd
import numpy as np
import time



class MyTestCase(unittest.TestCase):
    def test_something(self):
        data = {
            "material_type": 0,
            "goods_id": 28085,
            "scene": 1,
            "stock_up_version": 7,
            "consumer_version": [
                11
            ],
            "strategy_version_id": 3,
            "sim_version": 1,
            "commodity_list": [
                4166
            ],
            "supplier_stocks": 10000.0,
            "goods_warranty_days": 28,
            "batch_produce_date": "2023-09-30",
            "ro_after_launch": 20.0,
            "callback": "https://echarts5060.lkcoffee.com/callback"
        }

        # 回调信号置零
        df_status = pd.DataFrame()
        read_api.df_to_hdfs(df_status, '/projects/luckyml/dinghuo/temp/kai.feng/df_status/temp/', format='parquet')

        response = requests.post('https://echarts5060.lkcoffee.com/new_material_simulation', data=json.dumps(data))
        # 等待回调成功
        for _ in range(60):
            time.sleep(5)  # 每秒检查一次
            df_status = read_api.read_one_file("/projects/luckyml/dinghuo/temp/kai.feng/df_status/temp/")
            if not df_status.empty:
                print(response.status_code)
                print(response.text)
                pred_calculation_day = None
                pred_calc_day = DayStr.get_dt_str(pred_calculation_day)


                # 单元测试1
                log.debug('单元测试1  新品全国有货率')
                df_actual = read_api.read_one_file(
                    f"/user/hive/warehouse/dw_ads_scm_alg.db/alg_control_tower_newp_nation_goods_avl_result"
                    f"/dt={pred_calc_day}/alg_control_tower_newp_nation_goods_avl_result.parquet")

                df_actual = df_actual.query(
                    "goods_id == 28085 and consumer_version == 11 and predict_dt <= '2023-10-05'")

                cols = ['predict_dt', 'goods_id', 'shop_avl_rate', 'wh_avl_days', 'shop_avl_days', 'wh_consume_amount']
                df_actual = df_actual[cols]

                df_expect = pd.DataFrame({'predict_dt': ['2023-09-28','2023-09-29','2023-09-30','2023-10-01','2023-10-02',
                                                        '2023-10-03','2023-10-04','2023-10-05'],
                                         'goods_id': [28085, 28085, 28085, 28085, 28085, 28085, 28085, 28085],
                                        'shop_avl_rate': [0.0,0.0,0.0,0.0, 0.0, 0.12125262554897842,0.20861982223712894, 0.29007751937984494],
                         'wh_avl_days': [77, 76, 75, 74, 73, 72, 71, 70],
                         'shop_avl_days': [0, 0, 0, 0, 0, 0, 0, 0],
                         'wh_consume_amount': [0.0,0.0,7608000.0, 7284000.0,7428000.0,9216000.0,10020000.0,20268000.0]})


                dft.check_df_equal(df_expect, df_actual)

                # 单元测试2
                log.debug('单元测试2 新品全国模拟结果')
                df_actual_new_simulation = read_api.read_one_file(
                    f"/user/hive/warehouse/dw_ads_scm_alg.db/alg_control_tower_newp_nation_goods_stock_simulation_result"
                    f"/dt={pred_calc_day}/alg_control_tower_newp_nation_goods_stock_simulation_result.parquet")

                cols_input = ['goods_id', 'material_type', 'stock_up_version', 'consumer_version', 'strategy_version_id',
                              'sim_version','first_delivery_quantity', 'first_delivery_days', 'goods_stock_up_quantity',
                              'goods_stock_up_days','raw_stock_quantity', 'raw_stock_days', 'total_days', 'w1', 'w2',
                              'w3', 'w4', 'first_deadline','goods_deadline', 'raw_deadline', 'plan_sale_duration',
                              'over_plan_quantity', 'is_expire_risk','estimated_expired_quantity']

                df_actual_new_simulation_expect = pd.DataFrame([( 28085, 0, 7, 11, 3, 1, 1.762596e+09, 85, 4.30044e+08,
                   72, 3.580668e+09, 664, 821, 39.97, 40.19, 40.41, 39.4 , '2023-12-22', '2024-03-03', '2025-12-27',
                    67, 4.052736e+09, 1, 2.14068e+08), (28085, 0, 7,  7, 3, 1, 1.762596e+09, 19, 4.30044e+08,  8, 3.580668e+09,
                 50,  77, 58.98, 59.3 , 59.63, 58.14, '2023-10-17', '2023-10-25', '2023-12-14',
                 67, 3.265932e+09, 1, 2.14068e+08)],columns=cols_input)

                dft.check_df_equal(df_actual_new_simulation, df_actual_new_simulation_expect)




                break
        else:  # 如果60秒后回调还没有成功，抛出异常
            raise AssertionError('Callback did not succeed within 300 seconds')

    def test_sub_new(self):
        data2 = {
            "material_type": 0,
            "goods_id": 28257,
            "scene": 2,
            "stock_up_version": 7,
            "consumer_version": [11],
            "strategy_version_id": 4,
            "sim_version": 1,
            "commodity_list": [4129],
            "callback": "https://echarts5060.lkcoffee.com/callback"
        }

        # 回调信号置零
        df_status = pd.DataFrame()
        read_api.df_to_hdfs(df_status, '/projects/luckyml/dinghuo/temp/kai.feng/df_status/temp/', format='parquet')

        response = requests.post('https://echarts5060.lkcoffee.com/subnew_material_simulation', data=json.dumps(data2))
        # 等待回调成功
        for _ in range(60):
            time.sleep(5)  # 每秒检查一次
            df_status = read_api.read_one_file("/projects/luckyml/dinghuo/temp/kai.feng/df_status/temp/")
            if not df_status.empty:
                print(response.status_code)
                print(response.text)
                pred_calculation_day = None
                pred_calc_day = DayStr.get_dt_str(pred_calculation_day)

                # 单元测试1
                log.debug('单元测试1  次新品全国有货率')
                df_actual = read_api.read_one_file(
                    f"/user/hive/warehouse/dw_ads_scm_alg.db/alg_control_tower_newp_nation_goods_avl_result"
                    f"/dt={pred_calc_day}/alg_control_tower_newp_nation_goods_avl_result.parquet")

                df_actual = df_actual.query(
                    "goods_id == 28257 and consumer_version == 11 and predict_dt <= '2023-09-15'")

                cols = ['predict_dt', 'goods_id', 'shop_avl_rate', 'wh_avl_days', 'shop_avl_days', 'wh_consume_amount']
                df_actual = df_actual[cols]

                df_expect = pd.DataFrame(
                    [('2023-09-11', 28257, 0.97712153, 11, 15, 428100.),
                     ('2023-09-12', 28257, 0.98004219, 10, 16, 440700.),
                     ('2023-09-13', 28257, 0.98580237, 9, 15, 449100.),
                     ('2023-09-14', 28257, 0.98572124, 8, 15, 495900.),
                     ('2023-09-15', 28257, 0.98555898, 7, 14, 597300.)]
                , columns=cols)

                dft.check_df_equal(df_expect, df_actual)

                # 单元测试2
                log.debug('单元测试2 次新品全国模拟结果')
                df_actual_subnew_simulation = read_api.read_one_file(
                    f"/user/hive/warehouse/dw_ads_scm_alg.db/alg_control_tower_subnewp_nation_goods_stock_simulation_result"
                    f"/dt={pred_calc_day}/alg_control_tower_subnewp_nation_goods_stock_simulation_result.parquet")



                cols_input = ['goods_id', 'material_type', 'stock_up_version', 'consumer_version', 'strategy_version_id', 'sim_version',
                           'stock_quantity', 'stock_days', 'cg_trs_quantity', 'cg_trs_days', 'po_quantity', 'po_days',
                           'replenish_quantity', 'replenish_days', 'total_days', 'w1', 'w2', 'w3', 'w4', 'stock_deadline',
                           'cg_trs_deadline', 'po_deadline', 'replenish_deadline']

                df_actual_subnew_simulation_expect = pd.DataFrame([( 28257, 0, 7, 11, 4, 1, 10869600., 11, 0., 0, 455998., 1, 144000., 0, 12, 18.32, 18.65, 18.98, 19.3 , '2023-09-22', '2023-09-22', '2023-09-23', '2023-09-23'),
                            (28257, 0, 7,  7, 4, 1, 10869600., 19, 0., 0, 455998., 1, 144000., 0, 20,  9.13,  9.43, 14.99, 24.05, '2023-09-30', '2023-09-30', '2023-10-01', '2023-10-01')], columns=cols_input)

                dft.check_df_equal(df_actual_subnew_simulation, df_actual_subnew_simulation_expect)

                break
        else:  # 如果60秒后回调还没有成功，抛出异常
            raise AssertionError('Callback did not succeed within 300 seconds')


if __name__ == '__main__':
    unittest.main()
