# encoding: utf-8
# @author: jieqin.lin
# @created: 2023-01-03
# @file: projects/purchase/alg_hive01_new_purchase_shop_com_pred_future_x.py

"""
新品首采未来N天门店商品杯量预测表落库

上游 ~ 9：30am 完成

建表语句
DROP TABLE IF EXISTS  `dw_ads_scm_alg.new_purchase_shop_com_pred_future_x`;
CREATE TABLE `dw_ads_scm_alg.new_purchase_shop_com_pred_future_x`(
`date` string COMMENT '日期',
`plan_id` bigint COMMENT '商品计划ID',
`one_category_id` bigint COMMENT '商品大类ID',
`two_category_id` bigint COMMENT '商品中类ID',
`commodity_id` bigint COMMENT '商品ID',
`dept_id` bigint COMMENT '门店ID',
`city_id` bigint COMMENT '城市ID',
`avg_commodity` double COMMENT '门店商品中类日均单品杯量预估',
`pred_0` double COMMENT '全国门店商品中类日均单品杯量预估的均值',
`product_pred` double COMMENT '产品杯量预估',
`pred_ratio` double COMMENT '预测调整系数',
`pred` double COMMENT '门店新品杯量预估',
`is_sell_pred` bigint COMMENT '预测门店商品是否有售卖:1有售卖 0没有售卖',
`open` bigint COMMENT '门店是否营业:1营业 0不营业',
`adj_pred` double COMMENT '衰减后门店新品杯量预估')
COMMENT '新品首采未来N天门店商品杯量预测表'
PARTITIONED BY (
`dt` string)
STORED AS Parquet

"""

from utils.c52_hdfs_to_hive import mlclient_pandas_to_hive
from utils_offline.a00_imports import log, DayStr, argv_date, bip2, read_api
from utils.a91_wx_work_send import qiyeweixin_bot


# 新品首采未来N天门店商品杯量预测表落库，落库HIVE
def alg_hive__new_purchase_shop_com_pred_future_x(pred_calculation_day=None):
    """
    测试运行
    -------
    from xa_dh.projects.metric.alg_hive__new_purchase_shop_com_pred_future_x import *
    alg_hive__new_purchase_shop_com_pred_future_x("2022-11-23")
    """

    pred_calc_day = DayStr.get_dt_str(pred_calculation_day)
    pred_minus1_day = DayStr.n_day_delta(pred_calc_day, n=-1)

    log.h2(f"新品首采未来N天门店商品杯量预测表，落库HIVE data_dt={pred_calc_day} run at {DayStr.now2()}")

    ld_pred = read_api.read_one_folder(
        bip2("model/new_purchase", "new_purchase_shop_com_launch_plan_date_future_x_adj_2", pred_calc_day))
    # # 历史推单
    # df_his = read_api.read_dt_folder(
    #     bip2("model/new_purchase_metric", "new_purchase_com_pur_metric_po_push_history_2"), pred_minus1_day)
    # plan_id_list = df_his['plan_id'].drop_duplicates().tolist()

    if len(ld_pred) == 0:
        log.h2(f"新品首采预测使用降级方案")
        run_message = '新品首采预测使用降级方案\n'
        ld_pred = read_api.read_one_folder(
            bip2("model/new_purchase", "new_purchase_shop_com_launch_plan_date_future_x_adj_2", pred_minus1_day))
        qiyeweixin_bot(run_message, send_to=3, mention_number_list=["18664985916"])

    ld_pred.rename(columns={'dt': 'date'}, inplace=True)
    df_output = ld_pred.query("week_label !='W0'")[['date', 'plan_id', 'one_category_id', 'two_category_id',
                                                    'commodity_id', 'dept_id', 'city_id',
                                                    'avg_commodity', 'pred_0', 'product_pred',
                                                    'pred_ratio', 'pred', 'is_sell_pred', 'open', 'adj_pred']].copy()
    for cc in ['plan_id', 'one_category_id', 'two_category_id', 'commodity_id', 'dept_id', 'city_id', 'open']:
        df_output[cc] = df_output[cc].astype("int64")

    log.debug(f"数据量{len(df_output)}")
    # 排除历史推单
    # mlclient_pandas_to_hive(df_output.query(f"plan_id not in {plan_id_list}"),
    #                         "new_purchase_shop_com_pred_future_x", dt=pred_calc_day, dt_partition=True)
    mlclient_pandas_to_hive(df_output,
                            "new_purchase_shop_com_pred_future_x", dt=pred_calc_day, dt_partition=True)


if __name__ == '__main__':
    argv_date(alg_hive__new_purchase_shop_com_pred_future_x)