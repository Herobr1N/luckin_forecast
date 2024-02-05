from utils.file_utils import save_file
import multiprocessing

import math
import pandas as pd
from meta import *
from datetime import datetime, timedelta
from pandas import DataFrame
import numpy as np
from config.config import logger
from scipy.stats import norm

spark = Meta('yanxin.lu')
today = datetime.today().date()
yesterday = today - timedelta(days=1)


def get_random(low=-2, high=3, size=1):
    return int(np.random.randint(low=low, high=high, size=size)[0])


def sim_store(df_transit, cur_dt, sim_cur_dt, vlt, rep, wh_dept_id, goods_id):
    # 模拟入库
    dt_receive_plan = pd.to_datetime(sim_cur_dt).date() + timedelta(days=vlt)
    dt_receive = dt_receive_plan + timedelta(days=get_random())
    df_transit = pd.concat([df_transit, pd.DataFrame.from_records([{'dt': str(dt_receive), 'wh_dept_id': wh_dept_id, 'goods_id': goods_id,
                                                                    'transit': rep, 'transit_mdq': rep, 'order_dt': cur_dt,
                                                                    'plan_order_dt': str(dt_receive_plan)}])])

    logger.info(f"++++ 仓库:【{wh_dept_id}】,货物:【{goods_id}】, 计划入仓数量:【{rep}】, 计划入仓时间:【{dt_receive}】, 下单时间:【{cur_dt}】")
    return df_transit


def std_model(cur_dt, wh_dept_id, goods_id, vlt, rep, df_transit):
    return sim_store(df_transit, cur_dt, cur_dt, vlt, rep, wh_dept_id, goods_id)


def rq_model(cur_dt: str, wh_dept_id: int, goods_id: int, vlt: int, df_pred: DataFrame, df_transit: DataFrame, df_wh_dly: DataFrame, sim_ss: float,
             sim_ro: float, mdq: float, bp: int, sim_begin: float):
    # 若入仓频率高于1天一次，则每天发货，发货数量保证尽量下批货入仓前不触发断仓预警
    bp_rq = 1
    logger.info(f"第一次入仓阶段, 仓库:{wh_dept_id}, 货物:{goods_id}")
    """
        进行一次周期型补货，判断第一个VLT是否需要补货
    """
    vlt_first = vlt
    # 计算目标库存
    dt_bp_vlt = pd.to_datetime(cur_dt).date() + timedelta(days=vlt_first + bp_rq - 1)
    dmd_bp_vlt = df_pred.query(f"(predict_dt>='{cur_dt}') and (predict_dt <= '{dt_bp_vlt}')")['demand'].sum()
    ti = dmd_bp_vlt + sim_ss + sim_ro
    # 计算补货量
    sim_transit = df_transit.query(f"(dt>='{cur_dt}') and (dt <= '{dt_bp_vlt}') and (wh_dept_id == {wh_dept_id}) and (goods_id == {goods_id})")[
        'transit_mdq'].sum()
    rep = max(np.ceil(ti - sim_begin - sim_transit), 0)
    # 凑N倍的MOQ
    rep_mdq = np.ceil(rep / mdq) * mdq

    logger.info(f"推算第一个补货点, 需求区间:【{cur_dt}, {dt_bp_vlt}】, 在途总量:【{sim_transit / mdq}】个Q, 总需求量：【{dmd_bp_vlt / mdq}】个Q, 总入仓量:【{rep_mdq / mdq}】个Q")
    if rep_mdq > 0:
        # 模拟入库
        df_transit = sim_store(df_transit, cur_dt, cur_dt, vlt, rep_mdq, wh_dept_id, goods_id)

    sim_cur_dt = cur_dt
    # 计算第一次入仓+1天的期初
    sim_end_dt = str(pd.to_datetime(sim_cur_dt).date() + timedelta(days=vlt_first + bp_rq - 1))
    while sim_cur_dt <= sim_end_dt:
        dly_out = df_wh_dly.query(f"dt == '{sim_cur_dt}'")['wh_dly_out'].values[0]
        sim_transit = df_transit.query(f"dt == '{sim_cur_dt}' and (wh_dept_id == {wh_dept_id}) and (goods_id == {goods_id})")['transit_mdq'].sum()
        sim_end = max(sim_begin - dly_out + sim_transit, 0)
        logger.debug(
            f"更新第一次补货期间库存：{sim_cur_dt}, 仓库:{wh_dept_id}, 货物:{goods_id}, 期初：{sim_begin}, 模拟入库：{sim_transit / mdq}个Q, 实际出库:{dly_out}，当日期末：{sim_end}")
        sim_begin = sim_end
        sim_cur_dt = str(pd.to_datetime(sim_cur_dt).date() + timedelta(days=1))

    """
        在剩余的BP-1天中，连续Review，判断是否需要入库
    """
    vlt = 1
    rest_days = bp - 1
    while rest_days > 0:
        # 计算目标库存
        dt_bp_vlt = pd.to_datetime(sim_cur_dt).date() + timedelta(days=vlt + bp_rq - 1)
        dmd_bp_vlt = df_pred.query(f"(predict_dt>='{sim_cur_dt}') and (predict_dt <= '{dt_bp_vlt}')")['demand'].sum()
        ti = dmd_bp_vlt + sim_ss + sim_ro
        # 计算补货量
        sim_transit = \
        df_transit.query(f"(dt>='{sim_cur_dt}') and (dt <= '{dt_bp_vlt}') and (wh_dept_id == {wh_dept_id}) and (goods_id == {goods_id})")[
            'transit_mdq'].sum()
        rep = max(np.ceil(ti - sim_begin - sim_transit), 0)
        # 凑N倍的MOQ
        rep_mdq = np.ceil(rep / mdq) * mdq
        logger.info(f"连续Review:{sim_cur_dt}, 覆盖区间:【{sim_cur_dt}, {dt_bp_vlt}】, 在途总量:【{sim_transit / mdq}】个Q, 总需求量:【{rep_mdq / mdq}】个Q")
        if rep_mdq > 0:
            # 模拟入库
            df_transit = sim_store(df_transit, cur_dt, sim_cur_dt, vlt, rep_mdq, wh_dept_id, goods_id)
            # logger.info(f"++++ 仓库:【{wh_dept_id}】,货物:【{goods_id}】, 计划入仓数量:【{rep_mdq/moq}】个Q, 计划入仓时间:【{dt_receive}】, 下单时间:【{cur_dt}】")

        # 当天在途
        sim_transit = df_transit.query(f"dt=='{sim_cur_dt}'")['transit_mdq'].sum()
        # 当日实际出库量
        dly_out = df_wh_dly.query(f"(dt == '{sim_cur_dt}')")
        # 超出模拟范围
        if len(dly_out) == 0:
            logger.info('### 超出模拟范围，BREAK ###')
            break
        dly_out = dly_out['wh_dly_out'].values[0]
        # 计算期末
        sim_end = max(sim_begin - dly_out + sim_transit, 0)
        logger.debug(f"更新库存：{sim_cur_dt}, 仓库:{wh_dept_id}, 货物:{goods_id}, 期初：{sim_begin}, 模拟入库：{sim_transit / mdq}, 实际出库:{dly_out}，当日期末：{sim_end}")
        sim_begin = sim_end
        sim_cur_dt = pd.to_datetime(sim_cur_dt).date() + timedelta(days=1)
        rest_days -= 1
    return df_transit


def reorder_model(policy, dt: str, df_pred: DataFrame, df_wh_dly: DataFrame, df_transit: DataFrame, df_ss_ro: DataFrame, sim_begin: float, vlt: int,
                  vlt_std, bp: int, z: int, ro: int, pur_ratio: int, mdq_pur: int, wh_dept_id: int, goods_id: int) -> DataFrame:
    # review point
    # 计算目标库存
    dt_vlt = str(pd.to_datetime(dt).date() + timedelta(days=vlt - 1))
    dt_bp = str(pd.to_datetime(dt_vlt).date() + timedelta(days=bp))
    dt_bp_vlt = str(pd.to_datetime(dt).date() + timedelta(days=vlt + bp - 1))

    # VLT期间预测
    dmd_vlt = df_pred.query(f"(predict_dt >= '{dt}') and (predict_dt <= '{dt_vlt}')")['demand'].sum()
    # BP期间预测
    dmd_bp = df_pred.query(f"(predict_dt > '{dt_vlt}') and (predict_dt <= '{dt_bp}')")['demand'].sum()
    # BP+VLT 总需求
    dmd_bp_vlt = dmd_vlt + dmd_bp
    # BP+VLT 均值
    dmd_avg_bp_vlt = dmd_bp_vlt / (vlt + bp)

    # 历史出库std
    dt_dmd_start = str(pd.to_datetime(dt).date() - timedelta(days=vlt + bp))
    dmd_std = df_wh_dly.query(f"(dt>='{dt_dmd_start}') and (dt < '{dt}')")['wh_dly_out'].std()
    # 预测安全库
    sim_ss = np.round(z * dmd_std * math.sqrt(vlt + bp) + z * dmd_avg_bp_vlt * vlt_std)
    sim_ro = ro * dmd_avg_bp_vlt

    # 实际出库
    dly_out_base = df_wh_dly.query(f"(dt>='{dt}') and (dt <= '{dt_bp_vlt}')")['wh_dly_out']
    # 实际出库日均
    dly_out_avg = dly_out_base.mean()
    # 实际出库STD
    dly_out_std = dly_out_base.std()
    # 实际安全库存
    ss = np.round(z * dly_out_std * math.sqrt(vlt + bp) + z * dly_out_avg * vlt_std)
    ro = ro * dly_out_avg
    df_ss_ro = pd.concat([df_ss_ro, pd.DataFrame.from_records([{'dt': dt, 'wh_dept_id': wh_dept_id, 'goods_id': goods_id, 'sim_ss': sim_ss,
                                                                'sim_ro': sim_ro, 'sim_dmd_avg': dmd_avg_bp_vlt, 'ss': ss, 'ro': ro,
                                                                'dly_avg': dly_out_avg if dly_out_avg > 0 else np.nan}])])

    # VLT期间在途
    sim_transit_vlt = df_transit.query(f"(dt >= '{dt}') and (dt <= '{dt_vlt}') and (wh_dept_id == {wh_dept_id}) and (goods_id == {goods_id})")[
        'transit_mdq'].sum()
    # BP期间在途
    sim_transit_bp = df_transit.query(f"(dt > '{dt_vlt}') and (dt <= '{dt_bp}') and (wh_dept_id == {wh_dept_id}) and (goods_id == {goods_id})")[
        'transit_mdq'].sum()
    sim_transit = sim_transit_vlt + sim_transit_bp

    rop = dmd_vlt + sim_ss + sim_ro - sim_transit_vlt

    if rop > sim_begin:
        ti = dmd_bp_vlt + sim_ss + sim_ro
        # 原始采购量
        rep = max(np.ceil(ti - sim_begin - sim_transit), 0)
        # 采购单位
        rep_pur = np.ceil(rep / pur_ratio) * pur_ratio
        logger.info(
            f"订货日：【{dt}】, VLT: 【[{dt}, {dt_vlt}]】, BP区间:【({dt_vlt}, {dt_bp}]】, Rop: {rop}, TI: {ti}, Beg:{sim_begin}, 在途总量:【{sim_transit}】, 需采购量:【{rep}】,")
        if rep_pur > 0:
            # 模拟入仓
            if goods_id in (52, 354):
                logger.info(f"多频次入仓模型")
                df_transit = rq_model(dt, wh_dept_id, goods_id, vlt, df_pred, df_transit, df_wh_dly, sim_ss, sim_ro, mdq_pur * pur_ratio, bp,
                                      sim_begin)
            else:
                df_transit = std_model(dt, wh_dept_id, goods_id, vlt, rep_pur, df_transit)

    return df_transit, df_ss_ro, dmd_avg_bp_vlt, sim_transit


def inv_sim(dt, df_config, beg_cnt, df_pred, df_wh_dly, df_transit, df_ss_ro, df_res, dict_beg_inv, key_beg_inv, review_model=reorder_model,
            policy=std_model) -> DataFrame:
    wh_dept_id, wh_name, goods_id, bp, ro, sl, vlt, vlt_std, pur_ratio, mdq_pur = \
    df_config[['wh_dept_id', 'wh_name', 'goods_id', 'bp', 'ro', 'sl', 'vlt', 'vlt_std', 'pur_ratio', 'mdq_pur']].values[0]

    end_cnt, dly_out, transit = df_wh_dly.query(f"dt == '{dt}'")[['end_avl_inv', 'wh_dly_out', 'wh_pur_cnt']].values[0]
    pred_demand = df_pred['demand'].values[0]
    """ 补货模型 """
    df_transit, df_ss_ro, pred_bp_vlt_dmd_avg, trs_bp_vlt = review_model(policy, dt, df_pred, df_wh_dly, df_transit, df_ss_ro, beg_cnt, vlt, vlt_std,
                                                                         bp, norm.ppf(sl), ro, pur_ratio, mdq_pur, wh_dept_id, goods_id)

    transit_base = df_transit.query(f"(dt == '{dt}') and (wh_dept_id == {wh_dept_id}) and (goods_id == {goods_id})")
    if len(transit_base) == 0:
        sim_transit = 0
        order_dt = None
        plan_order_dt = None
    else:
        sim_transit = transit_base['transit_mdq'].sum()
        order_dt = transit_base['order_dt'].values[0]
        plan_order_dt = transit_base['plan_order_dt'].values[0]

    sim_end = max(beg_cnt - dly_out + sim_transit, 0)
    dict_beg_inv[key_beg_inv] = sim_end
    df_res.loc[len(df_res)] = [dt, wh_dept_id, wh_name, goods_id, beg_cnt, sim_end, end_cnt, sim_transit, transit, pred_demand, pred_bp_vlt_dmd_avg,
                               trs_bp_vlt, dly_out, order_dt, plan_order_dt]
    return df_res, df_transit, df_ss_ro


def cal_union_mdq(df: DataFrame, union_col: str, union_col_unit: str) -> DataFrame:
    """
    凑MQ逻辑
    同仓同供应商同MDQ，凑MDQ
    目标：各货物可用天数尽量保持一致
    :param df: 需凑记录
    :param union_col: 总采购量对应列
    :param union_col_unit: 当前总采购量对应列 的单位列
    :return:
    """
    print(union_col, union_col_unit)
    spark.save_hdfs(df)
    res_mdq_list = []
    for (supplier_code, wh_dept_id, union_mdq, moq_type), group in df.groupby(['supplier_code', 'wh_dept_id', 'union_mdq', 'moq_type']):
        logger.info(f"------- MDQ: {supplier_code}")
        # 单位采购单位可用天数
        group['day_per_pur'] = group['pur_ratio'] / group['pred_bp_vlt_dmd_avg']
        # 计算差异数量
        union_mdq, union_transit, unit = group.head(1)[['union_mdq', union_col, union_col_unit]].values[0]
        diff = union_mdq - union_transit
        # diff = 20
        # 根据可用天数，每次补可用天数最低的货物
        while diff > 0:
            group = group.sort_values(by='avl_days').reset_index(drop=True)
            group.loc[0, 'avl_days'] = group.loc[0, 'day_per_pur'] + group.loc[0, 'avl_days']
            group.loc[0, 'transit_mdq'] = group.loc[0, 'transit_mdq'] + group.loc[0, 'pur_ratio']
            diff -= unit
        group[['dt', 'order_dt', 'plan_order_dt']] = group[['dt', 'order_dt', 'plan_order_dt']].ffill()
        res_mdq_list.append(group)

    res_mdq = pd.concat(res_mdq_list)
    return res_mdq


def mdq_process(res_day, df_cg_config, df_transit):
    df_base = res_day[['wh_dept_id', 'goods_id', 'sim_beg', 'trs_bp_vlt', 'pred_bp_vlt_dmd_avg']] \
        .merge(df_cg_config, on=['wh_dept_id', 'goods_id'], how='inner') \
        .merge(df_transit, on=['wh_dept_id', 'goods_id'], how='left') \
        .reset_index()
    df_base.fillna({'transit': 0, 'transit_mdq': 0}, inplace=True)

    # 采购量-采购单位
    df_base['transit_pur'] = df_base['transit'] / df_base['pur_ratio']

    # 处理返回列
    res_columns = df_transit.columns

    """
    非联合凑MOQ 且 需要下单的货物， 凑至各自MOQ
    """
    part_one = df_base.query("is_union_mdq == 0 and (transit > 0)").reset_index()
    part_one['transit_mdq'] = part_one['transit'].where(part_one['transit_pur'] > part_one['mdq_pur'],
                                                        other=part_one['mdq_pur'] * part_one['pur_ratio'])
    res_part_one = part_one[res_columns]
    """
    联合凑MOQ货物
    1. 若需采购的各货物联合凑完之后满足MOQ要求，则不做其他处理
    2. 若需采购的各货物联合凑完之后不满足MOQ要求，拉齐各个货物（包括此次不需要采购的）的 当前库存 + VLT_BP期间在途 的可用天数，直至满足MOQ要求
    """
    part_two = df_base.query("is_union_mdq == 1").reset_index()
    # 采购量-重量
    part_two['transit_weight'] = part_two['transit_pur'] * part_two['pur_unit_gross_weight']
    # 采购量-体积
    part_two['transit_volume'] = part_two['transit_pur'] * part_two['pur_unit_volume']
    # 按单仓-单供应商汇总采购量，箱、重量、体积
    part_two[['transit_union_pur', 'transit_union_weight', 'transit_union_volume']] = \
    part_two.groupby(['supplier_code', 'wh_dept_id', 'moq_type', 'union_mdq'])[['transit_pur', 'transit_weight', 'transit_volume']].transform('sum')

    # 计算可用天数
    part_two['avl_days'] = (part_two['sim_beg'] + part_two['trs_bp_vlt'] + part_two['transit']) / part_two['pred_bp_vlt_dmd_avg']

    part_two = part_two[
        ['dt', 'wh_dept_id', 'goods_id', 'supplier_code', 'union_mdq', 'moq_type', 'transit_union_pur', 'pur_unit', 'transit_union_weight',
         'pur_unit_gross_weight', 'transit_union_volume', 'pur_unit_volume', 'pur_ratio', 'pred_bp_vlt_dmd_avg', 'avl_days', 'transit', 'transit_mdq',
         'order_dt', 'plan_order_dt']]
    res_part_two_list = []

    # 遍历按箱、按体积、按重量凑MDQ
    for (moq_type, moq_col) in dict({1: ['transit_union_pur', 'pur_unit'], 2: ['transit_union_volume', 'pur_unit_volume'],
                                     3: ['transit_union_weight', 'pur_unit_gross_weight']}).items():
        if len(part_two.query(f"moq_type == {moq_type}")) > 0:
            # 满足条件1的部分
            part_two_1 = part_two.query(f"{moq_col[0]} >= union_mdq")
            res_part_two_list.append(part_two_1)
            # 不满足的部分
            part_two_2_base = part_two.query(f"(0 < {moq_col[0]}) and ({moq_col[0]} < union_mdq)")
            if len(part_two_2_base) > 0:
                part_two_2 = cal_union_mdq(part_two_2_base, moq_col[0], moq_col[1])
                res_part_two_list.append(part_two_2)

    res_part_two = pd.concat(res_part_two_list)[res_columns] if len(res_part_two_list) > 0 else None
    # 过滤不要采购货物
    res_mdq = pd.concat([res_part_one, res_part_two]).query("transit_mdq > 0")
    return res_mdq


class Sim(multiprocessing.Process):
    def __init__(self, wh_dept_id: str, large_class_name, sim_start_dt, sim_len, df_cg_config, df_dly_inv, df_pred):
        multiprocessing.Process.__init__(self)
        self.wh_dept_id = wh_dept_id
        self.large_class_name = large_class_name
        self.sim_start_dt = sim_start_dt
        self.sim_len = sim_len
        self.df_cg_config = df_cg_config
        self.df_dly_inv = df_dly_inv
        self.df_pred = df_pred

    def run(self):

        dict_beg_inv = dict()
        _df_transit = pd.DataFrame(columns=['dt', 'wh_dept_id', 'goods_id', 'transit', 'transit_mdq', 'order_dt', 'plan_order_dt'])
        _df_ss_ro = pd.DataFrame(columns=['dt', 'wh_dept_id', 'goods_id', 'sim_ss', 'sim_ro', 'sim_dmd_avg', 'ss', 'ro', 'dly_avg'])
        res_everyday_list = []
        ss_ro_everyday_list = []
        transit_everyday_list = []
        # 模拟未来sim_len天
        for index in range(self.sim_len):
            sim_dt = str(pd.to_datetime(self.sim_start_dt).date() + timedelta(days=index))

            # 每日模拟结果
            res_every_goods_list = []
            # 遍历所有货物
            for (wh_dept_id, goods_id), group in self.df_cg_config.groupby(['wh_dept_id', 'goods_id']):
                logger.info(f"{sim_dt}, {wh_dept_id}, {goods_id}")
                df_config = self.df_cg_config.query(f"(goods_id == {goods_id})")
                df_wh_dly = self.df_dly_inv.query(f"(goods_id == {goods_id})")
                df_pred = self.df_pred.query(f"(goods_id == {goods_id}) and (dt == '{sim_dt}')")
                beg_cnt_base = df_wh_dly.query(f"dt == '{sim_dt}'")

                if (len(df_wh_dly) == 0) or (len(df_pred.query("demand > 0")) == 0):
                    continue

                beg_cnt = beg_cnt_base['beg_avl_inv'].values[0]
                # 记录期初，若有模拟期末，则取模拟期末
                key_beg_inv = f"{wh_dept_id}_{goods_id}"
                if dict_beg_inv.get(key_beg_inv):
                    beg_cnt = dict_beg_inv.get(key_beg_inv)
                else:
                    dict_beg_inv[key_beg_inv] = beg_cnt

                _df_res = pd.DataFrame(
                    columns=["dt", "wh_dept_id", "wh_name", "goods_id", "sim_beg", "sim_end", 'end_cnt', 'sim_transit', 'transit', "pred_demand",
                             'pred_bp_vlt_dmd_avg', 'trs_bp_vlt', 'dly_out', 'order_dt', 'plan_order_dt'])
                logger.info('Sim Start')
                _res_goods, _df_transit, _df_ss_ro = inv_sim(sim_dt, df_config, beg_cnt, df_pred, df_wh_dly, _df_transit, _df_ss_ro, _df_res,
                                                             dict_beg_inv, key_beg_inv)
                res_every_goods_list.append(_res_goods)

            # 每日所有货物结果
            _res_day = pd.concat(res_every_goods_list)
            # 若当日有下单，对下单货物进行MDQ处理
            trs_sim_dt = _df_transit.query(f"order_dt == '{sim_dt}'")
            if len(trs_sim_dt) > 0:
                logger.info("MDQ Process Start")
                transit_mdq = mdq_process(_res_day, self.df_cg_config, trs_sim_dt)
                _df_transit = pd.concat([_df_transit.query(f"order_dt != '{sim_dt}'"), transit_mdq])
                logger.info("MDQ Process End")
            # 保持每日结果
            res_everyday_list.append(_res_day)
        res = pd.concat(res_everyday_list)
        save_file(res, f'/data/purchase/simulate/v2/{today}/res/{self.wh_dept_id}/{self.large_class_name}.parquet')
        if len(ss_ro_everyday_list) > 0:
            save_file(pd.concat(ss_ro_everyday_list), f'/data/purchase/simulate/v2/{today}/ss_ro/{self.wh_dept_id}/{self.large_class_name}.parquet')
        if len(transit_everyday_list) > 0:
            save_file(pd.concat(transit_everyday_list), f'/data/purchase/simulate/v2/{today}/transit/{self.wh_dept_id}/{self.large_class_name}.parquet')


def get_config():
    # CDC、BP、RO
    cg_config_base = spark.sql(f"""
        SELECT
            goods_id
            , wh_dept_id
            , cdc_wh_dept_id
            , is_cdc
            , is_cdc_model
            , bp
            , ro
            , IF(service_level < service_level_limit, service_level_limit, service_level) AS sl
        FROM dw_ads_scm_alg.dim_automatic_order_base_cfg
        WHERE 1=1
            AND dt = DATE_SUB(CURRENT_DATE(), 0)
    """)
    # VLT, MDQ
    vlt_admin = spark.sql(f"""
            SELECT
                wh_dept_id
                , wh_name
                , base.goods_id
                , base.goods_name
                , base.large_class_name
                , base.spec_id
                , is_formula
                , admin_vlt
                , mdq_pur
                , pur_ratio
                , pur_unit_gross_weight
                , pur_unit_volume
                , 1 AS pur_unit
                , supplier_code
            FROM (
                SELECT DISTINCT
                    wh.wh_dept_id
                    , wh_name
                    , spec_info.goods_name
                    , spec_info.goods_id
                    , spec_info.spec_id
                    , spec_info.large_class_name
                    , admin_vlt
                    , mdq_pur
                    , pur_dly_ratio * dly_use_ratio AS pur_ratio
                    , pur_unit_gross_weight
                    , pur_unit_volume
                    , last_modify_time
                    , supplier_code
                    , RANK() OVER (PARTITION BY wh_dept_id, wh_name, goods_id, goods_name ORDER BY last_modify_time DESC) AS rnk -- 取最近修改的采购关注规格
                FROM (
                    SELECT
                        warehouse_id
                        , goods_spec_id AS spec_id
                        , vlt AS admin_vlt
                        , minimum_delivery AS mdq_pur
                        , supplier_mid AS supplier_code
                        , last_modify_time
                    FROM dw_dim.dim_goods_spec_city_config_d_his
                    WHERE dt = '{yesterday}'
                        AND vlt IS NOT NULL
                        AND city_purchase_status = 1
                        AND delete_flag = 1
                ) cfg
                INNER JOIN(
                    SELECT DISTINCT wh_id, wh_dept_id, wh_name
                    FROM dw_ads_scm_alg.dim_warehouse_city_shop_d_his
                    WHERE dt = (SELECT MAX(dt) FROM dw_ads_scm_alg.dim_warehouse_city_shop_d_his)
                ) wh ON wh.wh_id = cfg.warehouse_id
                LEFT JOIN dw_dim.dim_stock_spec_d_his spec_info
                    ON spec_info.dt = '{yesterday}'AND cfg.spec_id = spec_info.spec_id
            ) base
            LEFT JOIN dw_ads_scm_alg.dev_luckin_demand_forecast_category_info1 formula
                ON formula.dt = '{yesterday}' AND base.goods_id = formula.goods_id
            WHERE rnk == 1
        """)
    # 联合MDQ
    mdq_type = spark.sql(f"""
        SELECT
            wh_dept_id
            , spec_id
            , supplier_code
            , moq AS union_mdq
            , moq_limit_mode AS moq_type
            , 1 AS is_union_mdq
        FROM dw_ads_scm_alg.dim_wh_spec_moq_cfg
        WHERE 1=1
            AND dt = DATE_SUB(CURRENT_DATE(), 0)
    """)
    vlt_mdq = vlt_admin.merge(mdq_type, on=['wh_dept_id', 'spec_id', 'supplier_code'], how='left')
    # 调拨
    alt_admin = spark.sql("""
        SELECT DISTINCT
            outcome_wh_id AS out_wh_dept_id
            , income_wh_id AS wh_dept_id
            , cost_day AS alt_avg
        FROM lucky_stock.t_stock_allocation_time_config
    """)

    df_cg_config = cg_config_base.merge(vlt_mdq, on=['wh_dept_id', 'goods_id'], how='inner') \
        .merge(alt_admin, left_on=['wh_dept_id', 'cdc_wh_dept_id'], right_on=['wh_dept_id', 'out_wh_dept_id'], how='left')

    def get_vlt(row):
        alt, vlt_admin, is_cdc, is_cdc_model = row.alt_avg, row.admin_vlt, row.is_cdc, row.is_cdc_model
        vlt = alt if (is_cdc == 0) & (is_cdc_model == 1) else vlt_admin
        vlt_std = vlt * 0.2
        vlt_std = vlt_std if vlt_std < 7 else 7
        vlt_std = vlt_std if vlt_std > 2 else 2
        return vlt, vlt_std

    df_cg_config[['vlt', 'vlt_std']] = df_cg_config.apply(lambda x: get_vlt(x), axis=1, result_type='expand')
    df_cg_config.fillna({'is_union_mdq': 0}, inplace=True)
    return df_cg_config


def start():
    ths = []
    sim_start_dt = '2021-06-01'
    sim_len = 30
    sim_end_dt = pd.to_datetime(sim_start_dt).date() + timedelta(days=sim_len)

    df_dly_inv = spark.read_parquet(f"/user/yanxin.lu/sc/simulate/wh_dly_out/")
    df_pred_all = spark.read_parquet(f'/user/yanxin.lu/sc/simulate/demand/')
    df_cg_config = get_config()

    for wh_dept_id in df_cg_config['wh_dept_id'].drop_duplicates():
        logger.info(f"{wh_dept_id}")
        df_cg_config_wh = df_cg_config.query(f"wh_dept_id == {wh_dept_id}")
        df_dly_inv_wh = df_dly_inv.query(f"wh_dept_id == {wh_dept_id}")
        df_pred_all_wh = df_pred_all.query(f"wh_dept_id == {wh_dept_id} and (dt >= '{sim_start_dt}') and (dt <= '{sim_end_dt}')")

        for large_class_name in ['包装类', '零食', '轻食', '原料']:
            logger.info(f"{wh_dept_id}, {large_class_name}")
            _df_cg_config = df_cg_config_wh.query(f"large_class_name == '{large_class_name}'")
            _df_dly_inv = df_dly_inv_wh.query(f"large_class_name == '{large_class_name}'")
            _df_pred = df_pred_all_wh.query(f"large_class_name == '{large_class_name}'")
            thread = Sim(wh_dept_id, large_class_name, sim_start_dt, sim_len, _df_cg_config, _df_dly_inv, _df_pred)
            thread.daemon = True
            thread.start()
            ths.append(thread)

    for th in ths:
        th.join()


if __name__ == '__main__':
    start()
