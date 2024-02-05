# encoding: utf-8 
""" 
@Project:workspace 
@Created: 2023/9/8 
@Author: cuiyuhan 
"""
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from utils.dtype_monitor_transform import *


class BaseSimMethod:
    def dly_resign(self, df_resign):
        df_resign = df_resign.copy()

        # 如果仓库期初wh_begin >=订货单总量wh_order_sum，则配货量resign=order_num
        resign_ls = np.where(df_resign["wh_begin"] >= df_resign["wh_order_sum"], df_resign["order_num"], 0)
        df_resign["resign"] = resign_ls
        # 如果wh_begin <订货单总量wh_order_sum
        df_resign_less = df_resign[df_resign["wh_begin"] < df_resign["wh_order_sum"]].copy()

        if df_resign_less.empty:  # 当df_resign_less为空时，直接返回df_resign
            return df_resign

        # 按照订货量order_num从大到小排序
        df_resign_less.sort_values(by="order_num", ascending=False, inplace=True)

        # 记录当前仓库剩余库存
        wh_remain = df_resign_less["wh_begin"].values[0]

        i = 0
        while wh_remain > 0 and i < df_resign_less.shape[0]:
            amount_to_resign = min(df_resign_less.iloc[i, df_resign_less.columns.get_loc('dly_use_ratio')],
                                   df_resign_less.iloc[i, df_resign_less.columns.get_loc('order_num')] -
                                   df_resign_less.iloc[i, df_resign_less.columns.get_loc('resign')])
            if amount_to_resign > 0:
                df_resign_less.iloc[i, df_resign_less.columns.get_loc('resign')] += amount_to_resign
                wh_remain -= amount_to_resign
            i += 1
            if i == df_resign_less.shape[0]:
                i = 0

        # 合并两部分结果
        df_resign = pd.concat([df_resign[df_resign["resign"] > 0], df_resign_less])

        return df_resign

    def simulated_wh_resign(self, df_sub_re):
        df_resign_res = pd.DataFrame()
        for goods_id in df_sub_re["goods_id"].unique():
            sel_idx_goods = df_sub_re["goods_id"] == goods_id
            df_sub_re_goods = df_sub_re[sel_idx_goods].reset_index(drop=True).copy()

            for wh_dept_id in df_sub_re_goods["wh_dept_id"].unique():
                sel_idx_wh = df_sub_re_goods["wh_dept_id"] == wh_dept_id
                # 模拟短配
                df_sub_re_wh_goods = df_sub_re_goods[sel_idx_wh]
                df_sub_re_wh_res = self.dly_resign(df_sub_re_wh_goods)
                df_resign_res = pd.concat([df_resign_res, df_sub_re_wh_res])
        return df_resign_res

    def run_newp_shop_wh_simulate(self, df_sim_base, df_wh_begin, df_wh_transit):
        """
        含短配仓库门店订货模拟
        :param df_theory_order_base:
        :param df_stock_po_base:
        :param df_transit:
        :return:
        """
        log.debug(f'出库模拟开始')

        df_base = (df_sim_base
                   .sort_values(["dt", "dept_id", "goods_id"])
                   .reset_index(drop=True)
                   .copy())

        # 门店期初库存
        # df_base['shop_beg'] = df_base['shop_beg'] - df_base['after_check']
        df_sub_shop_beg = df_base.groupby(["dept_id", "goods_id"]).head(1)[["dept_id", "goods_id", "shop_beg"]]

        # 当天到货 和 未来在途
        cols_dly_num = ["dt", "dept_id", "goods_id", "expected_receive"]
        df_dly_num = df_base.query("expected_receive > 0")[cols_dly_num].copy()

        # 仓库当前库存
        df_wh_now = df_wh_begin[["wh_dept_id", "goods_id", "beg_wh_stock"]].drop_duplicates()

        # 仓库在途库存
        cols_wh_trans = ["dt", "wh_dept_id", "goods_id", "transit"]
        df_wh_trans = df_wh_transit.query("transit > 0")[cols_wh_trans]

        # 每日 计算结果
        df_res_shop = pd.DataFrame()
        df_res_order = pd.DataFrame()
        df_res_wh = pd.DataFrame()

        # 最基础列
        cols_sub = [
            'dept_id', 'goods_id', 'dt', 'pred_consume', 'ss_ro', 'closed',
            'bp_consume', 'vlt_consume', 'dly_use_ratio', 'can_order', 'receive_day', 'wh_dept_id'
        ]

        for dt in df_base["dt"].unique():

            # 当天基础数据
            df_sub = (df_base
                      .query(f"dt == '{dt}'")[cols_sub]
                      # 关联 当天期初
                      .merge(df_sub_shop_beg, "left", on=["dept_id", "goods_id"]).fillna(0)
                      .copy())

            # 关联 当天到货
            df_sub_rec = (df_dly_num.query(f"dt == '{dt}'")
                          .groupby(["dt", "dept_id", "goods_id"])["expected_receive"]
                          .sum().reset_index()
                          )
            df_sub = df_sub.merge(df_sub_rec, "left", on=["dt", "dept_id", "goods_id"]).fillna(0)

            # 计算门店期末
            df_sub['shop_end'] = np.clip(df_sub.eval("shop_beg + expected_receive - pred_consume"), 0, np.inf)
            # 准备 第二天期初数据（当天期末未第二天期初）
            df_sub_shop_beg = df_sub[["dept_id", "goods_id", "shop_end"]].rename(columns={"shop_end": "shop_beg"})

            # 记录门店数据
            df_res_shop = pd.concat([df_res_shop, df_sub])

            # =====
            # 计算订货量

            # 门店 在途信息
            df_sub_dly_num = (df_dly_num
                              .query(f"dt > '{dt}'")
                              .groupby(["dept_id", "goods_id"])["expected_receive"]
                              .sum().rename("dly_num")
                              .reset_index())

            # 准备 订货量计算 相关数据
            df_sub_order = (df_sub
                            # 筛选当前需要订货门店
                            .query("can_order == 1")
                            # 关联 在途
                            .merge(df_sub_dly_num, "left", on=["dept_id", "goods_id"])
                            .fillna(0))

            # 预估剩余库存
            df_sub_order['pred_vlt_end'] = np.clip(
                df_sub_order.eval("shop_beg + expected_receive + dly_num - vlt_consume"), 0, np.inf)

            # 订货量 = 需求- 门店现有库存
            df_sub_order['ti_amount'] = np.clip(df_sub_order.eval("bp_consume + ss_ro"), 0, np.inf)

            df_sub_order['order_calc'] = np.clip(df_sub_order.eval("ti_amount - pred_vlt_end"), 0, np.inf)
            # ① 0：不订货  ② 0 < 订货量 < 箱规：向上取整  ③ 箱规 < 订货量：四舍五入
            df_sub_order['order_num'] = np.where(
                df_sub_order.eval("0 < order_calc < dly_use_ratio"),
                df_sub_order["dly_use_ratio"], df_sub_order["order_calc"])
            df_sub_order['order_num'] = df_sub_order.eval("order_num / dly_use_ratio").round(0) * df_sub_order[
                "dly_use_ratio"]

            # =====
            # 计算仓库确认配货量

            df_sub_wh = df_wh_now.copy()
            df_sub_wh["dt"] = dt
            # 关联 在途
            df_sub_wh = df_sub_wh.merge(df_wh_trans, "left", on=["wh_dept_id", "goods_id", "dt"]).fillna(0)
            # 关联 当天订货量
            df_sub_order_all = (df_sub_order
                                .groupby(["wh_dept_id", "goods_id"])["order_num"]
                                .sum().rename("wh_order_sum")
                                .reset_index())
            df_sub_wh = df_sub_wh.merge(df_sub_order_all, "left", on=["wh_dept_id", "goods_id"]).fillna(0)
            df_sub_wh["wh_begin"] = df_sub_wh.eval("beg_wh_stock + transit")
            df_sub_wh["wh_end"] = np.clip(df_sub_wh.eval("wh_begin - wh_order_sum"), 0, np.inf)
            df_sub_wh["wh_short"] = np.clip(df_sub_wh.eval("wh_order_sum-wh_begin"), 0, np.inf)

            # 记录数据
            df_res_wh = pd.concat([df_res_wh, df_sub_wh])

            # ---

            # 准备数据：第二天 仓库期初
            df_wh_now = df_sub_wh[["wh_dept_id", "goods_id", "wh_end"]].rename(
                columns={"wh_end": "beg_wh_stock"}).copy()

            df_dly_num = df_dly_num.query(f"dt > '{dt}'")

            # 根据仓库库存 确认配货数量
            df_sub_wh_confirm = df_sub_order.merge(
                df_sub_wh[["wh_dept_id", "goods_id", "wh_order_sum", "wh_begin"]], "left").fillna(0)

            if len(df_sub_wh_confirm) > 0:
                df_sub_wh_confirm = self.simulated_wh_resign(df_sub_wh_confirm)

                # 记录数据
                df_res_order = pd.concat([df_res_order, df_sub_wh_confirm])

                # ---
                # 准备数据：第二天 门店在途信息

                df_sub_dly_future = (df_sub_wh_confirm
                                     .query("resign > 0")[["receive_day", "dept_id", "goods_id", "resign"]]
                                     .rename(columns={"receive_day": "dt", "resign": "expected_receive"}))

                df_dly_num = pd.concat([df_dly_num, df_sub_dly_future])
                log.debug(f'{dt}_finish')
        return df_res_shop, df_res_order, df_res_wh

    def run_newp_shop_wh_simulate_no_resign(self, df_sim_base, include_col=[]):
        """
        无短配门店订货模拟
        :param df_theory_order_base:
        :param include_col:
        :return:
        """

        log.debug(f'无短配订货模拟开始')

        df_base = (df_sim_base
                   .sort_values(["dt", "dept_id", "goods_id"])
                   .reset_index(drop=True)
                   .copy())

        # 门店期初库存
        # df_base['shop_beg'] = df_base['shop_beg'] - df_base['after_check']
        df_sub_shop_beg = df_base.groupby(["dept_id", "goods_id"]).head(1)[["dept_id", "goods_id", "shop_beg"]]

        # 当天到货 和 未来在途
        cols_dly_num = ["dt", "dept_id", "goods_id", "expected_receive"]
        df_dly_num = df_base.query("expected_receive > 0")[cols_dly_num].copy()

        # 每日 计算结果
        df_res_shop = pd.DataFrame()
        df_res_order = pd.DataFrame()

        # 最基础列
        cols_sub = [
                       'dept_id', 'goods_id', 'dt', 'pred_consume', 'ss_ro', 'closed',
                       'bp_consume', 'vlt_consume', 'dly_use_ratio', 'can_order', 'receive_day', 'wh_dept_id'
                   ] + include_col

        for dt in df_base["dt"].unique():
            # 当天基础数据
            df_sub = (df_base
                      .query(f"dt == '{dt}'")[cols_sub]
                      # 关联 当天期初
                      .merge(df_sub_shop_beg, "left", on=["dept_id", "goods_id"]).fillna(0)
                      .copy())

            # 关联 当天到货
            df_sub_rec = (df_dly_num.query(f"dt == '{dt}'")
                          .groupby(["dt", "dept_id", "goods_id"])["expected_receive"]
                          .sum().reset_index()
                          )
            df_sub = df_sub.merge(df_sub_rec, "left", on=["dt", "dept_id", "goods_id"]).fillna(0)

            # 计算门店期末
            df_sub['shop_end'] = np.clip(df_sub.eval("shop_beg + expected_receive - pred_consume"), 0, np.inf)
            # 准备 第二天期初数据（当天期末未第二天期初）
            df_sub_shop_beg = df_sub[["dept_id", "goods_id", "shop_end"]].rename(columns={"shop_end": "shop_beg"})

            # 记录门店数据
            df_res_shop = pd.concat([df_res_shop, df_sub])

            # =====
            # 计算订货量

            # 门店 在途信息
            df_sub_dly_num = (df_dly_num
                              .query(f"dt > '{dt}'")
                              .groupby(["dept_id", "goods_id"])["expected_receive"]
                              .sum().rename("dly_num")
                              .reset_index())

            # 准备 订货量计算 相关数据
            df_sub_order = (df_sub
                            # 筛选当前需要订货门店
                            .query("can_order == 1")
                            # 关联 在途
                            .merge(df_sub_dly_num, "left", on=["dept_id", "goods_id"])
                            .fillna(0))

            # 预估剩余库存
            df_sub_order['pred_vlt_end'] = np.clip(
                df_sub_order.eval("shop_beg + expected_receive + dly_num - vlt_consume"), 0, np.inf)

            # 订货量 = 需求- 门店现有库存
            df_sub_order['ti_amount'] = np.clip(df_sub_order.eval("bp_consume + ss_ro"), 0, np.inf)

            df_sub_order['order_calc'] = np.clip(df_sub_order.eval("ti_amount - pred_vlt_end"), 0, np.inf)
            # ① 0：不订货  ② 0 < 订货量 < 箱规：向上取整  ③ 箱规 < 订货量：四舍五入
            df_sub_order['order_num'] = np.where(
                df_sub_order.eval("0 < order_calc < dly_use_ratio"),
                df_sub_order["dly_use_ratio"], df_sub_order["order_calc"])
            df_sub_order['order_num'] = df_sub_order.eval("order_num / dly_use_ratio").round(0) * df_sub_order[
                "dly_use_ratio"]

            df_dly_num = df_dly_num.query(f"dt > '{dt}'")

            # 记录数据
            df_res_order = pd.concat([df_res_order, df_sub_order])

            # ---
            # 准备数据：第二天 门店在途信息

            df_sub_dly_future = (df_sub_order[["receive_day", "dept_id", "goods_id", "order_num"]]
                                 .rename(columns={"receive_day": "dt", "order_num": "expected_receive"}))

            df_dly_num = pd.concat([df_dly_num, df_sub_dly_future])

            log.debug(f'{dt}_finish')
        return df_res_order

    def get_cg_trans(self, df_transit_m1):

        df = (df_transit_m1
              .sort_values(["dt", "wh_dept_id", "goods_id"])
              .reset_index(drop=True)
              .copy())

        def dynamic_rolling(series, window_series):
            return series[::-1].rolling(window=int(window_series.iloc[0])).sum()[::-1]

        df['lt_consume'] = df.groupby(['wh_dept_id', 'goods_id']).apply(
            lambda x: dynamic_rolling(x['theory_order_sum'], x['lt_cg'])).reset_index(level=[0, 1],
                                                                                      drop=True)

        df['bp_lt_consume'] = df.groupby(['wh_dept_id', 'goods_id']).apply(
            lambda x: dynamic_rolling(x['theory_order_sum'], x['lt_cg'] + x['bp_cg'])).reset_index(
            level=[0, 1], drop=True)
        df['ss_amount'] = df.groupby(['wh_dept_id', 'goods_id']).apply(
            lambda x: x['bp_lt_consume'] / (x['lt_cg'] + x['bp_cg']) * x['ss_cg']).reset_index(
            level=[0, 1], drop=True)
        df = df.fillna(0)
        df['plan_finish_date'] = df.apply(lambda x: pd.to_datetime(x['dt']) + pd.to_timedelta(x.lt_cg, 'd'), axis=1)

        # 门店期初库存
        df_sub_wh_beg = df.groupby(["wh_dept_id", "goods_id"]).head(1)[["wh_dept_id", "goods_id", "beg_wh_stock"]]
        df_sub_wh_beg['beg_wh_stock'] = 0

        # 动态加入现有库存
        df_dly_num_p1 = df.groupby(['wh_dept_id', 'goods_id'], as_index=False).agg({'dt': 'min', 'beg_wh_stock': 'max'}) \
            .rename(columns={'beg_wh_stock': 'transit'})
        dly_col = ['wh_dept_id', 'goods_id', 'dt', 'transit']
        # 加入在途
        df_dly_num_p2 = df.query("transit>0")[dly_col]
        df_dly_concat = pd.concat([df_dly_num_p1[dly_col], df_dly_num_p2[dly_col]])
        df_dly_num = df_dly_concat \
            .groupby(["dt", "wh_dept_id", "goods_id"], as_index=False) \
            .agg({'transit': 'sum'})

        # 生成在途cg
        # dt_start = df.dt.min()
        cols_sub = ['dt', 'wh_dept_id', 'goods_id', 'theory_order_sum', 'lt_consume', 'bp_lt_consume', 'ss_amount',
                    'plan_finish_date', 'pur_use_ratio']

        # 仓库当前库存
        df_po_now = df[["wh_dept_id", "goods_id", "po_remain"]].drop_duplicates()

        df_wh_beg = pd.DataFrame()
        df_res_order = pd.DataFrame()
        df_res_po = pd.DataFrame()

        for dt in df['dt'].unique():
            log.debug(f'{dt}cg模拟开始')

            df_sub = df.query(f"dt=='{dt}'")[cols_sub] \
                .merge(df_sub_wh_beg, "left", on=["wh_dept_id", "goods_id"]) \
                .fillna(0)
            # 关联 当天到货
            df_sub_rec = (df_dly_num.query(f"dt == '{dt}'")
                          .groupby(["dt", "wh_dept_id", "goods_id"])["transit"]
                          .sum().reset_index()
                          )
            df_sub = df_sub.merge(df_sub_rec, "left", on=["dt", "wh_dept_id", "goods_id"]).fillna(0)
            df_sub['end_wh_stock'] = np.clip(df_sub.eval("beg_wh_stock + transit - theory_order_sum"), 0, np.inf)

            # 准备 第二天期初数据（当天期末未第二天期初）
            df_sub_wh_beg = df_sub[["wh_dept_id", "goods_id", "end_wh_stock"]].rename(
                columns={"end_wh_stock": "beg_wh_stock"})
            df_wh_beg = pd.concat([df_wh_beg, df_sub])

            df_sub_dly_num = (df_dly_num
                              .query(f"dt > '{dt}'")
                              .groupby(["wh_dept_id", "goods_id"])["transit"]
                              .sum().rename("vlt_transit")
                              .reset_index())
            df_sub_order = (df_sub
                            # 关联 在途
                            .merge(df_sub_dly_num, "left", on=["wh_dept_id", "goods_id"])
                            .fillna(0))
            df_sub_order['pred_vlt_end'] = np.clip(
                df_sub_order.eval("beg_wh_stock + transit + vlt_transit - lt_consume"), 0, np.inf)
            df_sub_order['ti_amount'] = np.clip(df_sub_order.eval("(bp_lt_consume-lt_consume) + ss_amount"), 0, np.inf)
            df_sub_order['if_trigger_cg'] = df_sub_order['pred_vlt_end'] <= df_sub_order['ti_amount']
            df_sub_order['cg_order_calc'] = np.clip(df_sub_order.eval("ti_amount - pred_vlt_end"), 0, np.inf)
            df_sub_order['cg_order_num'] = np.ceil(df_sub_order['cg_order_calc'] / df_sub_order['pur_use_ratio']) * \
                                           df_sub_order[
                                               "pur_use_ratio"]

            df_sub_po = df_po_now.copy()
            df_sub_po["dt"] = dt
            # 关联 当天订货量
            #     po在途暂未考虑
            #     df_sub_wh = df_sub_wh.merge(df_wh_trans, "left", on=["wh_dept_id", "goods_id", "dt"]).fillna(0)
            #     df_sub_wh["wh_begin"] = df_sub_wh.eval("beg_wh_stock + transit")
            col = ["wh_dept_id", "goods_id", "plan_finish_date", "transit", "vlt_transit","pred_vlt_end", "ti_amount", 'if_trigger_cg', "cg_order_calc",
                   "cg_order_num"]
            df_sub_po = df_sub_po.merge(df_sub_order[col], "left", on=["wh_dept_id", "goods_id"]).fillna(0)
            df_sub_po["po_end"] = np.clip(df_sub_po.eval("po_remain - cg_order_num"), 0, np.inf)
            df_sub_po["po_short"] = np.clip(df_sub_po.eval("cg_order_num-po_remain"), 0, np.inf)

            df_res_po = pd.concat([df_res_po, df_sub_po])

            # 准备数据：第二天 仓库期初
            df_po_now = df_sub_po[["wh_dept_id", "goods_id", "po_end"]].rename(
                columns={"po_end": "po_remain"}).copy()

            df_dly_num = df_dly_num.query(f"dt > '{dt}'")

            # 根据仓库库存 确认配货数量
            df_sub_cg_confirm = df_sub_po.fillna(0)
            df_sub_cg_confirm['cg_pur_amount'] = df_sub_cg_confirm['cg_order_num'].where(
                df_sub_cg_confirm['cg_order_num'] <= df_sub_cg_confirm['po_remain'], df_sub_cg_confirm['po_remain'])

            df_res_order = pd.concat([df_res_order, df_sub_cg_confirm])

            # ---
            # 准备数据：第二天 门店在途信息

            df_sub_dly_future = (df_sub_cg_confirm
                                 .query("cg_pur_amount > 0")[
                                     ["plan_finish_date", "wh_dept_id", "goods_id", "cg_pur_amount"]]
                                 .rename(columns={"plan_finish_date": "dt", "cg_pur_amount": "transit"}))

            df_dly_num = pd.concat([df_dly_num, df_sub_dly_future])
            df_dly_num['dt'] = pd.to_datetime(df_dly_num['dt'])
            if df_sub_po.po_end.sum() == 0:
                print('po_remain==0')

                break
        df_output = df_res_order.query("cg_pur_amount > 0")[
            ["plan_finish_date", "wh_dept_id", "goods_id", "cg_pur_amount"]].rename(
            columns={"plan_finish_date": "dt", "cg_pur_amount": "transit"})
        df_output['dt'] = pd.to_datetime(df_output['dt'])
        return df_output


class CalcAvlDayRate:
    """
    仓库可用天数、门店可用天数、有货率计算
    """

    def __init__(self):
        self.select_goods_id = None
        self.df_avl_d_his = None
        self.material_type = None
        self.scene = None
        self.stock_up_version = None
        self.consumer_version = None
        self.strategy_version_id = None
        self.sim_version = None
        self.min_cs_start = None
        self.sim_end_date = None
        self.min_launch_date = None
        self.df_params = None

    @staticmethod
    def calc_wh_avl_days(df_res_order, df_res_wh):
        """
        仓库可用天数计算
        :param df_res_order: 配货单模拟结果
        :param df_res_wh: 仓库期初期末模拟结果
        :return:
        """
        df_res_order = df_res_order.sort_values('dt')
        df_res_wh = df_res_wh.sort_values('dt')

        df_wh_res_order = df_res_order.groupby(['goods_id', 'dt'], as_index=False).agg(
            {'resign': 'sum', 'pred_consume': 'sum', 'order_num': 'sum'})
        # 未来出库序列
        dmd_dly_all = df_wh_res_order.sort_values('dt')
        dmd_dly_all = dmd_dly_all.rename(columns={'dt': 'predict_dt'})
        dmd_cumsum = pd.DataFrame()

        for dt in df_res_wh['dt'].unique():
            dmd_all_sub = dmd_dly_all.query(f"predict_dt>='{dt}'").copy()
            dmd_all_sub['cumsum'] = dmd_all_sub.groupby(['goods_id'], as_index=False).order_num.cumsum()
            dmd_all_sub = dmd_all_sub.rename(columns={'dt': 'predict_dt'})
            dmd_all_sub['dt'] = dt
            dmd_cumsum = pd.concat([dmd_all_sub, dmd_cumsum])

        df_nation_end = df_res_wh.groupby(['dt', 'goods_id'], as_index=False).agg({'wh_begin': 'sum'})
        dmd_all_mid = df_nation_end.merge(dmd_cumsum, on=['dt', 'goods_id'], how='left').dropna(subset=['predict_dt'])
        dmd_all_mid['left'] = dmd_all_mid['wh_begin'] - dmd_all_mid['cumsum']

        wh_avl_m1 = dmd_all_mid.query("left>0 and cumsum>0").sort_values(['dt', 'goods_id', 'predict_dt'],
                                                                         ascending=[True, True, False]) \
            .groupby(['dt', 'goods_id'], as_index=False).agg({'predict_dt': 'max'})
        wh_avl_m1['wh_avl_days'] = (wh_avl_m1['predict_dt'] - wh_avl_m1['dt']) / timedelta(days=1)
        wh_avl_m1['wh_avl_days'] = np.clip(wh_avl_m1['wh_avl_days'], 0, np.inf)
        wh_avl_m1 = wh_avl_m1[['dt', 'goods_id', 'wh_avl_days']]
        # 保留模拟时间序列
        df_wh_avl = dmd_all_mid[['goods_id', 'dt']].drop_duplicates().merge(wh_avl_m1, how='left').fillna(0)
        return df_wh_res_order, df_wh_avl

    @staticmethod
    def calc_shop_avl_days(df_res_shop):
        """
        门店可用天数计算
        :param df_res_shop: 门店期初期末模拟结果
        :return:
        """
        df_shop_avl = df_res_shop.groupby(['dt', 'goods_id'], as_index=False).agg(
            {'pred_consume': 'sum', 'shop_end': 'sum'})
        df_shop_avl = df_shop_avl.sort_values('dt')
        df_shop_avl['pred_consume_mean'] = df_shop_avl.iloc[::-1]['pred_consume'].rolling(window=7,
                                                                                          min_periods=1).mean()

        df_shop_avl['shop_avl_days'] = np.where(df_shop_avl['pred_consume_mean'] == 0, 0,
                                                np.floor(df_shop_avl['shop_end'] / df_shop_avl['pred_consume_mean']))
        return df_shop_avl[['dt', 'goods_id', 'shop_end', 'shop_avl_days']]

    @staticmethod
    def calc_shop_avl_rate(df_res_shop):
        """
        【23.9.2】版本改为计算期初有货率
        模拟有货率计算
        :param df_res_shop: 门店期初期末模拟结果
        :return:
        """
        df_res_shop['is_has_stock'] = (df_res_shop['shop_beg'] + df_res_shop['expected_receive']) > 0
        df_shop_avl_rate = df_res_shop.query("closed==0").groupby(['dt'], as_index=False).is_has_stock.sum()
        df_shop_avl_rate['shop_cnt'] = len(df_res_shop['dept_id'].unique())
        df_shop_avl_rate['shop_avl_rate'] = df_shop_avl_rate['is_has_stock']/df_shop_avl_rate['shop_cnt']
        return df_shop_avl_rate[['dt', 'shop_avl_rate']]

    def get_sim_avl_info(self, version_id, df_res_shop, df_res_order, df_res_wh):
        # 仓库部分可用天数计算
        df_wh_res_order, df_wh_avl = self.calc_wh_avl_days(df_res_order, df_res_wh)

        # 门店部分可用天数计算
        df_shop_avl = self.calc_shop_avl_days(df_res_shop)

        # 有货率计算
        df_shop_avl_rate = self.calc_shop_avl_rate(df_res_shop)

        # dt列:v9.2业务要求加入上市前11天

        df_dt_all = pd.DataFrame({'dt': pd.date_range(self.min_cs_start, self.sim_end_date)})
        df_dt_all['goods_id'] = self.select_goods_id

        # 整合
        df_avl_all = df_dt_all.merge(df_shop_avl_rate, how='left') \
            .merge(df_shop_avl, how='left') \
            .merge(df_wh_avl, how='left') \
            .merge(df_wh_res_order, how='left')
        # 处理空值
        df_avl_all[['pred_consume', 'order_num']] = df_avl_all[['pred_consume', 'order_num']].fillna(0)
        df_avl_all[['shop_avl_days', 'wh_avl_days']] = df_avl_all[['shop_avl_days', 'wh_avl_days']].fillna(0)

        # 加入历史有货率
        if len(self.df_avl_d_his) > 0:
            self.df_avl_d_his['dt'] = pd.to_datetime(self.df_avl_d_his['dt'])
            df_avl_all = df_avl_all.merge(self.df_avl_d_his, how='left')
        else:
            df_avl_all['shop_actual_avl_rate'] = np.nan
        df_avl_all['shop_avl_rate'] = df_avl_all['shop_actual_avl_rate'].fillna(df_avl_all['shop_avl_rate']).fillna(0)
        df_avl_all['shop_avl_rate'] = np.round(df_avl_all['shop_avl_rate'], 5)
        # 加入基础信息
        df_avl_all['material_type'] = self.material_type
        df_avl_all['scene'] = self.scene
        df_avl_all['stock_up_version'] = self.stock_up_version
        df_avl_all['consumer_version'] = version_id
        df_avl_all['strategy_version_id'] = self.strategy_version_id
        df_avl_all['sim_version'] = self.sim_version
        if self.scene == 1:
            df_avl_all['min_launch_date'] = self.min_launch_date
        else:
            df_avl_all['min_launch_date'] = np.nan

        df_avl_all['is_min_launch_day'] = df_avl_all['min_launch_date'] == df_avl_all['dt']
        df_avl_all['is_min_launch_day'] = df_avl_all['is_min_launch_day'].astype(int)
        if self.material_type == 0:
            df_avl_all['dly_use_ratio'] = np.nan
            df_avl_all['pur_use_ratio'] = np.nan
        else:
            df_avl_all['dly_use_ratio'] = self.df_params['delivery_use_ratio'].values[0]
            df_avl_all['pur_use_ratio'] = self.df_params['purchase_use_ratio'].values[0]

        # 更改列名、变更列类型
        choose_col = ['dt', 'goods_id', 'material_type', 'scene', 'stock_up_version', 'consumer_version',
                      'strategy_version_id',
                      'sim_version', 'shop_avl_rate', 'wh_avl_days', 'shop_avl_days', 'dly_use_ratio', 'pur_use_ratio',
                      'is_min_launch_day', 'pred_consume', 'order_num']
        df_avl_res = df_avl_all[choose_col]

        avl_col_name = {'dt': 'predict_dt', 'pred_consume': 'shop_consume_amount', 'order_num': 'wh_consume_amount'}
        df_avl_res = df_avl_res.rename(columns=avl_col_name)

        int_type_col = ['goods_id', 'material_type', 'stock_up_version', 'consumer_version', 'strategy_version_id',
                        'sim_version',
                        'wh_avl_days', 'is_min_launch_day', 'shop_avl_days']
        float_type_col = ['shop_avl_rate', 'dly_use_ratio', 'shop_consume_amount', 'wh_consume_amount']
        df_avl_res[int_type_col] = df_avl_res[int_type_col].astype('int')
        df_avl_res[float_type_col] = df_avl_res[float_type_col].astype(float)
        df_avl_res['predict_dt'] = df_avl_res['predict_dt'].dt.strftime('%Y-%m-%d')

        return df_avl_res

    @staticmethod
    def calculate_end_date(df, condition_col, theory_consume, dt_start):

        # calculate_end_date(df_transit_m1_nation, 'stock_end','stock_remain', last_seven_dly, dt_all)
        df_out = df.copy()

        condition_df_out = df_out[df_out[condition_col] == 0]
        dt_end = df_out.dt.max()
         # 没有截止时刻
        if condition_df_out.empty:
            target_remain = np.ceil(
                df_out.loc[df_out.dt == dt_end, condition_col].values[0] / theory_consume)
            dt_end = dt_start + pd.to_timedelta(target_remain, 'd')
        else:
            dt_end = max(condition_df_out.dt.min(), dt_start)
        return dt_end

    @staticmethod
    def calc_days_remain(df, seq_columns, consume_col):
        """
        计算订单可用量
        :param df:
        :param seq_columns:按使用顺序排序订单列，如['po1_amount', 'po2_amount', 'po3_amount']、['beg_wh_stock', 'cg_zt', 'po_zt', 'pp_zt']
        :param consume_col:出库列列名
        :return:
        """
        # calc_days_remain(df_transit_m1_nation,['beg_wh_stock', 'cg_zt', 'po_zt', 'pp_zt'],'dly_sum')
        # calc_days_remain(df_transit_m1_nation,['po1_amount', 'po2_amount', 'po3_amount'],'dly_sum')
        df_out = df.copy()
        df_out['previous_remain'] = 0

        for col in seq_columns:
            df_out[f'{col}_remain'] = np.clip(
                df_out[col] - np.clip(df_out[consume_col] - df_out['previous_remain'], 0, np.inf), 0, np.inf)
            df_out['previous_remain'] = df_out['previous_remain'] + df_out[col]
        # df_out = df_out.drop(columns=['previous_remain'])
        return df_out


def format_convert_subnew_goods_simulation(df):
    output_cols = ['goods_id', 'material_type', 'stock_up_version', 'consumer_version', 'strategy_version_id',
                   'sim_version',
                   'stock_quantity', 'stock_days', 'cg_trs_quantity', 'cg_trs_days', 'po_quantity', 'po_days',
                   'replenish_quantity', 'replenish_days', 'total_days', 'w1', 'w2', 'w3', 'w4', 'stock_deadline',
                   'cg_trs_deadline', 'po_deadline', 'replenish_deadline']
    df = df[output_cols]
    data_type = {'goods_id': 'int',
                 'material_type': 'int',
                 'stock_up_version': 'int',
                 'consumer_version': 'int',
                 'strategy_version_id': 'int',
                 'sim_version': 'int',
                 'stock_quantity': 'float',
                 'stock_days': 'int',
                 'cg_trs_quantity': 'float',
                 'cg_trs_days': 'int',
                 'po_quantity': 'float',
                 'po_days': 'int',
                 'replenish_quantity': 'float',
                 'replenish_days': 'int',
                 'total_days': 'int',
                 'w1': 'float',
                 'w2': 'float',
                 'w3': 'float',
                 'w4': 'float',
                 'stock_deadline': 'str',
                 'cg_trs_deadline': 'str',
                 'po_deadline': 'str',
                 'replenish_deadline': 'str'}

    for key, value in data_type.items():
        df[key] = df[key].astype(value)

    return df


def format_convert_new_goods_simulation(df):
    """
    新品模拟落库前 强制类型转化
    Parameters
    ----------
    df

    Returns
    -------

    """
    output_cols = ['goods_id', 'material_type', 'stock_up_version', 'consumer_version', 'strategy_version_id',
                   'sim_version',
                   'first_delivery_quantity', 'first_delivery_days', 'goods_stock_up_quantity', 'goods_stock_up_days',
                   'raw_stock_quantity', 'raw_stock_days', 'total_days', 'w1', 'w2', 'w3', 'w4', 'first_deadline',
                   'goods_deadline', 'raw_deadline', 'plan_sale_duration', 'over_plan_quantity', 'is_expire_risk',
                   'estimated_expired_quantity']
    df = df[output_cols]
    data_type = {'goods_id': 'int',
                 'material_type': 'int',
                 'stock_up_version': 'int',
                 'consumer_version': 'int',
                 'strategy_version_id': 'int',
                 'sim_version': 'int',
                 'first_delivery_quantity': 'float',
                 'first_delivery_days': 'int',
                 'goods_stock_up_quantity': 'float',
                 'goods_stock_up_days': 'int',
                 'raw_stock_quantity': 'float',
                 'raw_stock_days': 'int',
                 'total_days': 'int',
                 'w1': 'float',
                 'w2': 'float',
                 'w3': 'float',
                 'w4': 'float',
                 'first_deadline': 'str',
                 'goods_deadline': 'str',
                 'raw_deadline': 'str',
                 'plan_sale_duration': 'int',
                 'over_plan_quantity': 'float',
                 'is_expire_risk': 'int',
                 'estimated_expired_quantity': 'float',
                 }

    for key, value in data_type.items():
        df[key] = df[key].astype(value)

    return df


def format_convert_avl_result(df):
    """
    有货率落库前 强制类型转化
    Parameters
    ----------
    df

    Returns
    -------

    """
    output_cols = ['predict_dt', 'goods_id', 'material_type', 'scene', 'stock_up_version', 'consumer_version',
                   'strategy_version_id', 'sim_version', 'shop_avl_rate', 'wh_avl_days', 'shop_avl_days',
                   'dly_use_ratio',
                   'pur_use_ratio', 'is_min_launch_day', 'shop_consume_amount', 'wh_consume_amount']
    df = df[output_cols]
    data_type = {'predict_dt': 'str',
                 'goods_id': 'int',
                 'material_type': 'int',
                 'scene': 'int',
                 'stock_up_version': 'int',
                 'consumer_version': 'int',
                 'strategy_version_id': 'int',
                 'sim_version': 'int',
                 'shop_avl_rate': 'float',
                 'wh_avl_days': 'int',
                 'shop_avl_days': 'int',
                 'dly_use_ratio': 'float',
                 'pur_use_ratio': 'float',
                 'is_min_launch_day': 'int',
                 'shop_consume_amount': 'float',
                 'wh_consume_amount': 'float'}

    for key, value in data_type.items():
        df[key] = df[key].astype(value)

    return df
