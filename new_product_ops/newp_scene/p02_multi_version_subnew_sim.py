# encoding: utf-8 
""" 
@Project:workspace 
@Created: 2023/9/7 
@Author: cuiyuhan 
"""

from areas.t0_data.parse_dm_config import ro_setting_load
from utils.c52_hdfs_to_hive import mlclient_pandas_to_hive
import concurrent.futures
from projects.newp_scene.p00_sim_base_method import *
from projects.newp_scene.p00_etl_method import *
import logging

logger = logging.getLogger("alg_dh")


class SubSimVersion(BaseSimMethod, NewEtl, CalcAvlDayRate):
    def __init__(self, sub_variables, pred_calculation_day=None, is_test=False):
        NewEtl.__init__(self, pred_calculation_day)

        self.material_type = None
        self.select_goods_id = None
        self.scene = None
        self.stock_up_version = None
        self.consumer_version = None
        self.strategy_version_id = None
        self.sim_version = None
        self.commodity_list = []
        # 接受模拟参数
        log.debug(sub_variables)
        for key, value in sub_variables.items():
            setattr(self, key, value)

        self.select_goods_id = sub_variables["goods_id"]
        if isinstance(self.consumer_version, int):
            self.consumer_version = [self.consumer_version]
        self.version_pool = [self.stock_up_version] + self.consumer_version
        # 计划信息
        self.df_com_launch_plan = None
        self.max_launch_date = None
        self.min_launch_date = None
        self.min_pur_end_date = None
        self.max_pur_end_date = None
        self.min_cs_start = None
        self.launch_diff = None
        self.new_type = None
        self.sim_start_date = None
        self.sim_end_date = None

        #  其他数据初始化
        self.ld_receive_order_date = None
        self.df_future_vlt = None
        self.df_future_bp = None
        self.valid_shop_ratio = None
        self.wh_cup_config = None
        self.df_goods_daily = None
        self.df_params = None
        self.valid_formula = None
        self.true_formula = None
        self.ld_dt_label = None
        self.df_rt_dly = None
        self.df_rt_zt_valid_sel = None
        self.df_wh_beg = None
        self.df_avl_d_his = None
        # 初始化数据
        self.subnew_sim_data_load()
        # 计算中间结果
        self.df_bp_vlt_consume_ro_fill_ratio = None
        self.df_stock_po_base = None
        self.df_po_concat = None
        self.df_avl_concat = None
        self.is_test = is_test

    def subnew_sim_data_load(self):
        """
        次新品模拟基础数据读取
        :return:
        """
        log.title(
            f"select_goods_id = {self.select_goods_id} | commodity_list = {self.commodity_list} | scene = {self.scene} | material_type = {self.material_type} |version_pool = {self.version_pool} ")

        log.title(f"【次新品/小数仓】读取计划上市时间" + f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        self.df_com_launch_plan = self.read_com_launch_plan()

        self.max_launch_date = self.df_com_launch_plan['calc_launch_date'].max()
        self.min_launch_date = self.df_com_launch_plan['calc_launch_date'].min()
        self.min_pur_end_date = self.df_com_launch_plan['pur_end_date'].min()
        self.max_pur_end_date = self.df_com_launch_plan['pur_end_date'].max()
        self.min_cs_start = (pd.to_datetime(self.min_launch_date) - pd.to_timedelta(self.cold_start_dur, 'D'))
        self.launch_diff = (pd.to_datetime(self.max_launch_date) - pd.to_datetime(
            self.min_launch_date)) / pd.to_timedelta(1, 'D')
        self.new_type = 1 if self.max_launch_date == self.min_launch_date else 2 if self.launch_diff <= 11 else 3
        self.sim_start_date, self.sim_end_date = self.get_start_n_end()
        log.title(f"【次新品】模拟开始时间{self.sim_start_date}" + f'模拟结束时间{self.sim_end_date}')

        log.title(f"【次新品/calc】生成日期序列" + f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        self.ld_dt_label = self.get_ld_dt_label(self.sim_start_date, self.sim_end_date)
        # 小数仓数据
        log.title(
            f"【次新品/小数仓】读取门店订货日历" + f'scene=={self.scene} ｜' + f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        self.ld_receive_order_date, self.df_future_bp, self.df_future_vlt = self.read_receive_order_date(
            self.pred_calc_day)

        log.title(f"【次新品/小数仓】读取商品门店分配比例" + f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        self.valid_shop_ratio = self.read_shop_ratio()

        log.title(f"【次新品/小数仓】读取实历史有货率" + f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        self.df_avl_d_his = self.read_avl_d_his()

        log.title(f"【新品/rt_dim】读取商品-配置" + f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        self.wh_cup_config = self.load_cup_config()
        self.check_consume_data(self.wh_cup_config)

        log.title(f"【新品/rt_dim】读取实时配方" + f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        self.valid_formula = self.load_true_formula()

        log.title(f"【次新品/calc】计算门店-货物每日消耗量" + f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        self.df_goods_daily = self.get_all_version_shop_consume_new(self.wh_cup_config, self.valid_formula)

        log.title(f"【次新品/rt_dim】读取计划系统仓库货物配置" + f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        self.df_params = self.load_params()

        log.title(f"【次新品/准实时小数仓】读取实时配货数据" + f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        self.df_rt_dly = self.read_rt_dly()

        log.title(f"【次新品/准实时小数仓】读取实时仓库在途数据" + f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        self.df_rt_zt_valid_sel = self.read_rt_zt_valid_sel()

        log.title(f"【次新品/准实时小数仓】读取实时仓库库存数据" + f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        self.df_wh_beg = self.read_wh_beg()

    def ro_ss_load(self):
        # Try to load the result from the file
        df_ss_ro = read_api.read_one_folder(bip3('subnew_scene_sim', f'df_ss_ro_{self.select_goods_id}'),
                                            self.pred_calc_day)
        if df_ss_ro is None or df_ss_ro.empty:
            # If loading fails, run the original code and save the result
            # 读取dm配置
            print('ro loading ...')
            # ro
            df_shop_dly_cnt = read_api.read_dt_folder(bip1('operation', 'week_dly_day_cnt'), self.pred_minus1_day)
            ld_ro = ro_setting_load()
            ld_ro['ro'] = ld_ro['ro'].astype(float)
            df_ro = (ld_ro
                     .rename(columns={"dly_freq": "dly_day_cnt"})
                     [['goods_id', 'dly_day_cnt', 'ro']]
                     .drop_duplicates()
                     )
            df_ro = df_shop_dly_cnt.merge(df_ro)[['dept_id', 'goods_id', 'ro']]

            # 读取ss
            print('ss loading ...')
            df_shop_goods_ss = read_api.read_dt_folder(c_path.model.shop_goods_drink_food_safety_stock,
                                                       self.pred_calc_day)

            df_ss_qty = df_shop_goods_ss[['dept_id', 'goods_id', 'ss']]

            # 整合ss、ro
            df_ss_ro = df_ss_qty.merge(df_ro, how='outer').fillna(0)

            # Save the result to a file
            c_path_save_df(df_ss_ro, bip3('subnew_scene_sim', f'df_ss_ro_{self.select_goods_id}'),
                           self.pred_calc_day)

        return df_ss_ro

    @staticmethod
    def get_bp_vlt_consume(df_order_calendar, df_goods_daily, alias):
        df_order_calendar['order_day'] = pd.to_datetime(df_order_calendar['order_day'])
        df_goods_daily['predict_date'] = pd.to_datetime(df_goods_daily['predict_date'])
        df_order_calendar['predict_date'] = pd.to_datetime(df_order_calendar['predict_date'])
        df_order_calendar['order_day'] = pd.to_datetime(df_order_calendar['order_day'])

        df_consume_base = df_order_calendar \
            .rename(columns={'order_day': 'dt'}) \
            .merge(df_goods_daily)

        agg_col = ['dt', 'goods_id', 'dept_id', 'version_id']
        df_res = df_consume_base.groupby(agg_col).shop_com_consume.sum().rename(alias).reset_index()

        return df_res

    def get_all_version_sim_base(self):

        df_bp_consume = self.get_bp_vlt_consume(self.df_future_bp, self.df_goods_daily, 'bp_consume')
        df_vlt_consume = self.get_bp_vlt_consume(self.df_future_vlt, self.df_goods_daily, 'vlt_consume')

        df_order = self.ld_receive_order_date[["dept_id", "receive_day", "order_day", "can_order", "closed"]].rename(
            columns={'order_day': 'dt'})
        df_order.dt = pd.to_datetime(df_order.dt)

        df_bp_vlt_consume = self.ld_dt_label[['dt', 'dt_label', 'week_label', 'goods_id']] \
            .merge(self.df_goods_daily.rename(columns={'predict_date': 'dt'})) \
            .merge(df_order) \
            .merge(df_bp_consume, how='left') \
            .merge(df_vlt_consume, how='left') \
            .fillna({'shop_com_consume': 0, 'bp_consume': 0, 'vlt_consume': 0})

        # 加入ro
        df_bp_vlt_consume_res = df_bp_vlt_consume.copy()
        df_bp_vlt_consume_res = df_bp_vlt_consume_res.rename(columns={'shop_com_consume': 'pred_consume'})

        # # 加入ro
        df_norm_ss_ro = self.ro_ss_load()

        ro_sel_col = ['dept_id', 'goods_id', 'ss', 'ro']
        df_bp_vlt_consume_ro = df_bp_vlt_consume_res.merge(df_norm_ss_ro[ro_sel_col], how='left')

        df_bp_vlt_consume_ro['ss_ro'] = df_bp_vlt_consume_ro['ss'] + df_bp_vlt_consume_ro['ro'] * df_bp_vlt_consume_ro[
            'pred_consume']

        # 加入箱规信息
        df_wh_spec = read_api.read_hour_minute_folder(c_path.control_tower.cfg_city_status + self.pred_calc_day) \
            .pipe(convert_df_columns)

        ld_purchase_ratio = (read_api.read_dt_folder(bip2('stock', 'dim_stock_spec_d_his'))
        [['dly_use_ratio', 'goods_id', 'pur_dly_ratio']])
        # ld_purchase_ratio['moq'] = ld_purchase_ratio['dly_use_ratio']
        ld_purchase_ratio[['dly_use_ratio', 'pur_dly_ratio']] = ld_purchase_ratio[
            ['dly_use_ratio', 'pur_dly_ratio']].astype('float')
        df_pur_dly_ratio = df_wh_spec.merge(ld_purchase_ratio).groupby(['goods_id', 'wh_dept_id'], as_index=False) \
            .agg({'dly_use_ratio': 'max', 'pur_dly_ratio': 'max'})

        df_bp_vlt_consume_ro_ratio = df_bp_vlt_consume_ro.merge(df_pur_dly_ratio)

        df_shop_stock = \
            read_api.read_dt_folder(bip2('process', 'dws_stock_shop_goods_actual_consume'),
                                    self.pred_minus1_day).rename(
                columns={"end_shop_stock_cnt": "shop_beg"})[['goods_id', 'dept_id', 'shop_beg']]

        df_bp_vlt_consume_ro_ratio_begin = df_bp_vlt_consume_ro_ratio.merge(df_shop_stock, how='left').fillna(0)

        df_bp_vlt_consume_ro_ratio_begin_receive = df_bp_vlt_consume_ro_ratio_begin.merge(self.df_rt_dly,
                                                                                          how='left').fillna(0)
        df_bp_vlt_consume_ro_ratio_begin_receive.receive_day = pd.to_datetime(
            df_bp_vlt_consume_ro_ratio_begin_receive.receive_day)
        df_bp_vlt_consume_ro_ratio_begin_receive.dt = pd.to_datetime(df_bp_vlt_consume_ro_ratio_begin_receive.dt)

        return df_bp_vlt_consume_ro_ratio_begin_receive

    def get_df_stock_po_base(self, df_bp_vlt_consume_ro_fill_ratio):

        log.debug('calc_po_base_data')
        # po参数
        po_param = self.get_subnew_po_params()
        # cg参数
        cg_param = self.get_cg_params()
        # 中心仓调拨参数
        cdc_param = self.get_db_params()

        # 备货版本出库模拟
        df_stock_version_base = df_bp_vlt_consume_ro_fill_ratio.query(f"version_id=={self.stock_up_version}")

        df_stock_result = self.run_newp_shop_wh_simulate_no_resign(df_stock_version_base,
                                                                   ['version_id', 'pur_dly_ratio'])
        df_stock_wh = df_stock_result.groupby(
            ['version_id', 'goods_id', 'wh_dept_id', 'dt', 'dly_use_ratio',
             'pur_dly_ratio']).order_num.sum().reset_index()

        df_stock_wh_po = df_stock_wh.merge(po_param, how='left').fillna(0)

        df_stock_wh_po['sim_pp_start'] = self.pred_calc_day
        df_stock_wh_po['sim_pp_end'] = df_stock_wh_po.apply(
            lambda x: pd.to_datetime(x.sim_pp_start) + pd.to_timedelta(x.vlt_po + x.second_bp_po + x.ss_po, 'd'),
            axis=1)
        df_pp = df_stock_wh_po.loc[df_stock_wh_po.eval("sim_pp_start<dt<=sim_pp_end")]

        df_sim_pp_agg = df_pp.groupby(
            ['version_id', 'goods_id', 'wh_dept_id', 'dly_use_ratio', 'pur_dly_ratio']).order_num.sum().rename(
            'pp_zt').reset_index()
        df_sim_pp_agg['pur_use_ratio'] = df_sim_pp_agg['dly_use_ratio'] * df_sim_pp_agg['pur_dly_ratio']
        # 加入po在途
        df_sim_pp_agg['po_dist_ratio'] = df_sim_pp_agg['pp_zt'] / df_sim_pp_agg['pp_zt'].sum()

        po_nation_zt = self.df_rt_zt_valid_sel.query(f"order_type=='po'").total_count.sum()
        df_sim_pp_agg['po_zt'] = round(po_nation_zt * df_sim_pp_agg['po_dist_ratio'], 0)

        df_stock_po_base_beg = df_sim_pp_agg.merge(self.df_wh_beg, how='left').fillna(0)

        # 加入cg在途
        cg = self.df_rt_zt_valid_sel.query(f"order_type.isin(['cg','trs','fh'])") \
            .groupby(['goods_id', 'wh_dept_id']).total_count.sum().rename('cg_zt').reset_index()

        if len(cg) > 0:
            df_stock_po_base_res = df_stock_po_base_beg.merge(cg, how='left').fillna(0)
        else:
            df_stock_po_base_res = df_stock_po_base_beg
            df_stock_po_base_res['cg_zt'] = 0
        # 需求量-在途-库存 取箱规
        df_stock_po_base_res['pp_zt'] = df_stock_po_base_res['pp_zt'] - df_stock_po_base_res['po_zt'] - \
                                        df_stock_po_base_res['cg_zt'] - df_stock_po_base_res['beg_wh_stock']
        df_stock_po_base_res['pp_zt'] = np.clip(df_stock_po_base_res['pp_zt'], 0, np.inf)
        df_stock_po_base_res['pp_zt'] = np.ceil(df_stock_po_base_res['pp_zt'] / df_stock_po_base_res['pur_use_ratio']) * \
                                        df_stock_po_base_res['pur_use_ratio']
        # 加入cg参数
        df_stock_po_base_res_para = df_stock_po_base_res.merge(cg_param, how='left').fillna(0)

        if len(cdc_param) > 1:
            df_stock_po_base_out = df_stock_po_base_res_para.merge(cdc_param, how='left').fillna(0)
        else:
            df_stock_po_base_out = df_stock_po_base_res_para
            df_stock_po_base_out['wt'] = 0
        output_col = ['version_id', 'goods_id', 'wh_dept_id', 'pp_zt', 'pur_use_ratio',
                      'po_dist_ratio', 'po_zt', 'beg_wh_stock', 'cg_zt', 'bp_cg', 'ss_cg', 'vlt_cg', 'wt']

        return df_stock_po_base_out[output_col]

    def get_po_amount_last_info(self, version_id, df_theory_order_base, df_theory_dly):
        log.debug('po剩余量计算')
        nation_po = self.df_stock_po_base.groupby(['goods_id'], as_index=False).agg(
            {'beg_wh_stock': 'sum', 'cg_zt': 'sum', 'po_zt': 'sum', 'pp_zt': 'sum', 'pur_use_ratio': 'max'})
        df_theory_nation = df_theory_dly.groupby(['dt', 'goods_id'], as_index=False).agg({'theory_order_sum': 'sum'})
        df_transit_m1_nation = df_theory_nation.merge(nation_po, on=['goods_id'])
        df_transit_m1_nation['dly_sum'] = df_transit_m1_nation['theory_order_sum'].cumsum()

        # 计算po剩余量
        use_seq = ['beg_wh_stock', 'cg_zt', 'po_zt', 'pp_zt']
        df_transit_m2_nation = CalcAvlDayRate.calc_days_remain(df_transit_m1_nation, use_seq, 'dly_sum')

        # 采购单位转换
        log.debug('po单位换算')

        df_res = nation_po.copy()
        df_res['stock_quantity'] = np.ceil(df_res['beg_wh_stock'] / df_res['pur_use_ratio'])
        df_res['cg_trs_quantity'] = np.ceil(df_res['cg_zt'] / df_res['pur_use_ratio'])
        df_res['po_quantity'] = np.ceil(df_res['po_zt'] / df_res['pur_use_ratio'])
        df_res['replenish_quantity'] = np.ceil(df_res['pp_zt'] / df_res['pur_use_ratio'])

        # 截止日期
        log.debug('po截止日期计算')

        dt_end = df_transit_m2_nation.dt.max()
        dt_start = df_transit_m2_nation.dt.min()
        last_seven_dly = df_transit_m2_nation.sort_values('dt', ascending=1).query("theory_order_sum>0").iloc[
                         -7::].theory_order_sum.mean()

        stock_end = CalcAvlDayRate.calculate_end_date(df_transit_m2_nation, 'beg_wh_stock_remain', last_seven_dly,
                                                      dt_end)
        cg_start = max(stock_end, dt_end)
        cg_end = CalcAvlDayRate.calculate_end_date(df_transit_m2_nation, 'cg_zt_remain', last_seven_dly, cg_start)
        po_start = max(cg_end, dt_end)
        po_end = CalcAvlDayRate.calculate_end_date(df_transit_m2_nation, 'po_zt_remain', last_seven_dly, po_start)
        pp_start = max(po_end, dt_end)
        pp_end = CalcAvlDayRate.calculate_end_date(df_transit_m2_nation, 'pp_zt_remain', last_seven_dly, pp_start)

        if df_res.cg_zt.values[0] == 0:
            cg_end = max(stock_end, cg_end)
        if df_res.po_zt.values[0] == 0:
            po_end = max(cg_end, po_end)
        if df_res.pp_zt.values[0] == 0:
            pp_end = max(pp_end, po_end)

        df_res['stock_deadline'] = stock_end
        df_res['cg_trs_deadline'] = cg_end
        df_res['po_deadline'] = po_end
        df_res['replenish_deadline'] = pp_end

        df_res['stock_days'] = (stock_end - dt_start) / pd.to_timedelta(1, 'd')
        df_res['cg_trs_days'] = (cg_end - stock_end) / pd.to_timedelta(1, 'd')
        df_res['po_days'] = (po_end - cg_end) / pd.to_timedelta(1, 'd')
        df_res['replenish_days'] = (pp_end - po_end) / pd.to_timedelta(1, 'd')
        df_res['total_days'] = (pp_end - dt_start) / pd.to_timedelta(1, 'd')

        # 杯量计算
        log.debug('杯量计算')
        avg_cup_num = df_theory_order_base.groupby(['week_label', 'goods_id'], as_index=False).agg(
            {'pred_consume': 'mean'})
        avg_needs_num = self.valid_formula.sku_average_number.mean()
        avg_cup_num['avg_needs_num'] = avg_needs_num
        avg_cup_num['cup'] = avg_cup_num.pred_consume / avg_cup_num.avg_needs_num
        df_res['w1'] = round(avg_cup_num.loc[avg_cup_num.week_label == 'W1', 'cup'].values[0], 2)
        df_res['w2'] = round(avg_cup_num.loc[avg_cup_num.week_label == 'W2', 'cup'].values[0], 2)
        df_res['w3'] = round(avg_cup_num.loc[avg_cup_num.week_label == 'W3', 'cup'].values[0], 2)
        df_res['w4'] = round(avg_cup_num.loc[avg_cup_num.week_label == 'W4', 'cup'].values[0], 2)

        # 加入基础信息
        log.debug('模拟基础信息加入')
        df_res['material_type'] = self.material_type
        df_res['stock_up_version'] = self.stock_up_version
        df_res['consumer_version'] = version_id
        df_res['strategy_version_id'] = self.strategy_version_id
        df_res['sim_version'] = self.sim_version

        df_res = df_res[
            ['goods_id', 'material_type', 'stock_up_version', 'consumer_version', 'strategy_version_id',
             'sim_version',
             'stock_quantity', 'stock_days', 'cg_trs_quantity', 'cg_trs_days', 'po_quantity', 'po_days',
             'replenish_quantity', 'replenish_days', 'total_days', 'w1', 'w2', 'w3', 'w4', 'stock_deadline',
             'cg_trs_deadline', 'po_deadline', 'replenish_deadline']]

        # 当需求量均为0时，截止日期与可用天数兜底
        last_days_columns = ['stock_days', 'cg_trs_days', 'po_days', 'replenish_days', 'total_days']
        df_res[last_days_columns] = df_res[last_days_columns].fillna(0)

        columns_date = ['stock_deadline', 'cg_trs_deadline', 'po_deadline', 'replenish_deadline']
        df_res[columns_date] = df_res[columns_date].fillna(DayStr.n_day_delta(None, 365))

        df_res['stock_deadline'] = df_res['stock_deadline'].dt.strftime('%Y-%m-%d')
        df_res['po_deadline'] = df_res['po_deadline'].dt.strftime('%Y-%m-%d')
        df_res['replenish_deadline'] = df_res['replenish_deadline'].dt.strftime('%Y-%m-%d')

        return df_res

    def sim_all_version(self, version_id):
        log.debug(f'{version_id}模拟开始')
        # 消耗版本需求数据
        df_theory_order_base = self.df_bp_vlt_consume_ro_fill_ratio.query(f"version_id=={version_id}")
        # 消耗版本出库数据
        df_consumer_theory_order = self.run_newp_shop_wh_simulate_no_resign(df_theory_order_base, ['version_id'])
        df_theory_dly = df_consumer_theory_order.groupby(
            ['dt', 'version_id', 'goods_id', 'wh_dept_id']).order_num.sum().rename('theory_order_sum').reset_index()

        # 备货版本订单数据 ++ 消耗版本出库数据
        df_transit_m1 = self.df_stock_po_base.merge(df_theory_dly, on=['wh_dept_id', 'goods_id'],
                                                    suffixes=['_stock', '_consumer'])
        df_transit_m1['lt_cg'] = df_transit_m1['wt'] + df_transit_m1['vlt_cg']
        df_transit_m1['po_remain'] = df_transit_m1.po_zt

        # 备货版本订单数据 ++ 消耗版本出库 订单截止日期信息
        df_po_info = self.get_po_amount_last_info(version_id, df_theory_order_base, df_theory_dly)
        # 实际cg在途
        df_trans_real = \
            self.df_rt_zt_valid_sel.query(f"order_type!='po'").groupby(['wh_dept_id', 'goods_id', 'plan_finish_date'],
                                                                       as_index=False).agg(
                {'total_count': 'sum'}).rename(columns={'plan_finish_date': 'dt', 'total_count': 'transit'})[
                ['dt', 'wh_dept_id', 'goods_id', 'transit']]

        df_trans_real['dt'] = pd.to_datetime(df_trans_real['dt'])
        df_transit_m2 = df_transit_m1.merge(df_trans_real, how='left').fillna({'transit': 0})

        # 模拟cg在途
        df_transit_cal = self.get_cg_trans(df_transit_m2)
        # 整合实际在途 ++ 模拟在途
        df_transit_all = pd.concat([df_trans_real, df_transit_cal]).groupby(['dt', 'wh_dept_id', 'goods_id']).sum(
            'transit').reset_index()
        if len(df_transit_cal) == 0:
            df_transit_all = pd.DataFrame(columns=['dt', 'wh_dept_id', 'goods_id', 'transit'])

        # 备货版本订单数据 ++ 消耗版本出库 有货率信息
        df_res_shop, df_res_order, df_res_wh = self.run_newp_shop_wh_simulate(df_theory_order_base,
                                                                              self.df_stock_po_base,
                                                                              df_transit_all)
        df_avl_info = self.get_sim_avl_info(version_id, df_res_shop, df_res_order, df_res_wh)
        return df_po_info, df_avl_info

    def get_subnew_sim_res(self):

        log.debug("次新品模拟计算")
        self.df_bp_vlt_consume_ro_fill_ratio = self.get_all_version_sim_base()
        self.df_bp_vlt_consume_ro_fill_ratio.receive_day = pd.to_datetime(
            self.df_bp_vlt_consume_ro_fill_ratio.receive_day)
        self.df_bp_vlt_consume_ro_fill_ratio.dt = pd.to_datetime(self.df_bp_vlt_consume_ro_fill_ratio.dt)

        self.df_stock_po_base = self.get_df_stock_po_base(self.df_bp_vlt_consume_ro_fill_ratio)

        df_sim_res = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.sim_all_version, consume_v) for consume_v in
                       self.consumer_version + [self.stock_up_version]]
            for future in concurrent.futures.as_completed(futures):
                df_sim_res.append(future.result())

        df_po_list = [df[0] for df in df_sim_res]
        df_avl_list = [df[1] for df in df_sim_res]

        self.df_po_concat = pd.concat(df_po_list)
        self.df_po_concat = format_convert_subnew_goods_simulation(self.df_po_concat)

        self.df_avl_concat = pd.concat(df_avl_list)
        self.df_avl_concat = format_convert_avl_result(self.df_avl_concat)

        mlclient_pandas_to_hive(self.df_po_concat,
                                "alg_control_tower_subnewp_nation_goods_stock_simulation_result",
                                dt=self.pred_calc_day, dt_partition=True)
        if self.is_test:
            c_path_save_df(self.df_po_concat, bip3('subnew_scene_sim',
                                                   f'df_po_last_{self.select_goods_id}_{self.strategy_version_id}_{self.sim_version}'),
                           self.pred_calc_day)
            c_path_save_df(self.df_avl_concat, bip3('subnew_scene_sim',
                                                    f'df_avl_days_{self.select_goods_id}_{self.strategy_version_id}_{self.sim_version}'),
                           self.pred_calc_day)
        else:
            mlclient_pandas_to_hive(self.df_po_concat,
                                    "alg_control_tower_subnewp_nation_goods_stock_simulation_result",
                                    dt=self.pred_calc_day, dt_partition=True)
            mlclient_pandas_to_hive(self.df_avl_concat, "alg_control_tower_newp_nation_goods_avl_result",
                                    dt=self.pred_calc_day,
                                    dt_partition=True)
