# encoding: utf-8 
""" 
@Project:workspace 
@Created: 2023/8/18 
@Author: cuiyuhan 
"""

import concurrent.futures
from utils.c52_hdfs_to_hive import mlclient_pandas_to_hive
from projects.newp_scene.p00_sim_base_method import *
from projects.newp_scene.p00_etl_method import *

"""

"""


class newSimulation(BaseSimMethod, NewEtl, CalcAvlDayRate):

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
        self.supplier_stocks = None
        self.goods_warranty_days = None
        self.batch_produce_date = None
        self.ro_after_launch = None

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

        # 其他数据初始化
        self.ld_receive_order_date = None
        self.df_future_vlt = None
        self.df_future_bp = None
        self.valid_plan_scope = None
        self.valid_shop_ratio = None
        self.ld_shop_order_strategy = None
        self.df_wh_cup_config = None
        self.df_params = None
        self.valid_formula = None
        self.true_formula = None
        self.bp_days = None
        self.df_order_chance = None
        self.df_cal_ro_base = None
        self.ld_dt_label = None
        self.df_goods_daily = None
        self.df_avl_d_his = None
        # 初始化数据
        self.new_sim_data_load()
        # 计算中间结果
        self.expire_day = None
        self.estimated_expired_quantity = None
        self.df_bp_vlt_consume_ro_fill_ratio = None
        self.df_stock_po_base = None
        self.df_po_concat = None
        self.df_avl_concat = None
        self.is_test = is_test

    def new_sim_data_load(self):
        # 定时表
        log.title(
            f"select_goods_id = {self.select_goods_id} | commodity_list = {self.commodity_list} | scene = {self.scene} | material_type = {self.material_type} |version_pool = {self.version_pool} ")

        log.title(f"【新品/小数仓】读取计划上市时间" + f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        self.df_com_launch_plan = self.read_com_launch_plan()

        # 新品日期节点变量
        self.max_launch_date = self.df_com_launch_plan['calc_launch_date'].max()
        self.min_launch_date = self.df_com_launch_plan['calc_launch_date'].min()
        self.min_pur_end_date = self.df_com_launch_plan['pur_end_date'].min()
        self.max_pur_end_date = self.df_com_launch_plan['pur_end_date'].max()
        self.min_cs_start = (pd.to_datetime(self.min_launch_date) - pd.to_timedelta(self.cold_start_dur, 'D'))
        self.launch_diff = (pd.to_datetime(self.max_launch_date) - pd.to_datetime(
            self.min_launch_date)) / pd.to_timedelta(1, 'D')
        self.new_type = 1 if self.max_launch_date == self.min_launch_date else 2 if self.launch_diff <= 11 else 3

        self.sim_start_date, self.sim_end_date = self.get_start_n_end()
        log.title(f"【新品】模拟开始时间{self.sim_start_date}" + f'模拟结束时间{self.sim_end_date}')

        log.title(f"【新品/calc】生成日期序列" + f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        self.ld_dt_label = self.get_ld_dt_label(self.sim_start_date, self.sim_end_date)

        log.title(
            f"【新品/小数仓】读取门店订货日历" + f'scene=={self.scene} ｜' + f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        if pd.to_datetime(self.pred_calc_day) > self.sim_start_date:
            # 完整模拟整个冷启动时间节点
            log.debug(f'冷启动已开始，取冷启动开始前数据版本{self.sim_start_date.strftime("%Y-%m-%d")}')
            self.ld_receive_order_date, self.df_future_bp, self.df_future_vlt = self.read_receive_order_date(
                self.sim_start_date.strftime("%Y-%m-%d"))
        else:
            log.debug(f'{self.pred_calc_day}')
            self.ld_receive_order_date, self.df_future_bp, self.df_future_vlt = self.read_receive_order_date(
                self.pred_calc_day)

        log.title(f"【新品/小数仓】读取上市计划范围" + f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        self.valid_plan_scope = self.read_launch_plan_scope()

        log.title(f"【新品/小数仓】读取商品门店分配比例" + f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        self.valid_shop_ratio = self.read_shop_ratio()

        log.title(f"【次新品/小数仓】读取实历史有货率" + f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        self.df_avl_d_his = self.read_avl_d_his()

        # 新品订货机会
        log.title(f"【新品/小数仓】冷启动bp天数" + f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        self.bp_days = self.get_bp_days()

        log.title(f"【新品/小数仓】读取冷启动订货机会" + f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        self.df_order_chance = self.get_order_chance()

        log.title(f"【新品/小数仓】读取ro计算基础数据" + f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        self.df_cal_ro_base = self.get_cal_ro_base()

        # 数仓实时表
        log.title(f"【新品/rt_dim】读取ro配置" + f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        self.ld_shop_order_strategy = self.load_order_strategy()

        log.title(f"【新品/rt_dim】读取商品-配置" + f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        self.wh_cup_config = self.load_cup_config()
        self.check_consume_data(self.wh_cup_config)

        log.title(f"【新品/rt_dim】读取实时配方" + f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        self.valid_formula = self.load_valid_formula()

        log.title(f"【新品/calc】计算门店-货物每日消耗量" + f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        self.df_goods_daily = self.get_all_version_shop_consume_new(self.wh_cup_config, self.valid_formula)

        log.title(f"【新品/rt_dim】读取计划系统仓库货物配置" + f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        self.df_params = self.load_params()

    def get_newp_ro(self, ld_shop_order_strategy):
        log.debug('load_newp_ro')
        df_ro_cal_goods = read_api.read_one_folder(bip3('new_scene_sim', f'df_ss_ro_{self.select_goods_id}'),
                                                   self.pred_calc_day)
        if df_ro_cal_goods is None or df_ro_cal_goods.empty:
            # 自动化门店ro
            ro_cfg_sel_col = ['version_id', 'commodity_id', 'first_week_cup_quantity_avg', 'target_value']

            df_ro_cal_com = self.df_cal_ro_base.merge(self.valid_formula).merge(ld_shop_order_strategy[ro_cfg_sel_col],
                                                                                how='left').merge(
                self.valid_shop_ratio[['commodity_id', 'plan_id', 'dept_id', 'dis_ratio', 'wh_dept_id']])
            # N *（BP+RO) = kN *（BP+1)
            df_ro_cal_com['first_week_consume'] = df_ro_cal_com.eval(
                'sku_average_number*first_week_cup_quantity_avg*dis_ratio')
            df_ro_cal_com['target_consume'] = df_ro_cal_com.eval('sku_average_number*target_value*dis_ratio')

            goods_ro_sel = ['dept_id', 'wh_dept_id', 'ct_bp_days', 'ct_last_order_day', 'ct_last_receive_day',
                            'goods_id',
                            'version_id']
            df_ro_cal_goods = df_ro_cal_com.groupby(goods_ro_sel, as_index=False).agg(
                {'first_week_consume': 'sum', 'target_consume': 'sum'})
            df_ro_cal_goods['ro'] = round(
                df_ro_cal_goods.eval("target_consume/first_week_consume*(ct_bp_days+1)-ct_bp_days"),
                2)
            df_ro_cal_goods['ro'] = np.clip(df_ro_cal_goods['ro'], 2, 100)

            c_path_save_df(df_ro_cal_goods, bip3('new_scene_sim', f'df_ss_ro_{self.select_goods_id}'),
                           self.pred_calc_day)
        return df_ro_cal_goods

    # =============模拟po/cg==============
    @staticmethod
    def get_new_bp_vlt_consume(df_order_calendar, df_order_chance, df_goods_daily, filter, alias):
        df_order_calendar['order_day'] = pd.to_datetime(df_order_calendar['order_day'])
        df_goods_daily['predict_date'] = pd.to_datetime(df_goods_daily['predict_date'])
        df_order_calendar['predict_date'] = pd.to_datetime(df_order_calendar['predict_date'])
        df_order_calendar['order_day'] = pd.to_datetime(df_order_calendar['order_day'])

        df_consume_base = df_order_calendar \
            .rename(columns={'order_day': 'dt'}) \
            .merge(df_order_chance[['dt', 'dept_id', 'order_cnt', 'can_order', 'min_launch_date']]) \
            .merge(df_goods_daily)
        df_consume_base.shop_com_consume = np.where(df_consume_base.eval(filter), 0, df_consume_base.shop_com_consume)

        agg_col = ['dt', 'goods_id', 'dept_id', 'version_id']
        df_res = df_consume_base.groupby(agg_col).shop_com_consume.sum().rename(alias).reset_index()

        return df_res

    def get_all_version_sim_base(self):
        log.debug('calc_all_version_sim_base')

        # bp/vlt consume
        df_bp_consume = self.get_new_bp_vlt_consume(self.df_future_bp, self.df_order_chance, self.df_goods_daily,
                                                    "can_order==0",
                                                    'bp_consume')
        df_vlt_consume = self.get_new_bp_vlt_consume(self.df_future_vlt, self.df_order_chance, self.df_goods_daily,
                                                     "predict_date<min_launch_date",
                                                     'vlt_consume')
        order_cnt_sel_col = ['dt', 'goods_id', 'min_launch_date', 'dept_id', 'receive_day', 'can_order',
                             'closed', 'order_cnt']
        self.ld_dt_label['dt'] = pd.to_datetime(self.ld_dt_label['dt'])
        df_bp_vlt_consume = self.ld_dt_label[['dt', 'dt_label', 'week_label', 'goods_id']].merge(
            self.df_goods_daily.rename(columns={'predict_date': 'dt'}), how='left') \
            .merge(self.df_order_chance[order_cnt_sel_col], how='left') \
            .merge(df_bp_consume, how='left') \
            .merge(df_vlt_consume, how='left') \
            .fillna({'bp_consume': 0, 'vlt_consume': 0})

        df_bp_vlt_consume.loc[df_bp_vlt_consume.order_cnt > 1, 'can_order'] = 0
        # 加入ro
        df_bp_vlt_consume_res = df_bp_vlt_consume.copy()
        df_bp_vlt_consume_res.loc[df_bp_vlt_consume_res.eval("dt<min_launch_date"), 'shop_com_consume'] = 0
        df_bp_vlt_consume_res = df_bp_vlt_consume_res.rename(columns={'shop_com_consume': 'pred_consume'})

        # 加入ro
        df_ro_cal_goods = self.get_newp_ro(self.ld_shop_order_strategy)

        ro_sel_col = ['dept_id', 'goods_id', 'version_id', 'ct_last_order_day', 'ro']
        ro_rename = {'ct_last_order_day': 'dt', 'ct_last_receive_day': 'receive_day'}
        df_bp_vlt_consume_ro = df_bp_vlt_consume_res.merge(self.bp_days, how='left') \
            .fillna({'bp_days': 1}) \
            .merge(df_ro_cal_goods[ro_sel_col].rename(columns=ro_rename), how='left')

        df_bp_vlt_consume_ro_fill = df_bp_vlt_consume_ro.copy()
        df_bp_vlt_consume_ro_fill = df_bp_vlt_consume_ro_fill.fillna({'ro': self.ro_after_launch})

        df_bp_vlt_consume_ro_fill['ss_ro'] = df_bp_vlt_consume_ro_fill.eval("bp_consume/bp_days*ro").fillna(0)

        # 加入箱规信息
        df_wh_spec = read_api.read_hour_minute_folder(c_path.control_tower.cfg_city_status + self.pred_calc_day) \
            .pipe(convert_df_columns)
        ld_purchase_ratio = (read_api.read_dt_folder(bip2('stock', 'dim_stock_spec_d_his'))[
            ['dly_use_ratio', 'spec_id', 'goods_id', 'pur_dly_ratio']]) \
            .query(f"goods_id=={self.select_goods_id}")
        ld_purchase_ratio[['pur_dly_ratio', 'dly_use_ratio']] = ld_purchase_ratio[
            ['pur_dly_ratio', 'dly_use_ratio']].astype(float)
        ld_purchase_ratio_status = df_wh_spec.merge(ld_purchase_ratio)

        # 采购且关注中不含select_goods_id 读配置
        if len(ld_purchase_ratio_status) == 0:
            df_pur_dly_ratio = self.df_params.loc[(self.df_params['wh_dept_id'] == -1),
                                                  ['goods_id', 'purchase_use_ratio',
                                                   'delivery_use_ratio']].drop_duplicates()
            df_pur_dly_ratio[['purchase_use_ratio', 'delivery_use_ratio']] = df_pur_dly_ratio[
                ['purchase_use_ratio', 'delivery_use_ratio']].astype(float)

            df_pur_dly_ratio['pur_dly_ratio'] = round(
                df_pur_dly_ratio['purchase_use_ratio'] / df_pur_dly_ratio['delivery_use_ratio'])
            df_pur_dly_ratio['dly_use_ratio'] = round(df_pur_dly_ratio['delivery_use_ratio'])
            df_pur_dly_ratio = df_pur_dly_ratio[['goods_id', 'pur_dly_ratio', 'dly_use_ratio']]

        # 采购且关注中含select_goods_id
        elif len(ld_purchase_ratio_status) > 0:
            # 采购且关注状态下的仓
            df_valid = ld_purchase_ratio_status.query("city_purchase_status==1").groupby(['goods_id', 'wh_dept_id'],
                                                                                         as_index=False) \
                .agg({'dly_use_ratio': 'max', 'pur_dly_ratio': 'max'}) \
                .rename(columns={'dly_use_ratio': 'valid_dly_use_ratio', 'pur_dly_ratio': 'valid_pur_dly_ratio'})
            # 兜底：没有采购且关注状态的仓用有采购且关注的仓对应最大值
            df_all = ld_purchase_ratio_status.groupby(['goods_id'], as_index=False) \
                .agg({'dly_use_ratio': 'max', 'pur_dly_ratio': 'max'})
            df_all = df_all.merge(df_wh_spec[['wh_dept_id']].drop_duplicates(), how='cross')
            df_pur_dly_ratio = df_all.merge(df_valid, how='left')
            df_pur_dly_ratio['dly_use_ratio'] = df_pur_dly_ratio['valid_dly_use_ratio'].fillna(
                df_pur_dly_ratio['dly_use_ratio'])
            df_pur_dly_ratio['pur_dly_ratio'] = df_pur_dly_ratio['valid_pur_dly_ratio'].fillna(
                df_pur_dly_ratio['pur_dly_ratio'])
            df_pur_dly_ratio = df_pur_dly_ratio[['wh_dept_id', 'goods_id', 'pur_dly_ratio', 'dly_use_ratio']]
        else:
            raise ValueError(f" No available dly_use_ratio/pur_dly_ratio for selected goods.")

        df_bp_vlt_consume_ro_fill_ratio = df_bp_vlt_consume_ro_fill.merge(df_pur_dly_ratio)

        df_bp_vlt_consume_ro_fill_ratio['shop_beg'] = 0

        df_bp_vlt_consume_ro_fill_ratio['expected_receive'] = 0
        df_bp_vlt_consume_ro_fill_ratio = df_bp_vlt_consume_ro_fill_ratio.query("~receive_day.isnull()")
        return df_bp_vlt_consume_ro_fill_ratio

    def get_df_stock_po_base(self, df_bp_vlt_consume_ro_fill_ratio):
        log.debug('calc_po_base_data')
        # po参数
        po_param = self.get_po_params()
        # cg参数
        cg_param = self.get_cg_params()
        # 中心仓调拨参数
        cdc_param = self.get_db_params()

        # 备货版本无短配出库模拟
        df_stock_version_base = df_bp_vlt_consume_ro_fill_ratio.query(f"version_id=={self.stock_up_version}")
        df_stock_result = self.run_newp_shop_wh_simulate_no_resign(df_stock_version_base,
                                                                   ['version_id', 'pur_dly_ratio', 'min_launch_date'])
        df_stock_wh = df_stock_result.groupby(
            ['version_id', 'goods_id', 'wh_dept_id', 'dt', 'min_launch_date', 'dly_use_ratio',
             'pur_dly_ratio']).order_num.sum().reset_index()

        # 加入仓库参数
        df_stock_wh_po = df_stock_wh.merge(po_param, how='left') \
            .fillna(0)

        # 加入基础信息
        df_stock_wh_po['supplier_stocks'] = self.supplier_stocks
        df_stock_wh_po['scene'] = self.scene
        df_stock_wh_po['new_type'] = self.new_type

        # ======================================新品备货规则==========================================
        # 新增物料对应商品（组）同步上市 【场景1/2】
        # ①首批到仓量=[ 计划上市时间, 计划上市时间 + 首批到仓周期 ] 周期汇总的备货版本杯量对应的货物出库量
        # ②成品备货量=( 计划上市时间 + 首批到仓周期 , 计划上市时间 + 成品备货周期 ] 周期汇总的备货版本杯量对应的货物出库量
        #
        # ③原料备货量=(计划上市时间 + 成品备货周期, 计划上市时间 + 原料备货周期 ] 周期汇总的货物出库量
        # 新增物料对应商品组异步上市（上市日期间隔<=11）【场景3】
        # ①首批到仓量=[ min(计划上市时间), max(计划上市时间)+首批到仓周期 ] 周期汇总的备货版本杯量对应的货物出库量
        # ②成品备货量=( max(计划上市时间)+首批到仓周期,  max(计划上市时间) + 成品备货周期 ] 周期汇总的备货版本杯量对应的货物出库量
        # ③原料备货量=( max(计划上市时间) + 成品备货周期, max(计划上市时间) + 原料备货周期 ] 周期汇总的货物出库量
        #
        # 新增物料对应商品组异步上市（上市日期间隔>11）【场景4】
        # ①首批到仓量=[ min(计划上市时间), min(计划上市时间)+首批到仓周期 ] 周期汇总的备货版本杯量对应的货物出库量
        # ②成品备货量=( min(计划上市时间)+首批到仓周期,  min(计划上市时间) + 成品备货周期 ] 周期汇总的备货版本杯量对应的货物出库量
        # ③原料备货量=( min(计划上市时间) + 成品备货周期, min(计划上市时间) + 原料备货周期 ] 周期汇总的货物出库量
        df_stock_wh_po['po1_start'] = self.max_launch_date if self.new_type == 2 else self.min_launch_date
        df_stock_wh_po['po1_end'] = df_stock_wh_po.apply(
            lambda x: pd.to_datetime(x.po1_start) + pd.to_timedelta(x.wh_bp + x.wh_lt, 'd'), axis=1)
        df_stock_wh_po['po2_end'] = df_stock_wh_po.apply(
            lambda x: pd.to_datetime(x.po1_end) + pd.to_timedelta(x.pt, 'd'), axis=1)
        df_stock_wh_po['po3_end'] = df_stock_wh_po.apply(
            lambda x: pd.to_datetime(x.po2_end) + pd.to_timedelta(x.mt, 'd'), axis=1)

        df_stock_wh_po['pur_use_ratio'] = df_stock_wh_po['dly_use_ratio'] * df_stock_wh_po['pur_dly_ratio']
        df_po1 = df_stock_wh_po.loc[df_stock_wh_po.eval("dt<=po1_end")]
        df_po2 = df_stock_wh_po.loc[df_stock_wh_po.eval("po1_end<dt<=po2_end")]
        df_po3 = df_stock_wh_po.loc[df_stock_wh_po.eval("po2_end<dt<=po3_end")]

        df_po1_agg = df_po1.groupby(
            ['version_id', 'goods_id', 'wh_dept_id', 'pur_use_ratio']).order_num.sum().rename(
            'po1_amount').reset_index()
        df_po2_agg = df_po2.groupby(
            ['version_id', 'goods_id', 'wh_dept_id', 'pur_use_ratio']).order_num.sum().rename(
            'po2_amount').reset_index()
        df_po3_agg = df_po3.groupby(
            ['version_id', 'goods_id', 'wh_dept_id', 'pur_use_ratio']).order_num.sum().rename(
            'po3_amount').reset_index()

        # 分配供应商库存
        df_stock_po_res = df_po1_agg.merge(df_po2_agg, how='left').merge(df_po3_agg, how='left').fillna(0)
        all_need = df_stock_po_res['po1_amount'].sum() + df_stock_po_res['po2_amount'].sum() + df_stock_po_res[
            'po3_amount'].sum()
        df_stock_po_res['dis_ratio'] = (df_stock_po_res['po1_amount'] + df_stock_po_res['po2_amount'] + df_stock_po_res[
            'po3_amount']) / all_need
        df_stock_po_res['dis_supplier_stock'] = round(self.supplier_stocks * df_stock_po_res['dis_ratio'], 0) * \
                                                df_stock_po_res['pur_use_ratio']
        df_stock_po_res['po3_amount'] = df_stock_po_res['po3_amount'] + df_stock_po_res['dis_supplier_stock']

        # 模拟基础数据
        df_stock_po_base = df_stock_po_res.merge(cg_param, how='left').fillna(0)
        if len(cdc_param) > 1:
            df_stock_po_base_out = df_stock_po_base.merge(cdc_param, how='left').fillna(0)
        else:

            df_stock_po_base_out = df_stock_po_base
            df_stock_po_base_out['wt'] = 0
        df_stock_po_base_out['beg_wh_stock'] = df_stock_po_base_out['po1_amount']
        return df_stock_po_base_out

    # 【输出】po信息表
    def get_po_amount_last_info(self, version_id, df_theory_order_base, df_theory_dly):
        log.title('po展示信息表')
        log.debug('po剩余量计算')
        nation_po = self.df_stock_po_base.groupby(['goods_id'], as_index=False).agg(
            {'po1_amount': 'sum', 'po2_amount': 'sum', 'po3_amount': 'sum', 'pur_use_ratio': 'max'})
        df_theory_nation = df_theory_dly.groupby(['dt', 'goods_id'], as_index=False).agg({'theory_order_sum': 'sum'})
        df_transit_m1_nation = df_theory_nation.merge(nation_po, on=['goods_id'])
        df_transit_m1_nation['dly_sum'] = df_transit_m1_nation['theory_order_sum'].cumsum()

        # 计算po剩余量
        use_seq = ['po1_amount', 'po2_amount', 'po3_amount']
        df_transit_m2_nation = CalcAvlDayRate.calc_days_remain(df_transit_m1_nation, use_seq, 'dly_sum')

        # 采购单位转换
        log.debug('po单位换算')

        df_res = nation_po.copy()
        df_res['first_delivery_quantity'] = np.ceil(df_res['po1_amount'] / df_res['pur_use_ratio'])
        df_res['goods_stock_up_quantity'] = np.ceil(df_res['po2_amount'] / df_res['pur_use_ratio'])
        df_res['raw_stock_quantity'] = np.ceil(df_res['po3_amount'] / df_res['pur_use_ratio'])

        # 截止日期
        log.debug('po截止日期计算')

        dt_end = df_transit_m2_nation.dt.max()
        dt_start = df_transit_m2_nation.dt.min()
        last_seven_dly = df_transit_m2_nation.sort_values('dt', ascending=1).query("theory_order_sum>0").iloc[
                         -7::].theory_order_sum.mean()

        po1_end = CalcAvlDayRate.calculate_end_date(df_transit_m2_nation, 'po1_amount_remain', last_seven_dly, dt_end)
        po2_start = max(po1_end, dt_end)
        po2_end = CalcAvlDayRate.calculate_end_date(df_transit_m2_nation, 'po2_amount_remain', last_seven_dly,
                                                    po2_start)
        po3_start = max(po2_end, dt_end)
        po3_end = CalcAvlDayRate.calculate_end_date(df_transit_m2_nation, 'po3_amount_remain', last_seven_dly,
                                                    po3_start)

        # 日期兜底
        if df_res.po2_amount.values[0] == 0:
            po2_end = max(po2_end, po1_end)
        if df_res.po3_amount.values[0] == 0:
            po3_end = max(po2_end, po3_end)

        df_res['first_deadline'] = po1_end
        df_res['goods_deadline'] = po2_end
        df_res['raw_deadline'] = po3_end

        # 售卖周期相关
        log.debug('售卖周期计算')
        df_res['over_plan_quantity'] = df_transit_m2_nation.loc[
            df_transit_m2_nation.dt == df_transit_m2_nation.dt.max(), ['po1_amount_remain', 'po2_amount_remain',
                                                                       'po3_amount_remain']].values.sum()
        df_res['over_plan_quantity'] = np.ceil(df_res['over_plan_quantity'] / df_res['pur_use_ratio'])
        df_res['plan_sale_duration'] = (pd.to_datetime(self.sim_end_date) - pd.to_datetime(
            self.min_launch_date) + pd.to_timedelta(1, 'd')) / pd.to_timedelta(
            1, 'd')

        df_res['first_delivery_days'] = (po1_end - dt_start) / pd.to_timedelta(1, 'd')
        df_res['goods_stock_up_days'] = (po2_end - po1_end) / pd.to_timedelta(1, 'd')
        df_res['raw_stock_days'] = (po3_end - po2_end) / pd.to_timedelta(1, 'd')
        df_res['total_days'] = (po3_end - dt_start) / pd.to_timedelta(1, 'd')

        # 过期数量计算
        log.debug('过期数量计算')
        self.expire_day = pd.to_datetime(self.batch_produce_date) + pd.to_timedelta(
            self.goods_warranty_days, 'd')
        # 过期截止日期在模拟周期内
        if self.expire_day <= dt_end:
            all_consume = df_transit_m2_nation.loc[df_transit_m2_nation.dt < self.expire_day, 'theory_order_sum'].sum()
            po_amount_all = df_res['po1_amount'].values[0] + df_res['po2_amount'].values[0] + \
                            df_res['po3_amount'].values[0]
            df_res['estimated_expired_quantity'] = po_amount_all - all_consume
        # 过期截止日期在模拟周期外
        elif (self.expire_day > dt_end) & (self.expire_day < po3_end):
            all_consume_p1 = df_transit_m2_nation.loc[
                df_transit_m2_nation.dt < self.expire_day, 'theory_order_sum'].sum()
            po_amount_all = df_res['po1_amount'].values[0] + df_res['po2_amount'].values[0] + \
                            df_res['po3_amount'].values[0]
            remain_days = (po3_end - self.expire_day) / pd.to_timedelta(1, 'd')
            all_consume_p2 = remain_days * last_seven_dly
            df_res['estimated_expired_quantity'] = po_amount_all - all_consume_p1 - all_consume_p2
        else:
            df_res['estimated_expired_quantity'] = 0
        df_res['estimated_expired_quantity'] = np.ceil(df_res['estimated_expired_quantity'] / df_res['pur_use_ratio'])
        df_res['is_expire_risk'] = False if df_res['estimated_expired_quantity'].values[0] == 0 else True

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
        df_res['scene'] = self.scene
        df_res['stock_up_version'] = self.stock_up_version
        df_res['consumer_version'] = version_id
        df_res['strategy_version_id'] = self.strategy_version_id
        df_res['sim_version'] = self.sim_version

        df_res = df_res[
            ['goods_id', 'material_type', 'stock_up_version', 'consumer_version', 'strategy_version_id', 'sim_version',
             'first_delivery_quantity', 'first_delivery_days', 'goods_stock_up_quantity', 'goods_stock_up_days',
             'raw_stock_quantity', 'raw_stock_days', 'total_days', 'w1', 'w2', 'w3', 'w4', 'first_deadline',
             'goods_deadline', 'raw_deadline', 'plan_sale_duration', 'over_plan_quantity',
             'is_expire_risk', 'estimated_expired_quantity']].copy()

        df_res['first_deadline'] = df_res['first_deadline'].dt.strftime('%Y-%m-%d')
        df_res['goods_deadline'] = df_res['goods_deadline'].dt.strftime('%Y-%m-%d')
        df_res['raw_deadline'] = df_res['raw_deadline'].dt.strftime('%Y-%m-%d')
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
        df_transit_m1['po_remain'] = df_transit_m1.po3_amount + df_transit_m1.po2_amount

        # 备货版本订单数据 ++ 消耗版本出库 订单截止日期信息
        df_po_info = self.get_po_amount_last_info(version_id, df_theory_order_base, df_theory_dly)

        # 模拟cg单
        df_transit_m2 = df_transit_m1.copy()
        df_transit_m2['transit'] = 0
        df_transit = self.get_cg_trans(df_transit_m2)

        # 含短配逻辑出库模拟
        df_res_shop, df_res_order, df_res_wh = self.run_newp_shop_wh_simulate(df_theory_order_base,
                                                                              self.df_stock_po_base, df_transit)
        # 备货版本订单数据 ++ 消耗版本出库 有货率信息
        df_avl_info = self.get_sim_avl_info(version_id, df_res_shop, df_res_order, df_res_wh)
        return df_po_info, df_avl_info

    def get_sim_res(self):
        log.debug('全流程模拟开始')
        # 模拟宽表准备
        self.df_bp_vlt_consume_ro_fill_ratio = self.get_all_version_sim_base()
        # 计算备货po
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
        self.df_po_concat = format_convert_new_goods_simulation(self.df_po_concat)

        self.df_avl_concat = pd.concat(df_avl_list)
        self.df_avl_concat = format_convert_avl_result(self.df_avl_concat)

        if self.is_test:
            c_path_save_df(self.df_avl_concat, bip3('new_scene_sim',
                                                    f'df_avl_days_{self.select_goods_id}_{self.strategy_version_id}_{self.sim_version}'),
                           self.pred_calc_day)
            c_path_save_df(self.df_po_concat, bip3('new_scene_sim',
                                                   f'df_po_days_{self.select_goods_id}_{self.strategy_version_id}_{self.sim_version}'),
                           self.pred_calc_day)
        else:
            mlclient_pandas_to_hive(self.df_po_concat, "alg_control_tower_newp_nation_goods_stock_simulation_result",
                                    dt=self.pred_calc_day, dt_partition=True)
            mlclient_pandas_to_hive(self.df_avl_concat, "alg_control_tower_newp_nation_goods_avl_result",
                                    dt=self.pred_calc_day,
                                    dt_partition=True)
