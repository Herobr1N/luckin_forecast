import datetime
import numpy as np
import pandas as pd

from config.config import *
from utils.decorator import *
from utils.msg_utils import *
from dim_info.dim_base import Dim

pd.set_option('mode.chained_assignment', None)


class wh_arrival_monitor:
    def __init__(self) -> None:
        self.wh_info = Dim.dim_warehouse()
        self.wh_relation = Dim.dim_shop_city_warehouse_relation()
        self.valid_goods = Dim.dim_goods()
        self.get_formula = Dim.get_formula(is_agg=True)
        self.dim_valid_cmdty = Dim.dim_valid_cmdty()
        self.formula_agg = self.get_formula.merge(self.dim_valid_cmdty['cmdty_id'], on=['cmdty_id'], how='inner')
        self.dim_goods_spec = Dim.dim_goods_spec()
        self.new_product_push = spark.sql("""
                                    SELECT
                                        plan_id
                                        ,plan_name
                                        , wh_dept_id
                                        , cmdty_id
                                        , goods_id
                                        , DATE_FORMAT(update_time,'yyyy-MM-dd') AS update_time
                                        , push_status
                                        , pp_order AS spp_no
                                    FROM
                                        lucky_dataplatform.`t_commodity_purchase_strategy_push`
                                    WHERE (wh_dept_id IS NOT NULL)
                                        AND wh_dept_id NOT IN (8801, 170904, 328161, 337814)

                                    """)
        self.plan_info = spark.sql("""
                                     SELECT DISTINCT
                                        plan_id
                                        , plan_name
                                    FROM dw_dim.dim_cmdty_launch_plan_d_his plan
                                    WHERE dt = DATE_SUB(CURRENT_DATE(),1)""")
        self.days_ahead_launch = {
            'new_cg': 15,
            'new_db_qs': 8,
            'new_db_yl': 12,
            'old_goods': 15,
            'mrkt_goods': 8
        }

    def get_valid_cmdty_goods(self):
        """
        上市计划对应相关商品及货物
        :return: DataFrame
        """
        # 上市时间在三个月之内的计划
        valid_cmdty_plan_shop = spark.sql(f"""
        SELECT
            dt
          , launch_plan_id
          , cmdty_id
          , dept_id
          , MIN(launch_date) as launch_date
          , MIN(actual_launch_date) as actual_launch_date
        FROM dw_ads_scm_alg.`dm_purchase_cmdty_plan_time` --未来三个月内上市的新品
        WHERE (launch_date>='{today}'
           OR actual_launch_date>='{today}')
          AND (launch_date<='{today + timedelta(days=90)}'
           OR actual_launch_date<='{today + timedelta(days=90)}')
          AND (launch_plan_id is not null)
          AND dt = '{yesterday}'
        GROUP BY
            dt, launch_plan_id, cmdty_id, dept_id
    """)
        valid_cmdty_plan_shop.actual_launch_date = pd.to_datetime(valid_cmdty_plan_shop.actual_launch_date)
        valid_cmdty_plan_shop.launch_date = pd.to_datetime(valid_cmdty_plan_shop.launch_date)
        # 转仓维度
        shop_wh = self.wh_relation[['wh_dept_id', 'shop_dept_id']].drop_duplicates()
        valid_cmdty_plan_shop_wh = valid_cmdty_plan_shop.rename(columns={'dept_id': 'shop_dept_id'}).merge(shop_wh, on=['shop_dept_id'], how='left')
        valid_cmdty_plan_wh = valid_cmdty_plan_shop_wh.groupby(['launch_plan_id',
                                                                'cmdty_id',
                                                                'wh_dept_id'], as_index=False).agg({'launch_date': 'min', 'actual_launch_date': 'min'})
        # 判断实际上市时间与计划上市时间是否对应
        valid_cmdty_plan_wh.actual_launch_date = valid_cmdty_plan_wh.actual_launch_date.fillna(valid_cmdty_plan_wh.launch_date)
        valid_cmdty_plan_wh['same_plan_check'] = [True if np.ptp(valid_cmdty_plan_wh.loc[valid_cmdty_plan_wh.index == i, ['launch_date', 'actual_launch_date']].values)\
                                                              .astype('timedelta64[D]') < timedelta(days=30) else False for i in range(len(valid_cmdty_plan_wh))]
        valid_cmdty_plan_wh.loc[valid_cmdty_plan_wh['same_plan_check']==True, 'launch_date'] = valid_cmdty_plan_wh['actual_launch_date']
        valid_cmdty_plan_wh.loc[valid_cmdty_plan_wh['same_plan_check']==False, 'launch_date'] = valid_cmdty_plan_wh['launch_date']
        valid_cmdty_plan_wh = valid_cmdty_plan_wh[['launch_plan_id', 'cmdty_id', 'wh_dept_id', 'launch_date']]
        # 关联配方
        new_product_wh_goods = valid_cmdty_plan_wh.merge(self.formula_agg, on=['cmdty_id'], how='left').rename(columns={'launch_plan_id': 'plan_id'})
        # 降级--兜底配方中缺少某个商品的情况
        new_product_wh_goods = new_product_wh_goods[~new_product_wh_goods.goods_id.isnull()]
        new_product_wh_goods = new_product_wh_goods[~((new_product_wh_goods.wh_dept_id == 329235) & (new_product_wh_goods.goods_id.isin([7352, 7359])))]
        return new_product_wh_goods

    def split_single_use_goods(self):
        """
        区分已推送新品PP中的独用物料与共用物料
        :return:
        """
        new_product_wh_goods = self.get_valid_cmdty_goods()
        goods_cover_cmdty = self.formula_agg.merge(new_product_wh_goods[['cmdty_id', 'goods_id', 'plan_id']].drop_duplicates(), on=['cmdty_id', 'goods_id'], how='left').drop_duplicates()
        # 区分新老商品
        goods_cover_cmdty['old_product_flg'] = 0
        goods_cover_cmdty.loc[goods_cover_cmdty.plan_id.isnull(), 'old_product_flg'] = 1
        # 物料覆盖的老货物:包含实验货物
        old_cmdty_cover_cnt = goods_cover_cmdty[['cmdty_id', 'goods_id', 'old_product_flg']] \
            .drop_duplicates().groupby(['goods_id'], as_index=False) \
            .agg({'old_product_flg': 'sum'}).query('old_product_flg>0')
        # 解决goods_id有空值问题
        new_product_push = self.new_product_push
        new_product_push.goods_id = new_product_push.goods_id.fillna(-1).astype('int')
        push_status = new_product_wh_goods \
            .merge(new_product_push[['plan_id', 'plan_name', 'wh_dept_id', 'goods_id', 'push_status', 'update_time', 'spp_no']], on=['plan_id', 'wh_dept_id', 'goods_id'], how='left')
        push_status['new_goods'] = push_status['goods_id'].apply(lambda x: 0 if x in (old_cmdty_cover_cnt['goods_id'].to_list()) else 1)
        push_status = push_status.rename(columns={'update_time': 'pp_push_time'})
        # 新货物未推送部分
        new_goods_unpushed = push_status.query("push_status ==0 and new_goods == 1")
        # 新货物已推送部分
        new_goods_pushed = push_status.query("push_status==2 and new_goods==1")
        # 旧货物部分
        old_goods = push_status.query("new_goods==0")
        # 去除在其他商品中推送的货物
        new_goods_unpushed = new_goods_unpushed[~new_goods_unpushed.goods_id.isin(new_goods_pushed['goods_id'].to_list())]
        return new_goods_pushed, new_goods_unpushed, old_goods

    def get_mrkt_plan_goods(self):
        """
        获取有效营销计划及相关物料
        :return:
        """
        marketing_plan = spark.sql(f"""
            SELECT
              plan.plan_name
              , plan.id
              , plan.whole_country
              , IFNULL(date_format(plan.material_plan_launch_date,'yyyy-MM-dd'), date_format(plan.material_plan_start_use_date,'yyyy-MM-dd')) AS material_plan_launch_date
              , date_format(plan.material_plan_stop_use_date,'yyyy-MM-dd') AS material_plan_stop_use_date
              , pc.city_id
              , ps.goods_spec_id
              , spec.goods_id
              , spec.goods_name
              , spec.spec_name
              , spec.dly_use_ratio
              , spec.pur_dly_ratio
              , spec.large_class_name
            from
              lucky_commodity.`t_lucky_commodity_marketing_plan` AS plan
              LEFT JOIN lucky_commodity.`t_lucky_commodity_marketing_plan_city` AS pc ON pc.plan_id = plan.id
              LEFT JOIN lucky_commodity.`t_lucky_commodity_marketing_plan_goods_spec` AS ps ON ps.plan_id = plan.id
              LEFT JOIN dw_dim.dim_stock_spec_d_his AS spec ON spec.spec_id = ps.goods_spec_id
              AND spec.dt = '{yesterday}'
            WHERE
              (
                to_date(plan.material_plan_stop_use_date) > '{today}'
                OR plan.material_plan_stop_use_date IS NULL
              )
              AND (
                material_plan_start_use_date IS NULL
                OR to_date(plan.material_plan_start_use_date) > '{today}'
               )
              AND plan.plan_name NOT LIKE '实验%'
        """)
        wh_info = self.wh_info
        city_wh = self.wh_relation[['city_id', 'wh_dept_id']].drop_duplicates()

        # 区域性营销计划
        regional_plan = marketing_plan.query("whole_country == 0")
        # 全国性营销计划
        whole_country_plan = marketing_plan.query("whole_country == 1")

        # 关联仓库
        regional_plan_wh = regional_plan.merge(city_wh, on='city_id', how='left').groupby(['id', 'plan_name', 'goods_id', 'wh_dept_id', 'goods_spec_id', 'dly_use_ratio', 'pur_dly_ratio', 'goods_name', 'spec_name', 'large_class_name'], as_index=False).agg({'material_plan_launch_date': 'min', 'material_plan_stop_use_date': 'max'})
        whole_country_plan_wh = whole_country_plan.merge(wh_info['wh_dept_id'], how='cross').groupby(['id', 'plan_name', 'goods_id', 'wh_dept_id', 'goods_spec_id', 'dly_use_ratio', 'pur_dly_ratio', 'goods_name', 'spec_name', 'large_class_name'], as_index=False).agg({'material_plan_launch_date': 'min', 'material_plan_stop_use_date': 'max'})

        marketing_plan_wh = pd.concat([regional_plan_wh, whole_country_plan_wh]).drop_duplicates()

        return marketing_plan_wh

    @staticmethod
    def minus_x_month(date, x, start=True, strOrNot=True):
        """
        在计算x个月前的月初和月末
        :param date:
        :param x:x<0则向后推x个月 x>0向前推x个月
        :param start:是否输出月初
        :param strOrNot:是否输出str type 否则输出datetime.date
        :return:
        """
        if type(date) == str:
            date = datetime.strptime(date, '%Y-%m-%d')
        adj_days = (365.25 / 12 - 28) / 4  # 月和四周天数差异调整量
        if start:
            res = date - timedelta(weeks=x * 4 + np.ceil(adj_days * x))
            res = res - timedelta(days=res.day - 1)
        else:
            x = x - 1
            res = date - timedelta(weeks=x * 4 + np.ceil(adj_days * x))
            res = res - timedelta(days=res.day)
        if strOrNot:
            res = res.strftime('%Y-%m-%d')
        return res

    def get_suggest_arrival_time(self):
        """
        获取建议入仓日期
        推送新品新物料pp/cg未推送情况报警
        :return:
        """
        three_month_ago = self.minus_x_month(today, 3, start=True, strOrNot=True)[0:7]
        # 2022-04-19之后采购与调拨拆分

        cg_db_push_d_his = spark.sql(f"""
            (SELECT
                dt,
                    wh_dept_id,
                    wh_name,
                    large_class_name,
                    goods_id,
                    -- 存储类型 1 常温 2 冷冻 3 冷藏
                    storage_type,
                    spec_id,
                    purchase_type,
                    plan_finish_date,
                    supplier_no,
                    supplier_name,
                    DATE_FORMAT(update_time,'yyyy-MM-dd') AS cg_push_time,
                    order_no as scc_no
            FROM lucky_dataplatform.`t_dm_auto_cg_result`
            WHERE push_status = 2 AND dt >= '{three_month_ago}')
            UNION
            (SELECT
                dt,
                    wh_dept_id,
                    wh_name,
                    large_class_name,
                    goods_id,
                    -- 存储类型 1 常温 2 冷冻 3 冷藏
                    storage_type,
                    spec_id,
                    purchase_type,
                    plan_finish_date,
                    supplier_no,
                    supplier_name,
                    DATE_FORMAT(update_time,'yyyy-MM-dd') AS cg_push_time,
                    order_no as scc_no
            FROM lucky_dataplatform.`t_dm_auto_transfer_result`
            WHERE push_status = 2 AND dt >= '{three_month_ago}')
        """)
        cdc_config = spark.sql(f"""
                SELECT
                    wh_dept_id,
                    goods_id,
                    is_cdc,
                    is_cdc_model
                FROM dw_ads_scm_alg.dim_automatic_order_wh_goods_cfg
                WHERE dt  = '{today}'
            """)

        scc_cc_cg = spark.sql("""
            SELECT DISTINCT
                po_document_no
              , cc_no
              , scc_no
              , cg_order_no AS cg_no
              , cg_status
              , wh_dept_id
              , po_goods_id AS goods_id
              , po_spec_id AS spec_id
              , cg_total_count AS cg_plan_pur_cnt
              , cg_receive_count AS cg_receive_cnt
              , cg_remain_count AS cg_remain_cnt
              , cg_difference_count AS cg_diff_cnt
              , po_supplier_no AS supplier_no
              , spp_no
              , DATE_FORMAT(cg_create_time,'yyyy-MM-dd') AS cg_create_time
              , cg_plan_finish_time AS plan_finish_date
            FROM dw_ads_scm_alg.dws_pp_po_cc_cg_fh_chain
        """)

        new_goods_pushed, new_goods_unpushed_pp, old_goods = self.split_single_use_goods()

        cdc_config = cdc_config.rename(columns={'is_cdc': 'is_cdc_cfg', 'is_cdc_model': 'is_cdc_model_cfg'})
        # cg推送部分
        new_goods_cg = new_goods_pushed.merge(scc_cc_cg, on=['goods_id', 'wh_dept_id', 'spp_no'], how='left') \
            .merge(cdc_config, on=['goods_id', 'wh_dept_id'], how='left') \
            .merge(self.wh_info[['wh_dept_id', 'wh_name']].drop_duplicates(), on=['wh_dept_id'], how='left') \
            .merge(self.valid_goods[['goods_id', 'large_class_name']].drop_duplicates(), on=['goods_id'], how='left') \
            .query("cg_create_time>=pp_push_time")
        # 取首批cg单
        new_goods_cg['cg_rnk'] = new_goods_cg \
            .groupby(['goods_id', 'wh_dept_id', 'spp_no']) \
            .cg_create_time.rank(method='dense')
        new_goods_cg = new_goods_cg.query("cg_rnk==1")
        # 未推送cg部分--推送告警
        new_goods_unpushed_cg = new_goods_cg[new_goods_cg.cg_no.isnull()]
        # 已推送cg部分
        new_goods_pushed_cg = new_goods_cg[~new_goods_cg.cg_no.isnull()]
        # 调拨部分
        new_goods_db = new_goods_pushed.merge(cg_db_push_d_his, on=['goods_id', 'wh_dept_id'], how='left') \
            .merge(cdc_config, on=['goods_id', 'wh_dept_id'], how='left') \
            .query("(is_cdc_cfg==0 and is_cdc_model_cfg==1)")

        # 未推送db部分
        new_goods_unpushed_db = new_goods_db[~new_goods_db.scc_no.isnull()]
        # 报警：pp/cg/db未发起
        new_goods_unpushed_cg['缺失类型'] = '新品PP已推送，CG未推送'
        new_goods_unpushed_db['缺失类型'] = '新品PP已推送，城市仓调拨未发起'
        new_goods_unpushed_pp['缺失类型'] = '新品PP未推送'
        new_goods_unpushed_pp['pp_push_time'] = None
        # 独用物料cg/db未推送部分 前推13天
        date_limit = np.datetime64(today + timedelta(days=13))
        unpush_col = ['plan_id', 'plan_name', 'cmdty_id', 'wh_dept_id', 'launch_date', 'goods_id', 'goods_name', 'pp_push_time', '缺失类型']
        df_unpush = pd.concat([new_goods_unpushed_cg.loc[new_goods_unpushed_cg['launch_date'] < date_limit, unpush_col],
                               new_goods_unpushed_db.loc[new_goods_unpushed_db['launch_date'] < date_limit, unpush_col],
                               new_goods_unpushed_pp[unpush_col]])

        df_unpush = df_unpush.merge(self.dim_valid_cmdty[['cmdty_id', 'cmdty_name']], on=['cmdty_id']).merge(self.wh_info[['wh_dept_id', 'wh_name']], on=['wh_dept_id']) \
            .rename(columns={'plan_name': '计划名称', 'cmdty_name': '商品名称', 'wh_name': '仓库名称', 'goods_name': '货物名称', 'pp_push_time': '推送pp时间', 'launch_date': '上市时间'})
        df_unpush = df_unpush[['货物名称', '商品名称', '上市时间', '仓库名称', '缺失类型', '推送pp时间']].sort_values(['货物名称', '仓库名称', '缺失类型'])
        # df_unpush['负责人'] = '唐玉亮'
        if len(df_unpush) > 0:
            Message.send_file(df_unpush, f'{yesterday}_新品推送情况监控.csv', group=MSG_GROUP_THREE)

        # 构造建议到仓时间
        # 新品上市部分
        # 兜底cg计划完成时间
        new_goods_pushed_cg['suggest_arrive_time'] = new_goods_pushed_cg['launch_date'] - timedelta(days=self.days_ahead_launch.get('new_cg'))
        new_goods_pushed_cg['suggest_arrive_time'] = new_goods_pushed_cg['plan_finish_date'].fillna(new_goods_pushed_cg['suggest_arrive_time'])
        # 新品独用物料建议到仓时间
        new_goods_pushed_cg.suggest_arrive_time = new_goods_pushed_cg.suggest_arrive_time.astype('datetime64')
        # 确定监控范围
        new_goods_pushed = new_goods_pushed_cg[(new_goods_pushed_cg.suggest_arrive_time <= np.datetime64(today)) & (new_goods_pushed_cg.launch_date > np.datetime64(today))]

        # 新品共用物料建议到仓时间
        old_goods = old_goods \
            .merge(cdc_config, on=['goods_id', 'wh_dept_id'], how='left') \
            .merge(self.valid_goods[['goods_id', 'large_class_name']], on='goods_id', how='left')

        old_goods['suggest_arrive_time'] = old_goods['launch_date'] - timedelta(days=self.days_ahead_launch.get('old_goods'))
        old_goods.loc[(old_goods.is_cdc_cfg == 0) & (old_goods.is_cdc_model_cfg == 1) & (old_goods.large_class_name == '轻食'), 'suggest_arrive_time'] = old_goods['launch_date'] - timedelta(days=self.days_ahead_launch.get('new_db_qs'))
        old_goods.loc[(old_goods.is_cdc_cfg == 0) & (old_goods.is_cdc_model_cfg == 1) & (old_goods.large_class_name == '原料'), 'suggest_arrive_time'] = old_goods['launch_date'] - timedelta(days=self.days_ahead_launch.get('new_db_yl'))
        old_goods.suggest_arrive_time = old_goods.suggest_arrive_time.astype('datetime64')
        # 确定监控范围
        old_goods = old_goods[(old_goods.suggest_arrive_time <= np.datetime64(today)) & (old_goods.launch_date > np.datetime64(today))]

        # 营销活动部分
        marketing_plan_wh = self.get_mrkt_plan_goods()
        # 营销活动物料建议到仓时间
        marketing_plan_goods_arrive_time = marketing_plan_wh
        marketing_plan_goods_arrive_time['material_plan_launch_date'] = marketing_plan_goods_arrive_time['material_plan_launch_date'].astype('datetime64')
        marketing_plan_goods_arrive_time['suggest_arrive_time'] = marketing_plan_goods_arrive_time['material_plan_launch_date'] - timedelta(days=self.days_ahead_launch.get('mrkt_goods'))
        # 营销活动物料对应监控范围
        marketing_plan_goods_monitor_scope = marketing_plan_goods_arrive_time[(marketing_plan_goods_arrive_time.suggest_arrive_time <= np.datetime64(today)) & (marketing_plan_goods_arrive_time.material_plan_launch_date > np.datetime64(today))]
        marketing_plan_goods_monitor_scope = marketing_plan_goods_monitor_scope.rename(columns={'goods_spec_id': 'spec_id'})
        marketing_plan_goods_push = marketing_plan_goods_monitor_scope.merge(scc_cc_cg, on=['spec_id', 'goods_id', 'wh_dept_id'], how='left')
        return new_goods_pushed, old_goods, marketing_plan_goods_push

    def get_contact(self, df):
        """
        匹配缺失配置项负责人
        :param moq_missing:
        :return: dataframe
        """
        contact_emp = spark.sql("""
        SELECT
        -- 供应商商品信息表
            name
            , contact_emp_name
        FROM lucky_srm.`t_supplier_commodity_info` srm_cmdty_info
        INNER JOIN
        -- 供应商信息表
            (
                SELECT
                    id
                    ,supplier_code
                    ,supplier_name
                    ,enterprise_id
                FROM lucky_srm.`t_supplier_info`
                WHERE org_code RLIKE 0101
                    AND mdm_supplier_status = 1
                    AND cooperation_status = 0) supplier_info
            ON srm_cmdty_info.enterprise_id = supplier_info.enterprise_id

        """)
        contact_emp = contact_emp.rename(columns={'name': 'spec_name'})
        spec_contact = df.merge(contact_emp, on=['spec_name'], how='left')
        # 兜底
        spec_contact['contact_emp_name'] = spec_contact['contact_emp_name'].fillna('李瑞')
        return spec_contact

    def newp_old_goods_break_stock(self, old_goods):
        wh_stock = spark.sql(f"""
            SELECT
                dt
                , wh_dept_id
                , goods_id
                , SUM(begin_wh_stock_avl_cnt) AS begin_avl_cnt
                , SUM(end_wh_stock_avl_cnt) AS end_avl_cnt
            FROM dw_dws.dws_stock_warehouse_stock_adjust_begin_end_d_inc_summary
            WHERE dt = '{yesterday}'
                AND wh_dept_id in {tuple(self.wh_info['wh_dept_id'])}
                AND end_wh_stock_avl_cnt>=0
            GROUP BY dt, wh_dept_id, goods_id
        """)
        # 根据分仓平均配方筛选货物：去除不使用pla仓库的pla货物断仓提醒
        avg_formula = spark.sql("""
            SELECT DISTINCT
                plan_id
                , goods_id
                , wh_dept_id
                , wh_adjust_percent
            FROM
            -- 新品分仓平均配方表
                dw_ads_scm_alg.dim_commodity_wh_average_formula
            WHERE
                dt = CURRENT_DATE
                AND wh_dept_id IS NOT NULL
                AND adjusted_avg_config > 0
                AND plan_id > 2437  -- 切换平均配方开始计划ID
        """)

        # 上市计划物料监控
        goods_arrival_time_monitor_res = old_goods \
            .merge(avg_formula, on=['goods_id', 'wh_dept_id', 'plan_id'], how='inner') \
            .merge(wh_stock, on=['wh_dept_id', 'goods_id'], how='left')

        # 库存库存不足部分
        goods_arrival_time_monitor_res = goods_arrival_time_monitor_res.loc[~(goods_arrival_time_monitor_res.end_avl_cnt > 0)]
        # 负责人
        if len(goods_arrival_time_monitor_res) > 0:
            goods_arrival_time_monitor_res['contact_emp_name'] = '供应链规划'

            goods_arrival_time_monitor_res = goods_arrival_time_monitor_res \
                .merge(self.dim_valid_cmdty[['cmdty_id', 'cmdty_name']], on=['cmdty_id']) \
                .merge(self.wh_info[['wh_dept_id', 'wh_name']], on=['wh_dept_id'])
            goods_arrival_time_monitor_res['待办项'] = '及时补充库存'
            goods_arrival_time_monitor_res['监控类型'] = '新品上市老货物监控'
            goods_arrival_time_monitor_res = goods_arrival_time_monitor_res.rename(columns={'cmdty_name': '商品名称', 'wh_name': '未入仓仓库', 'goods_name': '货物名称', 'launch_date': '上市时间', 'contact_emp_name': '负责人'})
            goods_arrival_time_monitor_res = goods_arrival_time_monitor_res[['监控类型', '货物名称', '商品名称', '上市时间', '未入仓仓库', '负责人', '待办项']]
            Message.send_file(goods_arrival_time_monitor_res, f'{yesterday}_新品上市老货物断仓部分.csv', group=MSG_GROUP_THREE)

    @staticmethod
    def agg_by_plan_goods(df, agg_key):
        i = 0
        res = pd.DataFrame(columns=['计划名称', '上市范围', '上市日期', '货物大类', '货物名称', '供应商情况', '首批计划到货量(采购单位)', '首批实际到货量(采购单位)', '入仓率', '未到货仓库', '是否有延期风险'])

        for [k1, k2, k3, k4, k5], group in df.groupby(agg_key):
            supplier_num = '单一供应商'
            if len(group['supplier_no'].drop_duplicates().tolist()) > 1:
                supplier_num = '多供应商'
            cg_plan_arr = group['cg_plan_pur_cnt'].sum()
            cg_receive = group['cg_receive_cnt'].sum()
            arr_rate = cg_receive / cg_plan_arr
            # 未入仓部分超过计划10%则认为有风险
            un_arr_wh = ' ,'.join(group.query("cg_plan_pur_cnt>0 and cg_receive_cnt<0.97*cg_plan_pur_cnt").wh_name.drop_duplicates().tolist())
            delayed = '是' if len(un_arr_wh) > 0 else '否'
            res.loc[i] = [k1, k2, k3, k4, k5] + [supplier_num, cg_plan_arr, cg_receive, f'{round(arr_rate * 100, 2)}%', un_arr_wh, delayed]

            i = i + 1
        return res

    @log_wecom('新品入仓进度监控', P_TWO)
    def get_stock(self):
        """
        获取新品设计物料库存情况及推送报警
        :return:
        """

        # 监控范围
        new_goods, old_goods, marketing_plan_goods = self.get_suggest_arrival_time()
        old_goods = old_goods[['wh_dept_id', 'cmdty_id', 'plan_id', 'plan_name', 'goods_id', 'goods_name', 'new_goods', 'launch_date', 'suggest_arrive_time']].drop_duplicates()
        # 新品对应老货物--监控推送
        self.newp_old_goods_break_stock(old_goods)
        # 有效cg
        marketing_plan_goods = marketing_plan_goods.loc[~marketing_plan_goods.cg_no.isnull()]

        # 新品对应新货物
        if len(new_goods) > 0:
            # 上市计划对应城市范围
            city_plan = spark.sql(f"""
           SELECT cmdty.plan_id,
                whole_country,
                (CASE WHEN cmdty.whole_country = 1 THEN '全国' ELSE CONCAT_WS(',', collect_set(city.city_name)) END) AS launch_city
            FROM (SELECT dt, plan_id, whole_country FROM dw_dim.`dim_cmdty_launch_plan_d_his` WHERE dt = '{yesterday}') cmdty
                 LEFT JOIN dw_dim.`dim_cmdty_launch_plan_city_d_his` city
                           ON city.plan_id = cmdty.plan_id AND city.dt = cmdty.dt

            GROUP BY cmdty.plan_id, whole_country
                """)
            # 加入上市计划城市范围
            new_goods = new_goods.merge(city_plan[['plan_id', 'launch_city']], on=['plan_id'], how='left')
            new_goods_res = self.agg_by_plan_goods(new_goods, ['plan_name', 'launch_city', 'launch_date', 'large_class_name', 'goods_name'])
            new_goods_res['监控类型'] = '新品首采货物入仓监控'

        else:
            new_goods_res = pd.DataFrame()

        if len(marketing_plan_goods) > 0:
            marketing_plan_goods = marketing_plan_goods.merge(self.wh_info[['wh_dept_id', 'wh_name']], on='wh_dept_id', how='left')

            marketing_plan_city = spark.sql(f"""
            SELECT
                plan.id
              , plan.whole_country
              , (CASE WHEN plan.whole_country = 1 THEN '全国' ELSE CONCAT_WS(',', collect_set(pc.city_name)) END) AS launch_city

            FROM lucky_commodity.`t_lucky_commodity_marketing_plan` AS plan
            LEFT JOIN lucky_commodity.`t_lucky_commodity_marketing_plan_city` AS pc ON pc.plan_id = plan.id

            WHERE (
                        to_date(plan.material_plan_stop_use_date) > '{today}'
                    OR plan.material_plan_stop_use_date IS NULL
                )
              AND (
                    material_plan_start_use_date IS NULL
                    OR to_date(plan.material_plan_start_use_date) > '{today}'
                )
              AND plan.plan_name NOT LIKE '实验%'
            GROUP BY
                plan.id, whole_country""")
            # 加入上市计划城市范围
            marketing_plan_goods = marketing_plan_goods.merge(marketing_plan_city[['id', 'launch_city']], on=['id'], how='left')
            marketing_plan_res = self.agg_by_plan_goods(marketing_plan_goods, ['plan_name', 'launch_city', 'material_plan_launch_date', 'large_class_name', 'goods_name'])
            marketing_plan_res['监控类型'] = '营销活动货物入仓监控'
        else:
            marketing_plan_res = pd.DataFrame()
        res = pd.concat([new_goods_res, marketing_plan_res])
        if len(res) > 0:
            res['日期'] = yesterday
            res['距离上市日期天数'] = res['上市日期'] - pd.to_datetime(res['日期'])
            res['距离上市日期天数'] = res['距离上市日期天数'].astype('str').str[0:2]
            res = res[['日期', '监控类型', '货物大类', '货物名称', '供应商情况', '计划名称', '上市范围', '首批计划到货量(采购单位)', '首批实际到货量(采购单位)', '入仓率', '未到货仓库', '是否有延期风险', '上市日期', '距离上市日期天数']]
            res['备注'] = ''
            Message.send_file(res, f'{yesterday}_新品入仓监控.csv', group=MSG_GROUP_THREE)
            if len(res.loc[res['是否有延期风险'] == '是']) > 0:
                Message.send_msg('请及时关注未入仓货物', group=MSG_GROUP_THREE)
            else:
                Message.send_msg('新品相关首采货物均已正常入仓', group=MSG_GROUP_THREE)
        else:
            Message.send_msg('暂无新品相关首采货物', group=MSG_GROUP_THREE)


if __name__ == '__main__':
    wh_arrival_monitor().get_stock()
