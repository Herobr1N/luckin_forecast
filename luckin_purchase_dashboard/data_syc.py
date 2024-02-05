from utils import logger, spark, datetime
import pandas as pd


def load(date=datetime.today().date()):
    global dt, goods_list, wh_list, df_forecast_all, df_new_subnew, df_his_cost_detail, df_sea, df_plan, df_replace, df_plan_detail, df_demand_detail, df_inv, auth_info, dim_goods, dim_cmdty, dim_wh
    dt = date
    DEMAND_FORECAST_DAILY = f'/user/haoxuan.zou/demand_predict/{date}/demand_forecast_daily/'
    DEMAND_ADJUST = f'/user/haoxuan.zou/demand_predict/{date}/goods_adjust_weight/'

    # 需求预测总结果
    df_forecast_all = spark.read_parquet(DEMAND_FORECAST_DAILY + 'demand_all').query("goods_id!=0 and wh_dept_id!=0")
    logger.info("Load 需求预测总结果 Finish")
    # 新品&次新品
    df_new_subnew = spark.read_parquet(DEMAND_FORECAST_DAILY + 'new_cmdty_demand_detail').query("goods_id!=0 and wh_dept_id!=0")
    logger.info("Load 新品&次新品 Finish")
    df_sea = spark.read_parquet(DEMAND_ADJUST + 'sea_factor')
    logger.info("Load 季节因子 Finish")
    # 商品上下市计划
    df_plan = spark.read_parquet(DEMAND_ADJUST + 'cmdty_plan_factor')
    logger.info("Load 商品上下市计划 Finish")
    # 商品上下市计划详情
    df_plan_detail = spark.read_parquet(DEMAND_ADJUST + 'cmdty_plan_detail')
    logger.info("Load 商品上下市计划详情 Finish")
    # 节假日因子、开店计划、货物替换后
    if date <= datetime(year=2022, day=16, month=7).date():
        df_replace = spark.read_csv(f'/user/haoxuan.zou/demand_predict/{date}/goods_replace/replace_result')
        df_replace = df_replace.rename(columns = {'wh_id':'wh_dept_id'})
    else:
        df_replace = spark.read_parquet(f'/user/haoxuan.zou/demand_predict/{date}/goods_replace/replace_result')
    logger.info("Load 节假日因子、开店计划、货物替换后 Finish")
    # 各货物大类预测详情
    if date <= datetime(year=2022, day=16, month=7).date():
        cols_cost = ['dt', 'wh_id', 'goods_id', 'origin_dmd_cost', 'origin_dmd_dly', 'dmd_daily_base', 'dmd_daily_shop_inc', 'dmd_daily']
        qs_dmd = spark.read_csv(DEMAND_FORECAST_DAILY + "demand_qs/")[cols_cost]
        yl_dmd = spark.read_csv(DEMAND_FORECAST_DAILY + "demand_yl/")[cols_cost]
        ls_dmd = spark.read_csv(DEMAND_FORECAST_DAILY + "demand_ls/")[cols_cost]
        cols_cost.extend(['dmd_new_shop_daily'])
        bc_dmd_p1 = spark.read_csv(DEMAND_FORECAST_DAILY + "demand_bc_dmd_p1/")
        bc_dmd_p2 = spark.read_csv(DEMAND_FORECAST_DAILY + "demand_bc_dmd_p2/").query("~goods_id.isna()")
        bc_dmd_mid = bc_dmd_p1.merge(bc_dmd_p2, on=["dt", "year", "month", "wh_id", "goods_id"], how="outer").fillna(0)
        bc_dmd_mid['dmd_daily'] = bc_dmd_mid.dmd_daily_regular.astype('float') + bc_dmd_mid.dmd_new_shop_daily.astype('float')
        bc_dmd = bc_dmd_mid[cols_cost]

        df_demand_detail = pd.concat([qs_dmd, yl_dmd, ls_dmd, bc_dmd]).rename(columns={'wh_id':'wh_dept_id'}).fillna(0)
        df_demand_detail['dmd_daily_festival'] = df_demand_detail['dmd_daily']
    else:

        cols_cost = ['dt', 'wh_dept_id', 'goods_id', 'origin_dmd_cost', 'origin_dmd_dly', 'dmd_daily_base', 'dmd_daily_shop_inc', 'dmd_daily_festival',
                     'dmd_daily']
        cols_dly = ['dt', 'wh_dept_id', 'goods_id', 'origin_dmd_dly', 'dmd_daily_base', 'dmd_new_shop_daily', 'dmd_daily']
        qs_dmd = spark.read_parquet(DEMAND_FORECAST_DAILY + "demand_qs/")[cols_cost]
        yl_dmd = spark.read_parquet(DEMAND_FORECAST_DAILY + "demand_yl/")[cols_cost]
        ls_dmd = spark.read_parquet(DEMAND_FORECAST_DAILY + "demand_ls/")[cols_cost]
        cols_cost.extend(['dmd_new_shop_daily'])
        bc_dmd = spark.read_parquet(DEMAND_FORECAST_DAILY + "demand_bc/")[cols_cost]
        rh_dmd = spark.read_parquet(DEMAND_FORECAST_DAILY + "demand_rh/")[cols_dly]
        qj_dmd = spark.read_parquet(DEMAND_FORECAST_DAILY + "demand_qj/")[cols_dly]
        df_demand_detail = pd.concat([qs_dmd, yl_dmd, ls_dmd, bc_dmd, rh_dmd, qj_dmd])
    logger.info("Load 各大类需求明细 Finish")

    if date == datetime.today().date():
        dim_goods = spark.sql(f"""
                SELECT DISTINCT
                    goods_name
                    , goods_id
                FROM
                    dw_dim.dim_stock_spec_d_his
                WHERE dt = DATE_SUB(CURRENT_DATE(), 1)
            """)
        dim_cmdty = spark.sql('''
                    SELECT
                        cmdty_id
                        , cmdty_name
                    FROM dw_dim.dim_cmdty_d_his
                    WHERE dt = DATE_SUB(CURRENT_DATE(), 1)
                ''')
        dim_wh = spark.sql('''
                    SELECT DISTINCT wh_dept_id, wh_name
                    FROM dw_dim.dim_stock_warehouse_d_his
                    WHERE dt = DATE_SUB(CURRENT_DATE(), 1)
                        AND org_code = 0101
                        AND wh_type = 1
                        AND wh_dept_id IS NOT NULL
                        AND wh_dept_id NOT IN (8801, 170904, 328161, 337814)
                ''')
        logger.info("Load 维度表信息 Finish")
        import hashlib
        df_info = spark.sql(f"""
            SELECT
                emp_no
                , qywx_user_id
            FROM lucky_entwechat.`t_user_sync` wx_info
            LEFT JOIN lucky_ehr.t_ehr_employee lk_info ON wx_info.ehr_emp_no = lk_info.emp_no
            WHERE emp_no IS NOT NULL
        """)
        df_info['emp_hash'] = df_info['qywx_user_id'].apply(lambda x: hashlib.md5(x.encode('utf8')).hexdigest())
        df_info['emp_hash'] = df_info['emp_hash'].astype('str')
        auth_info = df_info[['emp_no', 'emp_hash']].to_dict(orient='tight')['data']
        logger.info("Load auth Finish")

        # 历史商品消耗分布
        df_his_cost_detail = spark.read_parquet('/user/yanxin.lu/sc/dev_tools/goods_cmdty_cost')
        logger.info("Load 历史消耗详情 Finish")
        # 历史库存情况
        df_inv = spark.read_parquet('/user/yanxin.lu/sc/dev_tools/dev_inv')
        logger.info("Load 历史库存情况 Finish")
    df_plan_detail.cmdty_id = df_plan_detail.cmdty_id.astype('int')
    df_forecast_all = df_forecast_all.merge(dim_goods, on=['goods_id'], how='left').merge(dim_wh, on=['wh_dept_id'], how='left')
    df_new_subnew = df_new_subnew.merge(dim_cmdty, on=['cmdty_id'], how='left')
    df_plan_detail = df_plan_detail.merge(dim_cmdty, on=['cmdty_id'], how='left')
    df_replace['goods_id'] = df_replace['goods_id'].astype('int')
    df_replace['dt'] = df_replace['dt'].astype('str')
    df_sea['dt'] = df_sea['dt'].astype('str')
    df_plan['dt'] = df_plan['dt'].astype('str')
    df_plan_detail['dt'] = df_plan_detail['dt'].astype('str')
    df_demand_detail['dt'] = df_demand_detail['dt'].astype('str')
    df_demand_detail['goods_id'] = df_demand_detail['goods_id'].astype('int')
    df_demand_detail[['origin_dmd_cost', 'origin_dmd_dly', 'dmd_daily']] = df_demand_detail[
        ['origin_dmd_cost', 'origin_dmd_dly', 'dmd_daily']].astype('float')
    df_demand_detail = df_demand_detail.fillna(0)

    goods_list = df_forecast_all[['goods_id', 'goods_name']].drop_duplicates().rename(columns={'goods_id': 'value', 'goods_name': 'label'}).to_dict(
        orient='records')
    wh_list = df_forecast_all[['wh_dept_id', 'wh_name']].drop_duplicates().rename(columns={'wh_dept_id': 'value', 'wh_name': 'label'}).to_dict(
        orient='records')

    logger.info("Forecast Load Finished")
