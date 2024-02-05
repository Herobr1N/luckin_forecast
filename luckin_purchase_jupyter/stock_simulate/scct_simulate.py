from utils.file_utils import save_file, read_folder, remove, save_hdfs
from utils.decorator import *
from config.paths import *
import multiprocessing
from config.config import *
import numpy as np
"""
水滴任务41236调度
已废弃
"""
pd.set_option('mode.chained_assignment', None)

HDFS_BASE = '/projects/luckyml/purchase/tower/simulate/spec_loss'
OUTPUT_BASE = '/data/scct/output/simulate/spec_loss'
OUTPUT_FULL_SPEC_LOCAL_PATH = f'{OUTPUT_BASE}/{today}/'
OUTPUT_FULL_SPEC_HDFS_PATH = f'{HDFS_BASE}/{today}/'


class scct_loss(multiprocessing.Process):
    def __init__(self, thread_name: str, data: DataFrame, output: str):
        multiprocessing.Process.__init__(self)
        self.thread_name = thread_name
        self.df = data
        self.output = output

    def run(self):
        logger.info(f"{self.thread_name} Start")
        df_emp = pd.DataFrame()
        goods_ls = self.df.goods_id.drop_duplicates().tolist()
        # 货物维度
        for goods in goods_ls:
            j = 0
            # last_end 上期期末
            last_end = 0
            # 需求份额分配
            dmd_share = 0
            group = self.df.query(f"goods_id =={goods}")
            # 按可订-报损最早时间对规格排序
            group.sort_values(['dt', 'wh_order_flg', 'rnk_final'], ascending=[1, 0, 0], inplace=True)

            spec_ls = group['spec_id'].drop_duplicates().tolist()
            # 仓-天-货-规格计数项i
            i = 0
            result = pd.DataFrame(columns=["dt", "wh_dept_id", "goods_id", "spec_id", "beg", "end", "dmd", 'actual_dly', "transit", "batch_amount", "batch_loss", "buffer", "dmd_share", "wh_order_flg", "rnk", 'spec_consume_rnk'])
            # 货物-天维度
            for (wh, goods, dt), dly in group.groupby(['wh_dept_id', 'goods_id', 'dt']):
                # 货物-天-规格维度
                for spec in range(len(spec_ls)):
                    dly_spec = dly.query(f"spec_id =={spec_ls[spec]}")
                    for row in dly_spec.itertuples():
                        dmd = float(getattr(row, "dmd")) if spec == 0 else dmd_share

                        simulate_already = result.query(f"spec_id =={spec_ls[spec]}")
                        # 计算期初
                        beg = float(getattr(row, "beg")) if j == 0 else simulate_already.iloc[-1, 5].sum()
                        # 实际可报损量 min（截止j日报损批次总量，期初）
                        available_batch_amount = min(simulate_already.loc[simulate_already.dt == f'{today}'].beg.sum(), simulate_already.batch_amount.sum())
                        # 实际出库
                        actual_dly = min(dmd, beg)
                        # 按规格计算 批次已使用量
                        buffer = max(simulate_already.query("beg > 0").actual_dly.sum() - available_batch_amount + simulate_already.batch_loss.sum(), 0) + actual_dly
                        # 无库存情况下无报损
                        if beg == 0:
                            batch_loss = 0
                            # buffer = 0
                        else:
                            if float(getattr(row, "batch_amount")) != 0:
                                if float(getattr(row, "batch_amount")) >= buffer:
                                    batch_loss = float(getattr(row, "batch_amount")) - buffer + dmd
                                    batch_loss = min(beg, batch_loss)
                                    buffer = 0
                                else:
                                    batch_loss = 0
                                    buffer = buffer - float(getattr(row, "batch_amount"))
                            else:
                                batch_loss = 0
                        if j == 0:
                            end = float(getattr(row, "beg")) + float(getattr(row, "transit")) - dmd - batch_loss
                            # beg = float(getattr(row, "beg"))
                        else:
                            end = simulate_already.iloc[-1, 5].sum() + float(getattr(row, "transit")) - dmd - batch_loss
                            # beg = last_end

                        # 需求份额分配
                        dmd_share = max(float(-1 * end), 0)
                        end = max(float(end), 0)

                        # last_end = end

                        result.loc[i] = [str(getattr(row, "dt").date())] + [wh, goods, spec_ls[spec]] + [beg] + [end] + [dmd] + [actual_dly] + [float(getattr(row, "transit")),
                                                                                                                                                float(getattr(row, "batch_amount")),
                                                                                                                                                batch_loss, buffer, dmd_share,
                                                                                                                                                float(getattr(row, "wh_order_flg")),
                                                                                                                                                float(getattr(row, "rnk_final")), spec]
                        i = i + 1
                j = j + 1
            df_emp = pd.concat([result, df_emp])
        df_emp = df_emp.fillna(0)
        save_file(df_emp, self.output + f'{self.thread_name}.csv')
        logger.info(f"{self.thread_name} END")
        return df_emp


def get_base_data():
    batch_one = spark.sql(f"""
        SELECT IF(DATE(shop_receiving_exp_date) < '{today}', '{today}', DATE(shop_receiving_exp_date)) AS dt
             , wh_dept_id
             , goods_id
             , spec_id
             , expiration_date
             , batch_no
             , SUM(end_wh_stock_cnt) AS batch_amount
        FROM dw_dws.`dws_stock_warehouse_stock_batch_adjust_d_inc_summary` wh_batch
        WHERE dt = '{yesterday}'
            AND end_wh_stock_cnt > 0 --期末库存=0则不存在报损情况
            AND shop_receiving_exp_date <= '{today + timedelta(days=120)}'
        GROUP BY IF(DATE(shop_receiving_exp_date) < '{today}', '{today}', DATE(shop_receiving_exp_date)),
                 wh_dept_id,
                 goods_id,
                 spec_id,
                 expiration_date,
                 batch_no
        HAVING SUM(end_wh_stock_cnt) > 0
    """)

    # 待检状态批次报损
    batch_two = spark.sql(f"""
            SELECT
                exp_date AS dt
                , wh_dept_id
                , goods_id
                , spec_id
                , expiration_date
                , batch_no
                , SUM(quarantine_cnt) AS batch_amount
            FROM (
                SELECT
                    wh_dept_id
                    , batch.spec_id
                    , goods_id
                    , expiration_date
                    , shop_receiving_period
                    , batch_no
                    -- 根据批次号、规格保质期、门店允收期，判断报损时间
                    , DATE_ADD(TO_DATE(batch_no, 'yyyyMMdd'), expiration_date - IFNULL(shop_receiving_period, 0) - 1) AS exp_date
                    , quarantine_cnt
                FROM dw_dwd.`fact_dwd_stock_warehouse_stock_batch_everyday_d_inc` AS batch
                LEFT JOIN dw_dim.dim_stock_spec_d_his AS spec_info ON spec_info.dt == '{yesterday}' AND spec_info.spec_id == batch.spec_id
                WHERE batch.dt = '{today}'
                    AND quarantine_cnt > 0
            )
            GROUP BY exp_date, wh_dept_id, goods_id, spec_id, expiration_date, batch_no
        """)
    base_batch = pd.concat([batch_one, batch_two]).dropna(subset=['expiration_date', 'dt'])
    base_batch = base_batch.groupby(['wh_dept_id', 'goods_id', 'spec_id', 'dt', 'batch_no'], as_index=False).agg({'batch_amount': 'sum'})

    # 期末可用库存
    stock_detail = spark.sql(f"""
                    SELECT
                        wh_dept_id
                        , goods_id
                        , spec_id
                        , beg_one + beg_two AS beg -- 合并待检库存与正常库存
                    FROM (
                        SELECT
                            wh_dept_id
                            , goods_id
                            , spec_id
                            , SUM(IFNULL(end_wh_stock_avl_cnt, 0)) AS beg_one
                            , SUM(IFNULL(end_wh_stock_quarantine_cnt, 0)) AS beg_two -- 待检库存
                        FROM dw_dws.`dws_stock_warehouse_stock_adjust_begin_end_d_inc_summary`
                        WHERE dt = '{yesterday}'
                            AND wh_dept_id IN (SELECT DISTINCT wh_dept_id FROM dw_ads_scm_alg.`dim_warehouse_city_shop_d_his` WHERE dt = '{yesterday}')
                        GROUP BY wh_dept_id, goods_id, spec_id
                    )
            """)

    # 在途
    wh_transit_base = spark.sql(f"""
        SELECT
            IF(DATE(plan_rec_date)<'{today}', '{today}', DATE(plan_rec_date)) AS dt,
            wh_dep_id AS wh_dept_id,
            goods_id,
            spec_id,
            SUM(transit_amount) AS transit
        FROM
            dw_ads_scm_alg.`dim_wh_spec_in_transit`
        WHERE dt = '{yesterday}'
            AND (
                (source = 4 AND status IN (3, 4)) -- FH
                OR (source = 3 AND status IN (2, 4, 7)) -- ALLOT
                OR (source = 1) -- CG
            )
            AND plan_rec_date <= '{today + timedelta(days=120)}'
        GROUP BY
            IF(DATE(plan_rec_date)<'{today}', '{today}', DATE(plan_rec_date)),
            wh_dept_id,
            goods_id,
            spec_id
    """)
    # zaitu
    wh_transit_base = wh_transit_base.query("transit>0")

    wh_order_spec = spark.sql(f"""
            SELECT
                wh_dept_id
                , spec_id
                , 1 as wh_order_flg
            FROM dw_dim.`dim_stock_warehouse_spec_d_his`
            WHERE dt = '{yesterday}'
                AND wh_status = 1
                AND wh_may_order_spec_status = 1
            """)

    dmd_forecast_day = spark.sql(f"""
            SELECT
                predict_dt AS dt,
                wh_dept_id,
                goods_id,
                demand AS dmd
            FROM
                dw_ads_scm_alg.`dm_wh_goods_demand_forecast_daily`
            WHERE dt = '{today}'
        """)
    dmd_forecast_day.dt = pd.to_datetime(dmd_forecast_day.dt)

    time_seq = pd.DataFrame(pd.date_range(start=today, periods=120, freq='d'))
    time_seq.columns = ['dt']
    # t-1期末-->当日期初
    mid0 = time_seq.merge(stock_detail, how='cross')
    # 扩展到未来120天
    wh_transit_base.dt = pd.to_datetime(wh_transit_base.dt)
    # 预测货物范围作为主维度
    mid1 = dmd_forecast_day.merge(mid0, on=['dt', 'wh_dept_id', 'goods_id'], how='left')
    # 有效规格：期初>0或在途>0的规格；库存+未来在途=0的部分默认未来无报损
    stock_valid_list = mid0.query("beg>0")[['goods_id', 'spec_id']].drop_duplicates()
    transit_valid_list = wh_transit_base.query("transit>0")[['goods_id', 'spec_id']].drop_duplicates()
    valid_list = stock_valid_list.merge(transit_valid_list, on=['goods_id', 'spec_id'], how='outer')
    mid2 = mid1.merge(wh_transit_base, on=['wh_dept_id', 'goods_id', 'spec_id', 'dt'], how='left')
    mid2 = mid2.loc[mid1.spec_id.isin(valid_list.spec_id.drop_duplicates().tolist())]

    # 加入可订库存
    mid3 = mid2.merge(wh_order_spec, on=['wh_dept_id', 'spec_id'], how='left')
    base_batch.dt = pd.to_datetime(base_batch.dt)
    # 加入批次信息
    mid4 = mid3.merge(base_batch, on=['wh_dept_id', 'spec_id', 'goods_id', 'dt'], how='left')
    mid4[['wh_order_flg', 'transit', 'batch_amount']] = mid4[['wh_order_flg', 'transit', 'batch_amount']].fillna(0)

    # 未来四个月没有报损批次的货物
    no_order_batch = mid4.groupby(['wh_dept_id', 'goods_id'], as_index=False).agg({'batch_amount': 'sum'})
    no_order_batch = no_order_batch.loc[no_order_batch.batch_amount == 0]
    no_order_batch['flg'] = 1
    no_order_batch = no_order_batch[['wh_dept_id', 'goods_id', 'flg']]
    mid5 = mid4.merge(no_order_batch, on=['wh_dept_id', 'goods_id'], how='left')
    mid5 = mid5[mid5.flg.isna()].drop(columns='flg')
    # 没有可订信息&未来四个月没有报损批次的规格
    no_order_batch_p2 = mid5.groupby(['wh_dept_id', 'goods_id', 'spec_id'], as_index=False).agg({'batch_amount': 'sum', 'wh_order_flg': 'sum'})
    no_order_batch_p2 = no_order_batch_p2.loc[(no_order_batch_p2.batch_amount == 0) & (no_order_batch_p2.wh_order_flg == 0)]
    no_order_batch_p2['flg'] = 1
    no_order_batch_p2 = no_order_batch_p2[['wh_dept_id', 'goods_id', 'spec_id', 'flg']]
    mid6 = mid5.merge(no_order_batch_p2, on=['wh_dept_id', 'goods_id', 'spec_id'], how='left')
    mid6 = mid6[mid6.flg.isna()].drop(columns='flg')
    # mid5 = mid5
    # 消耗排序
    # 可订/非可订两部分
    # spec_rnk
    spec_rnk = mid6.groupby(['wh_dept_id', 'goods_id', 'spec_id', 'wh_order_flg'], as_index=False).agg({'batch_amount': 'sum'})
    spec_rnk.loc[spec_rnk.batch_amount > 0, 'spec_rnk'] = 1
    spec_rnk.loc[spec_rnk.batch_amount == 0, 'spec_rnk'] = 0

    # batch_rnk
    batch_rnk = mid6.query("batch_amount>0").groupby(['wh_dept_id', 'goods_id', 'spec_id', 'wh_order_flg'], as_index=False).agg({'dt': 'min'})
    batch_rnk.rename(columns={'dt': 'min_dt'}, inplace=True)
    batch_rnk['batch_rnk'] = batch_rnk.groupby(['wh_dept_id', 'goods_id', 'wh_order_flg'], as_index=False).min_dt.rank(ascending=False)
    res = mid6.merge(spec_rnk[['wh_dept_id', 'spec_id', 'spec_rnk']], on=['wh_dept_id', 'spec_id'], how='left').merge(batch_rnk[['wh_dept_id', 'spec_id', 'batch_rnk']], on=['wh_dept_id', 'spec_id'], how='left')
    res['rnk_final'] = (res['spec_rnk'] + res['batch_rnk']).fillna(0)
    # res.sort_values(['wh_dept_id','dt','goods_id','wh_order_flg','rnk_final'], ascending=[1, 1, 1, 0, 0], inplace=True)
    return res


def start(sim_type, output_path, hdfs_path):
    @log_wecom(f'库存模拟-{sim_type}')
    def simulate():
        wh_goods_base_daily = get_base_data()
        wh_goods_base_daily.sort_values(['wh_dept_id', 'dt', 'goods_id', 'wh_order_flg', 'rnk_final'], ascending=[1, 1, 1, 0, 0], inplace=True)

        wh_ls = wh_goods_base_daily['wh_dept_id'].drop_duplicates()
        ths = []
        for wh_id in wh_ls:
            wh_goods_base_daily_thn = wh_goods_base_daily.query(f"wh_dept_id == {wh_id}")
            thread = scct_loss(f"wh_{wh_id}", wh_goods_base_daily_thn, output_path)
            thread.daemon = True
            thread.start()
            ths.append(thread)
        for th in ths:
            th.join()

        # 合并保存
        res_df = read_folder(output_path)
        res_df = res_df[['wh_dept_id', 'goods_id', 'spec_id', 'beg', 'end', 'actual_dly', 'transit', 'batch_loss', 'wh_order_flg', 'dt']]
        res_df[['beg', 'end', 'actual_dly', 'transit', 'batch_loss']] = np.abs(res_df[['beg', 'end', 'actual_dly', 'transit', 'batch_loss']])
        res_df = res_df.rename(columns={'actual_dly': 'dmd'})
        save_hdfs(data=res_df, hdfs_path=hdfs_path, file_name='result.parquet')

    simulate()


if __name__ == '__main__':
    logger.info('控制塔全规格库存模拟开始')
    start('控制塔全规格模拟', OUTPUT_FULL_SPEC_LOCAL_PATH, OUTPUT_FULL_SPEC_HDFS_PATH)
