from config.config import *
import numpy as np
from utils.file_utils import save_file, save_hdfs
import multiprocessing
from pandas import DataFrame

from utils.decorator import log_wecom

INPUT_PATH = f'/projects/luckyml/purchase/tower/monitor/{today}/cost_monitor_etl'
# 在途
TRS_PATH = f'/projects/luckyml/purchase/tower/monitor/{today}/transit'
# CG 在途明细
TRS_CG_PATH = f'/projects/luckyml/purchase/tower/monitor/{today}/transit_cg'
# 预测
PRED_PATH = f'/projects/luckyml/purchase/tower/monitor/{today}/pred'
# 损耗
LOSS_PATH = f'/projects/luckyml/purchase/tower/monitor/{today}/loss'
# 门店允收期
SHOP_AVL_PATH = f'/projects/luckyml/purchase/tower/monitor/{today}/shop_available'
# HDFS 结果路径
RES_DETAIL_HDFS_PATH = f'/projects/luckyml/purchase/tower/monitor/{today}/detail/'

RES_HDFS_PATH = f'/projects/luckyml/purchase/tower/monitor/{today}/res/'


def load_data():
    origin = spark.read_parquet(INPUT_PATH).query("large_class_name.isin(['原料', '轻食', '零食', '包装类', '日耗品']) and (small_class_name != '半成品')")

    float_column = ['end_avl_cnt', 'bp_alg', 'ss_days', 'ro', 'bp']
    int_column = ['wh_dept_id', 'goods_id', 'cdc_wh_dept_id', 'is_cdc', 'is_cdc_only', 'is_cdc_model']
    origin[float_column] = origin[float_column].astype(float)
    origin[float_column] = origin[float_column].fillna(0)
    origin[int_column] = origin[int_column]
    origin = origin.fillna({'result_level': ''})

    # 需求预测
    df_predict_base = spark.read_parquet(PRED_PATH)
    df_transit = spark.read_parquet(TRS_PATH)
    df_loss = spark.read_parquet(LOSS_PATH)
    # 门店允收期
    df_shop = spark.read_parquet(SHOP_AVL_PATH)

    # 汇总中心仓需求
    cdc_wh = origin[['wh_dept_id', 'cdc_wh_dept_id']].query("cdc_wh_dept_id > 0").drop_duplicates()
    df_predict_cdc = pd.merge(cdc_wh, df_predict_base, on=['wh_dept_id'], how='left') \
        .groupby(['predict_dt', 'cdc_wh_dept_id', 'goods_id'], as_index=False) \
        .agg({'demand': 'sum'})
    df_predict_cdc.columns = ['predict_dt', 'wh_dept_id', 'goods_id', 'demand_cdc']
    df_predict = pd.merge(df_predict_base, df_predict_cdc, on=['predict_dt', 'wh_dept_id', 'goods_id'], how='left')

    # 合并库存、需求、在途、报损
    df_base = origin.merge(df_predict, on=['wh_dept_id', 'goods_id']) \
        .merge(df_transit, on=['predict_dt', 'wh_dept_id', 'goods_id'], how='left') \
        .merge(df_loss, on=['predict_dt', 'wh_dept_id', 'goods_id'], how='left') \
        .fillna(0) \
        .merge(df_shop, on=['wh_dept_id', 'goods_id'], how='left') \
        .sort_values(by='predict_dt')

    df_inv = df_base[['predict_dt', 'wh_dept_id', 'wh_name', 'cdc_wh_dept_id', 'is_cdc', 'is_cdc_model', 'is_cdc_only', 'goods_id', 'goods_name',
                      'large_class_name', 'large_class_code', 'result_level', 'bp', 'ro', 'ss_days', 'vlt_alg', 'end_avl_cnt', 'FH', 'CG', 'TRS',
                      'total_transit', 'batch_loss_amount', 'demand', 'demand_cdc', 'max_available_days']]

    df_inv['predict_dt'] = pd.to_datetime(df_inv['predict_dt']).dt.date
    df_predict['predict_dt'] = pd.to_datetime(df_predict['predict_dt']).dt.date
    df_loss['predict_dt'] = pd.to_datetime(df_loss['predict_dt']).dt.date
    df_transit['predict_dt'] = pd.to_datetime(df_transit['predict_dt']).dt.date
    df_inv.set_index('predict_dt', inplace=True)
    df_predict.set_index('predict_dt', inplace=True)
    df_transit.set_index('predict_dt', inplace=True)
    df_loss.set_index('predict_dt', inplace=True)

    # 计算CG，最近一次CG，是否有未发货CG
    df_cg = spark.read_parquet(TRS_CG_PATH).query("goods_id == goods_id")
    df_cg['predict_dt'] = pd.to_datetime(df_cg['predict_dt']).dt.date
    df_cg['cg_dt'] = df_cg['predict_dt']
    df_cg.sort_values(by='predict_dt', inplace=True)
    # 最后一次未发货的CG，用于判断是否有未发货CG
    max_full_cg = df_cg.query("cg_full == 1").groupby(['wh_dept_id', 'goods_id'], as_index=False)['cg_dt'].max().rename(
        columns={'cg_dt': 'max_cg_dt'})

    df_cg_agg = df_cg.groupby(['predict_dt', 'wh_dept_id', 'goods_id', 'cg_dt'], as_index=False).agg({'transit_amount': 'sum'})

    # 构建基础维度
    df_dim = df_inv \
        .merge(df_cg_agg, on=['predict_dt', 'wh_dept_id', 'goods_id'], how='left') \
        .merge(max_full_cg, on=['wh_dept_id', 'goods_id'], how='left') \
        .sort_values(by='predict_dt')

    # 是否有未发货CG
    df_dim.loc[df_dim['predict_dt'] < df_dim['max_cg_dt'], 'has_full_cg'] = 1
    df_dim['has_full_cg'].fillna(0, inplace=True)
    # 最近一次CG
    df_dim['next_cg_dt'] = df_dim.groupby(['wh_dept_id', 'goods_id'])['cg_dt'].bfill()
    df_dim.set_index('predict_dt', inplace=True)
    return df_dim, df_predict, df_loss, df_transit


class SimTowerWarn(multiprocessing.Process):
    def __init__(self, wh_dept_id: str, df_inv: DataFrame, df_predict: DataFrame, df_loss: DataFrame, df_transit: DataFrame):
        multiprocessing.Process.__init__(self)
        self.wh_dept_id = wh_dept_id
        self._df_inv = df_inv
        self._df_predict = df_predict
        self._df_loss = df_loss
        self._df_transit = df_transit

    def cal_avl_days(self, beg_cnt, cdc_beg_cnt, dt, max_dt, max_avl_days, _df_predict, _df_loss, _df_transit):
        # 根据未来入仓情况，
        days_inv, cdc_days_inv = 0, 0
        # 含未来在途 连续可用天数
        days_trs, cdc_days_trs = 0, 0

        i = 0
        # 含当天入库的期初
        beg_cnt_trs, cdc_beg_cnt_trs = beg_cnt, cdc_beg_cnt
        while dt <= max_dt:
            demand = _df_predict.demand.get(dt, 0)
            trs = _df_transit.total_transit.get(dt, 0)
            loss = _df_loss.batch_loss_amount.get(dt, 0)

            end_pred = beg_cnt - demand - loss
            end_pred_trs = beg_cnt_trs + trs - demand - loss
            # 可用天数
            days_inv += 1 if end_pred > 0 else 0
            days_trs += 1 if end_pred_trs > 0 else 0

            beg_cnt = max(end_pred, 0)
            beg_cnt_trs = max(end_pred_trs, 0)

            """ 中心仓 """
            cdc_demand = _df_predict.demand_cdc.get(dt, 0)
            cdc_end_pred = cdc_beg_cnt - cdc_demand - loss
            cdc_end_pred_trs = cdc_beg_cnt_trs + trs - cdc_demand - loss

            cdc_days_inv += 1 if cdc_end_pred > 0 else 0
            cdc_days_trs += 1 if cdc_end_pred_trs > 0 else 0

            cdc_beg_cnt = max(cdc_end_pred, 0)
            cdc_beg_cnt_trs = max(cdc_end_pred_trs, 0)

            dt += timedelta(days=1)
            i += 1

        # 超过需求预测天数之外，以最后一天的期初/需求计算剩余可用天数
        if beg_cnt > 0:
            rest_days = round(beg_cnt / demand) if demand > 0 else np.inf
            days_inv += rest_days
            days_trs += rest_days
        days_inv = min(days_inv, max_avl_days)
        days_trs = min(days_trs, max_avl_days)

        if cdc_beg_cnt > 0:
            cdc_rest_days = round(cdc_beg_cnt / cdc_demand) if cdc_demand > 0 else np.inf
            cdc_days_inv += cdc_rest_days
            cdc_days_trs += cdc_rest_days
        cdc_days_inv = min(cdc_days_inv, max_avl_days)
        cdc_days_trs = min(cdc_days_trs, max_avl_days)
        return days_inv, days_trs, cdc_days_inv, cdc_days_trs

    def simulate(self, _df, df_inv, _df_predict, _df_loss, _df_transit):
        i = 0
        end_pred = 0
        cdc_end_pred = 0
        max_dt = today + timedelta(days=120)
        for index, row in df_inv[df_inv.index <= max_dt].iterrows():
            i += 1
            dt, wh_dept_id, goods_id, beg_cnt, cdc_beg_cnt, trs, loss, demand, cdc_demand, max_avl_days = index, row.wh_dept_id, row.goods_id, row.end_avl_cnt, row.end_avl_cnt, row.total_transit, row.batch_loss_amount, row.demand, row.demand_cdc, row.max_available_days
            if i > 1:
                beg_cnt = end_pred
                cdc_beg_cnt = cdc_end_pred
            days_inv, days_trs, cdc_days_inv, cdc_days_trs = self.cal_avl_days(beg_cnt, cdc_beg_cnt, dt, max_dt, max_avl_days, _df_predict, _df_loss,
                                                                               _df_transit)
            end_pred = max(beg_cnt + trs - demand - loss, 0)
            cdc_end_pred = max(cdc_beg_cnt + trs - cdc_demand - loss, 0)
            _df.loc[i] = [dt, wh_dept_id, goods_id, beg_cnt, demand, loss, trs, end_pred, days_inv, days_trs, cdc_beg_cnt, cdc_demand, cdc_end_pred,
                          cdc_days_inv, cdc_days_trs]
        return _df

    def run(self):
        res_list = []
        # 按货物遍历,计算可用天数
        for (wh_dept_id, goods_id), group in self._df_inv.groupby(['wh_dept_id', 'goods_id']):
            _df_inv = self._df_inv.query(f"goods_id == {goods_id}")
            _df_predict = self._df_predict.query(f"goods_id == {goods_id}")
            _df_loss = self._df_loss.query(f"goods_id == {goods_id}")
            _df_transit = self._df_transit.query(f"goods_id == {goods_id}")
            goods_res = pd.DataFrame(
                columns=['predict_dt', 'wh_dept_id', 'goods_id', 'beg_cnt', 'demand', 'loss', 'trs', 'end_pred', 'avl_days', 'avl_trs_days',
                         'cdc_beg_cnt', 'cdc_demand', 'cdc_end_pred', 'cdc_avl_days', 'cdc_avl_trs_days'])
            res_list.append(self.simulate(goods_res, _df_inv, _df_predict, _df_loss, _df_transit))
        res = pd.concat(res_list)
        cols = ['beg_cnt', 'demand', 'loss', 'trs', 'end_pred', 'avl_days', 'avl_trs_days', 'cdc_beg_cnt', 'cdc_demand', 'cdc_end_pred', 'cdc_avl_days', 'cdc_avl_trs_days']
        res[cols] = res[cols].astype('float')
        save_file(res, f'/data/purchase/tower/cal_avl_days/{self.wh_dept_id}/res.parquet')
        return res


def avl_days_process(df_inv, df_predict, df_loss, df_transit):
    ths = []
    for wh_dept_id in df_inv['wh_dept_id'].drop_duplicates():
        _df_inv = df_inv.query(f"wh_dept_id == {wh_dept_id}")
        _df_predict = df_predict.query(f"wh_dept_id == {wh_dept_id}")
        _df_loss = df_loss.query(f"wh_dept_id == {wh_dept_id}")
        _df_transit = df_transit.query(f"wh_dept_id == {wh_dept_id}")
        thread = SimTowerWarn(wh_dept_id, _df_inv, _df_predict, _df_loss, _df_transit)
        thread.daemon = True
        thread.start()
        ths.append(thread)

    for th in ths:
        th.join()
    logger.info('分仓计算可用天数计算完毕，合并数据')
    df_avl_days = pd.read_parquet('/data/purchase/tower/cal_avl_days/')

    return df_avl_days


def get_guide(row):
    """
    判断告警类型，策略建议
    """
    guide_list = []
    """
    需补充CG
        无CG：且min(仓库预计可用天数）=<vlt；
        有CG：仓库库存+CG在途预计可用天数=<vlt
        仅中心仓及中心仓对应的货物
    """
    warn_bu_cg = False
    if (((row.is_cdc == 1) and (row.is_cdc_model == 1)) or (row.is_cdc_only == 1)) \
            and (row.avl_trs_days <= row.vlt_alg):
        warn_bu_cg = True
    guide_list.append('补充CG单') if warn_bu_cg else None

    """
    仓库断仓风险
        日耗器具：min（仓库预计可用天数）<max(下一批CG到仓时间-today(),15)
        食品&包材类：min（仓库预计可用天数）<max(下一批CG到仓时间-today(),7)
    """
    warn_duan_cang = False
    next_cg = (datetime.strptime(row.next_cg_dt, '%Y-%m-%d').date() - today).days if row.next_cg_dt != row.next_cg_dt else 0
    if row.large_class_name in ['日耗品', '器具类', '工服类', '营销物料', '办公用品']:
        warn_duan_cang = row.avl_days < max(next_cg, 15)
    elif row.large_class_name in ['零食', '轻食', '原料', '包装类']:
        warn_duan_cang = row.avl_days < max(next_cg, 7)
    guide_list.append('提前入仓|安排调拨') if warn_duan_cang else None

    """
    库存积压
        一级物料：仓库预计可用天数>最大库存天数时
        其他食品类货物：仓库预计可用天数>60天时
        其他非食品类货物：仓库预计可用天数>90天时
    """
    warn_ji_ya = False
    if row.result_level == '一级':
        warn_ji_ya = row.avl_days > row.max_days
    elif row.large_class_name in ['零食', '轻食', '原料']:
        warn_ji_ya = row.avl_days > 60
    else:
        warn_ji_ya = row.avl_days > 90
    guide_list.append('延迟入仓|均仓调拨|营销清库存') if warn_ji_ya else None

    """
    CG过多
        一级物料：仓库预计可用天数-（下一批CG入仓时间-today（））> 最高库存天数（bp+ro+ss)，且有CG未发货
        中心仓对应的物料（非独立仓）：仓库库存/(中心仓消耗+城市仓消耗)>最高库存天数（bp+ro+ss)，且有CG未发货， 且有CG未发货
        其他非中心仓物料：仓库预计可用天数> 最高库存天数（bp+ro+ss)，且有CG未发货，且有CG未发货
        仅中心仓及中心仓模式的货物
    """
    # CG 过多
    warn_cg_duo = False
    if (row.has_full_cg == 1) and (row.is_cdc == 1) and (row.is_cdc_model == 1):
        next_cg = (datetime.strptime(row.next_cg_dt, '%Y-%m-%d').date() - today).days if row.next_cg_dt != row.next_cg_dt else 0
        std_days = row.bp + row.ro + row.ss_days
        if row.result_level == '一级':
            warn_cg_duo = (row.avl_days - next_cg) > (std_days if row.max_days is None else row.max_days)
        elif (row.is_cdc == 1) and (row.is_cdc_only != 1):  # 中心仓模式
            warn_cg_duo = row.cdc_avl_days > row.max_days
        else:
            warn_cg_duo = row.avl_days > row.max_days

    guide_list.append('取消CG') if warn_cg_duo else None

    is_warn = warn_bu_cg | warn_duan_cang | warn_ji_ya | warn_cg_duo
    guide = '|'.join(guide_list)
    return warn_bu_cg, warn_duan_cang, warn_ji_ya, warn_cg_duo, is_warn, guide


def get_warn(row):
    """
    获取告警类型及建议
    """
    warn_type, warn_type_comment = None, None

    if row.avl_days < row.min_days:
        warn_type_comment = '断仓'
        warn_type = 2
    elif row.avl_days > row.max_days:
        warn_type_comment = '呆滞'
        warn_type = 1
    else:
        warn_type_comment = '健康'
        warn_type = 3

    return warn_type_comment, warn_type


def get_warn_guide(row):
    warn_type_comment, warn_type = get_warn(row)
    warn_bu_cg, warn_duan_cang, warn_ji_ya, warn_cg_duo, is_warn, guide = get_guide(row)
    return warn_type_comment, warn_type, guide, warn_bu_cg, warn_duan_cang, warn_ji_ya, warn_cg_duo, is_warn


@log_wecom('控制塔-模拟告警', P_TWO)
def process():
    logger.info('数据载入')
    df_inv, df_predict, df_loss, df_transit = load_data()
    logger.info('开始计算可用天数')
    # 计算未来每一天可用天数
    df_avl_days = avl_days_process(df_inv, df_predict, df_loss, df_transit)
    # 基础数据
    df_final = pd.merge(df_inv, df_avl_days, on=['predict_dt', 'wh_dept_id', 'goods_id', 'demand'], how='left')

    """根据业务逻辑判断是否告警"""
    # 设置告警阈值, 仅判断未来4周告警
    df_final['max_days'] = df_final['ss_days'] + df_final['ro'] + 1.5 * df_final['bp']
    df_final['min_days'] = df_final['ss_days']
    df_final['next_cg_dt'] = df_final['next_cg_dt'].astype('str')
    df_final = df_final[(df_final.predict_dt <= (today + timedelta(days=28)))]
    logger.info('判断库存状态及策略建议')

    # 根据告警类型获取策略建议
    df_final[
        ['warn_type_comment', 'warn_type', 'warn_guide', 'warn_bu_cg', 'warn_duan_cang', 'warn_ji_ya', 'warn_cg_duo', 'is_warn']] = df_final.apply(
        lambda row: get_warn_guide(row), axis=1, result_type='expand')
    df_final['dt'] = today

    save_hdfs(data=df_final, hdfs_path=RES_DETAIL_HDFS_PATH, file_name='result.parquet')

    # HIVE表基础数据
    df_hive = df_final[['predict_dt', 'wh_name', 'wh_dept_id', 'goods_name', 'goods_id', 'large_class_name',
                        'large_class_code', 'warn_type_comment', 'warn_type', 'warn_guide', 'dt']]
    logger.info('Finish')
    save_hdfs(data=df_hive, hdfs_path=RES_HDFS_PATH, file_name='result.parquet')

    return df_final, df_hive


if __name__ == '__main__':
    process()
