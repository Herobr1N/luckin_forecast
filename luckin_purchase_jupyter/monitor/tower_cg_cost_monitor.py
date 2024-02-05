from config.config import *
import numpy as np

from utils.decorator import log_wecom
from utils.file_utils import save_hdfs
from utils.msg_utils import Message

INPUT_PATH = f'/projects/luckyml/purchase/tower/monitor/{today}/cost_monitor_etl'
# 在途
TRS_PATH = f'/projects/luckyml/purchase/tower/monitor/{today}/transit'
# 预测
PRED_PATH = f'/projects/luckyml/purchase/tower/monitor/{today}/pred'
# 损耗
LOSS_PATH = f'/projects/luckyml/purchase/tower/monitor/{today}/loss'
# 门店允收期
SHOP_AVL_PATH = f'/projects/luckyml/purchase/tower/monitor/{today}/shop_available'
# HDFS 结果路径
RES_HDFS_PATH = f'/projects/luckyml/purchase/tower/monitor/{today}/email_res/'


def get_warn(row):
    """
    判断告警类型
    """

    """
    需补充CG
        无CG：且min(仓库预计可用天数）=<vlt；
        有CG：仓库库存+CG在途预计可用天数=<vlt
        仅中心仓及中心仓对应的货物
    """
    warn_bu_cg = False
    if (((row.is_cdc == 1) and (row.is_cdc_model == 1)) or (row.is_cdc_only == 1)) \
            and (row.avl_days_with_cg <= row.vlt_alg):
        warn_bu_cg = True

    """
    仓库断仓风险
        日耗器具：min（仓库预计可用天数）<max(下一批CG到仓时间-today(),15)
        食品&包材类：min（仓库预计可用天数）<max(下一批CG到仓时间-today(),7)
    """
    warn_duan_cang = False
    next_cg = (datetime.strptime(row.min_cg_date, '%Y-%m-%d').date() - today).days if row.min_cg_date is not None else 0
    if row.large_class_name in ['日耗品', '器具类', '工服类', '营销物料', '办公用品']:
        warn_duan_cang = row.avl_days < max(next_cg, 15)
    elif row.large_class_name in ['零食', '轻食', '原料', '包装类']:
        warn_duan_cang = row.avl_days < max(next_cg, 7)

    """
    库存积压
        一级物料：仓库预计可用天数>最大库存天数时
        其他食品类货物：仓库预计可用天数>60天时
        其他非食品类货物：仓库预计可用天数>90天时
    """
    warn_ji_ya = False
    if row.result_level == '一级':
        warn_ji_ya = row.avl_days > row.dos_max
    elif row.large_class_name in ['零食', '轻食', '原料']:
        warn_ji_ya = row.avl_days > 60
    else:
        warn_ji_ya = row.avl_days > 90

    """
    CG过多
        一级物料：仓库预计可用天数-（下一批CG入仓时间-today（））> 最高库存天数（默认，bp+ro+ss)，且有CG未发货
        中心仓对应的物料（非独立仓）：仓库库存/(中心仓消耗+城市仓消耗)>bp+ro+ss，且有CG未发货， 且有CG未发货
        其他非中心仓物料：仓库预计可用天数>bp+ro+ss，且有CG未发货，且有CG未发货
        仅中心仓及中心仓模式的货物
    """
    # CG 过多
    warn_cg_duo = False
    if (row.cg_full == 1) and (row.is_cdc == 1) and (row.is_cdc_model == 1):
        next_cg = (datetime.strptime(row.min_cg_date, '%Y-%m-%d').date() - today).days if row.min_cg_date is not None else 0
        std_days = row.bp_alg + row.ro + row.ss_days
        if row.result_level == '一级':
            warn_cg_duo = (row.avl_days - next_cg) > (std_days if row.dos_max is None else row.dos_max)
        elif (row.is_cdc == 1) and (row.is_cdc_only != 1):
            warn_cg_duo = row.avl_days_cdc > std_days
        else:
            warn_cg_duo = row.avl_days > std_days

    """
    消耗波动
        abs（消耗同比）>30% or abs（环比L7日均）>30%
    """
    if row.large_class_name in ['零食', '轻食', '原料', '包装类']:
        warn_bo_dong = (abs(row.consume_ratio_T_1) > 0.3) or (abs(row.consume_ratio_L_7) > 0.3)
    else:
        warn_bo_dong = (abs(row.consume_ratio_T_1) > 1)

    is_warn = warn_bu_cg | warn_duan_cang | warn_ji_ya | warn_cg_duo | warn_bo_dong

    return warn_bu_cg, warn_duan_cang, warn_ji_ya, warn_cg_duo, warn_bo_dong, is_warn


def get_warn_msg(row):
    """
    获取告警类型及建议
    """
    warn_type_comment, warn_guide = None, None
    warn_type = row.warn_type
    if warn_type == 'warn_bu_cg':
        warn_type_comment = '库存不足'
        warn_guide = '补充CG单'
    if warn_type == 'warn_duan_cang':
        warn_type_comment = '库存不足'
        warn_guide = '提前入仓｜安排调拨'
    if warn_type == 'warn_ji_ya':
        warn_type_comment = '库存积压'
        warn_guide = '延迟入仓|均仓调拨|营销清库存'
    if warn_type == 'warn_cg_duo':
        warn_type_comment = '库存积压'
        warn_guide = '延迟入仓|取消CG'
    return warn_type_comment, warn_guide


def get_avl_days(df_inv):
    """
    计算库存可用天数
    :param df_inv: 当前库存
    """
    df_predict = spark.read_parquet(PRED_PATH)
    df_transit = spark.read_parquet(TRS_PATH)
    df_loss = spark.read_parquet(LOSS_PATH)
    df_shop = spark.read_parquet(SHOP_AVL_PATH)

    middle = df_inv.merge(df_predict, on=['wh_dept_id', 'goods_id']) \
        .merge(df_transit, on=['predict_dt', 'wh_dept_id', 'goods_id'], how='left') \
        .merge(df_loss, on=['predict_dt', 'wh_dept_id', 'goods_id'], how='left') \
        .fillna(0) \
        .sort_values(by='predict_dt')
    middle['avl_days_alg'] = middle.groupby(['wh_dept_id', 'goods_id'], as_index=False)['predict_dt'].rank()
    middle[['dmd_cum', 'transit_cum', 'loss_cum']] = middle.groupby(['wh_dept_id', 'goods_id'], as_index=False)[
        'demand', 'total_transit', 'batch_loss_amount'].cumsum()
    # 包含在途预计期末
    middle['end_pred_with_transit'] = middle['end_avl_cnt'] + middle['transit_cum'] - middle['loss_cum'] - middle['dmd_cum']
    # 不含在途的预计期末
    middle['end_pred'] = middle['end_avl_cnt'] - middle['loss_cum'] - middle['dmd_cum']

    # 最大可用天数
    avl_days_with_transit = middle.query("end_pred_with_transit >= 0").groupby(['wh_dept_id', 'goods_id'], as_index=False)['avl_days_alg'].max()
    avl_days = middle.query("end_pred >= 0").groupby(['wh_dept_id', 'goods_id'], as_index=False)['avl_days_alg'].max()

    final_days = avl_days_with_transit.merge(avl_days, on=['wh_dept_id', 'goods_id'], how='left') \
        .merge(df_shop, on=['wh_dept_id', 'goods_id'], how='left')

    final_days['avl_days_with_trs_alg'] = final_days['max_available_days'].where(final_days['avl_days_alg_x'] > final_days['max_available_days'],
                                                                                 other=final_days['avl_days_alg_x'])
    final_days['avl_days_alg'] = final_days['max_available_days'].where(final_days['avl_days_alg_y'] > final_days['max_available_days'],
                                                                        other=final_days['avl_days_alg_y'])

    return final_days


def load():
    origin = spark.read_parquet(INPUT_PATH).query("small_class_name != '半成品'")

    # 处理数据类型转化
    dly_column = ['dly_out', 'dly_out_t_1', 'dly_seven_out_avg', 'dly_out_last_week', 'dly_out_avg_workday',
                  'dly_out_avg_holiday']
    cost_column = ['actual_consume', 'cost_t_1', 'seven_cost_avg', 'cost_last_week', 'cost_avg_workday',
                   'cost_avg_holiday']
    consume_column = ['consume', 'consume_t_1', 'seven_consume_avg', 'consume_last_week', 'consume_avg_workday',
                      'consume_avg_holiday']
    float_column = cost_column + dly_column + ['dly_out_l30_avg', 'dly_out_l90_avg', 'end_avl_cnt', 'cg_transit', 'allot_transit',
                                               'cdc_wh_dept_id', 'is_cdc', 'is_cdc_model', 'is_cdc_only', 'bp_alg', 'ro']
    int_column = ['wh_dept_id', 'cdc_wh_dept_id', 'is_cdc', 'is_cdc_only', 'bp_alg', 'ro']

    origin[float_column] = origin[float_column].astype(float)
    origin[float_column] = origin[float_column].fillna(0)
    origin[int_column] = origin[int_column].astype(int)
    origin[float_column] = np.abs(origin[float_column])

    # 根据配方确定消耗口径
    origin[consume_column] = origin[dly_column]
    origin[consume_column] = origin[consume_column].where(origin['is_formula'] == 0, origin[cost_column], axis=0)

    # '日耗品', '器具类', '工服类', '营销物料', '办公用品' t-1 consume 对应L30, L7 对应 L90
    origin.loc[origin['large_class_name'].isin(['日耗品', '器具类', '工服类', '营销物料', '办公用品']), 'consume_t_1'] = origin[
        'dly_out_l30_avg']
    origin.loc[origin['large_class_name'].isin(['日耗品', '器具类', '工服类', '营销物料', '办公用品']), 'seven_consume_avg'] = origin[
        'dly_out_l90_avg']

    # 汇总中心仓L7日均
    cdc_wh = origin[['wh_dept_id', 'cdc_wh_dept_id']].query("cdc_wh_dept_id > 0").drop_duplicates()
    cdc_seven_consume_avg = pd.merge(cdc_wh, origin[['wh_dept_id', 'goods_id', 'seven_consume_avg']],
                                     on=['wh_dept_id'], how='left') \
        .groupby(['cdc_wh_dept_id', 'goods_id'], as_index=False) \
        .agg({'seven_consume_avg': 'sum'})

    cdc_seven_consume_avg.columns = ['wh_dept_id', 'goods_id', 'cdc_seven_consume_avg']

    origin = pd.merge(origin, cdc_seven_consume_avg, on=['wh_dept_id', 'goods_id'], how='left')

    return origin


def tower_warn(df):
    """
    控制塔告警底表
    """
    df_rpt = pd.melt(df, id_vars=['dt', 'wh_name', 'wh_dept_id', 'goods_name', 'goods_id', 'large_class_name', 'large_class_code']
                     , value_vars=['warn_bu_cg', 'warn_duan_cang', 'warn_ji_ya', 'warn_cg_duo']
                     , var_name='warn_type'
                     , value_name='flag').query("flag == True")
    df_rpt[['warn_type_comment', 'warn_guide']] = df_rpt.apply(lambda row: get_warn_msg(row), axis=1, result_type='expand')
    save_hdfs(data=df_rpt, hdfs_path=RES_HDFS_PATH, file_name='result.parquet')


@log_wecom('控制塔-采购业务告警', P_TWO)
def run():
    df = load()
    # 同比消耗(T_1)
    df['consume_ratio_T_1'] = (df['consume'] - df['consume_t_1']) / df['consume_t_1']
    # 环比消耗(L7)
    df['consume_ratio_L_7'] = (df['consume'] - df['seven_consume_avg']) / df['seven_consume_avg']

    # 预计可用天数(T_1)
    df['avl_days_T_1'] = df['end_avl_cnt'] / df['consume']
    df.loc[df['large_class_name'].isin(['日耗品', '器具类', '工服类', '营销物料', '办公用品']), 'avl_days_T_1'] = df['end_avl_cnt'] / df[
        'consume_t_1']
    # 预计可用天数(L7)
    df['avl_days'] = (df['end_avl_cnt'] + df['allot_transit']) / df['seven_consume_avg']

    # 预计可用天数（需求预测）avl_days_alg
    df_avl_days_alg = get_avl_days(df[['wh_dept_id', 'goods_id', 'end_avl_cnt']])
    df = df.merge(df_avl_days_alg, on=['wh_dept_id', 'goods_id'], how='left')

    # 仓库可用天数INV+CG+FH+ALLOT
    df['avl_days_with_cg'] = (df['end_avl_cnt'] + df['cg_transit'] + df['fh_transit'] + df['allot_transit']) / df['seven_consume_avg']
    # 中心仓的可用库存天数
    df['avl_days_cdc'] = (df['end_avl_cnt'] + df['allot_transit']) / df['cdc_seven_consume_avg']
    # 标记告警
    df[['warn_bu_cg', 'warn_duan_cang', 'warn_ji_ya', 'warn_cg_duo', 'warn_bo_dong', 'is_warn']] = df.apply(
        lambda row: get_warn(row), axis=1, result_type='expand')
    df_valid = df.query("is_warn == True").reset_index(drop=True)
    # 生成控制塔告警底表
    tower_warn(df_valid)
    # Email告警
    send_email(df_valid)


def send_email(df: DataFrame) -> None:
    df_email = df[
        ['wh_name', 'wh_dept_id', 'goods_name', 'goods_id', 'large_class_name', 'result_level', 'vlt_alg', 'bp_alg', 'ro', 'ss_days', 'end_avl_cnt',
         'cg_transit', 'fh_transit', 'allot_transit', 'consume', 'consume_t_1', 'seven_consume_avg', 'consume_ratio_T_1', 'consume_ratio_L_7',
         'avl_days_T_1', 'avl_days', 'avl_days_with_trs_alg', 'avl_days_alg',
         'cg_hist_list', 'min_cg_date', 'dos_max', 'dos_std',
         'cg_full', 'avl_days_with_cg', 'avl_days_cdc',
         'warn_bu_cg', 'warn_duan_cang', 'warn_ji_ya', 'warn_cg_duo', 'warn_bo_dong']]

    # 同步至HDFS

    df_email.columns = ['仓库名称', '仓库ID', '货物名称', '货物ID', '货物大类', '货物级别', 'VLT', 'BP', 'RO', 'SS', '期末可用库存',
                        'CG在途', 'FH在途', '调拨在途', '消耗', 'T_1消耗', 'L7日均消耗', '消耗同比变化', '消耗环比变化',
                        '仓库库存预计可用天数(相比昨日消耗)', '仓库库存预计可用天数(相比L7日均)', '含在途仓库库存预计可用天数(需求预测)', '仓库库存预计可用天数(需求预测)',
                        '历史未完成CG列表', '下一批CG入仓日期', '最高库存天数', '标准库存天数',
                        '是否有未发货CG', '含CG在途预计可用天数', '中心仓可用天数', '需补充CG', '仓库断仓风险', '库存积压', 'CG过多', '消耗波动，需关注']

    if (df_email is not None) & (len(df_email) > 0):
        to_email_list = ['youwei.xue@lkcoffee.com', "scp@lkcoffee.com", 'yuqi.zhai@lkcoffee.com',
                         "yuliang.tang@lkcoffee.com", "jingyuan.liu@lkcoffee.com", "xudong.zhang01@lkcoffee.com",
                         'wenliang.yao@lkcoffee.com', 'zheshun.shi@lkcoffee.com', 'yunzhu.jia@lkcoffee.com',"yuhan.cui@lkcoffee.com"]
        Message.send_email(df=df_email, to_emails=to_email_list, header='控制塔-采购业务告警')


if __name__ == '__main__':
    run()
