from utils.decorator import *
from utils.msg_utils import Message
from config.config import *


@log_wecom('DM自动CG配置监控', P_TWO)
def goods_spec_check():
    """
    DM系统CG配置表
    1.触发时间：每天下午17.30
    2.逻辑：
    ①对于重复配置或者配置有误的需提醒（保留当前逻辑）
    ②新增逻辑：对CG配置表中的货物&采购系统中【采购-关注】的货物进行校对，若存在采购系统中为【采购-关注】，但DM中未配置的需提醒
    1.剔除兰州仓；
    2.剔除上海设备仓；
    3.剔除轻食大类的半成品；
    4.剔除DM当前不支持推CG的大类（也就是说只看原料、轻食、包材、日耗这几类，其他的办公用品、器具这些目前还没有上DM推CG）
    """
    df = spark.sql(f"""
        SELECT
            t1.wh_dept_id AS 仓库部门ID
            , t1.wh_name  AS 仓库名称
            , t1.goods_id AS 货物ID
            , t1.goods_name AS 货物名称
            , t1.reason AS 异常原因
            , t2.large_class_name AS 货物大类
            , t1.dt AS 日期
        FROM dw_ads_scm_alg.dim_automatic_order_cfg_exp t1
        LEFT JOIN
             (SELECT 
                goods_id
                , large_class_name
                , large_class_id
                , small_class_name
              FROM dw_dim.dim_stock_good_d_his
              WHERE dt = '{yesterday}') t2
             ON (t1.goods_id = t2.goods_id)
        WHERE t1.dt = '{today}' 
            AND t1.wh_dept_id <> 329233
            AND t2.small_class_name <> '半成品'
            AND large_class_id IN (3, 4, 6, 29)
            AND reason <> '正常'
        ORDER BY t1.reason,
                 t2.large_class_name,
                 t1.wh_dept_id,
                 t1.goods_id
    """)
    if (df is not None) & (len(df) > 0):
        to_email_list = ["yuliang.tang@luckincoffee.com", "jingyuan.liu@luckincoffee.com", "xudong.zhang01@luckincoffee.com"]
        Message.send_email(df=df, to_emails=to_email_list, header='DM自动CG配置异常')


if __name__ == '__main__':
    goods_spec_check()