from config.config import spark
from dim_info.dim_base import Dim
from utils.msg_utils import Message


def email_send():
    df = spark.sql("""
        select
            t1.wh_dept_id,
            t1.wh_name,
            t1.goods_id,
            t1.goods_name,
            t1.reason,
            t2.large_class_name,
            t1.dt
        from
            dw_ads_scm_alg.dim_automatic_order_cfg_exp t1
        left join
            (select goods_id, large_class_name, large_class_id from dw_dim.dim_stock_good_d_his where dt = date_sub(current_date(), 1)) t2
            on (t1.goods_id = t2.goods_id)
        where t1.dt = current_date()
        order by
            t1.reason,
            t2.large_class_name,
            t1.wh_dept_id,
            t1.goods_id
    """)
    Message.send_email(df=df, to_emails=['yanxin.lu@lkcoffee.com', 'yuhan.cui@lkcoffee.com'], header='【测试】DM配置')


def wecom_send():
    Message.send_msg('我操作一下', Dim.get_wecom_ids(name="'余至诚', '崔与晗', '卢延新'"))


if __name__ == '__main__':
    wecom_send()