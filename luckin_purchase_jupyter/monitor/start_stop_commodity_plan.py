from meta import *
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from config.config import *
from dim_info.dim_base import Dim
from utils.file_utils import *
from utils.msg_utils import *
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 30)
pd.set_option('mode.chained_assignment', None)


def get_plan():
    """
    获取有效上下市计划
    返回上市计划df，下市计划df，联系人邮箱list
    :return:
    """
    # 获取有效上市计划
    start_plan = spark.sql(f"""
        SELECT
            base.dt
            , base.plan_id
            , base.cmdty_id
            , cmdty_plan.cmdty_name
            , cmdty_plan.plan_name
            , ehr.name as create_user
            , ehr.email AS create_usr_email
            , base.launch_date
            , COALESCE(cmdty_plan.lasting_days, 0) AS top_days
            , cmdty_plan.top_popularity_sales AS top_sales
            , plan_type
            , shop_num
            -- 短期计划：计划下市日期-计划上市日期
            , IF(DATEDIFF(cmdty_plan.cmdty_plan_abolish_date, cmdty_plan.cmdty_plan_launch_date) < IFNULL(cmdty_plan.first_purchasing_cycle, 30),
                 DATEDIFF(cmdty_plan.cmdty_plan_abolish_date, cmdty_plan.cmdty_plan_launch_date),
                 IFNULL(cmdty_plan.first_purchasing_cycle, 30)) - IFNULL(cmdty_plan.lasting_days, 0) AS norm_days
             , cmdty_plan.estimate_sales AS norm_sales
            , TO_DATE(cmdty_plan.cmdty_plan_abolish_date) AS abolish_date
        FROM
            (SELECT
                 shop_plan.dt
                 , shop_plan.launch_plan_id AS plan_id
                 , shop_plan.cmdty_id
                 , COUNT(shop_plan.dept_id) AS shop_num
                 -- 商品回归情况兜底：实际上市日期与计划上市日期不在同一年的情况
                 , shop_plan.launch_date

             FROM
                 dw_ads_scm_alg.dm_purchase_cmdty_plan_time shop_plan
             WHERE
             -- 商品回归情况兜底/取上市日期大于当日且小于未来120天的计划
                 ((shop_plan.launch_date >= '{today}') AND (shop_plan.launch_date <= DATE_ADD('{today}', 120)))
                 AND (shop_plan.launch_plan_id IS NOT NULL)
                 AND shop_plan.dt = '{yesterday}'
             GROUP BY shop_plan.dt, shop_plan.launch_plan_id, shop_plan.cmdty_id
             ) base
            LEFT JOIN (
            SELECT 
            dt, lasting_days,top_popularity_sales
            from
            dw_dim.`dim_cmdty_launch_plan_d_his`
            ) cmdty_plan
                      ON cmdty_plan.plan_id = base.plan_id AND cmdty_plan.cmdty_id = base.cmdty_id AND cmdty_plan.dt = base.dt
            LEFT JOIN lucky_ehr.`t_ehr_employee` ehr
                      ON cmdty_plan.create_user = ehr.id
            WHERE income_confirm_arrival_shop_date IS NULL

    """)
    start_plan[['abolish_date', 'launch_date', 'top_sales', 'norm_sales']] = start_plan[['abolish_date', 'launch_date', 'top_sales', 'norm_sales']].fillna('-')
    start_plan['type_name'] = start_plan.apply(lambda x: f'短期计划' if x.plan_type == 1 else f'上市计划' if x.plan_type == 2 else '-', axis=1)
    start_plan['days_remain'] = start_plan.apply(lambda x: (pd.to_datetime(x.launch_date) - pd.Timestamp(today)).days, axis=1)
    start_plan = start_plan.sort_values(['days_remain'])
    # 产品更改负责人:陈超-->刘宇捷
    start_plan['create_user'] = start_plan['create_user'].apply(lambda x : '刘宇捷' if x=='陈超' else x)
    start_plan['create_usr_email'] = start_plan['create_usr_email'].apply(lambda x : 'yujie.liu02@lkcoffee.com' if x=='chao.chen20@lkcoffee.com' else x)

    start_plan_push = start_plan[['plan_name', 'cmdty_name', 'type_name', 'launch_date', 'abolish_date', 'top_days', 'top_sales', 'norm_days', 'norm_sales', 'shop_num', 'create_user', 'days_remain']]
    start_plan_push['if_urgent'] = start_plan_push['type_name'] = start_plan_push.apply(lambda x: 2 if x.days_remain < 30 else 1 if x.days_remain < 45 else 0, axis=1)
    start_plan_push['type'] = '上市计划监控'

    # 获取有效下市计划
    stop_plan = spark.sql(f'''
        SELECT
            t1_1.cmdty_id
            , t1_4.plan_type
            , MAX(t1_1.abolish_date) AS abolish_date
            , abolish_plan_id
            , plan_name
            , cmdty_name
            , t1_1.dt
            , ehr.name AS create_user
            , ehr.email AS create_usr_email
        FROM
            dw_ads_scm_alg.dm_purchase_cmdty_plan_time t1_1
            LEFT JOIN (
            SELECT * FROM dw_dim.dim_cmdty_launch_plan_d_his WHERE dt = '{yesterday}'
            UNION
            SELECT * FROM dw_dim.dim_cmdty_launch_plan_d_his WHERE dt = '{yesterday}'

            ) t1_4
                      ON (t1_1.abolish_plan_id = t1_4.plan_id AND t1_1.cmdty_id = t1_4.cmdty_id)
             LEFT JOIN lucky_ehr.`t_ehr_employee` ehr
                              ON t1_4.create_user = ehr.id
        WHERE
            t1_1.dt = '{yesterday}'
            AND t1_1.launch_date < '{today}'
            AND t1_1.abolish_date > DATE_SUB('{today}', 30)
            AND t1_1.abolish_date <= DATE_ADD('{today}', 120)
        GROUP BY
            t1_1.cmdty_id, cmdty_name, plan_name, t1_1.dt, abolish_plan_id, t1_4.plan_type, ehr.name, ehr.email
        ''')
    stop_plan['type_name'] = stop_plan.apply(lambda x: '【下市计划缺失】-短期计划中计划下市时间' if x.plan_type == 1 else '【下市计划缺失】-上市计划中计划下市时间' if x.plan_type == 2 else '到期统一下市计划' if x.plan_type == 5 else '自然消耗下市计划' if x.plan_type == 6 else '-', axis=1)
    stop_plan['days_remain'] = stop_plan.apply(lambda x: (pd.to_datetime(x.abolish_date) - pd.Timestamp(today)).days, axis=1)
    stop_plan = stop_plan.sort_values('days_remain', ascending=1)
    stop_plan['if_urgent'] = stop_plan.apply(lambda x: 2 if abs(x.days_remain) < 30 else 1 if x.days_remain < 45 else 0, axis=1)
    # 产品更改负责人:陈超-->刘宇捷
    stop_plan['create_user'] = stop_plan['create_user'].apply(lambda x : '刘宇捷' if x=='陈超' else x)
    stop_plan['create_usr_email'] = stop_plan['create_usr_email'].apply(lambda x : 'yujie.liu02@lkcoffee.com' if x=='chao.chen20@lkcoffee.com' else x)

    stop_plan_push = stop_plan[['plan_name', 'cmdty_name', 'type_name', 'abolish_date', 'create_user', 'days_remain', 'if_urgent']]
    stop_plan_push['type'] = '下市计划监控'

    # 构造发送邮箱
    emp_email_ls = list(set(start_plan.create_usr_email.tolist() + stop_plan.create_usr_email.tolist()))

    return start_plan_push, stop_plan_push, emp_email_ls

def constructImage(df_plot, sort_name, title, plot_column_names, columnwidth):
    """
    构造图片类型表格
    :param df_plot: 画图dataframe
    :param sort_name: str 排序列
    :param title: 图片标题
    :param plot_column_names: list 图片表头名称
    :param columnwidth: 列宽
    :return:
    """
    plot_column_names = [f'<b>{x}</b>' for x in plot_column_names]
    # 构造颜色
    # color_pastel1 =  plt.cm.Pastel1(np.linspace(0,len(missing_gs_ls),len(missing_gs_ls)+1))
    color_pastel1 = px.colors.qualitative.Pastel1
    missing_gs_ls = df_plot[sort_name].drop_duplicates().to_list()
    df_plot['color'] = color_pastel1[len(missing_gs_ls)]
    for i in range(len(missing_gs_ls)):
        df_plot.loc[df_plot[sort_name] == missing_gs_ls[i], 'color'] = color_pastel1[np.random.randint(0, 5, 1)[0]]
    plot_colors = [df_plot['color'].tolist()]
    df_plot = df_plot.drop(sort_name, axis=1)
    fig = go.Figure(data=[go.Table(
        columnwidth=columnwidth,
        header=dict(
            values=[[x] for x in plot_column_names],
            line_color='darkslategray',
            fill_color='royalblue',
            align=['center', 'center', 'center', 'center', 'center'],
            font=dict(color='honeydew', size=12),
            height=30
        ),
        cells=dict(
            values=[df_plot[x].tolist() for x in df_plot.columns[:-1]],
            line_color='darkslategray',
            fill=dict(color=plot_colors),
            align=['center', 'center', 'left', 'center', 'left'],
            font_size=10,
            height=20)
    ),
    ])
    fig.update_layout(title=f'<b>{today}_{title}</b>', width=1000, height=max(((len(df_plot) + 6) * 30), 300), margin={'l': 5, 'r': 5, 't': 50, 'b': 10})
    return fig


if __name__ == '__main__':
    start_plan_push, stop_plan_push, emp_email_ls = get_plan()
    # 画图-推送
    start_title = '上市计划监控【今天-未来120天】'
    start_col_name = ['监控类型', '计划名称', '商品名称', '计划类型', '计划上市时间', '计划下市时间', 'top天数', 'top杯量', '非top天数', '非top杯量', '负责人', '距上市时间']
    stop_title = '下市计划监控【过去30天-未来120天】'
    stop_col_name = ['监控类型', '计划名称', '商品名称', '计划类型', '计划下市时间', '负责人', '距下市时间']
    # 保存表格地址
    data_path = f'/data/purchase/monitor/commodity_plan/commodity_plan_{today}.xlsx'
    folder_check(data_path)

    if len(start_plan_push) > 0:
        start_plan_plot = constructImage(start_plan_push[['type', 'plan_name', 'cmdty_name', 'type_name', 'launch_date', 'abolish_date',
                                                          'top_days', 'top_sales', 'norm_days', 'norm_sales',
                                                          'create_user', 'days_remain', 'if_urgent']], 'if_urgent', start_title, start_col_name, [30, 80, 60, 30, 32, 32, 20, 20, 25, 25, 25, 25, 25])
        Message.send_image(start_plan_plot, MSG_GROUP_ONE)
    if (len(stop_plan_push) > 0) and (len(stop_plan_push) < 50):
        stop_plan_plot = constructImage(stop_plan_push[['type', 'plan_name', 'cmdty_name', 'type_name', 'abolish_date', 'create_user',
                                                        'days_remain', 'if_urgent']], 'if_urgent', stop_title, stop_col_name, [30, 50, 30, 50, 20, 20, 20])
        Message.send_image(stop_plan_plot, MSG_GROUP_ONE)
    #     长度过长分成两个图片
    if len(stop_plan_push) > 50:
        stop_plan_plot1 = constructImage(stop_plan_push[0:50][['type', 'plan_name', 'cmdty_name', 'type_name', 'abolish_date', 'create_user',
                                                        'days_remain', 'if_urgent']], 'if_urgent', stop_title, stop_col_name, [30, 50, 30, 50, 20, 20, 20])

        stop_plan_plot2 = constructImage(stop_plan_push[50:][['type', 'plan_name', 'cmdty_name', 'type_name', 'abolish_date', 'create_user',
                                                              'days_remain', 'if_urgent']], 'if_urgent', stop_title, stop_col_name, [30, 50, 30, 50, 20, 20, 20])
        Message.send_image(stop_plan_plot1, MSG_GROUP_ONE)
        Message.send_image(stop_plan_plot2, MSG_GROUP_ONE)
    else:
        pass
    if len(emp_email_ls) > 0:
        Message.send_msg('请确认计划状态、上下市时间是否准确或存在变更，并于本月20号前反馈', mention_list=Dim.get_wecom_ids(email=emp_email_ls), group=MSG_GROUP_ONE)

    if (len(start_plan_push) > 0) or (len(stop_plan_push) > 0):
        # 构造发送文件
        stop_plan_push = stop_plan_push[['type', 'plan_name', 'cmdty_name', 'type_name', 'abolish_date', 'create_user',
                                         'days_remain']]
        start_plan_push = start_plan_push[['type', 'plan_name', 'cmdty_name', 'type_name', 'launch_date', 'abolish_date',
                                           'top_days', 'top_sales', 'norm_days', 'norm_sales',
                                           'create_user', 'days_remain']]
        start_plan_push.columns = start_col_name
        stop_plan_push.columns = stop_col_name

        df2Excel(data_path, [start_plan_push, stop_plan_push], [start_title, stop_title])
        commodity_msg = '附件为本月采购需求预测算法中所包含上下市计划基本信息，辛苦各位确认计划状态、上下市时间是否准确或存在变更，并及时反馈，感谢！'
        Message._Message__construct_email(header='上下市计划监控', msg=commodity_msg, cc_receiver=[], receiver=["sc-scp@lkcoffee.com", "yunzhong.li@lkcoffee.com", "yanxin.lu@lkcoffee.com", "yuhan.cui@lkcoffee.com"] + emp_email_ls, accessory_path=data_path, accessory_name='上下市计划监控.xlsx')