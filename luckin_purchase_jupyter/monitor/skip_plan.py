from meta import *
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from config.config import *
from dim_info.dim_base import Dim
from utils.file_utils import *
from utils.msg_utils import *


def get_skip_plan(path):
    replace_plan = spark.read_csv(path)
    skip_plan = replace_plan.query("(type_flg=='norm') and (main_cmdty_id!=cmdty_id)")[
        ['cmdty_id', 'main_cmdty_id', 'dept_id', 'city_id', 'abolish_date']]
    skip_plan.abolish_date = pd.to_datetime(skip_plan.abolish_date, format='%Y-%m-%d')
    skip_plan_agg = skip_plan.groupby(['cmdty_id', 'main_cmdty_id'], as_index=False).agg(
        {'dept_id': 'count', 'abolish_date': 'mean'})
    skip_plan_agg = skip_plan_agg.rename(columns={'dept_id': 'dept_num', 'city_id': 'city_num'})
    skip_plan_agg.abolish_date = skip_plan_agg.abolish_date.dt.date
    return skip_plan_agg


def get_skip_notice():
    """
    业务产品过滤部分下市计划
    :return:
    """
    yesterday_data = get_skip_plan(
        f"/user/haoxuan.zou/demand_predict/{yesterday}/commodity_plan/commodity_plan_replace")
    yesterday_data['flg'] = 1
    today_data = get_skip_plan(f"/user/haoxuan.zou/demand_predict/{today}/commodity_plan/commodity_plan_replace")
    today_data_check = today_data.merge(yesterday_data, on=['cmdty_id', 'main_cmdty_id'], how='left',
                                        suffixes=['_today', '_yesterday'])
    today_data_check['dept_diff'] = (today_data_check['dept_num_today'] - today_data_check['dept_num_yesterday']) / \
                                    today_data_check['dept_num_yesterday']
    today_data_check['flg'] = today_data_check['flg'].fillna(0)
    notice = today_data_check.query("flg==0")

    if len(notice) > 0:
        cmdty_info = spark.sql(f"""
                SELECT
                    cmdty_id
                    , cmdty_name
                    , CASE status WHEN 0 THEN '未完成配置'
                    WHEN 1 THEN '完成配置'
                    WHEN 2 THEN '已上线'
                    WHEN 3 THEN '已下线'
                    WHEN -1 THEN '删除' END AS cmdty_status
                FROM dw_dim.dim_cmdty_d_his
                WHERE dt = '{yesterday}'
                --去除无效商品id
                AND cmdty_name NOT LIKE '%弃用%'
            """)
        cmdty_info.cmdty_id = cmdty_info.cmdty_id.astype('str')
        notice = notice \
            .merge(cmdty_info[['cmdty_id', 'cmdty_name']], on=['cmdty_id'], how='left') \
            .merge(cmdty_info[['cmdty_id', 'cmdty_name']].rename(
            columns={'cmdty_id': 'main_cmdty_id', 'cmdty_name': 'main_cmdty_name'}), on=['main_cmdty_id'], how='left')
        notice['reason'] = '新增替代主计划'
        notice = notice[['cmdty_name', 'main_cmdty_name', 'abolish_date', 'reason', 'dept_diff']]
    else:
        notice = pd.DataFrame()
    return notice


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
    color_pastel1 = px.colors.qualitative.Pastel1
    missing_gs_ls = df_plot[sort_name].drop_duplicates().to_list()
    df_plot['color'] = color_pastel1[len(missing_gs_ls)]
    for i in range(len(missing_gs_ls)):
        df_plot.loc[df_plot[sort_name] == missing_gs_ls[i], 'color'] = color_pastel1[np.random.randint(0, 5, 1)[0]]
    plot_colors = [df_plot['color'].tolist()]
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
    fig.update_layout(title=f'<b>{today}_{title}</b>', height=max(((len(df_plot) + 3) * 30), 300),
                      margin={'l': 5, 'r': 5, 't': 50, 'b': 10})
    return fig


def run():
    notice = get_skip_notice()
    if len(notice) > 0:
        pic = constructImage(notice, 'reason', 'skip_plan', notice.columns.tolist(), [10, 10, 5, 5])
        Message.send_image(pic, MSG_GROUP_TEST)
    else:
        Message.send_msg('no skip_plan added', MSG_GROUP_TEST)


if __name__ == '__main__':
    run()