from config.config import *
from utils.msg_utils import Message, MSG_GROUP_THREE
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from pandas import DataFrame
import numpy as np
from dim_info.dim_base import Dim
from utils.decorator import *

BASE = f'/user/haoxuan.zou/demand_predict/{today}/demand_forecast_daily/'
INPUT_BC = BASE + 'demand_bc'
INPUT_YL = BASE + 'demand_yl'
INPUT_QS = BASE + 'demand_qs'
INPUT_LS = BASE + 'demand_ls'


def draw(df_table: DataFrame) -> Figure:
    headers = ['仓库名称', '货物名称', '货物ID', '货物大类', '货物级别','原始消耗模型', '消耗模型', '原始配送模型', '配送模型', '最终预测']
    # 根据货物级别标记颜色
    df_table['level_color'] = 'rgba(249, 250, 253, 1)'
    colors = px.colors.qualitative.Pastel1
    df_table.loc[df_table['result_level'] == '一级', 'level_color'] = colors[0]
    df_table.loc[df_table['result_level'] == '二级', 'level_color'] = colors[1]
    df_table.loc[df_table['result_level'] == '三级', 'level_color'] = colors[2]
    headers = [f'<b>{x}</b>' for x in headers]
    column_values = [df_table[col] for col in df_table.columns[:-1]]

    # 货物级别颜色
    column_color = []
    for index in range(len(headers)):
        if index == 4:
            column_color.append(df_table['level_color'].tolist())
        else:
            column_color.append(['rgba(249, 250, 253, 1)'])

    fig = go.Figure(data=[go.Table(
        columnwidth=[30, 80, 20, 25, 25, 35, 35, 35, 35, 35],
        header=dict(values=headers,
                    fill_color='rgba(89, 93, 98, 1)',
                    font=dict(color='white', size=13),
                    align='center'),
        cells=dict(values=column_values,
                   fill_color='rgba(249, 250, 253, 1)',
                   font=dict(color='black', size=13),
                   height=32,
                   align='center',
                   )
    )])
    fig.data[0]['cells']['fill']['color'] = column_color
    fig.update_layout(title=f'<b>{today} 销量预测基础数据异常监控</b>', width=1000, height=((len(df_table) + 2) * 50))
    return fig


@log_wecom('销量预测基础监控', P_TWO)
def run():
    columns = ['dt', 'wh_dept_id', 'goods_id', 'origin_dmd_cost', 'dmd_cost_filter', 'origin_dmd_dly', 'dmd_dly_filter', 'is_normal', 'dmd_daily_base']
    df_bc = spark.read_parquet(INPUT_BC)[columns]
    df_qs = spark.read_parquet(INPUT_QS)[columns]
    df_yl = spark.read_parquet(INPUT_YL)[columns]
    df_ls = spark.read_parquet(INPUT_LS)[columns]
    df_base = pd.concat([df_bc, df_qs, df_yl, df_ls]).query(f"(dt == '{today}') and (is_normal == 0)")
    df_base['goods_id'] = df_base['goods_id'].astype('int')
    df_base[['origin_dmd_cost', 'dmd_cost_filter', 'origin_dmd_dly', 'dmd_dly_filter', 'dmd_daily_base']] = np.round(df_base[['origin_dmd_cost', 'dmd_cost_filter', 'origin_dmd_dly', 'dmd_dly_filter', 'dmd_daily_base']], 2)
    df_wh = pd.merge(df_base, Dim.dim_warehouse('wh_dept_id', 'wh_name'), on='wh_dept_id', how='left')
    df_all = pd.merge(df_wh, Dim.dim_goods(), on='goods_id', how='left')
    df_valid = pd.merge(df_all, Dim.get_valid_goods('goods_id'), on='goods_id', how='inner')
    df = pd.merge(df_valid, Dim.get_band('goods_id', 'result_level'), on='goods_id', how='left')\
        .query("(small_class_name != '半成品') and (result_level != '不关注') and (origin_dmd_cost + dmd_cost_filter > 0)") \
        .sort_values(by=['result_level', 'goods_name'])

    df_table = df[['wh_name', 'goods_name', 'goods_id', 'large_class_name', 'result_level', 'origin_dmd_cost', 'dmd_cost_filter', 'origin_dmd_dly', 'dmd_dly_filter', 'dmd_daily_base']]
    df_table.loc[df_table['result_level'] == '一级', 'level'] = 1
    df_table.loc[df_table['result_level'] == '二级', 'level'] = 2
    df_table.loc[df_table['result_level'] == '三级', 'level'] = 3
    df_table = df_table.sort_values(by=['level', 'goods_name']).drop(columns='level')

    fig = draw(df_table)
    Message.send_image(fig, group=MSG_GROUP_THREE)


if __name__ == '__main__':
    run()
