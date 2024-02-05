import dash
from dash import Dash, html, dcc, Input, Output, callback
import plotly.express as px
import pandas as pd
from utils import today, datetime, timedelta, logger
from plotly.subplots import make_subplots
import data_syc as ds

dash.register_page(__name__)


def layout_forecast():
    return html.Div(
        children=[
            html.H1(children=f'消耗看板'),
            html.Div([
                dcc.DatePickerRange(
                    id='my-date-picker-range',
                    start_date=datetime.today().date() - timedelta(days=60),
                    end_date=datetime.today().date(),
                    display_format='YYYY-MM-DD',
                ),
                dcc.Dropdown(
                    options=ds.goods_list
                    , id='goods_id'
                    , value=354
                ),
                dcc.Dropdown(
                    options=ds.wh_list
                    , id='wh_dept_id'
                    , value=[4001]
                    , multi=True
                )
            ]
            ),

            dcc.Graph(
                id='fig_his_dis'
            ),

            dcc.Graph(
                id='fig_inv'
            ),

            dcc.Graph(
                id='fig_cost_all'
            )]
    )


layout = layout_forecast


@callback(
    Output('fig_his_dis', 'figure'),
    Input('goods_id', 'value'),
    Input('wh_dept_id', 'value'),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date'))
def update_fig_his(goods_id, wh_dept_id, start_date, end_date):
    logger.info("Visited")
    wh_dept_id_after = wh_dept_id if len(wh_dept_id) > 0 else 'wh_dept_id'
    """
        历史消耗分布
    """
    # 历史消耗
    fig_his_dis = px.bar()
    df_his_dis = ds.df_his_cost_detail.query(
        f"goods_id == {goods_id} and (wh_dept_id == {wh_dept_id_after}) and (dt >= '{start_date}') and (dt <= '{end_date}')") \
        .groupby(['dt', 'cmdty_name'], as_index=False) \
        .agg({'cmdty_cost': 'sum'})
    if len(df_his_dis) > 0:
        fig_his_dis = px.bar(df_his_dis, x='dt', y='cmdty_cost', color='cmdty_name', title='历史商品消耗占比分布')

    return fig_his_dis


@callback(
    Output('fig_inv', 'figure'),
    Output('fig_cost_all', 'figure'),
    Input('goods_id', 'value'),
    Input('wh_dept_id', 'value'),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date'))
def update_fig_dmd_inv(goods_id, wh_dept_id, start_date, end_date):
    """
        库存详情
    """
    wh_dept_id_after = wh_dept_id if len(wh_dept_id) > 0 else 'wh_dept_id'

    df_base = ds.df_inv.query(f"goods_id == {goods_id} and (wh_dept_id == {wh_dept_id_after}) and (dt >= '{start_date}') and (dt <= '{end_date}')") \
        .groupby(['dt', 'goods_id']) \
        .sum() \
        .reset_index() \
        .sort_values(by='dt')

    df_draw = pd.melt(df_base,
                      id_vars=['dt', 'goods_id'],
                      value_vars=['beg_avl_inv', 'wh_in_alloc_cnt', 'wh_pur_cnt', 'wh_out_alloc_cnt', 'wh_loss', 'dly_shop_new', 'dly_shop_regular'],
                      var_name='inv_type',
                      value_name='value')
    # 画图
    fig_inv = px.bar(df_draw,
                     x='dt',
                     y='value',
                     color='inv_type',
                     title='库存详情')
    # 实际消耗
    fig_cost = px.line(df_base, x='dt', y='shop_actual_consume')
    fig_cost.data[0]['line']['color'] = 'black'
    fig_cost.data[0]['name'] = 'actual_consume'
    fig_cost.data[0]['showlegend'] = True

    # 理论消耗
    fig_cost_theory = px.line(df_base, x='dt', y='shop_theory_consume')
    fig_cost_theory.data[0]['line']['color'] = 'green'
    fig_cost_theory.data[0]['name'] = 'theory_consume'
    fig_cost_theory.data[0]['showlegend'] = True

    fig_inv.add_traces(fig_cost.data)
    fig_inv.add_traces(fig_cost_theory.data)

    # 消耗对比
    df_cost = df_base.copy()
    cols = ['wh_dly_out', 'shop_actual_consume', 'shop_theory_consume']
    df_cost[cols] = - df_cost[cols]

    # 累计消耗
    cols_cum = [f'{x}_cum' for x in cols]
    df_cost[cols_cum] = df_cost.groupby(['wh_dept_id', 'goods_id'])[cols].cumsum()

    # MA14
    cols_avg_7 = [f'{x}_avg_7' for x in cols]
    df_cost[cols_avg_7] = df_cost.groupby(['wh_dept_id', 'goods_id'], as_index=False)[cols].transform(lambda x: x.rolling(7, 1).mean())

    # MA14
    cols_avg_14 = [f'{x}_avg_14' for x in cols]
    df_cost[cols_avg_14] = df_cost.groupby(['wh_dept_id', 'goods_id'], as_index=False)[cols].transform(lambda x: x.rolling(14, 1).mean())

    # MA21
    cols_avg_21 = [f'{x}_avg_21' for x in cols]
    df_cost[cols_avg_21] = df_cost.groupby(['wh_dept_id', 'goods_id'], as_index=False)[cols].transform(lambda x: x.rolling(21, 1).mean())

    fig_cost_all = make_subplots(rows=5, cols=1, subplot_titles=['出库 vs 实际消耗 vs 理论消耗', '累计消耗', 'MA_7', 'MA_14', 'MA_21'])
    fig_cost = px.line(df_cost, x='dt', y=cols)
    fig_cost_cum = px.line(df_cost, x='dt', y=cols_cum)
    fig_cost_avg_7 = px.line(df_cost, x='dt', y=cols_avg_7)
    fig_cost_avg_14 = px.line(df_cost, x='dt', y=cols_avg_14)
    fig_cost_avg_21 = px.line(df_cost, x='dt', y=cols_avg_21)

    fig_cost_all.add_traces(fig_cost.data, rows=1, cols=1)
    fig_cost_all.add_traces(fig_cost_cum.data, rows=2, cols=1)
    fig_cost_all.add_traces(fig_cost_avg_7.data, rows=3, cols=1)
    fig_cost_all.add_traces(fig_cost_avg_14.data, rows=4, cols=1)
    fig_cost_all.add_traces(fig_cost_avg_21.data, rows=5, cols=1)

    fig_cost_all.update_layout(height=1200, title='不同维度消耗详情')
    return fig_inv, fig_cost_all
