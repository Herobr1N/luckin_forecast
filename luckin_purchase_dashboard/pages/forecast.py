import dash
from dash import html, dcc, Input, Output, callback
import plotly.express as px
import pandas as pd
from utils import today, datetime, timedelta, logger
import data_syc as ds

dash.register_page(__name__, path='/')

def layout_forecast():
    return html.Div([
        html.H1(children=f'预测看板'),
        dcc.DatePickerSingle(
            id='date_picker',
            display_format='YYYY-MM-DD',
            min_date_allowed=datetime.today().date() - timedelta(180),
            max_date_allowed=datetime.today().date(),
            initial_visible_month=datetime.today().date(),
            date=datetime.today().date()),
        dcc.Loading(
            id="loading_ui",
            type="default",
            children=html.Button('Load', id='btn_load'),
            fullscreen=True
        ),
        html.Div([
            dcc.DatePickerRange(
                id='dt_pick_forecast',
                start_date=datetime.today().date(),
                end_date=datetime.today().date() + timedelta(days=119),
                display_format='YYYY-MM-DD',
            ),
            dcc.Dropdown(
                options=ds.goods_list
                , id='goods_id'
                , value=354
            ),
            dcc.Dropdown(
                ds.wh_list
                , id='wh_dept_id'
                , value=[4001]
                , multi=True
            )
        ]
        ),

        dcc.Graph(
            id='fig_forecast'
        ),

        dcc.Graph(
            id='fig_new'
        ),
        dcc.Graph(
            id='fig_factor'
        ),

        dcc.Graph(
            id='fig_plan_detail'
        ),

        dcc.Graph(
            id='fig_demand_detail'
        )]
    )


layout = layout_forecast


@callback(Output("dt_pick_forecast", "start_date"),
          Output("dt_pick_forecast", "end_date"),
          Output("btn_load", "n_clicks"),
          Input('btn_load', 'n_clicks'),
          Input('date_picker', 'date'))
def btn(n_clicks, date):
    dt = datetime.today().date()
    if n_clicks:
        dt = datetime.strptime(date, '%Y-%m-%d').date()
        print(dt)
        ds.load(dt)
    return dt, dt + timedelta(days=119), n_clicks


@callback(
    Output('fig_forecast', 'figure'),
    Input('goods_id', 'value'),
    Input('wh_dept_id', 'value'),
    Input('dt_pick_forecast', 'start_date'),
    Input('dt_pick_forecast', 'end_date'))
def update_fig_forecast(goods_id, wh_dept_id, start_date, end_date):
    logger.info("Visited")
    wh_dept_id_after = wh_dept_id if len(wh_dept_id) > 0 else 'wh_dept_id'
    # 最终预测
    dff = ds.df_forecast_all.query(
        f"goods_id == {goods_id} and (wh_dept_id == {wh_dept_id_after}) and (predict_dt >= '{start_date}') and (predict_dt <= '{end_date}')") \
        .groupby(['predict_dt', 'goods_id', 'type'], as_index=False) \
        .agg({'demand': 'sum'})
    fig_all = px.bar(dff
                     , x='predict_dt'
                     , y='demand'
                     , color='type'
                     , title='总需求预测分布')
    return fig_all


@callback(
    Output('fig_new', 'figure'),
    Input('goods_id', 'value'),
    Input('wh_dept_id', 'value'),
    Input('dt_pick_forecast', 'start_date'),
    Input('dt_pick_forecast', 'end_date'))
def update_fig_new(goods_id, wh_dept_id, start_date, end_date):
    wh_dept_id_after = wh_dept_id if len(wh_dept_id) > 0 else 'wh_dept_id'
    """
       新品&次新品
    """
    fig_new = px.bar()
    df_goods = ds.df_new_subnew.query(
        f"goods_id == {goods_id} and (wh_dept_id == {wh_dept_id_after}) and (predict_dt >= '{start_date}') and (predict_dt <= '{end_date}')") \
        .groupby(['predict_dt', 'cmdty_name', 'type'], as_index=False) \
        .agg({"demand": 'sum'})
    if len(df_goods) > 0:
        fig_new = px.bar(df_goods
                         , x='predict_dt'
                         , y='demand'
                         , color='cmdty_name'
                         , facet_row='type'
                         , height=600
                         , title='新品&次新品需求预测分布')
        fig_new.update_yaxes(matches=None)
    return fig_new


@callback(
    Output('fig_factor', 'figure'),
    Output('fig_plan_detail', 'figure'),
    Input('goods_id', 'value'),
    Input('wh_dept_id', 'value'),
    Input('dt_pick_forecast', 'start_date'),
    Input('dt_pick_forecast', 'end_date'))
def update_fig_factor(goods_id, wh_dept_id, start_date, end_date):
    wh_dept_id_after = wh_dept_id if len(wh_dept_id) > 0 else 'wh_dept_id'
    """
        因子系数
    """
    # 季节因子
    df_sea_res = ds.df_sea.query(f"goods_id == {goods_id} and (wh_dept_id == {wh_dept_id_after}) and (dt >= '{start_date}') and (dt <= '{end_date}')") \
        .groupby(['dt', 'goods_id'])[['goods_adjust_adjust', 'goods_cost_total']].sum().reset_index()
    df_sea_res['ratio_sea'] = df_sea_res['goods_adjust_adjust'] / df_sea_res['goods_cost_total']
    # 商品上下市
    df_plan_res = ds.df_plan.query(f"goods_id == {goods_id} and (wh_dept_id == {wh_dept_id_after}) and (dt >= '{start_date}') and (dt <= '{end_date}')") \
        .groupby(['dt', 'goods_id'])[['norm_adjust', 'norm_org', 'subnew_adjust', 'subnew_org']].sum().reset_index()
    df_plan_res['org_total'] = df_plan_res['norm_org'] + df_plan_res['subnew_org']
    df_plan_res['ratio_norm'] = df_plan_res['norm_adjust'] / df_plan_res['norm_org']
    df_plan_res['ratio_subnew'] = 1 - (df_plan_res['subnew_org'] / df_plan_res['org_total'])
    df_plan_res['ratio_plan'] = (df_plan_res['norm_adjust'] + df_plan_res['subnew_adjust']) / df_plan_res['org_total']

    df_factor = pd.merge(df_sea_res[['dt', 'goods_id', 'ratio_sea']],
                         df_plan_res[['dt', 'goods_id', 'ratio_norm', 'ratio_subnew', 'ratio_plan']],
                         on=['dt', 'goods_id'], how='outer') \
        .fillna(1)

    df_replace_res = \
        ds.df_replace.query(f"goods_id == {goods_id} and (wh_dept_id == {wh_dept_id_after}) and (dt >= '{start_date}') and (dt <= '{end_date}')") \
            .groupby(['dt', 'goods_id'])['dmd_daily'].sum().reset_index()

    df_ratio = df_replace_res.merge(df_factor[['dt', 'goods_id', 'ratio_sea', 'ratio_norm', 'ratio_subnew', 'ratio_plan']],
                                    on=['dt', 'goods_id'],
                                    how='left') \
        .rename(columns={'dmd_daily': 'dmd_origin'}) \
        .sort_values(by='dt')
    df_ratio['dmd_cmdty_plan'] = df_ratio['dmd_origin'] * df_ratio['ratio_plan']
    df_ratio['dmd_sea'] = df_ratio['dmd_origin'] * df_ratio['ratio_sea']
    df_ratio['dmd_final'] = df_ratio['dmd_origin'] * df_ratio['ratio_sea'] * df_ratio['ratio_plan']

    fig_factor = px.line(df_ratio
                         , x='dt'
                         , y=['dmd_origin', 'dmd_sea', 'dmd_cmdty_plan', 'dmd_final']
                         , title=f'季节&上下市因子')

    """
        上下市详情因子
    """
    fig_plan_detail = px.bar()
    cmdty_goods = ds.df_plan_detail.query(
        f"goods_id == {goods_id} and (wh_dept_id == {wh_dept_id_after}) and (dt >= '{start_date}') and (dt <= '{end_date}')") \
        .groupby(['dt', 'cmdty_name', 'goods_id']) \
        .sum().reset_index()
    if len(cmdty_goods) > 0:
        plan_detail = cmdty_goods.merge(df_plan_res,
                                        on=['dt', 'goods_id'],
                                        how='left',
                                        suffixes=['_cmdty', '_goods']) \
            .fillna(0) \
            .sort_values(by='dt')
        plan_detail['adjust_num_cmdty'] = plan_detail['subnew_org_cmdty'] - plan_detail['subnew_adjust_cmdty'] + plan_detail['norm_org_cmdty'] - \
                                          plan_detail['norm_adjust_cmdty']
        plan_detail['adjust_num_goods'] = plan_detail['subnew_org_goods'] - plan_detail['subnew_adjust_goods'] + plan_detail['norm_org_goods'] - \
                                          plan_detail['norm_adjust_goods']
        plan_detail['adjust_ratio'] = plan_detail['adjust_num_cmdty'] / plan_detail['adjust_num_goods']
        plan_detail = plan_detail.query("(adjust_ratio > 0)")
        if len(plan_detail) > 0:
            fig_plan_detail = px.bar(plan_detail, x='dt', y='adjust_ratio', color='cmdty_name', title='上下市因子分布')
            fig_plan_detail_line = px.line(plan_detail, x='dt', y='ratio_plan')
            fig_plan_detail_line.data[0]['line']['color'] = 'black'
            fig_plan_detail.add_traces(fig_plan_detail_line.data)

    return fig_factor, fig_plan_detail


@callback(
    Output('fig_demand_detail', 'figure'),
    Input('goods_id', 'value'),
    Input('wh_dept_id', 'value'),
    Input('dt_pick_forecast', 'start_date'),
    Input('dt_pick_forecast', 'end_date'))
def update_fig_dmd_detail(goods_id, wh_dept_id, start_date, end_date):
    """
        门店增长&节假日&新店拓展因子
    """
    wh_dept_id_after = wh_dept_id if len(wh_dept_id) > 0 else 'wh_dept_id'
    fig_demand_detail = px.line()
    detail_cols = ['origin_dmd_cost', 'origin_dmd_dly', 'dmd_daily_base', 'dmd_daily_shop_inc', 'dmd_daily_festival', 'dmd_new_shop_daily',
                   'dmd_daily']
    demand_detail = \
        ds.df_demand_detail.query(f"goods_id == {goods_id} and (wh_dept_id == {wh_dept_id_after}) and (dt >= '{start_date}') and (dt <= '{end_date}')") \
            .groupby(['dt', 'goods_id'])[detail_cols] \
            .sum().reset_index()
    if len(demand_detail) > 0:
        fig_demand_detail = px.line(demand_detail, x='dt', y=detail_cols, title='门店增长&节假日&新店因子(常规品)')

    return fig_demand_detail
