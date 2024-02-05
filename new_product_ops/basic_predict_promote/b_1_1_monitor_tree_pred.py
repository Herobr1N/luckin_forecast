# encoding: utf-8
# @created: 2023/11/30
# @author: jieqin.lin
# @file: projects/basic_predict_promote/b_1_1_monitor_tree_pred.py

"""
监控树模型的预测结果

1  监控++货物预测量对比++online++幼稚预测++过去历史真实++30天之和
2  监控++消耗突变(翻倍)
3  监控++统计模型管理器使用次数
4  监控++数据清洗++统计过滤次新品的个数
5  监控++趋势++过去历史真实120天++预测未来120天
6  监控++门店盘点记录++数据量是否有异常

生成数据

依赖数据
/user/haoxuan.zou/demand_predict/2023-12-01/demand_forecast_daily/demand_norm                       预测中间日志
bip3("model/basic_predict_promote_online", "stock_wh_goods_type_flg_theory_sale_cnt")               历史真实消耗数据
bip3("model/basic_predict_promote_online", "feature_engineering_wh_goods_normal")                   特征
bip3("model/basic_predict_promote_online", f"best_model_mix_boost_1_wh_goods_pred_normal_group_b")  最佳模型
"""

from __init__ import project_path
import os
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from z_config.cfg import DATA_OUTPUT_PATH
from areas.table_info.dh_dw_table_info import dh_dw
from projects.basic_predict_promote.b_0_0_utils_models import SelectGoodsList
from utils.a91_wx_work_send import qiyeweixin_image, qiyeweixin_bot
from utils_offline.a00_imports import log20 as log, DayStr, read_api, bip3, argv_date, bip2

f"Import from {project_path}"


def main_monitor_tree_pred(pred_calculation_day=None):
    # 监控++货物预测量对比++online++幼稚预测++过去历史真实++30天之和
    monitor_01_pred(pred_calculation_day)
    # 监控++消耗突变(翻倍)
    monitor_02_pred(pred_calculation_day)
    # 监控++统计模型管理器使用次数
    monitor_03_pred(pred_calculation_day)
    # 监控++数据清洗++统计过滤次新品的个数
    monitor_04_pred(pred_calculation_day)
    # 监控++趋势++过去历史真实120天++预测未来120天
    monitor_05_pred(pred_calculation_day, data_label='常规')
    monitor_05_pred(pred_calculation_day, data_label='总')
    # 监控++门店盘点记录++数据量是否有异常
    monitor_06_pred(pred_calculation_day)


def monitor_01_pred(pred_calculation_day=None):
    """
    监控++货物预测量对比++online++幼稚预测++过去历史真实++30天之和
    """
    days_to_include = 30
    pred_calc_day = DayStr.get_dt_str(pred_calculation_day)
    pred_minus1_day = DayStr.n_day_delta(pred_calc_day, n=-1)
    pred_minus30_day = DayStr.n_day_delta(pred_calc_day, n=-days_to_include)
    df_goods_info = dh_dw.dim_stock.goods_info()
    df_goods_info = df_goods_info[['goods_id', 'goods_name']].drop_duplicates()

    log.title(f"监控++货物预测量对比++online++幼稚预测++过去历史真实++30天之和 dt={pred_minus1_day} run at {DayStr.now2()}")

    # ------------------------
    # 长周期货物范围
    selector = SelectGoodsList()
    sel_goods_ls = selector.get_long_period_goods_id()

    # ----------------------
    # 货物范围 :{group_b,group_c,group_d,group_e,group_f}
    sel_all_goods_ls = selector.get_group_all_goods_ls()
    df_group_name = pd.DataFrame()
    for group, goods_list in sel_all_goods_ls.items():
        for goods_id in goods_list:
            temp_df = pd.DataFrame({'goods_id': [goods_id], 'group_name': [group]})
            df_group_name = pd.concat([df_group_name, temp_df], ignore_index=True)
    # ------------------------
    # 读取 采购代码中p_4_0_goods_demand_forecast_main 的中间日志

    DEMAND_BASE_PATH = f"/user/haoxuan.zou/demand_predict/{pred_calc_day}/"
    DEMAND_FORECAST_DAILY = DEMAND_BASE_PATH + "demand_forecast_daily/"
    dmd = read_api.read_one_folder(DEMAND_FORECAST_DAILY + "demand_norm")

    # 前30天 sum 预测++online
    dmd_sum = (dmd.sort_values(['dt'])
               .groupby(['wh_dept_id', 'goods_id'])
               .head(days_to_include)
               .groupby(['goods_id'])
               .agg({'dmd_daily': 'sum', 'dmd_daily_adj': 'sum'})
               .reset_index()
               )
    dmd_sum['goods_id'] = dmd_sum['goods_id'].astype(int)
    dmd_sum.query(f"goods_id in {sel_goods_ls}", inplace=True)

    # ------------------------
    # 过去30天历史真实
    df_stock_sell = read_api.read_dt_folder(
        bip3("model/basic_predict_promote_online", "stock_wh_goods_type_flg_theory_sale_cnt")
        , pred_minus30_day, pred_minus1_day)
    # 绝对值
    df_stock_sell["theory_sale_cnt"] = np.abs(df_stock_sell["theory_sale_cnt"])

    # ------------------------
    # sum 过去30天历史真实（常规品常规店 + 常规品新店）
    df_30 = (df_stock_sell.query(f"type_flg =='norm'")
             .groupby(['goods_id'])
             ['theory_sale_cnt'].sum()
             .rename('lag_30').reset_index()
             )

    # ------------------------
    # 幼稚预测
    df_1 = (df_stock_sell
            .query(f"dt =='{pred_minus1_day}' and type_flg =='norm'")
            .groupby(['goods_id'])
            ['theory_sale_cnt'].sum()
            .rename('lag_1').reset_index()
            )
    df_1['lag_1_sum'] = df_1.eval("lag_1 * 30")

    # ------------------------
    # merge
    dfc = (dmd_sum
           .merge(df_1)
           .merge(df_30)
           .merge(df_goods_info)
           .merge(df_group_name)
           )
    # 计算比例
    dfc['ratio'] = dfc.eval("dmd_daily/dmd_daily_adj").round(1)
    dfc['ratio_1'] = dfc.eval("dmd_daily/lag_1_sum").round(1)
    dfc['ratio_2'] = dfc.eval("dmd_daily/lag_30").round(1)
    # 排序
    dfc = dfc.sort_values(['ratio'], ascending=False)
    # 输出
    cols_output = ['goods_name','group_name', 'dmd_daily', 'dmd_daily_adj',
                   'lag_1_sum', 'lag_30', 'ratio', 'ratio_1', 'ratio_2']
    dfc = dfc[cols_output]

    if len(dfc):
        monitor_01_img(pred_calculation_day=pred_calculation_day, df_input=dfc,
                       is_send=True, send_to=300)


def monitor_01_img(pred_calculation_day=None, df_input=None, is_send=True, send_to=300):
    """
    监控++货物预测量对比++online++幼稚预测++过去历史真实++30天之和
    发生企业微信图片
    """
    pred_calc_day = DayStr.get_dt_str(pred_calculation_day)
    pred_minus1_day = DayStr.n_day_delta(pred_calc_day, n=-1)

    # 中文
    change_names_input = {'goods_name': '货物',
                          'group_name':'组',
                          'dmd_daily': '预测(tree)',
                          'dmd_daily_adj': '预测(online)',
                          'lag_1_sum': '预测(幼稚)',
                          'lag_30': '过去真实',
                          'ratio': 'tree/online',
                          'ratio_1': 'tree/幼稚',
                          'ratio_2': 'tree/真实'}

    df_input.rename(columns=change_names_input, inplace=True)

    def apply_color_img(value):
        """
        列的渐变填充，从深绿到浅绿
        """

        if value >= 2:
            color = '#ff9e81'  # 红色
        elif 0 <= value <= 0.5:
            color = 'rgb(204, 255, 204)'  # 深绿色
        else:
            color = '#ffffff'  # 白色

        return color

    # ------------------------
    # 画图
    df_plot = df_input.copy()

    s = len(df_plot)
    title_text = f'{pred_minus1_day}_30天预测之和对比_货物数{s}'
    n_rows = len(re.findall("<Br>", title_text))
    top_height = 35 + 20 * n_rows
    h = 100 + top_height + 35 * (s - 1)
    plot_column_names = [f'<b>{x}</b>' for x in df_plot.columns]
    plot_column_width = [25] * len(df_plot.columns)

    column_color_dict = {
        'tree/online': df_plot['tree/online'],
        'tree/幼稚': df_plot['tree/幼稚'],
        'tree/真实': df_plot['tree/真实']
    }

    column_color_ls = []
    for col in df_plot.columns:
        if col in column_color_dict:
            column_color_ls.append([apply_color_img(x) for x in column_color_dict[col]])
        else:
            column_color_ls.append(['#ffffff'] * len(df_plot))

    # Convert values to percentage format
    table_values = []
    for col in df_plot.columns:
        if col in ['货物','组']:
            table_values.append(df_plot[col].tolist())
        elif col in ['预测(tree)', '预测(online)', '预测(幼稚)', '过去真实', ]:
            table_values.append([str(round(x)) for x in df_plot[col].tolist()])
        else:
            table_values.append([f'{x:.0%}' for x in df_plot[col].tolist()])

    fig = go.Figure(data=[go.Table(
        columnwidth=plot_column_width,
        header=dict(
            values=plot_column_names,
            line_color='white',
            fill_color='rgb(104, 205, 104)',
            font=dict(color='black', size=12),
            height=40
        ),
        cells=dict(
            values=table_values,
            line_color='white',
            fill=dict(color=column_color_ls),
            font_size=12,
            height=35
        )
    )], layout=go.Layout(
        autosize=True,
        margin={'l': 5, 'r': 5, 't': top_height, 'b': 0},
        title=f"<b>{title_text}</b>",
    ))

    print("save")
    save_path = os.path.join(DATA_OUTPUT_PATH, f"images/base_pred/{title_text}.png")
    fig.write_image(save_path, width=sum(plot_column_width) * 6 + 50, height=h, scale=2, engine='kaleido')
    if is_send:
        qiyeweixin_image(save_path, send_to=send_to)


def monitor_02_pred(pred_calculation_day=None):
    """
    监控++消耗突变(翻倍)
    """
    pred_calc_day = DayStr.get_dt_str(pred_calculation_day)
    pred_minus1_day = DayStr.n_day_delta(pred_calc_day, n=-1)
    df_goods_info = dh_dw.dim_stock.goods_info()
    df_wh_info = dh_dw.dim_stock.wh_city_info()
    df_wh_info = df_wh_info[['wh_dept_id', 'wh_name']].drop_duplicates()
    df_goods_info = df_goods_info[['goods_id', 'goods_name']].drop_duplicates()
    log.title(f"监控++消耗突变(翻倍) dt={pred_minus1_day} run at {DayStr.now2()}")

    # ------------------------
    # 读取特征数据
    ld_feature = read_api.read_dt_folder(
        bip3("model/basic_predict_promote_online", "feature_engineering_wh_goods_normal"), pred_minus1_day)
    ld_feature.query(f"ds =='{pred_minus1_day}'", inplace=True)

    # 昨天跟前天比，跟7天均值比
    ld_feature['ratio_7'] = ld_feature.eval("y/lag_7_mean").round(1)
    ld_feature['ratio_1'] = ld_feature.eval("y/lag_1_mean").round(1)

    # 如果发生突变，突增大于等于2倍，则报警
    ld_feature['is_over'] = 0
    sel_idx = (ld_feature['ratio_1'] >= 2) | (ld_feature['ratio_7'] >= 2)
    ld_feature.loc[sel_idx, 'is_over'] = 1
    ld_feature.query("is_over==1", inplace=True)

    if len(ld_feature) > 0:
        df_feature = ld_feature.merge(df_wh_info).merge(df_goods_info)
        cols_output = ['wh_name', 'goods_name', 'y', 'lag_1_mean', 'lag_7_mean', 'ratio_1', 'ratio_7']
        df_feature = df_feature[cols_output]
        qiyeweixin_bot(msg="⚠️ 昨日消耗发生突变(翻倍)", send_to=300, mention_number_list=[18801057497])
        monitor_02_img(pred_calculation_day=pred_calculation_day, df_input=df_feature, is_send=True, send_to=300)
    else:
        qiyeweixin_bot(msg="🟢 昨日消耗无突变(翻倍)", send_to=300)


def monitor_02_img(pred_calculation_day=None, df_input=None, is_send=True, send_to=300):
    """
    监控++消耗突变(翻倍)
    发生企业微信图片
    """
    pred_calc_day = DayStr.get_dt_str(pred_calculation_day)
    pred_minus1_day = DayStr.n_day_delta(pred_calc_day, n=-1)

    # 中文
    change_names_input = {'wh_name': '仓库',
                          'goods_name': '货物',
                          'y': 'T消耗量',
                          'lag_1_mean': 'T-1消耗量',
                          'lag_7_mean': '7日均消耗量',
                          'ratio_1': 'T/T-1',
                          'ratio_7': 'T/7日均'}
    df_input.rename(columns=change_names_input, inplace=True)

    # ------------------------
    # 画图
    df_plot = df_input.copy()

    title_text = f'{pred_minus1_day}监控消耗突变(翻倍)'
    s = len(df_plot)
    n_rows = len(re.findall("<Br>", title_text))
    top_height = 35 + 20 * n_rows
    h = 100 + top_height + 35 * (s - 1)
    plot_column_names = [f'<b>{x}</b>' for x in df_plot.columns]
    plot_column_width = [25] * len(df_plot.columns)

    column_color_ls = []
    for col in df_plot.columns:
        column_color_ls.append(['#ffffff'] * len(df_plot))

    # Convert values to percentage format
    table_values = []
    for col in df_plot.columns:
        if col in ['仓库', '货物']:
            table_values.append(df_plot[col].tolist())
        elif col in ['T消耗量', 'T-1消耗量', '7日均消耗量']:
            table_values.append([str(round(x)) for x in df_plot[col].tolist()])
        else:
            table_values.append([f'{x:.0%}' for x in df_plot[col].tolist()])

    fig = go.Figure(data=[go.Table(
        columnwidth=plot_column_width,
        header=dict(
            values=plot_column_names,
            line_color='white',
            fill_color='rgb(104, 205, 104)',
            font=dict(color='black', size=12),
            height=40
        ),
        cells=dict(
            values=table_values,
            line_color='white',
            fill=dict(color=column_color_ls),
            font_size=12,
            height=35
        )
    )], layout=go.Layout(
        autosize=True,
        margin={'l': 5, 'r': 5, 't': top_height, 'b': 0},
        title=f"<b>{title_text}</b>",
    ))

    print("save")
    save_path = os.path.join(DATA_OUTPUT_PATH, f"images/base_pred/{title_text}.png")
    fig.write_image(save_path, width=sum(plot_column_width) * 6 + 50, height=h, scale=2, engine='kaleido')
    if is_send:
        qiyeweixin_image(save_path, send_to=send_to)


def monitor_03_pred(pred_calculation_day=None):
    """
    监控++统计模型管理器使用次数
    """
    pred_calc_day = DayStr.get_dt_str(pred_calculation_day)
    pred_minus1_day = DayStr.n_day_delta(pred_calc_day, n=-1)

    log.title(f"监控++统计模型管理器使用次数 dt={pred_minus1_day} run at {DayStr.now2()}")

    # ------------------------
    # 读取最佳模型
    data_label_ls = ['normal', 'new_shop_normal']
    goods_label_ls = ['group_b', 'group_c', 'group_e', 'group_f', 'group_g']

    df1 = pd.DataFrame()
    for data_label in data_label_ls:
        for goods_label in goods_label_ls:
            best_model = read_api.read_dt_folder(
                bip3("model/basic_predict_promote_online",
                     f"best_model_mix_boost_1_wh_goods_pred_{data_label}_{goods_label}")
                , pred_calc_day)
            best_model['data_label'] = data_label
            best_model['goods_label'] = goods_label
            df1 = pd.concat([df1, best_model])

    # ------------------------
    df1['MAPE'] = df1.eval("1 - last_1_acc")
    # mape准确度分级： [0，0.2, 0.4, 0.6, 0.8, 1]
    df1['pe20'] = df1['MAPE'].apply(lambda x: 1 if x <= 0.2 else 0)
    df1['pe40'] = df1['MAPE'].apply(lambda x: 1 if (x > 0.2 and x <= 0.4) else 0)
    df1['pe60'] = df1['MAPE'].apply(lambda x: 1 if (x > 0.4 and x <= 0.6) else 0)
    df1['pe80'] = df1['MAPE'].apply(lambda x: 1 if (x > 0.6 and x <= 0.8) else 0)
    df1['pe100'] = df1['MAPE'].apply(lambda x: 1 if (x > 0.8 and x <= 1) else 0)
    df1['pe_more'] = df1['MAPE'].apply(lambda x: 1 if x > 1 else 0)

    cols_agg = {'pe20': 'sum', 'pe40': 'sum', 'pe60': 'sum', 'pe80': 'sum', 'pe100': 'sum', 'pe_more': 'sum'}
    accuracy_levels = ['pe20', 'pe40', 'pe60', 'pe80', 'pe100', 'pe_more']

    df_result_all = (df1.groupby('model')
                     .agg(cols_agg).reset_index())

    # 总次数
    df_result_sum = (df1.groupby('model')
                     .size().rename("total").reset_index())
    df_result_all = df_result_all.merge(df_result_sum)

    # 占比 =  mape 在 20% 的次数/ 总次数
    for level in accuracy_levels:
        df_result_all[f'{level}_ratio'] = (df_result_all[level] / df_result_all['total']).round(4)
    df_result_all = df_result_all[['model', 'pe20_ratio', 'pe40_ratio', 'pe60_ratio', 'pe80_ratio', 'pe100_ratio']]

    # ------------------------
    df_best_model_cnt = (df1.groupby(['model'])['last_1_acc']
                         .agg(['size', 'mean', 'min'])
                         .round(3).reset_index()
                         )

    dfc = df_best_model_cnt.merge(df_result_all)

    if len(dfc):
        monitor_03_img(pred_calculation_day=pred_calculation_day, df_input=dfc,
                       is_send=True, send_to=300)


def monitor_03_img(pred_calculation_day=None, df_input=None, is_send=True, send_to=300):
    """
    监控++统计模型管理器使用次数
    发生企业微信图片
    """
    pred_calc_day = DayStr.get_dt_str(pred_calculation_day)

    # 中文
    change_names_input = {'model': '模型',
                          'size': '仓货使用次数',
                          'mean': 'ACC均值',
                          'min': 'ACC最小值',
                          'pe20_ratio': 'MAPE(0~0.2)',
                          'pe40_ratio': 'MAPE(0.2~0.4)',
                          'pe60_ratio': 'MAPE(0.4~0.6)',
                          'pe80_ratio': 'MAPE(0.6~0.8)',
                          'pe100_ratio': 'MAPE(0.8~1)'}
    df_input.rename(columns=change_names_input, inplace=True)

    # ------------------------
    # 画图
    df_plot = df_input.copy()

    title_text = f'统计模型管理器使用次数(常规品常规店+常规品新店)_{pred_calc_day}'
    s = len(df_plot)
    n_rows = len(re.findall("<Br>", title_text))
    top_height = 35 + 20 * n_rows
    h = 100 + top_height + 35 * (s - 1)
    plot_column_names = [f'<b>{x}</b>' for x in df_plot.columns]
    plot_column_width = [25] * len(df_plot.columns)

    column_color_ls = []
    for col in df_plot.columns:
        column_color_ls.append(['#ffffff'] * len(df_plot))

    # Convert values to percentage format
    table_values = []
    for col in df_plot.columns:
        if col in ['模型', '仓货使用次数']:
            table_values.append(df_plot[col].tolist())
        else:
            table_values.append([f'{x:.1%}' for x in df_plot[col].tolist()])

    fig = go.Figure(data=[go.Table(
        columnwidth=plot_column_width,
        header=dict(
            values=plot_column_names,
            line_color='white',
            fill_color='rgb(104, 205, 104)',
            font=dict(color='black', size=12),
            height=40
        ),
        cells=dict(
            values=table_values,
            line_color='white',
            fill=dict(color=column_color_ls),
            font_size=12,
            height=35
        )
    )], layout=go.Layout(
        autosize=True,
        margin={'l': 5, 'r': 5, 't': top_height, 'b': 0},
        title=f"<b>{title_text}</b>",
    ))

    print("save")
    save_path = os.path.join(DATA_OUTPUT_PATH, f"images/base_pred/{title_text}.png")
    fig.write_image(save_path, width=sum(plot_column_width) * 6 + 50, height=h, scale=2, engine='kaleido')
    if is_send:
        qiyeweixin_image(save_path, send_to=send_to)


def monitor_04_pred(pred_calculation_day=None):
    """
    监控++数据清洗++统计过滤次新品的个数
    """
    pred_calc_day = DayStr.get_dt_str(pred_calculation_day)
    pred_minus1_day = DayStr.n_day_delta(pred_calc_day, n=-1)
    pred_minus21_day = DayStr.n_day_delta(pred_calc_day, n=-21)

    log.title(f"监控++数据清洗++统计过滤次新品的个数 dt={pred_minus1_day} run at {DayStr.now2()}")

    # ------------------------
    # 货物范围
    selector = SelectGoodsList()
    sel_goods_ls = selector.get_long_period_goods_id()

    # ------------------------
    # 历史真实
    df_stock_sell = read_api.read_dt_folder(
        bip3("model/basic_predict_promote_online", "stock_wh_goods_type_flg_theory_sale_cnt")
        , pred_minus1_day)
    df_type_flg = (df_stock_sell
                   .groupby(['goods_id', 'type_flg'])
                   ['theory_sale_cnt'].sum().reset_index()
                   )
    sel_goods_ls_a = df_type_flg.query(f"goods_id in {sel_goods_ls} and type_flg =='subnew'")[
        'goods_id'].unique().tolist()

    # ------------------------
    # 上市计划
    com_plan = read_api.read_dt_folder(
        bip3("process", "dim_cmdty_online_plan_shop_d_his"), pred_minus1_day)
    sub_new_plan = com_plan.query(f"'{pred_minus21_day}' < actual_launch_date < '{pred_calc_day}'").copy()
    sub_new_plan = sub_new_plan.groupby(['commodity_id', 'actual_launch_date'])['dept_id'].count().rename(
        "shop_nums").reset_index()
    cols_sub_new_plan = ['commodity_id', 'actual_launch_date', 'shop_nums']
    sub_new_plan = sub_new_plan.groupby(['commodity_id']).tail(1)[cols_sub_new_plan]
    sub_new_plan.query("shop_nums >=100", inplace=True)
    sub_new_plan['type_flg_sub'] = 'subnew'

    # 配方
    ld_formula = read_api.read_dt_folder(bip2('process', 'commodity_goods_max'), pred_minus1_day)
    cols_sub_new_plan_goods = ['goods_id', 'actual_launch_date']
    sub_new_plan_goods = (sub_new_plan.merge(ld_formula)
        .query(f"goods_id in {sel_goods_ls}")
        .groupby(['goods_id']).tail(1)[cols_sub_new_plan_goods]
        )
    sel_goods_ls_b = sub_new_plan_goods.query(f"goods_id in {sel_goods_ls}")['goods_id'].unique().tolist()

    # ------------------------
    sel_goods_ls_c = list(set(sel_goods_ls_b) - set(sel_goods_ls_a))

    if len(sel_goods_ls_c) > 0:
        qiyeweixin_bot(msg=f"⚠️ 原始数据缺失次新品的过滤{sel_goods_ls_c}", send_to=300, mention_number_list=[18801057497])
    else:
        qiyeweixin_bot(msg=f"🟢 原始数据已过滤次新品{len(sel_goods_ls_a)}个", send_to=300)


def monitor_05_pred(pred_calculation_day=None, data_label='常规'):
    """
    监控++趋势++过去历史真实120天++预测未来120天
    折线图： 对比历史120天与预测未来120天
    """
    pred_calc_day = DayStr.get_dt_str(pred_calculation_day)
    start_date = DayStr.n_day_delta(pred_calc_day, n=-120)
    df_goods_info = dh_dw.dim_stock.goods_info()
    df_goods_info = df_goods_info[['goods_id', 'goods_name']].drop_duplicates()

    log.title(f"监控++趋势++过去历史真实120天++预测未来120天 dt={pred_calc_day} run at {DayStr.now2()}")

    # ------------------------
    # 货物范围
    selector = SelectGoodsList()
    sel_all_goods_ls = selector.get_group_all_goods_ls()

    # ------------------------
    # 历史120天， 常规真实  常规品 =  常规店常规品 + 新店常规品
    if data_label == '常规':
        df_his1 = read_api.read_dt_folder(
            bip3("model/basic_predict_promote_online", "feature_engineering_wh_goods_normal")
            , start_date, pred_calc_day)
        df_his2 = read_api.read_dt_folder(
            bip3("model/basic_predict_promote_online", "feature_engineering_wh_goods_new_shop_normal")
            , start_date, pred_calc_day)
        df_his = pd.concat([df_his1, df_his2])

    if data_label == '总':
        df_his = read_api.read_dt_folder(
            bip3("model/basic_predict_promote_online", "feature_engineering_wh_goods_all")
            , start_date, pred_calc_day)

    dfc_his = df_his.groupby(['ds', 'goods_id'])['y'].sum().reset_index()
    dfc_his.rename(columns={'ds': 'dt'}, inplace=True)

    # ------------------------
    # 预测值
    if data_label == '常规':
        df_pred = read_api.read_dt_folder(
            bip3("model/basic_predict_promote_online", "purchase_wh_goods_pred_normal"), pred_calc_day)
    if data_label == '总':
        df_pred = read_api.read_dt_folder(
            bip3("model/basic_predict_promote_online", "purchase_wh_goods_pred_all"), pred_calc_day)

    df_pred = df_pred[['predict_dt', 'wh_dept_id', 'goods_id', 'demand']]
    dfc_pred = df_pred.groupby(['predict_dt', 'goods_id'])['demand'].sum().reset_index()
    dfc_pred.rename(columns={"demand": "pred", "predict_dt": "dt"}, inplace=True)

    # ------------------------
    # 遍历 [group_b,group_c,group_e,group_f,group_g]
    for goods_label, sel_goods_ls in sel_all_goods_ls.items():
        # --------------
        # 真实值
        df_true_sum = dfc_his.query(f"goods_id in {sel_goods_ls}").copy()
        df_true_sum = df_true_sum.merge(df_goods_info)

        # --------------
        # 预测值
        df_pred_sum = dfc_pred.query(f"goods_id in {sel_goods_ls}").copy()
        df_pred_sum = df_pred_sum.merge(df_goods_info)

        log.debug(f"{goods_label}_真实货物数{df_true_sum['goods_id'].nunique()}, 预测货物数{df_pred_sum['goods_id'].nunique()}")
        if len(df_true_sum) > 0:
            monitor_05_img(pred_calculation_day, goods_label, data_label, df_true_sum, df_pred_sum)


def monitor_05_img(pred_calculation_day=None, goods_label='group_b', data_label='常规', df_true_sum=None,
                   df_pred_sum=None):
    """
    折线图： 对比历史120天与预测未来120天
    """
    pred_calc_day = DayStr.get_dt_str(pred_calculation_day)

    sel_goods_name = df_true_sum['goods_name'].unique().tolist()

    cols = 3  # 每行几列
    rows = len(sel_goods_name) // cols + 1

    fig = make_subplots(rows=rows,
                        cols=cols,
                        subplot_titles=sel_goods_name,  # 每个子图的 title
                        shared_yaxes=False,  # 共享y轴
                        horizontal_spacing=0.25 / cols,  # 子图与子图之间水平间距系数
                        vertical_spacing=0.2 / rows,  # 子图与子图之间垂直间距系数
                        row_heights=[450] * rows,  # 每一个行的高度
                        specs=[[{'secondary_y': True} for j in range(cols)] for i in range(rows)]
                        )

    for index in range(len(sel_goods_name)):
        row_index = index // cols + 1
        col_index = index % cols + 1

        df_plot = df_pred_sum.query(f"goods_name == '{sel_goods_name[index]}'")
        df_plot1 = df_true_sum.query(f"goods_name == '{sel_goods_name[index]}'")

        trace_0 = go.Scatter(
            x=df_plot.dt,
            y=df_plot['pred'],
            mode=' lines',
            name=f'{data_label}预测值',
            marker_color='blue'
        )
        trace_1 = go.Scatter(
            x=df_plot1.dt,
            y=df_plot1['y'],
            mode=' lines',
            name=f'{data_label}真实值',
            marker_color='red'
        )

        fig.add_trace(trace_0, secondary_y=False, row=row_index, col=col_index)
        fig.add_trace(trace_1, secondary_y=False, row=row_index, col=col_index)

    fig.update_layout(width=cols * 400, height=rows * 200, title=f'{data_label}_预测结果_{goods_label}_{pred_calc_day}')

    save_path = os.path.join(DATA_OUTPUT_PATH, f"images/base_pred/{data_label}_预测结果_{goods_label}_{pred_calc_day}.png")
    fig.write_image(save_path, scale=2, engine='kaleido')
    save_path = os.path.join(DATA_OUTPUT_PATH, f"images/base_pred/{data_label}_预测结果_{goods_label}_{pred_calc_day}.html")
    pio.write_html(fig, file=save_path)
    print(save_path)


def monitor_06_pred(pred_calculation_day=None):
    """
    监控++门店盘点记录++数据量是否有异常
    """
    pred_calc_day = DayStr.get_dt_str(pred_calculation_day)
    pred_minus2_day = DayStr.n_day_delta(pred_calc_day, n=-2)
    pred_minus1_day = DayStr.n_day_delta(pred_calc_day, n=-1)

    log.title(f"监控++门店盘点记录++数据量是否有异常 dt={pred_minus1_day} run at {DayStr.now2()}")

    # 前天 门店盘点记录
    df1 = read_api.read_dt_folder(
        bip3("process", "dwd_stock_shop_com_goods_stock_record"), pred_minus2_day)
    # 昨天 门店盘点记录
    df2 = read_api.read_dt_folder(
        bip3("process", "dwd_stock_shop_com_goods_stock_record"), pred_minus1_day)

    # 对比数据量
    ratio = len(df2) / len(df1)

    if ratio < 0.8:
        qiyeweixin_bot(msg=f"⚠️ 原始数据数据量缺失，只有{len(df2)}条，正常应有{len(df1)}条", send_to=300,
                       mention_number_list=[18801057497])
    elif ratio > 1.5:
        qiyeweixin_bot(msg=f"⚠️ 原始数据数据量翻倍，有{len(df2)}条，正常应有{len(df1)}条", send_to=300, mention_number_list=[18801057497])


if __name__ == '__main__':
    argv_date(main_monitor_tree_pred)
