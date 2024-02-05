# -*- coding: utf-8 -*-
import pandas as pd
import glob
from hdfs import InsecureClient
import requests
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import os
import base64
import hashlib

pd.set_option("display.max_columns", 200)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.min_rows', 20)
pd.set_option('display.max_rows', 100)

# HDFS client
HDFS_URIs = ['http://bimasterx02.luckycoffee.com:9870',
             'http://bimasterx03.luckycoffee.com:9870']
HDFS_URI = ';'.join(HDFS_URIs)
USERNAME = "yanxin.lu"
HDFS_PLAN_PATH = '/user/yanxin.lu/sc/monitor/plan'
HDFS_SIM_PATH = '/user/yanxin.lu/sc/monitor/similarity'
HDFS_CMDTY_PATH = '/user/yanxin.lu/sc/monitor/cmdty_info'
HDFS_CMDTY_SALE_PATH = '/user/yanxin.lu/sc/monitor/cmdty_sale'
HDFS_PURCHASE_STATUS_PATH = '/user/yanxin.lu/sc/monitor/goods_purchase_status'
HDFS_GOODS_COST_PATH = '/user/yanxin.lu/sc/monitor/goods_cost'
HDFS_GOODS_COST_WEEKLY_PATH = '/user/yanxin.lu/sc/monitor/goods_cost_weekly'
MSG_TYPE_PLAN = 'plan'
MSG_TYPE_GOODS = 'goods'
MSG_GROUP_ONE = '新品冷启动'
MSG_GROUP_TWO = '收益算法'
MSG_GROUP_THREE = '监控-测试'
MSG_GROUP_TEST = 'Test'
NOTIFY_MSG = 'MSG'
NOTIFY_IMAGE = 'IMAGE'
NOTIFY_FILE = 'FILE'
LOCAL_DIR = '/data/purchase/monitor/'

client = InsecureClient(HDFS_URI, user=USERNAME)
client.download(HDFS_PLAN_PATH, LOCAL_DIR, overwrite=True, n_threads=1)
client.download(HDFS_SIM_PATH, LOCAL_DIR, overwrite=True, n_threads=1)
client.download(HDFS_CMDTY_PATH, LOCAL_DIR, overwrite=True, n_threads=1)
client.download(HDFS_CMDTY_SALE_PATH, LOCAL_DIR, overwrite=True, n_threads=1)
client.download(HDFS_PURCHASE_STATUS_PATH, LOCAL_DIR, overwrite=True, n_threads=1)
client.download(HDFS_GOODS_COST_PATH, LOCAL_DIR, overwrite=True, n_threads=1)
client.download(HDFS_GOODS_COST_WEEKLY_PATH, LOCAL_DIR, overwrite=True, n_threads=1)

current_dt = datetime.today().date()
start_four_dt = current_dt - timedelta(days=28)


def load_data(filepath):
    """
    读取本地文件
    """
    df_list = []
    for file in glob.glob(filepath):
        temp = pd.read_parquet(file)
        df_list.append(temp)
    df_base = df_list[0] if 1 == len(df_list) else pd.concat(df_list, sort=False)
    return df_base


def getURL(group=MSG_GROUP_TEST):
    # test
    url = 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=930c0943-ae89-48dd-8f79-f4203c7fa835'
    if group == MSG_GROUP_ONE:
        # prod 新品配置
        url = 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=228d5fb3-40fc-451b-abfb-065445138ae8'
    elif group == MSG_GROUP_TWO:
        #  prod 收益-算法
        url = 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=3fd43103-3077-4909-8403-825eba17c6a9'
    # prod 监控-测试
    elif group == MSG_GROUP_THREE:
        url = 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=57ac898e-313c-4aa7-bf4e-f6a0e407f251'
    return url


def goods_level_rank(df):
    df['level_rank'] = 4
    df.loc[df['result_level'] == '一级', 'level_rank'] = 1
    df.loc[df['result_level'] == '二级', 'level_rank'] = 2
    df.loc[df['result_level'] == '三级', 'level_rank'] = 3
    return df


def sendFile(media_id, group=MSG_GROUP_TEST):
    data = {'msgtype': 'file',
            'file': {'media_id': media_id}
            }
    res = requests.post(url=getURL(group), json=data)
    print("## file 【%s】 send %s ##" % (media_id, res))


def sendImage(image_path, group=MSG_GROUP_TEST):
    with open(image_path, 'rb') as f:
        base64_data = base64.b64encode(f.read())
        base64_data = base64_data.decode()
    with open(image_path, 'rb') as f:
        fmd5 = hashlib.md5(f.read()).hexdigest()

    data = {
        "msgtype": "image",
        "image": {"base64": base64_data, "md5": fmd5}
    }

    headers = {"Content-Type": "text/plain"}
    res = requests.post(getURL(group), json=data, headers=headers)
    print("## image 【%s】 send %s ##" % (image_path, res))


def sendMsg(msg, group=MSG_GROUP_TEST, mention_list=[]):
    mention_list = ['LuYanXin'] if group == MSG_GROUP_TEST else mention_list
    headers = {"Content-Type": "application/json"}
    text_info = {
        "msgtype": "text",
        "text": {
            "content": msg,
            "mentioned_list": list(mention_list)
        }
    }
    res = requests.post(getURL(group), headers=headers, json=text_info)
    print("## msg 【%s】 send %s ##" % (msg, res))


def getProdConfigMsg(record):
    msg = '-1'
    items = list()
    if pd.isnull(record.norm_sale):
        items.append('预估单店杯量')
    if pd.isnull(record.cmdty_level):
        items.append('商品等级')
    if pd.isnull(record.top_sale) and (record.one_category_id == 22):
        items.append('人气top预销量')
    if pd.isnull(record.top_days) and (record.one_category_id == 22):
        items.append('top位持续天数')
    if len(items) > 0:
        msg = '未配置项：%s' % (items)
    return msg


def getMarketConfigMsg(record):
    msg = '-1'
    items = list()
    if pd.isnull(record.market_level) and (record.one_category_id == 22):
        items.append('营销等级')
    if len(items) > 0:
        msg = '未配置项：%s' % (items)
    return msg


def configProdCheck():
    """
    商品计划基本配置检查
    计划状态 plan_status
        1 - 已新建； 2 - 已启动； 3 - 已确认到仓； 4 - 已确认到店日期； 5 - 已确认停采日期； 6 - 已确认停售日期
        7 - 已确认推广计划； 8 - 已确认售卖日期； 9-已确认分仓数据
    商品状态 cmdty_status
        0 - 已创建； 1 - 已配置； 2 - 已上线； 3 - 已下线
    计划类型 plan_type
        1 - 短期计划； 2 - 上市计划； 3 - 下市计划； 4 - 强制下市计划
    """
    valid_plan = df_plan.query("plan_type in (1, 2) and (cmdty_status in (1,2)) and (plan_status > 1) and (plan_launch_date > '%s')" % current_dt)[
        ['plan_code', 'plan_status', 'cmdty_name', 'cmdty_status', 'one_category_id', 'two_category_id', 'norm_sale', 'top_sale', 'cmdty_level',
         'top_days']].drop_duplicates()
    config_res = None
    if len(valid_plan) > 0:
        valid_plan['msg'] = valid_plan.apply(lambda x: getProdConfigMsg(x), axis=1)
        config_df = valid_plan.query("msg != '-1'")
        config_df['type'] = '配置缺失'
        config_res = config_df[['plan_code', 'cmdty_name', 'two_category_id', 'type', 'msg']]
    return config_res


def configMarketCheck():
    """
    商品计划营销配置检查
    计划状态 plan_status
        1 - 已新建； 2 - 已启动； 3 - 已确认到仓； 4 - 已确认到店日期； 5 - 已确认停采日期； 6 - 已确认停售日期
        7 - 已确认推广计划； 8 - 已确认售卖日期； 9-已确认分仓数据
    商品状态 cmdty_status
        0 - 已创建； 1 - 已配置； 2 - 已上线； 3 - 已下线
    计划类型 plan_type
        1 - 短期计划； 2 - 上市计划； 3 - 下市计划； 4 - 强制下市计划
    """
    valid_plan = df_plan.query("plan_type in (1, 2) and (cmdty_status in (1, 2)) and (plan_status > 1) and (plan_launch_date > '%s')" % current_dt)[
        ['plan_code', 'plan_status', 'cmdty_name', 'cmdty_status', 'one_category_id', 'two_category_id', 'market_level']].drop_duplicates()
    # 白名单，小鹿茶SOE
    filter_one = ~(valid_plan['plan_code'].isin(
        ['PP20210816002', 'PP20210629011', 'PP20210629007', 'PP20210629006', 'PP20210629005', 'PP20210629009', 'PP20210629010', 'PP20210629008',
         'PP20210629012', 'PP20211026002', 'PP20211026003', 'PP20211026004']))
    valid_plan = valid_plan[filter_one]
    config_res = None
    if len(valid_plan) > 0:
        valid_plan['msg'] = valid_plan.apply(lambda x: getMarketConfigMsg(x), axis=1)
        config_df = valid_plan.query("msg != '-1'")
        config_df['type'] = '营销配置缺失'
        config_res = config_df[['plan_code', 'cmdty_name', 'two_category_id', 'type', 'msg']]
    return config_res


def simCheck():
    """
    检查相似品配置项
    """
    # 相似品影响系数和不为1
    df_factor = df_sim.groupby('plan_id', as_index=False).agg({'factor': 'sum'}).query("abs(factor - 1) > 0.0001")
    df_factor['msg'] = '相似品影响系数之和不为1'
    df_sim['start'] = pd.to_datetime(df_sim['start']).dt.date
    df_sim['end'] = pd.to_datetime(df_sim['end']).dt.date
    # 相似品开始日期 or 结束日期 晚于今日
    df_sim.loc[(df_sim.start >= current_dt) | (df_sim.end >= current_dt) | (df_sim.end < df_sim.start), 'flag'] = 1
    df_date = df_sim.query("flag == 1")
    df_date['msg'] = '相似品参考周期配置错误'
    sim_middle = pd.concat([df_factor[['plan_id', 'msg']], df_date[['plan_id', 'msg']]])
    sim_middle['type'] = '配置错误'
    plan_info = df_plan[['plan_id', 'plan_code', 'cmdty_name', 'two_category_id']].drop_duplicates()
    sim_res = pd.merge(sim_middle, plan_info, on='plan_id', how='inner')[['plan_code', 'cmdty_name', 'two_category_id', 'type', 'msg']]
    return sim_res


def duplicateCheck():
    """
    商品重复, 同一商品，同一范围，同一上市类型，同一上市时间
    """
    plan = df_plan
    plan['plan_dt'] = plan.apply(lambda x: x.plan_launch_date if x.plan_type in (1, 2) else x.plan_abolish_date, axis=1)
    duplicate = plan.groupby(['cmdty_id', 'city_id', 'plan_type', 'plan_dt'], as_index=False) \
        .agg({'plan_code': 'count'}) \
        .query("plan_code > 1") \
        .groupby(['cmdty_id', 'plan_type'], as_index=False) \
        .agg({'city_id': 'count'}) \
        .rename(columns={'city_id': 'city_cnt'})
    duplicate_res = None
    if len(duplicate) > 0:
        middle = pd.merge(duplicate, df_cmdty_info, on='cmdty_id', how='left')
        middle['type'] = '配置错误'
        middle['msg'] = middle.apply(lambda x: '重复城市数量：%s' % x.city_cnt, axis=1)
        middle['plan_code'] = ''
        duplicate_res = middle[['plan_code', 'cmdty_name', 'two_category_id', 'type', 'msg']]
    return duplicate_res


def launchDateCheck():
    """
    上市日期检查
    """
    valid_plan = df_plan.query("plan_type in (1, 2)")
    valid_plan['city_start_sale_dt'] = pd.to_datetime(valid_plan['city_start_sale_dt'])
    plan_start_dt = valid_plan.groupby('plan_code', as_index=False).agg({'city_start_sale_dt': 'min'}).rename(
        columns={'city_start_sale_dt': 'plan_start_sale_dt'})
    valid_plan = pd.merge(valid_plan, plan_start_dt, on='plan_code', how='left')
    # 实际售卖日期 晚于 配置上市日期
    after = valid_plan[valid_plan['plan_start_sale_dt'].isna() & (valid_plan['plan_actual_launch_date'] >= str(start_four_dt)) & (
                valid_plan['plan_actual_launch_date'] < str(current_dt))]
    after_res = None
    if len(after) > 0:
        after['type'] = '滞后售卖'
        after['msg'] = after.apply(
            lambda x: '计划实际上市-->%s, 最早售卖-->%s' % ('未配置' if x.plan_actual_launch_date is None else x.plan_actual_launch_date, x.plan_start_sale_dt),
            axis=1)
        after_res = after[['plan_code', 'cmdty_name', 'two_category_id', 'type', 'msg']]

    # 实际售卖日期 早于 配置上市日期
    before = valid_plan[((valid_plan['city_start_sale_dt'].dt.date < current_dt) & (valid_plan['plan_actual_launch_date'].isna())) | (
                (valid_plan['city_start_sale_dt'].astype('str') < valid_plan['plan_actual_launch_date']) & (
                    valid_plan['plan_actual_launch_date'] >= str(current_dt)))]
    # 过滤白名单
    # 厦门 ，广州 厚乳 拿铁 上海 花魁5.0
    filter_one = ~(before['plan_code'].isin(['PP20211012001', 'PP20211012002', 'PP20211012003', 'PP20210816002']) & (
        before['city_name'].isin(['广州', '厦门', '上海'])))
    # 小鹿茶SOE
    filter_two = ~(before['plan_code'].isin(
        ['PP20220301001', 'PP20220415001', 'PP20220415002', 'PP20220415003', 'PP20220415004', 'PP20220223001', 'PP20210705001', 'PP20210629011', 'PP20210629007',
         'PP20210629006', 'PP20210629005', 'PP20210629009', 'PP20210629010', 'PP20210629008', 'PP20210629012', 'PP20211201003', 'PP20211201002',
         'PP20211201001', 'PP20220209002', 'PP20220209003']))
    before = before[filter_one & filter_two]
    before_middle = before.groupby(['plan_code', 'cmdty_name', 'two_category_id'], as_index=False) \
        .agg({'city_id': 'count', 'city_name': 'unique'}) \
        .rename(columns={'city_id': 'city_cnt'})
    before_res = None
    if len(before_middle) > 0:
        before_middle['type'] = '提前售卖'
        before_middle['msg'] = before_middle.apply(
            lambda x: '%s等%s个城市' % (x.city_name[0:5], x.city_cnt) if x.city_cnt > 5 else '城市列表：%s' % x.city_name, axis=1)
        before_res = before_middle[['plan_code', 'cmdty_name', 'two_category_id', 'type', 'msg']]
    if after_res is None and before_res is None:
        return None
    launch_date = pd.concat([after_res, before_res])
    return launch_date


def salePredictCheck():
    """
    检查实际店均销量与预估销量差异
    """
    valid_plan = df_plan[((df_plan['plan_type'].isin([1, 2])) & (df_plan['cmdty_status'] == 2) & (
                df_plan['plan_actual_launch_date'] >= str(current_dt - timedelta(days=14)))) |
                         ((df_plan['plan_type'].isin([1, 2])) & (df_plan['cmdty_status'] == 2) & (df_plan['plan_actual_launch_date'].isna()))]
    # 白名单
    filter_one = ~(valid_plan['plan_code'].isin(['PP20210629011', 'PP20210629007', 'PP20210629006', 'PP20210629005',
                                                 'PP20210629009', 'PP20210629010', 'PP20210629008', 'PP20210629012']))
    valid_plan = valid_plan[filter_one]

    if valid_plan is None:
        return None
    actual_shop_sale_avg = pd.merge(df_cmdty_sale, valid_plan, on=['cmdty_id', 'city_id'], how='left') \
        .groupby(['plan_code'], as_index=False) \
        .agg({"shop_cnt": "sum", "sale_cnt": "sum"})
    actual_shop_sale_avg['shop_avg'] = actual_shop_sale_avg['sale_cnt'] / actual_shop_sale_avg['shop_cnt']
    middle = pd.merge(actual_shop_sale_avg, valid_plan[
        ['plan_code', 'cmdty_name', 'one_category_id', 'two_category_id', 'plan_actual_launch_date', 'norm_sale', 'top_sale']].drop_duplicates(),
                      on='plan_code', how='left')
    middle.loc[middle['top_sale'].isna(), 'top_sale'] = middle['norm_sale']
    # 实际店均销量 低于 预估norm的一半
    below = middle[middle['norm_sale'].astype('float') / middle['shop_avg'] >= 2]
    below_res = None
    if len(below) > 0:
        below['type'] = '销量低预期'
        below['msg'] = below.apply(lambda x: '实际店均：%.1f，预估店均：%.1f' % (x.shop_avg, x.norm_sale), axis=1)
        below_res = below[['plan_code', 'cmdty_name', 'two_category_id', 'type', 'msg']]
    # 饮品类，实际店均销量 高于 预估top位1.5倍
    above_drink_res = None
    above_drink = middle[(middle['one_category_id'] == 22) & (middle['shop_avg'] / middle['top_sale'].astype('float') >= 1.5)]
    if len(above_drink) > 0:
        above_drink['type'] = '销量高预期'
        above_drink['msg'] = above_drink.apply(lambda x: '实际店均：%.1f，预估TOP店均：%.1f' % (x.shop_avg, x.top_sale), axis=1)
        above_drink_res = above_drink[['plan_code', 'cmdty_name', 'two_category_id', 'type', 'msg']]
    # 食品类，实际店均销量 高于 预估norm 2倍
    above_food_res = None
    above_food = middle[(middle['one_category_id'] == 23) & (middle['shop_avg'] / middle['norm_sale'].astype('float') >= 2)]
    if len(above_food) > 0:
        above_food['type'] = '销量高预期'
        above_food['msg'] = above_food.apply(lambda x: '实际店均：%.1f，预估店均：%.1f' % (x.shop_avg, x.norm_sale), axis=1)
        above_food_res = above_food[['plan_code', 'cmdty_name', 'two_category_id', 'type', 'msg']]

    if below_res is None and above_food_res is None and above_drink_res is None:
        return None
    sale_res = pd.concat([below_res, above_drink_res, above_food_res])
    return sale_res


def purchaseStatusCheck():
    """
    检查采购状态配置
    """
    middle = df_purchase_status.query("purchase_status == 1") \
        .groupby(['wh_dept_id', 'goods_id', 'wh_name', 'large_class_id', 'goods_name'], as_index=False) \
        .agg({'spec_id': 'count', 'spec_name': 'unique'}) \
        .rename(columns={'spec_id': 'spec_cnt'}) \
        .query("spec_cnt > 1")
    purchase_status_res = None
    if len(middle) > 0:
        middle['type'] = '采购关注状态重复'
        middle['msg'] = middle.apply(lambda x: '重复规格：%s' % x.spec_name, axis=1)
        middle = pd.merge(middle, df_goods_cost[['wh_dept_id', 'goods_id', 'large_class_name', 'result_level']].drop_duplicates(),
                          on=['wh_dept_id', 'goods_id'], how='left')
        purchase_status_res = middle[['wh_name', 'goods_name', 'large_class_id', 'large_class_name', 'result_level', 'type', 'msg']]
    return purchase_status_res


def goodsCostCheck():
    """
    仓维度门店消耗异常
    """
    # 过滤不采购-不关注状态数据
    all_status = df_purchase_status.groupby(['wh_dept_id', 'goods_id'], as_index=False).agg({'purchase_status': 'count'})
    invalid_status = df_purchase_status.query("purchase_status == 3").groupby(['wh_dept_id', 'goods_id'], as_index=False).agg(
        {'purchase_status': 'count'})
    valid_goods = pd.merge(all_status, invalid_status, on=['wh_dept_id', 'goods_id'], how='left').query("purchase_status_x != purchase_status_y")[
        ['wh_dept_id', 'goods_id']]
    # 有效的消耗
    valid_goods_cost = pd.merge(df_goods_cost, valid_goods, on=['wh_dept_id', 'goods_id'], how='left')
    valid_goods = valid_goods_cost[(valid_goods_cost['seven_avg'] > 0) & (valid_goods_cost['sale_of_day'] == 1)]
    valid_goods['warn'] = valid_goods.apply(lambda x: 1 if (x.cost_cnt / x.seven_avg) >= 1.6 else 0, axis=1)
    goods_cost = valid_goods.query("warn == 1")
    goods_cost_res = None
    if len(goods_cost) > 0:
        goods_cost['type'] = '门店消耗量陡增'
        goods_cost['msg'] = goods_cost.apply(lambda x: "昨日消耗：%.0f；近7日均：%.0f" % (x.cost_cnt, x.seven_avg), axis=1)
        goods_cost_res = goods_cost[['wh_name', 'goods_name', 'large_class_id', 'large_class_name', 'result_level', 'type', 'msg']]
    return goods_cost_res


def constructMsg(df, msg_type):
    if df is not None:
        types = list()
        contents = list()
        for key, value in df.groupby('type'):
            types.append(key)
            contents.append(value)
        msg = ''
        for i in range(len(types)):
            msg += '%s<%s>异常%s\n' % ('*' * 8, types[i], '*' * 8)
            for record in contents[i].values.tolist():
                if msg_type == MSG_TYPE_PLAN:
                    msg += '计划编号：%s； 商品：%s；\n异常信息：%s\n' % (record[0], record[1], record[4])
                elif msg_type == MSG_TYPE_GOODS:
                    msg += '仓库：%s； 货物：%s；\n异常信息：%s\n' % (record[0], record[1], record[6])
        return msg


def constructImage(df, title):
    """
        构造图片
    """
    if (df is None) or len(df) < 0:
        return ''
    df = goods_level_rank(df)
    df = df.sort_values(by=['level_rank', 'goods_name'])

    df_plot = df[['type', 'wh_name', 'goods_name', 'result_level', 'msg', 'level_rank']]
    plot_column_names = ['异常类型', '仓库名称', '货物名称', '货物级别', '异常信息', ]
    plot_column_names = [f'<b>{x}</b>' for x in plot_column_names]
    # 构造颜色
    color_pastel1 = px.colors.qualitative.Pastel1
    df_plot['color'] = color_pastel1[8]
    df_plot.loc[df_plot['level_rank'] == 1, 'color'] = color_pastel1[0]
    df_plot.loc[df_plot['level_rank'] == 2, 'color'] = color_pastel1[1]
    df_plot.loc[df_plot['level_rank'] == 3, 'color'] = color_pastel1[2]
    plot_colors = [df_plot['color'].tolist()]

    layout = go.Layout(autosize=True, margin={'l': 5, 'r': 5, 't': 60, 'b': 0},
                       title='<b>异常大类：%s</b>' % title)

    fig = go.Figure(data=[go.Table(
        columnwidth=[10, 10, 20, 5, 30],
        header=dict(
            values=[[x] for x in plot_column_names],
            line_color='darkslategray',
            fill_color='royalblue',
            align=['center', 'center', 'center', 'center', 'center'],
            font=dict(color='white', size=12),
            height=40
        ),
        cells=dict(
            values=[df_plot[x].tolist() for x in df_plot.columns[:-2]],
            line_color='darkslategray',
            fill=dict(color=plot_colors),
            align=['center', 'center', 'left', 'center', 'left'],
            font_size=12,
            height=30)
    ),
    ], layout=layout)
    if not os.path.exists("images"):
        os.mkdir("images")
    save_path = os.path.join('images', 'purchase_%s.png' % title)

    fig.write_image(save_path, height=((len(df_plot) + 2) * 50), scale=2, engine='orca')
    return save_path


def constructFile(df, file_name, groupURL):
    """
    构造文件
    """
    file_name = '%s_%s.csv' % (file_name, current_dt)
    key = groupURL.split('=')[-1]
    id_url = "https://qyapi.weixin.qq.com/cgi-bin/webhook/upload_media?key=%s&type=file" % key
    if not os.path.exists("files"):
        os.mkdir("files")
    save_path = os.path.join('files', file_name)
    df.to_csv(save_path, header=True, index=False, encoding='utf_8_sig')
    data = {'media': open(save_path, 'rb')}
    msg = requests.post(url=id_url, files=data)
    dict_data = msg.json()
    media_id = dict_data['media_id']
    return media_id


def sendProxy(df, msg_type, msg_group, mobiles):
    # 受发送内容长度限制，分批次发送
    if df is not None:
        total_items = len(df)
        times = total_items // 30
        for i in range(times + 1):
            begin = i * 30
            end = min(begin + 30, total_items)
            sendMsg(constructMsg(df[begin:end], msg_type), msg_group, mobiles)


def wh_cost_weekly():
    """
    周-仓维度，消耗量异常
    """
    df_wh_cost = df_goods_cost_weekly
    df_wh_cost.sort_values(by=['wh_dept_id', 'goods_id', 'week_start_dt'], inplace=True)
    df_wh_cost['cost_cnt'] = df_wh_cost['cost_cnt'].round(1)

    df_wh_cost['previous_one'] = df_wh_cost.groupby(['wh_dept_id', 'goods_id'])['cost_cnt'].shift(1)
    df_wh_cost['previous_two'] = df_wh_cost.groupby(['wh_dept_id', 'goods_id'])['cost_cnt'].shift(2)
    wh_base = df_wh_cost.groupby(['wh_dept_id', 'goods_id'], as_index=False).tail(1).reset_index(drop=True)

    # 选择 采购且关注 状态数据
    valid_goods = df_purchase_status.query("purchase_status == 1").groupby(['wh_dept_id', 'goods_id'], as_index=False).agg(
        {'purchase_status': 'count'})
    # 计算差异
    wh_base['diff_value'] = wh_base['cost_cnt'] - wh_base['previous_one']
    wh_base['diff_ratio'] = np.round(wh_base['diff_value'] / wh_base['previous_one'], 2)

    # 阈值
    THRESHOLD = 0.25
    valid_item = wh_base.query("(diff_ratio <= -{}) | (diff_ratio >= {})".format(THRESHOLD, THRESHOLD))
    wh_cost = pd.merge(valid_item, valid_goods[['wh_dept_id', 'goods_id', 'purchase_status']],
                       on=['wh_dept_id', 'goods_id'],
                       how='left') \
        .query("purchase_status > 0")
    # 辅助排序
    wh_cost = goods_level_rank(wh_cost)
    wh_cost['cost_total'] = wh_cost.groupby('goods_id')['cost_cnt'].transform("sum")

    # 排序输出
    wh_cost_res = wh_cost.query("large_class_id in (3, 4, 6, 15)") \
        .sort_values(by=['level_rank', 'cost_total', 'diff_ratio'], ascending=[True, False, False]) \
        [['week_start_dt', 'wh_name', 'large_class_name', 'large_class_id', 'goods_name', 'result_level', 'cost_cnt', 'previous_one', 'previous_two',
          'diff_ratio']] \
        .rename(columns={
        'week_start_dt': '基础周',
        'wh_name': '仓库名称',
        'large_class_name': '货物大类',
        'goods_name': '货物名称',
        'result_level': '货物级别',
        'cost_cnt': '消耗（基础周W）',
        'previous_one': '消耗（W-1）',
        'previous_two': '消耗（W-2）',
        'diff_ratio': '变化率'})
    return wh_cost_res


def country_cost_week():
    """
    周-全国维度，消耗量
    """
    df_country_cost = df_goods_cost_weekly.groupby(['week_start_dt', 'large_class_id', 'large_class_name', 'goods_id', 'goods_name', 'result_level'],
                                                   as_index=False) \
        .agg({'cost_cnt': 'sum'}) \
        .sort_values(by=['goods_id', 'week_start_dt'])
    df_country_cost['cost_cnt'] = df_country_cost['cost_cnt'].round(1)

    df_country_cost['previous_one'] = df_country_cost.groupby(['goods_id'])['cost_cnt'].shift(1)
    df_country_cost['previous_two'] = df_country_cost.groupby(['goods_id'])['cost_cnt'].shift(2)
    country_base = df_country_cost.groupby(['goods_id'], as_index=False).tail(1).reset_index(drop=True)
    country_base['diff_value'] = country_base['cost_cnt'] - country_base['previous_one']
    country_base['diff_ratio'] = np.round(country_base['diff_value'] / country_base['previous_one'], 2)
    # 阈值
    THRESHOLD = 0.15
    valid_item = country_base.query("(diff_ratio <= -{}) | (diff_ratio >= {})".format(THRESHOLD, THRESHOLD))

    valid_goods = df_purchase_status.query("purchase_status == 1").groupby(['goods_id'], as_index=False).agg({'purchase_status': 'count'})
    country_cost = pd.merge(valid_item, valid_goods[['goods_id', 'purchase_status']],
                            on=['goods_id'],
                            how='left') \
        .query("purchase_status > 0")
    country_cost = goods_level_rank(country_cost)
    # 辅助排序
    country_cost['cost_total'] = country_cost.groupby('goods_id')['cost_cnt'].transform("sum")

    # 排序输出
    country_cost_res = country_cost.query("large_class_id in (3, 4, 6)") \
        .sort_values(by=['level_rank', 'large_class_id', 'diff_ratio'], ascending=[True, True, False]) \
        [['week_start_dt', 'large_class_name', 'goods_name', 'result_level', 'cost_cnt', 'previous_one', 'previous_two', 'diff_ratio']]

    return country_cost_res


def draw_country_cost_week(df_table):
    """
    周维度消耗，Pandas To Table
    """
    headers = ['基础周', '货物大类', '货物名称', '货物级别', '消耗（基础周W）', '消耗（W-1）', '消耗（W-2）', '变化率']

    def get_value_color(value):
        base = 'rgb({:1f}, 0, 0)' if value > 0 else 'rgb(0, {:1f}, 0)'
        return base.format(min(abs(value) + .35, 1) * 255)

    df_table['value_color'] = df_table['diff_ratio'].apply(lambda x: get_value_color(x))

    # 根据货物级别标记颜色
    df_table['level_color'] = 'rgba(249, 250, 253, 1)'
    colors = color_pastel1 = px.colors.qualitative.Pastel1
    df_table.loc[df_table['result_level'] == '一级', 'level_color'] = colors[0]
    df_table.loc[df_table['result_level'] == '二级', 'level_color'] = colors[1]
    df_table.loc[df_table['result_level'] == '三级', 'level_color'] = colors[2]

    headers = [f'<b>{x}</b>' for x in headers]
    column_values = [df_table[col] for col in df_table.columns[:-2]]
    # 变化率颜色
    font_color = [['black'] for x in range(7)]
    font_color.extend([df_table['value_color']])
    # 货物级别颜色
    column_color = []
    for index in range(len(headers)):
        if index == 2:
            column_color.append(df_table['level_color'].tolist())
        else:
            column_color.append(['rgba(249, 250, 253, 1)'])

    fig = go.Figure(data=[go.Table(
        columnwidth=[25, 15, 50, 15, 30, 30, 30, 20],
        header=dict(values=headers,
                    fill_color='rgba(89, 93, 98, 1)',
                    font=dict(color='white', size=13),
                    align='center'),
        cells=dict(values=column_values,
                   fill_color='rgba(249, 250, 253, 1)',
                   font=dict(color=font_color, size=13),
                   height=32,
                   align='center',
                   )
    )])
    fig.data[0]['cells']['fill']['color'] = column_color
    fig.update_layout(title='<b>全国周维度消耗量异常</b>')
    if not os.path.exists("images"):
        os.mkdir("images")
    save_path = os.path.join('images', 'country_cost_weekly.png')
    fig.write_image(save_path, height=((len(df_table) + 2) * 50), width=1000, scale=2, engine='orca')
    return save_path


def runCmdtyPlan():
    res_prod_config = configProdCheck()
    res_market_config = configMarketCheck()
    res_sim = simCheck()
    res_duplicate = duplicateCheck()
    res_launch = launchDateCheck()
    res_sale_predict = salePredictCheck()

    if res_duplicate is not None or res_launch is not None or res_sim is not None or \
            res_prod_config is not None or res_market_config is not None or res_sale_predict is not None:
        res_df = pd.concat([res_duplicate, res_launch, res_prod_config, res_market_config, res_sim, res_sale_predict]).reset_index(drop=True)
        # 销量异常
        sale_predict = res_df.query("type in ('销量低预期', '销量高预期')")
        if len(sale_predict) > 0:
            sendProxy(sale_predict, MSG_TYPE_PLAN, MSG_GROUP_ONE, [])
            res_df = res_df.drop(sale_predict.index)

        # 营销配置，吴姝菡 17775455493
        market_config = res_df.query("type == '营销配置缺失'")
        if len(market_config) > 0:
            sendProxy(market_config, MSG_TYPE_PLAN, MSG_GROUP_ONE, ['17775455493'])
            res_df = res_df.drop(market_config.index)

        # 咖啡，经典饮品 彭雨欣 PengYuXin, 陈瑞敏 2021083323, 刘得真 2021082263
        coffee_exfreezo = res_df.query("two_category_id in (51, 57)")
        if len(coffee_exfreezo) > 0:
            # 全局过滤白名单, 不@提醒
            white_list = coffee_exfreezo.query("plan_code in ['']")
            if len(white_list) > 0:
                sendProxy(white_list, MSG_TYPE_PLAN, MSG_GROUP_ONE, [])
            other = coffee_exfreezo.drop(white_list.index)
            if len(other) > 0:
                sendProxy(other, MSG_TYPE_PLAN, MSG_GROUP_ONE, ['PengYuXin', '2021083323', '2021082263'])
        # 小鹿茶, 瑞纳冰, 左秀萍 15021255139，陈超 13405058565
        tea = res_df.query("two_category_id in (51, 73)")
        if len(tea) > 0:
            sendProxy(tea, MSG_TYPE_PLAN, MSG_GROUP_ONE, ['2021121650'])
        # 健康轻食 王静 WangJing 原子禹、沈杨雯（2021040539）、王朵、陈俊威（2021070650）
        food = res_df.query("two_category_id == 47")
        if len(food) > 0:
            sendProxy(food, MSG_TYPE_PLAN, MSG_GROUP_ONE, ['duowang', '2021040539', '2021070650', 'YuanZiYu'])
        # 其它
        other = res_df.query("two_category_id not in (51, 55, 47, 73)")
        if len(other) > 0:
            sendProxy(other, MSG_TYPE_PLAN, MSG_GROUP_ONE, [])


def notifyByGroup(df, notify_type, dim=''):
    # 原料 王佳 15122481820，李富 18519509080
    yuan_liao = df.query("large_class_id == 6")
    if len(yuan_liao) > 0:
        if notify_type == NOTIFY_MSG:
            sendProxy(yuan_liao, MSG_TYPE_GOODS, MSG_GROUP_ONE, ['15122481820', '18519509080'])
        elif notify_type == NOTIFY_IMAGE:
            if len(yuan_liao) < 50:
                sendImage(constructImage(yuan_liao, '原料类'), group=MSG_GROUP_THREE)
            else:
                sendFile(constructFile(yuan_liao, '原料类', getURL(MSG_GROUP_THREE)),
                         group=MSG_GROUP_THREE)
        elif notify_type == NOTIFY_FILE:
            sendFile(constructFile(yuan_liao, '【{}】原料类'.format(dim), getURL(MSG_GROUP_THREE)),
                     group=MSG_GROUP_THREE)

    # 轻食 王青 13581923360
    qing_shi = df.query("large_class_id == 3")
    if len(qing_shi) > 0:
        if notify_type == NOTIFY_MSG:
            sendProxy(qing_shi, MSG_TYPE_GOODS, MSG_GROUP_ONE, ['13581923360'])
        elif notify_type == NOTIFY_IMAGE:
            if len(qing_shi) < 50:
                sendImage(constructImage(qing_shi, '轻食类'), group=MSG_GROUP_THREE)
            else:
                sendFile(constructFile(qing_shi, '轻食类', getURL(MSG_GROUP_THREE)),
                         group=MSG_GROUP_THREE)
        elif notify_type == NOTIFY_FILE:
            sendFile(constructFile(qing_shi, '【{}】轻食类'.format(dim), getURL(MSG_GROUP_THREE)),
                     group=MSG_GROUP_THREE)

    # 包材，零件 李紫涵 13466643166
    bao_cai_ling_jian = df.query("large_class_id in (4, 15)")
    if len(bao_cai_ling_jian) > 0:
        if notify_type == NOTIFY_MSG:
            sendProxy(bao_cai_ling_jian, MSG_TYPE_GOODS, MSG_GROUP_ONE, ['13466643166'])
        elif notify_type == NOTIFY_IMAGE:
            if len(bao_cai_ling_jian) < 50:
                sendImage(constructImage(bao_cai_ling_jian, '包装_零件类'), group=MSG_GROUP_THREE)
            else:
                sendFile(constructFile(bao_cai_ling_jian, '包装_零件类', getURL(MSG_GROUP_THREE)),
                         group=MSG_GROUP_THREE)
        elif notify_type == NOTIFY_FILE:
            sendFile(constructFile(bao_cai_ling_jian, '【{}】包装_零件类'.format(dim), getURL(MSG_GROUP_THREE)),
                     group=MSG_GROUP_THREE)

    # 日耗，器具，工服，办公用品，营销物料  翟羽 13811360444
    other = df.query("large_class_id in (29, 2, 22, 1, 7)")
    if len(other) > 0:
        if notify_type == NOTIFY_MSG:
            sendProxy(other, MSG_TYPE_GOODS, MSG_GROUP_ONE, ['13811360444'])
        elif notify_type == NOTIFY_IMAGE:
            if len(other) < 50:
                sendImage(constructImage(other, '日耗_器具_工服_办公_营销'), group=MSG_GROUP_THREE)
            else:
                sendFile(constructFile(other, '日耗_器具_工服_办公_营销', getURL(MSG_GROUP_THREE)),
                         group=MSG_GROUP_THREE)


def runPurchase():
    # 采购状态、日消耗
    res_purchase_status = purchaseStatusCheck()
    res_goods_cost = goodsCostCheck()

    if res_purchase_status is not None or res_goods_cost is not None:
        res_purchase = pd.concat([res_purchase_status, res_goods_cost]) \
            .sort_values(by=['result_level', 'goods_name']).reset_index(drop=True)
        # 消耗量异常
        goods_cost = res_purchase.query("(type == '门店消耗量陡增')")
        if len(goods_cost) > 0:
            # 通知一级物料
            level_one = goods_cost.query("result_level == '一级'")
            if len(level_one) > 0:
                sendImage(constructImage(level_one, '一级物料'), group=MSG_GROUP_TWO)
            # 分组通知
            notifyByGroup(goods_cost, NOTIFY_IMAGE)
            res_purchase = res_purchase.drop(goods_cost.index)
        # 配置异常
        if len(res_purchase) > 0:
            notifyByGroup(res_purchase, NOTIFY_MSG)

    #     周消耗，每周一触发
    if 0 == current_dt.weekday():
        res_wh_cost_weekly = wh_cost_weekly()
        if res_wh_cost_weekly is not None:
            notifyByGroup(res_wh_cost_weekly, NOTIFY_FILE, dim='周-分仓')
        res_country_cost_week = country_cost_week()
        if res_country_cost_week is not None:
            sendImage(draw_country_cost_week(res_country_cost_week), group=MSG_GROUP_THREE)


if __name__ == '__main__':
    print("%s -- ### Start ###" % datetime.now())
    # 读取plan
    df_plan = load_data(LOCAL_DIR + 'plan')
    # 读取相似品
    df_sim = load_data(LOCAL_DIR + 'similarity')
    # 读取商品信息
    df_cmdty_info = load_data(LOCAL_DIR + 'cmdty_info')
    # 读取商品销量信息
    df_cmdty_sale = load_data(LOCAL_DIR + 'cmdty_sale')
    # 读取采购状态配置
    df_purchase_status = load_data(LOCAL_DIR + 'goods_purchase_status')
    # 读取仓维度门店消耗量
    df_goods_cost = load_data(LOCAL_DIR + 'goods_cost')
    # 读取仓-周维度门店消耗量
    df_goods_cost_weekly = load_data(LOCAL_DIR + 'goods_cost_weekly')

    runCmdtyPlan()
    runPurchase()
    print("%s -- ### Finished ###" % datetime.now())
