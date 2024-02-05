from config.config import *
import numpy as np
from PIL import Image
import plotly.express as px

from utils.decorator import log_wecom
from utils.msg_utils import Message, MSG_GROUP_THREE, MSG_GROUP_TWO

from utils.file_utils import save_img


def simulate(df, _df_res):
    i = 0
    sim_end = 0
    for row_index, row in df.iterrows():
        i += 1
        dt, wh_dept_id, wh_name, goods_id, goods_name, spec_id, spec_name, ratio, demand, beg_cnt, total_transit, transit, loss = row[
            ['predict_dt', 'wh_dept_id', 'wh_name', 'goods_id', 'goods_name', 'spec_id', 'spec_name', 'ratio', 'max_demand', 'beg_cnt',
             'total_transit', 'transit', 'loss']]
        if i > 1:
            beg_cnt = sim_end
        sim_end = max(beg_cnt - demand + transit - loss, 0)
        _df_res.loc[i] = [dt, wh_dept_id, wh_name, goods_id, goods_name, spec_name, ratio, beg_cnt, demand, total_transit, transit, loss, sim_end]
    return _df_res


def get_concat_v(images):
    """
    合并图像
    :param images: Image List
    :return: Image
    """
    if images is None:
        return
    width, height = images[0].size
    dst = Image.new('RGB', (width, height * len(images)))
    for i in range(len(images)):
        dst.paste(images[i], (0, height * i))
    return dst


def draw_img(df, cmdty_name):
    """
    7、14、21、28天断仓热力图
    """
    img_list = []
    for i in range(1, 4):
        d = i * 7
        fig = px.density_heatmap(df.sort_values(by=['wh_dept_id', 'goods_id']),
                                 x='wh_name', y='goods_name', z=f'f{d}_days',
                                 title=f'{today} {cmdty_name} 相关原料 <b>未来{d}日断仓天数</b> 预估（以上市峰值消耗，含预计{d}日内入仓FH，TRS，CG）')
        fig.update_layout(height=500, width=1500,
                          xaxis_title="仓库名称",
                          yaxis_title="货物名称",
                          coloraxis_colorbar=dict(title='断仓天数'))
        img_list.append(Image.open(save_img(fig)))

    # 合并
    img_res = get_concat_v(img_list)
    return img_res


def load():
    df_base = spark.read_parquet('/user/yanxin.lu/sc/yeyun/base_info').sort_values(by='predict_dt')
    df_base[['max_demand', 'beg_cnt', 'ratio']] = df_base[['max_demand', 'beg_cnt', 'ratio']].astype('float')

    # 计算总在途数据
    df_total_trs = df_base.groupby(['wh_dept_id', 'goods_id'], as_index=False).agg({'transit': 'sum'}).rename(columns={'transit': 'total_transit'})
    df_base = pd.merge(df_base, df_total_trs, on=['wh_dept_id', 'goods_id'], how='left')

    return df_base


def send_msg(df):
    # 椰云拿铁
    cmdty_name = '椰云拿铁'
    df_yeyun = df.query("goods_id.isin([20818, 25463, 25456, 25316, 25533, 408, 411, 52, 4488]) ")
    # 发送图片消息
    Message.send_msg(msg=f'{today}_{cmdty_name} 未来断仓预估', group=MSG_GROUP_TWO)
    # 发送图片
    Message.send_image(draw_img(df_yeyun, cmdty_name), group=MSG_GROUP_TWO)
    # 发送详情文件
    Message.send_file(df_yeyun, file_name=f'{today}_{cmdty_name}相关货物紧俏状况.csv', group=MSG_GROUP_THREE)


@log_wecom('椰云监控推送', P_ONE)
def process(df_base):
    # 分仓-货物模拟断仓
    res_list = []
    for (wh_dept_id, goods_id), group in df_base.groupby(['wh_dept_id', 'goods_id']):
        _df_res = pd.DataFrame(
            columns=['dt', 'wh_dept_id', 'wh_name', 'goods_id', 'goods_name', 'spec_name', 'ratio', 'beg', 'demand', 'total_transit', 'transit',
                     'loss', 'sim_end'])
        df = df_base.query(f"wh_dept_id == {wh_dept_id} and goods_id == {goods_id}")
        df_middle = simulate(df, _df_res)
        # 计算可用天数
        df_middle['avl_days'] = np.round((df_middle['beg'] + df_middle['total_transit']) / df_middle['demand'], 1)
        df_middle.loc[df_middle['sim_end'] == 0, 'need_cnt'] = df_middle['demand'] - df_middle['beg']
        df_middle['need_cum'] = df_middle['need_cnt'].cumsum()
        df_middle['rank'] = df_middle['dt'].rank()
        # 分别计算未来7、14、21、28天的断仓天数
        interval = [7 * i for i in range(1, 5)]
        for _dt in interval:
            need_sum = df_middle.query(f"rank <= {_dt}")['need_cnt'].sum()
            need_dmd = df_middle['demand'][1]
            need_days = np.ceil(need_sum / need_dmd)
            df_middle[f'f{_dt}_cnt'] = need_sum
            df_middle[f'f{_dt}_days'] = need_days
        res_list.append(df_middle)

    # 合并结果
    df_total = pd.concat(res_list).reset_index(drop=True)
    df_res = df_total.query("rank == 1").sort_values(by=['goods_id', 'f7_cnt', 'f14_cnt', 'f21_cnt', 'f28_cnt', 'avl_days'],
                                                     ascending=[1, 0, 0, 0, 0, 1])[
        ['dt', 'wh_dept_id', 'wh_name', 'goods_id', 'goods_name', 'spec_name', 'ratio', 'beg', 'total_transit', 'demand', 'avl_days', 'f7_cnt',
         'f7_days', 'f14_cnt', 'f14_days', 'f21_cnt', 'f21_days', 'f28_cnt', 'f28_days']]
    send_msg(df_res)


if __name__ == '__main__':
    process(load())
