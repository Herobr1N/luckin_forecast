from config.config import *
import numpy as np

VLT = 7
BP = 15
SS = 7
RO = 9
pur_ratio = 12000
MOQ = 2000 * pur_ratio

result = pd.DataFrame(columns=["dt", "wh_dept_id", "goods_id", "sim_beg", "sim_end", 'end_cnt', 'sim_transit', 'sim_ss',
                               'ss', 'sim_ro', 'ro', "pred_demand", 'pred_demand_neg', 'dly_out', 'order_dt'])
df_transit = pd.DataFrame(columns=['dt', 'transit', 'order_dt'])


def base_period_model(df):
    i = 0
    sim_end = 0
    for row_index, row in df.iterrows():
        i += 1
        dt, wh_dept_id, goods_id, begin_cnt, end_cnt, pred_demand, demand_avg, dly_out, dly_out_avg = row['dt'], row.wh_dept_id, row.goods_id, row.begin_cnt, row.end_cnt, row.pred_demand, row.demand_avg, row.dly_out, row.dly_out_avg
        if i > 1:
            # 当前期初
            begin_cnt  = sim_end
        sim_ss = SS * demand_avg
        sim_ro = RO * demand_avg
        ss = - SS * dly_out_avg
        ro = - RO * dly_out_avg
        # 周期性补货日
        if dt[-2:] in ['01', '15']:
            # 计算目标库存
            dt_bp_vlt = pd.to_datetime(dt).date() + timedelta(days=VLT+BP-1)
            dmd_bp_vlt = df.query(f"(dt>='{dt}') and (dt <= '{dt_bp_vlt}')")['pred_demand'].sum()
            ti = dmd_bp_vlt + sim_ss + sim_ro
            # BP+VLT期间在途
            transit = df_transit.query(f"dt >= '{dt}' and (dt <= '{dt_bp_vlt}')")['transit'].sum()
            # 原始采购量
            rep = max(np.ceil(ti - begin_cnt - transit), 0)
            # 凑N倍的MOQ
            rep_moq = np.ceil(rep / MOQ) * MOQ
            # 模拟入仓
            dt_receive = str(pd.to_datetime(dt).date() + timedelta(days=VLT))
            df_transit = df_transit.append({'dt': dt_receive, 'transit': rep_moq, 'order_dt': dt}, ignore_index=True)
            print(f"周期性订货日：{dt}, 覆盖订货区间:【{dt},{dt_bp_vlt}】, 在途总量:{transit}, 需订货数量:{rep_moq}, 计划入仓时间:{dt_receive}")

        # 计算模拟期末
        # 当日在途
        transit_base = df_transit.query(f"dt == '{dt}'")
        if len(transit_base) == 0:
            transit = 0
            order_dt = None
        else:
            transit = transit_base['transit'].sum()
            order_dt = transit_base['order_dt'].values[0]
        sim_end = max(begin_cnt + dly_out + transit, 0)
        print(f"+++ 更新：{dt} 期初：{begin_cnt}, 实际出库:{dly_out}，当日入库：{transit}, 当日期末：{sim_end}")

        result.loc[i] = [dt, wh_dept_id, goods_id, begin_cnt, sim_end, end_cnt, transit, sim_ss, ss, sim_ro, ro, pred_demand, -pred_demand, dly_out, order_dt]

    result['sim_ss_ro'] = result['sim_ss'] + result['sim_ro']
    result['ss_ro'] = result['ss'] + result['ro']
    return result


def stored_calendar(total_rep, df_transit, cur_dt, begin_cnt, sim_ss, sim_ro):
    """
        入仓策略：
        1.第一次入仓前，周期性补货模型盘点第一次入仓时间VLT+K
        2.若有剩余量，在(VLT+k,T+BP】期间均匀入仓
    """
    # 判断第一个VLT是否需入仓
    vlt = VLT
    bp = 1
    sim_cur_dt = cur_dt
    k = 0
    is_first = True
    rest_cnt = total_rep
    while is_first:
        # 计算目标库存
        dt_bp_vlt = pd.to_datetime(cur_dt).date() + timedelta(days=vlt+k+bp-1)
        dmd_bp_vlt = df.query(f"(dt>='{cur_dt}') and (dt <= '{dt_bp_vlt}')")['pred_demand'].sum()
        ti = dmd_bp_vlt + sim_ss + sim_ro
        # 计算补货量
        transit = df_transit.query(f"(dt>='{cur_dt}') and (dt <= '{dt_bp_vlt}')")['transit'].sum()
        rep = max(np.ceil(ti - begin_cnt - transit), 0)
        # 凑N倍的MOQ
        rep_moq = np.ceil(rep / MOQ) * MOQ
        print(f"--- 推算第一个周期性补货点:{dt_bp_vlt}, 覆盖区间:【{cur_dt}, {dt_bp_vlt}】, Transit:{transit},Rep:{rep_moq}")
        if rep_moq > 0:
            # 模拟入库，加入在途中
            dt_receive = str(pd.to_datetime(cur_dt).date() + timedelta(days=vlt+k))
            df_transit = df_transit.append({'dt': dt_receive, 'transit': rep_moq, 'order_dt': cur_dt}, ignore_index=True)
            print(f"### 模拟下单点:{cur_dt}, VLT={vlt}, 需入库时间 {dt_receive}, 入库量：{rep_moq}")
            rest_cnt = total_rep - rep_moq
            is_first = False
        else:
            k += 1

    # 剩余量，均匀入仓
    if rest_cnt <= 0:
        print("BP周期内无需再次订货")
        return df_transit

    # 连续性开始
    sim_cur_dt = pd.to_datetime(cur_dt).date() + timedelta(days=vlt+k+1)
    # 连续性结束
    sim_end_dt = pd.to_datetime(cur_dt).date() + timedelta(days=VLT+BP-1)

    """模拟[T, T+vlt+k]计算期间库存，推算第T+vlt+k期末"""
    sim_period_start = pd.to_datetime(cur_dt).date()
    while sim_period_start < sim_cur_dt:
        dly_out = df.query(f"dt == '{sim_period_start}'")['dly_out'].values[0]
        transit = df_transit.query(f"dt == '{sim_period_start}'")['transit'].sum()
        end = max(begin_cnt + dly_out + transit, 0)
        print(f"+++ 更新第一次补货期间库存：{sim_period_start} 期初：{begin_cnt}, 实际出库:{dly_out}，当日入库：{transit}, 当日期末：{end}")
        begin_cnt = end
        # sim_period_start = str(pd.to_datetime(sim_period_start).date() + timedelta(days=1))
        sim_period_start = sim_period_start + timedelta(days=1)


    # 入仓频率(N天一次）
    freq = (BP-k-1) / (rest_cnt / MOQ)
    # 若入仓频率低于1天/次，则可隔N天一次，每次1个MOQ， 若入仓频率大于1天/次，则每天发货，每次N个MOQ

    if freq > 1:
        rep_freq, rep_per_cnt = freq, 1
    else:
        # 在BP-k-1期间，每次以rep_per_cnt的的数量入仓
        rep_per_cnt = np.ceil(1/freq)
        rep_freq = (BP-k-1) / (( rest_cnt / MOQ)/ rep_per_cnt)

    receive_index = 0
    # 已入仓总量
    total = rest_cnt

    print(f"剩余{rest_cnt/MOQ}个Q, 在[{sim_cur_dt}, {sim_end_dt}]期间({BP-k-1}天)均匀入仓，入仓频率:{rep_freq}, 每次入仓量:{rep_per_cnt}")


    while rest_cnt > 0:
        # 入仓日期
        dt_receive = str(pd.to_datetime(sim_cur_dt).date() + timedelta(days=receive_index))
        # 入仓数量
        rep_cnt = rep_per_cnt * MOQ if (rest_cnt / MOQ) > rep_per_cnt else rest_cnt
        df_transit = df_transit.append({'dt': dt_receive, 'transit': rep_cnt, 'order_dt': cur_dt}, ignore_index=True)
        # cg_order[dt_receive] = rep_cnt
        rest_cnt -= rep_cnt
        receive_index += rep_freq
        print(f"订货点:{cur_dt}, 总需订货:{total/MOQ}, 在 {dt_receive} 入仓{rep_cnt/MOQ}Q, 剩余{rest_cnt/MOQ}")

    return df_transit


def period_avg_model(df):
    i = 0
    sim_end = 0
    for row_index, row in df.iterrows():
        i += 1
        dt, wh_dept_id, goods_id, begin_cnt, end_cnt, pred_demand, demand_avg, dly_out, dly_out_avg = row['dt'], row.wh_dept_id, row.goods_id, row.begin_cnt, row.end_cnt, row.pred_demand, row.demand_avg, row.dly_out, row.dly_out_avg
        if i > 1:
            # 当前期初
            begin_cnt  = sim_end
        sim_ss = SS * demand_avg
        sim_ro = RO * demand_avg
        ss = - SS * dly_out_avg
        ro = - RO * dly_out_avg
        # 周期性补货，平均入仓
        if dt[-2:] in ['01', '15']:
            # 计算目标库存
            dt_bp_vlt = pd.to_datetime(dt).date() + timedelta(days=VLT+BP-1)
            dmd_bp_vlt = df.query(f"(dt>='{dt}') and (dt <= '{dt_bp_vlt}')")['pred_demand'].sum()
            ti = dmd_bp_vlt + sim_ss + sim_ro
            # BP+VLT期间在途
            transit = df_transit.query(f"dt >= '{dt}' and (dt <= '{dt_bp_vlt}')")['transit'].sum()
            # 原始采购量
            rep = max(np.ceil(ti - begin_cnt - transit), 0)
            # 凑N倍的MOQ
            rep_moq = np.ceil(rep / MOQ) * MOQ
            print(f"周期性订货日：{dt}, 覆盖订货区间:【{dt},{dt_bp_vlt}】, 在途总量:{transit}, 需订货数量:{rep_moq}")
            if rep_moq > 0:
                df_transit = stored_calendar(rep_moq, df_transit, dt, begin_cnt, sim_ss, sim_ro)
        # 当日在途
        transit_base = df_transit.query(f"dt == '{dt}'")
        if len(transit_base) == 0:
            transit = 0
            order_dt = None
        else:
            transit = transit_base['transit'].sum()
            order_dt = transit_base['order_dt'].values[0]
        sim_end = max(begin_cnt + dly_out + transit, 0)
        result.loc[i] = [dt, wh_dept_id, goods_id, begin_cnt, sim_end, end_cnt, transit, sim_ss, ss, sim_ro, ro, pred_demand, -pred_demand, dly_out, order_dt]

    result['sim_ss_ro'] = result['sim_ss'] + result['sim_ro']
    result['ss_ro'] = result['ss'] + result['ro']




def run(origin_df):
    df = origin_df.query("wh_dept_id == 329232 and (goods_id == 354)")


if __name__ == '__main__':
    origin = spark.read_parquet('/user/yanxin.lu/sc/tower/simulate').sort_values(by='dt').reset_index(drop=True)
    float_column = ['begin_cnt', 'end_cnt', 'pur_cnt', 'dly_out', 'dly_out_avg']
    origin[float_column] = origin[float_column].astype('float')
    run(origin)
