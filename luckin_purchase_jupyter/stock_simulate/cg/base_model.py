def model_one(df, df_transit, df_res):
    sim_end = 0
    i = 0
    for row_index, row in df.iterrows():
        i += 1
        dt, wh_dept_id, wh_name, goods_id, begin_cnt, end_cnt, pred_demand, demand_avg, dly_out, dly_out_avg, transit = row['dt'], row.wh_dept_id, row.wh_name, row.goods_id, row.begin_cnt, row.end_cnt, row.pred_demand, row.demand_avg, row.dly_out, row.dly_out_avg, row.pur_cnt
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
            sim_transit = df_transit.query(f"dt >= '{dt}' and (dt <= '{dt_bp_vlt}')")['transit'].sum()
            # 原始采购量
            rep = max(np.ceil(ti - begin_cnt - sim_transit), 0)
            # 凑N倍的MOQ
            rep_moq = np.ceil(rep / MOQ) * MOQ
            # 模拟入仓
            dt_receive = str(pd.to_datetime(dt).date() + timedelta(days=VLT))
            df_transit = df_transit.append({'dt': dt_receive, 'transit': rep_moq, 'order_dt': dt}, ignore_index=True)
            print(f"周期性订货日：{dt}, 覆盖订货区间:【{dt},{dt_bp_vlt}】, 在途总量:{transit/MOQ}个Q, 需订货数量:{rep_moq/MOQ}个Q, 计划入仓时间:{dt_receive}")

        # 计算模拟期末
        # 当日在途
        transit_base = df_transit.query(f"dt == '{dt}'")
        if len(transit_base) == 0:
            sim_transit = 0
            order_dt = None
        else:
            sim_transit = transit_base['transit'].sum()
            order_dt = transit_base['order_dt'].values[0]
        sim_end = max(begin_cnt + dly_out + sim_transit, 0)
        print(f"+++ 更新：{dt} 期初：{begin_cnt}, 实际出库:{dly_out}，当日入库：{sim_transit}, 当日期末：{sim_end}")

        df_res.loc[i] = [dt, wh_dept_id, wh_name, goods_id, begin_cnt, sim_end, end_cnt, sim_transit, transit, sim_ss, ss, sim_ro, ro, pred_demand, -pred_demand, dly_out, order_dt]

    df_res['sim_ss_ro'] = df_res['sim_ss'] + df_res['sim_ro']
    df_res['ss_ro'] = df_res['ss'] + df_res['ro']
    return df_res

def period_avg_calendar(df, total_rep, df_transit, cur_dt, begin_cnt, sim_ss, sim_ro):
    """
        入仓策略：
        1.第一次入仓前，周期性补货模型盘点第一次入仓时间VLT+K
        2.若有剩余量，在(VLT+k,T+BP】期间均匀入仓
    """
    # 判断第一个VLT是否需入仓
    vlt = VLT
    bp = 1
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
        rest_cnt -= rep_cnt
        receive_index += rep_freq
        print(f"订货点:{cur_dt}, 总需订货:{total/MOQ}, 在 {dt_receive} 入仓{rep_cnt/MOQ}Q, 剩余{rest_cnt/MOQ}")

    return df_transit

def model_two(df, df_transit, df_res):
    i = 0
    sim_end = 0
    for row_index, row in df.iterrows():
        i += 1
        dt, wh_dept_id, wh_name, goods_id, begin_cnt, end_cnt, pred_demand, demand_avg, dly_out, dly_out_avg, transit = row['dt'], row.wh_dept_id, row.wh_name, row.goods_id, row.begin_cnt, row.end_cnt, row.pred_demand, row.demand_avg, row.dly_out, row.dly_out_avg, row.pur_cnt
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
            sim_transit = df_transit.query(f"dt >= '{dt}' and (dt <= '{dt_bp_vlt}')")['transit'].sum()
            # 原始采购量
            rep = max(np.ceil(ti - begin_cnt - sim_transit), 0)
            # 凑N倍的MOQ
            rep_moq = np.ceil(rep / MOQ) * MOQ
            print(f"周期性订货日：{dt}, 覆盖订货区间:【{dt},{dt_bp_vlt}】, 在途总量:{transit}, 需订货数量:{rep_moq}")
            if rep_moq > 0:
                df_transit = period_avg_calendar(df, rep_moq, df_transit, dt, begin_cnt, sim_ss, sim_ro)
        # 当日在途
        transit_base = df_transit.query(f"dt == '{dt}'")
        if len(transit_base) == 0:
            sim_transit = 0
            order_dt = None
        else:
            sim_transit = transit_base['transit'].sum()
            order_dt = transit_base['order_dt'].values[0]
        sim_end = max(begin_cnt + dly_out + sim_transit, 0)
        df_res.loc[i] = [dt, wh_dept_id, wh_name, goods_id, begin_cnt, sim_end, end_cnt, sim_transit, transit, sim_ss, ss, sim_ro, ro, pred_demand, -pred_demand, dly_out, order_dt]

    df_res['sim_ss_ro'] = df_res['sim_ss'] + df_res['sim_ro']
    df_res['ss_ro'] = df_res['ss'] + df_res['ro']
    return df_res

def period_rq_calendar(df, total_rep, df_transit, cur_dt, begin_cnt, sim_ss, sim_ro):
    # 若入仓频率高于1天一次，则每天发货，发货数量保证尽量不触发断仓预警
    sim_cur_dt = cur_dt
    bp = 1

    """
        进行一次周期型补货，判断第一个VLT是否需要补货
    """
    vlt = VLT
    # 计算目标库存
    dt_bp_vlt = pd.to_datetime(sim_cur_dt).date() + timedelta(days=vlt+bp-1)
    dmd_bp_vlt = df.query(f"(dt>='{sim_cur_dt}') and (dt <= '{dt_bp_vlt}')")['pred_demand'].sum()
    ti = dmd_bp_vlt + sim_ss + sim_ro
    # 计算补货量
    transit = df_transit.query(f"(dt>='{sim_cur_dt}') and (dt <= '{dt_bp_vlt}')")['transit'].sum()
    rep = max(np.ceil(ti - begin_cnt - transit), 0)
    # 凑N倍的MOQ
    rep_moq = np.ceil(rep / MOQ) * MOQ
    print(f"--- 周期型补货日:{sim_cur_dt}, 覆盖区间:【{sim_cur_dt}, {dt_bp_vlt}】, Transit:{transit},Rep:{rep_moq}")
    if rep_moq > 0:
        # 模拟入库，加入在途中
        dt_receive = str(pd.to_datetime(sim_cur_dt).date() + timedelta(days=vlt))
        df_transit = df_transit.append({'dt': dt_receive, 'transit': rep_moq, 'order_dt': cur_dt}, ignore_index=True)
        print(f"    ### 模拟入库：模拟下单点:{sim_cur_dt}, VLT={vlt}, 需入库时间 {dt_receive}, 入库量：{rep_moq}")

    # 模拟计算期间库存
    sim_end_dt = str(pd.to_datetime(sim_cur_dt).date() + timedelta(days=VLT+bp-1))
    while sim_cur_dt <= sim_end_dt:
        dly_out = df.query(f"dt == '{sim_cur_dt}'")['dly_out'].values[0]
        transit = df_transit.query(f"dt == '{sim_cur_dt}'")['transit'].sum()
        end = max(begin_cnt + dly_out + transit, 0)
        print(f"+++ 更新：{sim_cur_dt} 期初：{begin_cnt}, 实际出库:{dly_out}，当日入库：{transit}, 当日期末：{end}")
        begin_cnt = end
        sim_cur_dt = str(pd.to_datetime(sim_cur_dt).date() + timedelta(days=1))

    # 周期剩余量
    rest_cnt = total_rep - rep_moq
    if rest_cnt <= 0:
        print("BP周期内无需再次订货")
        return df_transit
    """
        连续型盘点，判断是否需要入库
        在剩余的BP-1天中，入库剩余量rest_cnt
    """
    # 入仓频率(N天一次）
    freq = (BP-1) / (rest_cnt / MOQ)
    # 若入仓频率低于1天/次，则可隔N天一次，每次1个MOQ， 若入仓频率大于1天/次，则每天发货，每次N个MOQ
    rep_freq, rep_per_cnt = (freq, 1) if freq > 1 else (1, np.ceil(1/freq))
    print(f"进入连续性盘点，共需入库:{rest_cnt}, 入仓频率：{rep_freq}, 每次入仓数量:{rep_per_cnt}")
    vlt = 1
    while rest_cnt > 0:
        # 计算目标库存
        dt_bp_vlt = pd.to_datetime(sim_cur_dt).date() + timedelta(days=vlt+bp-1)
        dmd_bp_vlt = df.query(f"(dt>='{sim_cur_dt}') and (dt <= '{dt_bp_vlt}')")['pred_demand'].sum()
        ti = dmd_bp_vlt + sim_ss + sim_ro
        # 计算补货量
        transit = df_transit.query(f"(dt>='{sim_cur_dt}') and (dt <= '{dt_bp_vlt}')")['transit'].sum()
        rep = max(np.ceil(ti - begin_cnt - transit), 0)
        # 凑N倍的MOQ
        rep_moq = np.ceil(rep / MOQ) * MOQ
        print(f"--- 连续型盘点日:{sim_cur_dt}, 覆盖区间:【{sim_cur_dt}, {dt_bp_vlt}】, Transit:{transit},Rep:{rep_moq}")
        if rep_moq > 0:
            # 模拟入库，加入在途中
            dt_receive = str(pd.to_datetime(sim_cur_dt).date() + timedelta(days=vlt))
            df_transit = df_transit.append({'dt': dt_receive, 'transit': rep_moq, 'order_dt': cur_dt}, ignore_index=True)
            rest_cnt -= rep_moq
            print(f"    ### 模拟入库：模拟下单点:{sim_cur_dt}, VLT={vlt}, 需入库时间 {dt_receive}, 入库量：{rep_moq}")

        # 计算期末
        # 当天在途
        transit = df_transit.query(f"dt=='{sim_cur_dt}'")['transit'].sum()
        # 当日实际出库量
        dly_out = df.query(f"(dt == '{sim_cur_dt}')")
        # 超出模拟范围
        if len(dly_out) == 0:
            print('### OUT ###')
            break
        dly_out = dly_out['dly_out'].values[0]
        end = max(begin_cnt + dly_out + transit, 0)
        print(f"+++ 更新：{sim_cur_dt} 期初：{begin_cnt}, 实际出库:{dly_out}，当日入库：{transit}, 当日期末：{end}, 剩余量:{rest_cnt}")
        begin_cnt = end
        sim_cur_dt = pd.to_datetime(sim_cur_dt).date() + timedelta(days=1)
    return df_transit

def model_three(df, df_transit, df_res):
    i = 0
    sim_end = 0
    for row_index, row in df.iterrows():
        i += 1
        dt, wh_dept_id, wh_name, goods_id, begin_cnt, end_cnt, pred_demand, demand_avg, dly_out, dly_out_avg, transit = row['dt'], row.wh_dept_id, row.wh_name,row.goods_id, row.begin_cnt, row.end_cnt, row.pred_demand, row.demand_avg, row.dly_out, row.dly_out_avg, row.pur_cnt
        if i > 1:
            # 当前期初
            begin_cnt  = sim_end
        sim_ss = SS * demand_avg
        sim_ro = RO * demand_avg
        ss = - SS * dly_out_avg
        ro = - RO * dly_out_avg
        # 周期性补货，连续盘点入仓
        if dt[-2:] in ['01', '15']:
            dt_bp_vlt = pd.to_datetime(dt).date() + timedelta(days=VLT+BP-1)
            dmd_bp_vlt = df.query(f"(dt>='{dt}') and (dt <= '{dt_bp_vlt}')")['pred_demand'].sum()
            ti = dmd_bp_vlt + sim_ss + sim_ro
            # BP+VLT期间在途
            sim_transit = df_transit.query(f"dt >= '{dt}' and (dt <= '{dt_bp_vlt}')")['transit'].sum()
            # 原始采购量
            rep = max(np.ceil(ti - begin_cnt - sim_transit), 0)
            # 凑N倍的MOQ
            rep_moq = np.ceil(rep / MOQ) * MOQ
            print(f"周期性订货日：{dt}, 覆盖订货区间:【{dt},{dt_bp_vlt}】, 目标库存:{ti}, 在途总量:{transit}, 需订货数量:{rep_moq}")
            df_transit = period_rq_calendar(df, rep_moq, df_transit, dt, begin_cnt, sim_ss, sim_ro)
        # 当日在途
        transit_base = df_transit.query(f"dt == '{dt}'")
        if len(transit_base) == 0:
            sim_transit = 0
            order_dt = None
        else:
            sim_transit = transit_base['transit'].sum()
            order_dt = transit_base['order_dt'].values[0]
        sim_end = max(begin_cnt + dly_out + sim_transit, 0)

        df_res.loc[i] = [dt, wh_dept_id, wh_name, goods_id, begin_cnt, sim_end, end_cnt, sim_transit, transit, sim_ss, ss, sim_ro, ro, pred_demand, -pred_demand, dly_out, order_dt]

    df_res['sim_ss_ro'] = df_res['sim_ss'] + df_res['sim_ro']
    df_res['ss_ro'] = df_res['ss'] + df_res['ro']
    return df_res