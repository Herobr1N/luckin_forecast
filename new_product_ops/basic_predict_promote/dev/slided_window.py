

from datetime import timedelta

import numpy as np
import pandas as pd

from __init__ import project_path

f"Import from {project_path}"
"""
滑动窗口 - feature-engineering lag_n_mean字段
"""
def slide_window(yy,
                 start_date_list = ["2023-03-01" ,"2023-04-01" ,"2023-05-01" ,"2023-06-01" ,"2023-07-01" ,"2023-08-01","2023-09-01"],
                 window_size = 21,
                 test_size = 30):
    # 筛选要用的时间段数据
    result = yy[
        (yy["ds"] >= pd.to_datetime(min(start_date_list)) - timedelta(days = window_size)) &
        (yy["ds"] <= pd.to_datetime(max(start_date_list)) + timedelta(days = test_size))
        ].copy()

    result[f"yhat_{window_size}_window"] = 0
    result_norm_large = dict()
    # iterate over each warehouse and goods
    # 记录全周期mape，月mape

    for (wh_dept_id, goods_id), group_df in result.groupby(["wh_dept_id", "goods_id"]):

        # iterate over each update window date
        temp = pd.DataFrame()

        for i in range(len(start_date_list)):
            # print(date_range[i])
            # print("-------------")
            if pd.to_datetime(start_date_list[i]) <= group_df["ds"].min() :continue

            test_end_date = pd.to_datetime(start_date_list[i]) + timedelta(days = test_size)

            test = group_df.loc[
                (group_df["ds"] >= pd.to_datetime(start_date_list[i])) &(group_df["ds"] < test_end_date)
                ].copy()

            # 若测试集不够长则填充测试集
            if test_end_date > test["ds"].max():
                # 需要填充的行数
                rows_needed = test_size - len(test)

                df_fill = pd.DataFrame({"ds": pd.date_range(
                    test["ds"].max() + timedelta(days=1),
                    periods=rows_needed, freq="D")}
                )
                test = pd.concat([test, df_fill]).reset_index()
                # 缺失值填充
                test["wh_dept_id"].fillna(method="ffill", inplace=True)
                test["goods_id"].fillna(method="ffill", inplace=True)
                test["year"].fillna(test["ds"].dt.year, inplace=True)
                test["month"].fillna(test["ds"].dt.month, inplace=True)
                test["day"].fillna(test["ds"].dt.day, inplace=True)
                test["day_of_week"].fillna(test["ds"].dt.dayofweek, inplace=True)

            window_value = group_df[group_df["ds"] == pd.to_datetime(start_date_list[i])][f"lag_{window_size}_mean"].item()
            test.loc[:,f"yhat_{window_size}_window"] = window_value
            # 记录预测日
            test["predict_dt"] = pd.to_datetime(start_date_list[i])
            # 计算日mape
            test[f"mape_{window_size}_window"] = np.abs(1 - (test[f"yhat_{window_size}_window"]/test["y"]))

            temp = pd.concat([temp,test])
        #  月预测结果
        result_norm_large[(wh_dept_id,goods_id)] = temp

    return result_norm_large

"""
多种滑动平均
"""
def slide_window_summary(
    tt,
    start_date_list = ["2023-03-01","2023-04-01","2023-05-01","2023-06-01","2023-07-01","2023-08-01","2023-09-01"],
    window_list = [7,14,21,28],
    test_size = 30
):
    slide_window_result_all = {}
    #遍历每个窗口
    for window in window_list:
        dict_window_all = slide_window(tt,
               start_date_list = start_date_list,
               window_size = window,
               test_size = test_size)

        slide_window_result_all[window] = dict_window_all
    return slide_window_result_all