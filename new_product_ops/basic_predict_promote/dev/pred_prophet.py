
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error

sys.path.insert(0, "/home/dinghuo/alg_dh")

from utils_offline.a00_imports import dfu, log, DayStr, argv_date, dop, bip2, bip1, bip3, c_path, read_api, shuttle, \
    c_path_save_df, bip3_save_df2

"""
单点预测-单仓单货物
"""
def my_prophet(df_current_stock_all,
               today="2023-07-01",
               train_size=60,
               test_size=30,
               change_point_range=0.8,
               change_point_scaler=0.03,
               seasonal_m="additive",
               growth_m="logistic"):
    from prophet import Prophet
    # train test split
    train_end_date = pd.to_datetime(today)

    train_start_date = train_end_date - timedelta(days=train_size)
    # 训练集
    df_train = df_current_stock_all[
        (df_current_stock_all["ds"] < train_end_date)
        & (df_current_stock_all["ds"] >= train_start_date)
        ].copy().reset_index(drop=True)
    test_end_date = train_end_date + timedelta(days=test_size)
    # 测试集
    df_test = df_current_stock_all[
        (df_current_stock_all["ds"] >= train_end_date)
        & (df_current_stock_all["ds"] < test_end_date)
        ].copy().reset_index(drop=True)
    # 若测试集不够长则填充测试集
    if test_end_date > df_test["ds"].max():
        # 需要填充的行数
        rows_needed = test_size - len(df_test)

        df_fill = pd.DataFrame({"ds": pd.date_range(
            df_test["ds"].max() + timedelta(days=1),
            periods=rows_needed, freq="D")}
        )
        df_test = pd.concat([df_test, df_fill]).reset_index()
        # 缺失值填充
        df_test["wh_dept_id"].fillna(method="ffill", inplace=True)
        df_test["goods_id"].fillna(method="ffill", inplace=True)
        df_test["year"].fillna(df_test["ds"].dt.year, inplace=True)
        df_test["month"].fillna(df_test["ds"].dt.month, inplace=True)
        df_test["day"].fillna(df_test["ds"].dt.day, inplace=True)
        df_test["day_of_week"].fillna(df_test["ds"].dt.dayofweek, inplace=True)
    # define the cap and floor for logistic trend function
    cap = df_train["y"].max()
    floor = df_train["y"].min()

    df_train["cap"] = cap
    df_test["cap"] = cap
    df_train["floor"] = floor
    df_test["floor"] = floor
    # 创建prophet模型
    my_prophet_model = Prophet(
        growth=growth_m,
        changepoint_range=change_point_range,
        changepoint_prior_scale=change_point_scaler,
        seasonality_mode=seasonal_m,
        interval_width=0.95)
    # 添加exogenous variable
    my_prophet_model.add_regressor("month")
    # my_prophet.add_regressor("day")
    # my_prophet.add_regressor("day_of_week")

    # 节假日因子
    my_prophet_model.add_country_holidays(country_name="China")

    df_result = my_prophet_model.fit(df_train).predict(df_test)
    # 读取真实值
    df_result = df_test.merge(
        df_result,
        left_on="ds",
        right_on="ds",
        how="left"
    )
    # 计算mape & 存储预测日期
    df_result["predict_dt"] = pd.to_datetime(today)
    df_result["mape"] = np.abs(df_result["y"] - df_result["yhat"]) / df_result["y"]
    return df_result, df_train, my_prophet_model
"""
全月份预测 - 单仓单货物
"""
def predict_all_month(df_single_goods_norm,
               start_date_list = ["2023-03-01","2023-04-01","2023-05-01","2023-06-01","2023-07-01","2023-08-01","2023-09-01"],
               train_size = 60,
               test_size = 30):
    #训练集&测试集
    result_norm_large = pd.DataFrame()
    earlist_record_date = df_single_goods_norm["ds"].min()
    for date in start_date_list:
        #至少留两条
        if pd.to_datetime(date) < earlist_record_date + timedelta(2): continue
        #prophet
        df_result_norm,df_train_norm, my_model_norm = my_prophet(
            df_single_goods_norm,
            today = date,
            train_size = train_size,
            test_size = test_size,
            change_point_range = 0.99,
            change_point_scaler = 0.01)
        #存储结果
        result_norm_large = pd.concat([result_norm_large,df_result_norm])
    return result_norm_large

"""
各仓各货物汇总预测
"""
def predict_all(yy,
                start_date_list=["2023-03-01", "2023-04-01","2023-05-01", "2023-06-01", "2023-07-01", "2023-08-01","2023-09-01"],
                train_size=60,
                test_size=30):
    result_norm_dict = {}
    # 筛选要用的时间段数据
    yy_test = yy[
        (yy["ds"] >= pd.to_datetime(min(start_date_list)) - timedelta(days=train_size)) &
        (yy["ds"] <= pd.to_datetime(max(start_date_list)) + timedelta(days=test_size))
        ]
    for (wh_dept_id, goods_id), group_df in yy_test.groupby(["wh_dept_id", "goods_id"]):
        print(wh_dept_id, goods_id)
        # prophet
        result_norm_large = predict_all_month(
            group_df,
            start_date_list,
            train_size=train_size,
            test_size=test_size)
        # store
        result_norm_dict[(wh_dept_id, goods_id)] = result_norm_large
    return result_norm_dict


# 预测执行模块
def all_model_predict_save(tt,
                           execute_date=["2023-03-01"],
                           test_size=30):
    # 60天训练prophet
    prophet_result_dict_all_60_30 = predict_all(tt,
                                                start_date_list=execute_date,
                                                train_size=60,
                                                test_size=test_size)
    # 90天训练prophet
    prophet_result_dict_all_90_30 = predict_all(tt,
                                                start_date_list=execute_date,
                                                train_size=90,
                                                test_size=test_size)
    # 四种滑动平均
    slide_window_result_all_30 = slide_window_summary(
        tt,
        start_date_list=execute_date,
        window_list=[1, 7, 14, 21, 28],
        test_size=test_size)

    final_result = pd.DataFrame()
    # 遍历各仓各货物
    for key in prophet_result_dict_all_60_30.keys():
        # prophet模型抓取
        df_prophet_60_all = prophet_result_dict_all_60_30[key][["ds", "wh_dept_id", "goods_id",
                                                                "yhat", "y", "predict_dt"]].copy()
        df_prophet_90_all = prophet_result_dict_all_90_30[key][["ds", "wh_dept_id", "goods_id",
                                                                "yhat", "y", "predict_dt"]].copy()
        # 添加model名列
        df_prophet_60_all["model"] = "prophet_60"
        df_prophet_90_all["model"] = "prophet_90"
        final_result = pd.concat([final_result, df_prophet_60_all, df_prophet_90_all])
        for window in slide_window_result_all_30.keys():
            df_window = slide_window_result_all_30[window][key][[
                "ds", "wh_dept_id", "goods_id", f"yhat_{window}_window", "y", "predict_dt"
            ]].rename(columns={f"yhat_{window}_window": "yhat"}).copy()
            # 模型名字列
            df_window["model"] = f"window_{window}"
            final_result = pd.concat([final_result, df_window])

    final_result = final_result.rename(columns={"ds": "predict_dt",
                                                "yhat": "predict_demand",
                                                "predict_dt": "dt"})

    final_result = final_result.sort_values(by=["predict_dt", "goods_id", "wh_dept_id"]).reset_index(drop=True)
    # 输出格式变更
    final_result["goods_id"] = final_result["goods_id"].astype(int)
    final_result["wh_dept_id"] = final_result["wh_dept_id"].astype(int)

    final_result['dt'] = final_result['dt'] - timedelta(days=1)
    final_result['dt'] = final_result['dt'].dt.strftime('%Y-%m-%d')

    final_result['predict_dt'] = final_result['predict_dt'].dt.strftime('%Y-%m-%d')

    final_result['predict_demand'] = final_result['predict_demand'].round(1)

    return final_result

"""
销量预测未来30天
"""
def concurrent_predict(yy,
                       tt,
                       execute_date_list = ["2023-03-01","2023-04-01","2023-05-01","2023-06-01", "2023-07-01","2023-08-01", "2023-09-01"],
                       test_size = 30
                      ):
    """
    并行计算常规品和全量
    """
    import concurrent.futures

    #常规品
    def calculate_final_result_norm(yy, execute_date_list, test_size):
        final_result_norm = all_model_predict_save(yy, execute_date=execute_date_list, test_size=test_size)
        return final_result_norm
    #全量
    def calculate_final_result_all(tt,execute_date_list, test_size):
        final_result_all = all_model_predict_save(tt,execute_date=execute_date_list, test_size=test_size)
        return final_result_all

    if __name__ == "__main__":

        # Create a ThreadPoolExecutor with 2 worker threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_norm = executor.submit(calculate_final_result_norm, yy, execute_date_list, test_size)
            future_all = executor.submit(calculate_final_result_all, tt, execute_date_list, test_size)

            final_result_norm = future_norm.result()
            final_result_all = future_all.result()
    """
    销量预测表合并，按执行日期dt存
    """
    for dt in final_result_norm["dt"].unique():
        bip3_save_df2(
            final_result_norm[final_result_norm["dt"] == dt][
                ["predict_dt", "wh_dept_id", "goods_id", "predict_demand", "dt", "model"]
            ],
            table_folder_name='model_summary_normal',
            bip_folder='model/basic_predict_promote',
            output_name=f"normal_predict_model_summary",
            folder_dt=dt
        )
        bip3_save_df2(
            final_result_norm[final_result_norm["dt"] == dt][
                ["predict_dt", "wh_dept_id", "goods_id", "predict_demand", "dt", "model"]
            ],
            table_folder_name='model_summary_all',
            bip_folder='model/basic_predict_promote',
            output_name=f"all_predict_model_summary",
            folder_dt=dt
        )
    return final_result_norm,final_result_all
