from utils.decorator import *
import numpy as np
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from utils.file_utils import *
from config.paths import *

# WH_INFO_PATH = f'/data/purchase/input/wh_info/2022-02-12/'
# today = datetime(year = 2021,month = 9,day = 24).date()
# yesterday = today - timedelta(days = 1)
wh_inv_consume_path = f"{INPUT_BASE}/wh_inv_consume/"
# 输出图片
OUTPUT_RES_IMG = f"{OUTPUT_IMG_BASE}/long_cycle/{cur_month}/{today}/"

# 预测结果本地路径
OUTPUT_RESULT_LOCAL_PATH = f"{OUTPUT_BASE}/forecast/long_cycle/{today}/"

# 预测结果HDFS路径
DMD_FORECAST_RESULT_HDFS_PATH = f"{HDFS_BASE}/forecast/long_cycle/{today}/"

# 避免重复读取基础数据，加快运行速度
base_df = pd.read_parquet(wh_inv_consume_path, columns=['dt', 'wh_id', 'goods_id', 'shop_actual_consume_cnt_1d'])

# 参数
# historybest：g52 tau= 0.05 input_length= 750
goods_param = {
    52: {
        'input_length': 1450,
        'output_length': 120,
        'changepoints': None,
        # [i for i in ['2021-05-13','2021-07-25','2021-09-13','2021-10-01','2020-12-29','2021-04-03'] if datetime.strptime(i,'%Y-%m-%d').date()<today],
        'changepoint_prior_scale': 0.04,
        'changepoint_range': 0.8

    },
    354: {
        'input_length': 750,
        'output_length': 120,
        'changepoints': [i for i in ['2021-03-16', '2021-05-19', '2021-08-08', '2021-09-13', '2021-10-29', '2020-12-21',
                                     '2021-04-09', '2022-09-01', '2022-11-01'] if (
                                     datetime.strptime(i, '%Y-%m-%d').date() > today - timedelta(
                                 days=750 + 1) and datetime.strptime(i, '%Y-%m-%d').date() < today)],
        'changepoint_prior_scale': 0.05,
        'changepoint_range': 0.95
    }
}


class GoodsForecast:

    @staticmethod
    def read_folder(base_dir: str, dt_start: str, dt_end: str) -> DataFrame:
        """
        根据dt分区读取CSV文件夹
        :param base_dir: 文件夹根目录
        :param dt_start: DT开始
        :param dt_end: DT结束
        :return: Pandas DataFrame
        """
        dts = pd.date_range(dt_start, dt_end, freq='D')
        df_list = []
        for dt in dts:
            path = base_dir + str(dt.date())
            logger.info(path)
            df_list.append(pd.read_parquet(path))
        df_res = pd.concat(df_list)
        return df_res

    def load_data(self, base_df, dt_start, dt_end, goods_id) -> DataFrame:
        # 数据类型处理
        base_df.dt = base_df.dt.astype('datetime64')
        base_df.shop_actual_consume_cnt_1d = base_df.shop_actual_consume_cnt_1d.astype('float')
        base_df = base_df.query(f"goods_id=={goods_id} and dt >= '{dt_start}' and dt <= '{dt_end}'")

        # 全国维度数据
        df_nation = base_df.groupby(['dt'], as_index=False).agg({'shop_actual_consume_cnt_1d': 'sum'})
        df_nation.columns = ['ds', "y"]
        return df_nation, base_df

    @staticmethod
    def is_weekly_season(ds) -> bool:
        """
        自定义weekly周期，判断是否将当前日期加入自定义周期
        :param ds: 当前日期
        :return: True OR False
        """
        ds = pd.to_datetime(ds)
        # 春节
        cj_2019 = pd.date_range('2019-02-04', '2019-02-11', freq='D')
        cj_2021 = pd.date_range('2021-02-11', '2021-02-17', freq='D')
        cj_2022 = pd.date_range('2022-01-31', '2022-02-06', freq='D')
        # 国庆
        gq_2019 = pd.date_range('2019-10-01', '2019-10-07', freq='D')
        gq_2020 = pd.date_range('2020-10-01', '2020-10-07', freq='D')
        gq_2021 = pd.date_range('2021-10-01', '2021-10-07', freq='D')
        gq_2022 = pd.date_range('2022-10-01', '2022-10-07', freq='D')
        dts = cj_2019.union_many([cj_2021, cj_2022, gq_2019, gq_2020, gq_2021, gq_2022])
        return False if list(dts).count(ds) else True

    @staticmethod
    def is_yearly_season(ds) -> bool:
        """
        自定义Yearly周期，判断是否将当前日期加入自定义周期
        :param ds: 当前日期
        :return: True OR False
        """
        ds = pd.to_datetime(ds)
        dt_1 = pd.date_range('2019-01-01', '2019-02-03', freq='D')
        dt_2 = pd.date_range('2021-01-20', '2021-02-18', freq='D')
        # 疫情/与节假日解耦
        dt_3 = pd.date_range('2020-01-01', '2020-04-05', freq='D')
        dts = dt_1.union_many([dt_2, dt_3])
        return False if list(dts).count(ds) else True

    def season_create(self, df) -> DataFrame:
        """
        构造自定义周期
        """
        df['on_season'] = df['ds'].apply(self.is_weekly_season)
        df['off_season'] = ~df['ds'].apply(self.is_weekly_season)

        df['on_yearly'] = df['ds'].apply(self.is_yearly_season)
        df['off_yearly'] = ~df['ds'].apply(self.is_yearly_season)
        return df

    @staticmethod
    def get_holiday():
        # 调休
        change_workday = spark.read_csv(HOLIDAY_WORK_FILE, is_dir=False)
        holidays_workday = pd.DataFrame({
            'holiday': change_workday.iloc[0, 0],
            'ds': pd.to_datetime(eval(change_workday.iloc[0, 1])),
            'lower_window': 0,
            'upper_window': 0,
        })
        # 节假日
        hol = spark.read_csv(HOLIDAY_FILE, is_dir=False)
        hols = [holidays_workday]
        for i in range(hol.shape[0]):
            hol_item = pd.DataFrame({
                'holiday': 'h' + str(hol.iloc[i, 0]),
                'ds': pd.to_datetime(eval(hol.iloc[i, 1])),
                'lower_window': 0,
                'upper_window': 0,
            })
            hols.append(hol_item)
        holidays = pd.concat(hols)

        return holidays

    def prophet_train(self, df_train, periods, changepoints, changepoint_prior_scale=0.03,
                      holidays_prior_scale=10, seasonality_prior_scale=10, changepoint_range=0.8,
                      seasonality_mode='additive'):
        """
        模型训练
        :param df_train: 训练数据
        :param periods: 预测周期
        :param changepoint_prior_scale: 变点影响强度，解决过/欠拟合，越大越容易过拟合，默认0.05
        :param holidays_prior_scale: 节假日影响强度，越小影响越弱，默认10
        :param seasonality_prior_scale: 季节性影响强度，越大越容易过拟合，默认10
        :param seasonality_mode:
        :return: model模型, forecast:预测结果
        """
        holidays = self.get_holiday()
        df = self.season_create(df_train)
        df['cap'] = 1.5 * df['y'].quantile(0.9)
        m = Prophet(growth='logistic',
                    holidays=holidays,
                    changepoints=changepoints,
                    changepoint_range=changepoint_range,
                    holidays_prior_scale=holidays_prior_scale,
                    changepoint_prior_scale=changepoint_prior_scale,
                    yearly_seasonality=False,
                    seasonality_prior_scale=seasonality_prior_scale,
                    weekly_seasonality=False,
                    seasonality_mode=seasonality_mode)
        # 自定义季节性，fourier_order:越大越容易过拟合，周季节性默认值3，年季节性默认值10
        m.add_seasonality(name='weekly_on_season', period=7, fourier_order=3, condition_name='on_season')
        m.add_seasonality(name='weekly_off_season', period=7, fourier_order=3, condition_name='off_season')

        m.add_seasonality(name='yearly_on_season', period=365, fourier_order=10, condition_name='on_yearly')
        m.add_seasonality(name='yearly_off_season', period=365, fourier_order=10, condition_name='off_yearly')
        m.fit(df)

        future = m.make_future_dataframe(periods=periods)
        future['cap'] = 1.5 * df['y'].quantile(0.9)
        future['on_season'] = future['ds'].apply(self.is_weekly_season)
        future['off_season'] = ~future['ds'].apply(self.is_weekly_season)

        future['on_yearly'] = future['ds'].apply(self.is_yearly_season)
        future['off_yearly'] = ~future['ds'].apply(self.is_yearly_season)

        forecast = m.predict(future)

        return m, forecast

    def get_wh_goods_weight(self, df_org) -> DataFrame:
        """
        计算仓库占比
        :param goods_id: 货物ID
        :param dt_start: 参考起始日期
        :param dt_end: 参考结束日期
        :return: 各仓占比
        """
        # 指数滑动加权平均
        df = df_org.groupby(['wh_id']) \
            .ewm(span=6, min_periods=360, adjust=False, ignore_na=False) \
            .agg({'shop_actual_consume_cnt_1d': 'mean'}) \
            .reset_index().dropna()
        # 筛选最近一年记录
        latest_record = df.groupby('wh_id').agg({'level_1': 'max'}).rename(columns={'level_1': 'level_max'})
        df = df.merge(latest_record, on=['wh_id'], how='left') \
            .query("level_1 == level_max")[['wh_id', 'shop_actual_consume_cnt_1d']]

        # 有效仓库
        wh = pd.read_parquet(WH_INFO_PATH).rename(columns={'wh_dept_id': 'wh_id'})

        df = pd.merge(df, wh[["wh_id"]], on=["wh_id"], how="inner")
        df.columns = ["wh_id", "consume"]
        df["flg"] = 1
        weight = df[["flg", "consume"]].groupby("flg").agg("sum").reset_index()
        weight.columns = ["flg", "consume_total"]
        df = pd.merge(df, weight, on=["flg"], how="inner")
        df["weight"] = df["consume"] / df["consume_total"]

        return df[["wh_id", "weight", "flg"]]

    def model_default(self, df_train_org, goods_id: int):
        # 参考周期
        input_length = goods_param.get(goods_id).get('input_length')
        # 预测周期
        output_length = goods_param.get(goods_id).get('output_length')
        changepoints = goods_param.get(goods_id).get('changepoints')
        changepoint_prior_scale = goods_param.get(goods_id).get('changepoint_prior_scale')
        changepoint_range = goods_param.get(goods_id).get('changepoint_range')

        logger.info(f"预测货物:{goods_id}, 参考周期：{input_length}，预测周期：{output_length}")
        dt_start = today - timedelta(days=input_length)
        dt_end = yesterday
        # 构造训练数据
        logger.info(f'提取历史序列：货物{goods_id}，{dt_start}, {dt_end}')

        df_train = df_train_org[['ds', 'y']]
        # 训练
        logger.info(f'开始训练模型:tau={changepoint_prior_scale}')
        model, forecast = self.prophet_train(df_train, output_length, changepoints, changepoint_prior_scale,
                                             changepoint_range)
        logger.info('训练模型结束')
        y_forecast = forecast.iloc[-1 * output_length:, :][["ds", "yhat", "yhat_lower", "yhat_upper"]].reset_index()[
            ["ds", "yhat", "yhat_lower", "yhat_upper"]]
        # y_forecast["yhat_lower"] = y_forecast["yhat_upper"]
        return model, forecast, y_forecast

    @log_wecom('长周期预测', P_TWO)
    def run(self, goods_id: int) -> None:
        # 参考周期-->sample shape
        input_length = goods_param.get(goods_id).get('input_length')
        dt_start = today - timedelta(days=input_length)
        dt_end = yesterday
        df_train, df_all = self.load_data(base_df, dt_start, dt_end, goods_id)
        model, forecast, g1 = self.model_default(df_train, goods_id)
        # 仓库权重
        g1["flg"] = 1
        g1_wh_weight = self.get_wh_goods_weight(df_all)
        # 画图
        file_path = OUTPUT_RES_IMG + f'{goods_id}.png'
        folder_check(file_path)
        fig = model.plot(forecast)
        # 变点可视化
        add_changepoints_to_plot(fig.gca(), model, forecast)
        model.plot_components(forecast)
        fig.savefig(file_path)
        # 分仓计算
        forecast = pd.merge(g1, g1_wh_weight, on="flg")
        forecast["dmd"] = forecast["yhat"] * forecast["weight"]
        #     forecast["dmd"] = forecast["yhat_lower"] * forecast["weight"]

        forecast["goods_id"] = goods_id
        forecast = forecast[["ds", "wh_id", "goods_id", "yhat", "weight", "dmd"]]

        file_name = OUTPUT_RESULT_LOCAL_PATH + f'res_{goods_id}.csv'

        save_file(forecast, file_name)
        save_hdfs(forecast, DMD_FORECAST_RESULT_HDFS_PATH, f'res_{goods_id}.csv')
        # return forecast


if __name__ == '__main__':
    gf = GoodsForecast()
    gf.run(354)
    gf.run(52)
