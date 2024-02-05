from prophet import Prophet
import numpy as np
from config.paths import *
from utils.file_utils import *
from utils.decorator import *
from prophet.plot import add_changepoints_to_plot

# 冷热杯消耗
cup_consume_path = f'{INPUT_BASE}/wh_cup_consume_his_daily/{yesterday}/'
# today = datetime(year = 2021,month = 10 ,day = 24).date()
# yesterday = today-timedelta(days = 1)
one_year_ago = today - timedelta(days=365)

# 预测结果本地路径
OUTPUT_RESULT_LOCAL_PATH = f"{OUTPUT_BASE}/forecast/season_factor/{today}/"
OUTPUT_IMG_PATH = f"{OUTPUT_IMG_BASE}/season_factor/{cur_month}/{today}/"
# 预测结果HDFS路径
DMD_FORECAST_RESULT_HDFS_PATH = f"{HDFS_BASE}/factor/season/{today}/"

# 参考周期
INPUT_LENGTH = 785
# 预测周期
OUTPUT_LENGTH = 120


class SeasonFactor:
    @staticmethod
    def parameter_tune(dt):
        #     冷转热时期 -->tau降低+mode为加法：减小下降趋势 ；n升高：提升季节跟随
        if dt.month in [8, 9, 10, 11]:
            tau = 0.01  # 趋势调整参数
            n = 12
            mode = 'additive'  # 季节性计算方式
        #  其他区间
        else:
            tau = 0.05
            n = 10
            mode = 'multiplicative'
        return tau, n, mode

    @staticmethod
    def load_data(df, dt_start, dt_end, wh_id) -> DataFrame:
        """
        加载数据
        :param dt_start: 起始日期
        :param dt_end: 结束日期
        :param wh_id: 仓库wh_dept_id
        :return: 训练时间序列
        """
        df = df.query(f"(dt >= '{dt_start}') and (dt <= '{dt_end}') and (wh_id == {wh_id})")[["dt", "hot_weight"]]
        df.dt = df.dt.apply(lambda x: datetime.strptime(x, '%Y-%m-%d').date())
        df.columns = ["ds", "y"]
        return df

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
        dts = cj_2019.union(cj_2021).union(cj_2022)
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
        dt_2 = pd.date_range('2021-01-20', '2021-02-10', freq='D')
        dts = dt_1.union(dt_2)
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
        hols = pd.DataFrame()
        for i in range(hol.shape[0]):
            hol_item = pd.DataFrame({
                'holiday': 'h' + str(hol.iloc[i, 0]),
                'ds': pd.to_datetime(eval(hol.iloc[i, 1])),
                'lower_window': 0,
                'upper_window': 0,
            })
            hols = pd.concat([hols, hol_item])
        holidays = pd.concat([holidays_workday, hols], ignore_index=True)
        return holidays

    def get_special_day(self) -> DataFrame:
        """
        指定特殊日期
        :return: 特殊日期及影响范围
        """

        # 新冠疫情
        covid = pd.DataFrame({
            'holiday': 'xg',
            'ds': pd.to_datetime(['2020-01-23']),
            'lower_window': 0,
            'upper_window': 7 + 28 + 15,
        })

        # 公司暴雷
        company = pd.DataFrame({
            'holiday': 'market',
            'ds': pd.to_datetime(['2020-04-03']),
            'lower_window': 0,
            'upper_window': 6,
        })
        # 节假日
        holidays = self.get_holiday()

        special_days = pd.concat([covid, company, holidays])

        return special_days

    def prophet_train(self, df_train, periods, changepoint_prior_scale=0.05,
                      holidays_prior_scale=10, seasonality_prior_scale=10,
                      seasonality_mode='multiplicative', mcmc_samples=0):
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
        # 逻辑回归参数cap/floor
        # 切两刀：上下限区间宽度【是否能合理覆盖所有趋势因素；海南/哈尔滨/沈阳】/下限水平【气候原因下限制低；南宁/南昌/海南】
        diff = float(df.loc[df.ds >= one_year_ago, 'y'].max()) - float(df.loc[(df.y != 0), 'y'].min())
        df['floor'] = (float(df.loc[(df.y != 0), 'y'].min()) - 0.03) if ((float(df.loc[(df.y != 0), 'y'].min()) < 0.07) or (diff < 0.7)) else 0.1
        df['cap'] = float(df.loc[df.ds >= one_year_ago, 'y'].max())
        m = Prophet(growth='logistic',
                    holidays=holidays,
                    holidays_prior_scale=holidays_prior_scale,
                    changepoint_prior_scale=changepoint_prior_scale,
                    yearly_seasonality=False,
                    seasonality_prior_scale=seasonality_prior_scale,
                    weekly_seasonality=False,
                    seasonality_mode=seasonality_mode,
                    mcmc_samples=mcmc_samples)
        # 自定义季节性，fourier_order:越大越容易过拟合，周季节性默认值3，年季节性默认值10
        # m.add_seasonality(name='weekly_on_season', period=7, fourier_order=3, condition_name='on_season')
        # m.add_seasonality(name='weekly_off_season', period=7, fourier_order=3, condition_name='off_season')

        m.add_seasonality(name='yearly_on_season', period=365, fourier_order=3, condition_name='on_yearly')
        m.add_seasonality(name='yearly_off_season', period=365, fourier_order=3, condition_name='off_yearly')
        m.fit(df)

        future = m.make_future_dataframe(periods=periods)
        # 逻辑回归参数cap/floor
        future['floor'] = (float(df.loc[(df.y != 0), 'y'].min()) - 0.03) if ((float(df.loc[(df.y != 0), 'y'].min()) < 0.07) or (diff < 0.7)) else 0.1

        future['cap'] = float(df.loc[df.ds >= one_year_ago, 'y'].max())
        # future['on_season'] = future['ds'].apply(self.is_weekly_season)
        # future['off_season'] = ~future['ds'].apply(self.is_weekly_season)

        future['on_yearly'] = future['ds'].apply(self.is_yearly_season)
        future['off_yearly'] = ~future['ds'].apply(self.is_yearly_season)

        forecast = m.predict(future)

        return m, forecast

    def check_forecast_result(self, df, wh_id, dt_end, series_length, forecast_length):
        """
        训练模型+降级方案
        :param wh_id: 仓库wh_dept_id
        :param dt_end: 当前日期
        :param series_length: 参考历史序列长度
        :param forecast_length: 预测未来长度
        :return:
        """
        end = dt_end
        start = end - timedelta(days=series_length)
        tau, n, mode = self.parameter_tune(today)
        mcmc_samples = 0
        logger.info(f"train {wh_id}: {start} -> {end} \n changepoint_prior_scale = {tau}, seasonality_prior_scale = {n}, seasonality_mode = {mode}, mcmc_samples = {mcmc_samples}")
        df_train = self.load_data(df, start, end, wh_id)

        test = 100000
        while (test > 0.9) & (mcmc_samples <= 40):
            if test == 100000:
                pass
            else:
                logger.info(f"test == {test} > 0.5, MAP parameter search failed, mcmc_sample initiating, mcmc_samples == {mcmc_samples}")
            m, forecast = self.prophet_train(
                df_train=df_train,
                periods=forecast_length,
                changepoint_prior_scale=tau,
                seasonality_prior_scale=n,
                seasonality_mode=mode,
                mcmc_samples=mcmc_samples)

            # 过去7天样本均值
            past_7d_y = df_train.loc[df_train.ds >= end - timedelta(days=1 * 7), 'y'].mean()
            # 未来3天预测值均值
            next_3d_yhat = forecast.loc[(forecast.ds <= np.datetime64(end + timedelta(days=3))) & (forecast.ds > np.datetime64(end)), 'yhat'].mean()
            # 误差超过0.5即触发降级
            test = abs((next_3d_yhat - past_7d_y) / past_7d_y)
            # 兜底方案 --> mcmc_samples = 40
            mcmc_samples += 40
        logger.info(f"forecast {wh_id}: {today} -> {today + timedelta(days=forecast_length)}")
        return m, forecast

    @staticmethod
    def exp_decide(df):
        # 转为float类型
        df = df.astype('float')
        des = df.describe()
        Q1 = des["25%"]
        Q3 = des["75%"]
        QR = Q3 - Q1
        exp_line_up = Q3 + 1.5 * QR
        exp_line_low = Q1 - 1.5 * QR
        # print(Q1, Q3, QR, exp_line_up, exp_line_low)
        df_flg = df.apply(lambda x: 1 if x > exp_line_up or x < exp_line_low else 0)

        ret = len(df_flg[df_flg == 1]) / len(df_flg)
        print(ret)
        return df_flg

    @log_wecom('季节因子', P_TWO)
    def run(self):
        logger.info("开始计算季节因子")
        # 获取仓库信息
        wh_df = pd.read_parquet(WH_INFO_PATH)
        wh_df = pd.concat([wh_df, pd.DataFrame.from_records([{'wh_dept_id': -1, 'wh_name': '全国'}])], ignore_index=True)
        wh_df.columns = ["wh_id", "wh_name"]
        # 剔除兰州仓库
        wh_skip = [329233, 326919]

        # 输出结果
        wh_predict = pd.DataFrame()
        df_base = pd.read_parquet(cup_consume_path)
        # 分仓预测
        for wh in wh_df["wh_id"].tolist():
            if wh not in wh_skip:
                if wh_df[wh_df["wh_id"] == wh].shape[0] == 0:
                    break
                wh_name = wh_df[wh_df["wh_id"] == wh].iloc[0, 1]
                m, forecast = self.check_forecast_result(df_base, wh, yesterday, INPUT_LENGTH, OUTPUT_LENGTH)
                img_file = f"{OUTPUT_IMG_PATH}{wh_name}_{wh}.png"
                folder_check(img_file)
                fig = m.plot(forecast)
                add_changepoints_to_plot(fig.gca(), m, forecast)
                fig.savefig(img_file)
                forecast['wh_id'] = wh
                forecast = forecast[['ds', 'wh_id', 'yhat']].rename(columns={'ds': 'dt'})
                forecast_truncate = forecast.tail(OUTPUT_LENGTH)
                # 热饮占比 预测结果
                wh_predict = pd.concat([wh_predict, forecast_truncate])

        # 取预测值
        # 保存本地
        save_file(wh_predict, OUTPUT_RESULT_LOCAL_PATH + "hot_weight_predict.csv")
        # 上传HDFS
        save_hdfs(wh_predict, DMD_FORECAST_RESULT_HDFS_PATH, 'hot_weight_predict.csv')

        # 计算增长系数
        # 前21天平均占比
        last_21 = today - timedelta(days=21)
        logger.info(f"计算增长系数，计算历史21天热饮占比均值（{last_21}）")
        df_21_org = df_base.query(f"(dt >= '{last_21}') and (dt < '{today}')")
        df_21_filter = pd.DataFrame(columns=['wh_id', 'dt', 'hot_weight'])

        for index, val in df_21_org.groupby(['wh_id']):
            # 通过分位数剔除异常值
            val["mark"] = self.exp_decide(val['hot_weight'])
            val = val[val["mark"] == 0]
            val = val[['wh_id', 'dt', 'hot_weight'] + ['mark']]
            df_21_filter = pd.concat([df_21_filter, val])

        df_21 = df_21_filter.groupby("wh_id").agg({"hot_weight": "mean"}).reset_index()
        df_21.columns = ["wh_id", "yhat_base"]

        df = pd.merge(wh_predict, df_21, on="wh_id", how="left")

        df["increase_ratio"] = df[["yhat_base", "yhat"]].apply(lambda x: round((x[0] - x[1]) / (1 - x[0]), 5), axis=1)
        # 冷饮变化系数 [(1-yhat) - (1-yhat_base)] / (1-yhat_base)
        df['cold_ratio'] = (df['yhat_base'] - df['yhat']) / (1 - df['yhat_base'])
        # 热饮变化系数
        df['hot_ratio'] = (df['yhat'] - df['yhat_base']) / df['yhat_base']

        # 增长因子
        increase_ratio = OUTPUT_RESULT_LOCAL_PATH + "increase_ratio.csv"
        save_file(df, increase_ratio)
        save_hdfs(df, DMD_FORECAST_RESULT_HDFS_PATH, 'increase_ratio.csv')
        logger.info("季节因子计算完成了!")


SeasonFactor().run()
