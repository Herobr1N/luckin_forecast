import sys
from utils.decorator import *
import shutil
from config.config import *

# 仓库库存消耗
wh_inv_consume = {'hdfs': '/projects/luckyml/purchase/transfer/wh_inv_consume/',
                  'local': '/data/purchase/input/wh_inv_consume/'}
# 仓-商品-货物消耗
wh_cmdty_goods_consume = {'hdfs': '/projects/luckyml/purchase/transfer/wh_cmdty_goods_consume/',
                          'local': '/data/purchase/input/wh_cmdty_goods_consume/'}

# 商品售卖
wh_cmdty_sale_daily = {'hdfs': '/user/haoxuan.zou/transfer/wh_cmdty_sale_daily/',
                       'local': '/data/purchase/input/wh_cmdty_sale_daily/'}

# 冷热杯数量
wh_cup_consume_his_daily = {'hdfs': '/projects/luckyml/purchase/transfer/wh_cup_consume_his_daily/',
                            'local': '/data/purchase/input/wh_cup_consume_his_daily/'}

# 仓库信息
wh_info = {'hdfs': '/projects/luckyml/purchase/transfer/wh/', 'local': '/data/purchase/input/wh_info/'}

# 库存模拟配置
simulate_config = {'hdfs': '/projects/luckyml/purchase/transfer/cg_simulate_goods_base_daily/',
                   'local': '/data/purchase/input/cg_simulate_goods_base_daily/'}


class DataSyc:
    """
    HDFS与物理机数据同步
    """

    def __init__(self, dt_start=yesterday, dt_end=yesterday):
        """
        初始化
        :param dt_start: 开始日期，默认T-1
        :param dt_end: 结束日期，默认T-1
        """
        self.spark = Meta(USER)
        self.daily_task_list = [wh_cup_consume_his_daily, wh_info]
        self.monthly_task_list = [wh_inv_consume]
        if dt_start > dt_end:
            raise Exception('起始日期晚于开始日期')
        self.dt_start = dt_start
        self.dt_end = dt_end

    def data_syc(self, hdfs_path: str, local_path: str, reload=False) -> None:
        """
        从HDFS拉取同步数据
        :param hdfs_path: HDFS路径
        :param local_path: 本地路径
        :param reload: 覆盖本地文件夹
        :return:
        """
        logger.info(f'数据同步开始: \n FROM: {hdfs_path} \n TO: {local_path}')
        if reload & os.path.exists(local_path):
            shutil.rmtree(local_path)
            logger.info('本地文件夹已存在，重新覆盖')
        hdfs_dirs = self.spark.hdfs_dir_list(hdfs_path)
        if hdfs_dirs is None:
            logger.error(f'路径【{hdfs_path}】不存在')
        for path in hdfs_dirs:
            res = self.spark.download(path, local_path)
            logger.debug(f'{res} Done')
        logger.info(f'数据同步结束. TOTAL:{len(hdfs_dirs)}')

    @log_wecom('数据同步', P_TWO)
    def daily_syc(self) -> None:
        """
        日粒度同步
        """
        # 天粒度
        for task in self.daily_task_list:
            for dt in pd.date_range(start=self.dt_start, end=self.dt_end, freq='D'):
                day = str(dt.date())
                hdfs = task.get('hdfs') + day
                local = task.get('local') + day
                self.data_syc(hdfs, local)
        # 月粒度
        for task in self.monthly_task_list:
            # 未跨月日常同步
            if self.dt_start == self.dt_end:
                dt = str(self.dt_start.replace(day=1))
                hdfs = task.get('hdfs') + dt
                local = task.get('local') + dt
                self.data_syc(hdfs, local, True)
            else:
                for dt in pd.date_range(start=self.dt_start, end=self.dt_end, freq='MS'):
                    dt = str(dt.replace(day=1).date())
                    hdfs = task.get('hdfs') + dt
                    local = task.get('local') + dt
                    self.data_syc(hdfs, local, True)


if __name__ == '__main__':
    syc = DataSyc(sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else DataSyc()
    syc.daily_syc()
