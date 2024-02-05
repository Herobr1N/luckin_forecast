from utils.file_utils import save_file, read_folder, remove, save_hdfs
from utils.decorator import *
from config.paths import *
import multiprocessing
from config.config import *


INPUT_CG_HDFS_PATH = f'{HDFS_BASE}/transfer/cg_simulate_goods_base_daily/{yesterday}/'
INPUT_PLAN_HDFS_PATH = f'{HDFS_BASE}/transfer/plan_simulate_goods_base_daily/{yesterday}/'
OUTPUT_CG_LOCAL_PATH = f'{OUTPUT_BASE}/simulate/cg/{today}/'
OUTPUT_CG_HDFS_PATH = f'{HDFS_BASE}/simulate/cg/{today}/'
OUTPUT_PLAN_LOCAL_PATH = f'{OUTPUT_BASE}/simulate/plan/{today}/'
OUTPUT_PLAN_HDFS_PATH = f'{HDFS_BASE}/simulate/plan/{today}/'


class Thread(multiprocessing.Process):
    def __init__(self, thread_name: str, data: DataFrame, output: str):
        multiprocessing.Process.__init__(self)
        self.thread_name = thread_name
        self.df = data
        self.output = output

    def run(self):
        """
        模拟报损（goods_id为spec_id）：
        按批次旧-->新 依次进行消耗
        报损量 = (批次总量-截止报损日的消耗总量) if 批次总量>截止报损日的消耗总量 else 0
        :return:
        """
        logger.info(f"{self.thread_name} Start")
        result = pd.DataFrame(columns=["dt", "wh_dept_id", "goods_id", "beg", "end", "dmd", "transit", "batch_amount", "batch_loss"])

        i = 0
        for (k1, k2), group in self.df.groupby(["wh_dept_id", "goods_id"]):
            # 第几天：[0,120]
            j = 0
            # j-1期末/j日期初
            last_end = 0
            # !!!截止第j天【尚未到达允收期的最老批次】已消耗量
            buffer = 0
            for row in group.itertuples():
                j = j + 1
                i = i + 1
                # 批次损耗量
                batch_loss = 0
                # 报损量计算
                if float(getattr(row, "batch_amount")) != 0:
                    if float(getattr(row, "batch_amount")) >= buffer:
                        batch_loss = float(getattr(row, "batch_amount")) - buffer
                        buffer = 0
                    else:
                        batch_loss = 0
                        buffer = buffer - float(getattr(row, "batch_amount"))

                if j == 1:
                    # 期末 = 起初+在途-需求-报损
                    end = float(getattr(row, "beg")) + float(getattr(row, "transit")) - float(
                        getattr(row, "dmd")) - batch_loss
                    beg = float(getattr(row, "beg"))
                else:
                    # 期末 = 起初+在途-需求-报损
                    end = float(last_end) + float(getattr(row, "transit")) - float(getattr(row, "dmd")) - batch_loss
                    # 今日期末转为下一日期初
                    beg = last_end

                end = max(float(end), 0)

                last_end = end
                result.loc[i] = [str(getattr(row, "dt"))] + [k1, k2] + [beg] + [end] + [float(getattr(row, "dmd")),
                                                                                        float(getattr(row, "transit")),
                                                                                        float(getattr(row, "batch_amount")),
                                                                                        batch_loss]
                buffer = buffer + float(getattr(row, "dmd"))
        result = result.fillna(0)
        save_file(result, self.output + f'{self.thread_name}.csv')
        logger.info(f"{self.thread_name} END")


def start(sim_type, input_path, output_path, hdfs_path):
    @log_wecom(f'库存模拟-{sim_type}')
    def simulate():
        wh_goods_base_daily = spark.read_parquet(input_path)
        wh_goods_base_daily['dt'] = wh_goods_base_daily['dt'].astype('str')
        wh_goods_base_daily.sort_values(['wh_dept_id', 'goods_id', 'dt'], ascending=[0, 1, 1], inplace=True)
        wh_ls = wh_goods_base_daily['wh_dept_id'].drop_duplicates()
        ths = []
        for wh_id in wh_ls:
            wh_goods_base_daily_thn = wh_goods_base_daily.query(f"wh_dept_id == {wh_id}")
            thread = Thread("wh_{0}".format(wh_id), wh_goods_base_daily_thn, output_path)
            thread.daemon = True
            thread.start()
            ths.append(thread)
        for th in ths:
            th.join()

        # 合并保存
        result_file = output_path + 'result.csv'
        remove(result_file)
        res_df = read_folder(output_path)
        save_hdfs(data=res_df, hdfs_path=hdfs_path, file_name='result.parquet')

    simulate()


if __name__ == '__main__':
    logger.info('采购计划库存模拟开始')
    start('采购计划', INPUT_PLAN_HDFS_PATH, OUTPUT_PLAN_LOCAL_PATH, OUTPUT_PLAN_HDFS_PATH)
    logger.info('自动CG库存模拟开始')
    start('自动CG', INPUT_CG_HDFS_PATH, OUTPUT_CG_LOCAL_PATH, OUTPUT_CG_HDFS_PATH)
