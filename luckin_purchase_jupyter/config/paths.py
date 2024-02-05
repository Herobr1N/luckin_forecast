from config.config import today, yesterday, IS_PROD

HDFS_BASE = '/projects/luckyml/purchase'
HDFS_BASE_TEST = '/projects/luckyml/purchase/test'

"""本地文件路径"""
INPUT_BASE = '/data/purchase/input'
# 本地默认文件数据路径
OUTPUT_BASE = '/data/purchase/output'
# 本地图片默认输出路径
OUTPUT_IMG_BASE = '/data/jupyter_work_dir/projects/output' if IS_PROD else '/tf/projects/output'


""" Common """
# 仓库信息
WH_INFO_PATH = f'/data/purchase/input/wh_info/{yesterday}/'
HOLIDAY_WORK_FILE = f'{HDFS_BASE}/hand_config/workday.csv'
HOLIDAY_FILE = f'{HDFS_BASE}/hand_config/holiday.csv'
