from config.config import *
from utils.decorator import *
from utils.msg_utils import Message, MSG_GROUP_FIVE

BASE = f'/user/yanxin.lu/sc/wh_monitor/dt={today}/all/'


@log_wecom('库存监控推送', P_TWO)
def run():
    df = spark.read_csv(BASE)
    Message.send_file(df=df, file_name=f'{today}_库存监控.csv', group=MSG_GROUP_FIVE)


if __name__ == '__main__':
    run()
