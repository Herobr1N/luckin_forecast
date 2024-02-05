from meta import Meta
import yaml
from pathlib import Path
import os
from datetime import datetime, timedelta
import logging.config
import pandas as pd
from pandas import DataFrame
import socket

# 环境切换
IS_PROD = 'luckymlpurchase01-prod-py-vianetm5' == socket.gethostname()
ENV = 'Prod' if IS_PROD else 'Dev'
today = datetime.today().date()
yesterday = today - timedelta(days=1)
cur_month = today.strftime('%Y-%m')

abs_path = str(Path(__file__).parent.absolute())
"""
LOG配置
"""
with open(abs_path + "/log_conf.yml", "r") as f:
    dict_conf = yaml.safe_load(f)

logging.config.dictConfig(dict_conf)
logger_debug = logging.getLogger('logger_std_prod') if IS_PROD else logging.getLogger('logger_std_dev')
# 默认LOGGER
logger = logging.getLogger() if IS_PROD else logger_debug

"""
Email配置
"""
MAIL_USER = "sys_sender@lkcoffee.com"
MAIL_PWD = "QWER1234!@#$"
MAIL_HOST = "smtp.lkcoffee.com"
MAIL_PORT = 587
MAIL_SENDER = "sys_sender@lkcoffee.com"

"""
集群交互用户
"""
USER = 'yanxin.lu' if IS_PROD else 'p_opalgorithm_sc'
spark = Meta(USER)

"""
告警用户配置
"""
P_ALL = ['15201732462', '13604663817'] if IS_PROD else ['']
P_ONE = ['15201732462'] if IS_PROD else ['']
P_TWO = ['13604663817']


