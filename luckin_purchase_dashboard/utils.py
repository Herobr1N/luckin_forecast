import logging.config
from datetime import datetime, timedelta
from meta import Meta

logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

spark = Meta('yanxin.lu')

today = datetime.today().date()
