#!/bin/bash
# Task: 41233

. /projects/luckin_purchase_jupyter/schedule/base.sh
# 数据同步
python /projects/luckin_purchase_jupyter/forecast/data_syc.py
# 牛奶咖啡豆
python /projects/luckin_purchase_jupyter/forecast/dmd_forecast_long_cycle.py
