#!/bin/bash
# Task: 41437， 7:00触发

. /projects/luckin_purchase_jupyter/schedule/base.sh
# 库存监控结果推送
python /projects/luckin_purchase_jupyter/monitor/wh_monitor_push.py