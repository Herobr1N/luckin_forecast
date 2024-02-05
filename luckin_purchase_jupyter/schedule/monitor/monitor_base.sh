#!/bin/bash
# Task: 41235, 9:00触发

. /projects/luckin_purchase_jupyter/schedule/base.sh
# 基础销量预测监控
python /projects/luckin_purchase_jupyter/monitor/base_predict.py
# VLT、MOQ配置检查
python /projects/luckin_purchase_jupyter/monitor/vlt_moq_config_check.py
# 控制塔-业务告警
#python /projects/luckin_purchase_jupyter/monitor/tower_cg_cost_monitor.py
# 新品入仓
python /projects/luckin_purchase_jupyter/monitor/new_product_wh_arrival_monitor.py
# 椰云监控
#python /projects/luckin_purchase_jupyter/monitor/bis/duan_cang.py
# 过滤商品计划监控
python /projects/luckin_purchase_jupyter/monitor/skip_plan.py
