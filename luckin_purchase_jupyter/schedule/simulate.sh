#!/bin/bash
# Task: 23048

. /projects/luckin_purchase_jupyter/schedule/base.sh
# 数据同步
python /projects/luckin_purchase_jupyter/forecast/data_syc.py
# 采购计划&CG库存模拟
python /projects/luckin_purchase_jupyter/stock_simulate/simulate.py
