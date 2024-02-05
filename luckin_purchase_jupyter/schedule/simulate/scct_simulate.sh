#!/bin/bash
# Task: 41573 已废弃

. /projects/luckin_purchase_jupyter/schedule/base.sh

# 控制塔库存模拟
python /projects/luckin_purchase_jupyter/stock_simulate/scct_simulate.py
