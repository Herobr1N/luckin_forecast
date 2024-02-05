#!/bin/bash
# Task: 41234

. /projects/luckin_purchase_jupyter/schedule/base.sh
# 数据同步
python /projects/luckin_purchase_jupyter/forecast/data_syc.py

# 季节因子
python /projects/luckin_purchase_jupyter/forecast/season_factor.py
