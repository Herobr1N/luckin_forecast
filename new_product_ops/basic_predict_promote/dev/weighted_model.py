# /========================================
# File Name: weighted_model.py
# Abstract:
# Version:
# Author: Yang Zhuoshi
# Email: zhuoshi.yang@lkcoffee.com
# Date: 2023/10/16 11:55
# ========================================/
import pandas as pd

from __init__ import project_path

from utils_offline.a00_imports import DayStr, bip3_save_df

f"Import from {project_path}"


class WeightedModel:
    """
    通过加权融合的方式将各基础模型进行结果融合作为新的基础模型，
    其权重来源于过去3个月每个模型的平均acc值的softmax结果
    """
    def __init__(self, pred_calc_day: str, predict_res: pd.DataFrame, acc_res: pd.DataFrame):
        self.pred_calc_day = pred_calc_day
        self.predict_res = predict_res
        # 预测结果表，仓库-货物维度
        # ['predict_dt', 'wh_dept_id', 'goods_id', 'model', 'predict_demand']
        self.acc_res = acc_res
        # 过去3个月模型平均准确率表，仓库-货物维度
        # ['wh_dept_id', 'goods_id', 'model', 'last_3_acc']

    def _calc_weight(self):
        """
        Description
        -----------
        计算仓库-货物维度各模型权重
        """

        self.acc_res['total_acc'] = self.acc_res.groupby(['wh_dept_id', 'goods_id'])['last_3_acc'].transform('sum')
        self.acc_res['weight'] = self.acc_res['last_3_acc'] / self.acc_res['total_acc']
        selected_columns = ['wh_dept_id', 'goods_id', 'model', 'weight']
        # ['wh_dept_id', 'goods_id', 'model', 'weight']
        self.acc_res = self.acc_res[selected_columns]
        return

    def _calc_res(self):
        """
        Description
        -----------
        计算各模型融合结果

        Returns
        -------
        1. df [pd.DataFrame]
            加权融合每日预测结果
            ['predict_dt', 'wh_dept_id', 'goods_id', 'predict_demand']
        """
        df = self.predict_res.merge(self.acc_res, on=['wh_dept_id', 'goods_id', 'model'], how='left')
        # ['predict_dt', 'wh_dept_id', 'goods_id', 'model', 'predict_demand', 'weight']
        df['weight'].fillna(0, inplace=True)
        df['predict_demand'] = df['predict_demand'] * df['weight']
        df = df.groupby(['predict_dt', 'wh_dept_id', 'goods_id'], as_index=False)['predict_demand'].sum()
        # ['predict_dt', 'wh_dept_id', 'goods_id', 'predict_demand']
        return df

    def main(self):
        """
        Description
        -----------
        主函数
        """
        self._calc_weight()
        df = self._calc_res()
        bip3_save_df(df,
                     table_folder_name='test',
                     folder_dt=self.pred_calc_day,
                     bip_folder='test')
        return


def main(pred_calculation_day=None):
    """
    Description
    -----------
    对外暴露主函数

    Parameters
    ----------
    1. pred_calculation_day [Str] Default: None
        # 需要计算的日期，默认为空
    """
    pred_calc_day = DayStr.get_dt_str(pred_calculation_day)
    instance = WeightedModel(pred_calc_day, pd.DataFrame(), pd.DataFrame())
    instance.main()
    return
