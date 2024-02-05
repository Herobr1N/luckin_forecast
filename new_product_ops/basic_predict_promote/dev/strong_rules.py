# /========================================
# File Name: strong_rules.py
# Abstract:
# Version:
# Author: Yang Zhuoshi
# Email: zhuoshi.yang@lkcoffee.com
# Date: 2023/10/18 17:16
# ========================================/
import pandas as pd

from __init__ import project_path

from utils_offline.a00_imports import DayStr, bip3_save_df, read_api, bip3

f"Import from {project_path}"


class StrongRules:
    """
    强规则模块，用于筛选出违反强规则的仓库-货物-模型对，后续加权模型和模型融合时不再使用
    当前的强规则为预测值不能为负数，否则就记录该仓库-货物-模型对
    """
    def __init__(self, pred_calc_day: str):
        self.pred_calc_day = pred_calc_day
        self.model_summary_normal = read_api.read_dt_folder(bip3("model/basic_predict_promote",
                                                                 "model_summary_normal"),
                                                            self.pred_calc_day)
        self.model_summary_all = read_api.read_dt_folder(bip3("model/basic_predict_promote",
                                                              "model_summary_all"),
                                                         self.pred_calc_day)

    @staticmethod
    def _filter_negative(df: pd.DataFrame):
        """
        Description
        -----------
        筛选有负数的仓库-货物-模型对，输出违反规则的数据表

        Parameters
        ----------
        1. df [pd.DataFrame]
            # 需要被筛选的数据表

        Returns
        -------
        1. df [pd.DataFrame]
            违反规则的数据表
        """
        df = df[df['predict_demand'] < 0]
        df.drop_duplicates(subset=['wh_dept_id', 'goods_id', 'model'], inplace=True)
        selected_columns = ['wh_dept_id', 'goods_id', 'model']
        df = df[selected_columns]
        return df

    def main(self):
        """
        Description
        -----------
        主函数
        """
        self.model_summary_normal = self._filter_negative(self.model_summary_normal)
        self.model_summary_all = self._filter_negative(self.model_summary_all)
        bip3_save_df(self.model_summary_normal,
                     table_folder_name='model_summary_normal_unuseful',
                     folder_dt=self.pred_calc_day,
                     bip_folder="model/basic_predict_promote")
        bip3_save_df(self.model_summary_all,
                     table_folder_name='model_summary_all_unuseful',
                     folder_dt=self.pred_calc_day,
                     bip_folder="model/basic_predict_promote")
        return


def main(pred_calculation_day=None):
    """
    Description
    -----------
    对外暴露主函数

    Parameters
    ----------
    1. pred_calculation_day [Str] Default: None
        # 需要被计算的日期
    """
    pred_calc_day = DayStr.get_dt_str(pred_calculation_day)
    instance = StrongRules(pred_calc_day)
    instance.main()
    return


if __name__ == '__main__':
    main()
