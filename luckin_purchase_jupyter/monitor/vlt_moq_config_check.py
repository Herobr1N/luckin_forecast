from utils.decorator import *
from utils.msg_utils import *
from dim_info.dim_base import Dim


# today = datetime(year = 2022,month = 1,day  = 27).date()
# yesterday = today - timedelta(days = 1)

class vlt_moq_check:
    def __init__(self, test=True):
        if test:
            self.urlGroup = MSG_GROUP_TEST
        else:
            self.urlGroup = MSG_GROUP_THREE

    @staticmethod
    def missing_config():
        """
        vlt&moq&供应商配置缺失项
        :return: dataframe
        """
        # rtdw_dim.rt_dim_goods_spec_city_config_d_his _配置实时表
        # 查找配置项
        goods_city_config = spark.sql(f"""
            SELECT
                dt
                , goods_spec_id
                , warehouse_id
                , ifnull(vlt, -1) AS vlt
                , ifnull(supplier_mid, -1) AS supplier_mid
                , ifnull(minimum_delivery, -1) AS minimum_delivery
            FROM dw_dim.`dim_goods_spec_city_config_d_his`
            WHERE  dt = '{yesterday}' --test date
                AND city_purchase_status = 1
                AND delete_flag = 1
                AND org_code = '0101'
                AND goods_spec_id IS NOT NULL
                AND warehouse_id IS NOT NULL
        """)
        # 有效仓库信息
        wh_info = Dim.dim_warehouse()
        auto_cg_large_class = ('零食', '日耗品', '工服类',
                               '营销物料', '原料', '包装类', '轻食', '器具类', '办公用品')
        # 0024：三明治 0020：卷类 0071：半成品 0025：没有这个编号
        exclude_ls = ('TC0071', 'TC0025', 'TC0024', 'TC0020')
        # 有效规格
        valid_spec = spark.sql(f"""
            SELECT DISTINCT
                dt
                , goods_name
                , IFNULL(goods_id , -1)
                , spec_id
                , spec_name
                , large_class_name
                , small_class_name
            FROM
                dw_dim.dim_stock_spec_d_his
            WHERE dt = '{yesterday}'
                AND spec_id IS NOT NULL
                AND spec_status = 1
                AND goods_status = 1
                AND is_spec_deleted = 0
                AND small_class_code not in {exclude_ls}
                AND large_class_name in {auto_cg_large_class}
        """)
        goods_city_config = goods_city_config.rename(columns={'warehouse_id': 'wh_id',
                                                              'goods_spec_id': 'spec_id',
                                                              'vlt': 'VLT',
                                                              'minimum_delivery': 'MOQ',
                                                              'supplier_mid': '供应商'})
        mid1 = goods_city_config.merge(wh_info, on=['wh_id'], how='inner')
        mid = mid1.merge(valid_spec, on=['spec_id', 'dt'], how='inner')
        moq_missing = mid.loc[(mid['MOQ'] == -1) | (mid['VLT'].isin([-1, 0])) | (mid['供应商'] == -1)]
        return moq_missing

    def generate_warning_msg(self, df, columns=['MOQ', 'VLT', '供应商']):
        """
        合并、构造报警信息
        :param df:
        :param columns:
        :return: list
        """
        df_check = df[columns].isin([-1, 0])
        return [
            ''.join([f'{df_check.columns[i]}未配置 ' if df_check.iloc[x, i] else '' for i in range(df_check.shape[1])])
            for x in range(df_check.shape[0])]

    def get_contact(self, df):
        """
        匹配缺失配置项负责人
        :param moq_missing:
        :return: dataframe
        """
        contact_emp = spark.sql("""
        SELECT
        -- 供应商商品信息表
            name
            , contact_emp_name
        FROM lucky_srm.`t_supplier_commodity_info` srm_cmdty_info
        INNER JOIN
        -- 供应商信息表
            (
                SELECT
                    id
                    ,supplier_code
                    ,supplier_name
                    ,enterprise_id
                FROM lucky_srm.`t_supplier_info`
                WHERE org_code = 0101
                 AND supplier_code not in ('07', 'SPL00028') --去掉屏南
                    AND mdm_supplier_status = 1
                    AND cooperation_status = 0) supplier_info
            ON srm_cmdty_info.enterprise_id = supplier_info.enterprise_id

        """)
        contact_emp = contact_emp.rename(columns={'name': 'spec_name'})
        spec_contact = df.merge(contact_emp, on=['spec_name'], how='left')
        # 兜底
        spec_contact['contact_emp_name'] = spec_contact['contact_emp_name'].fillna('李瑞')
        return spec_contact

    @log_wecom('vlt_moq配置检查', P_TWO)
    def check_n_send(self):
        """
        检查配置项是否有缺失：是则推送监控，否则打印log
        :return:
        """
        moq_missing = self.missing_config()
        # 检查是否有配置缺失
        if len(moq_missing) == 0:
            Message.send_msg(f"{yesterday}_VLT&MOQ配置检查:无配置缺失", group=self.urlGroup)
        else:
            moq_missing.loc[:, '报警信息'] = self.generate_warning_msg(moq_missing)
            moq_missing_contact = self.get_contact(moq_missing)
            contact_emp_ls = moq_missing_contact['contact_emp_name'].drop_duplicates().to_list()
            moq_send = moq_missing_contact[['goods_name', 'spec_name', 'wh_name', '报警信息', 'contact_emp_name']]
            moq_send = moq_send.rename(columns={'goods_name': '货物名称',
                                                'spec_name': '规格名称',
                                                'wh_name': '仓库名称',
                                                '报警信息': '报警信息',
                                                'contact_emp_name': '联系人'}).sort_values(
                ['联系人', '规格名称', '仓库名称']).drop_duplicates()

            Message.send_file(moq_send, f'{yesterday}_VLT&MOQ配置检查.csv', group=self.urlGroup)
            Message.send_msg('请相关负责人尽快补全以上采购系统配置缺失项', group=self.urlGroup)


if __name__ == '__main__':
    vlt_moq_check(False).check_n_send()
