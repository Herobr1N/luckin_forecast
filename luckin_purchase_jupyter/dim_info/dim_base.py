from config.config import spark, today, yesterday
from pandas import DataFrame
from functools import wraps


def dim(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        df = func(*args, **kwargs)
        res = df[list(args)] if len(args) > 0 else df
        return res.copy()

    return wrapper


class Dim:

    @staticmethod
    @dim
    def dim_warehouse(*cols) -> DataFrame:
        """
        仓库维度信息
        :param cols: 列名
        :return: DataFrame
        """
        df = spark.sql('''
                SELECT DISTINCT 
                           wh_dept_id
                           , wh_id
                           , wh_name 
                   FROM dw_ads_scm_alg.dim_warehouse_city_shop_d_his
                   WHERE dt = (SELECT MAX(dt) FROM dw_ads_scm_alg.dim_warehouse_city_shop_d_his) 
            ''')
        return df

    @staticmethod
    @dim
    def get_valid_goods(*cols) -> DataFrame:
        """
        获取有效采购状态的货物
        :param cols: 列名
        :return: DataFrame
        """
        valid_goods = spark.sql("""
            SELECT DISTINCT 
                goods_id
                , goods_name
            FROM dw_ads_scm_alg.dim_wh_goods_spec_purchase_status
            WHERE dt = DATE_SUB(CURRENT_DATE(), 1) AND status = 1
        """)
        return valid_goods

    @staticmethod
    @dim
    def dim_shop_city_warehouse_relation(*cols) -> DataFrame:
        """
        仓库-城市-门店关联信息
        :param cols: 列名
        :return: DataFrame
        """
        df = spark.sql('''
                    SELECT DISTINCT 
                           wh_dept_id
                           , wh_name 
                           , city_id
                           , city_name
                           , shop_dept_id
                           , shop_name
                   FROM dw_ads_scm_alg.dim_warehouse_city_shop_d_his
                   WHERE dt = (SELECT MAX(dt) FROM dw_ads_scm_alg.dim_warehouse_city_shop_d_his) 
                  
                ''')
        return df

    @staticmethod
    @dim
    def dim_goods(*cols) -> DataFrame:
        """
        货物维度信息
        :param cols: 货物相关列名
        :return: DataFrame
        """
        df = spark.sql("""
            SELECT DISTINCT
                goods_name
                , goods_id
                , large_class_name
                , small_class_name
            FROM
                dw_dim.dim_stock_spec_d_his
            WHERE dt = DATE_SUB(CURRENT_DATE(), 1)
        """)
        return df

    @staticmethod
    @dim
    def dim_goods_spec(*cols) -> DataFrame:
        """
        获取货物-规格维度信息
        :param cols:
        :return: DataFrame
        """
        df = spark.sql("""
                    SELECT DISTINCT
                        IFNULL(goods_id , -1) as goods_id
                        , spec_id
                        , spec_name
                    FROM
                        dw_dim.dim_stock_spec_d_his
                    WHERE dt = DATE_SUB(CURRENT_DATE(), 1)
                        AND spec_id IS NOT NULL
                        AND spec_status = 1
                        AND goods_status = 1
                        AND is_spec_deleted = 0
                        """)
        return df


    @staticmethod
    @dim
    def dim_valid_cmdty(*cols) -> DataFrame:
        """
        商品基础信息
        :param cols: 列名
        :return: DataFrame
        """
        cmdty_info = spark.sql('''
            SELECT
                cmdty_id
                , cmdty_name
                , one_category_id
                , one_category_name
                , two_category_id
                , two_category_name
            FROM dw_dim.dim_cmdty_d_his
            WHERE dt = DATE_SUB(CURRENT_DATE(), 1) AND status NOT IN (-1,3)
            --去除无效商品id
            AND cmdty_name NOT LIKE '%弃用%'
        ''' )
        return cmdty_info

    @staticmethod
    @dim
    def get_formula(is_agg = False):
        """
        获取商品所对应配方
        :param national:全国维度/分仓维度配方方案
                is_agg: True：根据商品维度聚合 False：根据sku维度聚合
        :return: DataFrame
        """
        formula = spark.sql("""
            SELECT
                cmdty_id
                , sku_code
                , goods_id
                , goods_name
                , MAX(need_number) AS need_number
            FROM (
                SELECT
                    cmdty_id
                    , sku_code
                    , base.goods_id
                    , goods_info.goods_name
                    , need_number
                FROM (
                    SELECT 
                        cmdty_id
                        , sku_code
                        , goods_id
                        , need_number
                    FROM dw_ads_scm_alg.dim_cmdty_formula_a 
                    WHERE dt = CURRENT_DATE()
                ) base
                LEFT JOIN dw_dim.dim_stock_good_d_his goods_info ON goods_info.dt = DATE_SUB(CURRENT_DATE(), 1) AND base.goods_id = goods_info.goods_id
            )
            GROUP BY cmdty_id, sku_code, goods_id, goods_name
        """)
        agg = formula.groupby(['cmdty_id', 'goods_id', 'goods_name'], as_index = False).agg({'need_number':'max'})
        return agg if is_agg else formula

    @staticmethod
    @dim
    def get_band(*cols) -> DataFrame:
        """
        货物Band分级结果
        :param cols: Band相关列名
        :return: DataFrame
        """
        band = spark.sql("""
            SELECT 
                goods_id
                , goods_name
                , result_level
            FROM dw_ads_scm_alg.`dm_goods_band_country`
            WHERE dt = (SELECT MAX(dt) FROM dw_ads_scm_alg.`dm_goods_band_country`)
        """)
        return band



    @staticmethod
    def get_wecom_ids(emp_no=None, email=None, name=None):
        """
        通过员工工号/邮箱/姓名获取企业微信号
        :param emp_no: 员工工号
        :param email: 员工邮箱
        :param name: 员工姓名
        :return: list
        """
        condition = ''
        if emp_no is not None:
            if type(emp_no) != list:
                raise TypeError("emp_no must be list!")
            else:
                condition = f" AND emp_no == '{emp_no[0]}'" if len(emp_no) == 1 else f" AND emp_no IN {tuple(emp_no)}"

        elif email is not None:
            if type(email) != list:
                raise TypeError("email must be list!")
            else:
                condition = f" AND email == '{email[0]}'" if len(email) == 1 else f" AND email IN {tuple(email)}"

        elif name is not None:
            if type(name) != list:
                raise TypeError("name must be list!")
            else:
                condition = f" AND name == '{name[0]}'" if len(name) == 1 else f" AND name IN {tuple(name)}"

        df = spark.sql(f"""
                        SELECT
                            lk_info.emp_no
                            , lk_info.id AS emp_id
                            , lk_info.name AS emp_name
                            , lk_info.email AS emp_email
                            , wx_info.qywx_user_id AS wx_user_id
                            , wx_info.qywx_mobile AS wx_mobile
                        FROM lucky_entwechat.`t_user_sync` wx_info
                        LEFT JOIN lucky_ehr.t_ehr_employee lk_info ON wx_info.ehr_emp_no = lk_info.emp_no
                        WHERE 1 = 1 {condition}
                    """)

        return df['wx_user_id'].astype('str').drop_duplicates().tolist()
