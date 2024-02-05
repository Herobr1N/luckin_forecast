from config.config import *
from config.paths import HDFS_BASE, HDFS_BASE_TEST
from utils.decorator import log_file
import glob
from plotly.graph_objects import Figure


def folder_check(file_path: str) -> None:
    """
    判断路径是否存在，若不存在则创建
    :param file_path: 文件路径
    :return: None
    """
    folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, 0o755)
        logger.info(f'创建文件夹:{folder_path}')


@log_file
def save_file(df: DataFrame, file_path: str) -> str:
    """
    保存DataFrame到文件，支持CSV，Parqeut
    :param df: DataFrame
    :param file_path: 文件绝对路径
    :return: 保存路径
    """
    folder_check(file_path)
    if file_path.endswith('.csv'):
        df.to_csv(file_path, index=False, header=True, encoding='utf_8_sig')
    elif file_path.endswith('.parquet'):
        df.to_parquet(file_path)
    else:
        raise Exception('仅支持 CSV, Parquet文件')
    return file_path


@log_file
def save_hdfs(data: DataFrame, hdfs_path=None, file_name='jupyter.csv') -> str:
    """
    保存结果至HDFS
    :param data: DataFrame
    :param hdfs_path: HDFS路径
    :param file_name: 文件名称
    :return: 保存路径
    """

    if not IS_PROD & hdfs_path.startswith(HDFS_BASE):
        hdfs_path = hdfs_path.replace(HDFS_BASE, HDFS_BASE_TEST)
    spark.save_hdfs(data, hdfs_path, file_name)
    return hdfs_path + file_name


@log_file
def save_img(fig, img_path='random.png') -> str:
    """
    保存文件
    :param fig: 图像对象, 支持Figure or PIL.Image
    :param img_path: 图像路径
    :return: 图像路径
    """
    from PIL import Image
    if img_path == 'random.png':
        import uuid
        random_name = uuid.uuid4().hex
        img_path = f'/data/cache/temp/{random_name}.png'
    folder_check(img_path)
    if isinstance(fig, Figure):
        fig.write_image(img_path, scale=2, engine='orca')
    elif isinstance(fig, Image.Image):
        fig.save(img_path)
    return img_path


def read_folder(folder_path: str) -> DataFrame:
    """
    读取文件夹下的所有CSV文件
    :param folder_path: 文件夹路径
    :return: DataFrame
    """
    df_list = []
    for file in glob.glob(folder_path + "*.csv"):
        temp = pd.read_csv(file)
        df_list.append(temp)
    res = pd.concat(df_list)
    return res


def remove(path: str):
    """
    删除文件or文件夹
    :param path: 文件路径
    """
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            import shutil
            shutil.rmtree(path)
    return None


@log_file
def df2Excel(data_path: str, data_list: list, sheet_name_list: list):
    """
    将多个dataframe 保存到同一个excel的不同sheet
    :param data_path: 需要保存的文件地址及文件名
    :param data_list: 需要保存到excel的dataframe
    :param sheet_name_list: 每个sheet的名称
    :return:
    """

    write = pd.ExcelWriter(data_path, date_format='str')
    for da, sh_name in zip(data_list, sheet_name_list):
        da.to_excel(write, sheet_name=sh_name, index=False)
    write.save()