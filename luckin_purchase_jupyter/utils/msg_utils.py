from utils.file_utils import save_file, save_img, folder_check
from config.config import *
import requests
import base64
import hashlib
from plotly.graph_objects import Figure

MSG_GROUP_TEST = '【指标监控-自用】'
MSG_GROUP_ONE = '新品冷启动采购配置群'
MSG_GROUP_TWO = '【规划-算法】供应链业务沟通群'
MSG_GROUP_THREE = '监控-测试'
MSG_GROUP_FOUR = '小鹿智能采购配置监控'
MSG_GROUP_FIVE = '供应链规划组'


class Message:

    @staticmethod
    def get_url(group=MSG_GROUP_TEST):
        url = 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=930c0943-ae89-48dd-8f79-f4203c7fa835'
        if group == MSG_GROUP_ONE:
            url = 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=228d5fb3-40fc-451b-abfb-065445138ae8'
        elif group == MSG_GROUP_TWO:
            url = 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=3fd43103-3077-4909-8403-825eba17c6a9'
        elif group == MSG_GROUP_THREE:
            url = 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=57ac898e-313c-4aa7-bf4e-f6a0e407f251'
        # prod 小鹿智能采购配置监控
        # elif group == MSG_GROUP_FOUR:
        #     url = 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=7d3e82b9-7527-4b03-95d1-70ed561d45d8'
        elif group == MSG_GROUP_FIVE:
            url = 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=483ac2e6-c317-4d88-b460-aa00a52d17e2'
        return url

    @classmethod
    def send_file(cls, df: DataFrame, file_name, group=MSG_GROUP_TEST) -> None:
        """
        以文件形式发送消息，要求文件大小在5B~20M之间
        :param df: 消息内容Table格式
        :param file_name: 文件名称
        :param group: 消息群组
        :return: None
        """
        file_path = f'/data/cache/temp/{file_name}'
        save_path = save_file(df, file_path)
        key = cls.get_url(group).split('=')[-1]
        id_url = "https://qyapi.weixin.qq.com/cgi-bin/webhook/upload_media?key=%s&type=file" % key
        data = {'media': open(save_path, 'rb')}
        msg = requests.post(url=id_url, files=data)
        dict_data = msg.json()
        media_id = dict_data['media_id']
        data = {'msgtype': 'file',
                'file': {'media_id': media_id}
                }
        res = requests.post(url=cls.get_url(group), json=data)
        logger.info("## file 【%s】 send %s ##" % (media_id, res))

    @classmethod
    def send_image(cls, fig, group=MSG_GROUP_TEST) -> None:
        """
        以图片形式发送消息，图片（base64编码前）最大不能超过2M，支持JPG,PNG格式
        :param fig: Figure
        :param group: 消息群组
        :return: None
        """
        image_path = save_img(fig)
        with open(image_path, 'rb') as handler:
            base64_data = base64.b64encode(handler.read())
            base64_data = base64_data.decode()
        with open(image_path, 'rb') as handler:
            fmd5 = hashlib.md5(handler.read()).hexdigest()

        data = {
            "msgtype": "image",
            "image": {"base64": base64_data, "md5": fmd5}
        }

        headers = {"Content-Type": "text/plain"}
        res = requests.post(cls.get_url(group), json=data, headers=headers)
        logger.info("## image 【%s】 send %s ##" % (image_path, res))

    @classmethod
    def send_msg(cls, msg, mention_list: list = None, group=MSG_GROUP_TEST) -> None:
        """
        发送文本消息
        :param msg: 消息内容，最长不超过2048个字节
        :param mention_list: 企业微信userid 或者手机号码列表，提醒群中的指定成员(@某个成员)，@all表示提醒所有人
        :param group: 发送群组
        :return: None
        """
        if mention_list is None:
            mention_list = list()
        headers = {"Content-Type": "application/json"}
        text_info = {
            "msgtype": "text",
            "text": {
                "content": msg,
                "mentioned_mobile_list": list(mention_list),
                "mentioned_list": list(mention_list)
            }
        }
        res = requests.post(cls.get_url(group), headers=headers, json=text_info)
        logger.info("## msg 【%s】 send %s ##" % (msg, res))

    @classmethod
    def __construct_email(cls, header, msg, receiver, cc_receiver, accessory_path, accessory_name) -> None:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        from email.header import Header

        message = MIMEMultipart()
        message['Subject'] = Header(header, 'utf-8')
        message['From'] = MAIL_USER
        message['To'] = ','.join(receiver)
        message['Cc'] = ','.join(cc_receiver)
        message.attach(MIMEText(msg, 'html', 'utf-8'))

        att1 = MIMEText(open(accessory_path, 'rb').read(), 'base64', 'utf-8')
        att1["Content-Type"] = 'application/octet-stream'
        att1.add_header('Content-Disposition', 'attachment', filename=accessory_name)
        message.attach(att1)
        try:
            smtp = smtplib.SMTP(host=MAIL_HOST)
            smtp.connect(host=MAIL_HOST, port=MAIL_PORT)
            smtp.starttls()
            smtp.login(MAIL_USER, MAIL_PWD)
            smtp.sendmail(MAIL_SENDER, receiver, message.as_string())
        except smtplib.SMTPException as e:
            logger.error(e)

    @classmethod
    def send_email(cls, df, to_emails=None, cc_emails=None, msg=None, header=None) -> None:
        """
        发送邮件
        :param df: DataFrame
        :param to_emails: 收件人邮箱List
        :param cc_emails: 抄送人邮箱List
        :param msg: 邮箱正文内容
        :param header: 邮箱标题
        :return: None
        """
        if cc_emails is None:
            cc_emails = []
        if to_emails is None:
            to_emails = []
        import uuid
        random_name = uuid.uuid4().hex
        file_path = f'/data/cache/temp/{random_name}.xlsx'
        folder_check(file_path)
        df.to_excel(file_path, engine='xlsxwriter', index=False, encoding='utf-8')
        cls.__construct_email(header, msg, to_emails, cc_emails, file_path, f"{header}_{today}.xlsx")
        logger.info(f"Email-{header} 发送成功")
