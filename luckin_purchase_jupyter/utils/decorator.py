from config.config import ENV, logger
from functools import wraps


def log_wecom(task_type='default', mention_list: list = None):
    from utils.msg_utils import Message
    import traceback

    if mention_list is None:
        mention_list = list()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kw):
            try:
                res = func(*args, **kw)
                Message.send_msg(f'【{ENV}】\n✅ {task_type}任务运行成功')
                return res
            except:
                error_detail = traceback.format_exc()
                if len(error_detail) > 1500:
                    error_detail = error_detail.split('\n')[-2]
                logger.exception('程序异常', exc_info=True)
                msg = f'【{ENV}】\n❌任务失败\n类型：{task_type}\n{error_detail}'
                Message.send_msg(msg, mention_list)

        return wrapper

    return decorator


def log_file(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        logger.info(f'保存成功【{res}】')
        return res
    return wrapper
