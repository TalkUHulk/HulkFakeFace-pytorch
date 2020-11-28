import time
import logging

logger = logging.getLogger(__name__)


def timer(func):
    """
    Recording time
    :param func:
    :return:
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        logger.info('{} cost: {}s'.format(func.__name__, time.time() - start_time))
        return res

    return wrapper
