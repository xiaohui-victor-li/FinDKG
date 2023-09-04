import errno
import logging
import os
from datetime import datetime

from DKG import settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(module)s.%(funcName)s] %(message)s')

sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)


class FileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        mkdir_p(os.path.dirname(filename))
        logging.FileHandler.__init__(self, filename, mode, encoding, delay)


def mkdir_p(path):
    try:
        os.makedirs(path, exist_ok=True)  # Python>3.2
    except TypeError:
        try:
            os.makedirs(path)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else: raise


def get_log_root_path(graph_name, log_root_dir, result_root=settings.RESULT_ROOT):
    return os.path.join(result_root, graph_name, log_root_dir)


def add_logger_file_handler(graph_name, method_name, log_root_dir, log_time=None, fname_prefix=""):
    """log file path: RESULT_ROOT/{graph_name}/{method_name}/{graph_name}_{method_name}_{cur_time}.log"""
    if log_time is None:
        log_time = datetime.now().strftime("%y%m%d_%H%M%S")

    log_fname = f"{fname_prefix}{graph_name}_{method_name}_{log_time}.log"
    log_fpath = os.path.join(get_log_root_path(graph_name, log_root_dir), log_fname)

    file_handler = FileHandler(log_fpath)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
