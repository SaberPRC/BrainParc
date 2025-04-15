import os
import sys
import ants
import torch
import shutil
import logging

from logging import handlers
from itertools import product


def logging_init(name=None, PARENT_DIR=None):
    if PARENT_DIR==None:
        PARENT_DIR = os.path.split(os.path.realpath(__file__))[0]  # 父目录
    LOGGING_DIR = os.path.join(PARENT_DIR, "log")  # 日志目录
    LOGGING_NAME = "test"  # 日志文件名

    LOGGING_TO_FILE = True  # 日志输出文件
    LOGGING_TO_CONSOLE = False  # 日志输出到控制台

    LOGGING_WHEN = 'D'  # 日志文件切分维度
    LOGGING_INTERVAL = 1  # 间隔少个 when 后，自动重建文件
    LOGGING_BACKUP_COUNT = 15  # 日志保留个数，0 保留所有日志
    LOGGING_LEVEL = logging.DEBUG  # 日志等级
    LOGGING_suffix = "%Y.%m.%d.log"  # 旧日志文件名

    # 日志输出格式
    LOGGING_FORMATTER = "%(levelname)s - %(asctime)s - %(message)s"
    if not os.path.exists(LOGGING_DIR):
        os.makedirs(LOGGING_DIR)

    logger = logging.getLogger()
    logger.setLevel(LOGGING_LEVEL)
    formatter = logging.Formatter(LOGGING_FORMATTER)

    if name != None:
        filename = os.path.join(LOGGING_DIR, name)
    else:
        filename = os.path.join(LOGGING_DIR, LOGGING_NAME)

    if LOGGING_TO_FILE:
        file_handler = handlers.TimedRotatingFileHandler(filename=filename, when=LOGGING_WHEN, interval=LOGGING_INTERVAL, backupCount=LOGGING_BACKUP_COUNT)
        file_handler.suffix = LOGGING_suffix
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if LOGGING_TO_CONSOLE:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()

def set_initial(cfg):
    if cfg.general.resume_epoch < 0 and os.path.isdir(cfg.general.save_dir):
        print('Found non-empty save dir {}, \n {} to delete all files, {} to continue:'.format(cfg.general.save_dir,
                                                                                            'yes', 'no'), end=' ')
        choice = input().lower()
        if choice == 'yes':
            shutil.rmtree(cfg.general.save_dir)
        elif choice == 'no':
            pass
        else:
            raise ValueError('choice error')

    if not os.path.exists(cfg.general.save_dir):
        os.mkdir(cfg.general.save_dir)
        os.mkdir(os.path.join(cfg.general.save_dir, 'checkpoints'))
        os.mkdir(os.path.join(cfg.general.save_dir, 'pred'))

    if not os.path.isdir(os.path.join(cfg.general.save_dir, 'checkpoints')):
        os.mkdir(os.path.join(cfg.general.save_dir, 'checkpoints'))

    if not os.path.isdir(os.path.join(cfg.general.save_dir, 'pred')):
        os.mkdir(os.path.join(cfg.general.save_dir, 'pred'))


def calculate_patch_index(target_size, patch_size, overlap_ratio = 0.25):
    shape = target_size

    gap = int(patch_size[0] * (1-overlap_ratio))
    index1 = [f for f in range(shape[0])]
    index_x = index1[::gap]
    index2 = [f for f in range(shape[1])]
    index_y = index2[::gap]
    index3 = [f for f in range(shape[2])]
    index_z = index3[::gap]

    index_x = [f for f in index_x if f < shape[0] - patch_size[0]]
    index_x.append(shape[0]-patch_size[0])
    index_y = [f for f in index_y if f < shape[1] - patch_size[1]]
    index_y.append(shape[1]-patch_size[1])
    index_z = [f for f in index_z if f < shape[2] - patch_size[2]]
    index_z.append(shape[2]-patch_size[2])

    start_pos = list()
    loop_val = [index_x, index_y, index_z]
    for i in product(*loop_val):
        start_pos.append(i)
    return start_pos