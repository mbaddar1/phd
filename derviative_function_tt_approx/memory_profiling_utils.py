import logging
import sys

import numpy as np
import psutil
import torch
from memory_profiler import LogFile, profile


def create_memory_profiler_logger(logger_name, log_file_name, logging_level, log_format):
    # create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging_level)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_file_name)
    fh.setLevel(logging_level)

    # create formatter
    formatter = logging.Formatter(log_format)
    fh.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(fh)
    sys.stdout = LogFile(logger_name)


@profile
def f1():
    l = [5] * int(1e8)
    for i in range(2000):
        l.extend([10]*int(1e7))
        mem = psutil.virtual_memory()[2]
        print(f'i = {i} , mem = {mem}%')


if __name__ == '__main__':
    logger_name = f'memory_logger'
    log_file_name = f'memory_profiling_logs/test_memory_profiler.log'
    logging_level = logging.DEBUG
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    # create_memory_profiler_logger(logger_name, log_file_name, logging_level, FORMAT)
    f1()
    # l = []
    # for j in range(10):
    #     l.append(torch.tensor(np.ones(int(1e7))))
    # mb_const = 1024*1024
    #
    # print(sys.getsizeof(l)/mb_const)
    # for k in l:
    #     print(sys.getsizeof(k)/mb_const)
    # # 0.00017547607421875