import logging
import os.path
import pickle

import pandas as pd
from matplotlib import pyplot
import sys

import numpy as np
import psutil
import torch
from memory_profiler import LogFile, profile


# def create_memory_profiler_logger(logger_name, log_file_name, logging_level, log_format):
#     # create logger
#     logger = logging.getLogger(logger_name)
#     logger.setLevel(logging_level)
#
#     # create file handler which logs even debug messages
#     fh = logging.FileHandler(log_file_name)
#     fh.setLevel(logging_level)
#
#     # create formatter
#     formatter = logging.Formatter(log_format)
#     fh.setFormatter(formatter)
#
#     # add the handlers to the logger
#     logger.addHandler(fh)
#     sys.stdout = LogFile(logger_name)
#
#
# @profile
# def f1():
#     l = [5] * int(1e8)
#     for i in range(2000):
#         l.extend([10]*int(1e7))
#         mem = psutil.virtual_memory()[2]
#         print(f'i = {i} , mem = {mem}%')

def plot_memory_usage(train_meta_data_pkl, plot_core_name):
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    logger.info(f"""Virtual Memory total {train_meta_data_pkl["virtual_mem_tot_gb"]}""")
    logger.info(f"""Swap Memory total {train_meta_data_pkl["swap_mem_tot_gb"]}""")
    pyplot.xlabel("train_iterations")
    pyplot.ylabel("Virtual_mem_percentage")
    pyplot.title(
        f"""Percentage of virtual memory usage vs iterations, tot virtual memory = 
        {np.round(train_meta_data_pkl["virtual_mem_tot_gb"],1)} GB""")
    pd.Series(train_meta_data_pkl["iter_virtual_mem_perc"]).plot()
    pyplot.savefig(os.path.join('plots', f'vmem_iter_{plot_core_name}.png'))


if __name__ == '__main__':
    logger_name = f'memory_logger'
    log_file_name = f'memory_profiling_logs/test_memory_profiler.log'
    logging_level = logging.DEBUG
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging_level, format=FORMAT)
    # # create_memory_profiler_logger(logger_name, log_file_name, logging_level, FORMAT)
    # f1()
    # # l = []
    # # for j in range(10):
    # #     l.append(torch.tensor(np.ones(int(1e7))))
    # # mb_const = 1024*1024
    # #
    # # print(sys.getsizeof(l)/mb_const)
    # # for k in l:
    # #     print(sys.getsizeof(k)/mb_const)

    train_meta_data_dir = "train_meta_data"
    train_meta_data_filename = "train_meta_data_sum_sin_xj_pow_2_n_iter_300_2022-03-20T13:40:55.920454.pkl"
    train_meta_data_pkl_filepath = os.path.join(train_meta_data_dir, train_meta_data_filename)
    plot_core_name = train_meta_data_filename.replace("train_meta_data_", "").replace(".pkl", "")
    train_meta_data_pkl = pickle.load(open(train_meta_data_pkl_filepath, 'rb'))
    plot_memory_usage(train_meta_data_pkl=train_meta_data_pkl, plot_core_name=plot_core_name)
