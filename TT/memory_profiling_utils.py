import logging
import sys

from memory_profiler import LogFile


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