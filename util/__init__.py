import util.io_tool
import argparse
from util.io_tool import INFO, DEBUG, setup_logging, setup_seed
from util.tb_logger import TBLogger as _logger
from util.cfg import load_config, save_config
from util.type  import *

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else: 
        raise argparse.ArgumentTypeError('Boolean value expected.')

def TBLogger()->_logger:
    INFO("LOG_DIR: ",util.io_tool.LOG_DIR)
    return _logger(util.io_tool.LOG_DIR)
    ...

__all__=[
    "INFO","DEBUG",
    "setup_logging",
    "setup_seed",
    "TBLogger",
    "load_config","save_config"
    "ACT","OBS","RET","MASK",
    "str2bool"
    ]


