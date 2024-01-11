import util.io_tool
from util.io_tool import INFO, DEBUG, setup_logging, setup_seed
from util.tb_logger import TBLogger as _logger
from util.cfg import load_config, save_config
from util.type  import *

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
    "ACT","OBS","RET"]


