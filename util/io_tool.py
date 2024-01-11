import time
import os
import torch
import sys
import os.path as osp
from absl import flags
import logging
import numpy as np
import random
from pprint import pprint

DEBUG_ON = True

def __get_time_idx():
    return time.strftime("%d%H%M%S", time.localtime())

def INFO(*args,pp=False):
    """
    print information to console and log file
    pp: pretty print
    """
    # join args to a string
    sent = " ".join([str(arg) for arg in args])
    if pp:
        for arg in args:
            pprint(arg)
    else:
        print(sent)
    logging.info(sent)

def DEBUG(*args,pp=False):
    """
    print debug information to console and log file
    pp: pretty print
    """
    if not DEBUG_ON: return
    sent = "DEBUG: " + " ".join([str(arg) for arg in args])
    if pp:
        print("DEBUG: ", end="\n")
        for arg in args:
            pprint(arg)
    else:
        print(sent)
    logging.debug(sent)
    # logging.info(sent)

def setup_logging(expname="n"):
    idx = __get_time_idx()
    filename = osp.join("result",f"{expname}_"+str(idx),"run.log")
    
    if not osp.exists(osp.dirname(filename)):
        os.makedirs(osp.dirname(filename))

    logger = logging.getLogger()
    file_handler = logging.FileHandler(filename)
    formatter = logging.Formatter('[%(levelname)s]%(asctime)s: %(message)s', datefmt='%Y%m%d-%H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.addHandler(file_handler)

    logging.info(f"----------{time.asctime()}----------")
    

    logging.info(f"Sys Argv: {' '.join(sys.argv)}")
    return osp.dirname(filename)

def setup_seed(seed:int=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False