import torch 
from typing import Tuple 

ACT = int
# OBS = np.ndarray
OBS = torch.Tensor
REWARD = int
DONE = int
RET = Tuple[OBS, REWARD, DONE]