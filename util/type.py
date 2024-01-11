import torch 
from typing import Tuple 

ACT = int
# OBS = np.ndarray
OBS = torch.Tensor
REW = int
DONE = int
RET = Tuple[OBS, REW, DONE]