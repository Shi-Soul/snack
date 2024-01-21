import torch 
from typing import Tuple , Union

ONEDIM = torch.Tensor
ACT = ONEDIM
POL = torch.Tensor
# OBS = np.ndarray
OBS = torch.Tensor
REW = ONEDIM
DONE = ONEDIM
RET = Tuple[OBS, REW, DONE]