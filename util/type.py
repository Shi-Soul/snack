import torch 
from typing import Tuple , Union

ONEDIM = torch.Tensor
ACT = Union[int,ONEDIM]
# OBS = np.ndarray
OBS = torch.Tensor
REW = Union[int,ONEDIM]
DONE = Union[int,ONEDIM]
RET = Tuple[OBS, REW, DONE]