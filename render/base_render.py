import numpy as np
from util import ACT, OBS, RET, INFO, DEBUG

class BaseRender():
    def render(self, state: OBS)->None: 
        raise NotImplementedError
    
    
class TextRender(BaseRender):
    def __init__(self, cls=True):
        self.cls = cls
    def render(self, state: OBS)->None: 
        # print("Current State:")
        if self.cls:
            print('\033c',end='')
        shape=state[0].shape 
        print("-----"*(5+shape[1]))
        for x,y in np.ndindex(shape):
            if y == 0:
                print("|", end="")
            if state[0,x,y] == 1:
                print("H", end="")
            elif state[1,x,y] == 1:
                print("F", end="")
            elif state[2,x,y]>0:
                print("B", end="")
            else:
                print(" ", end="")
            if y == shape[1]-1:
                print("|")
        
        print("-----"*(5+shape[1]))
        print("F for Food, H for Head, B for Snake Body")