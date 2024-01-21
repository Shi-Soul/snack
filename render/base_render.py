import numpy as np
import time
from util import ACT, OBS, RET, INFO, DEBUG

class BaseRender():
    def render(self, state: OBS, time_step:int)->None: 
        raise NotImplementedError
    
    
class TextRender(BaseRender):
    def __init__(self, cls=True):
        self.cls = cls
    def render(self, state: OBS,  time_step:int)->None: 
        # print("Current State:")
        shape=state[0].shape 
        render_str =  "-"*(2+shape[1])+"\n"
        # print("-"*(5+shape[1]))
        for x,y in np.ndindex(shape):
            if y == 0:
                # print("|", end="")
                render_str += "|"
            if state[0,x,y] == 1:
                # print("H", end="")
                render_str += "H"
            elif state[1,x,y] == 1:
                # print("F", end="")
                render_str += "F"
            elif state[2,x,y]>0:
                # print("B", end="")
                render_str += "B"
            else:
                # print(" ", end="")
                render_str += " "
            if y == shape[1]-1:
                # print("|") 
                render_str += "|\n"
        
        # print("-"*(5+shape[1]))
        render_str += "-"*(2+shape[1])+"\n"
        time.sleep(0.05)
        if self.cls:
            print('\033c',end='')
        print(render_str, end="")
        print("F for Food, H for Head, B for Snake Body")
        print("Time step: ", time_step)