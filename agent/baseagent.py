from util import ACT, OBS, RET, INFO, DEBUG
import numpy as np
import torch
import time

class BaseAgent():
    def policy(self, state: OBS) -> (ACT):
        raise NotImplementedError("policy method not implemented")
    
class HumanAgent(BaseAgent):
    def policy(self,state:OBS) -> (ACT):
        print("Action: {0: 'Left', 1:'Right', 2:'Up', 3:'Down'}")
        print("Action Alias: {0: 'a', 1:'d', 2:'w', 3:'s'}")
        action = input("Please input your action:").replace("a","0").replace("d","1").replace("w","2").replace("s","3")
        while not action.isdecimal() or not int(action) in [0,1,2,3]:
            print("Invalid Action")
            action = input("Please input your action:").replace("a","0").replace("d","1").replace("w","2").replace("s","3")
        return int(action)

    

class RandomAgent(BaseAgent):
    def __init__(self,type:int =0):
        self.type = type
        if (type>0):
            self._move_kernel ={
                0: torch.tensor([[0,0,0],[1,0,0],[0,0,0]],dtype=torch.float32).flip(dims=(-2,-1)).reshape(1,1,3,3),
                1: torch.tensor([[0,0,0],[0,0,1],[0,0,0]],dtype=torch.float32).flip(dims=(-2,-1)).reshape(1,1,3,3),
                2: torch.tensor([[0,1,0],[0,0,0],[0,0,0]],dtype=torch.float32).flip(dims=(-2,-1)).reshape(1,1,3,3),
                3: torch.tensor([[0,0,0],[0,0,0],[0,1,0]],dtype=torch.float32).flip(dims=(-2,-1)).reshape(1,1,3,3),
            }
        
    def policy(self,state:OBS) -> (ACT):
        # time.sleep(0.5)
        action = np.random.randint(0,4)
        if(self.type>0):
            while not (self._check_action(state,action)):
                action = np.random.randint(0,4)
        return action
    
    def _check_action(self,state:OBS,action:ACT) -> bool:
        # if the action is valid, return True
        # else return False
        next_head = torch.conv2d(state[0].unsqueeze(0), 
                                    self._move_kernel[action],
                                    padding=1)[0]
        hit_wall = torch.abs(torch.sum(next_head))<1e-6
        hit_body = torch.sum(next_head*state[2])>0
        return not (hit_wall or hit_body)

