from util import ACT, OBS, RET, INFO, DEBUG
import numpy as np
class BaseAgent():
    def policy(self, state: OBS) -> (ACT):
        raise NotImplementedError("policy method not implemented")
    
class HumanAgent(BaseAgent):
    def policy(self,state:OBS) -> (ACT):
        self._print_state(state)
        print("Action: {0: 'Left', 1:'Right', 2:'Up', 3:'Down'}")
        action = input("Please input your action:")
        while not action.isdecimal() or not int(action) in [0,1,2,3]:
            print("Invalid Action")
            action = input("Please input your action:")
        return int(action)

    def _print_state(self, state: OBS)->None: 
        print("Current State:")
        print("F for Food, H for Head, B for Snake Body")
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
    

class RandomAgent(BaseAgent):
    def policy(self,state:OBS) -> (ACT):
        return np.random.randint(0,4)

