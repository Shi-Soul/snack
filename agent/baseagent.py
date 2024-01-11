from util import ACT, OBS, RET, INFO, DEBUG
import numpy as np
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
    def policy(self,state:OBS) -> (ACT):
        time.sleep(0.1)
        return np.random.randint(0,4)

