"""
Deprecated
"""

"""

action_space = ['Left', 'Right', 'Up', 'Down']
obs_space = (size,size,m) mat
    * m=4
    * type: 0 snake_head, 1 food, 2 snake_body, 3 empty;
    * hidden var: snake_length, snake_tail
reward: 1, -1, 0 (eat, die, move)
done  : timeout, die

die: hit wall, hit body 
"""
from typing import Union, NewType, Tuple
import numpy as np
# import torch

ACT = int
OBS = np.ndarray
# OBS = torch.Tensor
REWARD = int
DONE = int
RET = Tuple[OBS, REWARD, DONE]

class SnakeEnv():
    def __init__(self, init_length = 5, size=10, max_step=100):
        self.init_length = init_length
        self.size = size
        self.max_step = max_step
        
        self.state = None 
        self.time_step = None 
        self.current_length = None
        
        self._move_kernel ={
            0: np.array([[0,0,0],[1,0,0],[0,0,0]]),
            1: np.array([[0,0,0],[0,0,1],[0,0,0]]),
            2: np.array([[0,1,0],[0,0,0],[0,0,0]]),
            3: np.array([[0,0,0],[0,0,0],[0,1,0]]),
        }
        
        self._edge_mat = np.zeros((self.size, self.size))
        self._edge_mat[0,:] = 1
        self._edge_mat[-1,:] = 1
        self._edge_mat[:,0] = 1
        self._edge_mat[:,-1] = 1
        
        self.reset()
        
    def reset(self) -> RET:
        self.time_step = 0
        self.current_length = self.init_length
        
        obs = np.zeros((self.size,self.size,3))
        obs = self._generate_init_head(obs, self.init_length)
        obs = self._generate_food(obs)
        
        self.state = obs
        reward = 0 
        done = 0
        
        return (obs, reward, done)
    
    def step(self, action: ACT) -> RET: #TODO
        """
        time_step++
        
        if head hit wall or body:
            state = old_state
            reward=-1, done=1
            (wait for runner to reset)
        elif: head hit food:
            remove food
            snake_length++
            snake move
            generate new food
            reward=1, done=0
        else:
            snake move
            reward=0, done=0
            
        if time_step == max_step:
            done=1

        """
        return (obs, reward, done)
        
    def _generate_food(self, state: OBS) -> OBS:
        x, y = np.random.randint(0, self.size, 2)
        while (state[x,y,2] > 1):
            x, y = np.random.randint(0, self.size, 2)
        state[x,y,1] = 1
        return state
    
    def _generate_init_head(self, state: OBS, init_length: int) -> OBS:
        x, y = np.random.randint(0, self.size, 2)
        state[x,y,0] = 1
        state[x,y,2] = init_length
        return state
    
    def _move(self, state: OBS, action: ACT) -> OBS: #TODO:
        # Assert head won't hit wall or body
        
        # Head move: conv with kernel
        # FIXME: Need some way to conv2d
        state[:,:,0] = np.conv2d(state[:,:,0], self._move_kernel[action], mode='same')
        # Body move: score--, check >0
        state[:,:,2] = np.clip(state[:,:,2] - 1, 0, None)
        

        
        
        pass
