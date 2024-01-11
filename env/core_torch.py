"""

action_space = ['Left', 'Right', 'Up', 'Down']
obs_space = (m,size,size) mat
    * m=3
    * type: 0 snake_head, 1 food, 2 snake_body, 3 empty;
    * hidden var: snake_length, snake_tail
reward: 1, PENALTY, 0 (eat, die, move)
done  : timeout, die

die: hit wall, hit body 

"""
from typing import Any, Union, NewType, Tuple
import numpy as np
import torch

from util import ACT, OBS, RET, INFO, DEBUG
EPS=1e-6

class SnakeEnv():
    def __init__(self, init_length = 5, size=10, max_step=100, penalty=-10):
        self.init_length = init_length
        self.size = size
        self.max_step = max_step
        self.penalty = penalty
        
        # Current State
        self.state = torch.tensor([])
        self.time_step = 0
        self.current_length = 0
        
        self._move_kernel ={
            0: torch.tensor([[0,0,0],[1,0,0],[0,0,0]],dtype=torch.float32).flip(dims=(-2,-1)).reshape(1,1,3,3),
            1: torch.tensor([[0,0,0],[0,0,1],[0,0,0]],dtype=torch.float32).flip(dims=(-2,-1)).reshape(1,1,3,3),
            2: torch.tensor([[0,1,0],[0,0,0],[0,0,0]],dtype=torch.float32).flip(dims=(-2,-1)).reshape(1,1,3,3),
            3: torch.tensor([[0,0,0],[0,0,0],[0,1,0]],dtype=torch.float32).flip(dims=(-2,-1)).reshape(1,1,3,3),
        }
        
        self._edge_mat = torch.zeros((self.size, self.size))
        self._edge_mat[0,:] = 1
        self._edge_mat[-1,:] = 1
        self._edge_mat[:,0] = 1
        self._edge_mat[:,-1] = 1
        
        self.reset()
        
    def reset(self) -> RET:
        self.time_step = 0
        self.current_length = self.init_length
        
        obs = torch.zeros((3,self.size,self.size))
        obs = self._generate_init_head(obs, self.init_length)
        obs = self._generate_food(obs)
        
        self.state = obs
        reward = 0 
        done = 0
        
        return (obs, reward, done)
    
    def step(self, action: ACT) -> RET:
        """
        time_step++
        
        if head hit wall or body:
            state = old_state
            reward=-1, done=1
            (wait for runner to reset)
        elif: head hit food:
            snake move
            remove food
            snake_length++
            generate new food
            reward=1, done=0
        else:
            snake move
            reward=0, done=0
            
        if time_step == max_step:
            done=1

        """
        self.time_step += 1
        
        state = self.state
        next_head = torch.conv2d(state[0].unsqueeze(0), 
                                    self._move_kernel[action],
                                    padding=1)[0]
        # DEBUG("next_head: \n",next_head)
        hit_wall = torch.abs(torch.sum(next_head))<EPS
        hit_body = torch.sum(next_head*state[2])>0
        if hit_wall or hit_body:
            reward = self.penalty
            done = 1
            # DEBUG(f"hit_wall: {hit_wall}, hit_body: {hit_body}")
            self.state = state
            return (state, reward, done)
        
        # Move body
        state[0] = next_head
        state[2] = np.clip(state[2] - 1, 0, None) + state[0]*self.current_length
        
        # Hit food
        hit_food = torch.sum(next_head*state[1])>0
        if hit_food:
            state[1] = torch.zeros_like(state[1]) # remove food
            state = self._generate_food(state)
            self.current_length += 1
            reward = 1
            done = 0
        else:
            reward = 0
            done = 0
        # DEBUG(f"hit_food: {hit_food}")
        
        if self.time_step == self.max_step:
            done = 1
            # DEBUG(f"timeout")
            
        self.state = state
        return (state, reward, done)
        
    def get_hidden_state(self) -> Tuple[OBS, int, int]:
        return (self.state, self.time_step, self.current_length)
        
    def _generate_food(self, state: OBS) -> OBS:
        x, y = np.random.randint(0, self.size, 2)
        while (state[2,x,y] > 1):
            x, y = np.random.randint(0, self.size, 2)
        state[1,x,y] = 1
        return state
    
    def _generate_init_head(self, state: OBS, init_length: int) -> OBS:
        x, y = np.random.randint(0, self.size, 2)
        state[0,x,y] = 1
        state[2,x,y] = init_length
        return state
    
