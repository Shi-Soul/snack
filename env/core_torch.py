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
import torch.nn.functional as F
from util import ACT, OBS, RET, ONEDIM, INFO, DEBUG
# ONEDIM: torch.Tensor, [N,1], float32

EPS=1e-6
BIG=1e6
ACTIVELY_RESET = True

class SnakeEnv():
    def __init__(self, init_length = 5, size=10, max_step=100, rew_penalty=-10, rew_nothing=-1, rew_food=10):
        self.init_length = init_length
        self.size = size
        self.max_step = max_step
        self.rew_penalty = rew_penalty
        self.rew_nothing = rew_nothing
        self.rew_food = rew_food
        
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
        
        # self._edge_mat = torch.zeros((self.size, self.size))
        # self._edge_mat[0,:] = 1
        # self._edge_mat[-1,:] = 1
        # self._edge_mat[:,0] = 1
        # self._edge_mat[:,-1] = 1
        
        # self.reset()
        
    def reset(self) -> RET:
        self.time_step = 0
        self.current_length = self.init_length
        
        obs = torch.zeros((3,self.size,self.size))
        obs = self._generate_init_head(obs, self.init_length)
        obs = self._generate_food(obs)
        
        self.state: OBS = obs
        reward = 0 
        done = 0
        
        return (obs.unsqueeze(0), torch.tensor([[reward]]), torch.tensor([[done]]))
    
    def step(self, action: ACT) -> RET:
        """
        time_step++
        
        if head hit wall or body:
            state = reset state
            reward=-1, done=0
            ( actively reset )
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
        time_out = self.time_step == self.max_step
        
        next_head = torch.conv2d(state[0].unsqueeze(0), 
                                    self._move_kernel[action], # type: ignore
                                    padding=1)[0]
        # DEBUG("next_head: \n",next_head)
        hit_wall = torch.abs(torch.sum(next_head))<EPS
        hit_body = torch.sum(next_head*state[2])>0
        if hit_wall or hit_body:
            # DEBUG(f"hit_wall: {hit_wall}, hit_body: {hit_body}")
            if ACTIVELY_RESET:
                reward = self.rew_penalty
                done = 0
                state = torch.zeros((3,self.size,self.size))
                state = self._generate_init_head(state, self.init_length)
                state = self._generate_food(state)
                self.current_length = self.init_length
            else:
                reward = self.rew_penalty
                done = 1
            self.state = state
            return (state.unsqueeze(0), torch.tensor([[reward]]), torch.tensor([[done or time_out]]))
        
        # Hit food
        hit_food = torch.sum(next_head*state[1])>0
        
        # Move body
        state[0] = next_head
        state[2] = np.clip(state[2] - (1- (hit_food).to(torch.float32)), 0, None) + state[0]*self.current_length # type: ignore
        
        if hit_food:
            state[1] = torch.zeros_like(state[1]) # remove food
            state = self._generate_food(state)
            self.current_length += 1
            reward = self.rew_food # DEBUG
            # reward = 1 
            done = 0
        else:
            reward = self.rew_nothing 
            done = 0
        
            
        self.state = state
        return (state.unsqueeze(0), torch.tensor([[reward]]), torch.tensor([[done or time_out]]))
        
    def get_hidden_state(self) -> \
        Tuple[OBS, ONEDIM, ONEDIM, ONEDIM]:
        return (self.state, torch.tensor([[self.time_step]]), 
                torch.tensor([[self.current_length]]), torch.tensor([[0]]))
        
    def _generate_food(self, state: OBS) -> OBS:
        # For vectorization, we need to generate random food in fixed number
        # of steps. We should avoid logical loop in the code.
        
        # x, y = np.random.randint(0, self.size, 2)
        # while (state[2,x,y] > 1):
        #     x, y = np.random.randint(0, self.size, 2)
        # state[1,x,y] = 1
        P = (state[2]==0).to(torch.float32).reshape(-1)
        # P = P/(torch.sum(P,dim=(-1,),keepdim=True))
        food = torch.multinomial(P, 1)
        x, y = food//self.size, food%self.size
        state[1,x,y] = 1
        
        return state
    
    def _generate_init_head(self, state: OBS, init_length: int) -> OBS:
        x, y = np.random.randint(0, self.size, 2)
        state[0,x,y] = 1
        state[2,x,y] = init_length
        return state
    
class VectorizedSnakeEnv(SnakeEnv):
    def __init__(self, num_batch, device="cuda", init_length = 5, size=10, max_step=100, rew_penalty=-10, rew_nothing=-1, rew_food=10,rew_difficulty=0.3,rew_pow_len=2):
        self.n = num_batch
        self.device = device
        self.init_length = init_length
        self.size = size
        self.max_step = max_step
        self.rew_penalty = rew_penalty
        self.rew_nothing = rew_nothing
        self.rew_food = rew_food
        self.rew_difficulty = rew_difficulty
        self.rew_pow_len = rew_pow_len
        
        # Current State
        self.state = torch.tensor([])
        self.time_step = torch.tensor([])
        self.death_count = torch.tensor([])
        self.current_length = torch.tensor([])
        
        self._move_kernel =torch.concat([
            torch.tensor([[0,0,0],[1,0,0],[0,0,0]],dtype=torch.float32).flip(dims=(-2,-1)).reshape(1,1,3,3),
            torch.tensor([[0,0,0],[0,0,1],[0,0,0]],dtype=torch.float32).flip(dims=(-2,-1)).reshape(1,1,3,3),
            torch.tensor([[0,1,0],[0,0,0],[0,0,0]],dtype=torch.float32).flip(dims=(-2,-1)).reshape(1,1,3,3),
            torch.tensor([[0,0,0],[0,0,0],[0,1,0]],dtype=torch.float32).flip(dims=(-2,-1)).reshape(1,1,3,3),
        ]).to(self.device)
        # [4,1,3,3]
        
        
        # self.reset()
        
    def get_hidden_state(self) -> \
        Tuple[OBS, ONEDIM, ONEDIM, ONEDIM]:
        return (self.state, self.time_step, self.current_length, self.death_count)
    
    def step(self, action: ACT) -> RET: 
        # action: [N,1] int64
        """
        time_step++
        
        Compute Next Head
        
        Compute Event
        
        ACTIVELY RESET
        
        MOVE SNAKE
        
        REMOVE FOOD
        
        if head hit wall or body:
            state = reset state
            reward=-1, done=0
            ( actively reset )
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
        time_out = (self.time_step == self.max_step).to(torch.float32)
        # [N,1] float32
        
        one_hot_action = F.one_hot(action.to(torch.int64), num_classes=4).to(torch.float32) # type: ignore
        # [N,1,4]
        next_head = torch.conv2d(state[:,:1], 
                                    self._move_kernel,
                                    padding=1)
        # [N,4,size,size]
        next_head = torch.einsum('ijkl,iaj->ikl',next_head,one_hot_action)
        # [N,size,size]
        
        # Compute Event
        hit_wall = torch.abs(torch.sum(next_head,dim=(-1,-2),keepdim=False)[:,None])<EPS
        # [N,1]
        hit_body = (torch.sum(next_head*state[:,2],dim=(-1,-2),keepdim=False)>0)[:,None]
        hit_food = (torch.sum(next_head*state[:,1],dim=(-1,-2),keepdim=False)>0)[:,None]
        die = torch.logical_or(hit_wall, hit_body)
        hit_food,die = hit_food.to(torch.float32), die.to(torch.float32)
        self.death_count += die
        self.current_length = (self.current_length + hit_food) * (1-die) + self.init_length * die
        
        # ACTIVELY RESET
        state = torch.zeros((self.n,3,self.size,self.size),device=self.device)*die[:,None,None] + state*(1-die[:,None,None])
        state = self._generate_init_head(state, self.init_length, mask=die) 
        
        # Move body
        state[:,0] = next_head * (1-die[:,None]) + state[:,0] * die[:,None]
        state[:,2] = (torch.clip(state[:,2] - (1- (hit_food[:,None])), 0, None) + state[:,0]*self.current_length[:,None]) * (1-die[:,None]) + state[:,2] * die[:,None] # type: ignore
        
        # Remove food
        state[:,1] = torch.zeros_like(state[:,1])*hit_food[:,None] + state[:,1]*(1-hit_food[:,None])
        state = self._generate_food(state, mask=torch.logical_or(die,hit_food).to(torch.float32))
        
        # assert torch.all( torch.sum(state[:,0],dim=(-1,-2))==1  ), "Head is not unique"
        
        done = time_out
        reward = (self.rew_nothing * (1-hit_food)  + (self.rew_food+(self.current_length**self.rew_pow_len)*self.rew_difficulty) * hit_food) * (1-die) + self.rew_penalty * die
            
        # DEBUG("time step end",state)
        self.state = state
        return (state, reward, done)
        
    def reset(self) -> RET:
        self.time_step = torch.zeros((self.n,1),dtype=torch.float32,device=self.device)
        self.death_count = torch.zeros((self.n,1),dtype=torch.float32,device = self.device)
        self.current_length = torch.ones((self.n,1),
                                         dtype=torch.float32,device=self.device) * self.init_length
        
        obs = torch.zeros((self.n,3,self.size,self.size),device=self.device)
        obs = self._generate_init_head(obs, self.init_length)
        obs = self._generate_food(obs)
        
        self.state = obs
        reward = torch.zeros((self.n,1),dtype=torch.float32,device=self.device)
        done = torch.zeros((self.n,1),dtype=torch.float32,device=self.device)
        
        return (obs, reward, done)
    
    def _generate_food(self, state: OBS, 
                       mask: Union[None,ONEDIM]=None) -> OBS:
        # state: [N,3,size,size]
        # mask : [N,1] float32
        # mask = 1 means this batch sample need to generate food, 0 for not.
        
        # For vectorization, we need to generate random food in fixed number
        # of steps. We should avoid logical loop in the code.
        
        if mask is None:
            mask = torch.ones((self.n,1),dtype=torch.float32,device=self.device)
        P = ((state[:,2]==0).to(torch.float32).reshape(self.n,-1))
        P = P + (1-mask+(P.sum(dim=(-1),keepdim=True)==0).to(torch.float32))*BIG  
        # P: [N, size*size,]
        # add BIG number, to avoid those case that mask=0 but state[2] is all nonzero
        # It will cause the multinomial function to fail.
        
        food = torch.multinomial(P, 1)
        # food: [N, 1]
        
        x, y = food//self.size, food%self.size
        # x,y : [N, 1]
        
        state[torch.arange(self.n),1,x[:,0],y[:,0]] += mask[:,0]
        # assert torch.all( torch.sum(state[:,1],dim=(-1,-2))==1  ), "Food is not unique"
        # state[1,x,y] = 1
        
        return state
    
    def _generate_init_head(self, state: OBS, 
                            init_length: Union[int,ONEDIM], 
                            mask: Union[None,ONEDIM]=None) -> OBS:
        # state: [N,3,size,size]
        # init_length: [N,1] or int
        # mask : [N,1] float32
        # Assert state is empty for those mask==1.
        if mask is None:
            mask = torch.ones((self.n,1),dtype=torch.float32,device=self.device)
        
        # x, y = np.random.randint(0, self.size, 2)
        Pos = torch.randint(0, self.size, (2,self.n),device=self.device)
        
        state[torch.arange(self.n),0,Pos[0],Pos[1]] += mask[:,0]
        state[:,2] += mask[:,None] * init_length * state[:,0]
        # DEBUG("generate_init_head",state)
        # assert torch.all( torch.sum(state[:,0],dim=(-1,-2))==1  ), "Head is not unique"
        # state[0,x,y] = 1
        # state[2,x,y] = init_length
        return state