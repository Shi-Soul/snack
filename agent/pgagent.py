from agent import BaseAgent
import torch 
from torch import nn
import torch.nn.functional as F
import numpy as np


class PGAgent(BaseAgent):
    def __init__(self, model: nn.Module, device, epsilon=0.1):
        self.model = model
        self.device = device
        self.epsilon = epsilon
        
        self.is_train = False
        
    def policy(self, state):
                
        if self.is_train:
            # epsilon greedy
            if np.random.rand() < self.epsilon:
                action = np.random.randint(0, 4)
                return action 
            
        state = state.to(self.device)
        action_prob = self.model(state)
        action = action_prob.argmax().item()
        
        return action
    
    def policy_prob(self, state:torch.Tensor)->torch.Tensor:
        # state: [N, C, H, W]
        # action_prob: [N, 4]
        # action: [N]
        # N = state.shape[0]
        state = state.to(self.device)
        action_prob:torch.Tensor = self.model(state)
        
        # action = action_prob.argmax(-1).item()
        # if self.is_train:
        #     # epsilon greedy
        #     if np.random.rand() < self.epsilon:
        #         action = np.random.randint(0, 4, size=N)
        #         # action = torch.randint(0, 4, size=(N,), device=self.device)
    
        return action_prob
    
    def train(self, is_train):
        self.is_train = is_train
        self.model.train(is_train)



