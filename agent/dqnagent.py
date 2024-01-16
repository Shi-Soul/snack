from agent import BaseAgent
import torch 
from torch import nn
import torch.nn.functional as F
import numpy as np


class DQNAgent(BaseAgent):
    def __init__(self, model: nn.Module, device, epsilon=0.1):
        self.q_net = model
        # Q net
        self.q_net.to(device)
        self.device = device
        self.epsilon = epsilon
        
        self.is_train = False
        
    def policy(self, state:torch.Tensor)->torch.Tensor:
        # state: [N, C, H, W]
        # action: [N, 1]
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        N = state.shape[0]
        
        if self.is_train:
            # epsilon greedy
            if np.random.rand() < self.epsilon:
                action = torch.randint(0, 4, size=(N,1), device=self.device)
                return action 
            
        state = state.to(self.device)
        q_value:torch.Tensor = self.q_net(state)
        action = q_value.argmax(-1,keepdim=True).to(torch.float32)
        
        return action
    
    def policy_prob(self, state:torch.Tensor)->torch.Tensor:
        # state: [N, C, H, W]
        # q_value: [N, 4]
        # action_prob: [N, 4]
        
        # N = state.shape[0]
        state = state.to(self.device)
        q_value:torch.Tensor = self.q_net(state)
        # action_prob = torch.softmax(action_prob,dim=-1)
        action_prob = F.one_hot(q_value.argmax(-1),4).to(torch.float32)
        # if self.is_train:
        #     # epsilon greedy
        #     if np.random.rand() < self.epsilon:
        #         action = np.random.randint(0, 4, size=N)
        #         # action = torch.randint(0, 4, size=(N,), device=self.device)
    
        return action_prob
    
    def train(self, is_train):
        self.is_train = is_train
        self.q_net.train(is_train)

    def load_model(self, path):
        self.q_net.load_state_dict(torch.load(path))

    def save_model(self, path):
        torch.save(self.q_net.state_dict(), path)