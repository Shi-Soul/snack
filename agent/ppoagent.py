from agent import BaseAgent
import torch 
from torch import nn
import torch.nn.functional as F
import numpy as np


class PPOAgent(BaseAgent):
    def __init__(self, model: nn.Module, device, epsilon=0.1):
        self.model = model
        self.model.to(device)
        self.device = device
        self.epsilon = epsilon
        
        self.is_train = False
        self.use_argmax = False
        
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
        action_prob = self.model(state)
        action_prob = torch.softmax(action_prob,dim=-1,)
        action_dist = torch.distributions.Categorical(action_prob)
        action = action_dist.sample()
            # action = np.random.choice(4, p=action_prob.cpu().detach().numpy().squeeze())
        # assert action.shape == (N,1)
        return action.reshape(-1,1)
    
    def policy_prob(self, state:torch.Tensor)->torch.Tensor:
        # state: [N, C, H, W]
        # q_value: [N, 4]
        # action_prob: [N, 4]
        state = state.to(self.device)
        action_prob:torch.Tensor = self.model(state)
        action_prob = torch.softmax(action_prob,dim=-1)
        # action = action_prob.argmax(-1).item()
        if self.is_train:
            action_prob = action_prob*(1-self.epsilon) + self.epsilon/4
        #     # epsilon greedy
    
        return action_prob
    
    def train(self, is_train):
        self.is_train = is_train
        self.model.train(is_train)

    def update_epsilon(self, epsilon):
        self.epsilon = epsilon

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)