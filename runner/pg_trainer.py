import torch 
from torch import nn
import torch.nn.functional as F
import numpy as np
import logging
import os
import time
import sys
import yaml
from datetime import datetime
from collections import deque

from util import INFO, DEBUG, TBLogger
from runner import BaseRunner
from agent import PGAgent

def _get_nn_small(out_channel):
    policy_net = nn.Sequential(
            # Input: (batch_size, 3, 5, 5)
            nn.Conv2d(in_channels=out_channel, out_channels=16,
                        kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
                    # (batch_size, 16, 2, 2)
            nn.Flatten(),
                    # (batch_size, 16*2*2)
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )
    return policy_net

def _get_nn_normal(out_channel):
    policy_net = nn.Sequential(
            # Input: (batch_size, 3, 5, 5)
            nn.Conv2d(in_channels=out_channel, out_channels=16,
                        kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
                    # (batch_size, 32, 2, 2)
            nn.Flatten(),
                    # (batch_size, 32*2*2)
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
    return policy_net


class PGTrainer(BaseRunner):
    def __init__(self, env, config):
        self.device = config['device']
        self.use_tb = config['use_tb_logger']
        self.train_epoch = config['train_epoch']
        self.update_steps = config['update_steps']
        self.test_freq = config['test_freq']
        self.gamma = 0.99
        self.eps = 1e-6
        self.epsilon = 0.1
        self.lr = 1e-3
        self.num_episodes = 200
        self.buffer_size = 300
        
        self.env = env
        
        policy_net = _get_nn_small(config['obs_channel'])
        
        self.agent = PGAgent(policy_net, self.device, epsilon=self.epsilon)
        self.agent.train(True)
        
        self.optimizer = torch.optim.Adam(policy_net.parameters(), lr=self.lr)
        
        self.buffer = deque(maxlen=self.buffer_size)
        
        if self.use_tb:
            self.tb_logger = TBLogger()

    def run(self):
        loss_list = []
        score_mean_list = []
        score_std_list = []
        try:
            for epoch in range(self.train_epoch):
                loss = self.train_step()
                loss_list.append(loss)
                
                if self.use_tb:
                    self.tb_logger.log_scalar(loss, 'loss', epoch)
                
                if epoch % self.test_freq == 0 or epoch == self.train_epoch - 1:
                    score_mean, score_std = self.test()
                    score_mean_list.append(score_mean)
                    score_std_list.append(score_std)
                    if self.use_tb:
                        self.tb_logger.log_scalar(score_mean, 'score_mean', epoch)
                        self.tb_logger.log_scalar(score_std, 'score_std', epoch)
                    INFO(f'Epoch: {epoch}, score: {score_mean:.3f} +- {score_std:.3f}')
        except KeyboardInterrupt:
            INFO("Training interrupted.")
        result = {'loss': loss_list, 'score_mean': score_mean_list, 'score_std': score_std_list}
        # INFO(result,pp=True)
        INFO("Training finished.")
        return result

    def sample_episode(self):
        eps_state = []
        eps_action = []
        eps_action_prob = []
        eps_reward = []
        
        state, reward, done = self.env.reset()
        # state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        while not done:
            # action = self.agent.policy(state)
            state = state.unsqueeze(0)
            # action, action_prob = self.agent.policy_prob(state)
            action = self.agent.policy(state)
            next_state, reward, done = self.env.step(action)
            # next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # eps.append((state, action, reward))
            eps_state.append(state)
            eps_action.append(action)
            # eps_action_prob.append(action_prob)
            eps_reward.append(reward)
            
            state = next_state
        return eps_state, eps_action, eps_action_prob, eps_reward
        # return eps_state, eps_action, eps_reward

    def _update(self):
        # Sample episodes from buffer
        # Calculate loss
        # Update policy
        
        loss = torch.tensor([0], dtype=torch.float32, device=self.device)
        for _ in range(self.update_steps):
            indices = np.random.randint(0, len(self.buffer))
            # 优化: 用矩阵运算代替循环
            eps_state, eps_action, _, eps_reward = self.buffer[indices]
            
            # Compute reward to go
            eps_reward = torch.tensor(eps_reward, dtype=torch.float32, device=self.device)
            reward_to_go = torch.zeros_like(eps_reward, dtype=torch.float32, device=self.device)
            N = len(eps_reward)
            j = N - 1
            while j >= 0:
                reward_to_go[j-1] = eps_reward[j] + self.gamma * reward_to_go[j]
                j -= 1
            # DEBUG(torch.concat(eps_state).shape)
            action_prob = self.agent.policy_prob(torch.concat(eps_state))
            log_prob = torch.log(action_prob[torch.arange(N),torch.tensor(eps_action)])
                
            loss = -torch.sum(log_prob * reward_to_go)
        loss = loss / self.update_steps
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # loss_sum += loss.item()
        assert not np.isnan(loss.item()), 'loss is nan'
        return loss.item()
        
    def train_step(self):
        for episode in range(self.num_episodes):
            eps_state, eps_action, eps_action_prob, eps_reward = self.sample_episode()
                
            self.buffer.append((eps_state, eps_action, eps_action_prob, eps_reward))
        loss = self._update()
        return loss

    def test(self):
        score = []
        for episode in range(100):
            sample = self.sample_episode()
            score.append(sum(sample[3]))
        score = np.array(score)
        mean, std = score.mean(), score.std()
        return mean, std




