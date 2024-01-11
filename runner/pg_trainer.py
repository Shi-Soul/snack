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
        self.log_dir = config['log_dir']
        self.device = config['device']
        self.use_tb = config['use_tb_logger']
        self.train_epoch = config['train_epoch']
        self.update_steps = config['update_steps']
        self.test_freq = config['test_freq']
        self.gamma = config["pg_gamma"]
        # self.eps = 1e-6
        self.epsilon = config['pg_epsilon']
        self.lr = config['pg_lr']
        self.num_episodes = config['pg_num_episodes']
        self.buffer_size = config['pg_buffer_size']
        
        
        self.env = env
        
        policy_net = _get_nn_small(config['obs_channel']).to(self.device)
        
        self.agent = PGAgent(policy_net, self.device, epsilon=self.epsilon)
        self.agent.train(True)
        
        self.optimizer = torch.optim.Adam(policy_net.parameters(), lr=self.lr)
        
        self.buffer = deque(maxlen=self.buffer_size)
        
        if self.use_tb:
            self.tb_logger = TBLogger()

    def run(self):
        best_score = -np.inf
        loss_list = []
        score_mean_list = []
        score_std_list = []
        time_step_list = []
        snake_length_list = []
        try:
            for epoch in range(self.train_epoch):
                loss = self.train_step()
                loss_list.append(loss)
                
                if self.use_tb:
                    self.tb_logger.log_scalar(loss, 'loss', epoch)
                
                if epoch % self.test_freq == 0 or epoch == self.train_epoch - 1:
                    # score_mean, score_std = self.test()
                    score_mean, score_std, time_step, snake_length = self.test()
                    score_mean_list.append(score_mean)
                    score_std_list.append(score_std)
                    time_step_list.append(time_step)
                    snake_length_list.append(snake_length)
                    
                    if score_mean > best_score:
                        best_score = score_mean
                        self.save_model(epoch, score_mean, score_std)
                    
                    if self.use_tb:
                        # self.tb_logger.log_scalar(score_mean, 'score_mean', epoch)
                        # self.tb_logger.log_scalar(score_std, 'score_std', epoch)
                        self.tb_logger.log_scalars({
                            "score_mean": score_mean,
                            "score_std": score_std,
                            "time_step": time_step,
                            "snake_length": snake_length
                            }, 'eval', epoch)
                    INFO(f'Epoch: {epoch}, score: {score_mean:.3f} +- {score_std:.3f}')
        except KeyboardInterrupt:
            INFO("Training interrupted.")
        self.save_model(-1,score_mean_list[-1],score_std_list[-1])
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
        _,time_step,snake_length = self.env.get_hidden_state()
        return eps_state, eps_action, eps_action_prob, eps_reward, time_step, snake_length
        # return eps_state, eps_action, eps_reward

    def get_nstep_reward(self, eps_reward):
        reward_to_go = torch.zeros_like(eps_reward, dtype=torch.float32, device=self.device)
        N = len(eps_reward)
        j = N - 1
        while j >= 0:
            reward_to_go[j-1] = eps_reward[j] + self.gamma * reward_to_go[j]
            j -= 1
        return reward_to_go
    
    def _update(self):
        # Sample episodes from buffer
        # Calculate loss
        # Update policy
        
        loss = torch.tensor([0], dtype=torch.float32, device=self.device)
        for _ in range(self.update_steps):
            indices = np.random.randint(0, len(self.buffer))
            eps_state, eps_action, _, eps_reward = self.buffer[indices]
            
            eps_reward = torch.tensor(eps_reward, dtype=torch.float32, device=self.device)
            reward_to_go = self.get_nstep_reward(eps_reward)
            
            action_prob = self.agent.policy_prob(torch.concat(eps_state))
            N = len(eps_reward)
            log_prob = torch.log(action_prob[torch.arange(N),torch.tensor(eps_action)])
                
            loss += -torch.sum(log_prob * reward_to_go)
            
        loss = loss / self.update_steps
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # loss_sum += loss.item()
        assert not np.isnan(loss.item()), 'loss is nan'
        return loss.item()
        
    def train_step(self):
        self.agent.train(True)
        for episode in range(self.num_episodes):
            eps_state, eps_action, eps_action_prob, eps_reward, _,_ = self.sample_episode()
                
            self.buffer.append((eps_state, eps_action, eps_action_prob, eps_reward))
        loss = self._update()
        return loss


    def test(self,N=100):
        self.agent.train(False)
        score = []
        time_step = []
        snake_length = []
        for episode in range(N):
            sample = self.sample_episode()
            score.append(sum(sample[3]))
            time_step.append(sample[4])
            snake_length.append(sample[5])
        score = np.array(score)
        s_mean, std = score.mean(), score.std()
        return s_mean, std, sum(time_step)/N, sum(snake_length)/N

    def save_model(self, epoch, score_mean, score_std):
        model_path = os.path.join(self.log_dir,'model',
                                  f'pg_{epoch}_{score_mean:.3f}_{score_std:.3f}.pth')
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        torch.save(self.agent.model.state_dict(), model_path)
        INFO(f'Save model to {model_path}')



