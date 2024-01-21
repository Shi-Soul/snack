import random
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
from agent import DQNAgent


def _get_nn_xs(out_channel):
    policy_net = nn.Sequential(
            # Input: (batch_size, 3, 5, 5)
            nn.Conv2d(in_channels=out_channel, out_channels=6,
                        kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=6, out_channels=8,
                        kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveMaxPool2d((2,2)),
                    # (batch_size, 8, 2, 2)
            nn.Flatten(),
                    # (batch_size, 8*2*2)
            nn.Linear(32, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 4),
        )
    return policy_net

def _get_nn_small(out_channel):
    # Design for 5*5
    policy_net = nn.Sequential(
            # Input: (batch_size, 3, 5, 5)
            nn.Conv2d(in_channels=out_channel, out_channels=16,
                        kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((2,2)),
            # nn.MaxPool2d(kernel_size=2, stride=2),
                    # (batch_size, 16, 2, 2)
            nn.Flatten(),
                    # (batch_size, 16*2*2)
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )
    return policy_net

def _get_nn_normal(out_channel):
    # Design for 25*25
    policy_net = nn.Sequential(
            # Input: (batch_size, 3, 25, 25)
            nn.Conv2d(in_channels=out_channel, out_channels=16,
                        kernel_size=7, stride=1, padding=3),
            # (N,16,25,25)
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=16, out_channels=16,
                        kernel_size=7, stride=3, padding=3),
            nn.SiLU(),
            # 25-K+2P/S+1 = 25-7+2*3/3+1 = 9
            # (N,16,9,9)
            nn.AdaptiveMaxPool2d((3,3)),
            nn.Conv2d(in_channels=16, out_channels=32,
                        kernel_size=3, stride=2, padding=0),
            # (N,32,4,4) 9-3+2*0/2+1 = 4
            nn.SiLU(),
            nn.AdaptiveMaxPool2d((2,2)),
                    # (batch_size, 32, 2, 2)
            nn.Flatten(),
            nn.Dropout(0.1),
                    # (batch_size, 32*2*2)
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 4),
        )
    return policy_net

_get_nn_dict = {
    'xs': _get_nn_xs,
    'small': _get_nn_small,
    'normal': _get_nn_normal,
}

class DQNTrainer(BaseRunner):
    def __init__(self, env, config):
        self.log_dir = config['log_dir']
        self.device = config['device']
        self.use_tb = config['use_tb_logger']
        self.train_epoch = config['train_epoch']
        self.update_steps = config['update_steps']
        self.target_update_steps = config['dqn_target_update_steps']
        self.test_freq = config['test_freq']
        self.gamma = config["dqn_gamma"]
        self.eps = 1e-10
        self.epsilon = config['dqn_epsilon']
        self.lr = config['dqn_lr']
        self.num_episodes = config['dqn_num_episodes']
        self.batch_size = config['dqn_batch_size']
        self.buffer_size = config['dqn_buffer_size']
        self.use_target_q = False
        self.update_step=0
        # self._reward_baseline = 0.1
        
        self.env = env
        self.use_vec_env = config['use_vec_env']
        self.n_env = config['n_env']
        assert self.num_episodes % self.n_env == 0, "num_episodes should be divisible by n_env"
        
        q_net:nn.Module = _get_nn_dict[config['dqn_net']](config['obs_channel']).to(self.device)
        if self.use_target_q:
            self.target_q_net = _get_nn_dict[config['dqn_net']](config['obs_channel']).to(self.device)
            self.target_q_net.load_state_dict(q_net.state_dict())
            self.target_q_net.requires_grad_(False)
        else:
            self.target_q_net = q_net
        self.agent = DQNAgent(q_net, self.device, epsilon=self.epsilon)
        self.agent.train(True)
        
        self.optimizer = torch.optim.Adam(q_net.parameters(), lr=self.lr)
        self.critic = nn.MSELoss()
        
        self.buffer = deque(maxlen=self.buffer_size)
        
        if self.use_tb:
            self.tb_logger = TBLogger()

    def run(self):
        self.update_step=0
        best_score = -np.inf
        loss_list = []
        score_mean = -np.inf
        score_std = np.inf
        try:
            for epoch in range(0,self.train_epoch):
                loss, mean_q = self.train_step() 
                loss_list.append(loss)
                
                if self.use_tb:
                    self.tb_logger.log_scalar(loss, 'loss', epoch)
                    self.tb_logger.log_scalar(mean_q, 'q', epoch)
                
                if epoch % self.test_freq == 0 or epoch == self.train_epoch - 1:
                    # score_mean, score_std = self.test()
                    result = self.test(self.n_env)
                    score_mean = result['score_mean']
                    score_std = result['score_std']
                    if score_mean > best_score:
                        best_score = score_mean
                        self.save_model(epoch, score_mean, score_std)
                    if epoch % 200 == 0:
                        self.save_model(epoch, score_mean, score_std)
                    
                    if self.use_tb:
                        # self.tb_logger.log_scalar(score_mean, 'score_mean', epoch)
                        # self.tb_logger.log_scalar(score_std, 'score_std', epoch)
                        self.tb_logger.log_scalars(result, 'eval', epoch)
                        self.tb_logger.flush()
                    INFO(f'Epoch: {epoch}, score: {score_mean:.3f} +- {score_std:.3f}, snake_length: {result["snake_length"]:.3f}, die_count: {result["die_count"]:.3f}')
        except KeyboardInterrupt:
            INFO("Training interrupted.")
        self.save_model(-1,score_mean,score_std)
        result = {'loss': loss_list, 'score_mean': score_mean, 'score_std': score_std}
        # INFO(result,pp=True)
        INFO("Training finished.")
        return result

    def _sample_episode_vec(self):
        # [T+1, N, C, H, W], [T, N, 1], [T+1, N, 1], 
        # [N, 1], [N, 1], [N, 1]
        eps_state = []
        eps_action = []
        eps_reward = []
        state, reward, done = self.env.reset()
        while not torch.all(done):
            action = self.agent.policy(state)
            eps_state.append(state)
            eps_action.append(action)
            eps_reward.append(reward)
            state, reward, done = self.env.step(action)
        eps_state.append(state)
        eps_reward.append(reward)
        eps_state, eps_action, eps_reward, = torch.stack(eps_state), torch.stack(eps_action), torch.stack(eps_reward)
        _,time_step,snake_length,death_count = self.env.get_hidden_state()
        return eps_state, eps_action, eps_reward, time_step, snake_length, death_count
        ...
        
    def _sample_episode_single(self):
        eps_state = []
        eps_action = []
        eps_reward = []
        state, reward, done = self.env.reset()
        while not done:
            action = self.agent.policy(state)
            eps_state.append(state)
            eps_action.append(action)
            eps_reward.append(reward)
            state, reward, done = self.env.step(action)
        eps_state.append(state)
        eps_reward.append(reward)
        _,time_step,snake_length = self.env.get_hidden_state()
        # _,time_step,snake_length,death_count = self.env.get_hidden_state()
        return eps_state, eps_action, eps_reward, time_step, snake_length, None
        ...
        
    def sample_episode(self):
        if self.use_vec_env:
            return self._sample_episode_vec()
        else:
            return self._sample_episode_single()
    
    def train_step(self):
        # Sample episodes from buffer
        # Calculate loss
        # Update policy
        
        self.agent.train(True)
        for episode in range(self.num_episodes//self.n_env):
            eps_state, eps_action, eps_reward, _,_,_ = self.sample_episode()
            # Reorganize it into a list of episode of (s,a,r,s')
            eps_old_state = eps_state[:-1]
            # eps_action
            eps_reward = eps_reward[1:] 
            eps_next_state = eps_state[1:]
            T, N, C, H, W = eps_old_state.shape
            
            
            eps_old_state_list = torch.split(eps_old_state.reshape(T*N,C,H,W), 1, dim=0)
            eps_action_list = torch.split(eps_action.reshape(T*N,1), 1, dim=0)
            eps_reward_list = torch.split(eps_reward.reshape(T*N,1), 1, dim=0)
            eps_next_state_list = torch.split(eps_next_state.reshape(T*N,C,H,W), 1, dim=0)
            
            combined_eps_iter = zip(
                eps_old_state_list, eps_action_list, 
                eps_reward_list, eps_next_state_list)
            self.buffer.extend(combined_eps_iter)
            
            
            # self.buffer.append((eps_state, eps_action, eps_reward))
        
        total_loss = 0
        total_q_value = 0
        for i in range(self.update_steps):
            self.update_step += 1
            sample = random.sample(self.buffer, self.batch_size)
            eps_old_state, eps_action, eps_reward, eps_next_state = zip(*sample)
            eps_old_state = torch.cat(eps_old_state)
            eps_action = torch.cat(eps_action)
            eps_reward = torch.cat(eps_reward)
            eps_next_state = torch.cat(eps_next_state)
            
            q_value = self.agent.q_net(eps_old_state)
            q_value = q_value[torch.arange(self.batch_size),eps_action.reshape(-1).to(torch.int64)]
            # q_value_next = self.agent.q_net(eps_next_state)
            q_value_next = self.target_q_net(eps_next_state)
            q_value_next = torch.max(q_value_next, dim=1)[0]
            
            loss = self.critic(q_value, eps_reward.squeeze() + self.gamma * q_value_next)
            total_loss += loss.item()
            total_q_value += q_value.mean().item()
            
            if not torch.isnan(loss):
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                INFO("Loss is nan") 
            
            if self.update_step % self.target_update_steps == 0 and self.use_target_q:
                self.target_q_net.load_state_dict(self.agent.q_net.state_dict())
                self.target_q_net.requires_grad_(False)
                
        return total_loss / self.update_steps, total_q_value / self.update_steps

    def test(self,N=128):
        self.agent.train(False)
        score = []
        # time_step = torch.tensor([0.],device=self.device)
        snake_length = torch.tensor([0.],device=self.device)
        die_count = torch.tensor([0.],device=self.device)
        
        for episode in range(N//self.n_env):
            sample = self.sample_episode()
            eps_score = sample[2].sum(dim=(-1,-3))
            
            score.append(eps_score)
            # time_step += sample[3].sum()
            snake_length += sample[4].sum()
            die_count += sample[5].sum()
            
        print("DEBUG: ",[i.squeeze() for i in sample],"\neps_score: ", eps_score)
            
        score = torch.concat(score)
        s_mean, std = score.mean(), score.std()
        result = {
            'score_mean': s_mean.item(),
            'score_std': std.item(),
            # 'time_step': time_step.item()/N,
            'snake_length': snake_length.item()/N,
            'die_count': die_count.item()/N,
        }
        return result

    def save_model(self, epoch, score_mean, score_std):
        model_path = os.path.join(self.log_dir,'model',
                                  f'dqn_{epoch}_{score_mean:.3f}_{score_std:.3f}.pth')
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        self.agent.save_model(model_path)
        INFO(f'Save model to {model_path}')



