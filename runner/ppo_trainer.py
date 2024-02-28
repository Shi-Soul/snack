import random
import torch 
from torch import nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np
import logging
import os
import time
import sys
import yaml
from datetime import datetime
from collections import deque
from typing import Tuple, List, Dict, Union
from env import SnakeEnv
from env.core_torch import VectorizedSnakeEnv

from util import INFO, DEBUG, TBLogger, OBS, ONEDIM, ACT, REW
from runner import BaseRunner
from agent import PPOAgent
from util.type import POL


def _get_nn_xs(out_channel,out_dim=4):
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
            nn.Linear(8, out_dim),
        )
    return policy_net

def _get_nn_small(out_channel,out_dim=4):
    # Design for 5*5, Can also be used for 15*15
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
            nn.Linear(32, out_dim),
        )
    INFO(policy_net)
    INFO("Total parameters: ",sum(p.numel() for p in policy_net.parameters() if p.requires_grad))
    return policy_net

def _get_nn_mid(out_channel,out_dim=4):
    # Design for 15*15
    policy_net = nn.Sequential(
                # Input: (batch_size, 3, 15, 15)
            nn.Conv2d(in_channels=out_channel, out_channels=4,
                        kernel_size=7, stride=2, padding=3),
                # 15-K+2P/S+1 = (15-7+2*3)/2+1 = 8
                # (N,4,8,8)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
                # (N, 4, 4, 4)
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=4, out_channels=32,
                        kernel_size=3, stride=1, padding=0),
            nn.SiLU(),
                # 4-3+2*0/1+1 = 2
                # (N,32,2,2)
            nn.AdaptiveMaxPool2d((2,2)),
                # (N, 32, 2, 2)
            nn.Flatten(),
            nn.Dropout(0.1),
                # (N, 32*2*2)
            nn.Linear(128, 32),
            nn.SiLU(),
            nn.Linear(32, out_dim),
        )
    INFO(policy_net)
    INFO("Total parameters: ",sum(p.numel() for p in policy_net.parameters() if p.requires_grad))
    return policy_net

def _get_nn_normal(out_channel,out_dim=4):
    # Design for 25*25
    policy_net = nn.Sequential(
                # Input: (batch_size, 3, 25, 25)
            nn.Conv2d(in_channels=out_channel, out_channels=8,
                        kernel_size=7, stride=3, padding=3),
                # 25-K+2P/S+1 = 25-7+2*3/3+1 = 9
                # (N,4,9,9)
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=8, out_channels=32,
                        kernel_size=5, stride=1, padding=0),
            nn.SiLU(),
                # 9-5+2*2/1+1 = 5
                # (N,16,5,5)
            nn.AdaptiveMaxPool2d((2,2)),
                # (N, 32, 2, 2)
            nn.Flatten(),
            nn.Dropout(0.1),
                # (N, 32*2*2)
            nn.Linear(128, 32),
            nn.SiLU(),
            nn.Linear(32, out_dim),
        )
    INFO(policy_net)
    INFO("Total parameters: ",sum(p.numel() for p in policy_net.parameters() if p.requires_grad))
    return policy_net

_get_nn_dict = {
    'xs': _get_nn_xs,
    'small': _get_nn_small,
    'mid': _get_nn_mid,
    'normal': _get_nn_normal,
}

class PPOTrainer(BaseRunner):
    def __init__(self, env: SnakeEnv , config):
        ppo_config = config['ppo']
        self.log_dir = config['log_dir']
        self.device = config['device']
        self.use_tb = config['use_tb_logger']
        
        self.train_epoch = config['train_epoch']
        self.update_steps = config['update_steps']
        self.test_freq = config['test_freq']
        
        self.target_update_steps = ppo_config['target_update_steps']
        self.gamma = ppo_config["gamma"]
        self.eps = 1e-10
        self.explore_epsilon = ppo_config['explore_epsilon']
        self.clip_epsilon = ppo_config['clip_epsilon']
        # self.lr = ppo_config['lr']
        self.num_episodes = ppo_config['num_episodes']
        self.batch_size = ppo_config['batch_size']
        self.buffer_size = ppo_config['buffer_size']
        self.use_target_q = ppo_config['use_target_q']
        self.use_lr_scheduler = ppo_config['use_lr_scheduler']
        self.update_step=0
        # self._reward_baseline = 0.1
        
        self.env = env
        self.use_vec_env = config['use_vec_env']
        self.n_env = config['n_env']
        assert self.num_episodes % self.n_env == 0, "num_episodes should be divisible by n_env"
        
        self.v_net:nn.Module = _get_nn_dict[ppo_config['net']](config['obs_channel'],1).to(self.device)
        if self.use_target_q:
            self.target_v_net = _get_nn_dict[ppo_config['net']](config['obs_channel'],1).to(self.device)
            self.target_v_net.load_state_dict(self.v_net.state_dict())
            self.target_v_net.requires_grad_(False)
        else:
            self.target_v_net = self.v_net
        self.policy_net = _get_nn_dict[ppo_config['net']](config['obs_channel']).to(self.device)
        
        self.agent = PPOAgent(self.policy_net, self.device, epsilon=self.explore_epsilon)
        self.agent.train(True)
        
        self.v_optimizer = torch.optim.Adam(self.v_net.parameters(), 
                                            lr=ppo_config["critic_lr"])

        
        self.p_optimizer = torch.optim.Adam(self.policy_net.parameters(), 
                                            lr=ppo_config["actor_lr"])
        
        if self.use_lr_scheduler:
            # self.v_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                # self.v_optimizer, gamma=0.98)
            self.v_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.v_optimizer, T_max=self.train_epoch, eta_min=1e-10)
            self.p_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.p_optimizer, T_max=self.train_epoch, eta_min=1e-9)
        
        self.v_critic = nn.MSELoss()
        
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
                ret = self.train_step() 
                loss = ret["loss"]
                loss_list.append(loss)
                
                if self.use_tb:
                    # self.tb_logger.log_scalar(loss, 'loss', epoch)
                    # self.tb_logger.log_scalar(ret["critic"], 'critic_loss', epoch)
                    # self.tb_logger.log_scalar(ret[1], 'actor_loss', epoch)
                    # self.tb_logger.log_scalar(ret[2], 'advantage', epoch)
                    self.tb_logger.log_scalars(ret, 'train', epoch)
                
                if epoch % self.test_freq == 0 or epoch == self.train_epoch - 1:
                    # score_mean, score_std = self.test()
                    result = self.test(self.n_env)
                    score_mean = result['score_mean']
                    score_std = result['score_std']
                    if score_mean > best_score:
                        best_score = score_mean
                        self.save_model(epoch, score_mean, score_std)
                    if epoch % 500 == 0:
                        self.save_model(epoch, score_mean, score_std)
                    
                    if self.use_tb:
                        self.tb_logger.log_scalars(result, 'eval', epoch)
                        self.tb_logger.flush()
                    INFO(f'Epoch: {epoch}, score: {score_mean:.3f} +- {score_std:.3f}, eat_count: {result["eat_count"]:.3f}, die_count: {result["die_count"]:.3f}')
        except KeyboardInterrupt:
            INFO("Training interrupted.")
        self.save_model(-1,score_mean,score_std)
        result = {'loss': loss_list, 'score_mean': score_mean, 'score_std': score_std}
        
        INFO("Training finished.")
        return result

    def _sample_episode_vec(self):
        # [T+1, N, C, H, W], [T, N, 1], [T+1, N, 1], 
        # [N, 1], [N, 1], [N, 1]
        eps_state = []
        eps_action = []
        eps_log_action_prob = []
        eps_reward = []
        state, reward, done = self.env.reset()
        while not torch.all(done):
            action_prob = self.agent.policy_prob(state)
            action = D.Categorical(action_prob).sample().reshape(-1,1)
            # action = self.agent.policy(state)
            eps_state.append(state)
            eps_action.append(action)
            eps_log_action_prob.append(torch.log(action_prob).detach().gather(1,action))
            eps_reward.append(reward)
            state, reward, done = self.env.step(action)
        eps_state.append(state)
        eps_reward.append(reward)
        # eps_state, eps_action, , eps_reward, = torch.stack(eps_state), torch.stack(eps_action), torch.stack(eps_reward)
        eps_state, eps_action, eps_log_action_prob, eps_reward, = torch.stack(eps_state), torch.stack(eps_action), torch.stack(eps_log_action_prob), torch.stack(eps_reward)
        _,time_step,eat_count,death_count = self.env.get_hidden_state()
        return eps_state, eps_action, eps_log_action_prob, eps_reward, \
                time_step, eat_count, death_count
        ...
        
    def _sample_episode_single(self):
        raise NotImplementedError
        # eps_state = []
        # eps_action = []
        # eps_reward = []
        # state, reward, done = self.env.reset()
        # while not done:
        #     action = self.agent.policy(state)
        #     eps_state.append(state)
        #     eps_action.append(action)
        #     eps_reward.append(reward)
        #     state, reward, done = self.env.step(action)
        # eps_state.append(state)
        # eps_reward.append(reward)
        # _,time_step,snake_length,_ = self.env.get_hidden_state()
        # eps_state, eps_action, eps_reward, = torch.stack(eps_state), torch.stack(eps_action), torch.stack(eps_reward)
        # # _,time_step,snake_length,death_count = self.env.get_hidden_state()
        # return eps_state, eps_action, eps_reward, time_step, snake_length, torch.tensor([[0]])
        # ...
        
    def sample_episode(self)->Tuple[torch.Tensor, ACT, POL, REW, ONEDIM, ONEDIM, ONEDIM] : 
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
            eps_state, eps_action, eps_log_action_prob, eps_reward, _,_,_ = self.sample_episode()
            # Reorganize it into a list of episode of (s,a,r,s')
            eps_old_state = eps_state[:-1]
            # eps_action
            eps_reward = eps_reward[1:] 
            eps_next_state = eps_state[1:]
            T, N, C, H, W = eps_old_state.shape
            
            
            eps_old_state_list = torch.split(eps_old_state.reshape(T*N,C,H,W), 1, dim=0)
            eps_action_list = torch.split(eps_action.reshape(T*N,1), 1, dim=0)
            eps_log_action_prob_list = torch.split(eps_log_action_prob.reshape(T*N,1), 1, dim=0)
            eps_reward_list = torch.split(eps_reward.reshape(T*N,1), 1, dim=0)
            eps_next_state_list = torch.split(eps_next_state.reshape(T*N,C,H,W), 1, dim=0)
            
            combined_eps_iter = zip(
                eps_old_state_list, eps_action_list, eps_log_action_prob_list,
                eps_reward_list, eps_next_state_list)
            self.buffer.extend(combined_eps_iter)
            
            
            # self.buffer.append((eps_state, eps_action, eps_reward))
        
        total_v_loss = 0
        total_a_loss = 0
        total_a_value = 0
        # total_ratio = 0
        # max_ratio = 0
        # min_ratio = 10
        ratio_list=[]
        for i in range(self.update_steps):
            self.update_step += 1
            sample = random.sample(self.buffer, self.batch_size)
            eps_old_state, eps_action, log_action_prob, eps_reward, eps_next_state = zip(*sample)
            eps_old_state = torch.cat(eps_old_state)
            eps_action = torch.cat(eps_action)
            old_log_prob = torch.cat(log_action_prob)
            eps_reward = torch.cat(eps_reward)
            eps_next_state = torch.cat(eps_next_state)
            
            v = self.v_net(eps_old_state)
            v_next = self.target_v_net(eps_next_state)
            td_target = eps_reward + self.gamma * v_next
            td_diff = (td_target - v).detach()
            
            new_action_prob = self.agent.policy_prob(eps_old_state).gather(1, eps_action)
            ratio = torch.exp(torch.log(new_action_prob) - old_log_prob)
            clamp_ratio = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon)
            
            actor_loss:torch.Tensor = -torch.min(ratio*td_diff, clamp_ratio*td_diff).mean()

            critic_loss:torch.Tensor = self.v_critic(v, td_target.detach())
            
            self.p_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.p_optimizer.step()
            self.v_optimizer.zero_grad()
            critic_loss.backward()
            self.v_optimizer.step()
            
            total_v_loss += critic_loss.item()
            total_a_loss += actor_loss.item()
            total_a_value += td_diff.mean().item()
            # total_ratio += ratio.mean().item()
            # max_ratio = max(max_ratio, ratio.max().item())
            # min_ratio = min(min_ratio, ratio.min().item())
            ratio_list.append(ratio.mean().item())
            
            if self.update_step % self.target_update_steps == 0 and self.use_target_q:
                self.target_v_net.load_state_dict(self.v_net.state_dict())
                self.target_v_net.requires_grad_(False)
             
        if self.use_lr_scheduler:
            self.v_lr_scheduler.step()
            self.p_lr_scheduler.step()
        self.agent.update_epsilon(self.explore_epsilon*np.cos((self.update_step+1)/(self.train_epoch*self.update_steps)*np.pi/2))   
        # return total_loss / self.update_steps, total_q_value / self.update_steps
        result = {
            "loss":(total_v_loss + total_a_loss) / self.update_steps,
            "critic_loss":total_v_loss / self.update_steps, 
            "actor_loss":total_a_loss / self.update_steps, 
            "advantage":total_a_value / self.update_steps,
            "mean_ratio":sum(ratio_list)/len(ratio_list),
            "std_ratio":np.std(ratio_list),
            "max_ratio":max(ratio_list),
            "min_ratio":min(ratio_list),
            }
        return result

    def test(self,N=128):
        self.agent.train(False)
        score = []
        # time_step = torch.tensor([0.],device=self.device)
        eat_count = torch.tensor([0.],device=self.device)
        die_count = torch.tensor([0.],device=self.device)
        sample = [torch.tensor([0],device=self.device)]  # Assign a default value to "sample"
        eps_score = torch.tensor([0.],device=self.device)
        for episode in range(N//self.n_env):
            sample = self.sample_episode()
            eps_score = sample[3].sum(dim=(-1,-3))
            
            score.append(eps_score)
            # time_step += sample[3].sum()
            eat_count += sample[4+1].sum()
            die_count += sample[5+1].sum()
            
        print("DEBUG: ",[i.squeeze()[:10] for i in sample[1:]],"\neps_score: ", eps_score)
            
        score = torch.concat(score)
        s_mean, std = score.mean(), score.std()
        result = {
            'score_mean': s_mean.item(),
            'score_std': std.item(),
            # 'time_step': time_step.item()/N,
            'eat_count': eat_count.item()/N,
            'die_count': die_count.item()/N,
        }
        return result

    def save_model(self, epoch, score_mean, score_std):
        model_path = os.path.join(self.log_dir,'model',
                                  f'ppo_{epoch}_{score_mean:.3f}_{score_std:.3f}.pth')
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        self.agent.save_model(model_path)
        INFO(f'Save model to {model_path}')



