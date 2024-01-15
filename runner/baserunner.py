from typing import Sequence
import torch 

from env import VectorizedSnakeEnv
from util import DEBUG,INFO

class BaseRunner():
    def __init__(self, agent, env, render, *args, **kwargs):
        self.agent = agent
        self.env = env
        self.render = render
        if isinstance(env, VectorizedSnakeEnv):
            self.use_vec_env = True
            assert env.n == 1, "VectorizedSnakeEnv.n must be 1"
        else: 
            self.use_vec_env = False
        
        
    def run(self):
        """
            obs, reward, done = self.env.reset()
            while not done:
                action = self.agent.policy(obs)
                obs, reward, done = self.env.step(action)
        """
        rewards: Sequence[int] = []
        obs, reward, done = self.env.reset()
        try:
            while not done:
                self.render.render(obs[0],self.env.time_step)
                action = self.agent.policy(obs)
                DEBUG(f"action: {action}")
                if self.use_vec_env:
                    action = torch.tensor(action).reshape(-1,1)
                obs, reward, done = self.env.step(action)
                DEBUG(f"reward: {reward}, done: {done}")
            
                rewards.append(reward)
        except KeyboardInterrupt:
            INFO("\nUser Quit!")
        _,time_step,snake_length = self.env.get_hidden_state()
        return sum(rewards), time_step, snake_length
        