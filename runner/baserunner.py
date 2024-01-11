from typing import Sequence
from util import DEBUG,INFO
class BaseRunner():
    def __init__(self, agent, env, *args, **kwargs):
        self.agent = agent
        self.env = env
        
    def run(self):
        """
            obs, reward, done = self.env.reset()
            while not done:
                action = self.agent.policy(obs)
                obs, reward, done = self.env.step(action)
        """
        rewards: Sequence[int] = []
        obs, reward, done = self.env.reset()
        while not done:
            try:
                action = self.agent.policy(obs)
            except KeyboardInterrupt:
                INFO("\nUser Quit!")
                break
            DEBUG(f"action: {action}")
            
            obs, reward, done = self.env.step(action)
            DEBUG(f"reward: {reward}, done: {done}")
            
            rewards.append(reward)
        _,time_step,snake_length = self.env.get_hidden_state()
        return sum(rewards), time_step, snake_length
        