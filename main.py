
import torch
import argparse
import os
from util import load_config, save_config, setup_logging, setup_seed, INFO, DEBUG, str2bool
from agent import HumanAgent, RandomAgent, PGAgent, DQNAgent, PPOAgent
from env import SnakeEnv,VectorizedSnakeEnv
from runner import BaseRunner, PGTrainer, DQNTrainer, PPOTrainer
from render import TextRender

def play(params):
    # agent = HumanAgent()
    # agent = RandomAgent(1)
    from runner.pg_trainer import _get_nn_dict
    agent = PGAgent(_get_nn_dict['small'](params['obs_channel']), params['device'])
    agent.load_model("result/pg_t3_r_12152820/model/pg_33030_-17.552_4.356.pth")
    agent.train(False)
    
    env = SnakeEnv(params["init_length"], params["size"], params["max_step"], params["rew_penalty"], params["rew_nothing"], params["rew_food"])
    render = TextRender()
    
    runner = BaseRunner(agent, env, render)
    ret = runner.run()
    print(ret)
    ...
    
def play_vec(params):
    # agent = HumanAgent()
    # from runner.dqn_trainer import _get_nn_dict
    # agent = DQNAgent(_get_nn_dict['normal'](params['obs_channel']), params['device'])
    from runner.ppo_trainer import _get_nn_dict
    agent = PPOAgent(_get_nn_dict['small'](params['obs_channel']), params['device'])
    # agent = DQNAgent(_get_nn_dict['small'](params['obs_channel']), params['device'])
    agent.load_model("/home/wjxie/wjxie/env/snack/result/ppo_mid_21165929/model/ppo_90_298.397_298.850.pth")
    agent.train(False)
    
    
    env = VectorizedSnakeEnv(1,"cuda",params["init_length"], params["size"], params["max_step"], params["rew_penalty"], params["rew_nothing"], params["rew_food"],params['rew_difficulty'])
    render = TextRender()
    runner = BaseRunner(agent, env, render)
    ret = runner.run()
    print(ret)
    

def pg_train(params):
    log_dir = setup_logging(params['exp_name'])
    INFO("Training PG Agent")
    params['log_dir'] = log_dir
    INFO("Params: ",params,pp=True)
    save_config(params, log_dir)
    
    env = SnakeEnv(params["init_length"], params["size"], params["max_step"], params["rew_penalty"], params["rew_nothing"], params["rew_food"])
    # render = TextRender()
    
    runner = PGTrainer(env, params)
    ret = runner.run()
    INFO(ret,pp=True)
    print("Logging to: ",log_dir)

def dqn_train(params):
    log_dir = setup_logging(params['exp_name'])
    INFO("Training DQN Agent")
    params['log_dir'] = log_dir
    INFO("Params: ",params,pp=True)
    save_config(params, log_dir)
    
    assert params['use_vec_env'], "DQN only support vectorized env"
    env = VectorizedSnakeEnv(params["n_env"],"cuda",params["init_length"], params["size"], params["max_step"], params["rew_penalty"], params["rew_nothing"], params["rew_food"],params['rew_difficulty'])

    runner = DQNTrainer(env, params)
    ret = runner.run()
    INFO(ret,pp=True)
    print("Logging to: ",log_dir)
    
def ppo_train(params):
    log_dir = setup_logging(params['exp_name'])
    INFO("Training PPO Agent")
    params['log_dir'] = log_dir
    INFO("Params: ",params,pp=True)
    save_config(params, log_dir)
    
    # assert params['use_vec_env'], "DQN only support vectorized env"
    env = VectorizedSnakeEnv(params["n_env"],"cuda",params["init_length"], params["size"], params["max_step"], params["rew_penalty"], params["rew_nothing"], params["rew_food"],params['rew_difficulty'])

    runner = PPOTrainer(env, params)
    ret = runner.run()
    INFO(ret,pp=True)
    print("Logging to: ",log_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='cfgs/cfg.yaml')
    parser.add_argument('--play', type=str2bool, default=False)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--algo', type=str, default="ppo")
    args = parser.parse_args() # type: ignore
    
    params = load_config(args.config)
    params['device'] = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    if args.seed != -1:
        params['seed'] = args.seed
    setup_seed(params['seed'])
    
    if args.play:
        play_vec(params)
    else:
        if args.algo == 'dqn':
            dqn_train(params)
        elif args.algo == 'pg':
            pg_train(params)
        elif args.algo == 'ppo':
            ppo_train(params)
        else:
            raise NotImplementedError
    
    print("Done!")
    

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(">> BUG: ",e); import pdb; pdb.post_mortem()




