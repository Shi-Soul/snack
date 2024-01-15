
import torch
import argparse
import os
from util import load_config, save_config, setup_logging, setup_seed, INFO, DEBUG, str2bool
from agent import HumanAgent, RandomAgent, PGAgent
from env import SnakeEnv,VectorizedSnakeEnv
from runner import BaseRunner, PGTrainer
from render import TextRender

def play(params):
    agent = HumanAgent()
    # agent = RandomAgent(1)
    from runner.pg_trainer import _get_nn_dict
    # agent = PGAgent(_get_nn_dict['small'](params['obs_channel']), params['device'])
    # agent.load_model("result/pg_t3_r_12152820/model/pg_33030_-17.552_4.356.pth")
    # agent.train(False)
    
    env = SnakeEnv(params["init_length"], params["size"], params["max_step"], params["rew_penalty"], params["rew_nothing"], params["rew_food"])
    render = TextRender()
    
    runner = BaseRunner(agent, env, render)
    ret = runner.run()
    print(ret)
    ...
    
def play_vec(params):
    agent = HumanAgent()
    
    env = VectorizedSnakeEnv(1,params["init_length"], params["size"], params["max_step"], params["rew_penalty"], params["rew_nothing"], params["rew_food"])
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
    # agent = RandomAgent(1)
    env = SnakeEnv(params["init_length"], params["size"], params["max_step"], params["rew_penalty"], params["rew_nothing"], params["rew_food"])
    # render = TextRender()
    
    runner = PGTrainer(env, params)
    ret = runner.run()
    INFO(ret,pp=True)
    print("Logging to: ",log_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='cfgs/cfg.yaml')
    parser.add_argument('--play', type=str2bool, default=False)
    parser.add_argument('--seed', type=int, default=-1)
    args = parser.parse_args()
    
    params = load_config(args.config)
    params['device'] = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    if args.seed != -1:
        params['seed'] = args.seed
    setup_seed(params['seed'])
    
    if args.play:
        play_vec(params)
    else:
        pg_train(params)
    
    print("Done!")
    

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(">> BUG: ",e); import pdb; pdb.post_mortem()




