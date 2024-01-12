
import torch
import argparse

from util import load_config, save_config, setup_logging, setup_seed, INFO, DEBUG, str2bool
from agent import HumanAgent, RandomAgent, PGAgent
from env import SnakeEnv
from runner import BaseRunner, PGTrainer
from render import TextRender

def play(params):
    # agent = HumanAgent()
    # agent = RandomAgent(1)
    from runner.pg_trainer import _get_nn_small, _get_nn_normal
    agent = PGAgent(_get_nn_small(params['obs_channel']), params['device'])
    agent.load_model("/home/wjxie/wjxie/env/snack/result/pg_t3_11193730/model/pg_90_-1.470_3.659.pth")
    agent.train(False)
    
    env = SnakeEnv(params["init_length"], params["size"], params["max_step"], params["penalty"])
    render = TextRender()
    
    runner = BaseRunner(agent, env, render)
    ret = runner.run()
    print(ret)
    ...

def pg_train(params):
    log_dir = setup_logging(params['exp_name'])
    INFO("Training PG Agent")
    params['log_dir'] = log_dir
    INFO("Params: ",params,pp=True)
    save_config(params, log_dir)
    # agent = RandomAgent(1)
    env = SnakeEnv(params["init_length"], params["size"], params["max_step"], params["penalty"])
    # render = TextRender()
    
    runner = PGTrainer(env, params)
    ret = runner.run()
    INFO(ret,pp=True)
    print("Logging to: ",log_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='cfgs/cfg.yaml')
    parser.add_argument('--play', type=str2bool, default=False)
    args = parser.parse_args()
    
    params = load_config(args.config)
    params['device'] = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    
    setup_seed(params['seed'])
    
    if args.play:
        play(params)
    else:
        pg_train(params)
    
    print("Done!")
    

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(">> BUG: ",e); import pdb; pdb.post_mortem()




