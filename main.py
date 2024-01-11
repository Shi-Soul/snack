
import torch

from util import load_config, save_config, setup_logging, setup_seed, INFO, DEBUG
from agent import HumanAgent, RandomAgent, PGAgent
from env import SnakeEnv
from runner import BaseRunner, PGTrainer
from render import TextRender
CONFIG_FILE = 'cfg.yaml'

def play(params):
    agent = HumanAgent()
    # agent = RandomAgent(1)
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
    params = load_config(CONFIG_FILE)
    params['device'] = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    
    setup_seed(params['seed'])
    
    pg_train(params)
    
    print("Done!")
    

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(">> BUG: ",e); import pdb; pdb.post_mortem()




