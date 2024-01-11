from util import load_config, setup_logging, setup_seed, INFO, DEBUG
from agent import HumanAgent, RandomAgent
from env import SnakeEnv
from runner import BaseRunner
CONFIG_FILE = 'cfg.yaml'

def main():
    params = load_config(CONFIG_FILE)
    setup_seed(params['seed'])
    
    agent = HumanAgent()
    env = SnakeEnv(params["init_length"], params["size"], params["max_step"])
    runner = BaseRunner(agent, env)
    ret = runner.run()
    print(ret)
    
    print("Done!")
    

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(">> BUG: ",e); import pdb; pdb.post_mortem()




