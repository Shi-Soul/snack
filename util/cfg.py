import yaml
import os
def load_config(yaml_path:str)-> dict:
    try:
        with open(yaml_path, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    except:
        print('尝试UTF-8编码....')
        with open(yaml_path, 'r', encoding='UTF-8') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    # print(params,params.__class__);exit(0)
    return params

def save_config(dict_value: dict, save_path: str):
    """dict保存为yaml"""
    with open(os.path.join(save_path,"cfg.yaml"), 'w') as file:
        file.write(yaml.dump(dict_value, allow_unicode=True))
        
def save_config_file(cfg_path: str, save_path: str):
    """Copy cfgfile to the path"""
    with open(cfg_path, 'r') as file:
        cfg = file.read()
    with open(os.path.join(save_path,"cfg.yaml"), 'w') as file:
        file.write(cfg)