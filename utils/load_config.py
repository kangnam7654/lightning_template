import yaml

def load_config(config_path=None):
    with open(config_path) as cfg:
        config = yaml.full_load(cfg)
    return config
