import yaml
from box import Box


def load_config(cfg_path=None):
    with open(cfg_path, 'r') as f:
        config = Box(yaml.full_load(f))
    return config


if __name__ == '__main__':
    pass