import os
from pathlib import Path


class GetPaths:
    def __init__(self):
        self.configs_dir = self.get_configs_folder()

    @staticmethod
    def get_project_root(*paths):
        root_dir = os.path.join(Path(__file__).parents[2], *paths)
        return root_dir

    @staticmethod
    def get_data_folder(*paths):
        data_dir = os.path.join(Path(__file__).parents[2], 'data', *paths)
        return data_dir

    @staticmethod
    def get_configs_folder(*paths):
        configs_dir = os.path.join(Path(__file__).parents[2], 'configs', *paths)
        return configs_dir
