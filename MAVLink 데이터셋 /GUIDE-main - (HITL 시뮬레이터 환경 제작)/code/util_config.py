import json
import os


class Config:
    def __init__(self) -> None:
        self.path_config = {}
        self.param_config = {}
        self.load_path_config('./config/config_path.json')
        self.load_param_config('./config/config_parameter.json')

    def load_path_config(self, config_file_path: str) -> None:
        with open(config_file_path, 'r', encoding='utf-8') as config_file:
            config = json.load(config_file)
        root_path = config['root_path']

        for field_name in config:
            if field_name == 'root_path':
                continue
            if config[field_name][:2] == '.\\' or config[field_name][:2] == './':
                config[field_name] = os.path.join(root_path, config[field_name][2:])

        self.path_config = config

    def get_path_config(self) -> dict:
        return self.path_config

    def reload_path_config(self, config_file_path: str = './config/config_path.json') -> None:
        self.load_path_config(config_file_path)

    def load_param_config(self, config_file_path: str) -> None:
        with open(config_file_path, 'r', encoding='utf-8') as config_file:
            config = json.load(config_file)

        self.param_config = config

    def get_param_config(self) -> dict:
        return self.param_config

    def reload_param_config(self, config_file_path: str = './config/config_parameter.json') -> None:
        self.load_param_config(config_file_path)

    def save_param_config(self, config_file_path: str = './config/config_parameter.json') -> None:
        with open(config_file_path, 'w', encoding='utf-8') as config_file:
            json.dump(self.param_config, config_file, indent=4)


CONFIG = Config()


def make_dirs(path_str: str) -> None:
    _, ext = os.path.splitext(path_str)

    if ext:
        target_dir = os.path.dirname(path_str)
    else:
        target_dir = path_str

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)


def get_param_config() -> dict:
    return CONFIG.get_param_config()


def get_path_config() -> dict:
    return CONFIG.get_path_config()


def get_gan_type() -> str:
    param_config = get_param_config()
    gan_type = param_config['seqgan']['gan_type_template'].format(param_config['seqgan']['unrolled_step'],
                                                                  param_config['seqgan']['is_wasserstein'])
    return gan_type
