# file with functions to jointly load configs from default.cfg
import os
import logging
import configparser
from pathlib import Path


def load_default_config(config_dict):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = str(Path(script_dir)) + '/default.cfg'
    config = configparser.ConfigParser()
    config.read(config_file)
    for s in config.sections():
        config_dict.update({k: eval(v) for k, v in config.items(s)})
    return config_dict

if __name__ == "__main__":
    print(load_default_config(dict()))
