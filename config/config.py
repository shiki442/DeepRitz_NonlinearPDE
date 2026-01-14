from ml_collections import ConfigDict
import argparse

def load_config(config_file=None):
    if config_file is None:
        parser = argparse.ArgumentParser(description="Basic paser")
        parser.add_argument("--config_path", type=str, help="Path to the configuration file", default="")
        args = parser.parse_args()
        config_file = args.config_path
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    cfg = ConfigDict(config)
    return cfg