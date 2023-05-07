import datetime
import random
import pickle
import os

from helper.paths import dataset_path


def exp_name_stamp() -> str:
    """
    Get a unique experiment name based on current timestamp for saving checkpoints.
    Returns:
        str
    """

    now_time = datetime.datetime.now()
    st = now_time.strftime(f'%m%d_%H%M%S_{random.randint(0, 9999):04d}')
    return st

def load(file):
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data


def save(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def read_config(yaml_path: str):
    import yaml
    with open(yaml_path, 'r') as fin:
        configs = yaml.safe_load(fin)
    return configs
