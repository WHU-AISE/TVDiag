import json
import pickle


def load_pkl(file):
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data


def save_pkl(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)

def load_json(file):
    data = None
    with open(file, 'r', encoding='utf8') as fp:
        data = json.load(fp)
    return data

def save_json(file, data: dict):
    with open(file, 'w') as f:
        json.dump(data, f)