import json
import pickle


def load(file):
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data

def load_json(file):
    with open(file, 'r', encoding='utf8') as fp:
        data = json.load(fp)
    return data


def save(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)

def save_json(file, data: dict):
    with open(file, 'w') as f:
        json.dump(data, f)
    print('Save successfully!')