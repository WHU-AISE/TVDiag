import os
import random

import dgl
import torch
import numpy as np
from torch.utils.data import DataLoader
from helper.io import read_config, load
from helper.paths import config_path
from TVDiag import TVDiag
from process.GaiaProcess import GaiaProcess
from process.AIOps22Process import AIOps22Process
import logging
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_labels = torch.tensor(labels)
    return batched_graph, batched_labels


def build_dataloader(cfg: dict):
    reconstruct = cfg['common_args']['reconstruct']
    dataset_name = cfg['common_args']['dataset_name']
    dataset_args = cfg['dataset'][dataset_name]

    # N_S = dataset_args['N_S']
    N_A = dataset_args['N_A']
    N_I = dataset_args['N_I']
    epochs = dataset_args['epochs']
    metric_embedding_dim = dataset_args['metric_embedding_dim']
    trace_embedding_dim = dataset_args['trace_embedding_dim']
    log_embedding_dim = dataset_args['log_embedding_dim']

    if dataset_name == 'gaia':
        processor = GaiaProcess(cfg)
        train_data, test_data = processor.process(reconstruct=reconstruct)
    elif dataset_name == 'aiops22-pod':
        processor = AIOps22Process(cfg)
        train_data, test_data = processor.process(reconstruct=reconstruct)
    else:
        raise NotImplementedError
    train_dataloader = DataLoader(train_data, batch_size=dataset_args['batch_size'], shuffle=True, collate_fn=collate)
    test_dataloader = DataLoader(test_data, batch_size=dataset_args['batch_size'], shuffle=False, collate_fn=collate)

    return {'N_A': N_A,
            'N_I': N_I,
            'epochs': epochs,
            'metric_embedding_dim': metric_embedding_dim,
            'trace_embedding_dim': trace_embedding_dim,
            'log_embedding_dim': log_embedding_dim}, \
           train_dataloader, test_dataloader

if __name__ == '__main__':
    cfg = read_config(config_path)
    common_args = cfg['common_args']
    set_seed(common_args['seed'])
    model_args = cfg['model']
    
    data_args, train_dl, test_dl = build_dataloader(cfg)
    
    model_args.update(data_args)
    model_args.update(common_args)

    model = TVDiag(model_args, 'cpu', train_dl, test_dl)

    model.train()