import time
import pandas as pd
import random
import torch.backends.cudnn as cudnn
from config.exp_config import Config
from helper.logger import get_logger
import torch
import numpy as np
from core.TVDiag import TVDiag
from helper.Result import Result
from process.EventProcess import EventProcess
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def build_dataloader(config: Config, logger):
    reconstruct = config.reconstruct
    processor = EventProcess(config, logger)
    train_data, aug_data, test_data = processor.process(reconstruct=reconstruct)

    return train_data, aug_data, test_data


def train_and_evaluate(config: Config, log_dir, exp_name):
    set_seed(2)
    logger = get_logger(log_dir, exp_name)
    logger.info("Load dataset")
    train_data, aug_data, test_data = build_dataloader(config, logger)
    logger.info("Training...")
    model = TVDiag(config, logger, log_dir)
    model.train(train_data, aug_data)
    res: Result = model.evaluate(test_data)
    return res.export_df(exp_name)


if __name__ == '__main__':
    for dataset in ['gaia', 'sockshop']:
        config = Config(dataset)
        config.reconstruct = False # Directly use trained alert features
        train_and_evaluate(config, f'./logs/{dataset}', dataset)
