import argparse
import os
import random
import torch.backends.cudnn as cudnn
from helper.logger import get_logger
import dgl
import torch
import numpy as np
from torch.utils.data import DataLoader
from core.TVDiag import TVDiag
from process.EventProcess import EventProcess
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

# common
parser.add_argument('--seed', type=int, default=2)
parser.add_argument('--log_step', type=int, default=20)
parser.add_argument('--eval_period', type=int, default=10)
parser.add_argument('--reconstruct', type=bool, default=False)
parser.add_argument('--gpu_devices', type=str, default='0')

# dataset
parser.add_argument('--dataset', type=str, default='gaia', help='name of dataset')
parser.add_argument('--N_T', type=int, default=5, help='number of failure types')
parser.add_argument('--N_I', type=int, default=10, help='number of instances')

# TVDiag
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--patience', type=int, default=10)

parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--seq_hidden', type=int, default=128)
parser.add_argument('--linear_hidden', type=list, default=[64])
parser.add_argument('--graph_hidden', type=int, default=64)
parser.add_argument('--graph_out', type=int, default=32)
parser.add_argument('--feat_drop', type=float, default=0)
parser.add_argument('--attn_drop', type=float, default=0)
parser.add_argument('--aggregator', type=str, default='lstm')
parser.add_argument('--TO', action='store_true')
parser.add_argument('--CM', action='store_true')
parser.add_argument('--temperature', type=float, default=0.3)
parser.add_argument('--guide_weight', type=float, default=0.1)
parser.add_argument('--dynamic_weight', action='store_true')
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--aug_percent', type=float, default=0.2)
args = parser.parse_args()

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


def build_dataloader(args, logger):
    reconstruct = args.reconstruct
    processor = EventProcess(args, logger)
    train_data, test_data = processor.process(reconstruct=reconstruct)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    return train_dataloader, test_data

if __name__ == '__main__':
    logger = get_logger(f'logs/{args.dataset}', 'TVDiag')
    use_gpu = torch.cuda.is_available()
    set_seed(args.seed)

    if use_gpu:
        logger.info("Currently using GPU {}".format(args.gpu_devices))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
        device='cuda'
    else:
        logger.info("Currently using CPU (GPU is highly recommended)")
        device = 'cpu'

    logger.info("Load dataset")
    train_dl, test_data = build_dataloader(args, logger)
    
    logger.info("Training...")
    model = TVDiag(args, logger, device)

    model.train(train_dl)
    model.evaluate(test_data)