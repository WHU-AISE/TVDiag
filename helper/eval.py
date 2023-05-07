import numpy as np
import torch
from helper.io import *
from sklearn.metrics import precision_score, recall_score, f1_score


def accuracy(output, target, topk=(1, 5)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append((correct_k / batch_size).item())
    return tuple(res)


def precision(output, target, k=5):
    _, pred = output.topk(k, 1, True, True)
    y_pred = pred.detach().numpy()
    y_true = target.detach().numpy().reshape(-1, 1)
    pre = precision_score(y_pred[:, 0], y_true, average='weighted')

    return pre


def recall(output, target, k=5):
    _, pred = output.topk(k, 1, True, True)
    y_pred = pred.detach().numpy()
    y_true = target.detach().numpy().reshape(-1, 1)
    rec = recall_score(y_pred[:, 0], y_true, average='weighted')

    return rec


def f1score(output, target, k=5):
    _, pred = output.topk(k, 1, True, True)
    y_pred = pred.detach().numpy()
    y_true = target.detach().numpy().reshape(-1, 1)
    f1 = f1_score(y_pred[:, 0], y_true, average='weighted')

    return f1

# if __name__ == '__main__':
#     output = torch.randint(low=0, high=6, size=[8, 10])
#     target = torch.ones(8, dtype=torch.long)
#     print(accuracy(output, target))
