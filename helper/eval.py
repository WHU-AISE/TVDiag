import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score


def HR(output, target, topk=(1, 5)):
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

def NDCG(output, target, topk=(3,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # correct for 3  (3, num)
    # [[ True, False, True, ..., False, False],
    #  [ False, False, False, ..., False, False],
    #  [ False, False, False, ..., True, False]]
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        sum_correct = correct[:k].sum(1)
        respective_score = sum_correct / torch.log2(torch.arange(len(sum_correct))+2)
        total_score = respective_score.sum()
        res.append((total_score / batch_size).item())
    return tuple(res)

def target_rank(output, target, k=10):
    _, pred = output.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    ranks = []
    for col in range(correct.size(1)):
        try:
            idx=torch.where(correct[:, col] == target[col])[0].item() + 1
        except:
            idx=10
        ranks.append(idx)
    
    return ranks


def precision(output, target, k=5):
    _, pred = output.topk(k, 1, True, True)
    y_pred = pred.cpu().detach().numpy()
    y_true = target.cpu().detach().numpy().reshape(-1, 1)
    pre = precision_score(y_true, y_pred[:, 0], average='weighted')

    return pre


def recall(output, target, k=5):
    _, pred = output.topk(k, 1, True, True)
    y_pred = pred.cpu().detach().numpy()
    y_true = target.cpu().detach().numpy().reshape(-1, 1)
    rec = recall_score(y_true, y_pred[:, 0], average='weighted')

    return rec


def f1score(output, target, k=5):
    _, pred = output.topk(k, 1, True, True)
    y_pred = pred.cpu().detach().numpy()
    y_true = target.cpu().detach().numpy().reshape(-1, 1)
    f1 = f1_score(y_true, y_pred[:, 0], average='weighted')

    return f1