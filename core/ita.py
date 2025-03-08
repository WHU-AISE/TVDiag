
from collections import OrderedDict
import copy
import torch
import torch.nn.functional as F
from core.model.MainModel import MainModel

# Adapted from https://github.com/google-research/google-research/blob/master/tag/celeba/CelebA.ipynb


def cal_task_affinity(model: MainModel, 
                      optimizer, 
                      batch_graphs, 
                      type_labels,
                      device):
    model1 = copy.deepcopy(model)
    model2 = copy.deepcopy(model)

    lr = optimizer.param_groups[0]['lr']
    weight_decay = optimizer.param_groups[0]['weight_decay']
    opt1 = torch.optim.Adam(model1.parameters(), lr=lr, weight_decay=weight_decay)
    opt2 = torch.optim.Adam(model2.parameters(), lr=lr, weight_decay=weight_decay)

    ####################################
    ####   calculate the affinity   ####
    ####    RCL -> FTI              ####
    ####################################
    _, _, root_logit, type_logit = model1(batch_graphs)
    l_rcl = cal_rcl_loss(root_logit, batch_graphs, device)
    # l_rcl = F.cross_entropy(root_logit, instance_labels)

    l_fti = F.cross_entropy(type_logit, type_labels)
    # update model with l_rcl
    l_rcl.backward()
    opt1.step()
    _, _,  _, type_logit2 = model1(batch_graphs)
    l_fti2 = F.cross_entropy(type_logit2, type_labels)
    Z_r2f = 1 - (l_fti2.detach().item()/l_fti.detach().item())

    ####################################
    ####   calculate the affinity   ####
    ####    FTI -> RCL              ####
    ####################################
    _, _, root_logit, type_logit = model2(batch_graphs)
    # l_rcl = F.cross_entropy(root_logit, instance_labels)
    l_rcl = cal_rcl_loss(root_logit, batch_graphs, device)
    l_fti = F.cross_entropy(type_logit, type_labels)
    # update model with l_fti
    l_fti.backward()
    opt2.step()
    _, _, root_logit2, _ = model1(batch_graphs)
    l_rcl2 = cal_rcl_loss(root_logit2, batch_graphs, device)
    Z_f2r = 1 - (l_rcl2.detach().item()/l_rcl.detach().item())

    return Z_r2f, Z_f2r


def cal_rcl_loss(root_logit, batch_graphs, device):
    num_nodes_list = batch_graphs.batch_num_nodes()
    total_loss = None
    
    start_idx = 0
    for idx, num_nodes in enumerate(num_nodes_list):
        end_idx = start_idx + num_nodes
        node_logits = root_logit[start_idx : end_idx].reshape(1, -1)
        root = batch_graphs.ndata["root"][start_idx : end_idx].tolist().index(1)

        loss = F.cross_entropy(node_logits, torch.LongTensor([root]).view(1).to(device))

        if total_loss is None:
            total_loss = loss
        else:
            total_loss += loss
        start_idx += num_nodes
        
    l_rcl = total_loss / len(num_nodes_list)
    return l_rcl