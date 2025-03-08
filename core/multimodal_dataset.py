import torch
from torch.utils.data import Dataset
import dgl
from core.aug import *


class MultiModalDataSet(Dataset):
    def __init__(self):
        self.data = []

    def add_data(self, metric_Xs, trace_Xs, log_Xs, global_root_id, failure_type_id, local_root, nodes, edges):
        node_num = len(nodes)
        graph = dgl.graph(edges, num_nodes=node_num)
        graph.ndata["metric"] = torch.FloatTensor(metric_Xs)
        graph.ndata["trace"] = torch.FloatTensor(trace_Xs)
        graph.ndata["log"] = torch.FloatTensor(log_Xs)
        # graph.ndata["logs"] = torch.zeros(logs[i].shape)
        root_labels = [0] * len(nodes)
        root_labels[nodes.index(local_root)] = 1
        graph.ndata["root"] = torch.LongTensor(root_labels)

        in_degrees = graph.in_degrees()
        zero_indegree_nodes = [i for i in range(len(in_degrees)) if in_degrees[i].item() == 0]
        for node in zero_indegree_nodes:
            graph.add_edges(node, node)
        
        self.data.append((graph, (global_root_id, failure_type_id)))
           

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
