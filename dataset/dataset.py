import torch
from torch.utils.data import Dataset
import dgl
from helper.aug import *


class MultiModelDataSet(Dataset):
    def __init__(self, metrics, traces, logs, instance_labels, type_labels, nodes, edges):
        self.data = []
        node_num = len(nodes)
        for i in range(len(instance_labels)):
            graph = dgl.graph(edges, num_nodes=node_num)
            graph.ndata["metrics"] = torch.FloatTensor(metrics[i])
            # graph.ndata["metrics"] = torch.zeros(metrics[i].shape)
            graph.ndata["traces"] = torch.FloatTensor(traces[i])
            # graph.ndata["traces"] = torch.zeros(traces[i].shape)
            graph.ndata["logs"] = torch.FloatTensor(logs[i])
            # graph.ndata["logs"] = torch.zeros(logs[i].shape)

            # self-loop
            in_degrees = graph.in_degrees()
            zero_indegree_nodes = [i for i in range(len(in_degrees)) if in_degrees[i].item() == 0]
            for node in zero_indegree_nodes:
                graph.add_edges(node, node)

            root, type = instance_labels[i], type_labels[i]
            self.data.append((graph, (root, type)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
