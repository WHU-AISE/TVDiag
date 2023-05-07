"""
    https://github.com/Shen-Lab/GraphCL
"""

import copy
import random
from helper import io

import dgl
import torch


def aug_drop_node(graph, root, drop_percent=0.2):
    """
        drop non-root nodes
    """
    num = graph.number_of_nodes()  # number of nodes of one graph
    drop_num = int(num * drop_percent)  # number of drop nodes
    aug_graph = copy.deepcopy(graph)
    all_node_list = [i for i in range(num) if i != root]
    drop_node_list = random.sample(all_node_list, drop_num)
    aug_graph.remove_nodes(drop_node_list)
    # aug_graph = add_self_loop_if_not_in(aug_graph)
    return aug_graph


def aug_drop_node_list(graph_list, labels, drop_percent):
    graph_num = len(graph_list)  # number of graphs
    aug_list = []
    for i in range(graph_num):
        aug_graph = aug_drop_node(graph_list[i], labels[i], drop_percent)
        aug_list.append(aug_graph)
    return aug_list


def aug_random_walk(graph, root, drop_percent=0.2):
    """
        random walk from root
    """
    rg = dgl.reverse(graph, copy_ndata=False, copy_edata=False)
    num_edge = rg.number_of_edges()  # number of edges of one graph
    retain_num = num_edge - int(num_edge * drop_percent)  # number of retain edges
    trace = dgl.sampling.random_walk(rg, [root], length=retain_num, return_eids=True)[1]
    edges = trace.flatten()
    subgraph = dgl.edge_subgraph(graph, edges, store_ids=False)
    return subgraph


def aug_random_walk_list(graph_list, labels, drop_percent):
    graph_num = len(graph_list)  # number of graphs
    aug_list = []
    for i in range(graph_num):
        sub_graph = aug_random_walk(graph_list[i], labels[i], drop_percent)
        aug_list.append(sub_graph)
    return aug_list


def add_self_loop_if_not_in(graph):
    in_degrees = graph.in_degrees()
    zero_indegree_nodes = [i for i in range(len(in_degrees)) if in_degrees[i].item() == 0]
    for node in zero_indegree_nodes:
        graph.add_edges(node, node)
    return graph